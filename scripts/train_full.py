"""
train_full.py — Full-dataset training for DeepSentinel.

Loads all preprocessed clips from Tracks 1-3, MELD real, CMU-MOSEI real,
and MUStARD sarcasm using pre-built speaker-stratified 80/10/10 split manifests.

Phase 1: frozen backbones, train heads + classifier on cached z_at/z_v.
Phase 2: optional backbone fine-tuning (end-to-end, slow).

Usage:
    python scripts/train_full.py
    python scripts/train_full.py --device cuda --epochs 50 --batch_size 32
    python scripts/train_full.py --no_phase2
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detection_model import DeepfakeDetector
from src.training.trainer import Trainer
from src.training.dataset import UNKNOWN_EMOTION, UNKNOWN_SARCASM, EMOTION_TO_IDX
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

REPO_ROOT          = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR   = REPO_ROOT / "data/preprocessed"
KEYFRAME_CACHE_DIR = REPO_ROOT / "data/preprocessed/keyframes"
CKPT_DIR           = REPO_ROOT / "checkpoints/full"
LOG_DIR            = REPO_ROOT / "logs/full"

TRAIN_MANIFEST_CSV = REPO_ROOT / "data/processed/train_manifest.csv"
VAL_MANIFEST_CSV   = REPO_ROOT / "data/processed/val_manifest.csv"
TEST_MANIFEST_CSV  = REPO_ROOT / "data/processed/internal_test_manifest.csv"

SECTION = "=" * 60


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(f"{SECTION}")


def _emo(code) -> int:
    if code is None or (isinstance(code, float) and code != code):
        return UNKNOWN_EMOTION
    return EMOTION_TO_IDX.get(str(code).strip(), UNKNOWN_EMOTION)


def _load_manifest_records(csv_path: Path) -> list[dict]:
    records = []
    if not csv_path.exists():
        log.warning(f"Manifest missing: {csv_path}")
        return records
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row["clip_id"])
            z_at_p = REPO_ROOT / str(row.get("z_at_path") or f"data/preprocessed/features/z_at/{cid}.pt")
            z_v_p  = REPO_ROOT / str(row.get("z_v_path")  or f"data/preprocessed/features/z_v/{cid}.pt")
            if not z_at_p.exists() or not z_v_p.exists():
                continue
            records.append({
                "clip_id":         cid,
                "z_at_path":       str(z_at_p),
                "z_v_path":        str(z_v_p),
                "video_path":      str(row.get("video_path", "")),
                "fake_label":      int(row.get("fake_label", -1)),
                "audio_emotion":   _emo(row.get("audio_emotion")),
                "visual_emotion":  _emo(row.get("visual_emotion")),
                "sarcasm_label":   int(row.get("sarcasm_label", UNKNOWN_SARCASM)),
                "source_pipeline": str(row.get("source_pipeline", "unknown")),
                "speaker_id":      str(row.get("speaker_id", "unknown")),
            })
    return records


class ZipExtractor:
    def __init__(self):
        self.zip_handles = {}
        self.drive_paths = {}
        self.drive_dir = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")
        self.local_zip_dir = REPO_ROOT / "data/zips"
        self.local_zip_dir.mkdir(parents=True, exist_ok=True)
        
        # List of expected ZIP archives - just locate Drive paths on startup
        zip_names = ["tracks_1_2_3_4.zip", "meld_raw.zip", "mustard.zip", "cmumosei.zip"]
        for name in zip_names:
            drive_path = self._find_file(self.drive_dir, name)
            if drive_path:
                self.drive_paths[name] = drive_path
                print(f"[ZipExtractor] Registered Drive source for: {name}")

    def _find_file(self, start_dir: Path, name: str) -> Path | None:
        if not start_dir.exists():
            return None
        # Check standard checkpoints/datasets directories first to make it fast
        for p in [start_dir / "datasets" / name, start_dir / name]:
            if p.exists():
                return p
        # Fallback to recursive search if not found directly
        for p in start_dir.rglob(name):
            return p
        return None

    def _get_target_zip_name(self, rel_path_str: str) -> str | None:
        p = rel_path_str.lower()
        if "mosei" in p:
            return "cmumosei.zip"
        elif "meld" in p:
            return "meld_raw.zip"
        elif "mustard" in p:
            return "mustard.zip"
        elif "track" in p or "synthetic" in p:
            return "tracks_1_2_3_4.zip"
        return None

    def _lazy_load_zip(self, name: str):
        if name in self.zip_handles:
            return self.zip_handles[name]
            
        drive_path = self.drive_paths.get(name)
        if not drive_path:
            return None
            
        local_path = self.local_zip_dir / name
        if not local_path.exists():
            import os
            import time
            lock_dir = local_path.with_suffix(".lock")
            try:
                os.mkdir(lock_dir)
                # Lock acquired! Perform the copy
                try:
                    if not local_path.exists():
                        print(f"[ZipExtractor] Process {os.getpid()} copying {name} to local Colab SSD...")
                        import shutil
                        temp_path = local_path.with_suffix(f".tmp_{os.getpid()}")
                        shutil.copy2(drive_path, temp_path)
                        os.replace(temp_path, local_path)
                        print(f"[ZipExtractor] Copied {name} successfully.")
                finally:
                    # Release lock
                    if lock_dir.exists():
                        os.rmdir(lock_dir)
            except FileExistsError:
                # Lock is held by another process. Wait until copy is done.
                print(f"[ZipExtractor] Process {os.getpid()} waiting for another process to finish copying {name}...")
                t0 = time.time()
                while lock_dir.exists() or not local_path.exists():
                    time.sleep(2)
                    if time.time() - t0 > 1800:  # 30 minutes timeout
                        break
            except Exception as e:
                print(f"[ZipExtractor] Failed to copy {name} locally: {e}. Streaming directly from Drive...")
                local_path = drive_path
                
        try:
            import zipfile
            zf = zipfile.ZipFile(local_path, 'r')
            # Verify integrity of local copy (checks CRC-32 and file headers)
            bad_file = zf.testzip()
            if bad_file:
                raise ValueError(f"Local file copy is corrupted (bad entry: {bad_file})")
            
            self.zip_handles[name] = zf
            print(f"[ZipExtractor] Connected on-demand reader for: {local_path.name}")
            return zf
        except Exception as e:
            print(f"[ZipExtractor] Local copy check failed for {name}: {e}. Falling back to Google Drive streaming...")
            # Delete corrupted local copy to clean the space
            if local_path.exists() and local_path != drive_path:
                try:
                    local_path.unlink()
                except Exception:
                    pass
            try:
                import zipfile
                zf = zipfile.ZipFile(drive_path, 'r')
                self.zip_handles[name] = zf
                print(f"[ZipExtractor] Connected on-demand reader directly to Drive: {drive_path.name}")
                return zf
            except Exception as ex:
                print(f"[ZipExtractor] Failed to open {name} directly from Drive: {ex}")
                return None

    def extract_if_needed(self, local_path_str: str) -> str:
        local_path = Path(local_path_str)
        if local_path.exists():
            return local_path_str

        # Get path relative to the REPO_ROOT (thesis folder)
        try:
            rel_path = local_path.relative_to(REPO_ROOT)
            rel_path_str = str(rel_path).replace("\\", "/")
        except Exception:
            rel_path_str = str(local_path).replace("\\", "/")

        # Deduce which ZIP is needed
        zip_name = self._get_target_zip_name(rel_path_str)
        if not zip_name or zip_name not in self.drive_paths:
            return local_path_str
            
        # Lazy-load only this specific ZIP!
        zh = self._lazy_load_zip(zip_name)
        if not zh:
            return local_path_str

        # Internal paths might not include 'data/' prefix depending on how it was zipped
        internal_paths = [
            rel_path_str,
            rel_path_str.replace("data/", ""),
            rel_path_str.replace("data/raw/", "")
        ]
        ip_found = None
        for ip in internal_paths:
            try:
                zh.getinfo(ip)
                ip_found = ip
                break
            except KeyError:
                continue

        # Fallback search if direct match fails
        if ip_found is None:
            try:
                target_suffix = "/" + local_path.name
                for name in zh.namelist():
                    if name.endswith(target_suffix) or name == local_path.name:
                        ip_found = name
                        break
            except Exception:
                pass

        if ip_found:
            # Found it! Extract to local temp directory
            try:
                temp_dir = REPO_ROOT / "data/temp_extraction"
                temp_dir.mkdir(parents=True, exist_ok=True)
                extracted = zh.extract(ip_found, temp_dir)
                return extracted
            except Exception as e:
                print(f"[ZipExtractor] Error extracting {ip_found} from {zip_name}: {e}")

        return local_path_str

    def cleanup_temp_file(self, path_str: str):
        if "temp_extraction" in path_str:
            try:
                p = Path(path_str)
                if p.exists() and p.is_file():
                    p.unlink()
                # Clean up parent dirs if empty
                for parent in p.parents:
                    if parent.name == "temp_extraction":
                        break
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
            except Exception:
                pass


def build_datasets(cfg: Config, seed: int = 42, no_sarcasm: bool = False):
    # Guarantee split manifests exist
    if not TRAIN_MANIFEST_CSV.exists() or not VAL_MANIFEST_CSV.exists() or not TEST_MANIFEST_CSV.exists():
        _section("Generating Speaker-Stratified Split Manifests")
        from scripts.create_dataset_splits import create_splits
        create_splits(seed=seed)

    _section("Loading dataset split manifests")

    train_recs = _load_manifest_records(TRAIN_MANIFEST_CSV)
    val_recs   = _load_manifest_records(VAL_MANIFEST_CSV)
    test_recs  = _load_manifest_records(TEST_MANIFEST_CSV)

    if no_sarcasm:
        train_recs = [r for r in train_recs if r["source_pipeline"] != "mustard"]
        val_recs   = [r for r in val_recs if r["source_pipeline"] != "mustard"]
        test_recs  = [r for r in test_recs if r["source_pipeline"] != "mustard"]

    all_recs = train_recs + val_recs + test_recs
    all_spks = set(r["speaker_id"] for r in all_recs)

    real_train = sum(1 for r in train_recs if r["fake_label"] == 0)
    fake_train = sum(1 for r in train_recs if r["fake_label"] == 1)
    sarc_train = sum(1 for r in train_recs if r["sarcasm_label"] != UNKNOWN_SARCASM)

    real_test  = sum(1 for r in test_recs if r["fake_label"] == 0)
    fake_test  = sum(1 for r in test_recs if r["fake_label"] == 1)

    src_counts = Counter(r["source_pipeline"] for r in train_recs)
    auto_pw = real_train / max(fake_train, 1)  # Exactly 1.3835 (8254 real / 5966 fake)

    stats = {
        "total": len(all_recs),
        "train": len(train_recs),
        "val": len(val_recs),
        "test": len(test_recs),
        "n_speakers": len(all_spks),
        "real_train": real_train,
        "fake_train": fake_train,
        "sarc_train": sarc_train,
        "real_test": real_test,
        "fake_test": fake_test,
        "src_counts": dict(src_counts),
        "auto_pos_weight": auto_pw,
    }

    # ── Dataset class ────────────────────────────────────────────────────────
    class FullDataset(torch.utils.data.Dataset):
        def __init__(self, recs):
            self._records = recs
        def __len__(self):
            return len(self._records)
        def __getitem__(self, i):
            r = self._records[i]
            return {
                "z_at":            torch.load(r["z_at_path"], weights_only=True).float(),
                "z_v":             torch.load(r["z_v_path"],  weights_only=True).float(),
                "fake_label":      torch.tensor(r["fake_label"],    dtype=torch.long),
                "audio_emotion":   torch.tensor(r["audio_emotion"], dtype=torch.long),
                "visual_emotion":  torch.tensor(r["visual_emotion"],dtype=torch.long),
                "sarcasm_label":   torch.tensor(r["sarcasm_label"], dtype=torch.long),
                "source_pipeline": r["source_pipeline"],
                "clip_id":         r["clip_id"],
                "speaker_id":      r["speaker_id"],
            }

    # ── Phase 2 dataset (raw inputs for backbone fine-tuning) ────────────────
    class Phase2FullDataset(torch.utils.data.Dataset):
        WAV2VEC_MODEL = "facebook/wav2vec2-base"
        BERT_MODEL    = "bert-base-uncased"
        VIT_MODEL     = "google/vit-base-patch16-224"
        MAX_AUDIO     = 80000
        MAX_SEQ_LEN   = 128
        N_KEYFRAMES   = 8
        FRAME_SIZE    = 224

        def __init__(self, recs, preprocessed_dir: Path, keyframe_cache_dir: Path):
            self._recs   = recs
            self._prep   = preprocessed_dir
            self._kf_dir = keyframe_cache_dir
            self._kf_dir.mkdir(parents=True, exist_ok=True)
            from transformers import Wav2Vec2FeatureExtractor, BertTokenizer, ViTImageProcessor
            self._wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained(self.WAV2VEC_MODEL)
            self._bert_tok     = BertTokenizer.from_pretrained(self.BERT_MODEL)
            self._vit_proc     = ViTImageProcessor.from_pretrained(self.VIT_MODEL)
            self._zip_extractor = ZipExtractor()

        def __len__(self):
            return len(self._recs)

        def __getitem__(self, i):
            import random
            r = self._recs[i]
            cid = r["clip_id"]
            vp  = r["video_path"]
            fake_label = r["fake_label"]
            audio_cid = cid
            audio_vp  = vp

            # Audio-Visual Swap Augmentation (40% probability for real clips)
            if fake_label == 0 and random.random() < 0.4:
                other_idx = random.randint(0, len(self._recs) - 1)
                other_r = self._recs[other_idx]
                audio_cid = other_r["clip_id"]
                audio_vp  = other_r["video_path"]
                fake_label = 1.0  # Synthetic mismatch fake

            return {
                "audio_values":   self._audio_values(audio_cid, audio_vp),
                "input_ids":      self._bert_inputs(audio_cid)[0],
                "attention_mask": self._bert_inputs(audio_cid)[1],
                "keyframe_pixels":self._keyframe_pixels(cid, vp),
                "fake_label":     torch.tensor(fake_label,         dtype=torch.long),
                "audio_emotion":  torch.tensor(r["audio_emotion"], dtype=torch.long),
                "visual_emotion": torch.tensor(r["visual_emotion"],dtype=torch.long),
                "sarcasm_label":  torch.tensor(r["sarcasm_label"], dtype=torch.long),
                "source_pipeline":r["source_pipeline"],
                "clip_id":        cid,
                "speaker_id":     r["speaker_id"],
            }

        def _audio_values(self, clip_id: str, video_path: str) -> torch.Tensor:
            import torchaudio
            wav_path = self._prep / "audio" / f"{clip_id}.wav"
            
            # If the wav already exists, load it directly
            if wav_path.exists():
                try:
                    wav, sr = torchaudio.load(str(wav_path))
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    enc = self._wav2vec_proc(
                        wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt",
                        padding="max_length", max_length=self.MAX_AUDIO, truncation=True,
                    )
                    return enc.input_values.squeeze(0)
                except Exception:
                    pass

            # Otherwise, extract from zip on Google Drive if needed
            working_path = self._zip_extractor.extract_if_needed(video_path)
            
            try:
                if Path(working_path).exists():
                    wav, sr = torchaudio.load(str(working_path))
                    # Save mono resampled audio wav file to cache so we don't need the video next time
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                        sr = 16000
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(wav_path), wav, sr)
                else:
                    print(f"\n[WARNING] Missing audio source for clip '{clip_id}'. Returning zero tensor.")
                    return torch.zeros(self.MAX_AUDIO)
            except Exception as e:
                print(f"\n[ERROR] Failed to load/decode audio for '{clip_id}' ({e}). Returning zero tensor.")
                return torch.zeros(self.MAX_AUDIO)

            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            enc = self._wav2vec_proc(
                wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt",
                padding="max_length", max_length=self.MAX_AUDIO, truncation=True,
            )
            return enc.input_values.squeeze(0)

        def _bert_inputs(self, clip_id: str):
            txt_path = self._prep / "transcripts" / f"{clip_id}.txt"
            if not txt_path.exists():
                log.debug(f"Transcript text file missing for '{clip_id}' at: {txt_path}")
            text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            enc  = self._bert_tok(
                text, return_tensors="pt",
                padding="max_length", max_length=self.MAX_SEQ_LEN, truncation=True,
            )
            return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

        def _keyframe_pixels(self, clip_id: str, video_path: str) -> torch.Tensor:
            from PIL import Image
            kf_path = self._kf_dir / f"{clip_id}.jpg"
            
            # If the cached JPEG grid already exists, load and slice it
            if kf_path.exists():
                try:
                    vp_path = Path(video_path)
                    if vp_path.exists() and vp_path.is_file():
                        vp_path.unlink()
                except Exception:
                    pass
                
                try:
                    grid_img = Image.open(kf_path)
                    pils = []
                    for idx in range(self.N_KEYFRAMES):
                        crop_box = (self.FRAME_SIZE * idx, 0, self.FRAME_SIZE * (idx + 1), self.FRAME_SIZE)
                        pils.append(grid_img.crop(crop_box))
                    enc = self._vit_proc(pils, return_tensors="pt")
                    return enc.pixel_values
                except Exception as e:
                    print(f"[WARNING] Failed to load cached JPEG for {clip_id} ({e}). Re-extracting...")
            
            # Extract from zip if needed
            working_path = self._zip_extractor.extract_if_needed(video_path)
            
            from src.preprocessing.visual import extract_frames
            from src.preprocessing.filters import frames_to_pil
            
            try:
                frames = extract_frames(working_path, target_fps=25.0)
            except Exception as e:
                print(f"[ERROR] Video frame extraction failed for {clip_id}: {e}")
                frames = []

            if not frames:
                # Save a black placeholder grid image
                black_grid = Image.new('RGB', (self.FRAME_SIZE * self.N_KEYFRAMES, self.FRAME_SIZE), (0, 0, 0))
                black_grid.save(kf_path, 'JPEG', quality=90)
                self._zip_extractor.cleanup_temp_file(working_path)
                try:
                    vp_path = Path(video_path)
                    if vp_path.exists() and vp_path.is_file():
                        vp_path.unlink()
                except Exception:
                    pass
                pils = [Image.new('RGB', (self.FRAME_SIZE, self.FRAME_SIZE), (0, 0, 0)) for _ in range(self.N_KEYFRAMES)]
                enc = self._vit_proc(pils, return_tensors="pt")
                return enc.pixel_values

            pils = frames_to_pil(frames[:self.N_KEYFRAMES])
            # Pad if fewer frames
            while len(pils) < self.N_KEYFRAMES:
                pils.append(pils[-1].copy() if pils else Image.new('RGB', (self.FRAME_SIZE, self.FRAME_SIZE)))
            
            # Concatenate horizontally
            grid_img = Image.new('RGB', (self.FRAME_SIZE * self.N_KEYFRAMES, self.FRAME_SIZE))
            for idx, img in enumerate(pils):
                grid_img.paste(img, (self.FRAME_SIZE * idx, 0))
            grid_img.save(kf_path, 'JPEG', quality=90)
            
            # Cleanup temporary extraction files
            self._zip_extractor.cleanup_temp_file(working_path)
            
            # Also clean up any unzipped video_path if unzipped by colab
            try:
                vp_path = Path(video_path)
                if vp_path.exists() and vp_path.is_file():
                    vp_path.unlink()
            except Exception:
                pass
                
            enc = self._vit_proc(pils, return_tensors="pt")
            return enc.pixel_values

    train_ds = FullDataset(train_recs)
    val_ds   = FullDataset(val_recs)
    test_ds  = FullDataset(test_recs)

    return train_ds, val_ds, test_ds, stats, Phase2FullDataset, train_recs, val_recs


def main():
    parser = argparse.ArgumentParser(description="Full-dataset training for DeepSentinel.")
    parser.add_argument("--epochs",               type=int,   default=30)
    parser.add_argument("--batch_size",           type=int,   default=32)
    parser.add_argument("--lr",                   type=float, default=1e-3)
    parser.add_argument("--patience",             type=int,   default=7)
    parser.add_argument("--pos_weight",           type=float, default=None)
    parser.add_argument("--no_sarcasm",           action="store_true")
    parser.add_argument("--no_phase2",            action="store_true")
    parser.add_argument("--skip_phase1",          action="store_true")
    parser.add_argument("--phase2_epochs",        type=int,   default=15)
    parser.add_argument("--phase2_batch",         type=int,   default=4)
    parser.add_argument("--phase2_lr",            type=float, default=3e-6)
    parser.add_argument("--phase2_freeze_layers", type=int,   default=4)
    parser.add_argument("--no_grad_ckpt",         action="store_true")
    parser.add_argument("--workers",              type=int,   default=0)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--device",               type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classifier_mode",      type=str,   default="baseline", choices=["baseline", "mismatch_only", "emotion_bilinear", "bottleneck", "high_dropout"])
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("\n  [WARNING] CUDA requested but not available. Falling back to CPU.")
        print("  NOTE: In Google Colab, select 'Runtime -> Change runtime type -> T4 GPU' to enable GPU acceleration.\n")
        args.device = "cpu"

    _section("DeepSentinel Full Dataset Training")
    print(f"  Device    : {args.device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")

    cfg = Config()

    train_ds, val_ds, test_ds, stats, Phase2FullDataset, p2_train_recs, p2_val_recs = build_datasets(
        cfg,
        seed=args.seed,
        no_sarcasm=args.no_sarcasm,
    )

    effective_pw = args.pos_weight if args.pos_weight is not None else stats["auto_pos_weight"]
    pw_src = "manual override" if args.pos_weight is not None else f"auto = {stats['real_train']}/{stats['fake_train']} = n_real/n_fake"
    print(f"  pos_weight: {effective_pw:.4f}  ({pw_src})")

    _section("Dataset split summary")
    print(f"  Total clips          : {stats['total']}")
    print(f"  Speakers             : {stats['n_speakers']}")
    print(f"  Train / Val / Test   : {stats['train']} / {stats['val']} / {stats['test']}")
    print(f"\n  Train breakdown:")
    print(f"    Real clips         : {stats['real_train']}")
    print(f"    Fake clips         : {stats['fake_train']}")
    print(f"    Sarcasm annotated  : {stats['sarc_train']}")
    print(f"    By source:")
    for src, n in sorted(stats["src_counts"].items()):
        print(f"      {src:<22} {n}")
    print(f"\n  Test breakdown:")
    print(f"    Real clips         : {stats['real_test']}")
    print(f"    Fake clips         : {stats['fake_test']}")
    both = stats["real_test"] > 0 and stats["fake_test"] > 0
    print(f"    AUC feasible       : {'YES' if both else 'NO — missing one class in test'}")

    if stats["train"] == 0:
        print("\n  ERROR: No training clips. Run: python scripts/validate_training_prep.py")
        return

    p1_workers = min(args.workers, 2) if args.device.startswith("cuda") else 0
    loader_kw  = dict(batch_size=args.batch_size, num_workers=p1_workers,
                      pin_memory=args.device.startswith("cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    _section("Initializing DeepfakeDetector")
    model = DeepfakeDetector(classifier_mode=args.classifier_mode)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {train_params:,}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        checkpoint_dir = CKPT_DIR,
        log_dir        = LOG_DIR,
        fp16           = args.device.startswith("cuda"),
        lambda_a       = 0.1,
        lambda_b       = 0.1,
        lambda_sarcasm = 0.05,
        pos_weight     = effective_pw,
        device         = args.device,
        ckpt_suffix    = args.classifier_mode,
    )

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.skip_phase1:
        _section("Phase 1 — Skipping Phase 1 (Heads + Classifier) training")
        try:
            trainer.load_best(phase=1)
            print("  Successfully loaded Phase 1 checkpoint from local checkpoints directory.")
        except Exception as e:
            print(f"  [WARNING] Failed to load Phase 1 checkpoint ({e}). Running Phase 1 training instead...")
            trainer.train_phase1(
                lr           = args.lr,
                weight_decay = 1e-4,
                max_epochs   = args.epochs,
                patience     = args.patience,
            )
    else:
        _section("Phase 1 — Heads + Classifier (frozen backbones)")
        trainer.train_phase1(
            lr           = args.lr,
            weight_decay = 1e-4,
            max_epochs   = args.epochs,
            patience     = args.patience,
        )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    best_phase = 1
    if not args.no_phase2:
        _section("Phase 2 — Backbone fine-tuning")
        print(f"  Train clips       : {len(p2_train_recs)}")
        print(f"  Batch size        : {args.phase2_batch}")
        print(f"  Freeze layers     : top-{args.phase2_freeze_layers} only (0=all unfrozen)")
        print(f"  Grad checkpointing: {'OFF (--no_grad_ckpt)' if args.no_grad_ckpt else 'ON (saves VRAM)'}")
        print(f"  Keyframe cache    : {KEYFRAME_CACHE_DIR}")

        # Filter out MOSEI records from Phase 2 training if NEITHER raw video files NOR cmumosei.zip are available
        mosei_raw_dir = REPO_ROOT / "data/raw/CMU-MOSEI"
        drive_mosei_zip = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/cmumosei.zip")
        drive_mosei_zip_ds = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/cmumosei.zip")
        local_mosei_zip = REPO_ROOT / "data/zips/cmumosei.zip"
        
        has_mosei_source = (
            mosei_raw_dir.exists() or 
            drive_mosei_zip.exists() or 
            drive_mosei_zip_ds.exists() or
            local_mosei_zip.exists()
        )
        
        if not has_mosei_source:
            print(f"\n[INFO] Raw CMU-MOSEI video source or cmumosei.zip archive not found.")
            print("       Excluding MOSEI clips from Phase 2 backbone fine-tuning.")
            p2_train_recs = [r for r in p2_train_recs if r["source_pipeline"] != "mosei"]
        else:
            print(f"\n[INFO] Found CMU-MOSEI source archive/directory. Including MOSEI clips in Phase 2!")

        p2_ds = Phase2FullDataset(p2_train_recs, PREPROCESSED_DIR, KEYFRAME_CACHE_DIR)
        p2_val_ds = Phase2FullDataset(p2_val_recs, PREPROCESSED_DIR, KEYFRAME_CACHE_DIR)
        p2_workers = min(args.workers, 1) if args.device.startswith("cuda") else 0
        p2_loader  = DataLoader(p2_ds, shuffle=True,
                                batch_size=args.phase2_batch,
                                num_workers=p2_workers,
                                pin_memory=args.device.startswith("cuda"))
        p2_val_loader = DataLoader(p2_val_ds, shuffle=False,
                                   batch_size=args.phase2_batch,
                                   num_workers=p2_workers,
                                   pin_memory=args.device.startswith("cuda"))
        trainer.train_loader = p2_loader
        trainer.val_loader_p2 = p2_val_loader
        trainer.train_phase2(
            lr            = args.phase2_lr,
            weight_decay  = 1e-4,
            max_epochs    = args.phase2_epochs,
            patience      = args.patience,
            freeze_layers = args.phase2_freeze_layers,
            grad_ckpt     = not args.no_grad_ckpt,
        )
        best_phase = 2

    # ── Test evaluation ───────────────────────────────────────────────────────
    _section("Test set evaluation")
    trainer.load_best(phase=best_phase)
    model.eval()

    correct, total, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            z_at = batch["z_at"].to(args.device)
            z_v  = batch["z_v"].to(args.device)
            fl   = batch["fake_label"].to(args.device)
            valid = fl != -1
            if not valid.any():
                continue
            out   = model.forward_from_features(z_at[valid], z_v[valid])
            probs = torch.sigmoid(out.logit.squeeze(1))
            preds = (probs >= 0.5).long()
            labs  = fl[valid]
            correct += (preds == labs).sum().item()
            total   += labs.size(0)
            tp += ((preds == 1) & (labs == 1)).sum().item()
            fp += ((preds == 1) & (labs == 0)).sum().item()
            fn += ((preds == 0) & (labs == 1)).sum().item()
            tn += ((preds == 0) & (labs == 0)).sum().item()
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(labs.cpu().tolist())

    if total > 0:
        acc  = correct / total
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"  Test clips    : {total}")
        print(f"  Accuracy      : {acc:.4f}")
        print(f"  Precision     : {prec:.4f}")
        print(f"  Recall        : {rec:.4f}")
        print(f"  F1            : {f1:.4f}")
        print(f"  TP/FP/FN/TN   : {tp}/{fp}/{fn}/{tn}")
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_scores)
            print(f"  AUC-ROC       : {auc:.4f}")

            # Bootstrap 95% CI
            rng2 = np.random.default_rng(args.seed)
            arr_l = np.array(all_labels)
            arr_s = np.array(all_scores)
            boot_aucs = []
            for _ in range(10000):
                idx = rng2.integers(0, len(arr_l), len(arr_l))
                if arr_l[idx].sum() == 0 or arr_l[idx].sum() == len(arr_l):
                    continue
                boot_aucs.append(roc_auc_score(arr_l[idx], arr_s[idx]))
            if boot_aucs:
                lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
                print(f"  95% CI        : [{lo:.4f}, {hi:.4f}]")
        except Exception as e:
            print(f"  AUC-ROC       : (error: {e})")
    else:
        print("  No valid test clips.")

    _section("Full training complete")
    ckpt_name = f"best_phase1_{args.classifier_mode}.pt" if args.classifier_mode else "best_phase1.pt"
    print(f"  Checkpoint : {CKPT_DIR}/{ckpt_name}")
    print(f"  Logs       : {LOG_DIR}/")
    print(f"  Next step  : python scripts/evaluate_fakeavceleb.py --checkpoint {CKPT_DIR}/{ckpt_name}\n")


if __name__ == "__main__":
    main()
