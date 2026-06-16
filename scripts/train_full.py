"""
train_full.py — Full-dataset training for DeepSentinel.

Loads all preprocessed clips from Tracks 1-4, MELD real, CMU-MOSEI real,
and MUStARD sarcasm. Uses speaker-stratified 80/10/10 split.
Phase 1: frozen backbones, train heads + classifier on cached z_at/z_v.
Phase 2: optional backbone fine-tuning (end-to-end, slow).

Usage:
    python scripts/train_full.py
    python scripts/train_full.py --device cuda --epochs 50 --batch_size 32
    python scripts/train_full.py --no_phase2
    python scripts/train_full.py --no_track4   # exclude Track4 (conflicting signal)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
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

SECTION = "=" * 60


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(f"{SECTION}")


def _emo(code) -> int:
    if code is None or (isinstance(code, float) and code != code):
        return UNKNOWN_EMOTION
    return EMOTION_TO_IDX.get(str(code).strip().upper(), UNKNOWN_EMOTION)


def _load_tracks(cfg: Config, include_track4: bool, cached: set[str]) -> list[dict]:
    records = []

    # Tracks 1-3: CREMA-D fakes
    emo_map = {"ANG": "ANG", "DIS": "DIS", "FEA": "FEA", "HAP": "HAP", "NEU": "NEU", "SAD": "SAD"}
    for track_name, meta_path in [
        ("track1", cfg.paths.track1_meta),
        ("track2", cfg.paths.track2_meta),
        ("track3", cfg.paths.track3_meta),
    ]:
        p = Path(meta_path)
        if not p.exists():
            log.warning(f"SKIP {track_name}: {p} not found")
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = str(row["output_stem"])
                if stem not in cached:
                    continue
                z_at = PREPROCESSED_DIR / "features/z_at" / f"{stem}.pt"
                z_v  = PREPROCESSED_DIR / "features/z_v"  / f"{stem}.pt"
                if not z_at.exists() or not z_v.exists():
                    continue
                # Actor ID from stem or actor_id column
                actor_id = str(row.get("actor_id", "") or stem.split("_")[2] if len(stem.split("_")) > 2 else stem)
                spk = f"crema_{actor_id}"
                records.append({
                    "clip_id":         stem,
                    "z_at_path":       str(z_at),
                    "z_v_path":        str(z_v),
                    "video_path":      str(row["output_path"]),
                    "fake_label":      int(row.get("label", 1)),
                    "audio_emotion":   _emo(row.get("audio_emotion")),
                    "visual_emotion":  _emo(row.get("video_emotion")),
                    "sarcasm_label":   UNKNOWN_SARCASM,
                    "source_pipeline": track_name,
                    "speaker_id":      spk,
                })

    # Track 4: MELD emotion-mismatch fakes
    if include_track4:
        p = Path(cfg.paths.track4_meta)
        if p.exists():
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stem = str(row["output_stem"])
                    if stem not in cached:
                        continue
                    z_at = PREPROCESSED_DIR / "features/z_at" / f"{stem}.pt"
                    z_v  = PREPROCESSED_DIR / "features/z_v"  / f"{stem}.pt"
                    if not z_at.exists() or not z_v.exists():
                        continue
                    spk = f"meld_fake_{row.get('video_speaker', 'UNK')}"
                    records.append({
                        "clip_id":         stem,
                        "z_at_path":       str(z_at),
                        "z_v_path":        str(z_v),
                        "video_path":      str(row["output_path"]),
                        "fake_label":      int(row.get("label", 1)),
                        "audio_emotion":   _emo(row.get("audio_emotion")),
                        "visual_emotion":  _emo(row.get("video_emotion")),
                        "sarcasm_label":   UNKNOWN_SARCASM,
                        "source_pipeline": "track4",
                        "speaker_id":      spk,
                    })

    return records


def _load_meld_real(cfg: Config, cached: set[str]) -> list[dict]:
    records = []
    p = Path(cfg.paths.meld_real_csv)
    if not p.exists():
        log.warning(f"SKIP meld_real: {p} not found")
        return records
    meld_emo_map = {
        "neutral": "NEU", "happy": "HAP", "sad": "SAD",
        "angry": "ANG", "fear": "FEA", "disgust": "DIS", "surprise": "NEU",
    }
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row["clip_id"])
            if cid not in cached:
                continue
            z_at = PREPROCESSED_DIR / "features/z_at" / f"{cid}.pt"
            z_v  = PREPROCESSED_DIR / "features/z_v"  / f"{cid}.pt"
            if not z_at.exists() or not z_v.exists():
                continue
            raw_emo = str(row.get("emotion", "neutral")).lower()
            emo_code = meld_emo_map.get(raw_emo, "NEU")
            spk = f"meld_{str(row.get('speaker', 'UNK')).replace(' ', '_')}"
            records.append({
                "clip_id":         cid,
                "z_at_path":       str(z_at),
                "z_v_path":        str(z_v),
                "video_path":      str(row["video_path"]),
                "fake_label":      0,
                "audio_emotion":   _emo(emo_code),
                "visual_emotion":  _emo(emo_code),
                "sarcasm_label":   UNKNOWN_SARCASM,
                "source_pipeline": "meld_real",
                "speaker_id":      spk,
            })
    return records


def _load_mosei_real(cfg: Config, cached: set[str]) -> list[dict]:
    records = []
    p = Path(cfg.paths.mosei_real_csv)
    if not p.exists():
        log.warning(f"SKIP mosei_real: {p} not found")
        return records
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row["clip_id"])
            if cid not in cached:
                continue
            z_at = PREPROCESSED_DIR / "features/z_at" / f"{cid}.pt"
            z_v  = PREPROCESSED_DIR / "features/z_v"  / f"{cid}.pt"
            if not z_at.exists() or not z_v.exists():
                continue
            records.append({
                "clip_id":         cid,
                "z_at_path":       str(z_at),
                "z_v_path":        str(z_v),
                "video_path":      str(row["video_path"]),
                "fake_label":      0,
                "audio_emotion":   UNKNOWN_EMOTION,
                "visual_emotion":  UNKNOWN_EMOTION,
                "sarcasm_label":   UNKNOWN_SARCASM,
                "source_pipeline": "mosei_real",
                "speaker_id":      f"mosei_{cid}",
            })
    return records


def _load_mustard(cached: set[str]) -> list[dict]:
    records = []
    # Try full mustard CSV via config, fall back to smoke sarcasm manifest
    mustard_json = REPO_ROOT / "data/raw/MUStARD/repo/data/sarcasm_data.json"
    mustard_vid  = REPO_ROOT / "data/raw/MUStARD/raw_data/utterances_final"
    if mustard_json.exists() and mustard_vid.exists():
        import json
        data = json.loads(mustard_json.read_text(encoding="utf-8"))
        for key, entry in data.items():
            cid = key
            if cid not in cached:
                # try stem match
                mp4 = mustard_vid / f"{key}.mp4"
                if not mp4.exists():
                    continue
                cid = key
            z_at = PREPROCESSED_DIR / "features/z_at" / f"{cid}.pt"
            z_v  = PREPROCESSED_DIR / "features/z_v"  / f"{cid}.pt"
            if not z_at.exists() or not z_v.exists():
                continue
            sarc = 1 if entry.get("sarcasm", False) else 0
            spk  = f"mustard_{str(entry.get('speaker', 'UNK')).replace(' ', '_')}"
            records.append({
                "clip_id":         cid,
                "z_at_path":       str(z_at),
                "z_v_path":        str(z_v),
                "video_path":      str(mustard_vid / f"{key}.mp4"),
                "fake_label":      -1,
                "audio_emotion":   UNKNOWN_EMOTION,
                "visual_emotion":  UNKNOWN_EMOTION,
                "sarcasm_label":   sarc,
                "source_pipeline": "mustard",
                "speaker_id":      spk,
            })
        return records

    # fallback: smoke sarcasm CSV
    fallback = REPO_ROOT / "data/processed/smoke_manifests/smoke_sarcasm.csv"
    if fallback.exists():
        with open(fallback, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = str(row.get("path", ""))
                if not path:
                    continue
                cid = Path(path).stem
                z_at = PREPROCESSED_DIR / "features/z_at" / f"{cid}.pt"
                z_v  = PREPROCESSED_DIR / "features/z_v"  / f"{cid}.pt"
                if not z_at.exists() or not z_v.exists():
                    continue
                sarc = int(row.get("sarcasm_label", UNKNOWN_SARCASM))
                spk  = str(row.get("speaker_id", f"mustard_{cid}"))
                records.append({
                    "clip_id":         cid,
                    "z_at_path":       str(z_at),
                    "z_v_path":        str(z_v),
                    "video_path":      path,
                    "fake_label":      -1,
                    "audio_emotion":   UNKNOWN_EMOTION,
                    "visual_emotion":  UNKNOWN_EMOTION,
                    "sarcasm_label":   sarc,
                    "source_pipeline": "mustard",
                    "speaker_id":      spk,
                })
    return records


def build_datasets(cfg: Config, seed: int = 42, include_track4: bool = True, no_sarcasm: bool = False):
    cached = set(p.stem for p in (PREPROCESSED_DIR / "features/z_at").glob("*.pt"))

    _section("Loading full dataset manifests")

    records: list[dict] = []

    track_recs = _load_tracks(cfg, include_track4, cached)
    records.extend(track_recs)
    print(f"  Tracks 1-{'4' if include_track4 else '3'} (fakes) : {len(track_recs)}")

    meld_recs = _load_meld_real(cfg, cached)
    records.extend(meld_recs)
    print(f"  MELD real            : {len(meld_recs)}")

    mosei_recs = _load_mosei_real(cfg, cached)
    records.extend(mosei_recs)
    print(f"  CMU-MOSEI real       : {len(mosei_recs)}")

    if not no_sarcasm:
        must_recs = _load_mustard(cached)
        records.extend(must_recs)
        print(f"  MUStARD sarcasm      : {len(must_recs)}")

    print(f"  Total records        : {len(records)}")

    # ── Speaker-stratified 80/10/10 split ────────────────────────────────────
    rng = np.random.default_rng(seed)
    spk_map: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        spk_map[r["speaker_id"]].append(i)

    def _split_speakers(spk_list, train_frac=0.80, val_frac=0.10):
        arr = list(spk_list)
        rng.shuffle(arr)
        n    = len(arr)
        n_tr = max(1, int(n * train_frac))
        n_va = max(1, int(n * val_frac))
        return set(arr[:n_tr]), set(arr[n_tr:n_tr + n_va]), set(arr[n_tr + n_va:])

    real_det_spk = [s for s, idxs in spk_map.items()
                    if any(records[i]["fake_label"] == 0 for i in idxs)]
    fake_det_spk = [s for s, idxs in spk_map.items()
                    if any(records[i]["fake_label"] == 1 for i in idxs)]
    must_spk     = [s for s, idxs in spk_map.items()
                    if all(records[i]["fake_label"] == -1 for i in idxs)]

    tr_r, va_r, te_r = _split_speakers(real_det_spk)
    tr_f, va_f, te_f = _split_speakers(fake_det_spk)
    tr_m, va_m, te_m = _split_speakers(must_spk)

    train_spk = tr_r | tr_f | tr_m
    val_spk   = va_r | va_f | va_m

    train_idx, val_idx, test_idx = [], [], []
    for spk, idxs in spk_map.items():
        if spk in train_spk:   train_idx.extend(idxs)
        elif spk in val_spk:   val_idx.extend(idxs)
        else:                  test_idx.extend(idxs)

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

        def __len__(self):
            return len(self._recs)

        def _audio_values(self, clip_id: str) -> torch.Tensor:
            import torchaudio
            wav_path = self._prep / "audio" / f"{clip_id}.wav"
            wav, sr  = torchaudio.load(str(wav_path))
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
            text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            enc  = self._bert_tok(
                text, return_tensors="pt",
                padding="max_length", max_length=self.MAX_SEQ_LEN, truncation=True,
            )
            return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

        def _keyframe_pixels(self, clip_id: str, video_path: str) -> torch.Tensor:
            kf_path = self._kf_dir / f"{clip_id}.pt"
            if kf_path.exists():
                return torch.load(kf_path, weights_only=True)
            import numpy as np
            from src.preprocessing.visual import (
                extract_frames, optical_flow_gate, detect_and_align_faces,
            )
            from src.preprocessing.filters import select_keyframes, frames_to_pil
            frames = extract_frames(video_path, target_fps=25.0)
            if not frames:
                frames = [np.zeros((self.FRAME_SIZE, self.FRAME_SIZE, 3), dtype=np.uint8)]
            gated        = optical_flow_gate(frames, 0.3)
            face_results = detect_and_align_faces(gated, "insightface", 0.7)
            if not face_results:
                face_results = detect_and_align_faces(gated, "insightface", 0.0)
            if not face_results:
                face_results = [(f, 1.0) for f in (gated or frames)]
            crops     = [r[0] for r in face_results]
            scores    = [r[1] for r in face_results]
            keyframes = select_keyframes(crops, scores, k=self.N_KEYFRAMES)
            pil_imgs  = frames_to_pil(keyframes, size=self.FRAME_SIZE)
            pixels    = self._vit_proc(images=pil_imgs, return_tensors="pt").pixel_values
            torch.save(pixels, kf_path)
            return pixels

        def __getitem__(self, i):
            r       = self._recs[i]
            cid     = r["clip_id"]
            ids, am = self._bert_inputs(cid)
            return {
                "audio_values":    self._audio_values(cid),
                "input_ids":       ids,
                "attention_mask":  am,
                "keyframe_pixels": self._keyframe_pixels(cid, r["video_path"]),
                "fake_label":      torch.tensor(r["fake_label"],    dtype=torch.long),
                "audio_emotion":   torch.tensor(r["audio_emotion"], dtype=torch.long),
                "visual_emotion":  torch.tensor(r["visual_emotion"],dtype=torch.long),
                "sarcasm_label":   torch.tensor(r["sarcasm_label"], dtype=torch.long),
                "source_pipeline": r["source_pipeline"],
                "clip_id":         cid,
            }

    train_ds      = FullDataset([records[i] for i in train_idx])
    val_ds        = FullDataset([records[i] for i in val_idx])
    test_ds       = FullDataset([records[i] for i in test_idx])
    p2_train_recs = [records[i] for i in train_idx]

    src_counts = Counter(records[i]["source_pipeline"] for i in train_idx)
    real_train = sum(1 for i in train_idx if records[i]["fake_label"] == 0)
    fake_train = sum(1 for i in train_idx if records[i]["fake_label"] == 1)
    sarc_train = sum(1 for i in train_idx if records[i]["sarcasm_label"] != UNKNOWN_SARCASM)
    real_test  = sum(1 for i in test_idx  if records[i]["fake_label"] == 0)
    fake_test  = sum(1 for i in test_idx  if records[i]["fake_label"] == 1)

    auto_pos_weight = real_train / fake_train if fake_train > 0 else 1.0

    stats = {
        "total":          len(records),
        "train":          len(train_idx),
        "val":            len(val_idx),
        "test":           len(test_idx),
        "n_speakers":     len(spk_map),
        "real_train":     real_train,
        "fake_train":     fake_train,
        "sarc_train":     sarc_train,
        "real_test":      real_test,
        "fake_test":      fake_test,
        "src_counts":     dict(src_counts),
        "auto_pos_weight": auto_pos_weight,
    }
    return train_ds, val_ds, test_ds, stats, Phase2FullDataset, p2_train_recs


def main():
    parser = argparse.ArgumentParser(description="Full-dataset DeepSentinel training")
    parser.add_argument("--config",         default=None)
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device",         default=_default_device)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--workers",        type=int,   default=0)
    parser.add_argument("--no_sarcasm",     action="store_true")
    parser.add_argument("--no_track4",      action="store_true",
                        help="Exclude Track 4 fakes (emotion-mismatch MELD — conflicting signal at early training)")
    parser.add_argument("--no_phase2",      action="store_true")
    parser.add_argument("--phase2_epochs",        type=int,   default=10)
    parser.add_argument("--phase2_lr",            type=float, default=1e-5)
    parser.add_argument("--phase2_batch",         type=int,   default=1)
    parser.add_argument("--phase2_freeze_layers", type=int,   default=2,
                        help="Unfreeze only top N transformer layers per backbone (0=all). Default 2 saves VRAM.")
    parser.add_argument("--no_grad_ckpt",         action="store_true",
                        help="Disable gradient checkpointing in Phase 2 (faster but needs more VRAM)")
    parser.add_argument("--pos_weight",     type=float, default=None,
                        help="BCE pos_weight override. Default: auto-computed as n_real_train/n_fake_train.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    _section("DeepSentinel — Full Dataset Training")
    print(f"  Device    : {args.device}")
    print(f"  Batch     : {args.batch_size}")
    print(f"  Epochs    : {args.epochs} (patience={args.patience})")
    print(f"  LR        : {args.lr}")
    print(f"  Track 4   : {'EXCLUDED (--no_track4)' if args.no_track4 else 'included'}")
    print(f"  Sarcasm   : {'EXCLUDED (--no_sarcasm)' if args.no_sarcasm else 'included (MUStARD)'}")
    print(f"  Phase 2   : {'SKIP' if args.no_phase2 else f'ON — LR={args.phase2_lr}, epochs={args.phase2_epochs}, batch={args.phase2_batch}'}")

    train_ds, val_ds, test_ds, stats, Phase2FullDataset, p2_train_recs = build_datasets(
        cfg,
        seed=args.seed,
        include_track4=not args.no_track4,
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
        print("\n  ERROR: No training clips. Run: python scripts/preprocess_all.py --device cuda")
        return

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=args.device.startswith("cuda"))
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    _section("Initializing DeepfakeDetector")
    model = DeepfakeDetector()
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
        lambda_a       = 0.5,
        lambda_b       = 0.5,
        lambda_sarcasm = 0.3,
        pos_weight     = effective_pw,
        device         = args.device,
    )

    # ── Phase 1 ───────────────────────────────────────────────────────────────
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

        p2_ds = Phase2FullDataset(p2_train_recs, PREPROCESSED_DIR, KEYFRAME_CACHE_DIR)
        p2_loader = DataLoader(p2_ds, shuffle=True,
                               batch_size=args.phase2_batch,
                               num_workers=args.workers,
                               pin_memory=args.device.startswith("cuda"))
        trainer.train_loader = p2_loader
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
    print(f"  Checkpoint : {CKPT_DIR}/best_phase{best_phase}.pt")
    print(f"  Logs       : {LOG_DIR}/")
    print(f"  Next step  : python scripts/evaluate_fakeavceleb.py --checkpoint {CKPT_DIR}/best_phase{best_phase}.pt\n")


if __name__ == "__main__":
    main()
