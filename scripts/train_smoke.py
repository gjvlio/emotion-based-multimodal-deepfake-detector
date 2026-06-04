"""
train_smoke.py — Smoke-test training run on small manifests.

Uses smoke_detector.csv + smoke_sarcasm.csv (~1040 clips total).
Purpose: verify full training pipeline works before committing to full dataset.
NOT for reporting results — dataset too small.

Usage:
    python scripts/train_smoke.py
    python scripts/train_smoke.py --epochs 15 --batch_size 16
    python scripts/train_smoke.py --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detection_model import DeepfakeDetector
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

REPO_ROOT          = Path(__file__).resolve().parents[1]
SMOKE_DIR          = REPO_ROOT / "data/processed/smoke_manifests"
PREPROCESSED_DIR   = REPO_ROOT / "data/preprocessed"
KEYFRAME_CACHE_DIR = REPO_ROOT / "data/preprocessed/keyframes"
CKPT_DIR           = REPO_ROOT / "checkpoints/smoke"
LOG_DIR            = REPO_ROOT / "logs/smoke"

SECTION = "=" * 60


def _section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(f"{SECTION}")


def build_datasets(preprocessed_dir: Path, seed: int = 42, no_sarcasm: bool = False):
    """
    Load smoke manifests → speaker-independent 80/10/10 split.
    Returns (train_ds, val_ds, test_ds, stats_dict).
    """
    import csv, json, numpy as np
    from src.training.dataset import UNKNOWN_EMOTION, UNKNOWN_SARCASM

    detector_csv = SMOKE_DIR / "smoke_detector.csv"
    sarcasm_csv  = SMOKE_DIR / "smoke_sarcasm.csv"

    if not detector_csv.exists():
        raise FileNotFoundError(f"Smoke manifest not found: {detector_csv}\nRun: python scripts/build_smoke_manifest.py")

    # ── Load detector clips ────────────────────────────────────────────────────
    def _emo(code):
        from src.training.dataset import EMOTION_TO_IDX
        if code is None or (isinstance(code, float) and code != code):
            return UNKNOWN_EMOTION
        return EMOTION_TO_IDX.get(str(code).strip(), UNKNOWN_EMOTION)

    records = []

    with open(detector_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row.get("path", "")
            if not path:
                continue
            clip_id = Path(path).stem
            z_at = preprocessed_dir / "features/z_at" / f"{clip_id}.pt"
            z_v  = preprocessed_dir / "features/z_v"  / f"{clip_id}.pt"
            if not z_at.exists() or not z_v.exists():
                continue
            label = int(row.get("label", -1))
            source = row.get("source", "unknown")
            spk = row.get("speaker_id") or clip_id.split("_")[0]
            aud_emo = _emo(row.get("audio_emotion_label") or row.get("emotion_label", ""))
            vis_emo = _emo(row.get("visual_emotion_label") or row.get("emotion_label", ""))
            records.append({
                "clip_id":         clip_id,
                "z_at_path":       str(z_at),
                "z_v_path":        str(z_v),
                "video_path":      path,
                "fake_label":      label,
                "audio_emotion":   aud_emo,
                "visual_emotion":  vis_emo,
                "sarcasm_label":   UNKNOWN_SARCASM,
                "source_pipeline": source,
                "speaker_id":      spk,
            })

    # ── Load MUStARD sarcasm clips ─────────────────────────────────────────────
    if not no_sarcasm and sarcasm_csv.exists():
        with open(sarcasm_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = row.get("path", "")
                if not path:
                    continue
                clip_id = Path(path).stem
                z_at = preprocessed_dir / "features/z_at" / f"{clip_id}.pt"
                z_v  = preprocessed_dir / "features/z_v"  / f"{clip_id}.pt"
                if not z_at.exists() or not z_v.exists():
                    continue
                sarc = int(row.get("sarcasm_label", UNKNOWN_SARCASM))
                spk = row.get("speaker_id") or clip_id.split("_")[0]
                records.append({
                    "clip_id":         clip_id,
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

    if not records:
        raise RuntimeError(
            "No preprocessed clips found. Run preprocessing first:\n"
            "  python scripts/preprocess_all.py --smoke"
        )

    # ── Stratified speaker-independent 80/10/10 split ────────────────────────
    # Real and fake clips come from completely different speaker pools (different datasets).
    # Splitting them together allows all fake speakers to end up in train/val.
    # Fix: three pools split independently so test always contains real + fake + sarcasm.
    rng = np.random.default_rng(seed)
    spk_map = defaultdict(list)
    for i, r in enumerate(records):
        spk_map[r["speaker_id"]].append(i)

    def _split_speakers(spk_list, train_frac=0.80, val_frac=0.10):
        """Shuffle and split a speaker list into train/val/test index sets."""
        arr = list(spk_list)
        rng.shuffle(arr)
        n      = len(arr)
        n_tr   = max(1, int(n * train_frac))
        n_va   = max(1, int(n * val_frac))
        return set(arr[:n_tr]), set(arr[n_tr:n_tr + n_va]), set(arr[n_tr + n_va:])

    # Three independent pools:
    # real_det_spk: speakers that have at least one real clip (fake_label==0)
    # fake_det_spk: speakers that have at least one fake clip (fake_label==1)
    # must_spk:     speakers whose ALL clips are MUStARD (fake_label==-1)
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
    # test_spk  = te_r | te_f | te_m  (implicit)

    train_idx, val_idx, test_idx = [], [], []
    for spk, idxs in spk_map.items():
        if spk in train_spk:   train_idx.extend(idxs)
        elif spk in val_spk:   val_idx.extend(idxs)
        else:                  test_idx.extend(idxs)

    n = len(spk_map)

    # ── Inline dataset class ───────────────────────────────────────────────────
    class SmokeDataset(torch.utils.data.Dataset):
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

    # ── Phase 2 dataset (raw inputs for end-to-end backbone training) ──────────
    class Phase2SmokeDataset(torch.utils.data.Dataset):
        """
        Loads raw inputs for Phase 2 (unfrozen backbone) training.
        Audio: WAV from preprocessed cache → Wav2Vec2 feature extractor.
        Text:  transcript from preprocessed cache → BERT tokenizer.
        Video: keyframe pixels extracted from original MP4 (cached to KEYFRAME_CACHE_DIR).
        Val uses SmokeDataset (z_at/z_v) — only train loader needs this.
        """
        WAV2VEC_MODEL = "facebook/wav2vec2-base"
        BERT_MODEL    = "bert-base-uncased"
        VIT_MODEL     = "google/vit-base-patch16-224"
        MAX_AUDIO     = 80000    # 5s × 16kHz — 30s causes OOM (1500-frame attention on 6GB VRAM)
        MAX_SEQ_LEN   = 128
        N_KEYFRAMES   = 8
        FRAME_SIZE    = 224

        def __init__(self, recs, preprocessed_dir: Path, keyframe_cache_dir: Path):
            self._recs  = recs
            self._prep  = preprocessed_dir
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
            return enc.input_values.squeeze(0)  # (MAX_AUDIO,)

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
            face_results = detect_and_align_faces(gated, "retinaface", 0.7)
            if not face_results:
                face_results = detect_and_align_faces(gated, "retinaface", 0.0)
            if not face_results:
                face_results = [(f, 1.0) for f in (gated or frames)]
            crops     = [r[0] for r in face_results]
            scores    = [r[1] for r in face_results]
            keyframes = select_keyframes(crops, scores, k=self.N_KEYFRAMES)
            pil_imgs  = frames_to_pil(keyframes, size=self.FRAME_SIZE)
            pixels    = self._vit_proc(images=pil_imgs, return_tensors="pt").pixel_values
            torch.save(pixels, kf_path)
            return pixels  # (K, 3, 224, 224)

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

    train_ds   = SmokeDataset([records[i] for i in train_idx])
    val_ds     = SmokeDataset([records[i] for i in val_idx])
    test_ds    = SmokeDataset([records[i] for i in test_idx])
    p2_train_recs = [records[i] for i in train_idx]  # same split, raw inputs

    # Source breakdown
    from collections import Counter
    src_counts  = Counter(records[i]["source_pipeline"] for i in train_idx)
    real_train  = sum(1 for i in train_idx if records[i]["fake_label"] == 0)
    fake_train  = sum(1 for i in train_idx if records[i]["fake_label"] == 1)
    sarc_train  = sum(1 for i in train_idx if records[i]["sarcasm_label"] != UNKNOWN_SARCASM)
    real_test   = sum(1 for i in test_idx  if records[i]["fake_label"] == 0)
    fake_test   = sum(1 for i in test_idx  if records[i]["fake_label"] == 1)

    stats = {
        "total":       len(records),
        "train":       len(train_idx),
        "val":         len(val_idx),
        "test":        len(test_idx),
        "n_speakers":  n,
        "real_train":  real_train,
        "fake_train":  fake_train,
        "sarc_train":  sarc_train,
        "real_test":   real_test,
        "fake_test":   fake_test,
        "src_counts":  dict(src_counts),
    }
    return train_ds, val_ds, test_ds, stats, Phase2SmokeDataset, p2_train_recs


def main():
    parser = argparse.ArgumentParser(description="Smoke-test training run")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--workers",    type=int,   default=0,
                        help="DataLoader num_workers (0=main thread, safe on Windows)")
    parser.add_argument("--no_sarcasm", action="store_true",
                        help="Skip MUStARD sarcasm clips (detector-only training)")
    parser.add_argument("--no_phase2",  action="store_true",
                        help="Skip Phase 2 (backbone fine-tuning) — Phase 1 only")
    parser.add_argument("--phase2_epochs",  type=int,   default=10,
                        help="Max epochs for Phase 2 (default 10)")
    parser.add_argument("--phase2_lr",      type=float, default=1e-5,
                        help="Learning rate for Phase 2 (default 1e-5)")
    parser.add_argument("--phase2_batch",   type=int,   default=2,
                        help="Batch size for Phase 2 (lower — 3 backbones in VRAM, default 2)")
    args = parser.parse_args()

    _section("SMOKE TEST — DeepSentinel Training Pipeline")
    print(f"  Purpose  : verify pipeline, NOT for result reporting")
    print(f"  Sarcasm  : {'EXCLUDED (--no_sarcasm)' if args.no_sarcasm else 'included (MUStARD)'}")
    print(f"  Device   : {args.device}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Epochs   : {args.epochs} (early stopping patience={args.patience})")
    print(f"  LR       : {args.lr}")
    print(f"  Seed     : {args.seed}")
    print(f"  Phase 2  : {'SKIP (--no_phase2)' if args.no_phase2 else f'ON — LR={args.phase2_lr}, epochs={args.phase2_epochs}, batch={args.phase2_batch}'}")

    # ── Build datasets ─────────────────────────────────────────────────────────
    _section("Building datasets from smoke manifests")
    train_ds, val_ds, test_ds, stats, Phase2SmokeDataset, p2_train_recs = build_datasets(
        PREPROCESSED_DIR, seed=args.seed, no_sarcasm=args.no_sarcasm
    )

    print(f"  Total preprocessed clips : {stats['total']}")
    print(f"  Speakers                 : {stats['n_speakers']}")
    print(f"  Train / Val / Test       : {stats['train']} / {stats['val']} / {stats['test']}")
    print(f"\n  Train breakdown:")
    print(f"    Real clips   : {stats['real_train']}")
    print(f"    Fake clips   : {stats['fake_train']}")
    print(f"    Sarcasm annot: {stats['sarc_train']}")
    print(f"    By source:")
    for src, n in sorted(stats["src_counts"].items()):
        print(f"      {src:<20} {n}")
    print(f"\n  Test breakdown (AUC requires both classes):")
    print(f"    Real clips   : {stats['real_test']}")
    print(f"    Fake clips   : {stats['fake_test']}")
    both_classes = stats['real_test'] > 0 and stats['fake_test'] > 0
    print(f"    AUC feasible : {'YES' if both_classes else 'NO — test split lacks one class'}")

    if stats["train"] == 0:
        print("\n  ERROR: No training samples. Preprocessing may not have completed.")
        print("  Run: python scripts/preprocess_all.py --smoke --force")
        return

    # ── DataLoaders ────────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size  = args.batch_size,
        num_workers = args.workers,
        pin_memory  = args.device.startswith("cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ──────────────────────────────────────────────────────────────────
    _section("Initializing DeepfakeDetector")
    model = DeepfakeDetector()
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {train_params:,}")
    print(f"  Architecture:")
    print(f"    EmotionHeadA  : Z_at(1536) -> 256 -> 6")
    print(f"    EmotionHeadB  : Z_v(768)   -> 256 -> 6")
    print(f"    SarcasmHead   : Z_at(1536) -> 256 -> 1")
    print(f"    CBP           : (1536, 768) -> 8192")
    print(f"    Delta         : |softmax(A) - softmax(B)| -> 6")
    print(f"    Classifier    : 8199 -> 512 -> 128 -> 1")

    # ── Trainer ────────────────────────────────────────────────────────────────
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
        device         = args.device,
    )

    # ── Phase 1 training ───────────────────────────────────────────────────────
    trainer.train_phase1(
        lr           = args.lr,
        weight_decay = 1e-4,
        max_epochs   = args.epochs,
        patience     = args.patience,
    )

    # ── Phase 2 training ───────────────────────────────────────────────────────
    best_phase = 1
    if not args.no_phase2:
        _section("Phase 2 — Backbone fine-tuning")
        print(f"  Building Phase 2 dataset (raw inputs)...")
        print(f"  NOTE: First epoch extracts keyframe pixels from MP4 — takes ~5-10min.")
        print(f"        Cached to {KEYFRAME_CACHE_DIR} — instant on subsequent runs.")

        p2_train_ds = Phase2SmokeDataset(p2_train_recs, PREPROCESSED_DIR, KEYFRAME_CACHE_DIR)
        p2_loader_kwargs = dict(
            batch_size  = args.phase2_batch,
            num_workers = args.workers,
            pin_memory  = args.device.startswith("cuda"),
        )
        p2_train_loader = DataLoader(p2_train_ds, shuffle=True, **p2_loader_kwargs)

        # Swap train_loader to Phase 2 raw-input loader; val stays cached (z_at/z_v)
        trainer.train_loader = p2_train_loader

        trainer.train_phase2(
            lr           = args.phase2_lr,
            weight_decay = 1e-4,
            max_epochs   = args.phase2_epochs,
            patience     = args.patience,
        )
        best_phase = 2

    # ── Quick test set evaluation ──────────────────────────────────────────────
    _section("Test set evaluation (smoke — indicative only)")
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
        print(f"  Test clips evaluated : {total}")
        print(f"  Accuracy             : {acc:.4f}")
        print(f"  Precision            : {prec:.4f}")
        print(f"  Recall               : {rec:.4f}")
        print(f"  F1                   : {f1:.4f}")
        print(f"  TP/FP/FN/TN          : {tp}/{fp}/{fn}/{tn}")

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_scores)
            print(f"  AUC-ROC              : {auc:.4f}")
        except Exception:
            print(f"  AUC-ROC              : (sklearn not available)")

        print(f"\n  NOTE: Smoke results are NOT reportable — dataset too small ({total} test clips).")
        print(f"        Run full 80-10-10 on complete dataset for real metrics.")
    else:
        print("  No valid test clips (all MUStARD / fake_label=-1).")

    _section("Smoke test complete")
    print(f"  Checkpoint : {CKPT_DIR}/best_phase{best_phase}.pt")
    print(f"  Logs       : {LOG_DIR}/")
    print(f"  Next step  : run full preprocessing + train on complete dataset\n")


if __name__ == "__main__":
    main()
