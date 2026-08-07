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
    auto_pw = (real_train / max(fake_train, 1))

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

        def __len__(self):
            return len(self._recs)

        def __getitem__(self, i):
            r = self._recs[i]
            cid = r["clip_id"]
            vp  = r["video_path"]
            return {
                "audio_values":   self._audio_values(cid),
                "input_ids":      self._bert_inputs(cid)[0],
                "attention_mask": self._bert_inputs(cid)[1],
                "pixel_values":   self._keyframe_pixels(cid, vp),
                "fake_label":     torch.tensor(r["fake_label"],    dtype=torch.long),
                "audio_emotion":  torch.tensor(r["audio_emotion"], dtype=torch.long),
                "visual_emotion": torch.tensor(r["visual_emotion"],dtype=torch.long),
                "sarcasm_label":  torch.tensor(r["sarcasm_label"], dtype=torch.long),
                "source_pipeline":r["source_pipeline"],
                "clip_id":        cid,
                "speaker_id":     r["speaker_id"],
            }

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
            from src.preprocessing.visual import extract_frames
            from src.preprocessing.filters import frames_to_pil
            frames = extract_frames(video_path, target_fps=25.0)
            if not frames:
                px = torch.zeros(self.N_KEYFRAMES, 3, self.FRAME_SIZE, self.FRAME_SIZE)
                torch.save(px, kf_path)
                return px
            pils = frames_to_pil(frames[:self.N_KEYFRAMES])
            enc  = self._vit_proc(pils, return_tensors="pt")
            px   = enc.pixel_values
            if px.shape[0] < self.N_KEYFRAMES:
                pad = px[-1:].repeat(self.N_KEYFRAMES - px.shape[0], 1, 1, 1)
                px  = torch.cat([px, pad], dim=0)
            torch.save(px, kf_path)
            return px

    train_ds = FullDataset(train_recs)
    val_ds   = FullDataset(val_recs)
    test_ds  = FullDataset(test_recs)

    return train_ds, val_ds, test_ds, stats, Phase2FullDataset, train_recs


def main():
    parser = argparse.ArgumentParser(description="Full-dataset training for DeepSentinel.")
    parser.add_argument("--epochs",               type=int,   default=30)
    parser.add_argument("--batch_size",           type=int,   default=32)
    parser.add_argument("--lr",                   type=float, default=1e-3)
    parser.add_argument("--patience",             type=int,   default=7)
    parser.add_argument("--pos_weight",           type=float, default=None)
    parser.add_argument("--no_sarcasm",           action="store_true")
    parser.add_argument("--no_phase2",            action="store_true")
    parser.add_argument("--phase2_epochs",        type=int,   default=5)
    parser.add_argument("--phase2_batch",         type=int,   default=4)
    parser.add_argument("--phase2_lr",            type=float, default=1e-5)
    parser.add_argument("--phase2_freeze_layers", type=int,   default=10)
    parser.add_argument("--no_grad_ckpt",         action="store_true")
    parser.add_argument("--workers",              type=int,   default=0)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--device",               type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classifier_mode",      type=str,   default="baseline", choices=["baseline", "mismatch_only", "emotion_bilinear", "bottleneck", "high_dropout"])
    args = parser.parse_args()

    _section("DeepSentinel Full Dataset Training")
    print(f"  Device    : {args.device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")

    cfg = Config()

    train_ds, val_ds, test_ds, stats, Phase2FullDataset, p2_train_recs = build_datasets(
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

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
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
        lambda_a       = 0.5,
        lambda_b       = 0.5,
        lambda_sarcasm = 0.3,
        pos_weight     = effective_pw,
        device         = args.device,
        ckpt_suffix    = args.classifier_mode,
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
    ckpt_name = f"best_phase1_{args.classifier_mode}.pt" if args.classifier_mode else "best_phase1.pt"
    print(f"  Checkpoint : {CKPT_DIR}/{ckpt_name}")
    print(f"  Logs       : {LOG_DIR}/")
    print(f"  Next step  : python scripts/evaluate_fakeavceleb.py --checkpoint {CKPT_DIR}/{ckpt_name}\n")


if __name__ == "__main__":
    main()
