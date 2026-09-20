"""
evaluate_sarcasm.py — Held-out MUStARD sarcasm evaluation.

Reuses the EXACT speaker-stratified train/val/test split from train_full.py
(same seed) so the reported numbers come from MUStARD clips the model never
trained or validated on. Reports Accuracy/Precision/Recall/F1/AUC at the
sarcasm decision threshold (0.5, see architecture docs), plus a bootstrap
95% CI on AUC — same convention as evaluate_fakeavceleb.py.

Usage:
    python scripts/evaluate_sarcasm.py --checkpoint checkpoints/full/best_phase1.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.detection_model import DeepfakeDetector
from src.utils.config import Config
from src.evaluation.significance import binary_metrics

SECTION = "=" * 60
SARCASM_THRESHOLD = 0.5  # Bayes-optimal for a sigmoid head trained on class-balanced MUStARD (Castro et al., 2019)


def _section(title: str) -> None:
    print(f"\n{SECTION}\n  {title}\n{SECTION}")


def _load_train_full():
    path = Path(__file__).resolve().parent / "train_full.py"
    spec = importlib.util.spec_from_file_location("train_full", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def collect_sarcasm_scores(model, loader, device):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        z_at = batch["z_at"].to(device)
        z_v  = batch["z_v"].to(device)
        sl   = batch["sarcasm_label"]
        out  = model.forward_from_features(z_at, z_v)
        p_sarc = torch.sigmoid(out.sarcasm.squeeze(1)).cpu()
        mask = sl != -1
        if mask.any():
            labels.extend(sl[mask].tolist())
            scores.extend(p_sarc[mask].tolist())
    return np.array(labels), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SarcasmHead on the held-out MUStARD test split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default=None)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Must match the seed used for the training run being evaluated")
    parser.add_argument("--no_track4",  action="store_true",
                        help="Must match whatever the checkpoint was actually trained with")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return

    _section("Sarcasm evaluation - held-out MUStARD test split")
    cfg = Config.from_yaml(args.config)

    train_full = _load_train_full()
    _, _, test_ds, stats, _, _ = train_full.build_datasets(
        cfg, seed=args.seed, include_track4=not args.no_track4, no_sarcasm=False,
    )
    print(f"  Test split    : {stats['test']} clips  ({stats['n_speakers']} speakers total)")

    model = DeepfakeDetector(
        wav2vec_model = cfg.model.wav2vec_model,
        bert_model    = cfg.model.bert_model,
        vit_model     = cfg.model.vit_model,
        n_emotions    = cfg.model.n_emotions,
        proj_dim      = cfg.model.proj_dim,
        dropout_heads = cfg.model.dropout_heads,
        dropout_cls   = cfg.model.dropout_classifier,
    ).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Checkpoint    : {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    labels, scores = collect_sarcasm_scores(model, loader, args.device)

    if len(labels) == 0:
        print("\n  No MUStARD-labeled clips in the test split - nothing to evaluate.")
        print("  Check data/raw/MUStARD/repo/data/sarcasm_data.json exists and --seed/--no_track4 match training.")
        return

    m = binary_metrics(labels, scores, SARCASM_THRESHOLD)
    _section("Results")
    print(f"  MUStARD test clips : {len(labels)}  (sarcastic={int(labels.sum())}, not={int((labels == 0).sum())})")
    print(f"  Threshold          : {SARCASM_THRESHOLD}")
    print(f"  Accuracy           : {m['accuracy']:.4f}")
    print(f"  Precision          : {m['precision']:.4f}")
    print(f"  Recall             : {m['recall']:.4f}")
    print(f"  F1                 : {m['f1']:.4f}")

    if len(np.unique(labels)) == 2:
        rng = np.random.default_rng(42)
        n = len(labels)
        boot_aucs = []
        for _ in range(10_000):
            idx = rng.integers(0, n, size=n)
            b_lab = labels[idx]
            if len(np.unique(b_lab)) == 2:
                boot_aucs.append(roc_auc_score(b_lab, scores[idx]))
        lo, hi = (np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (float("nan"), float("nan")))
        print(f"  AUC-ROC            : {m['auc']:.4f}  [95% CI: {lo:.4f}-{hi:.4f}]  (10,000 resamples, seed=42)")
    else:
        print("  AUC-ROC            : N/A (only one class present in test split)")


if __name__ == "__main__":
    main()
