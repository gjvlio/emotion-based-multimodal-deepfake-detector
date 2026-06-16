"""
baseline_logreg.py — Logistic Regression baseline for deepfake detection.

Takes the same cached (Z_at, Z_v) features used by the deep model.
Concatenates them → (2304,) vector per clip.
Trains sklearn LogisticRegression on train split, evaluates on test split.
Reports: AUC, F1, Accuracy, Precision, Recall, Confusion Matrix.

Usage:
    python scripts/baseline_logreg.py
    python scripts/baseline_logreg.py --C 0.1 --out results/baseline.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_split_features(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Load all (Z_at, Z_v, label) from a DeepfakeDataset into numpy arrays."""
    X, y = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    for batch in loader:
        z_at = batch["z_at"].numpy()          # (B, 1536)
        z_v  = batch["z_v"].numpy()           # (B, 768)
        feat = np.concatenate([z_at, z_v], axis=1)  # (B, 2304)
        label = batch["fake_label"].numpy()   # (B,)
        X.append(feat)
        y.append(label)
    return np.concatenate(X), np.concatenate(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", default="data/preprocessed")
    parser.add_argument("--C",    type=float, default=1.0,
                        help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument("--out",  default="results/baseline_logreg.json")
    args = parser.parse_args()

    import yaml
    from src.training.dataset import DeepfakeDataset

    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    preprocessed_dir = PROJECT_ROOT / args.preprocessed_dir

    print("Building dataset splits...")
    train_ds, val_ds, test_ds = DeepfakeDataset.stratified_split(
        preprocessed_dir=preprocessed_dir,
        train_ratio=cfg["training"]["train_ratio"],
        val_ratio=cfg["training"]["val_ratio"],
        seed=args.seed,
        track1_meta  = PROJECT_ROOT / paths["track1_meta"],
        track2_meta  = PROJECT_ROOT / paths["track2_meta"],
        track3_meta  = PROJECT_ROOT / paths["track3_meta"],
        track4_meta  = PROJECT_ROOT / paths["track4_meta"],
        meld_real_csv  = PROJECT_ROOT / paths["meld_real_csv"],
        mosei_real_csv = PROJECT_ROOT / paths["mosei_real_csv"],
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    print("Loading train features...")
    X_train, y_train = load_split_features(train_ds)
    print("Loading test features...")
    X_test,  y_test  = load_split_features(test_ds)

    print(f"Feature dim: {X_train.shape[1]} | Train samples: {len(y_train)} | Test samples: {len(y_test)}")

    print(f"Training LogisticRegression (C={args.C})...")
    clf = LogisticRegression(
        C=args.C,
        max_iter=1000,
        random_state=args.seed,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_prob)
    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    results = {
        "model":     "LogisticRegression",
        "C":         args.C,
        "auc":       round(auc, 4),
        "accuracy":  round(acc, 4),
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "confusion_matrix": cm,
        "n_train":   int(len(y_train)),
        "n_test":    int(len(y_test)),
    }

    print("\n=== Logistic Regression Baseline ===")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
