"""
evaluate_all_models.py
======================
Loads all four trained classifier models (baseline, mismatch_only, bottleneck,
and high_dropout) from `checkpoints/full/` and evaluates them on the internal
test split (`data/processed/internal_test_manifest.csv`).

Outputs a formatted markdown report comparing test metrics (ACC, PREC, REC, F1, AUC, 95% CI).

Usage:
    python scripts/evaluate_all_models.py [--device cuda]
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.detection_model import DeepfakeDetector
from src.training.dataset import UNKNOWN_EMOTION, EMOTION_TO_IDX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "checkpoints/full"
TEST_MANIFEST_CSV = REPO_ROOT / "data/processed/internal_test_manifest.csv"


class SimpleTestDataset(Dataset):
    def __init__(self, recs):
        self._records = recs
    def __len__(self):
        return len(self._records)
    def __getitem__(self, i):
        r = self._records[i]
        return {
            "z_at": torch.load(r["z_at_path"], weights_only=True).float(),
            "z_v": torch.load(r["z_v_path"], weights_only=True).float(),
            "fake_label": torch.tensor(r["fake_label"], dtype=torch.long),
        }


def _emo(code) -> int:
    if code is None or (isinstance(code, float) and code != code):
        return -1
    return EMOTION_TO_IDX.get(str(code).strip(), -1)


def load_test_records() -> list[dict]:
    records = []
    if not TEST_MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Test manifest not found: {TEST_MANIFEST_CSV}")
    with open(TEST_MANIFEST_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row["clip_id"])
            z_at_p = REPO_ROOT / str(row.get("z_at_path") or f"data/preprocessed/features/z_at/{cid}.pt")
            z_v_p  = REPO_ROOT / str(row.get("z_v_path")  or f"data/preprocessed/features/z_v/{cid}.pt")
            if not z_at_p.exists() or not z_v_p.exists():
                continue
            records.append({
                "clip_id": cid,
                "z_at_path": str(z_at_p),
                "z_v_path": str(z_v_p),
                "fake_label": int(row.get("fake_label", -1)),
                "audio_emotion": _emo(row.get("audio_emotion")),
                "visual_emotion": _emo(row.get("visual_emotion")),
                "source_pipeline": str(row.get("source_pipeline", "unknown")),
                "speaker_id": str(row.get("speaker_id", "unknown")),
            })
    return records


def evaluate_model(model_mode: str, device: str, records: list[dict]) -> dict:
    ckpt_name = f"best_phase1_{model_mode}.pt" if model_mode != "baseline" else "best_phase1_baseline.pt"
    ckpt_path = CKPT_DIR / ckpt_name

    if not ckpt_path.exists():
        log.warning(f"Checkpoint for mode '{model_mode}' does not exist at: {ckpt_path}")
        return {}

    # Initialize model in the correct mode
    model = DeepfakeDetector(classifier_mode=model_mode).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    ds = SimpleTestDataset(records)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    correct, total, tp, fp, fn, tn = 0, 0, 0, 0, 0, 0
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            z_at = batch["z_at"].to(device)
            z_v  = batch["z_v"].to(device)
            fl   = batch["fake_label"].to(device)
            valid = fl != -1
            if not valid.any():
                continue
            out = model.forward_from_features(z_at[valid], z_v[valid])
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

    if total == 0:
        return {}

    acc = correct / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    try:
        auc = roc_auc_score(all_labels, all_scores)
        # Bootstrap 95% CI
        rng = np.random.default_rng(42)
        arr_l = np.array(all_labels)
        arr_s = np.array(all_scores)
        boot_aucs = []
        for _ in range(5000):
            idx = rng.integers(0, len(arr_l), len(arr_l))
            if arr_l[idx].sum() == 0 or arr_l[idx].sum() == len(arr_l):
                continue
            boot_aucs.append(roc_auc_score(arr_l[idx], arr_s[idx]))
        lo, hi = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (0.0, 0.0)
    except Exception as e:
        auc, lo, hi = 0.0, 0.0, 0.0

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "ci_lower": lo,
        "ci_upper": hi,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all four trained models on the internal test split.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("EVALUATING CLASSIFIER ARCHITECTURES ON THE INTERNAL TEST SET")
    print("=" * 68)

    try:
        records = load_test_records()
        print(f"Loaded {len(records):,} valid test records from internal test split.")
    except Exception as e:
        print(f"Error loading test manifest: {e}")
        return

    results = {}
    for mode in ["baseline", "mismatch_only", "emotion_bilinear", "bottleneck", "high_dropout"]:
        print(f"Evaluating mode: {mode.upper()}...")
        res = evaluate_model(mode, args.device, records)
        if res:
            results[mode] = res

    if not results:
        print("\nNo trained checkpoints found to evaluate. Run the training scripts first.")
        return

    # Print Results Table
    print("\n" + "=" * 80)
    print(f"{'Classifier Mode':<17} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'AUC-ROC (95% CI)':<22}")
    print("-" * 82)
    for mode, res in results.items():
        auc_str = f"{res['auc']:.4f} [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
        print(f"{mode.upper():<17} | {res['acc']:.4f}   | {res['prec']:.4f}    | {res['rec']:.4f}   | {res['f1']:.4f}   | {auc_str:<22}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
