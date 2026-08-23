"""
Evaluate DeepSentinel Phase 1 / Phase 2 models on the 10% Internal Test Split (1,469 clips).
Usage:
    python scripts/evaluate_internal_test.py --checkpoint checkpoints/full/best_phase1_bottleneck.pt
    python scripts/evaluate_internal_test.py --checkpoint checkpoints/full/best_phase2_bottleneck.pt
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.detection_model import DeepfakeDetector
from src.training.dataset import Phase1Dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate on 10% In-Domain Internal Test Split")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/full/best_phase1_bottleneck.pt")
    parser.add_argument("--manifest", type=str, default="data/processed/internal_test_manifest.csv")
    parser.add_argument("--prep_dir", type=str, default="data/preprocessed")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint)

    # Drive fallback
    if not ckpt_path.exists():
        drive_path = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode") / ckpt_path.name
        if drive_path.exists():
            import shutil
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_path, ckpt_path)

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    print("=" * 60)
    print("  IN-DOMAIN INTERNAL TEST EVALUATION (10% Split, Unseen Actors)")
    print("=" * 60)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Manifest   : {args.manifest}")
    print(f"  Device     : {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_keys = ckpt["model_state"].keys()

    mode = "bottleneck" if ("bilinear_proj.weight" in state_keys or "proj_ln.weight" in state_keys) else "baseline"
    model = DeepfakeDetector(classifier_mode=mode).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    prep_path = Path(args.prep_dir)
    manifest_path = REPO_ROOT / args.manifest if not Path(args.manifest).is_absolute() else Path(args.manifest)

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return

    ds = Phase1Dataset(manifest_path, prep_path)
    print(f"  Total Test Samples: {len(ds)} clips")

    if len(ds) == 0:
        print("ERROR: No valid preprocessed feature tensors found for internal test clips.")
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            z_at = batch["z_at"].to(device)
            z_v  = batch["z_v"].to(device)
            labs = batch["fake_label"].cpu().tolist()

            out = model.forward_from_features(z_at, z_v)
            scores = torch.sigmoid(out.logit).squeeze(1).cpu().tolist()

            all_scores.extend(scores)
            all_labels.extend(labs)

    preds_50 = [1 if s >= 0.5 else 0 for s in all_scores]

    acc_50 = accuracy_score(all_labels, preds_50)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, preds_50, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) == 2 else None

    print("\n" + "=" * 60)
    print("  RESULTS ON IN-DOMAIN INTERNAL TEST SPLIT")
    print("=" * 60)
    print(f"  Clips Evaluated : {len(all_labels)}")
    print(f"  Accuracy (0.50) : {acc_50 * 100:.2f}%")
    print(f"  Precision (0.50): {prec * 100:.2f}%")
    print(f"  Recall (0.50)   : {rec * 100:.2f}%")
    print(f"  F1-Score (0.50) : {f1:.4f}")
    if auc is not None:
        print(f"  AUC-ROC         : {auc:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
