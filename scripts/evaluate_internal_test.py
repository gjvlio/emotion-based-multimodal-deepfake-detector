"""
Evaluate DeepSentinel Phase 1 / Phase 2 models on the 10% Internal Test Split (1,469 clips).
Includes:
- Deepfake Detection Binary Metrics (Accuracy, Precision, Recall, Specificity, F1, AUC-ROC)
- Multi-Task Auxiliary Metrics:
    * Emotion Head A (Audio-Text Emotion) 6-Class Accuracy & F1
    * Emotion Head B (Visual Emotion) 6-Class Accuracy & F1
    * Sarcasm Head (Binary Sarcasm Detection) Accuracy on MUStARD
    * Emotion Delta (Δ Disparity) Mean Analysis: Real vs. Fake

Usage:
    python scripts/evaluate_internal_test.py --checkpoint checkpoints/full/best_phase1_bottleneck.pt
    python scripts/evaluate_internal_test.py --checkpoint checkpoints/full/best_phase2_bottleneck.pt
"""

import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.detection_model import DeepfakeDetector

EMOTION_MAP = {
    0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Fear", 5: "Disgust"
}

EMOTION_TO_IDX = {
    "NEU": 0, "neutral": 0,
    "HAP": 1, "happy": 1, "joy": 1,
    "SAD": 2, "sad": 2, "sadness": 2,
    "ANG": 3, "angry": 3, "anger": 3,
    "FEA": 4, "fear": 4, "fearful": 4,
    "DIS": 5, "disgust": 5, "disgusted": 5,
    "surprise": 0, "frustrated": 3, "excited": 1,
}

def _parse_emo(val) -> int:
    if val is None or str(val).strip() == "" or str(val).strip() == "-1":
        return -1
    return EMOTION_TO_IDX.get(str(val).strip(), -1)


class InternalTestDataset(Dataset):
    """Loads precomputed (z_at, z_v) feature tensors with emotion & sarcasm annotations."""
    def __init__(self, manifest_csv: Path, prep_dir: Path):
        self.records = []
        if not manifest_csv.exists():
            print(f"[WARNING] Manifest missing: {manifest_csv}")
            return
        
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = str(row.get("output_stem") or row.get("clip_id") or "")
                if not cid:
                    continue
                z_at_p = prep_dir / "features" / "z_at" / f"{cid}.pt"
                z_v_p  = prep_dir / "features" / "z_v"  / f"{cid}.pt"
                
                if z_at_p.exists() and z_v_p.exists():
                    fake_lab = int(row.get("fake_label", row.get("label", -1)))
                    aud_emo = _parse_emo(row.get("audio_emotion"))
                    vis_emo = _parse_emo(row.get("visual_emotion"))
                    sarc = int(row.get("sarcasm_label", -1))

                    self.records.append({
                        "clip_id": cid,
                        "z_at_path": z_at_p,
                        "z_v_path": z_v_p,
                        "fake_label": fake_lab,
                        "audio_emotion": aud_emo,
                        "visual_emotion": vis_emo,
                        "sarcasm_label": sarc,
                        "source_pipeline": str(row.get("source_pipeline", "unknown")),
                    })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        z_at = torch.load(r["z_at_path"], weights_only=True)
        z_v  = torch.load(r["z_v_path"],  weights_only=True)
        return {
            "clip_id": r["clip_id"],
            "z_at": z_at,
            "z_v": z_v,
            "fake_label": r["fake_label"],
            "audio_emotion": r["audio_emotion"],
            "visual_emotion": r["visual_emotion"],
            "sarcasm_label": r["sarcasm_label"],
            "source_pipeline": r["source_pipeline"],
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on 10% In-Domain Internal Test Split (Detection & Emotion)")
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

    print("=" * 65)
    print("  IN-DOMAIN MULTI-TASK EVALUATION (Internal Test Split - 10%)")
    print("=" * 65)
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

    ds = InternalTestDataset(manifest_path, prep_path)
    print(f"  Loaded Valid Test Tensors: {len(ds)} clips")

    if len(ds) == 0:
        print("ERROR: No valid preprocessed feature tensors found for internal test clips. Check data/preprocessed/features/.")
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Accumulators for Detection
    det_scores, det_labels = [], []
    
    # Accumulators for Emotion Heads
    emo_a_preds, emo_a_labels = [], []
    emo_b_preds, emo_b_labels = [], []
    
    # Accumulators for Delta disparity (Real vs. Fake)
    deltas_real, deltas_fake = [], []

    # Accumulators for Sarcasm
    sarc_scores, sarc_labels = [], []

    with torch.no_grad():
        for batch in loader:
            z_at = batch["z_at"].to(device)
            z_v  = batch["z_v"].to(device)
            
            fake_labs = batch["fake_label"].tolist()
            aud_emos  = batch["audio_emotion"].tolist()
            vis_emos  = batch["visual_emotion"].tolist()
            sarcs     = batch["sarcasm_label"].tolist()

            out = model.forward_from_features(z_at, z_v)
            
            # 1. Detection Scores
            scores = torch.sigmoid(out.logit).squeeze(1).cpu().tolist()
            for s, l in zip(scores, fake_labs):
                if l in (0, 1):
                    det_scores.append(s)
                    det_labels.append(l)

            # 2. Emotion Head A (Audio-Text)
            a_preds = out.emo_a.argmax(dim=-1).cpu().tolist()
            for p, l in zip(a_preds, aud_emos):
                if l != -1 and 0 <= l <= 5:
                    emo_a_preds.append(p)
                    emo_a_labels.append(l)

            # 3. Emotion Head B (Visual)
            b_preds = out.emo_b.argmax(dim=-1).cpu().tolist()
            for p, l in zip(b_preds, vis_emos):
                if l != -1 and 0 <= l <= 5:
                    emo_b_preds.append(p)
                    emo_b_labels.append(l)

            # 4. Emotion Disparity Delta norm
            delta_norms = out.delta.norm(dim=-1).cpu().tolist()
            for d_val, l in zip(delta_norms, fake_labs):
                if l == 0:
                    deltas_real.append(d_val)
                elif l == 1:
                    deltas_fake.append(d_val)

            # 5. Sarcasm Head
            s_probs = torch.sigmoid(out.sarc).squeeze(1).cpu().tolist()
            for sp, sl in zip(s_probs, sarcs):
                if sl in (0, 1):
                    sarc_scores.append(sp)
                    sarc_labels.append(sl)

    # ── 1. Deepfake Detection Results ──────────────────────────────────────────
    real_cnt = sum(1 for l in det_labels if l == 0)
    fake_cnt = sum(1 for l in det_labels if l == 1)
    preds_50 = [1 if s >= 0.5 else 0 for s in det_scores]

    acc_50 = accuracy_score(det_labels, preds_50)
    prec, rec, f1, _ = precision_recall_fscore_support(det_labels, preds_50, average="binary", zero_division=0)
    auc = roc_auc_score(det_labels, det_scores) if len(set(det_labels)) == 2 else None

    tp = sum(1 for p, l in zip(preds_50, det_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds_50, det_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds_50, det_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds_50, det_labels) if p == 0 and l == 0)

    print("\n" + "=" * 65)
    print("  1. DEEPFAKE DETECTION BENCHMARK (Binary Incongruency)")
    print("=" * 65)
    print(f"  Clips Evaluated : {len(det_labels)} (Real={real_cnt}, Fake={fake_cnt})")
    print(f"  Accuracy (0.50) : {acc_50 * 100:.2f}%")
    print(f"  Precision (0.50): {prec * 100:.2f}%")
    print(f"  Recall (0.50)   : {rec * 100:.2f}%")
    print(f"  Specificity(0.5): {(tn / max(real_cnt, 1)) * 100:.2f}%")
    print(f"  F1-Score (0.50) : {f1:.4f}")
    print(f"  TP/FP/FN/TN     : {tp}/{fp}/{fn}/{tn}")
    if auc is not None:
        print(f"  AUC-ROC         : {auc:.4f}")

    # ── 2. Emotion Heads Results ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  2. MULTI-TASK EMOTION HEAD ACCURACY (6 Emotion Classes)")
    print("=" * 65)
    
    if emo_a_labels:
        acc_a = accuracy_score(emo_a_labels, emo_a_preds)
        _, _, f1_a, _ = precision_recall_fscore_support(emo_a_labels, emo_a_preds, average="weighted", zero_division=0)
        print(f"  Emotion Head A (Audio-Text) : Acc = {acc_a * 100:.2f}% | F1 = {f1_a:.4f} (N = {len(emo_a_labels)} clips)")
    else:
        print("  Emotion Head A (Audio-Text) : N/A (no annotated test clips)")

    if emo_b_labels:
        acc_b = accuracy_score(emo_b_labels, emo_b_preds)
        _, _, f1_b, _ = precision_recall_fscore_support(emo_b_labels, emo_b_preds, average="weighted", zero_division=0)
        print(f"  Emotion Head B (Visual Face): Acc = {acc_b * 100:.2f}% | F1 = {f1_b:.4f} (N = {len(emo_b_labels)} clips)")
    else:
        print("  Emotion Head B (Visual Face): N/A (no annotated test clips)")

    # ── 3. Emotion Disparity Delta (Δ) Breakdown ──────────────────────────────
    print("\n" + "=" * 65)
    print("  3. AFFECTIVE DISPARITY DELTA (||Δ|| Discrepancy Signal)")
    print("=" * 65)
    mean_d_real = float(np.mean(deltas_real)) if deltas_real else 0.0
    mean_d_fake = float(np.mean(deltas_fake)) if deltas_fake else 0.0
    print(f"  Mean ||Δ|| on Real Videos (Emotion-Congruent): {mean_d_real:.4f}")
    print(f"  Mean ||Δ|| on Fake Videos (Emotion-Incongruent): {mean_d_fake:.4f}")
    if mean_d_real > 0:
        print(f"  Disparity Contrast Ratio (Fake / Real)       : {mean_d_fake / mean_d_real:.2f}x higher on fakes")

    # ── 4. Sarcasm Detection Results (MUStARD) ────────────────────────────────
    print("\n" + "=" * 65)
    print("  4. SARCASM HEAD ACCURACY (MUStARD Benchmark)")
    print("=" * 65)
    if sarc_labels:
        s_preds = [1 if p >= 0.5 else 0 for p in sarc_scores]
        acc_sarc = accuracy_score(sarc_labels, s_preds)
        auc_sarc = roc_auc_score(sarc_labels, sarc_scores) if len(set(sarc_labels)) == 2 else None
        auc_str = f"| AUC = {auc_sarc:.4f}" if auc_sarc is not None else ""
        print(f"  Sarcasm Detection Accuracy: {acc_sarc * 100:.2f}% {auc_str} (N = {len(sarc_labels)} clips)")
    else:
        print("  Sarcasm Detection : N/A (no MUStARD clips in test set)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
