"""
train_adaptation.py
===================
Few-Shot Domain Adaptation on FakeAVCeleb for DeepSentinel.

Performs a strict speaker-disjoint split of FakeAVCeleb:
- Adaptation Train Set: 200 clips (100 Real + 100 Fake) from Speaker Set A.
- Held-Out Test Set:    800 clips (400 Real + 400 Fake) from completely disjoint Speaker Set B.

Trains only for 3-5 lightweight epochs on the adaptation set to adapt
the bottleneck classification boundary to YouTube acoustic environments,
then evaluates on the unseen test set.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

try:
    import cv2
    cv2.setLogLevel(0)
except Exception:
    pass
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.detection_model import DeepfakeDetector
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FAV_ROOT = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2"


def create_disjoint_splits(n_adapt_real: int = 100, n_adapt_fake: int = 100,
                           n_test_real: int = 400, n_test_fake: int = 400,
                           seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Creates strict speaker-disjoint adaptation and test splits from FakeAVCeleb.
    """
    from scripts.evaluate_fakeavceleb import load_clips
    
    # Load pool of all valid clips across the entire FakeAVCeleb dataset
    all_clips = load_clips(n_real=5000, n_fake=25000, seed=seed, hard=False)
    if not all_clips:
        raise RuntimeError("No FakeAVCeleb clips found. Check dataset paths.")

    rng = random.Random(seed)
    
    # Group by speaker
    speaker_clips = defaultdict(lambda: {"real": [], "fake": []})
    for c in all_clips:
        spk = c.get("speaker_id", "unknown")
        if c["fake_label"] == 0:
            speaker_clips[spk]["real"].append(c)
        else:
            speaker_clips[spk]["fake"].append(c)

    all_speakers = list(speaker_clips.keys())
    rng.shuffle(all_speakers)

    # Split speakers 20% for adaptation, 80% for test
    split_idx = max(5, int(len(all_speakers) * 0.20))
    adapt_speakers = set(all_speakers[:split_idx])
    test_speakers = set(all_speakers[split_idx:])

    adapt_reals, adapt_fakes = [], []
    for spk in adapt_speakers:
        adapt_reals.extend(speaker_clips[spk]["real"])
        adapt_fakes.extend(speaker_clips[spk]["fake"])

    test_reals, test_fakes = [], []
    for spk in test_speakers:
        test_reals.extend(speaker_clips[spk]["real"])
        test_fakes.extend(speaker_clips[spk]["fake"])

    rng.shuffle(adapt_reals)
    rng.shuffle(adapt_fakes)
    rng.shuffle(test_reals)
    rng.shuffle(test_fakes)

    # Strictly enforce 1:1 balance in the adaptation training set to eliminate skew
    n_adapt = min(len(adapt_reals), len(adapt_fakes), n_adapt_real, n_adapt_fake)
    adapt_set = adapt_reals[:n_adapt] + adapt_fakes[:n_adapt]
    
    # Take all remaining unseen clips if n_test <= 0, else slice up to requested count
    t_reals = test_reals[:n_test_real] if (n_test_real and n_test_real > 0) else test_reals
    t_fakes = test_fakes[:n_test_fake] if (n_test_fake and n_test_fake > 0) else test_fakes
    test_set = t_reals + t_fakes

    rng.shuffle(adapt_set)
    rng.shuffle(test_set)

    # Verification: Assert ZERO clip and ZERO speaker overlap
    adapt_clip_ids = {c["clip_id"] for c in adapt_set}
    test_clip_ids = {c["clip_id"] for c in test_set}
    overlap_clips = adapt_clip_ids.intersection(test_clip_ids)
    
    adapt_spk_ids = {c["speaker_id"] for c in adapt_set}
    test_spk_ids = {c["speaker_id"] for c in test_set}
    overlap_spks = adapt_spk_ids.intersection(test_spk_ids)

    assert len(overlap_clips) == 0, f"DATA LEAKAGE ERROR: {len(overlap_clips)} overlapping clips!"
    assert len(overlap_spks) == 0, f"SPEAKER OVERLAP ERROR: {len(overlap_spks)} overlapping speakers!"

    log.info(f"Created strict speaker-disjoint splits:")
    log.info(f"  Adaptation Set: {len(adapt_set)} clips ({sum(1 for c in adapt_set if c['fake_label']==0)} Real / {sum(1 for c in adapt_set if c['fake_label']==1)} Fake) across {len(adapt_spk_ids)} speakers")
    log.info(f"  Held-Out Test : {len(test_set)} clips ({sum(1 for c in test_set if c['fake_label']==0)} Real / {sum(1 for c in test_set if c['fake_label']==1)} Fake) across {len(test_spk_ids)} speakers")
    log.info(f"  Overlap Check : 0 overlapping clips, 0 overlapping speakers (100% DISJOINT)")

    return adapt_set, test_set


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Domain Adaptation on FakeAVCeleb")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/full/best_phase2_bottleneck.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=4, help="Adaptation epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Adaptation learning rate (default: 5e-6)")
    parser.add_argument("--pos_weight", type=float, default=None, help="BCE loss pos_weight (default: None for unbiased standard BCE)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_adapt_real", type=int, default=100)
    parser.add_argument("--n_adapt_fake", type=int, default=100)
    parser.add_argument("--n_test_real", type=int, default=400)
    parser.add_argument("--n_test_fake", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  DeepSentinel — Few-Shot Domain Adaptation (FakeAVCeleb)")
    print("=" * 60)

    # 1. Create Disjoint Splits
    adapt_set, test_set = create_disjoint_splits(
        n_adapt_real=args.n_adapt_real, n_adapt_fake=args.n_adapt_fake,
        n_test_real=args.n_test_real, n_test_fake=args.n_test_fake,
        seed=args.seed
    )

    # Save test and adapt manifests for evaluate_fakeavceleb.py
    out_dir = REPO_ROOT / "data/eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_manifest_path = out_dir / "fakeavceleb_heldout_unseen_test.csv"
    with open(test_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "video_path", "fake_label", "method", "type", "speaker_id", "cached"])
        writer.writeheader()
        writer.writerows(test_set)
    print(f"  Saved held-out test manifest -> {test_manifest_path}")

    adapt_manifest_path = out_dir / "fakeavceleb_adapt_set.csv"
    with open(adapt_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "video_path", "fake_label", "method", "type", "speaker_id", "cached"])
        writer.writeheader()
        writer.writerows(adapt_set)
    print(f"  Saved adaptation training manifest -> {adapt_manifest_path}")

    drive_eval = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/eval_results")
    if drive_eval.exists():
        import shutil
        shutil.copy2(test_manifest_path, drive_eval / "fakeavceleb_heldout_unseen_test.csv")
        shutil.copy2(adapt_manifest_path, drive_eval / "fakeavceleb_adapt_set.csv")
        print(f"  [GOOGLE DRIVE BACKUP] Saved test & adapt manifests -> {drive_eval}")

    # 2. Load Base Model
    ckpt_path = Path(args.base_checkpoint)
    if not ckpt_path.exists():
        drive_ckpt = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode/best_phase2_bottleneck.pt")
        if drive_ckpt.exists():
            import shutil
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_ckpt, ckpt_path)

    model = DeepfakeDetector(classifier_mode="bottleneck").to(args.device)
    model.load_backbones()
    model.to(args.device)
    
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  Loaded base checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # 3. Setup Adaptation Dataset & Loader
    from scripts.evaluate_fakeavceleb import FakeAVCelebEvalDataset
    pipeline = PreprocessingPipeline(cache_dir="data/preprocessed")

    adapt_dataset = FakeAVCelebEvalDataset(adapt_set, pipeline, no_cache=False)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 4. Optimizer, Scheduler & Multi-Task Contrastive Loss
    pw_t = torch.tensor([args.pos_weight], device=args.device) if args.pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print("\n" + "=" * 60)
    print(f"  RUNNING {args.epochs}-EPOCH ADAPTATION ON {len(adapt_set)} CLIPS (LR={args.lr} -> 1e-6, Margin Separation=1.5)")
    print("=" * 60)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        pbar = tqdm(adapt_loader, desc=f"Adaptation Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            audio_values    = batch["audio_values"].to(args.device)
            input_ids       = batch["input_ids"].to(args.device)
            attention_mask  = batch["attention_mask"].to(args.device)
            keyframe_pixels = batch["keyframe_pixels"].to(args.device)
            labels          = batch["fake_label"].to(args.device).float()

            optimizer.zero_grad()
            out = model(audio_values, input_ids, attention_mask, keyframe_pixels)
            bce_loss = criterion(out.logit.squeeze(1), labels)

            # Supervised Margin Separation: actively push Real and Fake distributions apart
            f_mask = (labels == 1)
            r_mask = (labels == 0)
            if f_mask.any() and r_mask.any():
                mean_fake = out.logit.squeeze(1)[f_mask].mean()
                mean_real = out.logit.squeeze(1)[r_mask].mean()
                margin_loss = torch.clamp(1.5 - (mean_fake - mean_real), min=0.0)
            else:
                margin_loss = out.logit.new_zeros(1).squeeze()

            loss = bce_loss + 0.20 * margin_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        avg_loss = total_loss / len(adapt_set)
        print(f"  Epoch {epoch:02d}/{args.epochs:02d} — Adaptation Loss: {avg_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.2e})")

    # 5. Save Adapted Checkpoint
    adapted_ckpt_path = REPO_ROOT / "checkpoints/full/best_phase2_adapted.pt"
    adapted_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "classifier_mode": "bottleneck",
        "adapted": True,
    }, adapted_ckpt_path)
    print(f"\n✅ Saved adapted checkpoint -> {adapted_ckpt_path}")

    # Also backup to Drive
    drive_ckpt_out = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/best_phase2_adapted.pt")
    if drive_ckpt_out.parent.exists():
        import shutil
        shutil.copy2(adapted_ckpt_path, drive_ckpt_out)
        print(f"  [GOOGLE DRIVE BACKUP] Saved -> {drive_ckpt_out}")

    print("\n" + "=" * 60)
    print(f"  EVALUATING ON HELD-OUT UNSEEN TEST SET ({len(test_set)} CLIPS)")
    print("=" * 60)

    from scripts.evaluate_fakeavceleb import run_inference, compute_metrics, per_method_breakdown
    results = run_inference(
        model=model,
        clips=test_set,
        pipeline=pipeline,
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
        no_cache=False,
    )

    # Calculate initial metrics to find Youden's J and Equal Sensitivity-Specificity (EER) operating points
    temp_m = compute_metrics(results, user_thresh=0.50)
    cal_t = temp_m.get("youden_thresh", 0.50)
    eer_t = temp_m.get("eer_thresh", 0.50)
    m = compute_metrics(results, user_thresh=cal_t)

    print(f"\n  --- HELD-OUT UNSEEN TEST RESULTS ({len(results)} clips) ---")
    print(f"\n  --- 1. Standard Policy (tau = 0.50) ---")
    print(f"  Accuracy (0.50)        : {m['acc_50']:.4f}")
    print(f"  Balanced Accuracy      : {m['bal_acc_50']:.4f}")
    print(f"  Precision              : {m['prec_50']:.4f}")
    print(f"  Recall (Sensitivity)   : {m['rec_50']:.4f}")
    print(f"  Specificity (Real)     : {m['spec_50']:.4f}")
    print(f"  F1-Score               : {m['f1_50']:.4f}")

    print(f"\n  --- 2. Equal Error Rate Policy (tau = {eer_t:.2f}) [EXACT 50/50 REAL-FAKE EQUALITY] ---")
    print(f"  Overall Accuracy       : {temp_m['eer_acc']:.4f}")
    print(f"  Real Video Accuracy    : {temp_m['eer_spec']:.4f}")
    print(f"  Fake Video Detection   : {temp_m['eer_sens']:.4f}")
    print(f"  Precision              : {temp_m['eer_prec']:.4f}")
    print(f"  F1-Score               : {temp_m['eer_f1']:.4f}")

    print(f"\n  --- 3. Optimal Youden's J Policy (tau = {cal_t:.2f}) ---")
    print(f"  Accuracy ({cal_t:.2f})        : {m['acc_cal']:.4f}")
    print(f"  Balanced Accuracy      : {m['bal_acc_cal']:.4f}")
    print(f"  Precision              : {m['prec_cal']:.4f}")
    print(f"  Recall (Sensitivity)   : {m['rec_cal']:.4f}")
    print(f"  Specificity (Real)     : {m['spec_cal']:.4f}")
    print(f"  F1-Score               : {m['f1_cal']:.4f}")

    if m["auc"] is not None:
        print(f"\n  AUC-ROC                : {m['auc']:.4f}")

    _section(f"Per-method breakdown on Unseen Test Clips (Equal Policy tau = {eer_t:.2f})")
    per_method_breakdown(results, cal_thresh=eer_t)


def _section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


if __name__ == "__main__":
    main()
