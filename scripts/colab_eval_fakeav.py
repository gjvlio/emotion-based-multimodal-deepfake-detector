"""
colab_eval_fakeav.py
====================
Standalone FakeAVCeleb Evaluation Script for Google Colab.
Loads fine-tuned Phase 2 or Phase 1 checkpoints and runs end-to-end evaluation with:
- Temperature Scaling (T=0.5)
- Youden's J Threshold Calibration
- Balanced 500 Real / 500 Fake sampling
"""
import sys
import os
import shutil
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Standalone FakeAVCeleb Colab Evaluator")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Defaults to latest Phase 2 model on Drive or local disk.")
    parser.add_argument("--n_real", type=int, default=500, help="Number of real clips to sample")
    parser.add_argument("--n_fake", type=int, default=500, help="Number of fake clips to sample")
    parser.add_argument("--mode", type=str, default="bottleneck", choices=["bottleneck", "baseline"])
    parser.add_argument("--save_csv", type=str, default="benchmark_results_calibrated.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("  STANDALONE FAKEAVCELEB BENCHMARK EVALUATION (COLAB CELL 4)")
    print("=" * 60)

    drive_p2 = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode/best_phase2_bottleneck.pt")
    local_p2 = REPO_ROOT / "checkpoints/full/bottleneck_mode/best_phase2_bottleneck.pt"
    local_p2_alt = REPO_ROOT / "checkpoints/full/best_phase2_bottleneck.pt"

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif drive_p2.exists():
        ckpt_path = drive_p2
    elif local_p2.exists():
        ckpt_path = local_p2
    elif local_p2_alt.exists():
        ckpt_path = local_p2_alt
    else:
        print("ERROR: No checkpoint found! Train Stage 1 or Stage 2 first.")
        sys.exit(1)

    print(f"  Selected Checkpoint : {ckpt_path}")
    print(f"  Sample Ratio        : {args.n_real} Real / {args.n_fake} Fake (Balanced)")

    cmd = (f"{sys.executable} scripts/evaluate_fakeavceleb.py "
           f"--checkpoint {ckpt_path} "
           f"--classifier_mode {args.mode} "
           f"--n_real {args.n_real} "
           f"--n_fake {args.n_fake} "
           f"--save_csv {args.save_csv}")

    print(f"\nRunning command:\n  {cmd}\n")
    ret = os.system(cmd)

    if ret == 0 and Path(args.save_csv).exists():
        drive_out_dir = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode")
        if drive_out_dir.exists():
            shutil.copy2(args.save_csv, drive_out_dir / args.save_csv)
            print(f"\n[BACKUP] Saved benchmark CSV results to Google Drive: {drive_out_dir / args.save_csv}")

if __name__ == "__main__":
    main()
