"""
evaluate_local_latest.py
=========================
Evaluates checkpoints/full/best_phase2_bottleneck_latest.pt locally across cached feature tensors.
"""
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def main():
    print("=" * 60)
    print("  LOCAL EVALUATION: Latest Phase 2 Model (best_phase2_bottleneck_latest.pt)")
    print("=" * 60)

    ckpt_path = REPO_ROOT / "checkpoints/full/best_phase2_bottleneck_latest.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at: {ckpt_path}")
        return

    cmd_args = [
        sys.executable,
        "scripts/evaluate_fakeavceleb.py",
        "--checkpoint", str(ckpt_path),
        "--classifier_mode", "bottleneck",
        "--cached_only",
        "--force_features",
        "--n_real", "500",
        "--n_fake", "9500",
        "--save_csv", "benchmark_local_latest.csv",
    ]
    print(f"\nRunning command:\n  {' '.join(cmd_args)}\n")
    subprocess.run(cmd_args, check=True)

if __name__ == "__main__":
    main()
