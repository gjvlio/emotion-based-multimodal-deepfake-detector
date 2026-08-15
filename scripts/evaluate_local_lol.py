"""
evaluate_local_lol.py
======================
Local evaluation script to evaluate checkpoints/full/best_phase2_bottleneck_lol.pt 
directly from cached preprocessed features on disk.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def main():
    print("=" * 60)
    print("  LOCAL EVALUATION: Phase 4 Model (best_phase2_bottleneck_lol.pt)")
    print("=" * 60)

    ckpt_path = REPO_ROOT / "checkpoints/full/best_phase2_bottleneck_lol.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at: {ckpt_path}")
        return

    import subprocess
    cmd_args = [
        sys.executable,
        "scripts/evaluate_fakeavceleb.py",
        "--checkpoint", str(ckpt_path),
        "--classifier_mode", "bottleneck",
        "--cached_only",
        "--force_features",
        "--n_real", "500",
        "--n_fake", "9500",
        "--save_csv", "benchmark_local_lol.csv",
    ]
    print(f"\nRunning command:\n  {' '.join(cmd_args)}\n")
    subprocess.run(cmd_args, check=True)

if __name__ == "__main__":
    main()
