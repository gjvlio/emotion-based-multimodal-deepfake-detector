"""
colab_eval_fakeav.py
====================
Standalone FakeAVCeleb Evaluation Script for Google Colab.
Loads fine-tuned Phase 2 or Phase 1 checkpoints and runs end-to-end evaluation with:
- Automatic extraction of fakeavceleb.zip from Google Drive to local SSD
- Robust checkpoint validation
- Temperature Scaling (T=0.5)
- Youden's J Threshold Calibration
- Balanced 500 Real / 500 Fake sampling
"""
import sys
import os
import shutil
import zipfile
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DRIVE_BASE = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")

def search_drive_file(name: str) -> Path | None:
    if not DRIVE_BASE.exists():
        return None
    for root, _, files in os.walk(DRIVE_BASE):
        for f in files:
            if f.lower() == name.lower():
                return Path(root) / f
    stem = Path(name).stem.lower()
    for root, _, files in os.walk(DRIVE_BASE):
        for f in files:
            if f.endswith(".zip") and stem in f.lower():
                return Path(root) / f
    return None

def ensure_fakeavceleb_dataset():
    """Ensure FakeAVCeleb dataset videos and metadata are extracted on local SSD."""
    target_meta = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv"
    alt_meta = REPO_ROOT / "data/FakeAVCeleb_v1.2/meta_data.csv"
    
    has_meta = target_meta.exists() or alt_meta.exists()
    has_videos = False
    for p in [REPO_ROOT / "data/raw/FakeAVCeleb_v1.2", REPO_ROOT / "data/FakeAVCeleb_v1.2"]:
        if p.exists() and any(p.glob("**/*.mp4")):
            has_videos = True
            break

    if has_meta and has_videos:
        print("  [Dataset] FakeAVCeleb videos and metadata verified on local SSD.")
        return

    print("  [Dataset] FakeAVCeleb dataset missing or incomplete on local SSD. Extracting from Google Drive...")
    fav_zip = search_drive_file("fakeavceleb.zip")
    if not fav_zip:
        print("  [WARNING] fakeavceleb.zip not found in Drive. Checking for meta_data.csv...")
        csv_path = search_drive_file("meta_data.csv")
        if csv_path:
            target_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, target_meta)
            print(f"  Copied metadata CSV to: {target_meta}")
        return

    print(f"  Extracting {fav_zip.name} ({fav_zip.stat().st_size / 1e6:.1f} MB) to {REPO_ROOT / 'data/raw'} ...")
    dest = REPO_ROOT / "data/raw"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fav_zip) as zf:
        zf.extractall(dest)

    # Normalize folder name if extracted as data/FakeAVCeleb_v1.2
    alt_dir = REPO_ROOT / "data/raw/data/FakeAVCeleb_v1.2"
    target_dir = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2"
    if alt_dir.exists() and not target_dir.exists():
        shutil.move(str(alt_dir), str(target_dir))
    v_cnt = len(list(REPO_ROOT.glob("data/**/*.mp4")))
    print(f"  [Dataset] FakeAVCeleb extraction complete. Total MP4 videos found: {v_cnt:,}")

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

    # 1. Ensure test dataset is present
    ensure_fakeavceleb_dataset()

    # 2. Resolve best valid checkpoint (> 50 MB)
    drive_p2_bottleneck = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode/best_phase2_bottleneck.pt")
    local_p2_bottleneck = REPO_ROOT / "checkpoints/full/bottleneck_mode/best_phase2_bottleneck.pt"
    local_p2_alt        = REPO_ROOT / "checkpoints/full/best_phase2_bottleneck.pt"
    drive_p2_root       = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/best_phase2_bottleneck.pt")

    candidates = []
    if args.checkpoint:
        candidates.append(Path(args.checkpoint))
    candidates.extend([
        drive_p2_bottleneck,
        drive_p2_root,
        local_p2_bottleneck,
        local_p2_alt,
    ])

    ckpt_path = None
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 50000000: # > 50 MB valid checkpoint
            ckpt_path = cand
            break

    if not ckpt_path:
        print("ERROR: No valid checkpoint (> 50 MB) found! Please verify Phase 1 or Phase 2 training.")
        sys.exit(1)

    print(f"  Selected Checkpoint : {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
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
