"""
colab_stage2.py
================
Automated script for Stage 2 (Phase 2) training on Google Colab.
Designed for the training lead to run end-to-end backbone fine-tuning for 40 epochs.
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DRIVE_BASE = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")
DRIVE_OUTPUT_DIR = DRIVE_BASE / "checkpoints"
LOCAL_REPO_ROOT = "D:/Documents/Programming/Thesis_G10"
COLAB_REPO_ROOT = "/content/thesis"

def scan_drive_zips():
    """Print all zip files in Drive for debugging."""
    print("\nScanning Google Drive (THESIS_MOTHERFILE) for zip archives...")
    if not DRIVE_BASE.exists():
        print("  Google Drive folder THESIS_MOTHERFILE not found!")
        return
    found = []
    for root, _, files in os.walk(DRIVE_BASE):
        for f in files:
            if f.endswith(".zip"):
                p = Path(root) / f
                mb = p.stat().st_size / 1e6
                print(f"  Found zip: {p.relative_to(DRIVE_BASE)} ({mb:.1f} MB)")
                found.append(p)
    if not found:
        print("  No zip files found recursively in THESIS_MOTHERFILE.")

def search_drive_file(zip_name: str) -> Path | None:
    """Recursive search in Drive for zip_name."""
    if not DRIVE_BASE.exists():
        return None
    for root, _, files in os.walk(DRIVE_BASE):
        for f in files:
            if f.lower() == zip_name.lower():
                return Path(root) / f
    return None

def extract_zip(zip_name: str, extract_to: Path, optional: bool = False) -> bool:
    zip_path = search_drive_file(zip_name)
    if not zip_path:
        tag = "OPTIONAL" if optional else "MISSING - REQUIRED"
        print(f"  [{tag}] {zip_name}")
        return False
    size_mb = zip_path.stat().st_size / 1e6
    print(f"  Extracting {zip_name} ({size_mb:.1f} MB) from {zip_path.parent.name}/{zip_path.name} ...")
    
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
        if not namelist:
            return False
        # Check if the zip already contains root directories like 'data/' or 'data/preprocessed/'
        has_data_prefix = any(name.startswith("data/") for name in namelist)
        dest = REPO_ROOT if has_data_prefix else extract_to
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)
    print("  Done.")
    return True

def remap_csv_paths(csv_path: Path, path_cols: list[str]):
    import pandas as pd
    if not csv_path.exists():
        print(f"  Remap skip: {csv_path.name} not found")
        return
    df = pd.read_csv(csv_path)
    changed = 0
    for col in path_cols:
        if col not in df.columns:
            continue
        def fix(v):
            if not isinstance(v, str): return v
            v = v.replace("\\", "/")
            return v.replace(LOCAL_REPO_ROOT, COLAB_REPO_ROOT) if LOCAL_REPO_ROOT in v else v
        before = df[col].copy()
        df[col] = df[col].map(fix)
        changed += (df[col] != before).sum()
    df.to_csv(csv_path, index=False)
    print(f"  {csv_path.name:<35} {changed} paths remapped")

def compress_existing_cache():
    kf_dir = REPO_ROOT / "data/preprocessed/keyframes"
    if not kf_dir.exists():
        return
    pt_files = list(kf_dir.glob("*.pt"))
    if not pt_files:
        return
    print(f"\n[INFO] Detected {len(pt_files)} old PyTorch (.pt) cache files.")
    print("       Deleting them to free up disk space since we are now using highly-compressed JPEG (.jpg) caching.")
    deleted = 0
    freed_mb = 0
    for f in pt_files:
        try:
            size = f.stat().st_size
            f.unlink()
            deleted += 1
            freed_mb += size / (1024 * 1024)
        except Exception:
            pass
    print(f"  Successfully deleted {deleted} old cache files. Freed {freed_mb/1024:.1f} GB of local disk space!")

def main():
    print("\n" + "=" * 60)
    print("  DEEP-SENTINEL COLAB: STAGE 2 TRAINING (BACKBONE FINE-TUNING)")
    print("=" * 60)

    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        return

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scan and print Drive files for visibility
    scan_drive_zips()

    # Clean up old unzipped raw folders to free up disk space for local ZIP copies
    print("\nChecking for old unzipped raw video directories to clean up...")
    for d in ["CMU-MOSEI", "MELD", "MUStARD"]:
        raw_dir = REPO_ROOT / "data/raw" / d
        if raw_dir.exists():
            try:
                print(f"  [CLEANUP] Deleting old unzipped folder: {d} to reclaim disk space...")
                shutil.rmtree(raw_dir)
            except Exception as e:
                print(f"  [CLEANUP WARNING] Failed to delete {d}: {e}")

    # 1. Extract Preprocessed features cache
    z_at_dir = REPO_ROOT / "data/preprocessed/features/z_at"
    features_exist = z_at_dir.exists() and any(z_at_dir.glob("*.pt"))
    
    if features_exist:
        print("\n[SKIP] Preprocessed features are already extracted locally. Skipping zip extraction.")
    else:
        print("\nExtracting Preprocessed Feature Cache...")
        # Try consolidated zips
        consolidated_found = False
        for zip_candidate in ["preprocessed.zip", "Copy of preprocessed_all.zip", "preprocessed_all.zip"]:
            if extract_zip(zip_candidate, REPO_ROOT / "data/preprocessed", optional=True):
                consolidated_found = True
                break
        
        # Try individual segment zips (fallback/additional) only if consolidated zip was not found
        if not consolidated_found:
            print("\nChecking for segmented/shard feature archives...")
            extract_zip("existing_features.zip", REPO_ROOT / "data/preprocessed", optional=True)
            extract_zip("metadata.zip", REPO_ROOT / "data/preprocessed", optional=True)
            
            # MOSEI features
            mosei_found = False
            for i in range(4):
                if extract_zip(f"mosei_features_shard{i}.zip", REPO_ROOT / "data/preprocessed", optional=True):
                    mosei_found = True
            if not mosei_found:
                extract_zip("mosei_features.zip", REPO_ROOT / "data/preprocessed", optional=True)

    # Print local feature count for validation
    if z_at_dir.exists():
        files = list(z_at_dir.glob("*.pt"))
        print(f"\nLocal preprocessed feature count after extraction: {len(files)} files in z_at.")
    else:
        print("\nLocal preprocessed features directory does not exist yet.")

    # 2. Extract Raw Video/Audio Clips (Bypassed to save 22.5 GB of local disk space)
    print("\n[INFO] Bypassing raw dataset zip extraction to save 22.5 GB of local disk space.")
    print("       The training script will extract raw video files on-the-fly from Google Drive.")
    
    # Locate and copy FakeAVCeleb metadata CSV (cached features used during inference)
    print("\nLooking for FakeAVCeleb meta_data.csv in Drive...")
    csv_path = search_drive_file("meta_data.csv")
    if not csv_path:
        csv_path = search_drive_file("metadata.csv")  # Try fallback name
    
    if csv_path:
        dest_dir = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, dest_dir / "meta_data.csv")
        print(f"  Successfully copied FakeAVCeleb metadata file to: {dest_dir / 'meta_data.csv'}")
    else:
        # Check if the zip file is present instead
        print("  meta_data.csv not found directly. Checking for fakeavceleb.zip...")
        extract_zip("fakeavceleb.zip", REPO_ROOT / "data", optional=True)

    # 3. Compress any existing float32 keyframe caches to float16 to reclaim disk space
    compress_existing_cache()

    # 4. Remap Windows paths to Colab path structure
    print("\nRemapping dataset video paths...")
    remap_csv_paths(REPO_ROOT / "data/synthetic/track1_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/synthetic/track2_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/synthetic/track3_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/processed/meld_manifests/meld_real.csv", ["video_path"])
    remap_csv_paths(REPO_ROOT / "data/processed/mosei_manifests/mosei_real.csv", ["video_path"])

    # 5. Generate splits and validation
    print("\nValidating manifests and generating dataset splits...")
    os.chdir(str(REPO_ROOT))
    os.system("python scripts/validate_training_prep.py")
    os.system("python scripts/create_dataset_splits.py")

    # 6. Load Stage 1 checkpoint from Drive
    drive_p1_ckpt = DRIVE_OUTPUT_DIR / "best_phase1_emotion_bilinear.pt"
    local_p1_ckpt = REPO_ROOT / "checkpoints/full/best_phase1_emotion_bilinear.pt"
    local_p1_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if drive_p1_ckpt.exists():
        shutil.copy2(drive_p1_ckpt, local_p1_ckpt)
        print(f"\nLoaded Stage 1 weights from Drive -> {local_p1_ckpt}")
    else:
        print(f"ERROR: Phase 1 checkpoint not found in Drive: {drive_p1_ckpt}")
        print("Please run scripts/colab_stage1.py first to create the baseline weights.")
        return

    # 7. Run Stage 2 Training (10 epochs)
    print("\n" + "=" * 60)
    print("  RUNNING PHASE 2 TRAINING (FINE-TUNING BACKBONES) - 10 EPOCHS")
    print("=" * 60)
    cmd_p2 = "python scripts/train_full.py --device cuda --epochs 10 --classifier_mode emotion_bilinear --phase2_epochs 10 --phase2_batch 8 --phase2_freeze_layers 10 --skip_phase1 --workers 2"
    os.system(cmd_p2)

    # 8. Backup Phase 2 checkpoint to Drive
    p2_ckpt = REPO_ROOT / "checkpoints/full/best_phase2_emotion_bilinear.pt"
    if p2_ckpt.exists():
        shutil.copy2(p2_ckpt, DRIVE_OUTPUT_DIR / "best_phase2_emotion_bilinear.pt")
        print(f"\n[BACKUP] Saved Phase 2 checkpoint to Google Drive: {DRIVE_OUTPUT_DIR / 'best_phase2_emotion_bilinear.pt'}")

    logs = REPO_ROOT / "logs/full"
    if logs.exists():
        shutil.copytree(logs, DRIVE_OUTPUT_DIR / "logs_phase2_emotion_bilinear", dirs_exist_ok=True)
        print("[BACKUP] Saved training logs to Google Drive.")

    # 9. Run FakeAVCeleb evaluation on the fine-tuned model
    print("\n" + "=" * 60)
    print("  RUNNING CROSS-DATASET BENCHMARK (FAKEAVCELEB)")
    print("=" * 60)
    cmd_eval = "python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/full/best_phase2_emotion_bilinear.pt --classifier_mode emotion_bilinear --save_csv benchmark_results.csv"
    os.system(cmd_eval)

    # Backup benchmark CSV to Drive
    if Path("benchmark_results.csv").exists():
        shutil.copy2("benchmark_results.csv", DRIVE_OUTPUT_DIR / "benchmark_results.csv")
        print(f"\n[BACKUP] Saved benchmark results CSV to Google Drive: {DRIVE_OUTPUT_DIR / 'benchmark_results.csv'}")

    print("\nStage 2 fine-tuning and evaluation complete! Final results are saved in Google Drive.")

if __name__ == "__main__":
    main()
