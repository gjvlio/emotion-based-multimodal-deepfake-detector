"""
colab_stage1.py
================
Automated script for Stage 1 (Phase 1) training on Google Colab.
Designed for the training lead to train the emotion bilinear head for 40 epochs on cached features.
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
DRIVE_OUTPUT_DIR = DRIVE_BASE / "checkpoints/latest"

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
    """Recursive search in Drive for zip_name with exact and partial matching."""
    if not DRIVE_BASE.exists():
        return None
    # 1. Exact match search
    for root, _, files in os.walk(DRIVE_BASE):
        for f in files:
            if f.lower() == zip_name.lower():
                return Path(root) / f
    # 2. Substring match fallback
    stem = Path(zip_name).stem.lower()
    for root, _, files in os.walk(DRIVE_BASE):
        for f in files:
            if f.endswith('.zip') and stem in f.lower():
                return Path(root) / f
    return None

from typing import Optional

def extract_zip(zip_name: str, extract_to: Path, optional: bool = False, check_dir: Optional[Path] = None) -> bool:
    zip_path = search_drive_file(zip_name)
    if not zip_path:
        tag = "OPTIONAL" if optional else "MISSING - REQUIRED"
        print(f"  [{tag}] {zip_name}")
        return False
        
    size_mb = zip_path.stat().st_size / 1e6

    # Smart automatic sample-file extraction check
    try:
        with zipfile.ZipFile(zip_path) as zf:
            namelist = zf.namelist()
            if namelist:
                has_data_prefix = any(name.startswith("data/") for name in namelist)
                dest = REPO_ROOT if has_data_prefix else extract_to
                sample_files = [n for n in namelist if not n.endswith('/') and not n.startswith('__MACOSX')]
                if sample_files:
                    sample_check = dest / sample_files[min(5, len(sample_files)-1)]
                    if sample_check.exists():
                        print(f"  [ALREADY EXTRACTED] Skipping {zip_name} ({size_mb:.1f} MB) — already extracted on local SSD.")
                        return True
    except Exception:
        pass

    if check_dir is not None and check_dir.exists():
        try:
            if any(check_dir.iterdir()):
                print(f"  [ALREADY EXTRACTED] Skipping {zip_name} ({size_mb:.1f} MB) — directory {check_dir.name} already exists.")
                return True
        except Exception:
            pass

    print(f"  Extracting {zip_name} ({size_mb:.1f} MB) from {zip_path.parent.name}/{zip_path.name} ...")
    
    with zipfile.ZipFile(zip_path) as zf:
        namelist = zf.namelist()
        if not namelist:
            return False
        has_data_prefix = any(name.startswith("data/") for name in namelist)
        has_prep_prefix = any(name.startswith("preprocessed/") for name in namelist)
        
        if has_data_prefix:
            dest = REPO_ROOT
        elif has_prep_prefix:
            dest = REPO_ROOT / "data"
        else:
            dest = extract_to

        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)
        
        # Flatten nested preprocessed/preprocessed/ if present
        nested = REPO_ROOT / "data/preprocessed/preprocessed"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                target = REPO_ROOT / "data/preprocessed" / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))

    print("  Done.")
    return True

def main():
    print("\n" + "=" * 60)
    print("  DEEP-SENTINEL COLAB: STAGE 1 TRAINING (FROZEN BACKBONES)")
    print("=" * 60)

    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        return

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scan and print Drive files for visibility
    scan_drive_zips()

    # Extract Preprocessed features cache
    z_at_dir = REPO_ROOT / "data/preprocessed/features/z_at"
    features_exist = z_at_dir.exists() and any(z_at_dir.glob("*.pt"))
    
    if features_exist:
        print("\n[SKIP] Preprocessed features are already extracted locally. Skipping zip extraction.")
    else:
        print("\nExtracting Preprocessed Feature Cache...")
        # 1. Try consolidated zips
        consolidated_found = False
        for zip_candidate in ["preprocessed.zip", "Copy of preprocessed_all.zip", "preprocessed_all.zip"]:
            if extract_zip(zip_candidate, REPO_ROOT / "data/preprocessed", optional=True):
                consolidated_found = True
                break
        
        # 2. Try individual segment zips (fallback/additional)
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

    # Print local feature count for validation
    z_at_dir = REPO_ROOT / "data/preprocessed/features/z_at"
    if z_at_dir.exists():
        files = list(z_at_dir.glob("*.pt"))
        print(f"\nLocal preprocessed feature count after extraction: {len(files)} files in z_at.")
    else:
        print("\nLocal preprocessed features directory does not exist yet.")

    # Validate and generate manifests
    print("\nValidating preprocessed features and generating dataset splits...")
    os.chdir(str(REPO_ROOT))
    os.system("python scripts/validate_training_prep.py")
    os.system("python scripts/create_dataset_splits.py")

    # Run Stage 1 Training (40 epochs)
    print("\n" + "=" * 60)
    print("  RUNNING PHASE 1 TRAINING (BOTTLENECK_MODE) - 40 EPOCHS")
    print("=" * 60)
    cmd = "python scripts/train_full.py --device cuda --epochs 40 --classifier_mode bottleneck --no_phase2"
    os.system(cmd)

    # Copy checkpoints and logs to Drive under bottleneck_mode folder
    drive_mode_dir = DRIVE_BASE / "checkpoints/latest/bottleneck_mode"
    drive_mode_dir.mkdir(parents=True, exist_ok=True)
    ckpt = REPO_ROOT / "checkpoints/full/bottleneck_mode/best_phase1_bottleneck.pt"
    if not ckpt.exists():
        ckpt = REPO_ROOT / "checkpoints/full/best_phase1_bottleneck.pt"
    if ckpt.exists():
        shutil.copy2(ckpt, drive_mode_dir / "best_phase1_bottleneck.pt")
        print(f"\n[BACKUP] Saved Phase 1 bottleneck checkpoint to Google Drive: {drive_mode_dir / 'best_phase1_bottleneck.pt'}")

    logs = REPO_ROOT / "logs/full"
    if logs.exists():
        shutil.copytree(logs, drive_mode_dir / "logs_phase1_bottleneck", dirs_exist_ok=True)
        print("[BACKUP] Saved training logs to Google Drive.")

    print("\nStage 1 bottleneck mode complete! Model is ready for Stage 2.")

if __name__ == "__main__":
    main()
