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
        # 1. Attach preprocessed features directly from Drive without zip extraction
        print("\n[DRIVE FETCH] Attaching preprocessed features directly from Google Drive...")
        drive_prep_features = DRIVE_BASE / "preprocessed/features"
        local_prep_features = REPO_ROOT / "data/preprocessed/features"

        if drive_prep_features.exists():
            drive_at_count = len(list((drive_prep_features / "z_at").glob("*.pt"))) if (drive_prep_features / "z_at").exists() else 0
            local_at_count = len(list((local_prep_features / "z_at").glob("*.pt"))) if (local_prep_features / "z_at").exists() else 0

            if local_at_count == 0 and drive_at_count > 0:
                local_prep_features.parent.mkdir(parents=True, exist_ok=True)
                if local_prep_features.exists():
                    shutil.rmtree(local_prep_features, ignore_errors=True)
                try:
                    os.symlink(str(drive_prep_features), str(local_prep_features))
                    print(f"  [DRIVE ATTACHED] Instantly symlinked {drive_at_count:,} feature files from Drive (0-second extraction time).")
                except Exception:
                    (local_prep_features / "z_at").mkdir(parents=True, exist_ok=True)
                    (local_prep_features / "z_v").mkdir(parents=True, exist_ok=True)
                    synced = 0
                    for pt in (drive_prep_features / "z_at").glob("*.pt"):
                        local_at = local_prep_features / "z_at" / pt.name
                        local_v  = local_prep_features / "z_v" / pt.name
                        drive_v  = drive_prep_features / "z_v" / pt.name
                        if not local_at.exists():
                            shutil.copy2(pt, local_at)
                            synced += 1
                        if drive_v.exists() and not local_v.exists():
                            shutil.copy2(drive_v, local_v)
                    print(f"  [DRIVE FETCHED] Direct-fetched {synced:,} feature pairs from Google Drive.")

        # Print local feature count for validation
        z_at_dir = REPO_ROOT / "data/preprocessed/features/z_at"
        if z_at_dir.exists():
            files = list(z_at_dir.glob("*.pt"))
            print(f"\nLocal preprocessed feature count after Drive attach: {len(files):,} files in z_at.")

    # Locate and copy FakeAVCeleb metadata CSV (cached features used during inference)
    print("\nLooking for FakeAVCeleb meta_data.csv in Drive...")
    fav_meta = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv"
    if not fav_meta.exists():
        csv_path = search_drive_file("meta_data.csv")
        if csv_path:
            fav_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, fav_meta)
            print(f"  Successfully copied FakeAVCeleb metadata file to: {fav_meta}")
        else:
            print("  meta_data.csv not found directly. Extracting fakeavceleb.zip...")
            extract_zip("fakeavceleb.zip", REPO_ROOT / "data/raw", optional=True)
            # Check if extracted to data/FakeAVCeleb_v1.2 instead of data/raw/FakeAVCeleb_v1.2
            alt_meta = REPO_ROOT / "data/FakeAVCeleb_v1.2/meta_data.csv"
            if alt_meta.exists() and not fav_meta.exists():
                fav_meta.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(alt_meta.parent), str(fav_meta.parent))

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
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = f"python scripts/train_full.py --device {dev} --epochs 40 --classifier_mode bottleneck --no_phase2"
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
