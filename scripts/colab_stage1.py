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

def verify_dataset_completeness():
    """Pre-flight validator: ensures manifests and feature files are 100% complete before training."""
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT DATASET & FEATURE CACHE VALIDATOR")
    print("=" * 60)

    # 1. Check training turnover manifest in Drive if missing locally
    turnover_csv = REPO_ROOT / "data/processed/training_turnover_manifest.csv"
    if not turnover_csv.exists():
        drive_turnover = search_drive_file("training_turnover_manifest.csv")
        if drive_turnover and drive_turnover.exists():
            turnover_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_turnover, turnover_csv)
            print(f"  [FOUND ON DRIVE] Copied training_turnover_manifest.csv from Drive -> {turnover_csv}")

    # 2. Extract all feature archives from Drive
    print("\nChecking and extracting all feature archives from Google Drive...")
    extract_zip("preprocessed.zip", REPO_ROOT / "data/preprocessed", optional=True)
    extract_zip("existing_features.zip", REPO_ROOT / "data/preprocessed", optional=True)
    for i in range(4):
        extract_zip(f"mosei_features_shard{i}.zip", REPO_ROOT / "data/preprocessed", optional=True)
    extract_zip("mosei_features.zip", REPO_ROOT / "data/preprocessed", optional=True)
    extract_zip("metadata.zip", REPO_ROOT / "data", optional=True)

    # 3. Direct-sync loose features from Drive if local cache is incomplete
    drive_prep_features = DRIVE_BASE / "preprocessed/features"
    local_prep_features = REPO_ROOT / "data/preprocessed/features"
    (local_prep_features / "z_at").mkdir(parents=True, exist_ok=True)
    (local_prep_features / "z_v").mkdir(parents=True, exist_ok=True)

    local_at_count = len(list((local_prep_features / "z_at").glob("*.pt")))
    if local_at_count < 10000 and drive_prep_features.exists() and (drive_prep_features / "z_at").exists():
        drive_at_names = set(p.name for p in (drive_prep_features / "z_at").glob("*.pt"))
        local_at_names = set(p.name for p in (local_prep_features / "z_at").glob("*.pt"))
        missing_names  = drive_at_names - local_at_names
        
        if missing_names:
            synced = 0
            for name in missing_names:
                drive_at = drive_prep_features / "z_at" / name
                drive_v  = drive_prep_features / "z_v" / name
                local_at = local_prep_features / "z_at" / name
                local_v  = local_prep_features / "z_v" / name
                shutil.copy2(drive_at, local_at)
                synced += 1
                if drive_v.exists():
                    shutil.copy2(drive_v, local_v)
            print(f"  [DRIVE SYNC] Fast-synced {synced:,} missing feature pairs from Google Drive.")

    # 4. Flatten nested directory structure if present
    nested_prep = REPO_ROOT / "data/preprocessed/preprocessed"
    if nested_prep.exists() and nested_prep.is_dir():
        for item in nested_prep.iterdir():
            target = REPO_ROOT / "data/preprocessed" / item.name
            if not target.exists():
                shutil.move(str(item), str(target))

    # 5. Run validate_training_prep and create_dataset_splits
    print("\nValidating preprocessed features and generating dataset splits...")
    os.chdir(str(REPO_ROOT))
    os.system("python scripts/validate_training_prep.py")
    os.system("python scripts/create_dataset_splits.py")

    # 6. Final completeness verification summary
    train_csv = REPO_ROOT / "data/processed/train_manifest.csv"
    val_csv   = REPO_ROOT / "data/processed/val_manifest.csv"
    test_csv  = REPO_ROOT / "data/processed/internal_test_manifest.csv"

    n_train = 0
    n_val   = 0
    n_test  = 0
    if train_csv.exists():
        with open(train_csv, encoding="utf-8", errors="replace") as f:
            n_train = max(0, sum(1 for _ in f) - 1)
    if val_csv.exists():
        with open(val_csv, encoding="utf-8", errors="replace") as f:
            n_val = max(0, sum(1 for _ in f) - 1)
    if test_csv.exists():
        with open(test_csv, encoding="utf-8", errors="replace") as f:
            n_test = max(0, sum(1 for _ in f) - 1)

    z_at_cnt = len(list((local_prep_features / "z_at").glob("*.pt"))) if (local_prep_features / "z_at").exists() else 0
    z_v_cnt  = len(list((local_prep_features / "z_v").glob("*.pt"))) if (local_prep_features / "z_v").exists() else 0

    print("\n" + "=" * 60)
    print("  PRE-FLIGHT VALIDATION SUMMARY & COMPLETENESS REPORT")
    print("=" * 60)
    print(f"  Feature Tensors on Disk : {z_at_cnt:,} z_at files, {z_v_cnt:,} z_v files")
    print(f"  Train Manifest          : {n_train:,} clips {'[100% READY]' if n_train > 0 else '[MISSING/EMPTY]'}")
    print(f"  Val Manifest            : {n_val:,} clips {'[100% READY]' if n_val > 0 else '[MISSING/EMPTY]'}")
    print(f"  Internal Test Manifest  : {n_test:,} clips {'[100% READY]' if n_test > 0 else '[MISSING/EMPTY]'}")
    print(f"  Total Dataset Split     : {n_train + n_val + n_test:,} total clips")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  DEEP-SENTINEL COLAB: STAGE 1 TRAINING (EMOTION BILINEAR HEAD)")
    print("=" * 60)

    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        return

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_drive_zips()

    # Pre-flight dataset & feature cache validation
    verify_dataset_completeness()

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
    cmd = f"python scripts/train_full.py --device {dev} --epochs 50 --classifier_mode bottleneck --no_phase2"
    ret = os.system(cmd)
    if ret != 0:
        print(f"\n[ERROR] Stage 1 training failed with exit code {ret}.")
        sys.exit(1)

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
