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
from typing import Optional

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DRIVE_BASE = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")
DRIVE_OUTPUT_DIR = DRIVE_BASE / "checkpoints/latest"
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
                has_prep_prefix = any(name.startswith("preprocessed/") for name in namelist)
                if has_data_prefix:
                    dest = REPO_ROOT
                elif has_prep_prefix:
                    dest = REPO_ROOT / "data"
                else:
                    dest = extract_to

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

def verify_dataset_completeness():
    """Pre-flight validator: ensures manifests and feature files are 100% complete before Stage 2."""
    print("\n" + "=" * 60)
    print("  PRE-FLIGHT DATASET & FEATURE CACHE VALIDATOR (STAGE 2)")
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
    print("  PRE-FLIGHT VALIDATION SUMMARY & COMPLETENESS REPORT (STAGE 2)")
    print("=" * 60)
    print(f"  Feature Tensors on Disk : {z_at_cnt:,} z_at files, {z_v_cnt:,} z_v files")
    print(f"  Train Manifest          : {n_train:,} clips {'[100% READY]' if n_train > 0 else '[MISSING/EMPTY]'}")
    print(f"  Val Manifest            : {n_val:,} clips {'[100% READY]' if n_val > 0 else '[MISSING/EMPTY]'}")
    print(f"  Internal Test Manifest  : {n_test:,} clips {'[100% READY]' if n_test > 0 else '[MISSING/EMPTY]'}")
    print(f"  Total Dataset Split     : {n_train + n_val + n_test:,} total clips")
    print("=" * 60 + "\n")


def main():
    print("\n" + "=" * 60)
    print("  DEEP-SENTINEL COLAB: STAGE 2 TRAINING (BACKBONE FINE-TUNING)")
    print("=" * 60)

    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        return

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_drive_zips()

    # Pre-flight dataset & feature cache validation
    verify_dataset_completeness()



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
    if z_at_dir.exists():
        files = list(z_at_dir.glob("*.pt"))
        print(f"\nLocal preprocessed feature count after extraction: {len(files)} files in z_at.")
    else:
        print("\nLocal preprocessed features directory does not exist yet.")

    # 2. Extract Raw Video/Audio Clips (Skip if already extracted to local SSD)
    print("\n[COLAB PRO] Checking raw dataset directories on local SSD...")
    extract_zip("tracks_1_2_3_4.zip", REPO_ROOT / "data", optional=True, check_dir=REPO_ROOT / "data/synthetic/track1_fakes")
    extract_zip("meld_raw.zip", REPO_ROOT / "data", optional=True, check_dir=REPO_ROOT / "data/raw/MELD")
    extract_zip("mustard.zip", REPO_ROOT / "data", optional=True, check_dir=REPO_ROOT / "data/raw/MUStARD")
    extract_zip("cmumosei.zip", REPO_ROOT / "data", optional=True, check_dir=REPO_ROOT / "data/raw/CMU-MOSEI")
    
    # Extract FakeAVCeleb for cross-dataset evaluation
    print("\nChecking FakeAVCeleb test dataset...")
    extract_zip("fakeavceleb.zip", REPO_ROOT / "data", optional=True, check_dir=REPO_ROOT / "data/raw/FakeAVCeleb_v1.2")

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

    # 6. Load Stage 1 checkpoint from Drive (check latest/bottleneck_mode first)
    drive_mode_dir = DRIVE_BASE / "checkpoints/latest/bottleneck_mode"
    drive_mode_dir.mkdir(parents=True, exist_ok=True)
    
    drive_p1_ckpt = drive_mode_dir / "best_phase1_bottleneck.pt"
    local_p1_ckpt = REPO_ROOT / "checkpoints/full/bottleneck_mode/best_phase1_bottleneck.pt"
    local_p1_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if drive_p1_ckpt.exists():
        shutil.copy2(drive_p1_ckpt, local_p1_ckpt)
        print(f"\nLoaded Stage 1 bottleneck weights from Drive -> {local_p1_ckpt}")
    else:
        print(f"ERROR: Phase 1 bottleneck checkpoint not found in Drive: {drive_p1_ckpt}")
        print("Please run scripts/colab_stage1.py first to create the bottleneck baseline weights.")
        return

    # 7. Run Stage 2 Training (40 epochs)
    print("\n" + "=" * 60)
    print("  RUNNING PHASE 2 TRAINING (BOTTLENECK_MODE FINE-TUNING) - 40 EPOCHS")
    print("=" * 60)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Colab Pro GPU detection (A100 / V100 High-RAM optimization)
    p2_batch   = 4
    p2_workers = 0
    if dev == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[GPU DETECTED] {gpu_name} ({gpu_vram:.1f} GB VRAM)")
        if gpu_vram > 15.0 or "A100" in gpu_name or "V100" in gpu_name:
            p2_batch   = 16
            p2_workers = 2
            print("  [COLAB PRO OPTIMIZATION] Enabling High-Performance Batching (Batch=16, Workers=2) for ~2.5 mins/epoch!")
        else:
            print("  [COLAB STANDARD OPTIMIZATION] Enabling Safe Batching (Batch=4, Workers=0) to prevent OOM!")

    cmd_p2 = f"python scripts/train_full.py --device {dev} --epochs 50 --classifier_mode bottleneck --phase2_epochs 50 --phase2_batch {p2_batch} --phase2_freeze_layers 10 --skip_phase1 --workers {p2_workers} --patience 10"
    ret_p2 = os.system(cmd_p2)
    if ret_p2 != 0:
        print(f"\n[ERROR] Stage 2 training failed with exit code {ret_p2}.")
        sys.exit(1)

    # 8. Backup Phase 2 checkpoint to Drive under bottleneck_mode folder
    p2_ckpt = REPO_ROOT / "checkpoints/full/bottleneck_mode/best_phase2_bottleneck.pt"
    if not p2_ckpt.exists():
        p2_ckpt = REPO_ROOT / "checkpoints/full/best_phase2_bottleneck.pt"
    if p2_ckpt.exists():
        shutil.copy2(p2_ckpt, drive_mode_dir / "best_phase2_bottleneck.pt")
        print(f"\n[BACKUP] Saved Phase 2 bottleneck checkpoint to Google Drive: {drive_mode_dir / 'best_phase2_bottleneck.pt'}")

    logs = REPO_ROOT / "logs/full"
    if logs.exists():
        shutil.copytree(logs, drive_mode_dir / "logs_phase2_bottleneck", dirs_exist_ok=True)
        print("[BACKUP] Saved training logs to Google Drive.")

    # 9. Run FakeAVCeleb evaluation on the bottleneck fine-tuned model
    print("\n" + "=" * 60)
    print("  RUNNING CROSS-DATASET BENCHMARK (FAKEAVCELEB - BOTTLENECK MODE)")
    print("=" * 60)
    cmd_eval = f"python scripts/evaluate_fakeavceleb.py --checkpoint {p2_ckpt} --classifier_mode bottleneck --n_real 500 --n_fake 9500 --save_csv benchmark_results_bottleneck.csv"
    os.system(cmd_eval)

    # Backup benchmark CSV to Drive
    if Path("benchmark_results_bottleneck.csv").exists():
        shutil.copy2("benchmark_results_bottleneck.csv", drive_mode_dir / "benchmark_results_bottleneck.csv")
        print(f"\n[BACKUP] Saved bottleneck benchmark results CSV to Google Drive: {drive_mode_dir / 'benchmark_results_bottleneck.csv'}")

    print("\nStage 2 bottleneck fine-tuning and evaluation complete! Final results are saved in Google Drive under bottleneck_mode/.")

if __name__ == "__main__":
    main()
