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

def search_drive_file(zip_name: str) -> Path | None:
    for folder in [DRIVE_BASE, DRIVE_BASE / "datasets", DRIVE_BASE / "preprocessed"]:
        if folder.exists():
            for f in folder.iterdir():
                if f.name.lower() == zip_name.lower():
                    return f
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

def main():
    print("\n" + "=" * 60)
    print("  DEEP-SENTINEL COLAB: STAGE 2 TRAINING (BACKBONE FINE-TUNING)")
    print("=" * 60)

    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        return

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract Preprocessed features cache
    print("\nExtracting Preprocessed Feature Cache...")
    # Try consolidated zips
    consolidated_found = False
    for zip_candidate in ["preprocessed.zip", "Copy of preprocessed_all.zip", "preprocessed_all.zip"]:
        if extract_zip(zip_candidate, REPO_ROOT / "data/preprocessed", optional=True):
            consolidated_found = True
            break
    
    # Try individual segment zips (fallback/additional)
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

    # 2. Extract Raw Video/Audio Clips
    print("\nExtracting Raw Datasets (Phase 2)...")
    extract_zip("tracks_1_2_3_4.zip", REPO_ROOT / "data")
    extract_zip("meld_raw.zip", REPO_ROOT / "data")
    extract_zip("mustard.zip", REPO_ROOT / "data")
    extract_zip("Fakeavceleb.zip", REPO_ROOT / "data")

    # 3. Remap Windows paths to Colab path structure
    print("\nRemapping dataset video paths...")
    remap_csv_paths(REPO_ROOT / "data/synthetic/track1_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/synthetic/track2_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/synthetic/track3_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/processed/meld_manifests/meld_real.csv", ["video_path"])
    remap_csv_paths(REPO_ROOT / "data/processed/mosei_manifests/mosei_real.csv", ["video_path"])

    # 4. Generate splits and validation
    print("\nValidating manifests and generating dataset splits...")
    os.chdir(str(REPO_ROOT))
    os.system("python scripts/validate_training_prep.py")
    os.system("python scripts/create_dataset_splits.py")

    # 5. Load Stage 1 checkpoint from Drive
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

    # 6. Run Stage 2 Training (40 epochs)
    print("\n" + "=" * 60)
    print("  RUNNING PHASE 2 TRAINING (FINE-TUNING BACKBONES) - 40 EPOCHS")
    print("=" * 60)
    cmd_p2 = "python scripts/train_full.py --device cuda --epochs 40 --classifier_mode emotion_bilinear --phase2_epochs 40 --phase2_batch 4 --phase2_freeze_layers 10"
    os.system(cmd_p2)

    # 7. Backup Phase 2 checkpoint to Drive
    p2_ckpt = REPO_ROOT / "checkpoints/full/best_phase2_emotion_bilinear.pt"
    if p2_ckpt.exists():
        shutil.copy2(p2_ckpt, DRIVE_OUTPUT_DIR / "best_phase2_emotion_bilinear.pt")
        print(f"\n[BACKUP] Saved Phase 2 checkpoint to Google Drive: {DRIVE_OUTPUT_DIR / 'best_phase2_emotion_bilinear.pt'}")

    logs = REPO_ROOT / "logs/full"
    if logs.exists():
        shutil.copytree(logs, DRIVE_OUTPUT_DIR / "logs_phase2_emotion_bilinear", dirs_exist_ok=True)
        print("[BACKUP] Saved training logs to Google Drive.")

    # 8. Run FakeAVCeleb evaluation on the fine-tuned model
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
