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
DRIVE_OUTPUT_DIR = DRIVE_BASE / "checkpoints"

def search_drive_file(zip_name: str) -> Path | None:
    for folder in [DRIVE_BASE, DRIVE_BASE / "datasets", DRIVE_BASE / "preprocessed"]:
        if folder.exists():
            for f in folder.iterdir():
                if f.name.lower() == zip_name.lower():
                    return f
    return None

def extract_zip(zip_name: str, extract_to: Path) -> bool:
    zip_path = search_drive_file(zip_name)
    if not zip_path:
        print(f"  [MISSING - REQUIRED] {zip_name}")
        return False
    size_mb = zip_path.stat().st_size / 1e6
    print(f"  Extracting {zip_name} ({size_mb:.1f} MB) from {zip_path.parent.name}/{zip_path.name} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
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

    # Extract Preprocessed features cache
    print("\nExtracting Preprocessed Feature Cache...")
    preprocessed_ok = False
    for zip_candidate in ["preprocessed.zip", "Copy of preprocessed_all.zip", "preprocessed_all.zip"]:
        if extract_zip(zip_candidate, REPO_ROOT / "data/preprocessed"):
            preprocessed_ok = True
            break
    if not preprocessed_ok:
        print("ERROR: Could not find preprocessed feature zip in Drive.")
        return

    # Validate and generate manifests
    print("\nValidating preprocessed features and generating dataset splits...")
    os.chdir(str(REPO_ROOT))
    os.system("python scripts/validate_training_prep.py")
    os.system("python scripts/create_dataset_splits.py")

    # Run Stage 1 Training (40 epochs)
    print("\n" + "=" * 60)
    print("  RUNNING PHASE 1 TRAINING (EMOTION_BILINEAR) - 40 EPOCHS")
    print("=" * 60)
    cmd = "python scripts/train_full.py --device cuda --epochs 40 --classifier_mode emotion_bilinear --no_phase2"
    os.system(cmd)

    # Copy checkpoints and logs to Drive
    ckpt = REPO_ROOT / "checkpoints/full/best_phase1_emotion_bilinear.pt"
    if ckpt.exists():
        shutil.copy2(ckpt, DRIVE_OUTPUT_DIR / "best_phase1_emotion_bilinear.pt")
        print(f"\n[BACKUP] Saved Phase 1 checkpoint to Google Drive: {DRIVE_OUTPUT_DIR / 'best_phase1_emotion_bilinear.pt'}")

    logs = REPO_ROOT / "logs/full"
    if logs.exists():
        shutil.copytree(logs, DRIVE_OUTPUT_DIR / "logs_phase1_emotion_bilinear", dirs_exist_ok=True)
        print("[BACKUP] Saved training logs to Google Drive.")

    print("\nStage 1 complete! Model is ready for Stage 2.")

if __name__ == "__main__":
    main()
