"""
colab_run_stage.py
===================
Automated script to run Phase 1 + Phase 2 training and evaluation on Google Colab.
Designed to automatically locate datasets in the user's mounted Google Drive.

Steps:
1. Auto-detects Google Drive path.
2. Extracts preprocessed features zip.
3. Extracts raw video/audio datasets (for Phase 2).
4. Remaps dataset manifest paths from Windows format to Colab format.
5. Re-runs preprocessing validation and split manifests.
6. Runs Phase 1 training (emotion_bilinear mode) -> Saves checkpoint to Drive.
7. Runs Phase 2 training (emotion_bilinear mode) -> Saves checkpoint to Drive.
8. Runs internal test set evaluation comparing all modes.
9. Evaluates the fine-tuned model on the FakeAVCeleb benchmark.
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
    """Look for zip_name in base, datasets/ or preprocessed/ directories in Drive."""
    candidates = [
        DRIVE_BASE / zip_name,
        DRIVE_BASE / "datasets" / zip_name,
        DRIVE_BASE / "preprocessed" / zip_name,
    ]
    # Check exact case
    for c in candidates:
        if c.exists():
            return c
    # Check case-insensitive
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
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
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
    print("  AUTOMATED DEEP-SENTINEL COLAB TRAINING PIPELINE")
    print("=" * 60)

    # 1. Mount verification
    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        print("Please ensure your Drive is mounted and 'THESIS_MOTHERFILE' exists in MyDrive.")
        return

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Extract Preprocessed features cache
    print("\nExtracting Preprocessed Feature Cache...")
    preprocessed_ok = False
    for zip_candidate in ["preprocessed.zip", "Copy of preprocessed_all.zip", "preprocessed_all.zip"]:
        if extract_zip(zip_candidate, REPO_ROOT / "data/preprocessed"):
            preprocessed_ok = True
            break
    if not preprocessed_ok:
        print("ERROR: Could not find preprocessed feature zip in Drive.")
        return

    # 3. Extract Raw Datasets (for Phase 2)
    print("\nExtracting Raw Datasets (Phase 2)...")
    extract_zip("tracks_1_2_3_4.zip", REPO_ROOT / "data", optional=True)
    extract_zip("meld_raw.zip", REPO_ROOT / "data", optional=True)
    extract_zip("mustard.zip", REPO_ROOT / "data", optional=True)
    extract_zip("Fakeavceleb.zip", REPO_ROOT / "data", optional=True)

    # 4. Remap paths in manifests
    print("\nRemapping dataset paths from Windows to Colab formats...")
    remap_csv_paths(REPO_ROOT / "data/synthetic/track1_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/synthetic/track2_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/synthetic/track3_fakes/metadata.csv", ["output_path", "input_path"])
    remap_csv_paths(REPO_ROOT / "data/processed/meld_manifests/meld_real.csv", ["video_path"])
    remap_csv_paths(REPO_ROOT / "data/processed/mosei_manifests/mosei_real.csv", ["video_path"])

    # 5. Preprocessing validation and split manifests
    print("\nRe-generating splits and validating training manifests...")
    os.chdir(str(REPO_ROOT))
    os.system("python scripts/validate_training_prep.py")
    os.system("python scripts/create_dataset_splits.py")

    # 6. Run Phase 1 Training (10 Epochs, emotion_bilinear mode)
    print("\n" + "=" * 60)
    print("  RUNNING PHASE 1 TRAINING (EMOTION_BILINEAR)")
    print("=" * 60)
    cmd_p1 = "python scripts/train_full.py --device cuda --epochs 10 --classifier_mode emotion_bilinear --no_phase2"
    os.system(cmd_p1)

    # Backup Phase 1 checkpoint to Drive
    p1_ckpt = REPO_ROOT / "checkpoints/full/best_phase1_emotion_bilinear.pt"
    if p1_ckpt.exists():
        shutil.copy2(p1_ckpt, DRIVE_OUTPUT_DIR / "best_phase1_emotion_bilinear.pt")
        print(f"\n[BACKUP] Saved Phase 1 checkpoint to Google Drive: {DRIVE_OUTPUT_DIR / 'best_phase1_emotion_bilinear.pt'}")

    # 7. Run Phase 2 Training (10 Epochs, emotion_bilinear mode)
    # Check if raw clips are present to run Phase 2
    raw_ok = (REPO_ROOT / "data/synthetic/track1_fakes").exists()
    if raw_ok:
        print("\n" + "=" * 60)
        print("  RUNNING PHASE 2 TRAINING (FINE-TUNING BACKBONES)")
        print("=" * 60)
        cmd_p2 = "python scripts/train_full.py --device cuda --epochs 50 --classifier_mode bottleneck --phase2_epochs 10 --phase2_lr 1e-6 --phase2_freeze_layers 2"
        os.system(cmd_p2)

        # Backup Phase 2 checkpoint to Drive
        p2_ckpt = REPO_ROOT / "checkpoints/full/bottleneck_mode/best_phase2_bottleneck.pt"
        if not p2_ckpt.exists():
            p2_ckpt = REPO_ROOT / "checkpoints/full/best_phase2_bottleneck.pt"
        if p2_ckpt.exists():
            shutil.copy2(p2_ckpt, DRIVE_OUTPUT_DIR / "latest/bottleneck_mode/best_phase2_bottleneck.pt")
            print(f"\n[BACKUP] Saved Phase 2 checkpoint to Google Drive: {DRIVE_OUTPUT_DIR / 'latest/bottleneck_mode/best_phase2_bottleneck.pt'}")

        # 8. Run Evaluation (Compare all trained architectures)
        print("\n" + "=" * 60)
        print("  RUNNING CLASSIFIER MODE COMPARISONS")
        print("=" * 60)
        # Try to copy existing checkpoints from Drive so they can all be compared
        for m in ["baseline", "mismatch_only", "bottleneck", "high_dropout"]:
            drive_m_ckpt = DRIVE_OUTPUT_DIR / f"latest/bottleneck_mode/best_phase1_{m}.pt"
            local_m_ckpt = REPO_ROOT / "checkpoints/full" / (f"best_phase1_{m}.pt" if m != "baseline" else "best_phase1_baseline.pt")
            if drive_m_ckpt.exists() and not local_m_ckpt.exists():
                shutil.copy2(drive_m_ckpt, local_m_ckpt)
        os.system("python scripts/evaluate_all_models.py")

        # 9. Run FakeAVCeleb benchmark on the final model
        print("\n" + "=" * 60)
        print("  RUNNING CROSS-DATASET BENCHMARK (FAKEAVCELEB)")
        print("=" * 60)
        final_ckpt = p2_ckpt if p2_ckpt.exists() else p1_ckpt
        cmd_eval = f"python scripts/evaluate_fakeavceleb.py --checkpoint {final_ckpt} --classifier_mode bottleneck --n_real 500 --n_fake 9500 --save_csv benchmark_results_bottleneck.csv"
        os.system(cmd_eval)

        # Backup benchmark results to Drive
        if Path("benchmark_results_bottleneck.csv").exists():
            shutil.copy2("benchmark_results_bottleneck.csv", DRIVE_OUTPUT_DIR / "latest/bottleneck_mode/benchmark_results_bottleneck.csv")
            print(f"\n[BACKUP] Saved benchmark results CSV to Google Drive: {DRIVE_OUTPUT_DIR / 'latest/bottleneck_mode/benchmark_results_bottleneck.csv'}")
    else:
        print("\n[SKIP] Raw video/audio files missing in Drive datasets/. Skipping Phase 2 fine-tuning.")
        print("Running FakeAVCeleb evaluation on Phase 1 checkpoint instead...")
        cmd_eval = f"python scripts/evaluate_fakeavceleb.py --checkpoint {p1_ckpt} --classifier_mode bottleneck --n_real 500 --n_fake 9500 --save_csv benchmark_results_bottleneck.csv"
        os.system(cmd_eval)

        if Path("benchmark_results_bottleneck.csv").exists():
            shutil.copy2("benchmark_results_bottleneck.csv", DRIVE_OUTPUT_DIR / "latest/bottleneck_mode/benchmark_results_bottleneck.csv")
            print(f"\n[BACKUP] Saved benchmark results CSV to Google Drive: {DRIVE_OUTPUT_DIR / 'latest/bottleneck_mode/benchmark_results_bottleneck.csv'}")

    print("\nPipeline complete! Checkpoints and metrics are safely saved in your Google Drive.")

if __name__ == "__main__":
    main()
