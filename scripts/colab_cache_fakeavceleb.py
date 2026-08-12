"""
colab_cache_fakeavceleb.py — Incremental 10,000-clip FakeAVCeleb feature caching script for Google Colab.
Saves precomputed feature tensors directly to Google Drive after each clip.
Resumable: if Colab disconnects, re-running skips already-cached clips in 0 seconds.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import Config
from scripts.colab_stage1 import search_drive_file, extract_zip
from scripts.evaluate_fakeavceleb import load_clips

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DRIVE_BASE = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")
DRIVE_PREP_DIR = DRIVE_BASE / "preprocessed/features"
DRIVE_Z_AT_DIR = DRIVE_PREP_DIR / "z_at"
DRIVE_Z_V_DIR  = DRIVE_PREP_DIR / "z_v"

LOCAL_PREP_DIR = REPO_ROOT / "data/preprocessed/features"
LOCAL_Z_AT_DIR = LOCAL_PREP_DIR / "z_at"
LOCAL_Z_V_DIR  = LOCAL_PREP_DIR / "z_v"


def main():
    print("\n" + "=" * 60)
    print("  DEEP-SENTINEL COLAB: INCREMENTAL 10,000 FAKEAVCELEB FEATURE CACHER")
    print("=" * 60)

    if not DRIVE_BASE.exists():
        print(f"ERROR: Google Drive not mounted or path not found: {DRIVE_BASE}")
        return

    DRIVE_Z_AT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_Z_V_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_Z_AT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_Z_V_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Locate and extract fakeavceleb.zip if raw videos missing
    fav_meta = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv"
    if not fav_meta.exists():
        print("\nLooking for FakeAVCeleb meta_data.csv in Drive...")
        csv_path = search_drive_file("meta_data.csv")
        if csv_path:
            fav_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, fav_meta)
            print(f"  Successfully copied FakeAVCeleb metadata file to: {fav_meta}")
        else:
            print("  meta_data.csv not found directly. Extracting fakeavceleb.zip...")
            extract_zip("fakeavceleb.zip", REPO_ROOT / "data/raw", optional=True)
            alt_meta = REPO_ROOT / "data/FakeAVCeleb_v1.2/meta_data.csv"
            if alt_meta.exists() and not fav_meta.exists():
                fav_meta.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(alt_meta.parent), str(fav_meta.parent))

    # 2. Sample up to 10,000 clips (500 Real + 9,500 Fake)
    print("\nSampling 10,000 FakeAVCeleb clips for incremental caching...")
    clips = load_clips(n_real=500, n_fake=9500, seed=42, hard=True)
    print(f"  Total sampled target clips: {len(clips):,}")

    # 3. Initialize preprocessing pipeline
    cfg = Config.from_yaml(REPO_ROOT / "configs/config.yaml")
    pipeline = PreprocessingPipeline(
        cache_dir     = REPO_ROOT / "data/preprocessed",
        wav2vec_model = cfg.model.wav2vec_model,
        bert_model    = cfg.model.bert_model,
        whisper_model = cfg.model.whisper_model,
        vit_model     = cfg.model.vit_model,
        face_detector = cfg.preprocessing.face_detector,
        n_keyframes   = cfg.preprocessing.n_keyframes,
        frame_size    = cfg.preprocessing.frame_size,
        max_audio_sec = cfg.preprocessing.max_audio_seconds,
        device        = "cuda" if torch.cuda.is_available() else "cpu",
    )

    # 4. Incremental Caching Loop
    cached_count = 0
    newly_cached = 0
    failed_count = 0

    pbar = tqdm(clips, desc="Caching 10k FakeAVCeleb", unit="clip", dynamic_ncols=True)
    for c in pbar:
        clip_id = c["clip_id"]
        v_path  = c["video_path"]

        drive_at = DRIVE_Z_AT_DIR / f"{clip_id}.pt"
        drive_v  = DRIVE_Z_V_DIR  / f"{clip_id}.pt"
        local_at = LOCAL_Z_AT_DIR / f"{clip_id}.pt"
        local_v  = LOCAL_Z_V_DIR  / f"{clip_id}.pt"

        # Check if already in Drive or local SSD
        if drive_at.exists() and drive_v.exists():
            if not (local_at.exists() and local_v.exists()):
                try:
                    shutil.copy2(drive_at, local_at)
                    shutil.copy2(drive_v, local_v)
                except Exception:
                    pass
            cached_count += 1
            pbar.set_postfix(cached=cached_count, new=newly_cached, fail=failed_count)
            continue

        if local_at.exists() and local_v.exists():
            try:
                shutil.copy2(local_at, drive_at)
                shutil.copy2(local_v, drive_v)
            except Exception:
                pass
            cached_count += 1
            pbar.set_postfix(cached=cached_count, new=newly_cached, fail=failed_count)
            continue

        # Extract features using GPU pipeline
        feats = pipeline.process(clip_id, v_path, force=False)
        if feats is None:
            failed_count += 1
            pbar.set_postfix(cached=cached_count, new=newly_cached, fail=failed_count)
            continue

        # Save immediately to local SSD and sync to Google Drive for safety
        try:
            torch.save(feats.z_at.cpu(), local_at)
            torch.save(feats.z_v.cpu(),  local_v)
            shutil.copy2(local_at, drive_at)
            shutil.copy2(local_v,  drive_v)
            newly_cached += 1
            cached_count += 1
        except Exception as e:
            log.warning(f"Failed to sync {clip_id} to Drive: {e}")

        pbar.set_postfix(cached=cached_count, new=newly_cached, fail=failed_count)

    print("\n" + "=" * 60)
    print(f"  CACHING COMPLETE!")
    print(f"  Total Cached  : {cached_count:,} clips")
    print(f"  Newly Extracted: {newly_cached:,} clips")
    print(f"  Failed        : {failed_count:,} clips")
    print(f"  All features are safely stored on Google Drive: {DRIVE_PREP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
