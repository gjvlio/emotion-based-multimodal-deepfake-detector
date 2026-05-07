"""
preprocess_all.py — Run preprocessing pipeline on all clips in the dataset.

Reads metadata CSVs for all 4 tracks + real MELD + real CMU-MOSEI,
then runs PreprocessingPipeline.process() on each unprocessed clip.
Progress is implicitly tracked by cached .pt files — safe to interrupt/resume.

Usage (from repo root):
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --device cuda
    python scripts/preprocess_all.py --workers 4 --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _collect_clips(cfg: Config) -> list[tuple[str, str]]:
    """Return list of (clip_id, video_path) from all sources."""
    clips = []

    # Fake tracks 1–4
    for csv_path, id_col, path_col in [
        (cfg.paths.track1_meta,  "output_stem",  "output_path"),
        (cfg.paths.track2_meta,  "output_stem",  "output_path"),
        (cfg.paths.track3_meta,  "output_stem",  "output_path"),
        (cfg.paths.track4_meta,  "output_stem",  "output_path"),
    ]:
        p = Path(csv_path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            clips.append((str(row[id_col]), str(row[path_col])))

    # Real — MELD
    meld = Path(cfg.paths.meld_real_csv)
    if meld.exists():
        df = pd.read_csv(meld)
        for _, row in df.iterrows():
            clips.append((str(row["clip_id"]), str(row["video_path"])))

    # Real — CMU-MOSEI
    mosei = Path(cfg.paths.mosei_real_csv)
    if mosei.exists():
        df = pd.read_csv(mosei)
        for _, row in df.iterrows():
            clips.append((str(row["clip_id"]), str(row["video_path"])))

    return clips


def main():
    parser = argparse.ArgumentParser(description="Preprocess all dataset clips → Z_at, Z_v")
    parser.add_argument("--config",   default=None, help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--device",   default="cpu", help="Computation device (cpu / cuda)")
    parser.add_argument("--workers",  type=int, default=1, help="Parallel workers (default 1)")
    parser.add_argument("--force",    action="store_true", help="Reprocess even if cache exists")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    pipeline = PreprocessingPipeline(
        cache_dir       = cfg.paths.preprocessed_dir,
        wav2vec_model   = cfg.model.wav2vec_model,
        bert_model      = cfg.model.bert_model,
        whisper_model   = cfg.model.whisper_model,
        vit_model       = cfg.model.vit_model,
        face_detector   = cfg.preprocessing.face_detector,
        n_keyframes     = cfg.preprocessing.n_keyframes,
        frame_size      = cfg.preprocessing.frame_size,
        max_audio_sec   = cfg.preprocessing.max_audio_seconds,
        device          = args.device,
    )

    clips = _collect_clips(cfg)
    log.info(f"Total clips to process: {len(clips)}")

    already_done = sum(1 for cid, _ in clips if pipeline.is_cached(cid))
    log.info(f"Already cached: {already_done}")
    to_do = [(cid, vp) for cid, vp in clips if not pipeline.is_cached(cid) or args.force]
    log.info(f"Remaining: {len(to_do)}")

    if not to_do:
        log.info("All clips already preprocessed.")
        return

    failed = []

    def process_one(item):
        cid, vp = item
        result = pipeline.process(cid, vp, force=args.force)
        return cid, result is not None

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, item): item for item in to_do}
            for fut in tqdm(as_completed(futures), total=len(to_do), desc="Preprocessing"):
                cid, ok = fut.result()
                if not ok:
                    failed.append(cid)
    else:
        for item in tqdm(to_do, desc="Preprocessing"):
            cid, ok = process_one(item)
            if not ok:
                failed.append(cid)

    print(f"\nDone. Failed: {len(failed)}/{len(to_do)}")
    if failed:
        fail_file = Path(cfg.paths.preprocessed_dir) / "failed_clips.txt"
        fail_file.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed clip IDs written to {fail_file}")


if __name__ == "__main__":
    main()
