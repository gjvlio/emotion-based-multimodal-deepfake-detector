"""
segment_cmumosei.py — Cut CMU-MOSEI YouTube videos into utterance segments
using timestamps from CMU_MOSEI_Labels.csd.

Usage:
    python scripts/segment_cmumosei.py [--min_dur 2.0] [--max_dur 8.0] [--workers 4]

Reads:
    data/raw/CMU-MOSEI/videos/<video_id>.mp4  (downloaded by download_cmumosei_videos.sh)
    data/raw/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd

Writes:
    data/raw/CMU-MOSEI/segments/<video_id>_<seg_idx>.mp4
    data/raw/CMU-MOSEI/segments/manifest.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = REPO_ROOT / "data/raw/CMU-MOSEI/videos"
CSD_LABELS = REPO_ROOT / "data/raw/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd"
SEGMENTS_DIR = REPO_ROOT / "data/raw/CMU-MOSEI/segments"


def ffmpeg_cut(src: Path, start: float, end: float, dst: Path) -> bool:
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def load_segments(min_dur: float, max_dur: float):
    """Return list of (video_id, seg_idx, start, end) within duration range."""
    segments = []
    with h5py.File(str(CSD_LABELS), "r") as f:
        root = list(f.keys())[0]
        data = f[root]["data"]
        for vid in data.keys():
            intervals = data[vid]["intervals"][:]
            for i, (start, end) in enumerate(intervals):
                dur = end - start
                if min_dur <= dur <= max_dur:
                    segments.append((vid, i, float(start), float(end)))
    return segments


def process_segment(args):
    vid, seg_idx, start, end, video_path, out_path = args
    if out_path.exists():
        return "skip"
    if not video_path.exists():
        return "no_video"
    ok = ffmpeg_cut(video_path, start, end, out_path)
    return "ok" if ok else "fail"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_dur", type=float, default=2.0)
    parser.add_argument("--max_dur", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading segments from CSD ({args.min_dur}–{args.max_dur}s)...")
    segments = load_segments(args.min_dur, args.max_dur)
    print(f"Eligible segments: {len(segments)}")

    available_videos = set(p.stem for p in VIDEOS_DIR.glob("*.mp4"))
    print(f"Downloaded videos: {len(available_videos)}")

    tasks = []
    for vid, seg_idx, start, end in segments:
        video_path = VIDEOS_DIR / f"{vid}.mp4"
        out_path = SEGMENTS_DIR / f"{vid}_{seg_idx:03d}.mp4"
        tasks.append((vid, seg_idx, start, end, video_path, out_path))

    counts = {"ok": 0, "skip": 0, "no_video": 0, "fail": 0}
    manifest_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_segment, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), unit="seg"):
            t = futures[fut]
            vid, seg_idx, start, end, _, out_path = t
            status = fut.result()
            counts[status] += 1
            if status in ("ok", "skip"):
                manifest_rows.append({
                    "video_id": vid,
                    "seg_idx": seg_idx,
                    "start": f"{start:.3f}",
                    "end": f"{end:.3f}",
                    "duration": f"{end-start:.3f}",
                    "path": str(out_path.relative_to(REPO_ROOT)),
                })

    manifest_path = SEGMENTS_DIR / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"\nDone.")
    print(f"  ok:       {counts['ok']}")
    print(f"  skip:     {counts['skip']}")
    print(f"  no_video: {counts['no_video']}")
    print(f"  fail:     {counts['fail']}")
    print(f"Manifest: {manifest_path} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
