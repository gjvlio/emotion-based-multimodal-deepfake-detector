"""
validate_cmumosei_videos.py
===========================
Validates downloaded CMU-MOSEI videos after each batch.

Checks:
  1. Orphaned partial files (.f*.webm, .f*.mp4 fragments, .part files)
  2. Corrupted/truncated MP4s (ffprobe: no video stream, duration=0, read error)
  3. Zero-byte files

For each bad video ID:
  - Deletes the bad file(s)
  - Removes the ID from archive.txt so yt-dlp will re-download it

Then re-downloads all flagged IDs in one yt-dlp pass.

Usage (run from repo root):
    python scripts/validate_cmumosei_videos.py [--dry-run] [--workers 4]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT   = Path(__file__).resolve().parent.parent
VIDEOS_DIR  = REPO_ROOT / "data/raw/CMU-MOSEI/videos"
ARCHIVE     = VIDEOS_DIR / "archive.txt"
IDS_FILE    = REPO_ROOT / "data/raw/CMU-MOSEI/yt_ids_active.txt"
REPORT_FILE = VIDEOS_DIR / "validation_report.json"

PARTIAL_PATTERN = re.compile(r'\.(f\d+)\.(webm|mp4|m4a|opus)$|\.part$', re.I)


def find_orphans() -> list[Path]:
    return [p for p in VIDEOS_DIR.iterdir() if PARTIAL_PATTERN.search(p.name)]


def extract_video_id(path: Path) -> str:
    name = path.stem
    # strip .fNNN suffix if present (e.g. abc123.f251 -> abc123)
    return re.sub(r'\.f\d+$', '', name)


def probe_mp4(path: Path) -> tuple[bool, str]:
    """Return (ok, reason). Uses ffprobe to check video health."""
    if path.stat().st_size == 0:
        return False, "zero bytes"
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return False, f"ffprobe error: {result.stderr[:200].strip()}"
    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        codec_types = {s.get("codec_type") for s in streams}
        if "video" not in codec_types:
            return False, "no video stream"
        if "audio" not in codec_types:
            return False, "no audio stream"
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        dur = video_streams[0].get("duration")
        if dur is not None and float(dur) < 0.5:
            return False, f"duration too short ({dur}s)"
    except (json.JSONDecodeError, ValueError):
        return False, "ffprobe parse error"
    return True, "ok"


def check_one(mp4: Path) -> tuple[Path, bool, str]:
    try:
        ok, reason = probe_mp4(mp4)
    except subprocess.TimeoutExpired:
        ok, reason = False, "ffprobe timeout"
    except FileNotFoundError:
        print("ERROR: ffprobe not found — install ffmpeg and add to PATH", file=sys.stderr)
        sys.exit(1)
    return mp4, ok, reason


def load_archive() -> list[str]:
    if not ARCHIVE.exists():
        return []
    return ARCHIVE.read_text().splitlines()


def save_archive(lines: list[str]):
    ARCHIVE.write_text("\n".join(lines) + ("\n" if lines else ""))


def remove_from_archive(bad_ids: set[str], dry_run: bool) -> int:
    lines = load_archive()
    kept, removed = [], 0
    for line in lines:
        vid = line.split()[-1] if line.strip() else ""
        if vid in bad_ids:
            removed += 1
            if not dry_run:
                continue
        kept.append(line)
    if not dry_run:
        save_archive(kept)
    return removed


def redownload(bad_ids: set[str], dry_run: bool):
    if not bad_ids:
        return
    print(f"\nRe-downloading {len(bad_ids)} video(s)...")
    urls = [f"https://www.youtube.com/watch?v={vid}" for vid in sorted(bad_ids)]
    batch_file = VIDEOS_DIR / "_redownload_ids.txt"
    cookies_file = REPO_ROOT / "cookies.txt"
    if not dry_run:
        batch_file.write_text("\n".join(urls) + "\n")
        cmd = [
            "yt-dlp",
            "--batch-file",          str(batch_file),
            "--output",              str(VIDEOS_DIR / "%(id)s.%(ext)s"),
            "--format",              "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "--merge-output-format", "mp4",
            "--download-archive",    str(ARCHIVE),
            "--ignore-errors",
            "--no-warnings",
            "--continue",
            "--concurrent-fragments", "4",
            "--retries",             "5",
            "--fragment-retries",    "5",
            "--sleep-interval",      "1",
            "--max-sleep-interval",  "5",
            "--no-write-thumbnail",
            "--progress",
        ]
        if cookies_file.exists():
            cmd += ["--cookies", str(cookies_file)]
            print(f"  Using cookies: {cookies_file}")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        batch_file.unlink(missing_ok=True)
        if result.returncode != 0:
            print("WARNING: yt-dlp exited non-zero — some IDs may still be unavailable")
    else:
        print("  [dry-run] would re-download:", sorted(bad_ids))


def main():
    parser = argparse.ArgumentParser(description="Validate CMU-MOSEI videos and repair bad ones")
    parser.add_argument("--dry-run",  action="store_true", help="Report only, no deletions or re-downloads")
    parser.add_argument("--workers",  type=int, default=4, help="Parallel ffprobe workers")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY-RUN mode — no files will be modified\n")

    bad_ids:    set[str] = set()
    bad_files:  list[Path] = []
    report:     list[dict] = []

    # 1. Orphaned partial files
    print("Scanning for orphaned partial files...")
    orphans = find_orphans()
    for p in orphans:
        vid = extract_video_id(p)
        bad_ids.add(vid)
        bad_files.append(p)
        report.append({"file": str(p.name), "video_id": vid, "issue": "orphaned partial"})
        print(f"  ORPHAN  {p.name}  (id={vid})")

    # 2. Probe all MP4s
    mp4s = sorted(VIDEOS_DIR.glob("*.mp4"))
    print(f"\nProbing {len(mp4s)} MP4 files with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_one, p): p for p in mp4s}
        for fut in tqdm(as_completed(futures), total=len(mp4s), unit="file"):
            path, ok, reason = fut.result()
            if not ok:
                vid = extract_video_id(path)
                bad_ids.add(vid)
                bad_files.append(path)
                report.append({"file": path.name, "video_id": vid, "issue": reason})
                tqdm.write(f"  BAD     {path.name}  ({reason})")

    # Summary
    print(f"\n{'='*50}")
    print(f"Total MP4s checked : {len(mp4s)}")
    print(f"Orphaned partials  : {len(orphans)}")
    print(f"Bad/corrupt MP4s   : {len([r for r in report if 'orphaned' not in r['issue']])}")
    print(f"Unique bad IDs     : {len(bad_ids)}")
    print(f"{'='*50}")

    if not bad_ids:
        print("All videos healthy!")
        REPORT_FILE.write_text(json.dumps({"status": "clean", "checked": len(mp4s)}, indent=2))
        return

    REPORT_FILE.write_text(json.dumps({"bad": report, "bad_ids": sorted(bad_ids)}, indent=2))
    print(f"Report saved: {REPORT_FILE}")

    # 3. Delete bad files
    print(f"\nDeleting {len(bad_files)} bad file(s)...")
    for p in bad_files:
        print(f"  DELETE  {p.name}")
        if not args.dry_run:
            p.unlink(missing_ok=True)

    # 4. Remove from archive
    removed = remove_from_archive(bad_ids, args.dry_run)
    print(f"Removed {removed} archive entries for {len(bad_ids)} IDs")

    # 5. Re-download
    redownload(bad_ids, args.dry_run)

    print("\nValidation + repair complete.")


if __name__ == "__main__":
    main()
