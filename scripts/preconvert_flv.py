"""
preconvert_flv.py
=================
Pre-convert CREMA-D FLV files to VideoMP4/ for actors whose FLV→MP4 conversion
fails at generation time (currently actors 1061, 1062).

The generators check VideoMP4/ before VideoFlash/, so pre-converted MP4s bypass
the per-clip FLV conversion step entirely.

Usage (run from repo root):
    python scripts/preconvert_flv.py --cremad_dir data/raw/CREMA-D
    python scripts/preconvert_flv.py --cremad_dir data/raw/CREMA-D --actors 1061 1062
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def convert_flv_to_mp4(flv_path: Path, mp4_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(flv_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            str(mp4_path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  FAILED {flv_path.name}: {result.stderr[:200]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-convert CREMA-D FLVs to VideoMP4/ for specified actors"
    )
    parser.add_argument("--cremad_dir", required=True, help="CREMA-D root directory")
    parser.add_argument(
        "--actors", type=int, nargs="+", default=[1061, 1062],
        help="Actor IDs to convert (default: 1061 1062)"
    )
    args = parser.parse_args()

    cremad = Path(args.cremad_dir)
    flv_dir = cremad / "VideoFlash"
    mp4_dir = cremad / "VideoMP4"
    mp4_dir.mkdir(exist_ok=True)

    flvs = []
    for actor in args.actors:
        flvs.extend(sorted(flv_dir.glob(f"{actor}_*.flv")))

    print(f"Converting {len(flvs)} FLVs for actors {args.actors}")

    done, failed = 0, 0
    for flv in tqdm(flvs, desc="FLV→MP4"):
        mp4 = mp4_dir / (flv.stem + ".mp4")
        if mp4.exists():
            done += 1
            continue
        if convert_flv_to_mp4(flv, mp4):
            done += 1
        else:
            failed += 1

    print(f"\nConverted: {done}  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
