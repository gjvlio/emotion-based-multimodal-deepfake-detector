"""
migrate_stems.py
================
One-off migration: rename FAKE_T1_ prefix to FAKE_T2_ / FAKE_T3_ in all
Track 2 and Track 3 artefacts (video files, WAV caches, CSVs, JSON checkpoints).

Run from repo root:
    python scripts/migrate_stems.py

Safe to re-run: already-renamed files are skipped.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def rename_files(directory: Path, old_prefix: str, new_prefix: str) -> int:
    if not directory.exists():
        return 0
    count = 0
    for f in list(directory.iterdir()):
        if f.name.startswith(old_prefix):
            new_name = new_prefix + f.name[len(old_prefix):]
            f.rename(f.parent / new_name)
            count += 1
    return count


def patch_csv_stems(csv_path: Path, old_prefix: str, new_prefix: str,
                    cols: list[str]) -> int:
    if not csv_path.exists():
        print(f"  SKIP (not found): {csv_path}")
        return 0
    df = pd.read_csv(csv_path)
    changed = 0
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].str.startswith(old_prefix, na=False)
        if mask.any():
            df.loc[mask, col] = df.loc[mask, col].str.replace(
                old_prefix, new_prefix, regex=False
            )
            changed += mask.sum()
    df.to_csv(csv_path, index=False)
    return changed


def patch_json_checkpoint(json_path: Path, old_prefix: str, new_prefix: str) -> int:
    if not json_path.exists():
        print(f"  SKIP (not found): {json_path}")
        return 0
    with open(json_path) as f:
        data = json.load(f)
    old_completed = data.get("completed", [])
    new_completed = [
        (new_prefix + s[len(old_prefix):] if s.startswith(old_prefix) else s)
        for s in old_completed
    ]
    changed = sum(1 for a, b in zip(old_completed, new_completed) if a != b)
    data["completed"] = new_completed
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    return changed


def migrate_track(track_num: int):
    old_prefix = "FAKE_T1_"
    new_prefix = f"FAKE_T{track_num}_"
    suffix = "wav2lip" if track_num == 2 else "sadtalker"

    print(f"\n{'='*55}")
    print(f"Track {track_num}: {old_prefix} -> {new_prefix}")
    print(f"{'='*55}")

    # Video files
    videos_dir = BASE / f"data/synthetic/track{track_num}_fakes/videos"
    n = rename_files(videos_dir, old_prefix, new_prefix)
    print(f"  Renamed {n} video files in {videos_dir.name}/")

    # WAV cache files
    wav_dir = BASE / f"data/synthetic/track{track_num}_fakes/wav_tmp"
    n = rename_files(wav_dir, old_prefix, new_prefix)
    print(f"  Renamed {n} WAV cache files in wav_tmp/")

    # Pairs CSV (manifest)
    pairs_csv = BASE / f"data/processed/track1_manifests/track{track_num}_pairs.csv"
    n = patch_csv_stems(pairs_csv, old_prefix, new_prefix, ["output_stem"])
    print(f"  Patched {n} stems in track{track_num}_pairs.csv")

    # Retry CSV
    retry_csv = BASE / f"data/processed/track1_manifests/track{track_num}_retry.csv"
    n = patch_csv_stems(retry_csv, old_prefix, new_prefix, ["output_stem"])
    print(f"  Patched {n} stems in track{track_num}_retry.csv")

    # metadata.csv — both output_stem and output_path
    meta_csv = BASE / f"data/synthetic/track{track_num}_fakes/metadata.csv"
    n = patch_csv_stems(meta_csv, old_prefix, new_prefix, ["output_stem", "output_path"])
    print(f"  Patched {n} cells in metadata.csv")

    # failed.csv — output_stem
    failed_csv = BASE / f"data/synthetic/track{track_num}_fakes/failed.csv"
    n = patch_csv_stems(failed_csv, old_prefix, new_prefix, ["output_stem"])
    print(f"  Patched {n} stems in failed.csv")

    # Progress JSON checkpoint
    json_path = BASE / f"data/synthetic/track{track_num}_fakes/progress_track{track_num}.json"
    n = patch_json_checkpoint(json_path, old_prefix, new_prefix)
    print(f"  Patched {n} stems in progress_track{track_num}.json")


def main():
    print("Stem migration: FAKE_T1_ -> FAKE_T2_ / FAKE_T3_")
    migrate_track(2)
    migrate_track(3)
    print("\nDone.")


if __name__ == "__main__":
    main()
