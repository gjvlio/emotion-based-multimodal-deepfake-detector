"""
validate_generation.py
======================
Health check for Track 1 and Track 2 deepfake generation outputs.

Checks:
  - metadata.csv completeness vs checkpoint
  - Every listed output file exists on disk
  - ffprobe: each file has valid audio (and video for track2) streams
  - Duration > 0
  - Orphan files (on disk but not in metadata)
  - Emotion label distribution

Usage:
    python scripts/validate_generation.py
    python scripts/validate_generation.py --tracks 1
    python scripts/validate_generation.py --tracks 1 2
    python scripts/validate_generation.py --tracks 2 --out_dir data/synthetic/track2_fakes
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ── ffprobe helpers ────────────────────────────────────────────────────────────

def ffprobe_info(path: str) -> dict:
    """Return dict with 'duration', 'has_video', 'has_audio'. Empty dict on failure."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams', '-show_format',
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {}
        import json as _json
        data = _json.loads(result.stdout)
        streams = data.get('streams', [])
        fmt     = data.get('format', {})
        return {
            'duration':  float(fmt.get('duration', 0)),
            'has_video': any(s.get('codec_type') == 'video' for s in streams),
            'has_audio': any(s.get('codec_type') == 'audio' for s in streams),
        }
    except Exception:
        return {}


# ── Per-track validation ───────────────────────────────────────────────────────

def validate_track(track_num: int, out_dir: Path, expect_video: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"TRACK {track_num} VALIDATION  [{out_dir}]")
    print(f"{'='*60}")

    meta_csv    = out_dir / 'metadata.csv'
    failed_csv  = out_dir / 'failed.csv'
    chk_name    = f'progress_styletts.json' if track_num == 1 else f'progress_track{track_num}.json'
    checkpoint  = out_dir / chk_name

    if not meta_csv.exists():
        print(f"  ERROR: metadata.csv not found at {meta_csv}")
        return False

    meta = pd.read_csv(meta_csv)
    print(f"  metadata.csv rows   : {len(meta)}")

    if checkpoint.exists():
        chk = json.loads(checkpoint.read_text())
        print(f"  checkpoint count    : {chk['count']} (updated {chk.get('updated_at','')})")
        if chk['count'] != len(meta):
            print(f"  WARNING: checkpoint({chk['count']}) != metadata({len(meta)}) — {abs(chk['count']-len(meta))} mismatch")
    else:
        print(f"  WARNING: checkpoint not found at {checkpoint}")

    if failed_csv.exists():
        failed = pd.read_csv(failed_csv)
        print(f"  failed.csv rows     : {len(failed)}")
    else:
        print(f"  failed.csv          : not found (0 failures recorded)")

    # file existence check
    print(f"\n  [File existence]")
    missing = [row['output_path'] for _, row in meta.iterrows()
               if not Path(row['output_path']).exists()]
    print(f"    present : {len(meta) - len(missing)}/{len(meta)}")
    if missing:
        print(f"    MISSING : {len(missing)}")
        for p in missing[:5]:
            print(f"      {p}")
        if len(missing) > 5:
            print(f"      ... and {len(missing)-5} more")

    # ffprobe integrity check (sample 200 or all if <= 200)
    print(f"\n  [ffprobe integrity — sampling up to 200 files]")
    sample = meta.sample(min(200, len(meta)), random_state=42)
    bad_missing_stream = []
    bad_zero_duration  = []
    bad_ffprobe_fail   = []

    for _, row in sample.iterrows():
        p = row['output_path']
        if not Path(p).exists():
            continue
        info = ffprobe_info(p)
        if not info:
            bad_ffprobe_fail.append(p)
            continue
        if info['duration'] <= 0:
            bad_zero_duration.append(p)
        if expect_video and not info['has_video']:
            bad_missing_stream.append(p)
        if not info['has_audio']:
            bad_missing_stream.append(p)

    n_checked = len(sample) - len(missing[:len(sample)])
    print(f"    checked   : {n_checked}")
    print(f"    ffprobe ok: {n_checked - len(bad_ffprobe_fail) - len(bad_zero_duration) - len(bad_missing_stream)}")
    if bad_ffprobe_fail:
        print(f"    ffprobe fail (corrupt/unreadable): {len(bad_ffprobe_fail)}")
        for p in bad_ffprobe_fail[:3]:
            print(f"      {p}")
    if bad_zero_duration:
        print(f"    zero-duration files: {len(bad_zero_duration)}")
    if bad_missing_stream:
        print(f"    missing expected streams: {len(bad_missing_stream)}")

    # orphan check
    vid_dir = out_dir / 'videos'
    if vid_dir.exists():
        suffix = '_wav2lip.mp4' if track_num == 2 else '_sadtalker.mp4' if track_num == 3 else '.wav'
        disk_files = set(vid_dir.glob('*.mp4')) | set(vid_dir.glob('*.wav'))
        meta_paths = set(Path(p) for p in meta['output_path'])
        orphans = disk_files - meta_paths
        print(f"\n  [Orphan files in {vid_dir.name}/]")
        print(f"    on disk       : {len(disk_files)}")
        print(f"    in metadata   : {len(meta_paths)}")
        print(f"    orphans       : {len(orphans)}")

    # emotion distribution
    if 'audio_emotion' in meta.columns:
        dist = meta['audio_emotion'].value_counts()
        print(f"\n  [Emotion distribution (audio_emotion)]")
        for emo, cnt in dist.items():
            print(f"    {emo:4s}: {cnt}")

    ok = not missing and not bad_ffprobe_fail and not bad_zero_duration and not bad_missing_stream
    print(f"\n  RESULT: {'PASS' if ok else 'ISSUES FOUND'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Health check for Track 1 and Track 2 generation outputs"
    )
    parser.add_argument(
        '--tracks', nargs='+', type=int, default=[1, 2],
        help='Tracks to validate (default: 1 2)',
    )
    parser.add_argument(
        '--base_dir', default='data/synthetic',
        help='Base directory for synthetic outputs',
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    track_cfg = {
        1: (base / 'track1_fakes', False),
        2: (base / 'track2_fakes', True),
        3: (base / 'track3_fakes', True),
    }

    all_ok = True
    for t in args.tracks:
        if t not in track_cfg:
            print(f"Unknown track {t}, skipping.")
            continue
        out_dir, expect_video = track_cfg[t]
        ok = validate_track(t, out_dir, expect_video)
        all_ok = all_ok and ok

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'ISSUES FOUND — see above'}")
    print(f"{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
