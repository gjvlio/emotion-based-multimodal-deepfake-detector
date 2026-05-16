"""
generate_track4_musetalk.py

Generates Track 4 fakes using MuseTalk cross-speaker lip sync.
Reads meld_mismatch_pairs.csv (3,482 emotion-mismatched pairs).
Runs MuseTalk in batches — models load once per batch.
Supports resume: skips pairs whose output .mp4 already exists.
Ctrl+C safe: partial batch outputs are rescued before exit.

Usage:
    python scripts/generate_track4_musetalk.py
    python scripts/generate_track4_musetalk.py --batch_size 200 --start_idx 0
    python scripts/generate_track4_musetalk.py --max_pairs 2   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUSETALK_DIR  = PROJECT_ROOT / "tools" / "MuseTalk"
PAIRS_CSV     = PROJECT_ROOT / "data" / "processed" / "meld_manifests" / "meld_mismatch_pairs.csv"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "synthetic" / "track4_fakes"
META_OUT      = OUTPUT_DIR / "metadata.csv"
TMP_DIR       = OUTPUT_DIR / "_tmp"

UNET_MODEL   = MUSETALK_DIR / "models" / "musetalk" / "pytorch_model.bin"
UNET_CONFIG  = MUSETALK_DIR / "models" / "musetalk" / "musetalk.json"
WHISPER_DIR  = MUSETALK_DIR / "models" / "whisper"
FFMPEG_BIN   = Path("C:/ffmpeg/bin")

META_FIELDS = [
    "output_path", "video_clip", "audio_clip",
    "video_emotion", "audio_emotion",
    "video_speaker", "audio_speaker",
    "split", "label", "method", "output_stem",
]

_interrupted = False


def _handle_sigint(sig, frame):
    global _interrupted
    print("\n[!] Interrupt received — finishing current batch then exiting cleanly.")
    _interrupted = True


def load_pairs() -> list[dict]:
    with open(PAIRS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def output_path(stem: str) -> Path:
    return OUTPUT_DIR / f"{stem}.mp4"


def already_done(stem: str) -> bool:
    p = output_path(stem)
    return p.exists() and p.stat().st_size > 0


def resolve_path(rel: str) -> Path:
    return PROJECT_ROOT / rel.replace("\\", "/")


def duration_ok(row: dict, max_ratio: float = 1.5) -> bool:
    """Skip pairs where durations differ by more than max_ratio (avoids frozen frames)."""
    try:
        vd = float(row["video_duration"])
        ad = float(row["audio_duration"])
        if vd <= 0 or ad <= 0:
            return False
        ratio = max(vd, ad) / min(vd, ad)
        return ratio <= max_ratio
    except (KeyError, ValueError, ZeroDivisionError):
        return True


def rescue_partial(result_dir: Path, batch: list[dict]) -> list[dict]:
    """Move any .mp4 files produced in result_dir/v1/ to OUTPUT_DIR, return matching rows."""
    musetalk_out = result_dir / "v1"
    if not musetalk_out.exists():
        return []

    stem_to_row = {row["output_stem"]: row for row in batch}
    rescued: list[dict] = []

    for mp4 in musetalk_out.glob("*.mp4"):
        stem = mp4.stem
        dst  = output_path(stem)
        if not dst.exists():
            mp4.rename(dst)
        if stem in stem_to_row and not rescued.__contains__(stem_to_row[stem]):
            rescued.append(stem_to_row[stem])

    return rescued


def run_musetalk_batch(batch: list[dict], batch_idx: int) -> list[dict]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    config: dict = {}
    for i, row in enumerate(batch):
        stem     = row["output_stem"]
        vid_path = resolve_path(row["video_clip"])
        aud_path = resolve_path(row["audio_clip"])

        if not vid_path.exists():
            print(f"  SKIP (video missing): {vid_path.name}")
            continue
        if not aud_path.exists():
            print(f"  SKIP (audio missing): {aud_path.name}")
            continue

        config[f"task_{i}"] = {
            "video_path": str(vid_path),
            "audio_path": str(aud_path),
            "result_name": f"{stem}.mp4",
        }

    if not config:
        return []

    yaml_path = TMP_DIR / f"batch_{batch_idx:04d}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    result_dir = TMP_DIR / f"batch_{batch_idx:04d}"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", str(yaml_path),
        "--result_dir",       str(result_dir),
        "--unet_model_path",  str(UNET_MODEL),
        "--unet_config",      str(UNET_CONFIG),
        "--whisper_dir",      str(WHISPER_DIR),
        "--vae_type",         "sd-vae-ft-mse",
        "--ffmpeg_path",      str(FFMPEG_BIN),
        "--version",          "v1",
        "--use_float16",
        "--fps",              "25",
        "--gpu_id",           "0",
    ]

    print(f"  Running MuseTalk on {len(config)} pairs...")
    subprocess.run(cmd, cwd=str(MUSETALK_DIR), check=False)

    # Rescue all outputs produced — catches partial batches on interrupt
    completed = rescue_partial(result_dir, batch)

    missing = len(config) - len(completed)
    if missing:
        done_stems = {r["output_stem"] for r in completed}
        for row in batch:
            if row["output_stem"] not in done_stems and row["output_stem"] in {
                v["result_name"].removesuffix(".mp4") for v in config.values()
            }:
                print(f"  WARN: output not produced for {row['output_stem']}")

    return completed


def append_metadata(rows: list[dict]) -> None:
    write_header = not META_OUT.exists()
    with open(META_OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            stem = row["output_stem"]
            writer.writerow({
                "output_path":   str(output_path(stem)),
                "video_clip":    row["video_clip"],
                "audio_clip":    row["audio_clip"],
                "video_emotion": row["video_emotion"],
                "audio_emotion": row["audio_emotion"],
                "video_speaker": row["video_speaker"],
                "audio_speaker": row["audio_speaker"],
                "split":         row["split"],
                "label":         1,
                "method":        "musetalk_cross_speaker",
                "output_stem":   stem,
            })


def main() -> None:
    signal.signal(signal.SIGINT,  _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--start_idx",  type=int, default=0,
                        help="Skip first N pairs (for manual resume)")
    parser.add_argument("--max_pairs",  type=int, default=None,
                        help="Process at most N pairs (for smoke testing)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not UNET_MODEL.exists():
        print(f"ERROR: MuseTalk model not found at {UNET_MODEL}")
        sys.exit(1)

    all_pairs = load_pairs()
    pending   = [r for r in all_pairs[args.start_idx:]
                 if not already_done(r["output_stem"]) and duration_ok(r)]

    skipped_duration = sum(1 for r in all_pairs if not duration_ok(r))

    total = len(all_pairs)
    done  = sum(1 for r in all_pairs if already_done(r["output_stem"]))
    print(f"Pairs total={total}  already_done={done}  skipped_duration={skipped_duration}  pending={len(pending)}")

    if not pending:
        print("All pairs already generated.")
        return

    if args.max_pairs is not None:
        pending = pending[:args.max_pairs]
        print(f"max_pairs={args.max_pairs} — processing only first {len(pending)}")

    total_completed = 0
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        if _interrupted:
            break

        batch = pending[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
        print(f"\nBatch {batch_idx + 1}/{n_batches}  ({len(batch)} pairs)")

        completed = run_musetalk_batch(batch, batch_idx)
        append_metadata(completed)
        total_completed += len(completed)
        print(f"  Batch done: {len(completed)}/{len(batch)} succeeded | "
              f"Total: {total_completed}/{len(pending)}")

    status = "interrupted" if _interrupted else "finished"
    done_now = sum(1 for r in all_pairs if already_done(r["output_stem"]))
    print(f"\nTrack 4 generation {status}. {total_completed} clips produced this run.")
    print(f"Total done across all runs: {done_now}/{total}")
    print(f"Metadata written to {META_OUT}")
    if _interrupted:
        print("Resume with: python scripts/generate_track4_musetalk.py")
        print("(already-done clips are skipped automatically)")


if __name__ == "__main__":
    main()
