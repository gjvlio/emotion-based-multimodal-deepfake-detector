"""
Track 3 — Emotion Talking Head Generation (SadTalker)

Generates the most convincing class of deepfake in the pipeline:
both audio AND video are fully synthesised with the target emotion.
SadTalker generates a complete new face video from a single portrait image
driven by audio — identity and lip sync both synthesised, not reanimated.

Difference from previous tracks:
  Track 1 — synthetic audio only, original face video, lip mismatch visible
  Track 2 — synthetic audio, original face, lips reanimated (Wav2Lip)
  Track 3 — synthetic audio + fully generated face video (SadTalker),
             driven by audio; face is generated from actor portrait, not
             reanimated from original footage

Pipeline per clip:
  1. Extract synthesised audio from Track 1 _styletts.mp4
     (reuses Track 1 Method B audio; avoids re-running StyleTTS2/RVC)
  2. Load actor's best portrait frame as reference image
  3. Run SadTalker inference: portrait + audio -> full face video
     with lip sync and 3D head motion
  4. Write output to data/synthetic/track3_fakes/videos/

Usage:
  python src/track3/track3_generate.py \
    --track1_dir    data/synthetic/track1_fakes \
    --portraits_dir data/processed/actor_portraits \
    --sadtalker_dir tools/SadTalker \
    --out_dir       data/synthetic/track3_fakes \
    [--size         256]      # output resolution: 256 or 512 (default: 256)
    [--still]                 # minimal head motion (default: on)
    [--enhancer     gfpgan]   # face enhancer: gfpgan or none (default: none)
    [--resume]
"""

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CHECKPOINT_FILE = "progress_track3.json"
METADATA_FILE   = "metadata.csv"
FAILED_FILE     = "failed.csv"
METADATA_COLS   = [
    "output_stem", "track1_source", "portrait_used",
    "face_emotion", "audio_emotion", "actor_id", "sentence",
    "sadtalker_size", "timestamp",
]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(out_dir: Path) -> set:
    p = out_dir / CHECKPOINT_FILE
    return set(json.load(open(p))) if p.exists() else set()


def save_checkpoint(out_dir: Path, done: set):
    with open(out_dir / CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(done), f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         str(out_wav)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def parse_track1_stem(stem: str) -> dict | None:
    stem = stem.replace("_styletts", "")
    if not stem.startswith("FAKE_T1_"):
        return None
    try:
        parts = stem[len("FAKE_T1_"):].split("__AUDIO_")
        fp = parts[0].split("_")
        ap = parts[1].split("_")
        return {
            "actor_id":        fp[0],
            "sentence":        fp[1],
            "face_emotion":    fp[2],
            "face_intensity":  fp[3],
            "audio_emotion":   ap[2],
            "audio_intensity": ap[3],
        }
    except (IndexError, ValueError):
        return None


def find_portrait(portraits_dir: Path, actor_id: str) -> Path | None:
    p = portraits_dir / f"actor_{actor_id}" / "portrait.png"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# SadTalker inference
# ---------------------------------------------------------------------------

def run_sadtalker(
    sadtalker_dir: Path,
    portrait:      Path,
    audio_wav:     Path,
    out_path:      Path,
    size:          int  = 256,
    still:         bool = True,
    enhancer:      str  = "none",
) -> bool:
    """
    Call SadTalker's inference.py to generate a talking head video.

    SadTalker takes a single portrait image + driving audio and generates
    a full face video with 3D head motion and lip sync. Unlike Wav2Lip
    (Track 2) which reanimates an existing face video, SadTalker generates
    a completely new face sequence from the portrait.

    --still:   keeps head mostly stable, focuses motion on face/lips
    --preprocess full: processes the entire image (not just the face crop)
    --size:    output resolution (256 = faster, 512 = higher quality)
    """
    infer_script = sadtalker_dir / "inference.py"
    if not infer_script.exists():
        raise FileNotFoundError(
            f"SadTalker inference script not found at {infer_script}. "
            "Clone SadTalker to tools/SadTalker — see tools/README.md."
        )

    # SadTalker writes to result_dir with auto-generated filename
    tmp_result = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_result.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", str(audio_wav.resolve()),
        "--source_image", str(portrait.resolve()),
        "--result_dir",   str(tmp_result.resolve()),
        "--preprocess",   "full",
        "--size",         str(size),
    ]
    if still:
        cmd.append("--still")
    if enhancer and enhancer.lower() != "none":
        cmd += ["--enhancer", enhancer]

    result = subprocess.run(
        cmd,
        cwd=str(sadtalker_dir),
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        log.error(f"SadTalker inference failed:\n{result.stderr[-2000:]}")
        shutil.rmtree(str(tmp_result), ignore_errors=True)
        return False

    # Locate generated MP4 and move to final path
    mp4s = list(tmp_result.rglob("*.mp4"))
    if not mp4s:
        log.error("SadTalker returned success but no MP4 found in output dir.")
        shutil.rmtree(str(tmp_result), ignore_errors=True)
        return False

    shutil.move(str(mp4s[0]), str(out_path))
    shutil.rmtree(str(tmp_result), ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(args):
    track1_videos = Path(args.track1_dir) / "videos"
    portraits_dir = Path(args.portraits_dir)
    sadtalker_dir = Path(args.sadtalker_dir)
    out_dir       = Path(args.out_dir)
    vid_dir       = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    styletts_files = sorted(track1_videos.glob("*_styletts.mp4"))
    if not styletts_files:
        log.error(f"No _styletts.mp4 files in {track1_videos}. Run Track 1 Method B first.")
        sys.exit(1)
    log.info(f"Found {len(styletts_files)} Track 1 Method B clips.")

    done = load_checkpoint(out_dir) if args.resume else set()
    if done:
        log.info(f"Resuming -- {len(done)} clips already done.")

    meta_path   = out_dir / METADATA_FILE
    failed_path = out_dir / FAILED_FILE
    meta_exists = meta_path.exists()
    meta_f   = open(meta_path,   "a", newline="", encoding="utf-8")
    failed_f = open(failed_path, "a", newline="", encoding="utf-8")
    meta_w   = csv.DictWriter(meta_f,   fieldnames=METADATA_COLS)
    fail_w   = csv.DictWriter(failed_f, fieldnames=["output_stem", "error", "timestamp"])
    if not meta_exists:
        meta_w.writeheader()
        fail_w.writeheader()

    total     = len(styletts_files)
    succeeded = 0
    failed    = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, t1_path in enumerate(styletts_files, 1):
            stem     = t1_path.stem
            out_stem = stem.replace("_styletts", "_sadtalker")
            out_path = vid_dir / f"{out_stem}.mp4"

            if out_stem in done:
                continue

            log.info(f"[{i}/{total}] {stem}")

            info = parse_track1_stem(stem)
            if info is None:
                log.warning("  Could not parse stem, skipping.")
                continue

            actor_id = info["actor_id"]

            portrait = find_portrait(portraits_dir, actor_id)
            if portrait is None:
                msg = f"Portrait not found for actor {actor_id}. Run extract_actor_frames.py first."
                log.error(f"  {msg}")
                fail_w.writerow({"output_stem": out_stem, "error": msg,
                                 "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            tmp_wav = Path(tmpdir) / f"{out_stem}.wav"
            if not extract_audio(t1_path, tmp_wav):
                msg = "ffmpeg audio extraction failed"
                fail_w.writerow({"output_stem": out_stem, "error": msg,
                                 "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            try:
                ok = run_sadtalker(
                    sadtalker_dir, portrait, tmp_wav, out_path,
                    size=args.size, still=not args.no_still,
                    enhancer=args.enhancer,
                )
            except FileNotFoundError as e:
                log.error(str(e))
                sys.exit(1)
            except subprocess.TimeoutExpired:
                ok = False
                fail_w.writerow({"output_stem": out_stem, "error": "SadTalker timed out",
                                 "timestamp": datetime.now().isoformat()})
                failed += 1
                continue
            except Exception as e:
                log.error(f"  SadTalker error: {e}", exc_info=True)
                fail_w.writerow({"output_stem": out_stem, "error": str(e),
                                 "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            if ok and out_path.exists():
                meta_w.writerow({
                    "output_stem":   out_stem,
                    "track1_source": t1_path.name,
                    "portrait_used": portrait.name,
                    "face_emotion":  info["face_emotion"],
                    "audio_emotion": info["audio_emotion"],
                    "actor_id":      actor_id,
                    "sentence":      info["sentence"],
                    "sadtalker_size": args.size,
                    "timestamp":     datetime.now().isoformat(),
                })
                done.add(out_stem)
                succeeded += 1
                if succeeded % 50 == 0:
                    save_checkpoint(out_dir, done)
                    meta_f.flush()
            else:
                fail_w.writerow({"output_stem": out_stem,
                                 "error": "SadTalker returned failure or no output file",
                                 "timestamp": datetime.now().isoformat()})
                failed += 1

    save_checkpoint(out_dir, done)
    meta_f.close()
    failed_f.close()
    log.info(f"\nDone. Succeeded: {succeeded}  Failed: {failed}  Total: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Track 3 -- SadTalker talking head generation."
    )
    parser.add_argument("--track1_dir",    required=True)
    parser.add_argument("--portraits_dir", required=True)
    parser.add_argument("--sadtalker_dir", required=True)
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--size",          type=int, default=256,
                        choices=[256, 512])
    parser.add_argument("--no_still",      action="store_true",
                        help="Allow free head motion (default: still mode)")
    parser.add_argument("--enhancer",      default="none",
                        choices=["none", "gfpgan"])
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
