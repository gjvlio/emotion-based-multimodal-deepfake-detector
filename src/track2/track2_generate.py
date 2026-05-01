"""
Track 2 — Face Reenactment Deepfake Generation

Takes Track 1 Method B fakes (synthetic audio + original face) and runs
Wav2Lip to sync the actor's lip movements to the synthesised audio.

This closes the lip-sync gap that made Track 1 detectable, producing a
significantly harder-to-detect fake:
  - Face expresses original emotion (unchanged from CREMA-D)
  - Lips move in sync with the synthesised audio
  - Voice expresses a different emotion (from Track 1 Method B)

Pipeline per clip:
  1. Extract synthesised audio from Track 1 _styletts.mp4 via ffmpeg
  2. Run Wav2Lip: original CREMA-D face video + extracted audio → reanimated video
  3. Write output to data/synthetic/track2_fakes/videos/

Usage (run from src/track2/):
  python track2_generate.py \
    --track1_dir  ../../data/synthetic/track1_fakes \
    --cremad_dir  ../../data/raw/CREMA-D \
    --wav2lip_dir ../../tools/Wav2Lip \
    --out_dir     ../../data/synthetic/track2_fakes \
    [--resume]
"""

import argparse
import csv
import json
import logging
import os
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

CHECKPOINT_FILE = "progress_track2.json"
METADATA_FILE = "metadata.csv"
FAILED_FILE = "failed.csv"
METADATA_COLS = [
    "output_stem", "track1_source", "cremad_face_video",
    "face_emotion", "audio_emotion", "actor_id",
    "sentence", "wav2lip_model", "timestamp",
]


# ---------------------------------------------------------------------------
# Checkpoint helpers  (same resume pattern as Track 1)
# ---------------------------------------------------------------------------

def load_checkpoint(out_dir: Path) -> set:
    path = out_dir / CHECKPOINT_FILE
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(out_dir: Path, done: set):
    with open(out_dir / CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(done), f, indent=2)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, out_wav: Path) -> bool:
    """Extract audio track from an MP4 into a WAV file."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(out_wav),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.error(f"ffmpeg audio extract failed:\n{result.stderr}")
        return False
    return True


def video_has_audio(video_path: Path) -> bool:
    """Return True if the video file contains an audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Wav2Lip inference
# ---------------------------------------------------------------------------

def run_wav2lip(
    wav2lip_dir: Path,
    face_video: Path,
    audio_wav: Path,
    out_video: Path,
    model_path: Path,
) -> bool:
    """
    Call Wav2Lip's inference.py to reanimate lip movements.

    Wav2Lip takes the original face video and the target audio, then
    generates a new video where the mouth region is reanimated to match
    the audio while the rest of the face is left unchanged.
    """
    inference_script = wav2lip_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"Wav2Lip inference.py not found at {inference_script}. "
            f"Clone Wav2Lip to tools/Wav2Lip and download the model checkpoint. "
            f"See tools/README.md."
        )

    result = subprocess.run(
        [
            sys.executable, str(inference_script),
            "--checkpoint_path", str(model_path),
            "--face",           str(face_video),
            "--audio",          str(audio_wav),
            "--outfile",        str(out_video),
            "--nosmooth",       # skip temporal smoothing for cleaner frames
        ],
        cwd=str(wav2lip_dir),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        log.error(f"Wav2Lip failed:\n{result.stderr[-2000:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def parse_track1_stem(stem: str) -> dict | None:
    """
    Parse a Track 1 Method B output stem into its components.

    Expected format: FAKE_T1_{actor}_{sentence}_{face_emo}_{intensity}__AUDIO_{actor}_{sentence}_{audio_emo}_{intensity}_styletts
    Example:         FAKE_T1_1001_DFA_ANG_XX__AUDIO_1001_DFA_HAP_MD_styletts
    """
    stem = stem.replace("_styletts", "")
    if not stem.startswith("FAKE_T1_"):
        return None
    try:
        parts = stem[len("FAKE_T1_"):].split("__AUDIO_")
        face_parts  = parts[0].split("_")   # actor, sentence, emotion, intensity
        audio_parts = parts[1].split("_")
        return {
            "actor_id":      face_parts[0],
            "sentence":      face_parts[1],
            "face_emotion":  face_parts[2],
            "face_intensity": face_parts[3],
            "audio_emotion": audio_parts[2],
            "audio_intensity": audio_parts[3],
        }
    except (IndexError, ValueError):
        return None


def find_cremad_video(cremad_dir: Path, actor: str, sentence: str,
                      emotion: str, intensity: str) -> Path | None:
    """Locate the original CREMA-D video for the given clip."""
    stem = f"{actor}_{sentence}_{emotion}_{intensity}"
    for ext in (".mp4", ".flv"):
        for subdir in ("VideoFlash", "VideoMP4", ""):
            candidate = cremad_dir / subdir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def find_wav2lip_model(wav2lip_dir: Path) -> Path | None:
    """Find the Wav2Lip model checkpoint (.pth) in the checkpoints/ folder."""
    for name in ("wav2lip_gan.pth", "wav2lip.pth"):
        candidate = wav2lip_dir / "checkpoints" / name
        if candidate.exists():
            return candidate
    return None


def load_filter_stems(filter_csv: str | None) -> set | None:
    """Return set of base output_stems to process, or None to process all."""
    if filter_csv is None:
        return None
    import csv as _csv
    stems = set()
    with open(filter_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            stems.add(row["output_stem"])
    return stems


def generate(args):
    track1_videos = Path(args.track1_dir) / "videos"
    cremad_dir    = Path(args.cremad_dir)
    wav2lip_dir   = Path(args.wav2lip_dir)
    out_dir       = Path(args.out_dir)
    vid_dir       = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Find Wav2Lip checkpoint
    model_path = find_wav2lip_model(wav2lip_dir)
    if model_path is None:
        log.error(
            "No Wav2Lip checkpoint found in tools/Wav2Lip/checkpoints/. "
            "Download wav2lip_gan.pth — see tools/README.md."
        )
        sys.exit(1)
    log.info(f"Using Wav2Lip model: {model_path.name}")

    # Collect Track 1 Method B source files
    filter_stems = load_filter_stems(args.filter_csv)
    all_styletts = sorted(track1_videos.glob("*_styletts.mp4"))
    if filter_stems is not None:
        styletts_files = [f for f in all_styletts
                          if f.stem.replace("_styletts", "") in filter_stems]
        log.info(f"Filter: {len(styletts_files)}/{len(all_styletts)} clips selected via --filter_csv.")
    else:
        styletts_files = all_styletts
    if not styletts_files:
        log.error(f"No _styletts.mp4 files to process in {track1_videos}.")
        sys.exit(1)
    log.info(f"Processing {len(styletts_files)} Track 1 clips.")

    # Resume support
    done = load_checkpoint(out_dir) if args.resume else set()
    if done:
        log.info(f"Resuming — {len(done)} clips already completed.")

    # Output metadata / failed writers
    meta_path   = out_dir / METADATA_FILE
    failed_path = out_dir / FAILED_FILE
    meta_exists = meta_path.exists()
    meta_f   = open(meta_path,   "a", newline="", encoding="utf-8")
    failed_f = open(failed_path, "a", newline="", encoding="utf-8")
    meta_writer   = csv.DictWriter(meta_f,   fieldnames=METADATA_COLS)
    failed_writer = csv.DictWriter(failed_f, fieldnames=["output_stem", "error", "timestamp"])
    if not meta_exists:
        meta_writer.writeheader()
        failed_writer.writeheader()

    total = len(styletts_files)
    succeeded = 0
    failed    = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, t1_path in enumerate(styletts_files, 1):
            stem = t1_path.stem          # e.g. FAKE_T1_1001_DFA_ANG_XX__AUDIO_1001_DFA_HAP_MD_styletts
            out_stem = stem.replace("_styletts", "_wav2lip")
            out_path = vid_dir / f"{out_stem}.mp4"

            if out_stem in done:
                continue

            log.info(f"[{i}/{total}] {stem}")

            info = parse_track1_stem(stem)
            if info is None:
                log.warning(f"  Could not parse stem, skipping: {stem}")
                continue

            # Find the original CREMA-D face video (unmodified mouth movements)
            face_video = find_cremad_video(
                cremad_dir,
                info["actor_id"], info["sentence"],
                info["face_emotion"], info["face_intensity"],
            )
            if face_video is None:
                msg = f"CREMA-D face video not found for {info}"
                log.error(f"  {msg}")
                failed_writer.writerow({"output_stem": out_stem, "error": msg,
                                        "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            # Extract synthesised audio from the Track 1 MP4
            tmp_wav = Path(tmpdir) / f"{out_stem}.wav"
            if not extract_audio(t1_path, tmp_wav):
                msg = "ffmpeg audio extraction failed"
                failed_writer.writerow({"output_stem": out_stem, "error": msg,
                                        "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            # Run Wav2Lip
            try:
                ok = run_wav2lip(wav2lip_dir, face_video, tmp_wav, out_path, model_path)
            except subprocess.TimeoutExpired:
                ok = False
                msg = "Wav2Lip subprocess timed out"
                log.error(f"  {msg}")
                failed_writer.writerow({"output_stem": out_stem, "error": msg,
                                        "timestamp": datetime.now().isoformat()})
                failed += 1
                continue
            except Exception as e:
                ok = False
                log.error(f"  Wav2Lip error: {e}", exc_info=True)
                failed_writer.writerow({"output_stem": out_stem, "error": str(e),
                                        "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            if ok and out_path.exists():
                meta_writer.writerow({
                    "output_stem":       out_stem,
                    "track1_source":     t1_path.name,
                    "cremad_face_video": face_video.name,
                    "face_emotion":      info["face_emotion"],
                    "audio_emotion":     info["audio_emotion"],
                    "actor_id":          info["actor_id"],
                    "sentence":          info["sentence"],
                    "wav2lip_model":     model_path.name,
                    "timestamp":         datetime.now().isoformat(),
                })
                done.add(out_stem)
                succeeded += 1
                if succeeded % 50 == 0:
                    save_checkpoint(out_dir, done)
                    meta_f.flush()
            else:
                failed_writer.writerow({
                    "output_stem": out_stem,
                    "error":       "Wav2Lip returned failure or output file missing",
                    "timestamp":   datetime.now().isoformat(),
                })
                failed += 1

    save_checkpoint(out_dir, done)
    meta_f.close()
    failed_f.close()

    log.info(f"\nDone. Succeeded: {succeeded}  Failed: {failed}  Total: {total}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Track 2 — Wav2Lip face reenactment on Track 1 Method B fakes."
    )
    parser.add_argument("--track1_dir",  required=True,
                        help="Path to Track 1 output dir (contains videos/*_styletts.mp4)")
    parser.add_argument("--cremad_dir",  required=True,
                        help="Root CREMA-D directory (for original face videos)")
    parser.add_argument("--wav2lip_dir", required=True,
                        help="Path to cloned Wav2Lip tool directory")
    parser.add_argument("--out_dir",     required=True,
                        help="Output directory for Track 2 fakes")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip clips already present in the checkpoint file")
    parser.add_argument("--filter_csv",  default=None,
                        help="Only process clips whose output_stem appears in this pairs CSV "
                             "(e.g. track2_pairs.csv from sample_by_track.py)")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
