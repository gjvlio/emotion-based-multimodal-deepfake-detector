"""
Track 3 — High-Quality Emotion Talking Head Generation

Generates the most convincing class of deepfake in the pipeline:
both audio AND video are fully synthesised with the target emotion,
and the face is generated (not reanimated) using a per-actor fine-tuned
Hallo model — making it the hardest to detect.

Difference from previous tracks:
  Track 1 — synthetic audio only, original face, lip mismatch visible
  Track 2 — synthetic audio, original face, lips reanimated (Wav2Lip)
  Track 3 — synthetic audio + fully generated face video, emotion-consistent,
             per-actor LoRA ensures identity fidelity

Pipeline per clip:
  1. Load per-actor LoRA weights for Hallo (identity fine-tune)
  2. Extract synthesised audio from Track 1 _styletts.mp4
     (reuses Track 1 Method B audio; avoids re-running StyleTTS2/RVC)
  3. Load actor's best portrait frame as reference image
  4. Run Hallo inference: portrait + audio + LoRA → full face video
     with lip sync AND facial emotion expression
  5. Write output to data/synthetic/track3_fakes/videos/

Usage:
  python src/track3/track3_generate.py \
    --track1_dir    data/synthetic/track1_fakes \
    --portraits_dir data/processed/actor_portraits \
    --hallo_dir     tools/Hallo \
    --lora_dir      tools/Hallo/lora \
    --out_dir       data/synthetic/track3_fakes \
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

CHECKPOINT_FILE = "progress_track3.json"
METADATA_FILE   = "metadata.csv"
FAILED_FILE     = "failed.csv"
METADATA_COLS   = [
    "output_stem", "track1_source", "portrait_used",
    "face_emotion", "audio_emotion", "actor_id", "sentence",
    "lora_path", "hallo_model", "timestamp",
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


def find_actor_ckpt(ckpt_dir: Path, actor_id: str) -> Path | None:
    """Return per-actor fine-tuned Hallo checkpoint dir if it exists."""
    p = ckpt_dir / f"actor_{actor_id}"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Hallo inference
# ---------------------------------------------------------------------------

def run_hallo(
    hallo_dir:    Path,
    portrait:     Path,
    audio_wav:    Path,
    out_video:    Path,
    actor_ckpt:   Path | None,
) -> bool:
    """
    Call Hallo's inference script to generate a talking head video.

    Hallo takes a YAML config (configs/inference/default.yaml) with model paths,
    plus CLI overrides for the per-clip inputs:
      --source_image  : actor portrait PNG (reference identity)
      --driving_audio : synthesised WAV with target emotion
      --output        : output MP4 path
      --audio_ckpt_dir: optional per-actor fine-tuned checkpoint dir

    The model generates a video where:
      - Face identity is anchored to the portrait via InsightFace embeddings
      - Lip movements are driven by audio via wav2vec features
      - Facial expression and head motion emerge from audio prosody,
        making the synthesised emotion visible in the generated face
    """
    infer_script = hallo_dir / "scripts" / "inference.py"
    if not infer_script.exists():
        raise FileNotFoundError(
            f"Hallo inference script not found at {infer_script}. "
            "Clone Hallo to tools/Hallo — see tools/README.md."
        )

    cmd = [
        sys.executable, str(infer_script),
        "--config",         str(hallo_dir / "configs" / "inference" / "default.yaml"),
        "--source_image",   str(portrait),
        "--driving_audio",  str(audio_wav),
        "--output",         str(out_video),
        "--pose_weight",    "1.0",
        "--face_weight",    "1.0",
        "--lip_weight",     "1.0",
        "--face_expand_ratio", "1.2",
    ]

    # If a per-actor fine-tuned checkpoint exists, use it instead of the base model
    if actor_ckpt and actor_ckpt.exists():
        cmd += ["--audio_ckpt_dir", str(actor_ckpt)]

    result = subprocess.run(
        cmd,
        cwd=str(hallo_dir),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        log.error(f"Hallo inference failed:\n{result.stderr[-2000:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(args):
    track1_videos = Path(args.track1_dir) / "videos"
    portraits_dir = Path(args.portraits_dir)
    hallo_dir     = Path(args.hallo_dir)
    ckpt_dir      = Path(args.lora_dir)
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
        log.info(f"Resuming — {len(done)} clips already done.")

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
            out_stem = stem.replace("_styletts", "_hallo")
            out_path = vid_dir / f"{out_stem}.mp4"

            if out_stem in done:
                continue

            log.info(f"[{i}/{total}] {stem}")

            info = parse_track1_stem(stem)
            if info is None:
                log.warning(f"  Could not parse stem, skipping.")
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

            actor_ckpt = find_actor_ckpt(ckpt_dir, actor_id)
            if actor_ckpt is None:
                log.warning(f"  No fine-tuned checkpoint for actor {actor_id} — using base Hallo model.")

            tmp_wav = Path(tmpdir) / f"{out_stem}.wav"
            if not extract_audio(t1_path, tmp_wav):
                msg = "ffmpeg audio extraction failed"
                fail_w.writerow({"output_stem": out_stem, "error": msg,
                                 "timestamp": datetime.now().isoformat()})
                failed += 1
                continue

            try:
                ok = run_hallo(hallo_dir, portrait, tmp_wav, out_path, actor_ckpt)
            except FileNotFoundError as e:
                log.error(str(e))
                sys.exit(1)
            except subprocess.TimeoutExpired:
                ok = False
                fail_w.writerow({"output_stem": out_stem, "error": "Hallo timed out",
                                 "timestamp": datetime.now().isoformat()})
                failed += 1
                continue
            except Exception as e:
                log.error(f"  Hallo error: {e}", exc_info=True)
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
                    "lora_path":     str(actor_ckpt) if actor_ckpt else "",
                    "hallo_model":   "hallo_v2",
                    "timestamp":     datetime.now().isoformat(),
                })
                done.add(out_stem)
                succeeded += 1
                if succeeded % 50 == 0:
                    save_checkpoint(out_dir, done)
                    meta_f.flush()
            else:
                fail_w.writerow({"output_stem": out_stem,
                                 "error": "Hallo returned failure or no output file",
                                 "timestamp": datetime.now().isoformat()})
                failed += 1

    save_checkpoint(out_dir, done)
    meta_f.close()
    failed_f.close()
    log.info(f"\nDone. Succeeded: {succeeded}  Failed: {failed}  Total: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Track 3 — Hallo talking head generation with per-actor LoRA."
    )
    parser.add_argument("--track1_dir",    required=True)
    parser.add_argument("--portraits_dir", required=True)
    parser.add_argument("--hallo_dir",     required=True)
    parser.add_argument("--lora_dir",      required=True)
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
