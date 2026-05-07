"""
track4_generate.py
==================
Track 4 deepfake generation: cross-speaker Wav2Lip on MELD clips.

Takes meld_pairs.csv (50% of MELD) and produces fake clips where the face
video of speaker A is lip-synced to the real audio of speaker B.  No voice
synthesis — both face and audio come from genuine MELD utterances, making
this an in-the-wild cross-speaker lip-sync attack.

Contrast with CREMA-D tracks:
  Track 1 — lab audio swap, original lips (visible mismatch)
  Track 2 — lab audio swap, Wav2Lip lips (matched)
  Track 3 — lab fully synthesised face + audio
  Track 4 — in-the-wild cross-speaker real audio, Wav2Lip lips (this script)

Pipeline per pair:
  1. Extract WAV from donor (audio_clip) via ffmpeg
  2. Wav2Lip: reanimate lips in video_clip to match donor audio
     (retry with resize_factor 2 then 4 on face-detection failure)
  3. Write *_wav2lip.mp4 to output dir

Usage (run from repo root):
    python src/track4/track4_generate.py \\
        --pairs_csv   data/processed/meld_manifests/meld_pairs.csv \\
        --out_dir     data/synthetic/track4_fakes \\
        --wav2lip_dir tools/Wav2Lip \\
        --resume

    # 25% partition runs:
    python ... --max_clips 1714              # batch 1
    python ... --max_clips 3428 --resume     # batch 2
    python ... --max_clips 5142 --resume     # batch 3
    python ... --resume                      # batch 4 (all remaining)

Outputs:
    data/synthetic/track4_fakes/
        videos/               -- fake .mp4 clips (*_wav2lip.mp4)
        metadata.csv          -- completed clip log
        failed.csv            -- failed clips with error messages
        progress_track4.json  -- checkpoint for resuming
"""

import os
import sys
import json
import argparse
import subprocess
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

_orig_torch_load = None
try:
    import torch
    _orig_torch_load = torch.load
    def _torch_load_compat(*args, weights_only=False, **kwargs):
        return _orig_torch_load(*args, weights_only=weights_only, **kwargs)
    torch.load = _torch_load_compat
except ImportError:
    pass

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list[str], timeout: int = 120) -> bool:
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error'] + cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.debug(f"ffmpeg error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timeout")
        return False
    except FileNotFoundError:
        log.error("ffmpeg not found — install it and add to PATH")
        sys.exit(1)


def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> bool:
    return run_ffmpeg([
        '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', '1', wav_path,
    ])


# ── Wav2Lip ────────────────────────────────────────────────────────────────────

def find_wav2lip_model(wav2lip_dir: Path) -> Path | None:
    for name in ("wav2lip_gan.pth", "wav2lip.pth"):
        p = wav2lip_dir / "checkpoints" / name
        if p.exists():
            return p
    return None


def run_wav2lip(wav2lip_dir: Path, face_mp4: str, audio_wav: str,
                out_video: str, model_path: Path) -> bool:
    inference_script = wav2lip_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"Wav2Lip inference.py not found at {inference_script}. "
            "Clone Wav2Lip to tools/Wav2Lip and download wav2lip_gan.pth."
        )
    for resize in (1, 2, 4):
        cmd = [
            sys.executable, str(inference_script.resolve()),
            "--checkpoint_path", str(model_path.resolve()),
            "--face",    os.path.abspath(face_mp4),
            "--audio",   os.path.abspath(audio_wav),
            "--outfile", os.path.abspath(out_video),
            "--nosmooth",
        ]
        if resize > 1:
            cmd += ["--resize_factor", str(resize)]
        result = subprocess.run(
            cmd,
            cwd=str(wav2lip_dir.resolve()),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and os.path.exists(out_video):
            return True
        if "Face not detected" in result.stderr:
            log.warning(f"Face not detected (resize_factor={resize}), retrying...")
            continue
        log.debug(f"Wav2Lip failed (resize={resize}):\n{result.stderr[-800:]}")
        return False
    log.error("Wav2Lip face detection failed at all resize factors.")
    return False


# ── Per-clip generation ────────────────────────────────────────────────────────

def generate_clip(row: pd.Series, out_dir: Path, wav2lip_dir: Path,
                  wav2lip_model: Path) -> dict:
    video_clip = row["video_clip"]
    audio_clip = row["audio_clip"]
    out_stem   = row["output_stem"] + "_wav2lip"

    if not os.path.exists(video_clip):
        return {"status": "failed", "error": f"video_clip not found: {video_clip}"}
    if not os.path.exists(audio_clip):
        return {"status": "failed", "error": f"audio_clip not found: {audio_clip}"}

    video_dir = out_dir / "videos"
    wav_dir   = out_dir / "wav_tmp"
    video_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    out_video = str(video_dir / f"{out_stem}.mp4")
    donor_wav = str(wav_dir   / f"{out_stem}_donor.wav")

    if not extract_audio(audio_clip, donor_wav):
        return {"status": "failed", "error": "audio extraction from donor failed"}

    ok = run_wav2lip(wav2lip_dir, video_clip, donor_wav, out_video, wav2lip_model)

    if not ok or not os.path.exists(out_video):
        return {"status": "failed", "error": "Wav2Lip failed"}

    return {
        "status":         "done",
        "output_path":    out_video,
        "method":         "wav2lip_cross_speaker",
        "video_speaker":  row["video_speaker"],
        "audio_speaker":  row["audio_speaker"],
        "video_emotion":  row["video_emotion"],
        "audio_emotion":  row["audio_emotion"],
        "video_clip_id":  row["video_clip_id"],
        "audio_clip_id":  row["audio_clip_id"],
        "label":          1,
    }


# ── Progress checkpoint ────────────────────────────────────────────────────────

def load_progress(checkpoint: Path) -> set:
    if not checkpoint.exists():
        return set()
    with open(checkpoint) as f:
        return set(json.load(f).get("completed", []))


def save_progress(checkpoint: Path, completed: set):
    with open(checkpoint, "w") as f:
        json.dump({
            "completed":  sorted(completed),
            "updated_at": datetime.now().isoformat(),
            "count":      len(completed),
        }, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track 4: cross-speaker Wav2Lip deepfakes from MELD"
    )
    parser.add_argument("--pairs_csv",   required=True,
                        help="meld_pairs.csv from sample_meld.py")
    parser.add_argument("--out_dir",     default="data/synthetic/track4_fakes")
    parser.add_argument("--wav2lip_dir", required=True,
                        help="Path to Wav2Lip directory (contains inference.py)")
    parser.add_argument("--max_clips",   type=int, default=None,
                        help="Cap total clips — use for 25%% batches: "
                             "1714 / 3428 / 5142 / (omit for all)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip clips already in checkpoint")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.addHandler(logging.FileHandler(log_file))
    log.info(f"Log: {log_file}")

    wav2lip_dir = Path(args.wav2lip_dir)
    wav2lip_model = find_wav2lip_model(wav2lip_dir)
    if wav2lip_model is None:
        log.error("No Wav2Lip checkpoint found in tools/Wav2Lip/checkpoints/.")
        sys.exit(1)
    log.info(f"Wav2Lip model: {wav2lip_model.name}")

    pairs = pd.read_csv(args.pairs_csv)
    log.info(f"Loaded {len(pairs)} pairs from {args.pairs_csv}")

    checkpoint = out_dir / "progress_track4.json"
    completed  = load_progress(checkpoint) if args.resume else set()
    if completed:
        log.info(f"Resuming: {len(completed)} clips already done")
        pairs = pairs[~pairs["output_stem"].isin(completed)].reset_index(drop=True)
        log.info(f"  {len(pairs)} remaining")

    if args.max_clips:
        pairs = pairs.head(args.max_clips)
        log.info(f"Capped to {args.max_clips} clips (batch mode)")

    meta_csv   = out_dir / "metadata.csv"
    failed_csv = out_dir / "failed.csv"
    results = pd.read_csv(meta_csv).to_dict("records")   if (args.resume and meta_csv.exists())   else []
    failed  = pd.read_csv(failed_csv).to_dict("records") if (args.resume and failed_csv.exists()) else []
    n_done, n_failed = 0, 0

    log.info("Starting Track 4 generation — cross-speaker Wav2Lip on MELD")
    log.info(f"Output dir: {out_dir}")

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Generating"):
        stem = row["output_stem"]
        try:
            result = generate_clip(row, out_dir, wav2lip_dir, wav2lip_model)
        except Exception as e:
            result = {"status": "failed", "error": str(e)}

        result["output_stem"] = stem
        result["timestamp"]   = datetime.now().isoformat()

        if result["status"] == "done":
            results.append(result)
            completed.add(stem)
            n_done += 1
        else:
            failed.append(result)
            n_failed += 1
            log.debug(f"FAILED {stem}: {result.get('error', '?')}")

        if (n_done + n_failed) % 50 == 0:
            save_progress(checkpoint, completed)
            if results:
                pd.DataFrame(results).to_csv(meta_csv, index=False)
            if failed:
                pd.DataFrame(failed).to_csv(failed_csv, index=False)

    save_progress(checkpoint, completed)
    if results:
        pd.DataFrame(results).to_csv(meta_csv, index=False)
        log.info(f"Saved metadata: {meta_csv}")
    if failed:
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        log.info(f"Failed clips: {failed_csv}")

    print("\n" + "=" * 55)
    print("TRACK 4 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Generated successfully:  {n_done}")
    print(f"Failed:                  {n_failed}")
    print(f"Output directory:        {out_dir}/videos/")
    print("=" * 55)


if __name__ == "__main__":
    main()
