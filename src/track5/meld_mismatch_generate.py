"""
meld_mismatch_generate.py
=========================
Track 5 deepfake generation: emotion-mismatch lip-sync via MuseTalk on MELD.

Takes meld_mismatch_pairs.csv (50% of MELD, video_emotion != audio_emotion)
and runs MuseTalk lip-sync: donor audio is used to drive face video, producing
a clip where the face lip-syncs to an emotionally mismatched voice.

Manipulation signal: face shows emotion A, lips/voice carry emotion B.
This tests whether the detector learns audio-visual emotional consistency
AND low-level lip/face artefacts (unlike the pure audio-swap baseline).

Contrast with other tracks:
  Track 1 — fake TTS+RVC audio, original face, lip mismatch
  Track 2 — fake TTS+RVC audio, Wav2Lip lips
  Track 3 — fully synthesised face + audio (SadTalker)
  Track 4 — MELD, real cross-speaker audio, Wav2Lip lips
  Track 5 — MELD, real cross-EMOTION audio, MuseTalk lips (this script)

Usage (run from repo root):
    python src/track5/meld_mismatch_generate.py \\
        --pairs_csv data/processed/meld_manifests/meld_mismatch_pairs.csv \\
        --out_dir   data/synthetic/track5_fakes \\
        --resume

    # Limit VRAM: lower batch_size
    python ... --batch_size 2

Outputs:
    data/synthetic/track5_fakes/
        videos/                   -- fake .mp4 clips (*_musetalk.mp4)
        metadata.csv              -- completed clip log
        failed.csv                -- failed clips with error
        progress_track5.json      -- checkpoint for resuming
"""

import json
import logging
import os
import sys
import argparse
import subprocess
import tempfile
import threading
import time
import yaml
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list, timeout: int = 120) -> bool:
    try:
        r = subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error'] + cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            log.debug(f"ffmpeg stderr: {r.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timeout")
        return False
    except FileNotFoundError:
        log.error("ffmpeg not found")
        sys.exit(1)


def extract_audio(video_path: str, wav_path: str) -> bool:
    return run_ffmpeg([
        '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path,
    ])


def write_musetalk_config(config_path: str, video_path: str, audio_path: str,
                           bbox_shift: int = 0):
    cfg = {"task_0": {
        "video_path": os.path.abspath(video_path),
        "audio_path": os.path.abspath(audio_path),
        "bbox_shift":  bbox_shift,
    }}
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)


def find_output_mp4(result_dir: str) -> str | None:
    import glob
    hits = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
    return hits[0] if hits else None


def run_musetalk(musetalk_dir: Path, config_path: str, result_dir: str,
                 batch_size: int = 4, timeout: int = 600) -> tuple[bool, str]:
    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config",  os.path.abspath(config_path),
        "--result_dir",        os.path.abspath(result_dir),
        "--batch_size",        str(batch_size),
        "--version",           "v1",
        "--unet_model_path",   "models/musetalk/pytorch_model.bin",
        "--unet_config",       "models/musetalk/musetalk.json",
        "--vae_type",          "sd-vae-ft-mse",
    ]
    r = subprocess.run(
        cmd,
        cwd=str(musetalk_dir.resolve()),
        capture_output=True, text=True,
        timeout=timeout, env=env,
    )
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-500:].strip()
    return True, ""


# ── Per-clip generation ────────────────────────────────────────────────────────

def generate_clip(row: pd.Series, video_dir: Path, musetalk_dir: Path,
                  batch_size: int, bbox_shift: int) -> dict:
    stem        = row['output_stem']
    video_path  = row['video_clip']
    audio_src   = row['audio_clip']
    out_video   = str(video_dir / f"{stem}_musetalk.mp4")

    if not os.path.exists(video_path):
        return {'status': 'failed', 'error': f'video_clip not found: {video_path}'}
    if not os.path.exists(audio_src):
        return {'status': 'failed', 'error': f'audio_clip not found: {audio_src}'}

    with tempfile.TemporaryDirectory() as tmp:
        donor_wav   = os.path.join(tmp, "donor.wav")
        config_path = os.path.join(tmp, "config.yaml")
        result_dir  = os.path.join(tmp, "results")
        os.makedirs(result_dir)

        if not extract_audio(audio_src, donor_wav):
            return {'status': 'failed', 'error': 'donor audio extraction failed'}

        write_musetalk_config(config_path, video_path, donor_wav, bbox_shift)

        ok, err = run_musetalk(musetalk_dir, config_path, result_dir, batch_size)

        if not ok:
            return {'status': 'failed', 'error': f'MuseTalk failed: {err[:200]}'}

        src = find_output_mp4(result_dir)
        if src is None:
            return {'status': 'failed', 'error': 'MuseTalk produced no output mp4'}

        shutil.copy(src, out_video)

    if not os.path.exists(out_video):
        return {'status': 'failed', 'error': 'output file not found after copy'}

    size_mb = os.path.getsize(out_video) / 1_048_576
    return {
        'status':          'done',
        'output_path':     out_video,
        'output_mb':       round(size_mb, 1),
        'method':          'musetalk_emotion_mismatch',
        'video_speaker':   row['video_speaker'],
        'audio_speaker':   row['audio_speaker'],
        'video_emotion':   row['video_emotion'],
        'audio_emotion':   row['audio_emotion'],
        'video_clip_id':   row['video_clip_id'],
        'audio_clip_id':   row['audio_clip_id'],
        'label':           1,
    }


# ── Progress checkpoint ────────────────────────────────────────────────────────

def load_progress(checkpoint: Path) -> set:
    if not checkpoint.exists():
        return set()
    with open(checkpoint) as f:
        return set(json.load(f).get('completed', []))


def save_progress(checkpoint: Path, completed: set):
    with open(checkpoint, 'w') as f:
        json.dump({
            'completed':  sorted(completed),
            'updated_at': datetime.now().isoformat(),
            'count':      len(completed),
        }, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track 5: emotion-mismatch MuseTalk lip-sync deepfakes from MELD"
    )
    parser.add_argument(
        '--pairs_csv',
        default='data/processed/meld_manifests/meld_mismatch_pairs.csv',
    )
    parser.add_argument('--out_dir',      default='data/synthetic/track5_fakes')
    parser.add_argument('--musetalk_dir', default='tools/MuseTalk')
    parser.add_argument('--max_clips',    type=int, default=None)
    parser.add_argument('--batch_size',   type=int, default=4,
                        help='MuseTalk inference batch size (lower = less VRAM)')
    parser.add_argument('--bbox_shift',   type=int, default=0,
                        help='Vertical lip-region shift in pixels')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    musetalk_dir = Path(args.musetalk_dir)
    if not (musetalk_dir / "scripts" / "inference.py").exists():
        log.error(f"MuseTalk not found at {musetalk_dir}")
        log.error("Clone: git clone https://github.com/TMElyralab/MuseTalk tools/MuseTalk")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.addHandler(logging.FileHandler(log_file))
    log.info(f"Log: {log_file}")

    pairs_csv = Path(args.pairs_csv)
    if not pairs_csv.exists():
        log.error(f"Pairs CSV not found: {pairs_csv}")
        log.error("Run: python scripts/sample_meld_mismatch.py")
        sys.exit(1)

    pairs = pd.read_csv(pairs_csv)
    log.info(f"Loaded {len(pairs)} pairs")

    checkpoint = out_dir / 'progress_track5.json'
    completed  = load_progress(checkpoint) if args.resume else set()
    if completed:
        log.info(f"Resuming: {len(completed)} already done")
        pairs = pairs[~pairs['output_stem'].isin(completed)].reset_index(drop=True)
        log.info(f"  {len(pairs)} remaining")

    if args.max_clips:
        pairs = pairs.head(args.max_clips)
        log.info(f"Capped to {args.max_clips} clips")

    video_dir = out_dir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)

    meta_csv   = out_dir / 'metadata.csv'
    failed_csv = out_dir / 'failed.csv'
    results = pd.read_csv(meta_csv).to_dict('records')   if (args.resume and meta_csv.exists())   else []
    failed  = pd.read_csv(failed_csv).to_dict('records') if (args.resume and failed_csv.exists()) else []
    n_done, n_failed = 0, 0

    log.info("Starting Track 5 — MuseTalk emotion-mismatch lip-sync on MELD")
    log.info(f"Output: {out_dir}")
    log.info(f"MuseTalk: {musetalk_dir}  batch_size={args.batch_size}")

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Track5-MuseTalk"):
        stem = row['output_stem']
        t0 = time.time()
        try:
            result = generate_clip(row, video_dir, musetalk_dir,
                                   args.batch_size, args.bbox_shift)
        except Exception as e:
            result = {'status': 'failed', 'error': str(e)}

        result['output_stem'] = stem
        result['timestamp']   = datetime.now().isoformat()
        result['elapsed_s']   = round(time.time() - t0, 1)

        if result['status'] == 'done':
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
        log.info(f"Metadata: {meta_csv}")
    if failed:
        pd.DataFrame(failed).to_csv(failed_csv, index=False)
        log.info(f"Failed: {failed_csv}")

    print("\n" + "=" * 55)
    print("TRACK 5 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Generated successfully:  {n_done}")
    print(f"Failed:                  {n_failed}")
    print(f"Output directory:        {video_dir}")
    print("=" * 55)


if __name__ == '__main__':
    main()
