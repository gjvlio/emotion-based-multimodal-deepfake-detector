"""
track1_generate.py
==================
Step 2 of the Track 1 pipeline.

Reads the swap_pairs.csv produced by parse_cremad.py and generates
fake audio-video clips where the face emotion and voice emotion
deliberately mismatch.

METHOD A (default, fast): Cross-emotion audio swap
    Takes the video of clip X (face = emotion A) and replaces
    its audio with the audio of clip Y (voice = emotion B),
    where A ≠ B, same actor, same sentence.
    No model inference — just ffmpeg.
    ~0.5 seconds per clip. Recommended for 90% of your fakes.

METHOD B (optional, SOTA): StyleTTS 2 + RVC v2
    Synthesises a new audio waveform expressing emotion B from
    the sentence text, then uses RVC v2 to transfer the original
    speaker's timbre onto it.
    Requires model downloads. ~3-5 minutes per clip.
    Recommended for 10% of fakes to add diversity.

Usage:
    # Method A only (recommended starting point)
    python track1_generate.py \\
        --pairs_csv ./track1_manifests/swap_pairs.csv \\
        --out_dir   ./track1_fakes \\
        --method    swap

    # Method B (StyleTTS 2 + RVC) — set up models first
    python track1_generate.py \\
        --pairs_csv ./track1_manifests/swap_pairs.csv \\
        --out_dir   ./track1_fakes \\
        --method    styletts \\
        --rvc_models_dir ./rvc_models

    # Resume interrupted run (skips already-done clips)
    python track1_generate.py --resume ...same args...

Outputs:
    ./track1_fakes/
        videos/             — fake .mp4 clips (video = emotion A, audio = emotion B)
        metadata.csv        — completed clip log with paths, emotions, quality scores
        failed.csv          — clips that failed with error messages
        progress.json       — checkpoint for resuming
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
import tempfile
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch 2.6 changed torch.load to default weights_only=True, which breaks
# StyleTTS2 checkpoints that use non-standard globals. Patch back to False so
# the library's pretrained models load correctly (they come from trusted sources).
_orig_torch_load = torch.load
def _torch_load_compat(*args, weights_only=False, **kwargs):
    return _orig_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _torch_load_compat

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EMOTION_STYLE_MAP = {
    # Maps CREMA-D emotion codes to StyleTTS 2 style descriptors
    'ANG': 'angry',
    'DIS': 'disgusted',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

# ── StyleTTS 2 reference audio auto-discovery ─────────────────────────────────

def build_emotion_refs(cremad_dir: Path) -> dict[str, str]:
    """
    Pick one high-quality reference WAV per emotion from CREMA-D AudioWAV.
    StyleTTS 2 extracts vocal style from these to control emotional prosody.
    Preference order: HI intensity > XX > MD > LO.
    Returns dict: emotion_label -> wav_path  (e.g. 'angry' -> '/.../.wav')
    """
    audio_dir = cremad_dir / 'AudioWAV'
    if not audio_dir.exists():
        log.warning("AudioWAV not found — StyleTTS will use default (neutral) style")
        return {}

    INTENSITY_RANK = {'HI': 0, 'XX': 1, 'MD': 2, 'LO': 3}
    pattern = re.compile(r'^(\d{4})_([A-Z]{2,3})_([A-Z]{2,3})_([A-Z]{2})\.wav$')
    best: dict[str, tuple[str, int]] = {}  # emotion_label -> (path, rank)

    for f in audio_dir.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue
        _, _, emotion_code, intensity_code = m.groups()
        label = EMOTION_STYLE_MAP.get(emotion_code)
        if label is None:
            continue
        rank = INTENSITY_RANK.get(intensity_code, 99)
        if label not in best or rank < best[label][1]:
            best[label] = (str(f), rank)

    refs = {label: path for label, (path, _) in best.items()}
    if refs:
        log.info(f"StyleTTS reference WAVs selected: {sorted(refs.keys())}")
    else:
        log.warning("No reference WAVs found — StyleTTS will use default style")
    return refs


# ── ffmpeg helpers ─────────────────────────────────────────────────────────────

def run_ffmpeg(cmd: list[str], timeout: int = 120) -> bool:
    """Run an ffmpeg command silently. Returns True on success."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error'] + cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            log.debug(f"ffmpeg error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg timeout")
        return False
    except FileNotFoundError:
        log.error("ffmpeg not found — install it with: sudo apt install ffmpeg")
        sys.exit(1)


def get_duration(path: str) -> float | None:
    """Get media duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def mux_video_audio(video_path: str, audio_path: str, out_path: str,
                    target_fps: int = 30) -> bool:
    """
    Replace the audio track of a video with a new audio file.
    The output duration matches the VIDEO duration (audio is trimmed/padded).
    Both streams re-encoded to H.264 / AAC for compatibility.
    """
    video_dur = get_duration(video_path)
    if video_dur is None:
        return False

    # Build ffmpeg command:
    #   -i video  → input 0 (video stream)
    #   -i audio  → input 1 (audio stream)
    #   -map 0:v  → take video from input 0
    #   -map 1:a  → take audio from input 1
    #   -t        → trim output to video duration
    #   -shortest → just in case audio is shorter than video
    #   -af apad  → pad with silence if audio is shorter
    cmd = [
        '-i', video_path,
        '-i', audio_path,
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-t', str(video_dur),
        '-af', 'apad',
        '-shortest',
        '-r', str(target_fps),
        out_path,
    ]
    return run_ffmpeg(cmd)


def convert_flv_to_mp4(flv_path: str, mp4_path: str) -> bool:
    """Convert a .flv video to .mp4 for compatibility."""
    cmd = [
        '-i', flv_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        mp4_path,
    ]
    return run_ffmpeg(cmd)


def extract_audio_from_video(video_path: str, wav_path: str,
                             sample_rate: int = 16000) -> bool:
    """Extract audio from a video file as 16kHz mono WAV."""
    cmd = [
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        wav_path,
    ]
    return run_ffmpeg(cmd)


# ── Speaker verification (x-vector cosine similarity) ─────────────────────────

class XVectorVerifier:
    """
    Computes speaker embedding similarity between two audio clips
    using SpeechBrain's x-vector model.

    Used to filter RVC outputs: we only keep a converted clip if
    the speaker embedding cosine similarity >= threshold (ACE-Net used 0.75).
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from speechbrain.pretrained import SpeakerRecognition
            self._model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="./pretrained_models/xvect",
            )
            log.info("X-vector model loaded.")
        except ImportError:
            log.warning("speechbrain not installed — x-vector filtering disabled.")
            self._model = 'unavailable'

    def similarity(self, wav1: str, wav2: str) -> float | None:
        """Returns cosine similarity in [-1, 1] or None if unavailable."""
        self._load()
        if self._model == 'unavailable':
            return None
        try:
            score, _ = self._model.verify_files(wav1, wav2)
            return float(score)
        except Exception as e:
            log.debug(f"X-vector error: {e}")
            return None

    def passes(self, wav1: str, wav2: str) -> bool:
        """True if similarity >= threshold, or if verification is unavailable."""
        sim = self.similarity(wav1, wav2)
        if sim is None:
            return True  # Skip filter if model unavailable
        return sim >= self.threshold


# ── Method A: Cross-emotion audio swap (no model) ─────────────────────────────

class EmotionSwapGenerator:
    """
    Method A: Replace a video clip's audio with a same-actor,
    same-sentence, different-emotion audio clip.

    This is the primary Track 1 method. No model inference.
    Pure emotional mismatch, zero synthesis artifacts.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.video_dir = out_dir / 'videos'
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, row: pd.Series) -> dict:
        """
        Generate one fake clip from a swap pair row.
        Returns a result dict with status and output path.
        """
        video_path = row['video_path']
        audio_path = row['audio_path']
        out_stem   = row['output_stem']

        if pd.isna(video_path) or pd.isna(audio_path):
            return {'status': 'failed', 'error': 'missing video or audio path'}

        out_path = str(self.video_dir / f"{out_stem}.mp4")

        # If video is .flv, convert first
        with tempfile.TemporaryDirectory() as tmp:
            if video_path.endswith('.flv'):
                mp4_tmp = os.path.join(tmp, 'video.mp4')
                if not convert_flv_to_mp4(video_path, mp4_tmp):
                    return {'status': 'failed', 'error': 'flv→mp4 conversion failed'}
                video_path = mp4_tmp

            # Replace audio track
            ok = mux_video_audio(video_path, audio_path, out_path)

        if not ok or not os.path.exists(out_path):
            return {'status': 'failed', 'error': 'ffmpeg mux failed'}

        return {
            'status':        'done',
            'output_path':   out_path,
            'method':        'emotion_swap',
            'video_emotion': row['video_emotion'],
            'audio_emotion': row['audio_emotion'],
            'actor_id':      row['actor_id'],
            'sentence_key':  row['sentence_key'],
            'label':         1,
        }


# ── Method B: StyleTTS 2 + RVC (SOTA synthesis) ───────────────────────────────

class StyleTTSRVCGenerator:
    """
    Method B: Synthesise a new audio with target emotion using StyleTTS 2,
    then transfer the original speaker's voice using RVC v2.

    This produces genuinely synthesised fakes, giving your training set
    diversity beyond simple audio swapping.

    Setup required before using this:
      1. pip install styletts2 rvc-python
      2. Train an RVC model per actor:
             python train_rvc_voices.py --cremad_dir /path/to/CREMA-D
    """

    def __init__(self, out_dir: Path, applio_dir: Path,
                 verifier: XVectorVerifier,
                 ref_wavs: dict[str, str] | None = None):
        self.out_dir    = out_dir
        self.applio_dir = applio_dir   # path to cloned Applio (contains core.py)
        self.verifier   = verifier
        self.ref_wavs   = ref_wavs or {}
        self.video_dir  = out_dir / 'videos'
        self.wav_dir    = out_dir / 'wav_tmp'
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None

    def _load_tts(self):
        if self._tts is not None:
            return
        try:
            import nltk
            nltk.download('punkt_tab', quiet=True)
            from styletts2 import tts
            self._tts = tts.StyleTTS2()
            log.info("StyleTTS 2 loaded.")
        except ImportError:
            raise RuntimeError(
                "styletts2 not installed. Run: pip install styletts2"
            )

    def _get_applio_model_paths(self, actor_id: int) -> tuple[str, str]:
        """
        Return (pth_path, index_path) for an actor from Applio's logs dir.
        index_path may be '' if no index file exists yet.
        Raises FileNotFoundError if no .pth exists.
        """
        logs_dir  = self.applio_dir / 'logs' / f"actor_{actor_id}"
        pth_files = list(logs_dir.glob('*.pth')) if logs_dir.exists() else []
        if not pth_files:
            raise FileNotFoundError(
                f"No RVC model for actor {actor_id} in {logs_dir}.\n"
                f"Run train_rvc_voices.py --actors {actor_id} first."
            )
        idx_files = list(logs_dir.glob('added_*.index'))
        return str(pth_files[0]), (str(idx_files[0]) if idx_files else '')

    def _synthesise(self, text: str, target_emotion: str, out_wav: str) -> bool:
        """Use StyleTTS 2 to generate speech in target_emotion.

        target_emotion is an emotion label string ('angry', 'sad', etc.).
        A matching CREMA-D reference WAV is passed as target_voice so that
        StyleTTS 2 extracts the emotional prosody style from it.
        """
        self._load_tts()
        ref_wav = self.ref_wavs.get(target_emotion)  # None → neutral default
        if ref_wav is None:
            log.warning(
                f"No reference WAV for emotion '{target_emotion}' — "
                f"StyleTTS will use default style. Pass --cremad_dir to fix this."
            )
        try:
            wav = self._tts.inference(
                text,
                target_voice_path=ref_wav,  # emotion-specific reference audio
                output_sample_rate=24000,
                alpha=0.3,             # style blending weight
                beta=0.7,
                diffusion_steps=10,    # balance quality vs speed
                embedding_scale=1.0,
            )
            import soundfile as sf
            sf.write(out_wav, wav, 24000)
            return True
        except Exception as e:
            log.error(f"StyleTTS synthesis failed: {e}", exc_info=True)
            return False

    def _convert_voice(self, src_wav: str, actor_id: int, out_wav: str) -> bool:
        """Transfer actor vocal identity via Applio RVC inference (subprocess)."""
        try:
            pth_path, idx_path = self._get_applio_model_paths(actor_id)
        except FileNotFoundError as e:
            log.debug(str(e))
            return False

        result = subprocess.run(
            [
                sys.executable, 'core.py', 'infer',
                '--pitch',           '0',
                '--index_rate',      '0.3' if idx_path else '0',
                '--volume_envelope', '0.25',
                '--protect',         '0.33',
                '--f0_method',       'rmvpe',
                '--input_path',      src_wav,
                '--output_path',     out_wav,
                '--pth_path',        pth_path,
                '--index_path',      idx_path,
                '--split_audio',     'False',
                '--clean_audio',     'True',
                '--clean_strength',  '0.7',
                '--export_format',   'WAV',
            ],
            cwd=str(self.applio_dir),
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            log.debug(f"Applio infer failed: {result.stderr[:300]}")
        return result.returncode == 0 and os.path.exists(out_wav)

    def generate(self, row: pd.Series) -> dict:
        video_path     = row['video_path']
        actor_id       = int(row['actor_id'])
        sentence_text  = row['sentence_text']
        tgt_emotion    = EMOTION_STYLE_MAP[row['audio_emotion']]
        out_stem       = row['output_stem'] + '_styletts'

        if pd.isna(video_path):
            return {'status': 'failed', 'error': 'missing video path'}

        out_video = str(self.video_dir / f"{out_stem}.mp4")
        tts_wav   = str(self.wav_dir  / f"{out_stem}_tts.wav")
        rvc_wav   = str(self.wav_dir  / f"{out_stem}_rvc.wav")
        orig_wav  = str(self.wav_dir  / f"{out_stem}_orig.wav")

        with tempfile.TemporaryDirectory() as tmp:
            # Step 1: convert .flv → .mp4 if needed
            if video_path.endswith('.flv'):
                mp4_tmp = os.path.join(tmp, 'video.mp4')
                if not convert_flv_to_mp4(video_path, mp4_tmp):
                    return {'status': 'failed', 'error': 'flv conversion failed'}
                video_path = mp4_tmp

            # Step 2: synthesise speech in target emotion
            if not self._synthesise(sentence_text, tgt_emotion, tts_wav):
                return {'status': 'failed', 'error': 'StyleTTS synthesis failed'}

            # Step 3: transfer speaker identity via RVC
            try:
                if not self._convert_voice(tts_wav, actor_id, rvc_wav):
                    return {'status': 'failed', 'error': 'RVC conversion failed'}
            except FileNotFoundError as e:
                return {'status': 'failed', 'error': str(e)}

            # Step 4: x-vector verification (keep if speaker similarity >= 0.75)
            # Extract original audio for comparison
            extract_audio_from_video(video_path, orig_wav)
            if not self.verifier.passes(orig_wav, rvc_wav):
                return {'status': 'failed', 'error': 'x-vector similarity < 0.75'}

            # Step 5: mux converted audio into original video
            ok = mux_video_audio(video_path, rvc_wav, out_video)

        if not ok or not os.path.exists(out_video):
            return {'status': 'failed', 'error': 'mux failed'}

        return {
            'status':        'done',
            'output_path':   out_video,
            'method':        'styletts_rvc',
            'video_emotion': row['video_emotion'],
            'audio_emotion': row['audio_emotion'],
            'actor_id':      actor_id,
            'sentence_key':  row['sentence_key'],
            'label':         1,
        }


# ── Progress checkpoint ────────────────────────────────────────────────────────

def load_progress(checkpoint_path: Path) -> set:
    """Load set of already-completed output stems from checkpoint."""
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path) as f:
        data = json.load(f)
    return set(data.get('completed', []))


def save_progress(checkpoint_path: Path, completed: set):
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'completed':  list(completed),
            'updated_at': datetime.now().isoformat(),
            'count':      len(completed),
        }, f, indent=2)


# ── Main generation loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Track 1: Audio-tampered deepfake generator")
    parser.add_argument('--pairs_csv',     required=True,
                        help='swap_pairs.csv from parse_cremad.py')
    parser.add_argument('--out_dir',       default='./track1_fakes',
                        help='Output directory (default: ./track1_fakes)')
    parser.add_argument('--method',        choices=['swap', 'styletts', 'both'],
                        default='swap',
                        help='Generation method: swap (fast) | styletts (SOTA) | both')
    parser.add_argument('--applio_dir',    default=None,
                        help='Path to cloned Applio directory (needed for styletts method)')
    parser.add_argument('--cremad_dir',    default=None,
                        help='CREMA-D root dir — used to auto-select StyleTTS reference clips per emotion')
    parser.add_argument('--max_clips',     type=int, default=None,
                        help='Limit number of clips to generate (for testing)')
    parser.add_argument('--resume',        action='store_true',
                        help='Skip clips already in progress checkpoint')
    parser.add_argument('--xvector_threshold', type=float, default=0.75,
                        help='Min x-vector cosine similarity for RVC outputs (default: 0.75)')
    args = parser.parse_args()

    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = out_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.addHandler(logging.FileHandler(log_file))
    log.info(f"Log: {log_file}")

    # Load pairs
    pairs = pd.read_csv(args.pairs_csv)
    log.info(f"Loaded {len(pairs)} swap pairs from {args.pairs_csv}")

    # Filter to pending only
    if 'status' in pairs.columns:
        pairs = pairs[pairs['status'] == 'pending'].reset_index(drop=True)
        log.info(f"  {len(pairs)} pending (status='pending')")

    if args.max_clips:
        pairs = pairs.head(args.max_clips)
        log.info(f"  Capped to {args.max_clips} clips")

    # Resume checkpoint — per-method file so swap and styletts don't share keys
    checkpoint = out_dir / f'progress_{args.method}.json'
    completed  = load_progress(checkpoint) if args.resume else set()
    if completed:
        log.info(f"Resuming: {len(completed)} clips already done")
        pairs = pairs[~pairs['output_stem'].isin(completed)].reset_index(drop=True)
        log.info(f"  {len(pairs)} remaining")

    # Initialise generators
    generators = {}
    verifier   = XVectorVerifier(threshold=args.xvector_threshold)

    if args.method in ('swap', 'both'):
        generators['swap'] = EmotionSwapGenerator(out_dir)

    if args.method in ('styletts', 'both'):
        if not args.applio_dir:
            log.error("--applio_dir is required for --method styletts")
            sys.exit(1)
        ref_wavs = build_emotion_refs(Path(args.cremad_dir)) if args.cremad_dir else {}
        generators['styletts'] = StyleTTSRVCGenerator(
            out_dir, Path(args.applio_dir), verifier, ref_wavs
        )

    # Output collectors
    results   = []
    failed    = []
    n_done    = 0
    n_failed  = 0

    log.info(f"Starting generation — method: {args.method}")
    log.info(f"Output dir: {out_dir}")

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Generating"):
        stem = row['output_stem']

        for method_name, generator in generators.items():
            try:
                result = generator.generate(row)
            except Exception as e:
                result = {'status': 'failed', 'error': str(e)}

            result['output_stem']    = stem
            result['method_used']    = method_name
            result['sentence_text']  = row.get('sentence_text', '')
            result['timestamp']      = datetime.now().isoformat()

            if result['status'] == 'done':
                results.append(result)
                completed.add(stem)
                n_done += 1
            else:
                failed.append(result)
                n_failed += 1
                log.debug(f"FAILED {stem}: {result.get('error', '?')}")

        # Save checkpoint every 50 clips
        if (n_done + n_failed) % 50 == 0:
            save_progress(checkpoint, completed)
            if results:
                pd.DataFrame(results).to_csv(out_dir / 'metadata.csv', index=False)
            if failed:
                pd.DataFrame(failed).to_csv(out_dir / 'failed.csv', index=False)

    # Final save
    save_progress(checkpoint, completed)

    if results:
        meta_path = out_dir / 'metadata.csv'
        pd.DataFrame(results).to_csv(meta_path, index=False)
        log.info(f"Saved metadata: {meta_path}")

    if failed:
        fail_path = out_dir / 'failed.csv'
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        log.info(f"Failed clips: {fail_path}")

    # Summary
    print("\n" + "=" * 55)
    print("TRACK 1 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Generated successfully:  {n_done}")
    print(f"Failed:                  {n_failed}")
    print(f"Output directory:        {out_dir}/videos/")
    print(f"Metadata CSV:            {out_dir}/metadata.csv")
    if n_done:
        print(f"\nNext step:")
        print(f"  Add the genuine clips CSV from parse_cremad.py")
        print(f"  and this metadata.csv to your dataset loader.")
    print("=" * 55)


if __name__ == '__main__':
    main()
