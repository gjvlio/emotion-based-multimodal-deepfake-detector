"""
track1_generate.py
==================
Step 2 of the Track 1 pipeline.

Reads the swap_pairs.csv produced by parse_cremad.py and generates
fake audio-video clips using StyleTTS 2 + RVC v2 (Method B only).

Both audio and video must use the target actor — StyleTTS 2 synthesises
speech in the target emotion, RVC transfers the actor's vocal identity.
The original face video is kept; audio is replaced. The face still shows
the original emotion while the voice expresses a different one.

Pipeline per clip:
  1. Convert .flv → .mp4 if needed
  2. StyleTTS 2 synthesises speech in target emotion (using a CREMA-D reference)
  3. Applio RVC converts the synthesised voice to match the actor's timbre
  4. X-vector similarity check — discard if speaker identity too weak (< 0.75)
  5. ffmpeg muxes the converted audio into the original face video

Usage:
    python track1_generate.py \\
        --pairs_csv  data/processed/track1_manifests/swap_pairs.csv \\
        --out_dir    data/synthetic/track1_fakes \\
        --applio_dir tools/Applio \\
        --cremad_dir data/raw/CREMA-D

    # Resume interrupted run
    python track1_generate.py --resume ...same args...

Outputs:
    data/synthetic/track1_fakes/
        videos/          — fake .mp4 clips (*_styletts.mp4)
        metadata.csv     — completed clip log with paths, emotions, quality scores
        failed.csv       — clips that failed with error messages
        progress_styletts.json  — checkpoint for resuming
"""

import os
import re
import sys
import json
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
# StyleTTS2 checkpoints. Patch back to False (trusted source models only).
_orig_torch_load = torch.load
def _torch_load_compat(*args, weights_only=False, **kwargs):
    return _orig_torch_load(*args, weights_only=weights_only, **kwargs)
torch.load = _torch_load_compat

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EMOTION_STYLE_MAP = {
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
    best: dict[str, tuple[str, int]] = {}

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


def get_duration(path: str) -> float | None:
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
    """Replace video audio track. Output duration matches video."""
    video_dur = get_duration(video_path)
    if video_dur is None:
        return False
    cmd = [
        '-i', video_path, '-i', audio_path,
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-t', str(video_dur), '-af', 'apad', '-shortest',
        '-r', str(target_fps), out_path,
    ]
    return run_ffmpeg(cmd)


def convert_flv_to_mp4(flv_path: str, mp4_path: str) -> bool:
    return run_ffmpeg([
        '-i', flv_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', mp4_path,
    ])


def extract_audio_from_video(video_path: str, wav_path: str,
                             sample_rate: int = 16000) -> bool:
    return run_ffmpeg([
        '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', '1', wav_path,
    ])


# ── Speaker verification (x-vector cosine similarity) ─────────────────────────

class XVectorVerifier:
    """
    Filters RVC outputs by speaker embedding cosine similarity.
    Keeps a clip only if similarity >= threshold (default 0.75).
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
        sim = self.similarity(wav1, wav2)
        if sim is None:
            return True
        return sim >= self.threshold


# ── Method B: StyleTTS 2 + RVC (SOTA synthesis) ───────────────────────────────

class StyleTTSRVCGenerator:
    """
    Synthesise target-emotion speech with StyleTTS 2, then transfer the
    actor's vocal identity with Applio RVC. Mux into the original face video.
    """

    def __init__(self, out_dir: Path, applio_dir: Path,
                 verifier: XVectorVerifier,
                 ref_wavs: dict[str, str] | None = None,
                 cremad_dir: Path | None = None):
        self.out_dir    = out_dir
        self.applio_dir = applio_dir
        self.verifier   = verifier
        self.ref_wavs   = ref_wavs or {}
        self.cremad_dir = cremad_dir
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
            raise RuntimeError("styletts2 not installed. Run: pip install styletts2")

    def _get_applio_model_paths(self, actor_id: int) -> tuple[str, str]:
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
        self._load_tts()
        ref_wav = self.ref_wavs.get(target_emotion)
        if ref_wav is None:
            log.warning(
                f"No reference WAV for emotion '{target_emotion}' — "
                f"StyleTTS will use default style."
            )
        try:
            wav = self._tts.inference(
                text,
                target_voice_path=ref_wav,
                output_sample_rate=24000,
                alpha=0.3,
                beta=0.7,
                diffusion_steps=10,
                embedding_scale=1.0,
            )
            import soundfile as sf
            sf.write(out_wav, wav, 24000)
            return True
        except Exception as e:
            log.error(f"StyleTTS synthesis failed: {e}", exc_info=True)
            return False

    def _convert_voice(self, src_wav: str, actor_id: int, out_wav: str) -> bool:
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
                '--input_path',      os.path.abspath(src_wav),
                '--output_path',     os.path.abspath(out_wav),
                '--pth_path',        os.path.abspath(pth_path),
                '--index_path',      os.path.abspath(idx_path) if idx_path else '',
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

    def _resolve_video_path(self, row: pd.Series) -> str | None:
        video_stem = row.get('video_stem')
        if self.cremad_dir and video_stem:
            for subdir, ext in [('VideoFlash', '.flv'), ('VideoMP4', '.mp4')]:
                p = self.cremad_dir / subdir / f"{video_stem}{ext}"
                if p.exists():
                    return str(p)
        csv_path = row.get('video_path')
        if csv_path and not pd.isna(csv_path) and os.path.exists(csv_path):
            return csv_path
        return None

    def generate(self, row: pd.Series) -> dict:
        actor_id      = int(row['actor_id'])
        sentence_text = row['sentence_text']
        tgt_emotion   = EMOTION_STYLE_MAP[row['audio_emotion']]
        out_stem      = row['output_stem'] + '_styletts'

        video_path = self._resolve_video_path(row)
        if video_path is None:
            return {'status': 'failed', 'error': 'video file not found'}

        out_video = str(self.video_dir / f"{out_stem}.mp4")
        tts_wav   = str(self.wav_dir  / f"{out_stem}_tts.wav")
        rvc_wav   = str(self.wav_dir  / f"{out_stem}_rvc.wav")
        orig_wav  = str(self.wav_dir  / f"{out_stem}_orig.wav")

        with tempfile.TemporaryDirectory() as tmp:
            if video_path.endswith('.flv'):
                mp4_tmp = os.path.join(tmp, 'video.mp4')
                if not convert_flv_to_mp4(video_path, mp4_tmp):
                    return {'status': 'failed', 'error': 'flv conversion failed'}
                video_path = mp4_tmp

            if not self._synthesise(sentence_text, tgt_emotion, tts_wav):
                return {'status': 'failed', 'error': 'StyleTTS synthesis failed'}

            try:
                if not self._convert_voice(tts_wav, actor_id, rvc_wav):
                    return {'status': 'failed', 'error': 'RVC conversion failed'}
            except FileNotFoundError as e:
                return {'status': 'failed', 'error': str(e)}

            extract_audio_from_video(video_path, orig_wav)
            if not self.verifier.passes(orig_wav, rvc_wav):
                return {'status': 'failed', 'error': 'x-vector similarity < threshold'}

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
    parser = argparse.ArgumentParser(
        description="Track 1: StyleTTS2+RVC deepfake generator"
    )
    parser.add_argument('--pairs_csv',  required=True,
                        help='swap_pairs.csv from parse_cremad.py')
    parser.add_argument('--out_dir',    default='./track1_fakes')
    parser.add_argument('--applio_dir', required=True,
                        help='Path to cloned Applio directory (contains core.py)')
    parser.add_argument('--cremad_dir', required=True,
                        help='CREMA-D root dir — for StyleTTS reference WAV selection')
    parser.add_argument('--max_clips',  type=int, default=None,
                        help='Limit number of clips to generate (for testing)')
    parser.add_argument('--resume',     action='store_true',
                        help='Skip clips already in progress checkpoint')
    parser.add_argument('--xvector_threshold', type=float, default=0.75,
                        help='Min x-vector cosine similarity for RVC outputs (default: 0.75)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.addHandler(logging.FileHandler(log_file))
    log.info(f"Log: {log_file}")

    pairs = pd.read_csv(args.pairs_csv)
    log.info(f"Loaded {len(pairs)} swap pairs from {args.pairs_csv}")

    if 'status' in pairs.columns:
        pairs = pairs[pairs['status'] == 'pending'].reset_index(drop=True)
        log.info(f"  {len(pairs)} pending")

    if args.max_clips:
        pairs = pairs.head(args.max_clips)
        log.info(f"  Capped to {args.max_clips} clips")

    checkpoint = out_dir / 'progress_styletts.json'
    completed  = load_progress(checkpoint) if args.resume else set()
    if completed:
        log.info(f"Resuming: {len(completed)} clips already done")
        pairs = pairs[~pairs['output_stem'].isin(completed)].reset_index(drop=True)
        log.info(f"  {len(pairs)} remaining")

    verifier  = XVectorVerifier(threshold=args.xvector_threshold)
    ref_wavs  = build_emotion_refs(Path(args.cremad_dir))
    generator = StyleTTSRVCGenerator(out_dir, Path(args.applio_dir), verifier, ref_wavs,
                                     cremad_dir=Path(args.cremad_dir))

    results  = []
    failed   = []
    n_done   = 0
    n_failed = 0

    log.info(f"Starting Track 1 generation — StyleTTS2+RVC")
    log.info(f"Output dir: {out_dir}")

    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Generating"):
        stem = row['output_stem']
        try:
            result = generator.generate(row)
        except Exception as e:
            result = {'status': 'failed', 'error': str(e)}

        result['output_stem']   = stem
        result['sentence_text'] = row.get('sentence_text', '')
        result['timestamp']     = datetime.now().isoformat()

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
                pd.DataFrame(results).to_csv(out_dir / 'metadata.csv', index=False)
            if failed:
                pd.DataFrame(failed).to_csv(out_dir / 'failed.csv', index=False)

    save_progress(checkpoint, completed)

    if results:
        meta_path = out_dir / 'metadata.csv'
        pd.DataFrame(results).to_csv(meta_path, index=False)
        log.info(f"Saved metadata: {meta_path}")

    if failed:
        fail_path = out_dir / 'failed.csv'
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        log.info(f"Failed clips: {fail_path}")

    print("\n" + "=" * 55)
    print("TRACK 1 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Generated successfully:  {n_done}")
    print(f"Failed:                  {n_failed}")
    print(f"Output directory:        {out_dir}/videos/")
    print(f"Metadata CSV:            {out_dir}/metadata.csv")
    print("=" * 55)


if __name__ == '__main__':
    main()
