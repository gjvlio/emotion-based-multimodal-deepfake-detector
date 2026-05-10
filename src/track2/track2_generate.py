"""
track2_generate.py
==================
Track 2 deepfake generation: StyleTTS2 + RVC + Wav2Lip.

Takes track2_pairs.csv (non-overlapping 30% split of CREMA-D) and produces
fake clips where BOTH the audio AND lip movements are synthesised:
  - StyleTTS2 synthesises speech in the target emotion
  - Applio RVC transfers the actor's vocal identity
  - Wav2Lip reanimates the lip region of the original face video to match the
    synthesised audio

This is strictly harder to detect than Track 1 (audio-only swap) because the
visual lip sync inconsistency is also corrected.

Pipeline per clip:
  1. Resolve original CREMA-D face video (FLV or MP4)
  2. Convert FLV -> MP4 if needed (Wav2Lip prefers MP4)
  3. StyleTTS2 synthesises speech in target emotion
  4. Applio RVC converts synthesised voice to actor's timbre
  5. X-vector similarity check (optional filter)
  6. Wav2Lip reanimates lips in original face video to match RVC audio
     (retries with resize_factor 2 then 4 on face-detection failure)

Usage (run from repo root):
    python src/track2/track2_generate.py \\
        --pairs_csv  data/processed/track1_manifests/track2_pairs.csv \\
        --out_dir    data/synthetic/track2_fakes \\
        --applio_dir tools/Applio \\
        --wav2lip_dir tools/Wav2Lip \\
        --cremad_dir data/raw/CREMA-D \\
        --resume

    # 25% partition runs (run one at a time):
    python ... --max_clips 567              # batch 1
    python ... --max_clips 1134 --resume    # batch 2 (resumes, adds next 567)
    python ... --max_clips 1701 --resume    # batch 3
    python ... --resume                     # batch 4 (all remaining)

Outputs:
    data/synthetic/track2_fakes/
        videos/               -- fake .mp4 clips (*_wav2lip.mp4)
        metadata.csv          -- completed clip log
        failed.csv            -- failed clips with error messages
        progress_track2.json  -- checkpoint for resuming
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

if 'numpy.exceptions' not in sys.modules:
    import types as _types
    _exc = _types.ModuleType('numpy.exceptions')
    for _name in ('AxisError', 'ComplexWarning', 'DTypePromotionError',
                  'ModuleDeprecationWarning', 'RankWarning',
                  'TooHardError', 'VisibleDeprecationWarning'):
        if hasattr(np, _name):
            setattr(_exc, _name, getattr(np, _name))
    sys.modules['numpy.exceptions'] = _exc

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


# ── StyleTTS2 reference audio auto-discovery ──────────────────────────────────

def build_emotion_refs(cremad_dir: Path) -> dict[str, str]:
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


# ── Speaker verification ───────────────────────────────────────────────────────

class XVectorVerifier:
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

    def passes(self, wav1: str, wav2: str) -> bool:
        self._load()
        if self._model == 'unavailable':
            return True
        try:
            score, _ = self._model.verify_files(wav1, wav2)
            return float(score) >= self.threshold
        except Exception as e:
            log.debug(f"X-vector error: {e}")
            return True


# ── Wav2Lip inference ──────────────────────────────────────────────────────────

def run_wav2lip(wav2lip_dir: Path, face_mp4: str, audio_wav: str,
                out_video: str, model_path: Path) -> bool:
    """
    Run Wav2Lip on an MP4 face video + WAV audio.
    Retries with resize_factor 2 then 4 on face-detection failure.
    face_mp4 must already be an MP4 (convert FLV before calling).
    """
    inference_script = wav2lip_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"Wav2Lip inference.py not found at {inference_script}. "
            "Clone Wav2Lip to tools/Wav2Lip and download wav2lip_gan.pth."
        )

    for resize in (1, 2, 4, 8):
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
            log.warning(f"Face not detected at resize_factor={resize}, retrying...")
            continue
        log.debug(f"Wav2Lip failed (resize={resize}):\n{result.stderr[-1000:]}")
        return False

    log.error("Wav2Lip face detection failed at all resize factors (1, 2, 4, 8).")
    return False


def find_wav2lip_model(wav2lip_dir: Path) -> Path | None:
    for name in ("wav2lip_gan.pth", "wav2lip.pth"):
        p = wav2lip_dir / "checkpoints" / name
        if p.exists():
            return p
    return None


# ── Track 2 generator ─────────────────────────────────────────────────────────

class Track2Generator:
    """
    Per-clip pipeline: StyleTTS2 -> RVC -> Wav2Lip.
    Produces _wav2lip.mp4 files from track2_pairs.csv rows.
    """

    def __init__(self, out_dir: Path, applio_dir: Path, wav2lip_dir: Path,
                 wav2lip_model: Path, verifier: XVectorVerifier,
                 ref_wavs: dict[str, str], cremad_dir: Path):
        self.out_dir       = out_dir
        self.applio_dir    = applio_dir
        self.wav2lip_dir   = wav2lip_dir
        self.wav2lip_model = wav2lip_model
        self.verifier      = verifier
        self.ref_wavs      = ref_wavs
        self.cremad_dir    = cremad_dir
        self.video_dir     = out_dir / 'videos'
        self.wav_dir       = out_dir / 'wav_tmp'
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
            log.info("StyleTTS2 loaded.")
        except ImportError:
            raise RuntimeError("styletts2 not installed. Run: pip install styletts2")

    def _get_applio_model_paths(self, actor_id: int) -> tuple[str, str]:
        logs_dir  = self.applio_dir / 'logs' / f"actor_{actor_id}"
        pth_files = list(logs_dir.glob('*.pth')) if logs_dir.exists() else []
        if not pth_files:
            raise FileNotFoundError(
                f"No RVC model for actor {actor_id} in {logs_dir}."
            )
        idx_files = list(logs_dir.glob('added_*.index'))
        return str(pth_files[0]), (str(idx_files[0]) if idx_files else '')

    def _synthesise(self, text: str, target_emotion: str, out_wav: str) -> bool:
        self._load_tts()
        ref_wav = self.ref_wavs.get(target_emotion)
        try:
            wav = self._tts.inference(
                text,
                target_voice_path=ref_wav,
                output_sample_rate=24000,
                alpha=0.3, beta=0.7,
                diffusion_steps=5,
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
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.debug(f"Applio infer failed: {result.stderr[:300]}")
        return result.returncode == 0 and os.path.exists(out_wav)

    def _resolve_video_path(self, row: pd.Series) -> str | None:
        video_stem = row.get('video_stem')
        if video_stem:
            for subdir, ext in [('VideoMP4', '.mp4'), ('VideoFlash', '.flv')]:
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
        out_stem      = row['output_stem'] + '_wav2lip'

        video_path = self._resolve_video_path(row)
        if video_path is None:
            return {'status': 'failed', 'error': 'video file not found'}

        out_video = str(self.video_dir / f"{out_stem}.mp4")
        tts_wav   = str(self.wav_dir   / f"{out_stem}_tts.wav")
        rvc_wav   = str(self.wav_dir   / f"{out_stem}_rvc.wav")
        orig_wav  = str(self.wav_dir   / f"{out_stem}_orig.wav")

        with tempfile.TemporaryDirectory() as tmp:
            # FLV -> MP4 (Wav2Lip needs MP4)
            face_mp4 = video_path
            if video_path.lower().endswith('.flv'):
                face_mp4 = os.path.join(tmp, 'face.mp4')
                if not convert_flv_to_mp4(video_path, face_mp4):
                    return {'status': 'failed', 'error': 'flv conversion failed'}

            if not self._synthesise(sentence_text, tgt_emotion, tts_wav):
                return {'status': 'failed', 'error': 'StyleTTS synthesis failed'}

            try:
                if not self._convert_voice(tts_wav, actor_id, rvc_wav):
                    return {'status': 'failed', 'error': 'RVC conversion failed'}
            except FileNotFoundError as e:
                return {'status': 'failed', 'error': str(e)}

            extract_audio_from_video(face_mp4, orig_wav)
            if not self.verifier.passes(orig_wav, rvc_wav):
                return {'status': 'failed', 'error': 'x-vector similarity < threshold'}

            ok = run_wav2lip(self.wav2lip_dir, face_mp4, rvc_wav,
                             out_video, self.wav2lip_model)

        if not ok or not os.path.exists(out_video):
            return {'status': 'failed', 'error': 'Wav2Lip failed'}

        return {
            'status':        'done',
            'output_path':   out_video,
            'method':        'styletts_rvc_wav2lip',
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
            'completed':  sorted(completed),
            'updated_at': datetime.now().isoformat(),
            'count':      len(completed),
        }, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track 2: StyleTTS2 + RVC + Wav2Lip deepfake generator"
    )
    parser.add_argument('--pairs_csv',   required=True,
                        help='track2_pairs.csv from sample_by_track.py')
    parser.add_argument('--out_dir',     default='data/synthetic/track2_fakes')
    parser.add_argument('--applio_dir',  required=True,
                        help='Path to Applio directory (contains core.py)')
    parser.add_argument('--wav2lip_dir', required=True,
                        help='Path to Wav2Lip directory (contains inference.py)')
    parser.add_argument('--cremad_dir',  required=True,
                        help='CREMA-D root directory')
    parser.add_argument('--max_clips',   type=int, default=None,
                        help='Cap total clips processed — use for 25%% batches: '
                             '567 / 1134 / 1701 / (omit for all)')
    parser.add_argument('--resume',      action='store_true',
                        help='Skip clips already in checkpoint')
    parser.add_argument('--xvector_threshold', type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.addHandler(logging.FileHandler(log_file))
    log.info(f"Log: {log_file}")

    wav2lip_dir = Path(args.wav2lip_dir)
    wav2lip_model = find_wav2lip_model(wav2lip_dir)
    if wav2lip_model is None:
        log.error("No Wav2Lip checkpoint found in tools/Wav2Lip/checkpoints/. "
                  "Download wav2lip_gan.pth.")
        sys.exit(1)
    log.info(f"Wav2Lip model: {wav2lip_model.name}")

    pairs = pd.read_csv(args.pairs_csv)
    log.info(f"Loaded {len(pairs)} pairs from {args.pairs_csv}")

    if 'status' in pairs.columns:
        pairs = pairs[pairs['status'] == 'pending'].reset_index(drop=True)
        log.info(f"  {len(pairs)} pending")

    checkpoint = out_dir / 'progress_track2.json'
    completed  = load_progress(checkpoint) if args.resume else set()
    if completed:
        log.info(f"Resuming: {len(completed)} clips already done")
        pairs = pairs[~pairs['output_stem'].isin(completed)].reset_index(drop=True)
        log.info(f"  {len(pairs)} remaining")

    if args.max_clips:
        pairs = pairs.head(args.max_clips)
        log.info(f"Capped to {args.max_clips} clips (batch mode)")

    verifier  = XVectorVerifier(threshold=args.xvector_threshold)
    ref_wavs  = build_emotion_refs(Path(args.cremad_dir))
    generator = Track2Generator(
        out_dir       = out_dir,
        applio_dir    = Path(args.applio_dir),
        wav2lip_dir   = wav2lip_dir,
        wav2lip_model = wav2lip_model,
        verifier      = verifier,
        ref_wavs      = ref_wavs,
        cremad_dir    = Path(args.cremad_dir),
    )

    meta_csv   = out_dir / 'metadata.csv'
    failed_csv = out_dir / 'failed.csv'
    results  = pd.read_csv(meta_csv).to_dict('records')   if (args.resume and meta_csv.exists())   else []
    failed   = pd.read_csv(failed_csv).to_dict('records') if (args.resume and failed_csv.exists()) else []
    n_done   = 0
    n_failed = 0

    log.info("Starting Track 2 generation — StyleTTS2 + RVC + Wav2Lip")
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
    print("TRACK 2 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Generated successfully:  {n_done}")
    print(f"Failed:                  {n_failed}")
    print(f"Output directory:        {out_dir}/videos/")
    print("=" * 55)


if __name__ == '__main__':
    main()
