"""
track3_generate.py
==================
Track 3 deepfake generation: StyleTTS2 + RVC + SadTalker.

Takes track3_pairs.csv (non-overlapping 50% split of CREMA-D) and produces
the most convincing class of fake — both audio AND face are fully synthesised:
  - StyleTTS2 synthesises speech in the target emotion
  - Applio RVC transfers the actor's vocal identity
  - SadTalker generates a complete talking-head video from a single portrait
    image driven by the synthesised audio (3DMM-based head pose + lip sync)

Difference from other tracks (all use the same unique source clips):
  Track 1 — fake audio, original face video (lip mismatch visible)
  Track 2 — fake audio, Wav2Lip lip-reanimated face (lips match audio)
  Track 3 — fake audio, fully generated face from portrait (identity + lips)

Pipeline per clip:
  1. Resolve CREMA-D face video, extract middle frame as portrait image
  2. StyleTTS2 synthesises speech in target emotion
  3. Applio RVC converts synthesised voice to actor's timbre
  4. X-vector similarity check (optional)
  5. SadTalker: portrait image + RVC audio -> full talking-head video

Usage (run from repo root):
    python src/track3/track3_generate.py \\
        --pairs_csv    data/processed/track1_manifests/track3_pairs.csv \\
        --out_dir      data/synthetic/track3_fakes \\
        --applio_dir   tools/Applio \\
        --sadtalker_dir tools/SadTalker \\
        --cremad_dir   data/raw/CREMA-D \\
        --resume

    # 25% partition runs (run one at a time, ~930 clips each):
    python ... --max_clips 930              # batch 1
    python ... --max_clips 1861 --resume    # batch 2
    python ... --max_clips 2791 --resume    # batch 3
    python ... --resume                     # batch 4 (all remaining)

Outputs:
    data/synthetic/track3_fakes/
        videos/               -- fake .mp4 clips (*_sadtalker.mp4)
        metadata.csv          -- completed clip log
        failed.csv            -- failed clips with error messages
        progress_track3.json  -- checkpoint for resuming
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
from glob import glob

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        log.warning("AudioWAV not found — StyleTTS will use default style")
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
    return run_ffmpeg(['-i', flv_path,
                       '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                       '-c:a', 'aac', mp4_path])


def extract_middle_frame(video_path: str, out_jpg: str) -> bool:
    """Extract middle frame from video as portrait image for SadTalker."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(result.stdout.strip())
        mid = duration / 2
    except Exception:
        mid = 0.5
    return run_ffmpeg([
        '-ss', str(mid), '-i', video_path,
        '-vframes', '1', '-update', '1', '-q:v', '2', out_jpg,
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


# ── SadTalker inference ────────────────────────────────────────────────────────

def run_sadtalker(sadtalker_dir: Path, portrait_jpg: str, audio_wav: str,
                  out_video: str, size: int = 256, still: bool = True) -> bool:
    """
    Run SadTalker inference.py to generate talking-head video.
    SadTalker writes to a timestamped subdir — we find and move the output.
    Must run with cwd=sadtalker_dir (SadTalker uses relative src/config paths).
    """
    inference_script = sadtalker_dir / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"SadTalker inference.py not found at {inference_script}. "
            "Clone SadTalker to tools/SadTalker."
        )

    with tempfile.TemporaryDirectory() as tmp_result:
        cmd = [
            sys.executable, str(inference_script.resolve()),
            "--source_image",  os.path.abspath(portrait_jpg),
            "--driven_audio",  os.path.abspath(audio_wav),
            "--checkpoint_dir", str((sadtalker_dir / "checkpoints").resolve()),
            "--result_dir",    tmp_result,
            "--size",          str(size),
            "--preprocess",    "crop",
        ]
        if still:
            cmd.append("--still")

        result = subprocess.run(
            cmd,
            cwd=str(sadtalker_dir.resolve()),
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            log.debug(f"SadTalker failed:\n{result.stderr[-1000:]}")
            return False

        # SadTalker writes to tmp_result/<timestamp>/*.mp4
        mp4s = glob(os.path.join(tmp_result, "**", "*.mp4"), recursive=True)
        if not mp4s:
            log.debug("SadTalker produced no MP4 output.")
            return False

        import shutil
        shutil.move(mp4s[0], out_video)
        return os.path.exists(out_video)


# ── Track 3 generator ─────────────────────────────────────────────────────────

class Track3Generator:
    """
    Per-clip pipeline: StyleTTS2 -> RVC -> SadTalker.
    Produces _sadtalker.mp4 files from track3_pairs.csv rows.
    """

    def __init__(self, out_dir: Path, applio_dir: Path, sadtalker_dir: Path,
                 verifier: XVectorVerifier, ref_wavs: dict[str, str],
                 cremad_dir: Path, size: int = 256, still: bool = True):
        self.out_dir      = out_dir
        self.applio_dir   = applio_dir
        self.sadtalker_dir = sadtalker_dir
        self.verifier     = verifier
        self.ref_wavs     = ref_wavs
        self.cremad_dir   = cremad_dir
        self.size         = size
        self.still        = still
        self.video_dir    = out_dir / 'videos'
        self.wav_dir      = out_dir / 'wav_tmp'
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
        out_stem      = row['output_stem'] + '_sadtalker'

        video_path = self._resolve_video_path(row)
        if video_path is None:
            return {'status': 'failed', 'error': 'video file not found'}

        out_video = str(self.video_dir / f"{out_stem}.mp4")
        tts_wav   = str(self.wav_dir   / f"{out_stem}_tts.wav")
        rvc_wav   = str(self.wav_dir   / f"{out_stem}_rvc.wav")
        orig_wav  = str(self.wav_dir   / f"{out_stem}_orig.wav")

        with tempfile.TemporaryDirectory() as tmp:
            # Convert FLV -> MP4 if needed, then extract portrait frame
            face_mp4 = video_path
            if video_path.lower().endswith('.flv'):
                face_mp4 = os.path.join(tmp, 'face.mp4')
                if not convert_flv_to_mp4(video_path, face_mp4):
                    return {'status': 'failed', 'error': 'flv conversion failed'}

            portrait_jpg = os.path.join(tmp, 'portrait.jpg')
            if not extract_middle_frame(face_mp4, portrait_jpg):
                return {'status': 'failed', 'error': 'portrait frame extraction failed'}

            if os.path.exists(rvc_wav):
                log.debug(f"Reusing cached RVC wav: {rvc_wav}")
            else:
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

            ok = run_sadtalker(self.sadtalker_dir, portrait_jpg, rvc_wav,
                               out_video, size=self.size, still=self.still)

        if not ok or not os.path.exists(out_video):
            return {'status': 'failed', 'error': 'SadTalker failed'}

        return {
            'status':        'done',
            'output_path':   out_video,
            'method':        'styletts_rvc_sadtalker',
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
        description="Track 3: StyleTTS2 + RVC + SadTalker deepfake generator"
    )
    parser.add_argument('--pairs_csv',     required=True,
                        help='track3_pairs.csv from sample_by_track.py')
    parser.add_argument('--out_dir',       default='data/synthetic/track3_fakes')
    parser.add_argument('--applio_dir',    required=True,
                        help='Path to Applio directory (contains core.py)')
    parser.add_argument('--sadtalker_dir', required=True,
                        help='Path to SadTalker directory (contains inference.py)')
    parser.add_argument('--cremad_dir',    required=True,
                        help='CREMA-D root directory')
    parser.add_argument('--size',          type=int, default=256,
                        choices=[256, 512],
                        help='SadTalker output resolution (default: 256)')
    parser.add_argument('--no_still',      action='store_true',
                        help='Allow head motion (default: still pose)')
    parser.add_argument('--max_clips',     type=int, default=None,
                        help='Cap total clips processed — use for 25%% batches: '
                             '930 / 1861 / 2791 / (omit for all)')
    parser.add_argument('--resume',        action='store_true',
                        help='Skip clips already in checkpoint')
    parser.add_argument('--xvector_threshold', type=float, default=0.75)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log.addHandler(logging.FileHandler(log_file))
    log.info(f"Log: {log_file}")

    sadtalker_dir = Path(args.sadtalker_dir)
    if not (sadtalker_dir / "inference.py").exists():
        log.error(f"SadTalker inference.py not found at {sadtalker_dir}.")
        sys.exit(1)

    pairs = pd.read_csv(args.pairs_csv)
    log.info(f"Loaded {len(pairs)} pairs from {args.pairs_csv}")

    if 'status' in pairs.columns:
        pairs = pairs[pairs['status'] == 'pending'].reset_index(drop=True)
        log.info(f"  {len(pairs)} pending")

    checkpoint = out_dir / 'progress_track3.json'
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
    generator = Track3Generator(
        out_dir       = out_dir,
        applio_dir    = Path(args.applio_dir),
        sadtalker_dir = sadtalker_dir,
        verifier      = verifier,
        ref_wavs      = ref_wavs,
        cremad_dir    = Path(args.cremad_dir),
        size          = args.size,
        still         = not args.no_still,
    )

    meta_csv   = out_dir / 'metadata.csv'
    failed_csv = out_dir / 'failed.csv'
    results  = pd.read_csv(meta_csv).to_dict('records')   if (args.resume and meta_csv.exists())   else []
    failed   = pd.read_csv(failed_csv).to_dict('records') if (args.resume and failed_csv.exists()) else []
    n_done   = 0
    n_failed = 0

    log.info("Starting Track 3 generation — StyleTTS2 + RVC + SadTalker")
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
        pd.DataFrame(results).to_csv(out_dir / 'metadata.csv', index=False)
        log.info(f"Saved metadata: {out_dir / 'metadata.csv'}")
    if failed:
        pd.DataFrame(failed).to_csv(out_dir / 'failed.csv', index=False)
        log.info(f"Failed clips: {out_dir / 'failed.csv'}")

    print("\n" + "=" * 55)
    print("TRACK 3 GENERATION COMPLETE")
    print("=" * 55)
    print(f"Generated successfully:  {n_done}")
    print(f"Failed:                  {n_failed}")
    print(f"Output directory:        {out_dir}/videos/")
    print("=" * 55)


if __name__ == '__main__':
    main()
