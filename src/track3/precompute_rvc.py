"""
precompute_rvc.py
=================
Pre-compute TTS + RVC wavs for track3 batch clips WITHOUT running SadTalker.
Once wavs exist in wav_tmp, track3_generate.py skips TTS+RVC and runs SadTalker only.

Usage (run from repo root):
    python src/track3/precompute_rvc.py \\
        --pairs_csv  data/processed/track1_manifests/track3_pairs.csv \\
        --out_dir    data/synthetic/track3_fakes \\
        --applio_dir tools/Applio \\
        --cremad_dir data/raw/CREMA-D \\
        --skip_done               # skip stems already in track3 checkpoint
        --start 930 --end 1861    # batch 2 index range (0-based, exclusive end)
"""

import os, sys, re, argparse, subprocess, logging, json
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.track3.track3_generate import (
    EMOTION_STYLE_MAP, build_emotion_refs, load_progress,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def synthesise(tts_model, text: str, emotion: str, ref_wavs: dict, out_wav: str) -> bool:
    ref = ref_wavs.get(emotion)
    try:
        import soundfile as sf
        wav = tts_model.inference(
            text,
            target_voice_path=ref,
            output_sample_rate=24000,
            alpha=0.3, beta=0.7,
            diffusion_steps=10,
            embedding_scale=1.0,
        )
        sf.write(out_wav, wav, 24000)
        return True
    except Exception as e:
        log.error(f"TTS failed: {e}")
        return False


def convert_voice(applio_dir: Path, src_wav: str, actor_id: int, out_wav: str) -> bool:
    logs_dir  = applio_dir / 'logs' / f"actor_{actor_id}"
    pth_files = list(logs_dir.glob('*.pth')) if logs_dir.exists() else []
    if not pth_files:
        log.warning(f"No RVC model for actor {actor_id}")
        return False
    idx_files = list(logs_dir.glob('added_*.index'))
    pth = str(pth_files[0])
    idx = str(idx_files[0]) if idx_files else ''

    result = subprocess.run(
        [
            sys.executable, 'core.py', 'infer',
            '--pitch',           '0',
            '--index_rate',      '0.3' if idx else '0',
            '--volume_envelope', '0.25',
            '--protect',         '0.33',
            '--f0_method',       'rmvpe',
            '--input_path',      os.path.abspath(src_wav),
            '--output_path',     os.path.abspath(out_wav),
            '--pth_path',        os.path.abspath(pth),
            '--index_path',      os.path.abspath(idx) if idx else '',
            '--split_audio',     'False',
            '--clean_audio',     'True',
            '--clean_strength',  '0.7',
            '--export_format',   'WAV',
        ],
        cwd=str(applio_dir),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        log.debug(f"Applio failed: {result.stderr[:300]}")
    return result.returncode == 0 and os.path.exists(out_wav)


def load_wav_checkpoint(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return set(json.load(f).get('completed', []))


def save_wav_checkpoint(path: Path, completed: set):
    with open(path, 'w') as f:
        json.dump({'completed': sorted(completed), 'count': len(completed),
                   'updated_at': datetime.now().isoformat()}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_csv',  required=True)
    parser.add_argument('--out_dir',    default='data/synthetic/track3_fakes')
    parser.add_argument('--applio_dir', required=True)
    parser.add_argument('--cremad_dir', required=True)
    parser.add_argument('--start',      type=int, default=0,
                        help='Start row index in pairs_csv (0-based)')
    parser.add_argument('--end',        type=int, default=None,
                        help='End row index (exclusive). Omit = all remaining.')
    parser.add_argument('--skip_done',  action='store_true',
                        help='Skip stems already in track3 checkpoint')
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    wav_tmp  = out_dir / 'wav_tmp'
    wav_tmp.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv)
    if 'status' in pairs.columns:
        pairs = pairs[pairs['status'] == 'pending'].reset_index(drop=True)

    batch = pairs.iloc[args.start:args.end].reset_index(drop=True)
    log.info(f"Batch rows {args.start}:{args.end} → {len(batch)} clips")

    # skip stems already done in track3 checkpoint
    if args.skip_done:
        ckpt_done = load_progress(out_dir / 'progress_track3.json')
        before = len(batch)
        batch = batch[~batch['output_stem'].isin(ckpt_done)].reset_index(drop=True)
        log.info(f"Skipped {before - len(batch)} already in track3 checkpoint")

    # wav-level checkpoint (separate from track3 checkpoint)
    wav_ckpt_path = out_dir / 'progress_rvc_precompute.json'
    wav_done = load_wav_checkpoint(wav_ckpt_path)
    before = len(batch)
    batch = batch[~batch['output_stem'].isin(wav_done)].reset_index(drop=True)
    log.info(f"Skipped {before - len(batch)} already precomputed. {len(batch)} remaining.")

    # skip clips whose rvc wav already exists in wav_tmp
    need = []
    for _, row in batch.iterrows():
        stem    = row['output_stem'] + '_sadtalker'
        rvc_wav = wav_tmp / f"{stem}_rvc.wav"
        if rvc_wav.exists():
            wav_done.add(row['output_stem'])
        else:
            need.append(row)
    batch = pd.DataFrame(need).reset_index(drop=True)
    log.info(f"RVC wavs needed: {len(batch)}")

    if len(batch) == 0:
        log.info("All wavs already present. Done.")
        save_wav_checkpoint(wav_ckpt_path, wav_done)
        return

    # load TTS
    import nltk
    nltk.download('punkt_tab', quiet=True)
    from styletts2 import tts as styletts
    tts_model = styletts.StyleTTS2()
    log.info("StyleTTS2 loaded.")

    ref_wavs = build_emotion_refs(Path(args.cremad_dir))
    applio   = Path(args.applio_dir)

    n_ok = n_fail = 0
    for _, row in tqdm(batch.iterrows(), total=len(batch), desc="TTS+RVC"):
        stem       = row['output_stem'] + '_sadtalker'
        tts_wav    = str(wav_tmp / f"{stem}_tts.wav")
        rvc_wav    = str(wav_tmp / f"{stem}_rvc.wav")
        emotion    = EMOTION_STYLE_MAP[row['audio_emotion']]
        actor_id   = int(row['actor_id'])

        ok = synthesise(tts_model, row['sentence_text'], emotion, ref_wavs, tts_wav)
        if not ok:
            n_fail += 1
            log.warning(f"TTS failed: {row['output_stem']}")
            continue

        ok = convert_voice(applio, tts_wav, actor_id, rvc_wav)
        if not ok:
            n_fail += 1
            log.warning(f"RVC failed: {row['output_stem']}")
            continue

        wav_done.add(row['output_stem'])
        n_ok += 1

        if (n_ok + n_fail) % 20 == 0:
            save_wav_checkpoint(wav_ckpt_path, wav_done)

    save_wav_checkpoint(wav_ckpt_path, wav_done)
    log.info(f"Done. TTS+RVC ok={n_ok} fail={n_fail}")
    log.info(f"Wavs in: {wav_tmp}")


if __name__ == '__main__':
    main()
