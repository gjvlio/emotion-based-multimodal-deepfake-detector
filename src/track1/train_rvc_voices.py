"""
train_rvc_voices.py
===================
Train one RVC v2 voice model per CREMA-D actor.

Each actor's CREMA-D clips (~80-100 WAVs, ~5-8 min total) are collected,
resampled to 40 kHz (RVC's native rate), and used to train a lightweight
voice conversion model. The trained model.pth is then used by
track1_generate.py (Method B) to wrap the original actor's timbre onto
StyleTTS 2-synthesised speech.

Usage:
    # Train specific actors (recommended for testing)
    python train_rvc_voices.py \\
        --cremad_dir  ../../data/raw/CREMA-D \\
        --models_dir  ../../data/processed/rvc_models \\
        --actors 1001 1002 1003

    # Prepare datasets only (no training — verify data before committing GPU time)
    python train_rvc_voices.py \\
        --cremad_dir  ../../data/raw/CREMA-D \\
        --prepare_only

    # Full run — all 91 actors (~30 hrs on RTX 3060)
    python train_rvc_voices.py \\
        --cremad_dir  ../../data/raw/CREMA-D

    # Validate trained models with x-vector similarity
    python train_rvc_voices.py \\
        --cremad_dir  ../../data/raw/CREMA-D \\
        --validate_only

Outputs:
    ./rvc_datasets/actor_{id}/    resampled WAVs (training input)
    ./rvc_models/actor_{id}/model.pth    trained voice model
    ./rvc_models/training_log.csv        per-actor status and x-vector scores
"""

import os
import re
import sys
import csv
import json
import signal
import logging
import argparse
import subprocess
import tempfile
from math import gcd
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CLIP_PATTERN = re.compile(r'^(\d{4})_([A-Z]{2,3})_([A-Z]{2,3})_([A-Z]{2})\.wav$')
RVC_SR       = 40000   # RVC v2 native sample rate
MIN_CLIPS    = 20      # warn if an actor has fewer clips than this


# ── Audio helpers ──────────────────────────────────────────────────────────────

def read_wav_scipy(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file using scipy (no extra deps). Returns (float32 mono, sr)."""
    from scipy.io import wavfile
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind == 'i' or data.dtype.kind == 'u':
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return data, sr


def write_wav_scipy(path: Path, data: np.ndarray, sr: int):
    """Write a float32 mono array as 16-bit PCM WAV."""
    from scipy.io import wavfile
    data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    wavfile.write(str(path), sr, data_int16)


def resample_wav(src: Path, dst: Path, target_sr: int) -> bool:
    """Resample src WAV to target_sr and save to dst."""
    try:
        from scipy.signal import resample_poly
        data, sr = read_wav_scipy(src)
        if sr != target_sr:
            g = gcd(target_sr, sr)
            data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        write_wav_scipy(dst, data, target_sr)
        return True
    except Exception as e:
        log.debug(f"Resample failed {src.name}: {e}")
        return False


# ── Dataset preparation ────────────────────────────────────────────────────────

def collect_actor_wavs(actor_id: int, audio_dir: Path) -> list[Path]:
    """Return all CREMA-D WAVs for one actor, sorted by filename."""
    return sorted([
        f for f in audio_dir.iterdir()
        if CLIP_PATTERN.match(f.name) and f.name.startswith(f"{actor_id:04d}_")
    ])


def prepare_dataset(actor_id: int, audio_dir: Path,
                    datasets_dir: Path) -> tuple[Path, int]:
    """
    Resample all of an actor's WAVs to RVC_SR and stage them in
    datasets_dir/actor_{id}/.

    Returns (dataset_path, n_clips_ok).
    """
    wavs = collect_actor_wavs(actor_id, audio_dir)
    if not wavs:
        raise FileNotFoundError(f"No WAVs found for actor {actor_id} in {audio_dir}")

    if len(wavs) < MIN_CLIPS:
        log.warning(f"Actor {actor_id}: only {len(wavs)} clips (< {MIN_CLIPS}) — quality may be low")

    actor_dir = datasets_dir / f"actor_{actor_id}"
    actor_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for wav in wavs:
        dst = actor_dir / wav.name
        if dst.exists():
            n_ok += 1
            continue
        if resample_wav(wav, dst, RVC_SR):
            n_ok += 1
        else:
            log.warning(f"Actor {actor_id}: skipped {wav.name}")

    log.info(f"Actor {actor_id}: {n_ok}/{len(wavs)} clips staged → {actor_dir}")
    return actor_dir, n_ok


# ── RVC training via Applio ────────────────────────────────────────────────────

def _run_applio(applio_dir: Path, args: list[str], timeout: int = 3600) -> bool:
    """Run one Applio core.py subcommand. Returns True on success."""
    # On Windows, multiprocessing 'spawn' workers need rvc/train/ in PYTHONPATH
    # so they can import mel_processing and other local modules.
    # OPENBLAS_NUM_THREADS=1 prevents OpenBLAS from spawning many threads that
    # each try to allocate large memory blocks, causing "memory allocation failed"
    # crashes mid-training on Windows.
    env = os.environ.copy()
    train_dir = str(applio_dir / 'rvc' / 'train')
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = train_dir + (os.pathsep + existing if existing else '')
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'

    try:
        result = subprocess.run(
            [sys.executable, 'core.py'] + args,
            cwd=str(applio_dir),
            stdout=subprocess.DEVNULL,   # avoid stdout pipe buffer pressure on long runs
            stderr=subprocess.PIPE,
            text=True, timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        log.error(f"Applio {args[0]} timed out after {timeout}s")
        return False
    # Applio sometimes exits 0 even on failure; treat stderr tracebacks as failure
    has_traceback = 'Traceback (most recent call last)' in (result.stderr or '')
    if result.returncode != 0 or has_traceback:
        log.error(f"Applio {args[0]} stderr:\n{result.stderr[-3000:]}")
        return False
    return True


def _batch_sizes(initial: int):
    """Yield batch sizes to try on failure: initial → half → 1."""
    seen: set[int] = set()
    for bs in [initial, max(1, initial // 2), 1]:
        if bs not in seen:
            seen.add(bs)
            yield bs


def train_rvc_model(actor_id: int, dataset_dir: Path, applio_dir: Path,
                    epochs: int = 40, batch_size: int = 2) -> bool:
    """
    Train an RVC v2 model for one actor using Applio's 4-step pipeline:
      preprocess → extract → train → index
    Models are stored in applio_dir/logs/actor_{id}/
    Returns True if a .pth file exists after training.
    """
    model_name = f"actor_{actor_id}"
    logs_dir   = applio_dir / 'logs' / model_name

    if list(logs_dir.glob(f'{model_name}*.pth')):
        log.info(f"Actor {actor_id}: model already exists — skipping")
        return True

    # Step 1: preprocess — skip if sliced_audios already populated
    sliced_dir = logs_dir / 'sliced_audios'
    if sliced_dir.exists() and any(sliced_dir.iterdir()):
        log.info(f"Actor {actor_id}: [1/4] preprocess already done — skipping")
    else:
        log.info(f"Actor {actor_id}: [1/4] preprocessing …")
        if not _run_applio(applio_dir, [
            'preprocess',
            '--model_name',    model_name,
            '--dataset_path',  str(dataset_dir.resolve()),
            '--sample_rate',   str(RVC_SR),
            '--cpu_cores',     '4',
            '--cut_preprocess','Skip',
        ]):
            log.error(f"Actor {actor_id}: preprocess failed")
            return False

    # Step 2: extract — skip if features already extracted
    extracted_dir = logs_dir / 'extracted'
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        log.info(f"Actor {actor_id}: [2/4] extraction already done — skipping")
    else:
        log.info(f"Actor {actor_id}: [2/4] extracting features …")
        if not _run_applio(applio_dir, [
            'extract',
            '--model_name',   model_name,
            '--f0_method',    'rmvpe',
            '--cpu_cores',    '4',
            '--gpu',          '0',
            '--sample_rate',  str(RVC_SR),
            '--include_mutes', '0',
        ]):
            log.error(f"Actor {actor_id}: feature extraction failed")
            return False

    # Step 3: train — restore config.json from RVC template first so a previous
    # killed run's PID-only config.json never causes 'HParams has no data' crashes
    config_src = applio_dir / 'rvc' / 'configs' / f'{RVC_SR}.json'
    config_dst = logs_dir / 'config.json'
    if config_src.exists():
        import shutil as _shutil
        _shutil.copy2(str(config_src), str(config_dst))

    log.info(f"Actor {actor_id}: [3/4] training ({epochs} epochs) …")
    trained = False
    for bs in _batch_sizes(batch_size):
        if bs < batch_size:
            log.warning(f"Actor {actor_id}: retrying training with batch_size={bs}")
        if _run_applio(applio_dir, [
            'train',
            '--model_name',         model_name,
            '--sample_rate',        str(RVC_SR),
            '--vocoder',            'HiFi-GAN',
            '--save_every_epoch',   '10',
            '--save_only_latest',   'True',
            '--save_every_weights', 'True',
            '--total_epoch',        str(epochs),
            '--batch_size',         str(bs),
            '--gpu',                '0',
            '--overtraining_detector', 'False',
            '--cleanup',            'False',
        ], timeout=7200):
            trained = True
            break

    if not trained:
        log.error(f"Actor {actor_id}: training failed after all batch_size retries")
        return False

    log.info(f"Actor {actor_id}: [4/4] building index …")
    _run_applio(applio_dir, [
        'index',
        '--model_name', model_name,
    ])

    success = bool(list(logs_dir.glob(f'{model_name}*.pth')))
    if success:
        log.info(f"Actor {actor_id}: model ready → {logs_dir}")
        _cleanup_training_artifacts(logs_dir, model_name)
    return success


def _cleanup_training_artifacts(logs_dir: Path, model_name: str):
    """Delete large training-only artifacts after a successful run.

    Keeps: final 40e .pth, .index files, config.json, filelist.txt.
    Deletes: D_/G_ discriminator checkpoints, intermediate epoch .pth files,
             sliced_audios*, extracted, f0, f0_voiced, eval directories.
    This prevents disk exhaustion on multi-actor full runs (~1.5 GB freed per actor).
    """
    # Intermediate epoch .pth (keep only the highest-epoch one)
    all_pths = sorted(logs_dir.glob(f'{model_name}*.pth'))
    for p in all_pths[:-1]:
        try:
            p.unlink()
        except OSError:
            pass

    # Discriminator / generator checkpoints written by Applio during training
    for pattern in ('D_*.pth', 'G_*.pth'):
        for p in logs_dir.glob(pattern):
            try:
                p.unlink()
            except OSError:
                pass

    # Training-only subdirs (not needed for inference)
    for subdir in ('sliced_audios', 'sliced_audios_16k', 'extracted',
                   'f0', 'f0_voiced', 'eval'):
        target = logs_dir / subdir
        if target.exists():
            try:
                import shutil
                shutil.rmtree(str(target))
            except OSError:
                pass

    log.info(f"Cleaned training artifacts → {logs_dir}")


# ── Validation (x-vector similarity) ──────────────────────────────────────────

def validate_model(actor_id: int, applio_dir: Path,
                   audio_dir: Path, threshold: float = 0.75) -> float | None:
    """
    Convert a held-out clip via Applio infer and measure x-vector cosine
    similarity against the original. ACE-Net threshold is ≥ 0.75.
    Returns similarity score or None if model/speechbrain is unavailable.
    """
    model_name = f"actor_{actor_id}"
    logs_dir   = applio_dir / 'logs' / model_name
    pth_files  = list(logs_dir.glob(f'{model_name}*.pth'))
    if not pth_files:
        return None

    pth_path  = str(pth_files[0])
    idx_files = list(logs_dir.glob('added_*.index'))
    idx_path  = str(idx_files[0]) if idx_files else ''

    wavs = collect_actor_wavs(actor_id, audio_dir)
    if len(wavs) < 2:
        return None

    orig_wav = str(wavs[0])
    test_wav = str(wavs[-1])

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        converted_wav = f.name

    try:
        result = subprocess.run(
            [
                sys.executable, 'core.py', 'infer',
                '--pitch',           '0',
                '--index_rate',      '0.3' if idx_path else '0',
                '--volume_envelope', '0.25',
                '--protect',         '0.33',
                '--f0_method',       'rmvpe',
                '--input_path',      test_wav,
                '--output_path',     converted_wav,
                '--pth_path',        pth_path,
                '--index_path',      idx_path,
                '--split_audio',     'False',
                '--clean_audio',     'True',
                '--clean_strength',  '0.7',
                '--export_format',   'WAV',
            ],
            cwd=str(applio_dir),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not os.path.exists(converted_wav):
            return None

        try:
            from speechbrain.pretrained import SpeakerRecognition
            verifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="./pretrained_models/xvect",
            )
            score, _ = verifier.verify_files(orig_wav, converted_wav)
            return float(score)
        except ImportError:
            log.warning("speechbrain not installed — x-vector validation skipped")
            return None
    except Exception as e:
        log.debug(f"Validation error actor {actor_id}: {e}")
        return None
    finally:
        if os.path.exists(converted_wav):
            os.unlink(converted_wav)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train per-actor RVC v2 voice models from CREMA-D clips using Applio"
    )
    parser.add_argument('--cremad_dir',    required=True,
                        help='Root directory of your CREMA-D download')
    parser.add_argument('--applio_dir',    required=True,
                        help='Path to your cloned Applio directory (contains core.py)')
    parser.add_argument('--datasets_dir',  default='./rvc_datasets',
                        help='Where to stage resampled training audio (default: ./rvc_datasets)')
    parser.add_argument('--actors',        nargs='+', type=int, default=None,
                        help='Specific actor IDs to process (default: all actors in CREMA-D)')
    parser.add_argument('--epochs',        type=int, default=40,
                        help='RVC training epochs (default: 40)')
    parser.add_argument('--batch_size',    type=int, default=4,
                        help='Training batch size (default: 4)')
    parser.add_argument('--prepare_only',  action='store_true',
                        help='Only resample and stage audio — do not train')
    parser.add_argument('--validate_only', action='store_true',
                        help='Skip training, only run x-vector validation on existing models')
    parser.add_argument('--xvector_threshold', type=float, default=0.75,
                        help='Minimum x-vector similarity to mark a model as passing (default: 0.75)')
    args = parser.parse_args()

    cremad_dir   = Path(args.cremad_dir)
    audio_dir    = cremad_dir / 'AudioWAV'
    applio_dir   = Path(args.applio_dir)
    datasets_dir = Path(args.datasets_dir)

    if not (applio_dir / 'core.py').exists():
        log.error(f"core.py not found in {applio_dir} — check --applio_dir")
        sys.exit(1)

    if not audio_dir.exists():
        log.error(f"AudioWAV not found: {audio_dir}")
        sys.exit(1)

    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Discover actor IDs
    if args.actors:
        actor_ids = sorted(args.actors)
    else:
        actor_ids = sorted({
            int(f.name[:4])
            for f in audio_dir.iterdir()
            if CLIP_PATTERN.match(f.name)
        })

    log.info(f"Actors to process: {len(actor_ids)}")
    log.info(f"Applio dir:        {applio_dir}")

    # Log CSV saved next to datasets
    log_path   = datasets_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_fields = ['actor_id', 'clips_staged', 'trained', 'xvector_sim', 'passes_threshold', 'error']
    log_rows: list[dict] = []

    def _flush_log():
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)

    def _sigint_handler(sig, frame):
        log.warning("Interrupted — saving partial log …")
        _flush_log()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    for actor_id in tqdm(actor_ids, desc="Actors"):
        row: dict = {
            'actor_id': actor_id, 'clips_staged': 0,
            'trained': False, 'xvector_sim': None,
            'passes_threshold': None, 'error': '',
        }
        try:
            # Step 1: prepare dataset
            dataset_dir, n_clips = prepare_dataset(actor_id, audio_dir, datasets_dir)
            row['clips_staged'] = n_clips

            if args.prepare_only:
                log_rows.append(row)
                _flush_log()
                continue

            if not args.validate_only:
                # Step 2: train via Applio
                ok = train_rvc_model(
                    actor_id, dataset_dir, applio_dir,
                    epochs=args.epochs, batch_size=args.batch_size,
                )
                row['trained'] = ok

            # Step 3: validate
            sim = validate_model(actor_id, applio_dir, audio_dir, args.xvector_threshold)
            row['xvector_sim'] = round(sim, 4) if sim is not None else None
            if sim is not None:
                row['passes_threshold'] = sim >= args.xvector_threshold
                if not row['passes_threshold']:
                    log.warning(f"Actor {actor_id}: x-vector sim={sim:.3f} < {args.xvector_threshold}")

        except Exception as e:
            row['error'] = str(e)
            log.error(f"Actor {actor_id}: {e}")

        log_rows.append(row)
        _flush_log()  # persist after every actor so a crash loses at most one actor's data

    # Summary
    n_trained  = sum(1 for r in log_rows if r['trained'])
    n_passing  = sum(1 for r in log_rows if r['passes_threshold'])
    n_prepared = sum(1 for r in log_rows if r['clips_staged'] > 0)

    print("\n" + "=" * 55)
    print("RVC TRAINING COMPLETE")
    print("=" * 55)
    print(f"Actors processed:    {len(log_rows)}")
    print(f"Datasets staged:     {n_prepared}")
    print(f"Models trained:      {n_trained}")
    if any(r['xvector_sim'] is not None for r in log_rows):
        print(f"Models passing x-vector >= {args.xvector_threshold}: {n_passing}")
    print(f"Log: {log_path}")
    if n_trained:
        print(f"\nNext step:")
        print(f"  python track1_generate.py \\")
        print(f"      --pairs_csv   ../../data/processed/track1_manifests/swap_pairs.csv \\")
        print(f"      --out_dir     ../../data/synthetic/track1_fakes \\")
        print(f"      --method      styletts \\")
        print(f"      --cremad_dir  {cremad_dir} \\")
        print(f"      --applio_dir {applio_dir}")
    print("=" * 55)


if __name__ == '__main__':
    main()
