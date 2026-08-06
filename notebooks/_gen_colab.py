"""
_gen_colab.py — generate the 4 sharded MOSEI-preprocessing Colab notebooks AND 4 parallel training Colab notebooks.

Each of 4 people opens ONE preprocessing notebook (person1..4) and ONE training notebook (person1..4).
Preprocessing is sharded, while training is run in parallel with different seeds and person-specific checkpoint paths
to prevent write collisions in the shared Google Drive folder.

Run:  python notebooks/_gen_colab.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/gjvlio/emotion-based-multimodal-deepfake-detector.git"
BRANCH = "feat/webapp-integration"
NUM_SHARDS = 4
# One shared Drive folder holds everything: the segments RAR (leader uploads once)
# and the 4 output shard zips (one per person).
DRIVE_ROOT = "/content/drive/MyDrive/DeepSentinel_data"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def shard_notebook(shard: int, person: int) -> dict:
    cells = [
        md(
            f"# MOSEI Preprocessing — **Person {person} of {NUM_SHARDS}** (shard {shard})\n\n"
            f"You process **your quarter** of the 6,277 CMU-MOSEI clips → `z_at` / `z_v` features, "
            f"then upload `mosei_features_shard{shard}.zip` to the shared Drive folder.\n\n"
            "### One-time setup (leader does this once)\n"
            f"1. Create a shared Drive folder and share it with all 4 members.\n"
            f"2. Upload `segments.zip` (the 6,277 MOSEI clips) into it.\n"
            f"3. Everyone's `DRIVE_ROOT` (Cell 1) must point to that same folder.\n\n"
            "### Each person\n"
            "1. Runtime → Change runtime type → **T4 GPU**\n"
            "2. Run all cells, top to bottom (~10-15 min for a quarter)\n"
            "3. Confirm the final cell prints **shard complete** and the zip landed on Drive\n"
        ),
        md("## Cell 1 — Config (already set for your shard — don't change SHARD_INDEX)"),
        code(
            f"# ═══════════════════════════════════════════════════════════════════\n"
            f"#  PERSON {person} — SHARD {shard} of {NUM_SHARDS}.  Do not edit SHARD_INDEX.\n"
            f"# ═══════════════════════════════════════════════════════════════════\n"
            f"SHARD_INDEX = {shard}\n"
            f"NUM_SHARDS  = {NUM_SHARDS}\n\n"
            f'DRIVE_ROOT  = "{DRIVE_ROOT}"          # SHARED folder — SAME for all 4 people\n'
            f'DRIVE_SEG_ARCHIVE = DRIVE_ROOT + "/segments.zip"  # segments.zip (or .rar) — leader uploads once\n'
            f'DRIVE_OUTPUT_DIR  = DRIVE_ROOT                    # shard zips go here\n\n'
            f'REPO_URL    = "{REPO_URL}"\n'
            f'REPO_BRANCH = "{BRANCH}"\n\n'
            f'# ── AU saliency (keyframe scoring) — PREPROCESSING LEAD decides ──────────\n'
            f'#  USE_AU=False -> conf x sharpness (DEFAULT; matches the existing AU-off cache).\n'
            f'#  USE_AU=True  -> real Action Unit saliency via py-feat.\n'
            f'#    Only flip this for a FULL AU-on re-run of ALL clips into a SEPARATE store.\n'
            f'#    NEVER mix AU-on and AU-off features in one training set (invalid).\n'
            f'#  WARNING: py-feat needs an OLD numpy/torch stack that conflicts with Colab —\n'
            f'#    AU-on is designed as a LOCAL .venv-feat job and may fail here. Cell 3 will\n'
            f'#    HARD-STOP if USE_AU=True but py-feat cannot load (prevents silent AU-off).\n'
            f'USE_AU   = False\n'
            f'AU_TOP_K = 12   # AU runs only on the top-K frames by conf x sharpness (bounds cost)\n'
        ),
        md("## Cell 2 — Mount Drive"),
        code(
            "from google.colab import drive\n"
            "import os\n"
            "drive.mount('/content/drive')\n"
            "os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)\n"
            "assert os.path.exists(DRIVE_SEG_ARCHIVE), f'segments archive not found at {DRIVE_SEG_ARCHIVE} — check DRIVE_ROOT'\n"
            "print('Drive OK. Output →', DRIVE_OUTPUT_DIR)"
        ),
        md("## Cell 3 — Install tools + deps"),
        code(
            "!apt-get update -qq && apt-get install -y -qq unrar ffmpeg\n"
            "!pip install -q openai-whisper transformers timm insightface onnxruntime-gpu librosa soundfile torchaudio\n"
            "if USE_AU:\n"
            "    print('USE_AU=True -> installing py-feat (may conflict with Colab torch/numpy)...')\n"
            "    get_ipython().system('pip install -q py-feat')\n"
            "    try:\n"
            "        from feat import Detector  # verify the REAL py-feat loads (not a squatter / broken dep)\n"
            "        print('py-feat OK — AU-on available.')\n"
            "    except Exception as e:\n"
            "        raise RuntimeError(\n"
            "            f'USE_AU=True but py-feat failed to load ({e}). AU-on would SILENTLY fall back '\n"
            "            'to conf x sharpness, producing AU-OFF features labelled as AU-on. '\n"
            "            'Run AU-on locally in .venv-feat instead, or set USE_AU=False.')\n"
            "print('Deps installed.')"
        ),
        md("## Cell 4 — Extract the segments archive (.zip or .rar)"),
        code(
            "import subprocess, zipfile\n"
            "from pathlib import Path\n"
            "SEGMENTS_DIR = '/content/mosei_segments'\n"
            "os.makedirs(SEGMENTS_DIR, exist_ok=True)\n"
            "arc = DRIVE_SEG_ARCHIVE\n"
            "print(f'Extracting {arc} ({os.path.getsize(arc)/1e9:.2f} GB)...')\n"
            "if arc.lower().endswith('.zip'):\n"
            "    with zipfile.ZipFile(arc) as z: z.extractall(SEGMENTS_DIR)\n"
            "else:\n"
            "    r = subprocess.run(['unrar','x','-y',arc,SEGMENTS_DIR+'/'], capture_output=True, text=True)\n"
            "    if r.returncode != 0:\n"
            "        print(r.stderr[-1500:]); raise RuntimeError('unrar failed')\n"
            "mp4s = list(Path(SEGMENTS_DIR).rglob('*.mp4'))\n"
            "assert mp4s, 'no MP4s after extract'\n"
            "ACTUAL_SEGMENTS_DIR = str(mp4s[0].parent)\n"
            "print(f'{len(mp4s)} segments at {ACTUAL_SEGMENTS_DIR}')"
        ),
        md("## Cell 5 — Clone repo"),
        code(
            "REPO_DIR = '/content/thesis'\n"
            "if os.path.exists(REPO_DIR):\n"
            "    subprocess.run(['git','-C',REPO_DIR,'pull'], check=True)\n"
            "else:\n"
            "    subprocess.run(['git','clone','--branch',REPO_BRANCH,'--depth','1',REPO_URL,REPO_DIR], check=True)\n"
            "os.chdir(REPO_DIR)\n"
            "import sys; sys.path.insert(0, REPO_DIR)\n"
            "print('Repo @', REPO_BRANCH, '| CWD', os.getcwd())"
        ),
        md("## Cell 6 — Build MOSEI manifest + isolate MOSEI-only"),
        code(
            "import pandas as pd\n"
            "for d in ['data/preprocessed/features/z_at','data/preprocessed/features/z_v',\n"
            "          'data/preprocessed/audio','data/preprocessed/transcripts',\n"
            "          'data/processed/mosei_manifests']:\n"
            "    Path(d).mkdir(parents=True, exist_ok=True)\n"
            "segs = sorted(Path(ACTUAL_SEGMENTS_DIR).glob('*.mp4'))\n"
            "pd.DataFrame([{'clip_id':v.stem,'video_path':str(v.resolve())} for v in segs])\\\n"
            "  .to_csv('data/processed/mosei_manifests/mosei_real.csv', index=False)\n"
            "print(f'manifest: {len(segs)} clips')\n"
            "# remove other manifests so preprocess_all.py targets MOSEI only\n"
            "for f in ['data/processed/meld_manifests/meld_real.csv','data/raw/MUStARD/repo/data/sarcasm_data.json',\n"
            "          'data/synthetic/track1_fakes/metadata.csv','data/synthetic/track2_fakes/metadata.csv',\n"
            "          'data/synthetic/track3_fakes/metadata.csv','data/synthetic/track4_fakes/metadata.csv']:\n"
            "    p = Path(f)\n"
            "    if p.exists(): p.unlink()\n"
            "print('MOSEI-only.')"
        ),
        md("## Cell 7 — Verify GPU"),
        code(
            "import torch\n"
            "print('CUDA:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')\n"
            "from src.preprocessing.pipeline import PreprocessingPipeline  # import check\n"
            "print('repo imports OK')"
        ),
        md(
            "## Cell 8 — Run preprocessing (YOUR shard only)\n\n"
            f"Processes every {NUM_SHARDS}th clip starting at index {shard}. Resume-safe: re-run if it "
            "disconnects and it skips what you already did."
        ),
        code(
            f"au = f' --use_au --au_top_k {{AU_TOP_K}}' if USE_AU else ''\n"
            f"print('AU saliency:', 'ON (py-feat)' if USE_AU else 'OFF (conf x sharpness)')\n"
            f"!python scripts/preprocess_all.py --device cuda --num_shards {NUM_SHARDS} --shard {shard}{{au}} 2>&1"
        ),
        md("## Cell 9 — Verify + upload your shard zip to Drive"),
        code(
            "import shutil, json, time\n"
            "z_at = list(Path('data/preprocessed/features/z_at').glob('*.pt'))\n"
            "z_v  = list(Path('data/preprocessed/features/z_v').glob('*.pt'))\n"
            "failed_log = Path('data/preprocessed/failed_clips.txt')\n"
            "n_failed = len(failed_log.read_text().split()) if failed_log.exists() else 0\n"
            "print(f'z_at={len(z_at)}  z_v={len(z_v)}  failed={n_failed}')\n"
            "assert z_at and z_v, 'no features produced — check Cell 8 output'\n"
            "# zip features → Drive (per-shard name, no collision with other people)\n"
            f"zip_local = '/content/mosei_features_shard{shard}'\n"
            "shutil.make_archive(zip_local, 'zip', 'data/preprocessed', 'features')\n"
            f"dst = os.path.join(DRIVE_OUTPUT_DIR, 'mosei_features_shard{shard}.zip')\n"
            "shutil.copy2(zip_local + '.zip', dst)\n"
            "status = {'person': %d, 'shard': %d, 'z_at': len(z_at), 'z_v': len(z_v),\n"
            "          'failed': n_failed, 'finished_at': time.strftime('%%Y-%%m-%%d %%H:%%M:%%S')}\n"
            "json.dump(status, open(os.path.join(DRIVE_OUTPUT_DIR, f'shard{SHARD_INDEX}_status.json'),'w'), indent=2)\n"
            "print('\\n=== SHARD %d COMPLETE ===')\n"
            "print('uploaded →', dst, f'({os.path.getsize(dst)/1e6:.0f} MB)')\n"
            "print('tell the leader your shard is done.')" % (person, shard, shard)
        ),
    ]
    return notebook(cells)


def training_notebook(person: int, seed: int) -> dict:
    cells = [
        md(
            f"# DeepSentinel — Colab Training (Phase 1 + Phase 2) — **Person {person}** (Seed {seed})\n\n"
            f"You run Phase 1 + Phase 2 training with **your seed ({seed})** on a T4 GPU. "
            f"Checkpoints and logs are saved to Drive with a `_person{person}` suffix to avoid overwriting your team members' files.\n\n"
            "### Files to upload to Google Drive before running\n\n"
            "Ensure the leader has uploaded the Phase 1 and Phase 2 data zips to your shared Drive folder.\n"
        ),
        md("## Config — customized for your run"),
        code(
            f"# ═══════════════════════════════════════════════════════════════════\n"
            f"#  PERSON {person} — SEED {seed}.  Do not edit unless changing directories.\n"
            f"# ═══════════════════════════════════════════════════════════════════\n"
            f"PERSON_ID        = {person}\n"
            f"SEED             = {seed}\n"
            f'DRIVE_DIR        = "{DRIVE_ROOT}"       # folder with all zips\n'
            f'DRIVE_OUTPUT_DIR = "{DRIVE_ROOT}/checkpoints" # output for checkpoints\n\n'
            f'REPO_URL    = "{REPO_URL}"\n'
            f'REPO_BRANCH = "{BRANCH}"\n\n'
            f'# Local Windows repo root — used to remap video paths for Phase 2\n'
            f'LOCAL_REPO_ROOT = r"D:\\Documents\\Programming\\Thesis_G10"\n'
        ),
        md("## Step 1 — Mount Drive + install deps"),
        code(
            "from google.colab import drive\n"
            "import os\n"
            "drive.mount('/content/drive')\n"
            "os.makedirs(DRIVE_DIR, exist_ok=True)\n"
            "os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)\n"
            "print(f'Drive mounted.')\n"
            "print(f'  Input  : {DRIVE_DIR}')\n"
            "print(f'  Output : {DRIVE_OUTPUT_DIR}')"
        ),
        code(
            "!pip install -q transformers scikit-learn tensorboard timm\n"
            "print('Deps installed.')"
        ),
        md("## Step 2 — Clone repo"),
        code(
            "import subprocess, os\n"
            "REPO_DIR = '/content/thesis'\n\n"
            "if os.path.exists(REPO_DIR):\n"
            "    print('Repo exists — pulling...')\n"
            "    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)\n"
            "else:\n"
            "    print(f'Cloning {REPO_URL} @ {REPO_BRANCH} ...')\n"
            "    subprocess.run(\n"
            "        ['git', 'clone', '--branch', REPO_BRANCH, '--depth', '1', REPO_URL, REPO_DIR],\n"
            "        check=True\n"
            "    )\n"
            "os.chdir(REPO_DIR)\n"
            "import sys\n"
            "sys.path.insert(0, REPO_DIR)\n"
            "print(f'CWD: {os.getcwd()}')"
        ),
        md("## Step 3 — Extract all data"),
        code(
            "import zipfile, os\n"
            "from pathlib import Path\n\n"
            "REPO_DIR = '/content/thesis'\n"
            "os.chdir(REPO_DIR)\n\n"
            "def extract_zip(zip_name, extract_to=REPO_DIR, optional=False):\n"
            "    zip_path = os.path.join(DRIVE_DIR, zip_name)\n"
            "    if not os.path.exists(zip_path):\n"
            "        tag = 'OPTIONAL — SKIP' if optional else 'MISSING — REQUIRED'\n"
            "        print(f'  [{tag}] {zip_name}')\n"
            "        return False\n"
            "    size_mb = os.path.getsize(zip_path) / 1e6\n"
            "    print(f'  Extracting {zip_name} ({size_mb:.0f} MB) ...')\n"
            "    with zipfile.ZipFile(zip_path) as zf:\n"
            "        zf.extractall(extract_to)\n"
            "    print(f'  Done.')\n"
            "    return True\n\n"
            "# Create all expected dirs\n"
            "for d in [\n"
            "    'data/preprocessed/features/z_at',\n"
            "    'data/preprocessed/features/z_v',\n"
            "    'data/preprocessed/audio',\n"
            "    'data/preprocessed/transcripts',\n"
            "    'data/preprocessed/keyframes',\n"
            "    'data/processed/meld_manifests',\n"
            "    'data/processed/mosei_manifests',\n"
            "    'data/synthetic/track1_fakes',\n"
            "    'data/synthetic/track2_fakes',\n"
            "    'data/synthetic/track3_fakes',\n"
            "    'data/raw/MUStARD/repo/data',\n"
            "    'checkpoints/full',\n"
            "    'logs/full',\n"
            "]:\n"
            "    Path(d).mkdir(parents=True, exist_ok=True)\n\n"
            "print('=== Phase 1 data ===')\n"
            "extract_zip('existing_features.zip')\n"
            "extract_zip('metadata.zip')\n"
            "# MOSEI features arrive as 4 shard zips from the parallel preprocessing notebooks\n"
            "_got = False\n"
            "for _i in range(4):\n"
            "    if extract_zip(f'mosei_features_shard{_i}.zip', extract_to='data/preprocessed', optional=True):\n"
            "        _got = True\n"
            "if not _got:\n"
            "    extract_zip('mosei_features.zip', extract_to='data/preprocessed', optional=True)\n\n"
            "print()\n"
            "print('=== Phase 2 data (skipped if not uploaded) ===')\n"
            "extract_zip('audio_cache.zip',    optional=True)\n"
            "extract_zip('transcripts.zip',    optional=True)\n"
            "# Try to extract the consolidated datasets zips first, otherwise fall back to individual zips\n"
            "if not extract_zip('tracks_1_2_3_4.zip', optional=True):\n"
            "    extract_zip('track1_clips.zip',   optional=True)\n"
            "    extract_zip('track2_clips.zip',   optional=True)\n"
            "    extract_zip('track3_clips.zip',   optional=True)\n"
            "if not extract_zip('meld_raw.zip', optional=True):\n"
            "    extract_zip('meld_real_clips.zip',optional=True)\n"
            "if not extract_zip('mustard.zip', optional=True):\n"
            "    extract_zip('mustard_clips.zip',  optional=True)\n"
            "extract_zip('fakeavceleb.zip', optional=True)"
        ),
        md("## Step 4 — Verify features"),
        code(
            "import torch\n"
            "from pathlib import Path\n\n"
            "z_at = list(Path('data/preprocessed/features/z_at').glob('*.pt'))\n"
            "z_v  = list(Path('data/preprocessed/features/z_v').glob('*.pt'))\n"
            "wavs = list(Path('data/preprocessed/audio').glob('*.wav'))\n\n"
            "print(f'z_at .pt files : {len(z_at):>6}  (expect 14,000+)')\n"
            "print(f'z_v  .pt files : {len(z_v):>6}  (expect 14,000+)')\n"
            "print(f'audio WAVs     : {len(wavs):>6}  (0 = Phase 2 audio not uploaded yet)')\n\n"
            "if not z_at:\n"
            "    raise RuntimeError('No z_at features — existing_features.zip not extracted correctly')\n\n"
            "s_at = torch.load(z_at[0], weights_only=True)\n"
            "s_v  = torch.load(z_v[0],  weights_only=True)\n"
            "print(f'\\nz_at shape: {s_at.shape}  (expect [1536])')\n"
            "print(f'z_v  shape: {s_v.shape}   (expect [768])')\n"
            "print(f'GPU: {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')\n\n"
            "# Check metadata CSVs\n"
            "for c in [\n"
            "    'data/synthetic/track1_fakes/metadata.csv',\n"
            "    'data/synthetic/track2_fakes/metadata.csv',\n"
            "    'data/synthetic/track3_fakes/metadata.csv',\n"
            "    'data/processed/meld_manifests/meld_real.csv',\n"
            "    'data/processed/mosei_manifests/mosei_real.csv',\n"
            "]:\n"
            "    p = Path(c)\n"
            "    n = sum(1 for _ in open(p)) - 1 if p.exists() else -1\n"
            "    print(f'  {p.name:<30} {\"OK \" + str(n) + \" rows\" if n >= 0 else \"MISSING\"}')"
        ),
        md("---\n# Phase 1 — Frozen Backbones"),
        code(
            "import os\n"
            "os.chdir('/content/thesis')\n\n"
            "# Run training script passing custom SEED\n"
            "!python scripts/train_full.py \\\n"
            "    --device cuda \\\n"
            "    --no_track4 \\\n"
            "    --no_phase2 \\\n"
            "    --batch_size 32 \\\n"
            "    --epochs 50 \\\n"
            "    --patience 7 \\\n"
            "    --lr 1e-3 \\\n"
            "    --seed {SEED} \\\n"
            "    2>&1"
        ),
        code(
            "import shutil, os\n"
            "from pathlib import Path\n\n"
            "ckpt = Path('checkpoints/full/best_phase1.pt')\n"
            "if not ckpt.exists():\n"
            "    raise FileNotFoundError(f'Checkpoint not found: {ckpt} — check training output above')\n\n"
            "# Save as best_phase1_person{PERSON_ID}.pt\n"
            "dst = os.path.join(DRIVE_OUTPUT_DIR, f'best_phase1_person{PERSON_ID}.pt')\n"
            "shutil.copy2(str(ckpt), dst)\n"
            "print(f'Phase 1 checkpoint saved → {dst}  ({ckpt.stat().st_size/1e6:.1f} MB)')\n\n"
            "logs = Path('logs/full')\n"
            "if logs.exists() and any(logs.iterdir()):\n"
            "    shutil.copytree(str(logs), os.path.join(DRIVE_OUTPUT_DIR, f'logs_phase1_person{PERSON_ID}'), dirs_exist_ok=True)\n"
            "    print('Logs saved to Drive.')"
        ),
        md("---\n# Phase 2 — Backbone Fine-Tuning"),
        code(
            "import os\n"
            "from pathlib import Path\n\n"
            "required = [\n"
            "    ('audio_cache.zip',     '~1.5 GB'),\n"
            "    ('transcripts.zip',     '~0.7 MB'),\n"
            "]\n\n"
            "all_ok = True\n"
            "for fname, expected_size in required:\n"
            "    p = os.path.join(DRIVE_DIR, fname)\n"
            "    if os.path.exists(p):\n"
            "        mb = os.path.getsize(p) / 1e6\n"
            "        print(f'  OK     {fname:<28} {mb:>7.0f} MB')\n"
            "    else:\n"
            "        print(f'  MISS   {fname:<28} (expected {expected_size})')\n"
            "        all_ok = False\n\n"
            "# Also check either new zips or old individual zips\n"
            "has_tracks = os.path.exists(os.path.join(DRIVE_DIR, 'tracks_1_2_3_4.zip')) or \\\n"
            "             (os.path.exists(os.path.join(DRIVE_DIR, 'track1_clips.zip')) and \\\n"
            "              os.path.exists(os.path.join(DRIVE_DIR, 'track2_clips.zip')) and \\\n"
            "              os.path.exists(os.path.join(DRIVE_DIR, 'track3_clips.zip')))\n"
            "has_meld = os.path.exists(os.path.join(DRIVE_DIR, 'meld_raw.zip')) or \\\n"
            "           os.path.exists(os.path.join(DRIVE_DIR, 'meld_real_clips.zip'))\n"
            "has_mustard = os.path.exists(os.path.join(DRIVE_DIR, 'mustard.zip')) or \\\n"
            "              os.path.exists(os.path.join(DRIVE_DIR, 'mustard_clips.zip'))\n\n"
            "if not has_tracks: print('  MISS   tracks zip(s)'); all_ok = False\n"
            "else: print('  OK     tracks present')\n"
            "if not has_meld: print('  MISS   meld zip(s)'); all_ok = False\n"
            "else: print('  OK     meld present')\n"
            "if not has_mustard: print('  MISS   mustard zip(s)'); all_ok = False\n"
            "else: print('  OK     mustard present')\n\n"
            "if all_ok:\n"
            "    print('\\nAll Phase 2 files present. Run next cells.')\n"
            "else:\n"
            "    print('\\nMissing files — upload to Drive first, or run Phase 2 locally.')"
        ),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n\n"
            "COLAB_ROOT = '/content/thesis'\n"
            "LOCAL_ROOT = LOCAL_REPO_ROOT.replace('\\\\', '/')\n\n"
            "def remap_csv(csv_path, path_cols):\n"
            "    p = Path(csv_path)\n"
            "    if not p.exists():\n"
            "        print(f'  SKIP (not found): {csv_path}')\n"
            "        return\n"
            "    df = pd.read_csv(p)\n"
            "    changed = 0\n"
            "    for col in path_cols:\n"
            "        if col not in df.columns:\n"
            "            continue\n"
            "        def fix(v):\n"
            "            if not isinstance(v, str): return v\n"
            "            v = v.replace('\\\\', '/')\n"
            "            return v.replace(LOCAL_ROOT, COLAB_ROOT) if LOCAL_ROOT in v else v\n"
            "        before = df[col].copy()\n"
            "        df[col] = df[col].map(fix)\n"
            "        changed += (df[col] != before).sum()\n"
            "    df.to_csv(p, index=False)\n"
            "    print(f'  {p.name:<35} {changed} paths remapped')\n\n"
            "print('Remapping video paths...')\n"
            "remap_csv('data/synthetic/track1_fakes/metadata.csv', ['output_path', 'input_path'])\n"
            "remap_csv('data/synthetic/track2_fakes/metadata.csv', ['output_path', 'input_path'])\n"
            "remap_csv('data/synthetic/track3_fakes/metadata.csv', ['output_path', 'input_path'])\n"
            "remap_csv('data/processed/meld_manifests/meld_real.csv',  ['video_path'])\n"
            "remap_csv('data/processed/mosei_manifests/mosei_real.csv', ['video_path'])\n"
            "print('Done.')"
        ),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n\n"
            "df = pd.read_csv('data/synthetic/track1_fakes/metadata.csv')\n"
            "sample = df['output_path'].iloc[0]\n"
            "print(f'Sample remapped path: {sample}')\n"
            "ok = Path(sample).exists()\n"
            "print(f'File exists: {ok}')\n\n"
            "if not ok:\n"
            "    print('\\nFix: Check that:')\n"
            "    print(f'  1. tracks_1_2_3_4.zip or track1_clips.zip was extracted')\n"
            "    print(f'  2. LOCAL_REPO_ROOT matches your Windows repo root (current: {LOCAL_REPO_ROOT})')"
        ),
        code(
            "import os, sys, torch\n"
            "from pathlib import Path\n"
            "from torch.utils.data import DataLoader\n\n"
            "os.chdir('/content/thesis')\n"
            "sys.path.insert(0, '/content/thesis')\n\n"
            "from src.models.detection_model import DeepfakeDetector\n"
            "from src.training.trainer import Trainer\n"
            "from src.utils.config import Config\n"
            "from scripts.train_full import build_datasets\n\n"
            "DEVICE = 'cuda'\n"
            "PREPROCESSED_DIR   = Path('data/preprocessed')\n"
            "KEYFRAME_CACHE_DIR = Path('data/preprocessed/keyframes')\n"
            "CKPT_DIR           = Path('checkpoints/full')\n"
            "LOG_DIR            = Path('logs/full')\n\n"
            "cfg = Config.from_yaml()\n\n"
            "# Build datasets (speaker-stratified split) using custom SEED\n"
            "train_ds, val_ds, test_ds, stats, Phase2FullDataset, p2_train_recs = build_datasets(\n"
            "    cfg, seed=SEED, include_track4=False\n"
            ")\n"
            "print(f'Train: {stats[\"train\"]}  Val: {stats[\"val\"]}  Test: {stats[\"test\"]}')\n"
            "print(f'Real: {stats[\"real_train\"]}  Fake: {stats[\"fake_train\"]}  Sarc: {stats[\"sarc_train\"]}')\n\n"
            "p2_ds     = Phase2FullDataset(p2_train_recs, PREPROCESSED_DIR, KEYFRAME_CACHE_DIR)\n"
            "p2_loader = DataLoader(p2_ds, shuffle=True, batch_size=2, num_workers=2, pin_memory=True)\n"
            "val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)\n\n"
            "model    = DeepfakeDetector().to(DEVICE)\n"
            "ckpt_src = CKPT_DIR / 'best_phase1.pt'\n\n"
            "# Copy Phase 1 weights for this specific person\n"
            "import shutil\n"
            "drive_ckpt = os.path.join(DRIVE_OUTPUT_DIR, f'best_phase1_person{PERSON_ID}.pt')\n"
            "if os.path.exists(drive_ckpt):\n"
            "    shutil.copy2(drive_ckpt, str(ckpt_src))\n"
            "    print('Copied Phase 1 checkpoint from Drive.')\n"
            "else:\n"
            "    if ckpt_src.exists():\n"
            "        print('Using existing local best_phase1.pt.')\n"
            "    else:\n"
            "        raise FileNotFoundError(f'Phase 1 checkpoint best_phase1_person{PERSON_ID}.pt not found.')\n\n"
            "ckpt_data = torch.load(str(ckpt_src), map_location=DEVICE, weights_only=True)\n"
            "model.load_state_dict(ckpt_data['model_state'], strict=False)\n"
            "print(f'Loaded Phase 1 weights (epoch={ckpt_data[\"epoch\"]}, val_loss={ckpt_data[\"val_loss\"]:.4f})')\n\n"
            "trainer = Trainer(\n"
            "    model          = model,\n"
            "    train_loader   = p2_loader,\n"
            "    val_loader     = val_loader,\n"
            "    checkpoint_dir = CKPT_DIR,\n"
            "    log_dir        = LOG_DIR,\n"
            "    fp16           = True,\n"
            "    lambda_a       = 0.5,\n"
            "    lambda_b       = 0.5,\n"
            "    lambda_sarcasm = 0.3,\n"
            "    pos_weight     = stats['auto_pos_weight'],\n"
            "    device         = DEVICE,\n"
            ")\n\n"
            "trainer.train_phase2(\n"
            "    lr            = 1e-5,\n"
            "    weight_decay  = 1e-4,\n"
            "    max_epochs    = 5,\n"
            "    patience      = 5,\n"
            "    freeze_layers = 2,\n"
            "    grad_ckpt     = True,\n"
            ")\n"
            "print('Phase 2 complete.')"
        ),
        code(
            "import shutil, os\n"
            "from pathlib import Path\n\n"
            "for fname in ['best_phase2.pt', 'best_phase1.pt']:\n"
            "    src = Path(f'checkpoints/full/{fname}')\n"
            "    if src.exists():\n"
            "        base_name = fname.replace('.pt', '')\n"
            "        dst = os.path.join(DRIVE_OUTPUT_DIR, f'{base_name}_person{PERSON_ID}.pt')\n"
            "        shutil.copy2(str(src), dst)\n"
            "        print(f'Saved {fname} → Drive as {Path(dst).name}  ({src.stat().st_size/1e6:.1f} MB)')\n\n"
            "# Keyframe cache\n"
            "kf_dir   = Path('data/preprocessed/keyframes')\n"
            "kf_files = list(kf_dir.glob('*.pt'))\n"
            "if kf_files:\n"
            "    kf_mb = sum(f.stat().st_size for f in kf_files) / 1e6\n"
            "    print(f'\\nKeyframe cache: {len(kf_files)} files, {kf_mb:.0f} MB')\n"
            "    if kf_mb < 5000:\n"
            "        shutil.make_archive(f'/content/keyframe_cache_person{PERSON_ID}', 'zip', kf_dir.parent, kf_dir.name)\n"
            "        shutil.copy2(f'/content/keyframe_cache_person{PERSON_ID}.zip', os.path.join(DRIVE_OUTPUT_DIR, f'keyframe_cache_person{PERSON_ID}.zip'))\n"
            "        print('Keyframe cache saved.')\n\n"
            "# Logs\n"
            "logs = Path('logs/full')\n"
            "if logs.exists():\n"
            "    shutil.copytree(str(logs), os.path.join(DRIVE_OUTPUT_DIR, f'logs_phase2_person{PERSON_ID}'), dirs_exist_ok=True)\n"
            "    print('Logs saved.')"
        ),
        md(
            "## After training — local steps\n\n"
            "Download from Drive and place in repo:\n"
            "```\n"
            f"best_phase1_person{person}.pt  →  checkpoints/full/best_phase1.pt\n"
            f"best_phase2_person{person}.pt  →  checkpoints/full/best_phase2.pt\n"
            "```\n\n"
            "Evaluate on FakeAVCeleb:\n"
            "```powershell\n"
            "python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/full/best_phase2.pt\n"
            "```"
        )
    ]
    return notebook(cells)


def main():
    # Preprocessing notebooks
    for shard in range(NUM_SHARDS):
        person = shard + 1
        out = HERE / f"colab_preprocess_person{person}.ipynb"
        out.write_text(json.dumps(shard_notebook(shard, person), indent=1), encoding="utf-8")
        print(f"wrote {out.name}  (shard {shard})")

    # Training notebooks (parallel running with different seeds: 42, 43, 44, 45)
    for i in range(NUM_SHARDS):
        person = i + 1
        seed = 42 + i
        out = HERE / f"colab_training_person{person}.ipynb"
        out.write_text(json.dumps(training_notebook(person, seed), indent=1), encoding="utf-8")
        print(f"wrote {out.name}  (person {person}, seed {seed})")


if __name__ == "__main__":
    main()
