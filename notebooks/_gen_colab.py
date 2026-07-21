"""
_gen_colab.py — generate the 4 sharded MOSEI-preprocessing Colab notebooks.

Each of 4 people opens ONE notebook (person1..4), which processes its shard of the
6,277 CMU-MOSEI segments and drops `mosei_features_shard{i}.zip` into a SHARED Drive
folder. The training notebook then merges all four.

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


def main():
    for shard in range(NUM_SHARDS):
        person = shard + 1
        out = HERE / f"colab_preprocess_person{person}.ipynb"
        out.write_text(json.dumps(shard_notebook(shard, person), indent=1), encoding="utf-8")
        print(f"wrote {out.name}  (shard {shard})")


if __name__ == "__main__":
    main()
