"""
_gen_el_shard2.py — generate colab_preprocess_el.ipynb.

Shard-2 MOSEI preprocessing (AU-off completion pass), parallelizable across N accounts.
Each account edits ACCOUNT_ID + NUM_ACCOUNTS; the strided sub-partition + per-account
progress file + shared feature folder (disjoint clip filenames) => zero collision.

Every action cell re-derives its paths from disk, so running top-to-bottom never
NameErrors and any cell survives a re-run as long as the setup cells ran this session.

Run:  python notebooks/_gen_el_shard2.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/gjvlio/emotion-based-multimodal-deepfake-detector.git"
BRANCH = "feat/webapp-integration"
DRIVE_ROOT = "/content/drive/MyDrive/THESIS_MOTHERFILE"


def md(t):   return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def build():
    cells = [
        md(
            "# MOSEI Preprocessing — **Shard 2**, parallelized across accounts\n\n"
            "AU-off completion pass for shard 2's remaining clips. Multiple Google accounts run "
            "**this same notebook at once**, each on a disjoint slice — no collision ever.\n\n"
            "### Setup (once)\n"
            "1. Add a shortcut to the shared **THESIS_MOTHERFILE** Drive folder (right-click the shared "
            "link -> Add shortcut to Drive -> place in My Drive root).\n"
            "2. The leader must have uploaded `segments.zip` **and** `mosei_real.csv` (the one with the "
            "`processed` column, from `metadata.zip`) into that folder.\n\n"
            "### Each account\n"
            "1. **File -> Save a copy in Drive** (so your `ACCOUNT_ID` edit doesn't clobber others).\n"
            "2. Runtime -> Change runtime type -> **T4 GPU**.\n"
            "3. In **Cell 1**, set `NUM_ACCOUNTS` (how many accounts total) and `ACCOUNT_ID` (yours, 0..N-1).\n"
            "4. **Runtime -> Run all**. Resume-safe: if it disconnects, just Run all again.\n"
        ),
        md("## Cell 1 — Config  (set NUM_ACCOUNTS + ACCOUNT_ID)"),
        code(
            "# ═══════════════════════════════════════════════════════════════════\n"
            "#  SHARD 2 of 4.  Split across accounts: edit the two lines below.\n"
            "# ═══════════════════════════════════════════════════════════════════\n"
            "NUM_ACCOUNTS = 3      # how many accounts are working shard 2 (SAME on all of them)\n"
            "ACCOUNT_ID   = 0      # THIS account's id: 0 .. NUM_ACCOUNTS-1  (each account sets its own)\n\n"
            "SHARD_INDEX  = 2      # fixed — this notebook is shard 2\n"
            "NUM_SHARDS   = 4\n"
            "BATCH_SIZE   = 100    # clips per checkpoint-to-Drive batch\n\n"
            f'DRIVE_ROOT        = "{DRIVE_ROOT}"\n'
            'DRIVE_SEG_ARCHIVE = DRIVE_ROOT + "/segments.zip"\n'
            'DRIVE_STATUS_CSV  = DRIVE_ROOT + "/mosei_real.csv"   # MUST have the \'processed\' column (from metadata.zip)\n'
            'DRIVE_OUTPUT_DIR  = DRIVE_ROOT\n\n'
            f'REPO_URL    = "{REPO_URL}"\n'
            f'REPO_BRANCH = "{BRANCH}"\n\n'
            "assert 0 <= ACCOUNT_ID < NUM_ACCOUNTS, 'ACCOUNT_ID must be in 0 .. NUM_ACCOUNTS-1'\n"
            "print(f'shard {SHARD_INDEX} | account {ACCOUNT_ID}/{NUM_ACCOUNTS} | batch {BATCH_SIZE}')"
        ),
        md("## Cell 2 — Mount Drive + install deps"),
        code(
            "from google.colab import drive\n"
            "import os, subprocess, sys\n"
            "drive.mount('/content/drive')\n"
            "assert os.path.exists(DRIVE_SEG_ARCHIVE), f'segments.zip not found at {DRIVE_SEG_ARCHIVE} — check the Drive shortcut'\n"
            "assert os.path.exists(DRIVE_STATUS_CSV), f'mosei_real.csv not found at {DRIVE_STATUS_CSV} — leader uploads it (with the processed column)'\n"
            "os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)\n"
            "get_ipython().system('apt-get update -qq && apt-get install -y -qq unrar ffmpeg')\n"
            "get_ipython().system('pip install -q openai-whisper transformers timm insightface onnxruntime-gpu librosa soundfile torchaudio pandas')\n"
            "print('Drive OK + deps installed.')"
        ),
        md("## Cell 3 — Extract segments (skips if already extracted)"),
        code(
            "import zipfile\n"
            "from pathlib import Path\n"
            "SEGMENTS_DIR = '/content/mosei_segments'\n"
            "os.makedirs(SEGMENTS_DIR, exist_ok=True)\n"
            "if not list(Path(SEGMENTS_DIR).rglob('*.mp4')):\n"
            "    print(f'Extracting {DRIVE_SEG_ARCHIVE} ({os.path.getsize(DRIVE_SEG_ARCHIVE)/1e9:.2f} GB)...')\n"
            "    if DRIVE_SEG_ARCHIVE.lower().endswith('.zip'):\n"
            "        with zipfile.ZipFile(DRIVE_SEG_ARCHIVE) as z: z.extractall(SEGMENTS_DIR)\n"
            "    else:\n"
            "        r = subprocess.run(['unrar','x','-y',DRIVE_SEG_ARCHIVE,SEGMENTS_DIR+'/'], capture_output=True, text=True)\n"
            "        if r.returncode != 0: print(r.stderr[-1500:]); raise RuntimeError('unrar failed')\n"
            "mp4s = list(Path(SEGMENTS_DIR).rglob('*.mp4'))\n"
            "assert mp4s, 'no MP4s after extract'\n"
            "print(f'{len(mp4s)} segments ready at {mp4s[0].parent}')"
        ),
        md("## Cell 4 — Clone repo"),
        code(
            "REPO_DIR = '/content/thesis'\n"
            "if os.path.exists(REPO_DIR):\n"
            "    subprocess.run(['git','-C',REPO_DIR,'pull'], check=True)\n"
            "else:\n"
            "    subprocess.run(['git','clone','--branch',REPO_BRANCH,'--depth','1',REPO_URL,REPO_DIR], check=True)\n"
            "os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)\n"
            "print('repo @', REPO_BRANCH, '| cwd', os.getcwd())"
        ),
        md("## Cell 5 — Verify GPU"),
        code(
            "import torch\n"
            "print('CUDA:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')\n"
            "from src.preprocessing.pipeline import PreprocessingPipeline\n"
            "print('repo imports OK')"
        ),
        md(
            "## Cell 6 — Status check (self-contained; run anytime)\n\n"
            "Reads the ground truth (`.pt` files on Drive) — no dependency on the other cells' variables. "
            "Shows shard-2 progress across ALL accounts and this account's slice."
        ),
        code(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "_seg = {p.stem for p in Path('/content/mosei_segments').rglob('*.mp4')}\n"
            "_df  = pd.read_csv(DRIVE_STATUS_CSV)\n"
            "assert 'processed' in _df.columns, \"mosei_real.csv has no 'processed' column — upload the one from metadata.zip\"\n"
            "_rem = _df[_df['processed'].astype(str).str.lower()=='no'].sort_values('clip_id').reset_index(drop=True)\n"
            "_rem = _rem[_rem['clip_id'].isin(_seg)].reset_index(drop=True)\n"
            "_my_shard = list(_rem.iloc[SHARD_INDEX::NUM_SHARDS]['clip_id'])\n"
            "_my_work  = _my_shard[ACCOUNT_ID::NUM_ACCOUNTS]\n"
            "_featz = Path(DRIVE_OUTPUT_DIR)/f'shard{SHARD_INDEX}_features'/'z_at'\n"
            "_done  = {p.stem for p in _featz.glob('*.pt')} if _featz.exists() else set()\n"
            "_sd = sum(c in _done for c in _my_shard); _ad = sum(c in _done for c in _my_work)\n"
            "print(f'SHARD {SHARD_INDEX} (all accounts): {len(_my_shard)} clips | {_sd} done | {len(_my_shard)-_sd} pending')\n"
            "print(f'ACCOUNT {ACCOUNT_ID}/{NUM_ACCOUNTS}   : {len(_my_work)} clips | {_ad} done | {len(_my_work)-_ad} pending')\n"
            "_pend = len(_my_work)-_ad\n"
            "print(f'  -> {(_pend+BATCH_SIZE-1)//BATCH_SIZE} batches left for this account (BATCH_SIZE={BATCH_SIZE})')"
        ),
        md(
            "## Cell 7 — Run preprocessing for THIS account (batched, syncs to Drive each batch)\n\n"
            "Processes this account's disjoint slice of shard-2's remaining clips, in batches. After each "
            "batch, the new `.pt` files are copied to the shared `shard2_features/` on Drive and a "
            "per-account progress file is updated. Disconnect-safe: re-run and it skips anything already on Drive."
        ),
        code(
            "import os, json, shutil, time\n"
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "# re-derive paths from disk so this cell never depends on earlier cells' variables\n"
            "_seg_paths = list(Path('/content/mosei_segments').rglob('*.mp4'))\n"
            "assert _seg_paths, 'segments not extracted — run Cell 3'\n"
            "ACTUAL_SEGMENTS_DIR = str(_seg_paths[0].parent)\n"
            "assert os.path.exists('/content/thesis'), 'repo not cloned — run Cell 4'\n"
            "os.chdir('/content/thesis')\n\n"
            "# scope preprocess_all.py to MOSEI only\n"
            "for f in ['data/processed/meld_manifests/meld_real.csv','data/raw/MUStARD/repo/data/sarcasm_data.json',\n"
            "          'data/synthetic/track1_fakes/metadata.csv','data/synthetic/track2_fakes/metadata.csv',\n"
            "          'data/synthetic/track3_fakes/metadata.csv','data/synthetic/track4_fakes/metadata.csv',\n"
            "          'data/processed/smoke_manifests/smoke_sarcasm.csv','data/processed/smoke_manifests/smoke_detector.csv']:\n"
            "    p = Path(f)\n"
            "    if p.exists(): p.unlink()\n"
            "for d in ['data/preprocessed/features/z_at','data/preprocessed/features/z_v',\n"
            "          'data/preprocessed/audio','data/preprocessed/transcripts','data/processed/mosei_manifests']:\n"
            "    Path(d).mkdir(parents=True, exist_ok=True)\n\n"
            "# this account's disjoint slice of shard-2's remaining clips\n"
            "ext = {p.stem for p in _seg_paths}\n"
            "df  = pd.read_csv(DRIVE_STATUS_CSV)\n"
            "rem = df[df['processed'].astype(str).str.lower()=='no'].sort_values('clip_id').reset_index(drop=True)\n"
            "rem = rem[rem['clip_id'].isin(ext)].reset_index(drop=True)\n"
            "my_shard = rem.iloc[SHARD_INDEX::NUM_SHARDS].reset_index(drop=True)\n"
            "my_work  = my_shard.iloc[ACCOUNT_ID::NUM_ACCOUNTS].reset_index(drop=True)\n\n"
            "# shared feature dir (disjoint clip filenames -> no collision); resume off .pt already on Drive\n"
            "FEAT = os.path.join(DRIVE_OUTPUT_DIR, f'shard{SHARD_INDEX}_features')\n"
            "os.makedirs(os.path.join(FEAT,'z_at'), exist_ok=True); os.makedirs(os.path.join(FEAT,'z_v'), exist_ok=True)\n"
            "done = {p.stem for p in (Path(FEAT)/'z_at').glob('*.pt')}\n"
            "my_work = my_work[~my_work['clip_id'].isin(done)].reset_index(drop=True)\n"
            "my_work['video_path'] = my_work['clip_id'].apply(lambda c: str((Path(ACTUAL_SEGMENTS_DIR)/f'{c}.mp4').resolve()))\n"
            "print(f'account {ACCOUNT_ID}/{NUM_ACCOUNTS}: {len(my_work)} clips to do this session')\n\n"
            "PROG = os.path.join(DRIVE_OUTPUT_DIR, f'shard{SHARD_INDEX}_acc{ACCOUNT_ID}_progress.json')\n"
            "n_batches = (len(my_work) + BATCH_SIZE - 1)//BATCH_SIZE\n"
            "for i in range(0, len(my_work), BATCH_SIZE):\n"
            "    batch = my_work.iloc[i:i+BATCH_SIZE]\n"
            "    print(f'\\n=== account {ACCOUNT_ID} batch {i//BATCH_SIZE+1}/{n_batches} ({len(batch)} clips) ===')\n"
            "    batch.to_csv('data/processed/mosei_manifests/mosei_real.csv', index=False)\n"
            "    get_ipython().system('python scripts/preprocess_all.py --device cuda 2>&1')\n"
            "    for c in batch['clip_id']:\n"
            "        za = Path('data/preprocessed/features/z_at')/f'{c}.pt'; zv = Path('data/preprocessed/features/z_v')/f'{c}.pt'\n"
            "        if za.exists() and zv.exists():\n"
            "            shutil.copy2(za, Path(FEAT)/'z_at'/za.name); shutil.copy2(zv, Path(FEAT)/'z_v'/zv.name); done.add(c)\n"
            "    json.dump({'shard':SHARD_INDEX,'account':ACCOUNT_ID,'done':sorted(done),'at':time.strftime('%Y-%m-%d %H:%M:%S')},\n"
            "              open(PROG,'w'), indent=2)\n"
            "    print(f'synced -> {len(done)} clips now on Drive for shard {SHARD_INDEX}')\n"
            "print(f'\\naccount {ACCOUNT_ID} finished this session.')"
        ),
        md("## Cell 8 — Final shard-2 status (across ALL accounts) + zip if complete"),
        code(
            "import os, json, shutil, time\n"
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "_featz = Path(DRIVE_OUTPUT_DIR)/f'shard{SHARD_INDEX}_features'/'z_at'\n"
            "_done  = {p.stem for p in _featz.glob('*.pt')} if _featz.exists() else set()\n"
            "_seg   = {p.stem for p in Path('/content/mosei_segments').rglob('*.mp4')}\n"
            "_df    = pd.read_csv(DRIVE_STATUS_CSV)\n"
            "_rem   = _df[_df['processed'].astype(str).str.lower()=='no'].sort_values('clip_id')\n"
            "_rem   = _rem[_rem['clip_id'].isin(_seg)]\n"
            "_my_shard = list(_rem.iloc[SHARD_INDEX::NUM_SHARDS]['clip_id'])\n"
            "_sd = sum(c in _done for c in _my_shard)\n"
            "status = {'shard':SHARD_INDEX,'num_accounts':NUM_ACCOUNTS,'shard_done':_sd,'shard_total':len(_my_shard),\n"
            "          'complete':_sd>=len(_my_shard) and len(_my_shard)>0,'at':time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "json.dump(status, open(os.path.join(DRIVE_OUTPUT_DIR, f'shard{SHARD_INDEX}_status.json'),'w'), indent=2)\n"
            "print(f'SHARD {SHARD_INDEX} overall: {_sd}/{len(_my_shard)} clips done across all accounts')\n"
            "if status['complete']:\n"
            "    zl = f'/content/mosei_features_shard{SHARD_INDEX}'\n"
            "    shutil.make_archive(zl, 'zip', os.path.join(DRIVE_OUTPUT_DIR, f'shard{SHARD_INDEX}_features'))\n"
            "    dst = os.path.join(DRIVE_OUTPUT_DIR, f'mosei_features_shard{SHARD_INDEX}.zip')\n"
            "    shutil.copy2(zl + '.zip', dst)\n"
            "    print(f'\\n=== SHARD {SHARD_INDEX} COMPLETE (all accounts) ===  zipped -> {dst}')\n"
            "else:\n"
            "    print(f'\\n{len(_my_shard)-_sd} clips still pending across accounts — re-run the accounts that have work left.')\n"
            "print(status)"
        ),
    ]
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


def main():
    out = HERE / "colab_preprocess_el.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
