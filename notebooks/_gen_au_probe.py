"""
_gen_au_probe.py — generate colab_au_probe.ipynb.

A throwaway PROBE notebook that answers two questions before anyone commits days of GPU
to an AU-on re-preprocess:
  1. Can py-feat run on Colab?  (Colab pulls py-feat 2.x on Python 3.12 — the repo's
     visual.py is version-aware and handles both 2.x `Detectorv1` and 0.6.x `Detector`.)
  2. How long is AU-on z_v extraction per clip on a T4 -> real total for the full dataset.

NON-DESTRUCTIVE: only calls get_z_v() (returns a tensor, writes NOTHING) on a few sample
clips extracted to /content. It never touches any feature cache.

Run:  python notebooks/_gen_au_probe.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/gjvlio/emotion-based-multimodal-deepfake-detector.git"
BRANCH = "feat/mosei-preprocess-and-eval"   # has the AU code + version-aware visual.py
DRIVE_ROOT = "/content/drive/MyDrive/DeepSentinel_data"


def md(t):   return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def build():
    cells = [
        md(
            "# AU-saliency Probe — can Colab run it, and how long?\n\n"
            "Throwaway benchmark. Answers **(1)** does py-feat run on a Colab T4, and **(2)** how many "
            "seconds/clip AU-on `z_v` extraction takes -> real total for the full dataset.\n\n"
            "Colab installs **py-feat 2.x** on **Python 3.12**. The repo's `visual.py` is version-aware "
            "(handles 2.x `Detectorv1` + `detect()` and 0.6.x `Detector` + `detect_image()`), so no "
            "downgrade is needed — the old pinned torch 2.0.1 has no py3.12 wheel anyway.\n\n"
            "**Non-destructive:** only calls `get_z_v()` (returns a tensor, writes nothing) on a few sample "
            "clips in `/content`. It never touches any feature cache.\n\n"
            "### How to use\n"
            "1. Runtime -> Change runtime type -> **T4 GPU**.\n"
            "2. **Run all**. Read the verdict in the last cell.\n"
        ),
        md("## Cell 1 — Config"),
        code(
            "N_BENCH   = 12           # clips to benchmark\n"
            "AU_TOP_K  = 12           # AU runs on the top-K frames by conf x sharpness\n"
            f'DRIVE_ROOT        = "{DRIVE_ROOT}"\n'
            'DRIVE_SEG_ARCHIVE = DRIVE_ROOT + "/segments.zip"   # a few clips are sampled from here\n'
            f'REPO_URL    = "{REPO_URL}"\n'
            f'REPO_BRANCH = "{BRANCH}"\n\n'
            "# Full-dataset sizes for the extrapolation (clips needing AU-on z_v):\n"
            "TOTALS = [6277, 14240, 18369]   # MOSEI-only, current cache, all-sources\n"
        ),
        md("## Cell 2 — Install py-feat (2.x) + visual-pipeline deps"),
        code(
            "import subprocess, sys\n"
            "def sh(c): print('$', c); subprocess.run(c, shell=True)\n"
            "# purge any 'feat' squatter, install deps first, then real py-feat LAST\n"
            "sh('pip uninstall -y feat py-feat')\n"
            "sh('pip install -q transformers timm insightface onnxruntime-gpu opencv-python-headless librosa soundfile')\n"
            "sh('pip install -q py-feat')\n"
            "print('install done.')"
        ),
        md("## Cell 3 — Verify torch/CUDA + py-feat Detector loads"),
        code(
            "import torch, feat\n"
            "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(),\n"
            "      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')\n"
            "# py-feat 2.0 renamed Detector -> Detectorv1; accept either.\n"
            "Detector = getattr(feat, 'Detector', None) or getattr(feat, 'Detectorv1', None)\n"
            "assert Detector is not None, 'no Detector/Detectorv1 in py-feat — install failed'\n"
            "print('py-feat', getattr(feat, '__version__', '?'), '-> using', Detector.__name__)\n"
            "try:\n"
            "    _ = Detector(au_model='xgb', device=dev)\n"
            "    PYFEAT_WORKS = True\n"
            "    print('py-feat OK — Detector loaded on', dev, '=> this Colab CAN run AU-on.')\n"
            "except Exception as e:\n"
            "    PYFEAT_WORKS = False\n"
            "    import traceback; traceback.print_exc()\n"
            "assert PYFEAT_WORKS, 'Detector failed to instantiate — see traceback.'"
        ),
        md("## Cell 4 — Mount Drive, clone/pull repo, grab a few sample clips"),
        code(
            "from google.colab import drive\n"
            "import os, zipfile\n"
            "from pathlib import Path\n"
            "drive.mount('/content/drive')\n"
            "assert os.path.exists(DRIVE_SEG_ARCHIVE), f'segments.zip not found at {DRIVE_SEG_ARCHIVE}'\n"
            "REPO_DIR = '/content/thesis'\n"
            "if os.path.exists(REPO_DIR):\n"
            "    subprocess.run(['git','-C',REPO_DIR,'pull'], check=True)  # get the latest version-aware visual.py\n"
            "else:\n"
            "    subprocess.run(['git','clone','--branch',REPO_BRANCH,'--depth','1',REPO_URL,REPO_DIR], check=True)\n"
            "os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)\n"
            "SAMPLE_DIR = '/content/au_sample'; os.makedirs(SAMPLE_DIR, exist_ok=True)\n"
            "with zipfile.ZipFile(DRIVE_SEG_ARCHIVE) as z:\n"
            "    names = [n for n in z.namelist() if n.lower().endswith('.mp4')][:N_BENCH]\n"
            "    for n in names: z.extract(n, SAMPLE_DIR)\n"
            "clips = sorted(Path(SAMPLE_DIR).rglob('*.mp4'))\n"
            "print(f'{len(clips)} sample clips ready; repo @ {REPO_BRANCH}')"
        ),
        md(
            "## Cell 5 — Benchmark AU-on `z_v` (non-destructive)\n\n"
            "Calls `get_z_v(..., AU-on)` on each clip and times it. First clip is a warm-up "
            "(model downloads + first-call overhead) and is excluded from the average."
        ),
        code(
            "import time, statistics as st\n"
            "from src.preprocessing import visual\n"
            "print('py-feat API:', 'v2.x (Detectorv1)' if visual._PYFEAT_V2 else 'v0.6.x (Detector)')\n"
            "visual.configure_au(enabled=True, device=dev, top_k=AU_TOP_K)\n"
            "from src.preprocessing.visual import get_z_v\n\n"
            "def extract(v):\n"
            "    return get_z_v(str(v), vit_model_name='google/vit-base-patch16-224', detector='retinaface',\n"
            "                   n_keyframes=8, frame_size=224, target_fps=25.0, motion_threshold=0.3,\n"
            "                   confidence_threshold=0.7, device=dev)\n\n"
            "print('warming up (ViT + insightface + py-feat first call)...')\n"
            "_ = extract(clips[0])\n"
            "print('warm — benchmarking', len(clips)-1, 'clips:\\n')\n"
            "times = []\n"
            "for v in clips[1:]:\n"
            "    t0 = time.time(); z = extract(v); dt = time.time() - t0; times.append(dt)\n"
            "    print(f'  {v.name[:42]:<42} {dt:6.1f}s   z_v={tuple(z.shape)}')\n"
            "mean = st.mean(times); med = st.median(times)\n"
            "print(f'\\nAU-on z_v per clip:  mean={mean:.1f}s   median={med:.1f}s   (n={len(times)})')"
        ),
        md("## Cell 6 — Verdict: does Colab work, and how long for the full run?"),
        code(
            "print('='*64)\n"
            "print(f'  py-feat on this Colab : WORKS  (v{getattr(__import__(\"feat\"),\"__version__\",\"?\")})')\n"
            "print(f'  AU-on z_v per clip    : {med:.1f}s (median)')\n"
            "print('='*64)\n"
            "for total in TOTALS:\n"
            "    hrs = total * med / 3600\n"
            "    print(f'  {total:>6} clips  ->  {hrs:6.1f} GPU-hours  (~{hrs/24:.1f} days on ONE T4)')\n"
            "    print(f'              4-way shard  ->  ~{hrs/4:.1f}h wall-clock across 4 T4s')\n"
            "print()\n"
            "print('z_at is REUSED (AU-independent) -> the real run rebuilds ONLY z_v.')\n"
            "print('Real run (sharded on Colab OR local .venv-feat):')\n"
            "print('  preprocess_all.py --use_au --au_top_k 12 \\\\')\n"
            "print('     --out_dir data/preprocessed_au_on --reuse_zat_from data/preprocessed')\n"
            "print('  (--out_dir REQUIRED with --use_au — the AU-off baseline is never overwritten.)')"
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
    out = HERE / "colab_au_probe.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
