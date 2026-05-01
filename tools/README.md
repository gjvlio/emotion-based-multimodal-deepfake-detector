# Tools Setup

The `tools/` directory contains third-party tools used by the pipeline.
They are **excluded from version control** due to their size. Follow the
instructions below to set them up before running any pipeline scripts.

---

## Applio (RVC v2 — required for Track 1 Method B)

Applio provides the RVC voice-conversion CLI used to train per-actor voice
models and convert synthesised speech into the target speaker's voice.

### 1. Clone Applio

```bash
git clone https://github.com/IAHispano/Applio.git tools/Applio
cd tools/Applio
pip install -r requirements.txt
```

### 2. Download pretrained models

Applio requires several pretrained models to run. Download them by running
Applio's built-in installer or by following the model download section in
the [Applio README](https://github.com/IAHispano/Applio#readme).

Required files (paths relative to `tools/Applio/`):

| File | Purpose |
|------|---------|
| `rvc/models/predictors/rmvpe.pt` | Pitch predictor (RMVPE) |
| `rvc/models/predictors/fcpe.pt` | Pitch predictor (FCPE, fallback) |
| `rvc/models/embedders/contentvec/pytorch_model.bin` | Speaker embedding model |
| `rvc/models/pretraineds/hifi-gan/` | Pretrained vocoder |

### 3. Apply Windows compatibility patches

We made the following changes to Applio's source code for Windows /
single-GPU compatibility. Apply them after cloning.

---

#### Patch A — `rvc/train/train.py`: single-GPU Windows fix

**Problem:** On Windows with a single GPU, Applio spawns a child process via
`mp.Process`, which re-loads all CUDA DLLs and exhausts the Windows paging
file. Also, the default `DataLoader` `num_workers > 0` causes deadlocks on
Windows.

**Fix — in `train.py`, `run()` function, find the DataLoader call and set:**
```python
# Before:
train_loader = DataLoader(..., num_workers=<some value>, ...)

# After:
train_loader = DataLoader(..., num_workers=0, ...)
```

**Fix — in `train.py`, `start()` function, the single-GPU Windows branch
should call `run()` directly (not via `mp.Process`). Add this block before
the multi-GPU `mp.Process` loop:**
```python
if n_gpus == 1 and sys.platform == "win32":
    pid_data = {"process_pids": [os.getpid()]}
    with open(config_save_path, "w") as pid_file:
        json.dump(pid_data, pid_file, indent=4)
    run(0, 1, experiment_dir, pretrainG, pretrainD,
        total_epoch, save_every_weights, config, device, gpus[0])
    return
```

---

#### Patch B — `rvc/train/preprocess/preprocess.py`: Windows multiprocessing fix

**Problem:** `ProcessPoolExecutor` causes a fork-related crash on Windows when
used from a non-`__main__` context.

**Fix — replace `ProcessPoolExecutor` with `ThreadPoolExecutor`:**
```python
# Before:
with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:

# After:
with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
```

---

### Directory structure after setup

```
tools/
├── README.md          ← this file
└── Applio/
    ├── core.py        ← CLI entry point used by train_rvc_voices.py
    ├── rvc/
    │   ├── models/
    │   │   ├── predictors/    ← rmvpe.pt, fcpe.pt
    │   │   ├── embedders/     ← contentvec model
    │   │   └── pretraineds/   ← hifi-gan vocoder
    │   └── train/
    │       ├── train.py       ← apply Patch A
    │       └── preprocess/
    │           └── preprocess.py  ← apply Patch B
    └── logs/          ← created during training (actor models stored here)
```
