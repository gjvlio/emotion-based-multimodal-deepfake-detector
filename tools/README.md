# Tools Setup

The `tools/` directory contains third-party tools used by the pipeline.
They are **excluded from version control** due to their size. Follow the
instructions below to set them up before running any pipeline scripts.

---

## Wav2Lip (required for Track 2)

Wav2Lip reanimates the lip movements of a face video to sync with a target
audio file. Track 2 uses it to close the lip-sync gap left by Track 1, making
fakes significantly harder to detect.

### 1. Clone Wav2Lip

```bash
git clone https://github.com/Rudrabha/Wav2Lip.git tools/Wav2Lip
cd tools/Wav2Lip
pip install -r requirements.txt
```

### 2. Download model checkpoints

Wav2Lip requires a pretrained GAN checkpoint. Download it from the official
release and place it at `tools/Wav2Lip/checkpoints/wav2lip_gan.pth`.

```bash
mkdir -p tools/Wav2Lip/checkpoints

# Download wav2lip_gan.pth from the official Google Drive link:
# https://github.com/Rudrabha/Wav2Lip#getting-the-weights
# Place it at: tools/Wav2Lip/checkpoints/wav2lip_gan.pth
```

A face detection model is also required:

```bash
# Download s3fd face detection model:
# https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
mkdir -p tools/Wav2Lip/face_detection/detection/sfd
# Place s3fd-619a316812.pth at:
# tools/Wav2Lip/face_detection/detection/sfd/s3fd.pth
```

### 3. Apply compatibility patches

Three fixes are required to run Wav2Lip with Python 3.11 / PyTorch 2.6 / librosa ≥0.10:

---

#### Patch A — `audio.py`: librosa API change

`librosa.filters.mel()` no longer accepts positional arguments in librosa ≥0.10.

```python
# Before (line ~100):
return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                           fmin=hp.fmin, fmax=hp.fmax)

# After:
return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,
                           fmin=hp.fmin, fmax=hp.fmax)
```

---

#### Patch B — `inference.py`: PyTorch 2.6 `torch.load` default change

`torch.load()` defaults to `weights_only=True` in PyTorch 2.6, breaking old checkpoints.

```python
# Before (in _load(), ~line 161):
checkpoint = torch.load(checkpoint_path)
# and:
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

# After — add weights_only=False to both branches:
checkpoint = torch.load(checkpoint_path, weights_only=False)
# and:
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage,
                        weights_only=False)
```

---

#### Patch C — `inference.py`: `wav2lip_gan.pth` is a TorchScript archive

`wav2lip_gan.pth` is saved as a TorchScript archive, so `torch.load` dispatches
to `torch.jit.load` and `checkpoint["state_dict"]` raises `NotImplementedError`.

```python
# Before (load_model function, ~line 172):
checkpoint = _load(path)
s = checkpoint["state_dict"]

# After — add fallback for TorchScript archives:
try:
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
except (TypeError, NotImplementedError, KeyError):
    jit_model = torch.jit.load(path, map_location=device)
    s = jit_model.state_dict()
```

---

### 4. Verify setup

```bash
cd tools/Wav2Lip
python inference.py --help
```

### Directory structure after setup

```
tools/Wav2Lip/
├── inference.py                        ← entry point used by track2_generate.py
├── checkpoints/
│   └── wav2lip_gan.pth                 ← pretrained GAN model (download manually)
└── face_detection/
    └── detection/sfd/
        └── s3fd.pth                    ← face detector (download manually)
```

---

## SadTalker (required for Track 3)

SadTalker generates a complete talking head video from a single portrait image
driven by audio — lip sync and 3D head motion both synthesised. Unlike Wav2Lip
(Track 2) which reanimates an existing face video's mouth region, SadTalker
generates a **fully new face sequence** from just the portrait, making Track 3
fakes harder to detect.

Cross-platform (Windows + Linux), lighter than diffusion-based alternatives.

### 1. Clone SadTalker

```bash
git clone https://github.com/OpenTalker/SadTalker.git tools/SadTalker
cd tools/SadTalker
pip install -r requirements.txt
```

### 2. Download pretrained models

SadTalker provides a download script:

```bash
cd tools/SadTalker
# Linux/Mac:
bash scripts/download_models.sh
# Windows — run the commands inside download_models.sh manually, or:
python scripts/download_correct_model.py  # if available
```

Alternatively download manually and place in `tools/SadTalker/checkpoints/`:
- SadTalker models: ~400 MB total
- (Optional) GFPGAN face enhancer: ~340 MB — place in `tools/SadTalker/gfpgan/weights/`

See the [SadTalker README](https://github.com/OpenTalker/SadTalker#usage) for
direct download links.

### 3. Verify setup

```bash
cd tools/SadTalker
python inference.py \
  --driven_audio examples/driven_audio/bus_chinese.wav \
  --source_image examples/source_image/full_body_1.png \
  --result_dir results \
  --still --preprocess full --size 256
```

### Directory structure after setup

```
tools/SadTalker/
├── inference.py             <- entry point used by track3_generate.py
├── checkpoints/             <- pretrained SadTalker models (~400 MB)
├── gfpgan/weights/          <- optional face enhancer (~340 MB)
└── src/                     <- SadTalker source modules
```

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
