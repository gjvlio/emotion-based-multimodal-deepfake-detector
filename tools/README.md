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

## Hallo (required for Track 3)

Hallo is a diffusion-based talking head model from Fudan University. Track 3
uses it to generate fully synthesised face videos with emotion-driven lip sync,
head motion, and facial expression from a single portrait + audio file.

### 1. Clone Hallo

```bash
git clone https://github.com/fudan-generative-vision/hallo.git tools/Hallo
```

### 2. Install Python dependencies

Hallo requires `insightface` which needs Microsoft C++ Build Tools on Windows.
Install the hallo package only (without build-failing deps):

```bash
pip install -e tools/Hallo --no-deps
pip install diffusers transformers omegaconf einops accelerate face_alignment
pip install imageio-ffmpeg decord av librosa
```

### 3. Download pretrained models

Models are stored via git-lfs in HuggingFace. Run in the Hallo directory:

```bash
cd tools/Hallo
git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models
```

If git-lfs isn't installed (`git lfs version`), install it first:
- Windows: https://git-lfs.com
- Then re-run the clone or `git lfs pull` inside the pretrained_models directory

Total model size: ~12 GB across 12 LFS files.

### 4. Extract component weights (required for per-actor fine-tuning)

`train_stage2.py` expects individual `.pth` files, but HuggingFace ships a
single `net.pth`. Run the splitter once:

```bash
python src/track3/extract_hallo_components.py \
  --net_pth tools/Hallo/pretrained_models/hallo/net.pth \
  --out_dir tools/Hallo/pretrained_models/hallo
```

This writes `reference_unet.pth`, `denoising_unet.pth`, `face_locator.pth`,
`imageproj.pth`, and `audioproj.pth` into `pretrained_models/hallo/`.

### 5. Verify inference

```bash
cd tools/Hallo
python scripts/inference.py \
  --config configs/inference/default.yaml \
  --source_image examples/reference_images/1.jpg \
  --driving_audio examples/driving_audios/1.wav \
  --output .cache/test_output.mp4
```

### Directory structure after setup

```
tools/Hallo/
├── hallo/                  <- Python package (pip install -e . --no-deps)
├── scripts/
│   ├── inference.py        <- entry point used by track3_generate.py
│   ├── train_stage2.py     <- entry point used by finetune_hallo.py
│   └── data_preprocess.py  <- data pipeline for fine-tuning
├── configs/
│   ├── inference/default.yaml
│   └── train/stage2.yaml
└── pretrained_models/      <- downloaded via git-lfs (12 GB)
    ├── hallo/
    │   ├── net.pth                     <- combined checkpoint (4.9 GB)
    │   ├── reference_unet.pth          <- extracted by extract_hallo_components.py
    │   ├── denoising_unet.pth          <- extracted
    │   ├── face_locator.pth            <- extracted
    │   ├── imageproj.pth               <- extracted
    │   └── audioproj.pth               <- extracted
    ├── face_analysis/models/           <- InsightFace ONNX models
    ├── motion_module/mm_sd_v15_v2.ckpt <- AnimateDiff motion module (1.8 GB)
    ├── sd-vae-ft-mse/                  <- Stable Diffusion VAE (335 MB)
    ├── stable-diffusion-v1-5/unet/    <- SD1.5 UNet (3.4 GB)
    ├── wav2vec/wav2vec2-base-960h/    <- Audio encoder (378 MB)
    └── audio_separator/Kim_Vocal_2.onnx <- Vocal separator (67 MB)
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
