# Multimodal Deepfake Generation & Detection Pipeline

**Thesis G10** — Generating a labeled dataset of audio-visual deepfakes from the CREMA-D corpus, then training and evaluating detection models against them.

Deepfakes are created by **mismatching the emotional content of audio and video**: the actor's face expresses one emotion while their voice expresses a different one. Three pipeline tracks of increasing sophistication produce the fakes, so detectors can be evaluated against both simple and complex manipulations.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Dataset: CREMA-D](#dataset-crema-d)
- [Project Structure](#project-structure)
- [Track 1 — Emotion Audio Swap](#track-1--emotion-audio-swap)
  - [Method A: Direct Swap](#method-a-direct-swap-no-ai)
  - [Method B: AI Voice Conversion](#method-b-ai-voice-conversion-styletts2--rvc)
  - [Step 1 — Parse the dataset](#step-1--parse-the-dataset)
  - [Step 2 — Train voice models](#step-2--train-voice-models-method-b-only)
  - [Step 3 — Generate fakes](#step-3--generate-fake-videos)
- [Pipeline Status](#pipeline-status)

---

## Prerequisites

**Python 3.11** is required. Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

**External tools and datasets** must be set up manually before running the pipeline — they are too large for version control:

| Dependency | Setup guide |
|------------|-------------|
| CREMA-D dataset | See [`data/raw/README.md`](data/raw/README.md) |
| Applio (RVC v2 tool) | See [`tools/README.md`](tools/README.md) |

---

## Quick Start

```bash
# 1. Download CREMA-D (see data/raw/README.md)
# 2. Clone and configure Applio (see tools/README.md)

# 3. Parse CREMA-D and build the swap-pair manifest
python src/track1/parse_cremad.py \
  --cremad_dir data/raw/CREMA-D \
  --out_dir    data/processed/track1_manifests

# 4a. Generate Method A fakes (no AI needed, ~0.5 s/clip)
cd src/track1
python track1_generate.py \
  --pairs_csv ../../data/processed/track1_manifests/test_pairs_1001_1003.csv \
  --out_dir   ../../data/synthetic/track1_fakes \
  --method    swap

# 4b. Generate Method B fakes (requires trained voice models — run step 5 first)
python track1_generate.py \
  --pairs_csv  ../../data/processed/track1_manifests/test_pairs_1001_1003.csv \
  --out_dir    ../../data/synthetic/track1_fakes \
  --method     styletts \
  --cremad_dir ../../data/raw/CREMA-D \
  --applio_dir ../../tools/Applio

# 5. (Method B only) Train RVC voice models for target actors
cd ../..
python src/track1/train_rvc_voices.py \
  --cremad_dir   data/raw/CREMA-D \
  --applio_dir   tools/Applio \
  --datasets_dir data/processed/rvc_datasets \
  --actors 1001 1002 1003
```

> **Note:** Steps 4b and 5 can be run in any order — run 5 first (training), then 4b (generation).  
> Add `--resume` to any generation command to continue an interrupted run without reprocessing completed clips.

---

## Dataset: CREMA-D

**CREMA-D** (Crowd-Sourced Emotional Multimodal Actors Dataset) contains ~7,400 short video clips of 91 actors (IDs 1001–1091). Each actor speaks one of 12 sentences using one of 6 emotions at varying intensity levels.

Download instructions: [`data/raw/README.md`](data/raw/README.md)

### Filename format

```
1001_IEO_ANG_HI.wav
│    │   │   └─ Intensity : HI (high) | MD (medium) | LO (low) | XX (unspecified)
│    │   └───── Emotion   : ANG | DIS | FEA | HAP | NEU | SAD
│    └───────── Sentence  : IEO = "It's eleven o'clock"
│                           IWW = "I want your wardrobe"
│                           (12 sentence codes total)
└──────────────  Actor ID : 1001 – 1091
```

### Emotions

| Code | Emotion  | Code | Emotion  |
|------|----------|------|----------|
| ANG  | Angry    | HAP  | Happy    |
| DIS  | Disgusted| NEU  | Neutral  |
| FEA  | Fearful  | SAD  | Sad      |

---

## Project Structure

```
Thesis_G10/
│
├── data/
│   ├── raw/
│   │   ├── README.md              ← dataset download instructions
│   │   ├── CREMA-D/               ← original dataset (not in git)
│   │   │   ├── AudioWAV/          ← ~7,400 WAV audio clips
│   │   │   └── VideoFlash/        ← ~7,442 FLV video clips
│   │   ├── SAVEE/                 ← optional (future tracks)
│   │   └── MELD/                  ← optional (future tracks)
│   │
│   ├── processed/
│   │   ├── track1_manifests/
│   │   │   ├── swap_pairs.csv           ← 6,532 emotion-swap pairs (all actors)
│   │   │   ├── test_pairs_1001_1003.csv ← 215 pairs (actors 1001–1003 only)
│   │   │   ├── clips.csv                ← per-clip metadata
│   │   │   └── actor_stats.csv          ← clip counts per actor
│   │   ├── rvc_datasets/
│   │   │   └── actor_XXXX/              ← WAVs resampled to 40 kHz for RVC training
│   │   └── actor_portraits/             ← output of extract_actor_frames.py (not in git)
│   │       └── actor_XXXX/
│   │           ├── portrait.png         ← best neutral portrait frame for SadTalker
│   │           └── finetune/            ← additional diverse frames (10 by default)
│   │
│   └── synthetic/
│       ├── track1_fakes/
│       │   ├── videos/                  ← generated fake MP4 files (not in git)
│       │   ├── metadata.csv             ← label, method, and path per fake
│       │   └── failed.csv               ← clips that failed with error details
│       ├── track2_fakes/
│       │   ├── videos/                  ← Wav2Lip reanimated fakes (not in git)
│       │   ├── metadata.csv
│       │   └── failed.csv
│       └── track3_fakes/
│           ├── videos/                  ← SadTalker talking head fakes (not in git)
│           ├── metadata.csv
│           └── failed.csv
│
├── src/
│   ├── track1/
│   │   ├── parse_cremad.py        ← parses CREMA-D and builds pair manifests
│   │   ├── train_rvc_voices.py    ← trains one RVC voice model per actor
│   │   └── track1_generate.py     ← generates Track 1 fake videos
│   ├── track2/
│   │   └── track2_generate.py     ← Wav2Lip lip reenactment on Track 1 output
│   └── track3/
│       ├── extract_actor_frames.py ← extracts portrait + finetune frames per actor
│       └── track3_generate.py     ← SadTalker talking head generation
│
├── tools/
│   ├── README.md                  ← setup instructions for all external tools
│   ├── Applio/                    ← RVC v2 training tool (not in git, clone separately)
│   ├── Wav2Lip/                   ← lip reenactment tool (not in git, clone separately)
│   └── SadTalker/                 ← talking head generation (not in git, clone separately)
│
├── scripts/
│   ├── setup_track1.sh            ← environment setup script
│   └── peek_cmu_mosei.py          ← inspect CMU-MOSEI feature file shapes
│
├── .github/workflows/
│   ├── claude-code-review.yml     ← auto PR review (bugs + security)
│   └── claude.yml                 ← Claude PR assistant
│
├── requirements.txt
└── .gitignore
```

---

## Track 1 — Emotion Audio Swap

Track 1 creates the simplest class of fake: the audio track of a real video is replaced with audio of the **same actor saying the same sentence but with a different emotion**. The result is a clip where the face and voice express conflicting emotions — a labeled fake that preserves speaker identity and lip-sync plausibility.

Two methods are provided, producing two separate sets of fakes from the same pair manifest:

### Method A: Direct Swap (no AI)

`ffmpeg` strips the original audio track and replaces it with the target WAV file directly. No model is trained or invoked.

**Example:**

| | File | Emotion shown |
|-|------|---------------|
| Input video | `1001_IEO_ANG_HI.mp4` | Angry face |
| Replacement audio | `1001_IEO_HAP_MD.wav` | Happy voice |
| Output fake | `FAKE_T1_1001_IEO_ANG_HI__AUDIO_1001_IEO_HAP_MD.mp4` | Angry face + Happy voice |

This method is fast (~0.5 s/clip) and produces a perfectly valid fake for detection experiments, but the audio is unmodified real speech — making it easier for detectors that look for audio synthesis artifacts.

---

### Method B: AI Voice Conversion (StyleTTS2 + RVC)

Method B produces a more challenging fake by **synthesising the target emotion from scratch and then transferring the actor's vocal identity onto it**. This makes both the emotion and the voice identity convincing while keeping the audio-visual mismatch intact.

**Pipeline:**

```
[CREMA-D sentence text]
        │
        ▼
  StyleTTS 2                ← Neural TTS model
  synthesises speech         Generates natural-sounding speech in the
  in target emotion          target emotion style using a reference WAV
        │
        ▼
  RVC v2 (Applio)           ← Retrieval-based Voice Conversion
  wraps speech with          Converts the synthesised voice to match
  actor's vocal timbre       the actor's real vocal characteristics,
        │                    using a trained per-actor voice model
        ▼
  ffmpeg muxes               Replaces the original audio track in
  audio into video           the source video file
```

**Why this is harder to detect:** The synthesised audio contains real neural TTS artifacts and a real voice identity, unlike Method A where the audio is completely unaltered original speech.

---

### Step 1 — Parse the dataset

`src/track1/parse_cremad.py` scans the CREMA-D directory, parses every filename into its components (actor, sentence, emotion, intensity), and generates **emotion-swap pairs**: for every clip, it finds other clips from the **same actor and same sentence** but with a **different emotion**. These pairs are the inputs to the generation pipeline.

```bash
python src/track1/parse_cremad.py \
  --cremad_dir data/raw/CREMA-D \
  --out_dir    data/processed/track1_manifests
```

**Outputs:**

| File | Description |
|------|-------------|
| `swap_pairs.csv` | 6,532 pairs across all 91 actors |
| `test_pairs_1001_1003.csv` | 215 pairs for actors 1001–1003 (quick test subset) |
| `clips.csv` | One row per clip with parsed fields |
| `actor_stats.csv` | Clip count per actor |

---

### Step 2 — Train voice models (Method B only)

`src/track1/train_rvc_voices.py` trains a **per-actor RVC v2 voice model** using Applio's CLI. The model learns the actor's unique vocal characteristics (timbre, prosody pattern) from their CREMA-D clips, so that RVC can later transfer those characteristics onto any synthesised speech.

**Training stages per actor:**

```
(1) Stage     → Copy actor WAVs to rvc_datasets/, resample to 40 kHz
(2) Preprocess → Slice audio into training chunks, normalize volume
(3) Extract   → Extract ContentVec speaker embeddings + F0 pitch contours
(4) Train     → Train RVC v2 model for 40 epochs
(5) Index     → Build FAISS vector index for fast retrieval at inference time
(6) Validate  → Run inference on a held-out clip; measure x-vector cosine
                 similarity between converted and original voice (quality check)
```

```bash
# Train for test actors only (recommended first run — ~1 hr on RTX 3060)
python src/track1/train_rvc_voices.py \
  --cremad_dir   data/raw/CREMA-D \
  --applio_dir   tools/Applio \
  --datasets_dir data/processed/rvc_datasets \
  --actors 1001 1002 1003

# Full run — all 91 actors (~30 hrs on RTX 3060)
python src/track1/train_rvc_voices.py \
  --cremad_dir   data/raw/CREMA-D \
  --applio_dir   tools/Applio \
  --datasets_dir data/processed/rvc_datasets
```

**Trained model output:** `tools/Applio/logs/actor_XXXX/actor_XXXX_40e_880s.pth`

> See [`tools/README.md`](tools/README.md) for Windows-specific patches required before training.

---

### Step 3 — Generate fake videos

`src/track1/track1_generate.py` reads the pair manifest and produces one fake video per row.

> **Important:** Run from the `src/track1/` directory — the CSV uses paths relative to that location.

```bash
cd src/track1

# Method A — no models required, processes all pairs in the CSV
python track1_generate.py \
  --pairs_csv ../../data/processed/track1_manifests/swap_pairs.csv \
  --out_dir   ../../data/synthetic/track1_fakes \
  --method    swap

# Method B — requires trained .pth models (run Step 2 first)
python track1_generate.py \
  --pairs_csv  ../../data/processed/track1_manifests/swap_pairs.csv \
  --out_dir    ../../data/synthetic/track1_fakes \
  --method     styletts \
  --cremad_dir ../../data/raw/CREMA-D \
  --applio_dir ../../tools/Applio

# Add --resume to any command to skip already-completed clips
python track1_generate.py --resume [... same args ...]
```

**CLI flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--pairs_csv` | Yes | Path to the swap-pair manifest CSV |
| `--out_dir` | Yes | Directory for output videos and logs |
| `--method` | Yes | `swap` (Method A) or `styletts` (Method B) |
| `--cremad_dir` | Method B | Root CREMA-D directory (for reference audio) |
| `--applio_dir` | Method B | Applio tool directory (for RVC inference) |
| `--resume` | No | Skip clips already present in the checkpoint file |

**Output files:**

| File | Description |
|------|-------------|
| `videos/FAKE_T1_*.mp4` | Generated fake video clips |
| `metadata.csv` | Per-clip record: method, source/target emotions, file paths |
| `failed.csv` | Clips that failed, with error messages |
| `progress_{method}.json` | Checkpoint for `--resume` (updated every 50 clips) |

---

## Pipeline Status

| Stage | Status | Notes |
|-------|--------|-------|
| CREMA-D dataset | ✅ Ready | 7,442 videos + 7,400 WAVs |
| Swap pairs manifest | ✅ Done | 6,532 pairs → `swap_pairs.csv` |
| Test pairs manifest | ✅ Done | 215 pairs (actors 1001–1003) |
| RVC training — actors 1001–1003 | ✅ Done | 40-epoch models in `tools/Applio/logs/` |
| Track 1 Method A — test actors | ✅ Done | 215 / 215 clips |
| Track 1 Method B — test actors | ✅ Done | 214 / 215 clips (1 RVC timeout) |
| RVC training — all 91 actors | ⏳ Pending | Full run not started |
| Track 1 — all 91 actors | ⏳ Pending | Blocked on full training |
| Actor portrait extraction | ⏳ Pending | `extract_actor_frames.py` → `data/processed/actor_portraits/` |
| Track 2 (Wav2Lip) | ⏳ Pending | Script ready; blocked on Track 1 full run |
| Track 3 (SadTalker) | ⏳ Pending | Script ready; blocked on portrait extraction |
