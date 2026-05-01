# Multimodal Deepfake Generation & Detection Pipeline

**Thesis G10** — Generating a labeled dataset of audio-visual deepfakes from the CREMA-D corpus, then training and evaluating detection models against them.

Deepfakes are created by **mismatching the emotional content of audio and video**: the actor's face expresses one emotion while their voice expresses a different one. Three pipeline tracks of increasing sophistication produce the fakes, so detectors can be evaluated against both simple and complex manipulations.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Dataset: CREMA-D](#dataset-crema-d)
- [Project Structure](#project-structure)
- [Track 1 — StyleTTS2+RVC Deepfake Generation](#track-1--styletts2rvc-deepfake-generation)
  - [Step 1 — Parse the dataset](#step-1--parse-the-dataset)
  - [Step 2 — Sample by track](#step-2--sample-by-track)
  - [Step 3 — Train voice models](#step-3--train-voice-models)
  - [Step 4 — Generate fakes](#step-4--generate-fake-videos)
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
# 2. Clone and configure Applio, Wav2Lip, SadTalker (see tools/README.md)

# 3. Parse CREMA-D and build the pair manifest
python src/track1/parse_cremad.py \
  --cremad_dir data/raw/CREMA-D \
  --out_dir    data/processed/track1_manifests

# 4. Split pairs across tracks (20% T1 / 30% T2 / 50% T3, stratified by actor)
python scripts/sample_by_track.py \
  --pairs_csv data/processed/track1_manifests/swap_pairs.csv \
  --out_dir   data/processed/track1_manifests

# 5. Train RVC voice models (test actors — ~1 hr on RTX 3060)
python src/track1/train_rvc_voices.py \
  --cremad_dir   data/raw/CREMA-D \
  --applio_dir   tools/Applio \
  --datasets_dir data/processed/rvc_datasets \
  --actors 1001 1002 1003

# 6. Generate Track 1 fakes (StyleTTS2+RVC)
python src/track1/track1_generate.py \
  --pairs_csv  data/processed/track1_manifests/track1_pairs.csv \
  --out_dir    data/synthetic/track1_fakes \
  --applio_dir tools/Applio \
  --cremad_dir data/raw/CREMA-D
```

> Add `--resume` to any generation command to continue an interrupted run.

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

## Track 1 — StyleTTS2+RVC Deepfake Generation

Track 1 creates emotional-mismatch fakes: the audio of a real video is replaced with **AI-synthesised speech expressing a different emotion, cloned to match the actor's voice**. The face still shows the original emotion; the voice expresses a different one.

### How it works

Track 1 **synthesises the target emotion from scratch and then transfers the actor's vocal identity onto it**. This makes both the emotion and the voice identity convincing while keeping the audio-visual mismatch intact.

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

**Why Track 1 is harder to detect than naive swaps:** The synthesised audio carries real neural TTS artifacts and a genuine voice identity match, so simple audio-forensic detectors fail.

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

### Step 2 — Sample by track

`scripts/sample_by_track.py` splits the 6,532 swap pairs into three non-overlapping, actor-stratified manifests for the three pipeline tracks:

| Track | Fraction | Pairs | Rationale |
|-------|----------|-------|-----------|
| Track 1 | 20% | 1,271 | Simplest fakes — fewer needed for baseline |
| Track 2 | 30% | 1,994 | Medium difficulty |
| Track 3 | 50% | 3,267 | Hardest to detect — largest share |

```bash
python scripts/sample_by_track.py \
  --pairs_csv data/processed/track1_manifests/swap_pairs.csv \
  --out_dir   data/processed/track1_manifests
```

**Outputs:**

| File | Description |
|------|-------------|
| `track1_pairs.csv` | 1,271 pairs → Track 1 generation |
| `track2_pairs.csv` | 1,994 pairs → Track 2 filter |
| `track3_pairs.csv` | 3,267 pairs → Track 3 filter |

---

### Step 3 — Train voice models

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

### Step 4 — Generate fake videos

`src/track1/track1_generate.py` reads `track1_pairs.csv` and produces one `_styletts.mp4` per row. These outputs are also the audio source for Track 2 and Track 3.

```bash
python src/track1/track1_generate.py \
  --pairs_csv  data/processed/track1_manifests/track1_pairs.csv \
  --out_dir    data/synthetic/track1_fakes \
  --applio_dir tools/Applio \
  --cremad_dir data/raw/CREMA-D

# Add --resume to skip already-completed clips
python src/track1/track1_generate.py --resume [... same args ...]
```

**CLI flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--pairs_csv` | Yes | `track1_pairs.csv` from `sample_by_track.py` |
| `--out_dir` | Yes | Directory for output videos and logs |
| `--applio_dir` | Yes | Applio tool directory (for RVC inference) |
| `--cremad_dir` | Yes | Root CREMA-D directory (for StyleTTS reference audio) |
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
| Track split manifest | ✅ Done | 20/30/50 → `track1/2/3_pairs.csv` |
| RVC training — actors 1001–1003 | ✅ Done | 40-epoch models in `tools/Applio/logs/` |
| Track 1 — test actors (1001–1003) | ✅ Done | 214 / 214 clips (_styletts.mp4) |
| Track 2 — test actors | ⚠️ Partial | 48 / 214 done; resume with `--resume` |
| Track 3 — test actors | ✅ Done | 214 / 214 clips (_sadtalker.mp4) |
| Actor portrait extraction — all 91 | ✅ Done | `data/processed/actor_portraits/` |
| RVC training — all 91 actors | ⏳ Pending | Full run not started |
| Track 1 — all 91 actors (1,271 clips) | ⏳ Pending | Blocked on full RVC training |
| Track 2 — all 91 actors (1,994 clips) | ⏳ Pending | Blocked on Track 1 full run |
| Track 3 — all 91 actors (3,267 clips) | ⏳ Pending | Blocked on Track 1 full run |
