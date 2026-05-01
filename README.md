# Thesis G10 — Multimodal Deepfake Generation & Detection Pipeline

## Overview

This project generates a labeled dataset of audio-visual deepfakes from the
CREMA-D corpus, then uses that dataset to train and evaluate deepfake detection
models. Fakes are created by **mismatching the emotional content of the audio
and video** — the face shows one emotion, the voice expresses another.

Three tracks of increasing sophistication are used to produce fakes, so that
detectors can be evaluated against simple and complex manipulations alike.

---

## Source Dataset: CREMA-D

**Location:** `data/raw/CREMA-D/`

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset) contains ~7,400
short clips of 91 actors (IDs 1001–1091) each speaking one of 12 sentences
with one of 6 emotions at varying intensities.

### Filename convention

1001_IEO_ANG_HI.wav
│   │   │   └── Intensity: HI / MD / LO / XX (unspecified)
│   │   └── Emotion:   ANG DIS FEA HAP NEU SAD
│   └── Sentence:  IEO = "It's eleven o'clock"
│                  DFA = "Don't forget a jacket"
│                  (12 sentence codes total)
└── Actor ID:  1001 – 1091

### Key subdirectories

| Path | Contents |
|------|----------|
| `data/raw/CREMA-D/AudioWAV/` | ~7,400 `.wav` files, one per clip |
| `data/raw/CREMA-D/` (root) | `.flv` or `.mp4` video files |

---

## Project Directory Structure

Thesis_G10/
├── data/
│   ├── raw/
│   │   └── CREMA-D/             <- original dataset (do not modify)
│   │       ├── AudioWAV/        <- all WAV audio clips
│   │       └── .flv / .mp4      <- all video clips
│   │
│   ├── processed/
│   │   ├── track1_manifests/
│   │   │   └── swap_pairs.csv   <- 6,532 valid emotion-swap pairs
│   │   │
│   │   └── rvc_datasets/
│   │       ├── actor_1001/      <- 82 WAVs resampled to 40 kHz (training data)
│   │       ├── actor_1002/      <- 81 WAVs resampled to 40 kHz
│   │       ├── actor_1003/      <- 82 WAVs resampled to 40 kHz
│   │       └── training_log_.csv <- per-run training status for each actor
│   │
│   └── synthetic/
│       └── track1_fakes/
│           ├── videos/          <- generated fake MP4 files
│           ├── metadata.csv     <- per-fake label, method, paths
│           ├── progress.json    <- completed clip IDs (for resumable runs)
│           └── generation_.log  <- timestamped run logs
│
├── src/
│   ├── track1/
│   │   ├── train_rvc_voices.py  <- trains one RVC voice model per actor
│   │   └── track1_generate.py   <- generates Track 1 fake videos
│   ├── track2/                  <- (Track 2 scripts, TBD)
│   └── track3/                  <- (Track 3 scripts, TBD)
│
├── tools/
│   └── Applio/                  <- RVC v2 training tool (git submodule)
│       ├── core.py              <- CLI entry point for all Applio commands
│       └── logs/
│           ├── actor_1001/      <- Applio training artifacts for actor 1001
│           │   ├── sliced_audios/       <- preprocessed audio chunks (40 kHz)
│           │   ├── sliced_audios_16k/   <- preprocessed audio chunks (16 kHz)
│           │   ├── extracted/           <- contentvec speaker embeddings (.npy)
│           │   ├── f0/                  <- raw pitch contours (.npy)
│           │   ├── f0_voiced/           <- voiced-only pitch contours (.npy)
│           │   ├── model_info.json      <- dataset duration summary
│           │   └── actor_1001.pth       <- TRAINED MODEL (target output)
│           └── actor_1002/, actor_1003/, ...
│       └── rvc/
│           └── models/
│               ├── predictors/
│               │   ├── rmvpe.pt         <- pitch predictor model (must be downloaded)
│               │   └── fcpe.pt          <- alt pitch predictor (must be downloaded)
│               ├── embedders/
│               │   └── contentvec/      <- speaker embedding model (must be downloaded)
│               └── pretraineds/
│                   └── hifi-gan/        <- pretrained vocoder (must be downloaded)
│
├── scripts/
│   └── setup_track1.sh          <- environment setup script
│
└── requirements.txt

---

## Track 1 — Emotion Audio Swap

**What it does:** Replaces the audio of a video with audio from the same actor
saying the same sentence but with a different emotion. The face and voice now
express different emotions — a labeled fake.

**Why it works:** Using the same actor keeps vocal timbre consistent. Using the
same sentence keeps timing/lip-sync plausible.

### Method A: `swap` (simple, no AI)

ffmpeg replaces the audio track directly. Fast, no model needed.

**Input:** `1001_IEO_ANG_HI.mp4` + `1001_IEO_HAP_MD.wav`
**Output:** `FAKE_T1_1001_IEO_ANG_HI__AUDIO_1001_IEO_HAP_MD.mp4`
- Face shows: ANGRY
- Voice says: HAPPY

### Method B: `styletts` (AI voice conversion)

1. **StyleTTS 2** synthesizes the target sentence with the target emotion from scratch
2. **RVC (Retrieval-based Voice Conversion)** wraps the synthesized speech with the original actor's vocal timbre, using a trained per-actor voice model
3. The converted audio replaces the original audio track

This produces a more natural-sounding fake where the synthesized emotion is convincing AND the voice identity is preserved.

---

## Track 1: RVC Training Pipeline

Before Method B generation can run, a voice model must be trained for each
actor. This is handled by `src/track1/train_rvc_voices.py`.

### Steps per actor

Stage dataset -> Preprocess -> Extract features -> Train -> Build index -> Validate
     (1)            (2)              (3)            (4)        (5)          (6)

| Step | Applio command | What it does | Output |
|------|----------------|--------------|--------|
| 1 | _(Python)_ | Copies actor's CREMA-D WAVs into `rvc_datasets/actor_XXXX/`, resampled to 40 kHz | `data/processed/rvc_datasets/actor_XXXX/*.wav` |
| 2 | `core.py preprocess` | Slices audio into training chunks, normalizes | `logs/actor_XXXX/sliced_audios/` |
| 3 | `core.py extract` | Extracts contentvec embeddings + f0 pitch contours | `logs/actor_XXXX/extracted/`, `f0/`, `f0_voiced/` |
| 4 | `core.py train` | Trains RVC v2 model for 40 epochs | `logs/actor_XXXX/actor_XXXX.pth` |
| 5 | `core.py index` | Builds FAISS index for fast inference retrieval | `logs/actor_XXXX/added_*.index` |
| 6 | _(Python)_ | Runs inference on a held-out clip, measures x-vector cosine similarity vs original | `training_log_*.csv` column `xvector_sim` |

### Running the training
```bash
# Train specific actors (for testing)
python src/track1/train_rvc_voices.py \
  --cremad_dir data/raw/CREMA-D \
  --applio_dir tools/Applio \
  --datasets_dir data/processed/rvc_datasets \
  --actors 1001 1002 1003

# Full run — all 91 actors (~30 hrs on RTX 3060)
python src/track1/train_rvc_voices.py \
  --cremad_dir data/raw/CREMA-D \
  --applio_dir tools/Applio \
  --datasets_dir data/processed/rvc_datasets
```

---

## Track 1: Generation Pipeline

After training, generate fake clips using `src/track1/track1_generate.py`.

**Important:** Run from `src/track1/` — the pairs CSV uses paths relative to that directory.

```bash
cd src/track1

# Method A only — fast, no model needed (~0.5s/clip)
python track1_generate.py \
  --pairs_csv ../../data/processed/track1_manifests/swap_pairs.csv \
  --out_dir   ../../data/synthetic/track1_fakes \
  --method    swap

# Test run — actors 1001–1003 only (215 pairs, pre-filtered CSV)
python track1_generate.py \
  --pairs_csv ../../data/processed/track1_manifests/test_pairs_1001_1003.csv \
  --out_dir   ../../data/synthetic/track1_fakes \
  --method    swap \
  --resume

# Method B (StyleTTS 2 + RVC) — requires trained .pth models
python track1_generate.py \
  --pairs_csv  ../../data/processed/track1_manifests/test_pairs_1001_1003.csv \
  --out_dir    ../../data/synthetic/track1_fakes \
  --method     styletts \
  --cremad_dir ../../data/raw/CREMA-D \
  --applio_dir ../../tools/Applio \
  --resume

# Resume any interrupted run
python track1_generate.py --resume ...same args...
```

### Test workflow (actors 1001–1003)

A pre-filtered CSV `test_pairs_1001_1003.csv` (215 pairs) is available for quick
end-to-end testing before committing to all 91 actors.

---

## Current Pipeline Status (updated 2026-04-30, ~17:15)

| Stage | Status | Notes |
|-------|--------|-------|
| CREMA-D dataset | ✅ Ready | 7,442 FLV videos + 7,400 WAVs in `data/raw/CREMA-D/` |
| Swap pairs manifest | ✅ Done | 6,532 pairs in `track1_manifests/swap_pairs.csv` |
| Test pairs manifest | ✅ Done | 215 pairs (actors 1001–1003) in `test_pairs_1001_1003.csv` |
| RVC training (actors 1001–1003) | ✅ Done | 40-epoch models in `tools/Applio/logs/actor_XXXX/actor_XXXX_40e_880s.pth` |
| Track 1 Method A generation | ✅ Done | 215/215 clips for actors 1001–1003; `data/synthetic/track1_fakes/videos/` |
| Track 1 Method B generation | ✅ Done | 214/215 clips for actors 1001–1003; `data/synthetic/track1_fakes/videos/` |
| Track 2 / Track 3 | ⏳ Not started | TBD |
  