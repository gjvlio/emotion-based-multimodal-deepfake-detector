# Multimodal Deepfake Generation & Detection Pipeline

**Thesis G10** — A two-phase system that (1) generates a labeled dataset of audio-visual deepfakes using cross-modal emotional mismatch as the attack surface, and (2) trains a multimodal detector that exploits the same mismatch as the discriminative signal.

Core hypothesis: deepfake generators process audio and visual modalities independently, so they fail to preserve the biological cross-modal emotional coordination present in real human videos (Ekman & Friesen, 1969). The detector exploits this by comparing the emotion in the voice against the emotion on the face — disagreement indicates a fake.

---

## Table of Contents

- [System Overview](#system-overview)
- [Prerequisites](#prerequisites)
- [Datasets](#datasets)
- [Phase 1 — Deepfake Generation](#phase-1--deepfake-generation)
  - [Track 1 — Audio Swap (StyleTTS2 + RVC)](#track-1--audio-swap-styletts2--rvc)
  - [Track 2 — Audio Swap + Lip Correction (+ Wav2Lip)](#track-2--audio-swap--lip-correction--wav2lip)
  - [Track 3 — Full Face Synthesis (+ SadTalker)](#track-3--full-face-synthesis--sadtalker)
  - [Track 4 — Cross-Speaker Lip Sync on MELD](#track-4--cross-speaker-lip-sync-on-meld)
- [Phase 2 — Detection System](#phase-2--detection-system)
  - [Architecture](#architecture)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Pipeline Status](#pipeline-status)
- [Project Structure](#project-structure)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1 — GENERATION                        │
│                                                                  │
│  CREMA-D ──► Track 1 (StyleTTS2+RVC)        ┐                  │
│  (91 actors) ► Track 2 (+Wav2Lip)           ├─► FAKE samples   │
│               ► Track 3 (+SadTalker)        │                  │
│                                              ┘                  │
│  MELD ────► Track 4 (Wav2Lip cross-speaker) ──► FAKE samples   │
│  (TV clips) ► 50% kept as-is               ──► REAL samples   │
│                                                                  │
│  CMU-MOSEI ─► 100% kept as-is              ──► REAL samples   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2 — DETECTION                         │
│                                                                  │
│  Preprocessing: clip ──► Z_at (1536-dim) + Z_v (768-dim)        │
│                                                                  │
│  Detection model:                                                │
│    Z_at ──► Emotion Head A ──► emotion_A (6-class)              │
│    Z_v  ──► Emotion Head B ──► emotion_B (6-class)              │
│    Z_at, Z_v ──► Bilinear Fusion ──► fused (65536-dim)          │
│    Δ = |emotion_A − emotion_B|  ──► (6-dim)                     │
│    [fused ; Δ] ──► MLP ──► P(fake) ∈ [0, 1]                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

**Python 3.11+**. Install dependencies:

```bash
pip install -r requirements.txt
```

External tools are too large for version control — clone them manually:

| Tool | Location | Required for |
|------|----------|--------------|
| Applio (RVC v2) | `tools/Applio/` | Track 1/2/3 voice conversion |
| Wav2Lip | `tools/Wav2Lip/` | Track 2 + Track 4 |
| SadTalker | `tools/SadTalker/` | Track 3 |

See [`tools/README.md`](tools/README.md) for setup and Windows patches.

---

## Datasets

| Dataset | Type | Role |
|---------|------|------|
| **CREMA-D** | 91-actor lab recordings, 6 emotions | 100% fake source — Tracks 1/2/3 |
| **MELD** | TV dialogue (Friends), 7 speakers | 50% real + 50% fake source (Track 4) |
| **CMU-MOSEI** | In-the-wild YouTube sentiment/emotion | 100% real training samples |

> CREMA-D originals are **never used as real training samples** — they served as generation sources and reusing them as real would introduce label ambiguity.

### Emotion label space (6 classes)

| Index | Emotion | CREMA-D | MELD |
|-------|---------|---------|------|
| 0 | Neutral | NEU | neutral |
| 1 | Happy | HAP | joy |
| 2 | Sad | SAD | sadness |
| 3 | Angry | ANG | anger |
| 4 | Fear | FEA | fear |
| 5 | Disgust | DIS | disgust |

MELD `surprise` → mapped to neutral. CMU-MOSEI dimensional → mapped to nearest class.

---

## Phase 1 — Deepfake Generation

### The Core Audio Chain (Tracks 1–3)

All CREMA-D tracks share the same audio synthesis pipeline:

```
[Sentence text]
      │
      ▼
StyleTTS2                ← Neural TTS: synthesises speech in target emotion
      │                    using a reference WAV of that emotion as style guide
      ▼
RVC v2 (Applio)          ← Voice conversion: transfers actor's vocal timbre
      │                    onto synthesised audio using per-actor .pth model
      ▼
Emotionally mismatched audio in actor's own voice
```

**RVC models:** 91 trained models, one per actor (1001–1091). Each trained for 40 epochs on ~7–10 min of that actor's CREMA-D audio. `tools/Applio/logs/actor_XXXX/`.

### Track 1 — Audio Swap (StyleTTS2 + RVC)
> CREMA-D · 20% split (1,452 pairs)

```
StyleTTS2 → RVC → ffmpeg mux into original face video
```

Face is unchanged — lips still move for original emotion. Easiest to detect (lip-audio mismatch visible).

```bash
# Step 1: parse CREMA-D and build pair manifest
python src/track1/parse_cremad.py \
  --cremad_dir data/raw/CREMA-D \
  --out_dir    data/processed/track1_manifests

# Step 2: split pairs across tracks (actor-stratified)
python scripts/sample_by_track.py \
  --pairs_csv data/processed/track1_manifests/swap_pairs.csv \
  --out_dir   data/processed/track1_manifests

# Step 3: train RVC voice models (all 91 actors, ~16 hrs on RTX 3060)
python src/track1/train_rvc_voices.py \
  --cremad_dir   data/raw/CREMA-D \
  --applio_dir   tools/Applio \
  --datasets_dir data/processed/rvc_datasets

# Step 4: generate
python src/track1/track1_generate.py \
  --pairs_csv  data/processed/track1_manifests/track1_pairs.csv \
  --out_dir    data/synthetic/track1_fakes \
  --applio_dir tools/Applio \
  --cremad_dir data/raw/CREMA-D \
  --resume
```

---

### Track 2 — Audio Swap + Lip Correction (+ Wav2Lip)
> CREMA-D · 30% split (~2,267 pairs)

```
StyleTTS2 → RVC → Wav2Lip → output
```

Wav2Lip rewrites the lip region of the original face video to match the synthesised audio. Lips now match the wrong emotion — harder to detect than Track 1.

```bash
python src/track2/track2_generate.py \
  --pairs_csv  data/processed/track1_manifests/track2_pairs.csv \
  --out_dir    data/synthetic/track2_fakes \
  --applio_dir tools/Applio \
  --wav2lip_dir tools/Wav2Lip \
  --cremad_dir data/raw/CREMA-D \
  --resume

# 25% batches (run sequentially):
python ... --max_clips 567              # batch 1
python ... --max_clips 1134 --resume    # batch 2
python ... --max_clips 1701 --resume    # batch 3
python ... --resume                     # batch 4
```

**Known issues and fixes applied:**
- Wav2Lip S3FD face detector fails on low-resolution or dark frames. Generator retries with
  `resize_factor` 1 → 2 → 4 → 8 before marking a clip failed.
- Applio RVC timeout raised to 600 s for slow-converging actors.
- Actor 1047 black leader frames: same fix as Track 3 (`VideoMP4/` override, see above).
- 8 clips failed (actors 1027/1030/1047/1058). Retry with track2_retry.csv after fixes.

**Retry failed clips:**
```bash
python src/track2/track2_generate.py \
  --pairs_csv  data/processed/track1_manifests/track2_retry.csv \
  --out_dir    data/synthetic/track2_fakes \
  --applio_dir tools/Applio \
  --wav2lip_dir tools/Wav2Lip \
  --cremad_dir data/raw/CREMA-D \
  --resume
```

---

### Track 3 — Full Face Synthesis (+ SadTalker)
> CREMA-D · 50% split (~3,722 pairs)

```
StyleTTS2 → RVC → SadTalker → output
```

SadTalker generates a complete talking-head video from a single portrait frame (middle frame of original video) driven by the synthesised audio. The entire face is replaced — head pose, expressions, lip sync. Hardest to detect.

```bash
python src/track3/track3_generate.py \
  --pairs_csv     data/processed/track1_manifests/track3_pairs.csv \
  --out_dir       data/synthetic/track3_fakes \
  --applio_dir    tools/Applio \
  --sadtalker_dir tools/SadTalker \
  --cremad_dir    data/raw/CREMA-D \
  --resume

# 25% batches (~930 clips each):
python ... --max_clips 930              # batch 1
python ... --max_clips 1861 --resume    # batch 2
python ... --max_clips 2791 --resume    # batch 3
python ... --resume                     # batch 4
```

#### Two-phase strategy (batches 2–4)

End-to-end per clip is ~240 s (TTS+RVC ≈ 213 s + SadTalker ≈ 27 s). Batch 1 benefited from 930
pre-existing cached RVC wavs (`wav_tmp/`) left by a prior interrupted run, so SadTalker ran alone
at 27 s/clip and finished in ~6.2 h. For subsequent batches, pre-compute TTS+RVC first, then run
SadTalker-only:

```bash
# Phase A — TTS+RVC only (no SadTalker). Checkpoints every 20 clips.
python src/track3/precompute_rvc.py \
  --pairs_csv  data/processed/track1_manifests/track3_pairs.csv \
  --out_dir    data/synthetic/track3_fakes \
  --applio_dir tools/Applio \
  --cremad_dir data/raw/CREMA-D \
  --start 930 --end 1861 \   # batch 2 — adjust start/end per batch
  --skip_done                # skip stems already in track3 progress checkpoint

# Phase B — SadTalker-only. track3_generate.py auto-detects wav_tmp cache and skips TTS+RVC.
python src/track3/track3_generate.py \
  --pairs_csv     data/processed/track1_manifests/track3_pairs.csv \
  --out_dir       data/synthetic/track3_fakes \
  --applio_dir    tools/Applio \
  --sadtalker_dir tools/SadTalker \
  --cremad_dir    data/raw/CREMA-D \
  --max_clips 1861 --resume
```

**Phase A** writes `wav_tmp/{stem}_sadtalker_rvc.wav`. **Phase B** checks for that file before
calling StyleTTS2/RVC — on cache hit it proceeds directly to SadTalker.

**Known issues and fixes applied:**
- Actor 1047 `IEO_FEA_LO.flv` has black leader frames (0.03–0.23 s) that break face detection.
  Fix: trimmed MP4 at `data/raw/CREMA-D/VideoMP4/1047_IEO_FEA_LO.mp4`; generator checks
  `VideoMP4/` before `VideoFlash/`.
- Applio RVC timeout raised to 600 s (was 300 s — insufficient for slow actors).
- Actors 1061–1062 FLV files fail ffmpeg conversion at runtime. Fix: pre-converted all
  164 FLVs to `VideoMP4/` via `scripts/preconvert_flv.py`; generator finds MP4 directly.
- SadTalker `--preprocess crop` fails face detection on some actors. Generator now retries
  with `--preprocess full` before marking a clip failed.
- 637 clips failed (actors 1038/1047–1062, batch 3). Retry with track3_retry.csv after fixes.

**Retry failed clips:**
```bash
# Step 1: pre-convert FLVs for actors 1061-1062 (already done; skip if VideoMP4/ exists)
python scripts/preconvert_flv.py --cremad_dir data/raw/CREMA-D --actors 1061 1062

# Step 2: retry all 637 failed clips
python src/track3/track3_generate.py \
  --pairs_csv     data/processed/track1_manifests/track3_retry.csv \
  --out_dir       data/synthetic/track3_fakes \
  --applio_dir    tools/Applio \
  --sadtalker_dir tools/SadTalker \
  --cremad_dir    data/raw/CREMA-D \
  --resume
```

---

### Track 4 — Cross-Speaker Lip Sync on MELD
> MELD · 50% split · 4,909 clips ≥2.5 s → 2,522 pairs (2,387 real)

```
[MELD clip A — face video] + [MELD clip B — real audio] → Wav2Lip → fake
```

No speech synthesis — both face and audio are genuine MELD utterances from different speakers. Simulates putting real words in another person's mouth. Distinct from Tracks 1–3: no TTS artifacts.

```bash
# Step 1: split MELD 50/50 and build cross-speaker pairs
python scripts/sample_meld.py \
  --meld_dir data/raw/MELD/MELD-RAW/MELD.Raw \
  --out_dir  data/processed/meld_manifests

# Step 2: generate
python src/track4/track4_generate.py \
  --pairs_csv   data/processed/meld_manifests/meld_pairs.csv \
  --out_dir     data/synthetic/track4_fakes \
  --wav2lip_dir tools/Wav2Lip \
  --resume

# 25% batches (~1,268 clips each):
python ... --max_clips 1268              # batch 1
python ... --max_clips 2535 --resume     # batch 2
python ... --max_clips 3803 --resume     # batch 3
python ... --resume                      # batch 4
```

---

## Phase 2 — Detection System

### Architecture

The detector takes one video clip and outputs **P(fake) ∈ [0, 1]**.

```
                Z_at (1536-dim)          Z_v (768-dim)
                [Wav2Vec·BERT]           [ViT]
                      │                      │
            ┌─────────┤                      ├──────────┐
            │         │                      │          │
            ▼         │                      │          ▼
     Emotion Head A   │                      │   Emotion Head B
     (1536→256→6)     │                      │   (768→256→6)
      emotion_A       │                      │   emotion_B
                      └──────────┬───────────┘
                                 ▼
                         Bilinear Fusion
                    (proj_256 ⊗ proj_256 → 65,536)
                                 │
                                 ▼         Δ = |softmax(A) − softmax(B)|
                                 │                     │
                                 └──────[concat]───────┘
                                              │
                                        Classifier MLP
                                    (65,542 → 512 → 128 → 1)
                                              │
                                          sigmoid
                                              │
                                          P(fake)
```

**Key constraints:**
- Bilinear fusion operates on **raw embeddings only** — emotion probabilities are NOT fed into it
- Δ and bilinear output are **parallel inputs** to the classifier — not chained
- 256-dim projection before outer product is mandatory (avoids 768×768 = 590K OOM on RTX 3060)

### Z_at — Audio-Text Embedding (1536-dim)

```
Raw audio waveform
      ├──► Wav2Vec 2.0 (facebook/wav2vec2-base)
      │    mean-pool temporal dim → (768,)
      │
      └──► Whisper ASR → transcript text
           → BERT (bert-base-uncased) CLS token → (768,)

Z_at = concat([acoustic_768, linguistic_768]) → (1536,)
```

### Z_v — Visual Embedding (768-dim)

```
Video frames
      │
      ├── Coarse filter (Haar cascade — drop frames with no face)
      │
      ├── RetinaFace (MobileNet) — precise face bounding box + alignment
      │
      ├── Fine filter — select top 8 keyframes by (confidence × sharpness)
      │
      └── ViT (google/vit-base-patch16-224) — CLS token per keyframe
          mean-pool across keyframes → (768,)
```

---

### Preprocessing

Preprocessing caches feature tensors to disk. Run once before training.

```bash
# Process all clips from all tracks + real sources
python scripts/preprocess_all.py --device cuda

# With multiprocessing (CPU-heavy steps)
python scripts/preprocess_all.py --workers 4 --device cpu
```

**Cache layout:**
```
data/preprocessed/
├── audio/              {clip_id}.wav        — 16kHz mono WAV
├── transcripts/        {clip_id}.txt        — Whisper ASR transcript
└── features/
    ├── z_at/           {clip_id}.pt         — (1536,) tensor
    └── z_v/            {clip_id}.pt         — (768,)  tensor
```

Preprocessing is safe to interrupt and resume — already-cached clips are skipped.

---

### Training

Two-phase strategy:

**Phase 1 — Frozen backbones** (Wav2Vec2, BERT, ViT weights fixed):
- Trains only: emotion heads + bilinear fusion projections + classifier MLP
- Uses cached (Z_at, Z_v) tensors — no backbone forward pass
- Higher LR (1e-3), AdamW, early stopping (patience 5)

**Phase 2 — Full fine-tune** (all weights unfrozen):
- Backbones adapt to deepfake detection task at lower LR (1e-5)
- Requires raw audio/frames DataLoader (end-to-end backprop through backbones)

```bash
# Phase 1 — train on cached features
python scripts/train.py --phase 1 --device cuda

# Phase 2 — fine-tune all parameters (load Phase 1 checkpoint)
python scripts/train.py --phase 2 --device cuda \
  --resume checkpoints/best_phase1.pt
```

**Multi-task loss:**
```
L_total = L_BCE(P(fake), fake_label)
        + 0.5 × L_CE(emotion_A_logits, audio_emotion_label)
        + 0.5 × L_CE(emotion_B_logits, visual_emotion_label)
```

**Label assignment per sample type:**

| Sample | fake_label | audio_emotion | visual_emotion |
|--------|-----------|---------------|----------------|
| Real (MELD, CMU-MOSEI) | 0 | annotation | same as audio |
| Track 1 fake | 1 | target emotion | source emotion |
| Track 2 fake | 1 | target emotion | target emotion |
| Track 3 fake | 1 | target emotion | target emotion |
| Track 4 fake | 1 | audio speaker | video speaker |

---

### Evaluation

```bash
# Primary evaluation (RQ1 — detection accuracy + per-pipeline breakdown)
python scripts/evaluate.py \
  --checkpoint checkpoints/best_phase1.pt \
  --threshold 0.5

# With Δ-ablation (RQ2 — validates emotion mismatch contribution)
python scripts/evaluate.py \
  --checkpoint checkpoints/best_phase1.pt \
  --ablation

# With OOD benchmark check
python scripts/evaluate.py \
  --checkpoint checkpoints/best_phase1.pt \
  --ood_csv path/to/benchmark.csv
```

**RQ1 metrics:** Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix, per-pipeline breakdown.

**RQ2 metrics:** Pearson/Spearman correlation between ‖Δ‖ magnitude and P(fake) confidence. Ablation compares full-model F1 vs Δ-zeroed variant — the drop measures Δ's contribution.

---

## Pipeline Status

### Measured Generation Throughput (RTX 3060 Mobile, Windows 11)

| Stage | Time per clip | Per 930-clip batch | Notes |
|-------|--------------|-------------------|-------|
| RVC model training (91 actors) | — | ~16 h total | 40 epochs/actor on CREMA-D audio |
| TTS + RVC precompute | ~22 s | ~5.7 h | measured batch 2 (after model warm-up) |
| SadTalker (wav cached) | ~24 s | ~6.2 h | measured batch 1 (talking-head only) |
| Track 3 full end-to-end | ~46 s | ~11.9 h | TTS+RVC + SadTalker combined |
| Track 2 Wav2Lip | ~30–60 s | ~8–16 h | varies by resize_factor fallback depth |

**Projected Track 3 completion** (starting from batch 2, sequential):
- Each batch: precompute ~5.7 h + SadTalker pass ~6.2 h ≈ **~12 h per batch**
- Batches 2, 3, 4: ~12 h × 3 = **~36 h** of continuous GPU time remaining

### Generation Progress

| Track | Dataset | Total pairs | Done | Status |
|-------|---------|-------------|------|--------|
| Track 1 — StyleTTS2+RVC | CREMA-D 20% | 1,452 | **1,452** | ✅ 100% complete |
| Track 2 — +Wav2Lip | CREMA-D 30% | 2,267 | **2,267** | ✅ 100% complete |
| Track 3 — +SadTalker | CREMA-D 50% | 3,722 | **3,722** | ✅ 100% complete |
| Track 4 — Wav2Lip MELD | MELD 50% | 2,522 | **5** (test) | 🟡 Ready to run (batch 1 pending) |

### Data Preparation

| Step | Status |
|------|--------|
| CREMA-D parsing + pair manifest | ✅ 7,441 pairs (T1: 1,452 + T2: 2,267 + T3: 3,722) |
| Track split (20/30/50%) | ✅ track1/2/3_pairs.csv |
| RVC models — all 91 actors | ✅ All trained (40 epochs each) |
| MELD 50/50 split + pairs | ✅ 2,387 real + 2,522 pairs (≥2.5 s filter, 4,909 total) |
| CMU-MOSEI segmentation | 🟡 311 raw videos present, manifest pending |

### Detection System

| Component | Status |
|-----------|--------|
| Preprocessing module | ✅ Implemented (`src/preprocessing/`) |
| Detection model (emotion heads + bilinear + classifier) | ✅ Implemented (`src/models/`) |
| Training module (multi-task loss, two-phase trainer) | ✅ Implemented (`src/training/`) |
| Evaluation module (metrics, ablation, OOD) | ✅ Implemented (`src/evaluation/`) |
| Preprocessing run (cache Z_at, Z_v) | 🟡 Pending — CREMA-D generation complete, ready to run |
| Model training | 🟡 Pending — run after preprocessing |

---

## Project Structure

```
Thesis_G10/
│
├── configs/
│   └── default.yaml               ← all hyperparameters and paths
│
├── data/
│   ├── raw/
│   │   ├── CREMA-D/               ← 91 actors, 7,442 clips (not in git)
│   │   ├── MELD/                  ← TV dialogue clips (not in git)
│   │   └── CMU-MOSEI/             ← YouTube clips + labels (not in git)
│   │
│   ├── processed/
│   │   ├── track1_manifests/
│   │   │   ├── swap_pairs.csv          ← 7,442 emotion-swap pairs
│   │   │   ├── track1_pairs.csv        ← 1,484 pairs → Track 1
│   │   │   ├── track2_pairs.csv        ← 2,267 pairs → Track 2
│   │   │   └── track3_pairs.csv        ← 3,722 pairs → Track 3
│   │   ├── meld_manifests/
│   │   │   ├── meld_real.csv           ← 2,387 real clips (label=0, ≥2.5 s)
│   │   │   └── meld_pairs.csv          ← 2,522 cross-speaker pairs (label=1, ≥2.5 s)
│   │   ├── rvc_datasets/               ← resampled WAVs per actor (not in git)
│   │   └── actor_portraits/            ← SadTalker portrait frames (not in git)
│   │
│   ├── synthetic/
│   │   ├── track1_fakes/
│   │   │   ├── videos/                 ← 1,452 MP4 fakes (not in git)
│   │   │   ├── metadata.csv
│   │   │   └── failed.csv
│   │   ├── track2_fakes/
│   │   │   ├── videos/                 ← 2,267 MP4 fakes (not in git)
│   │   │   ├── metadata.csv
│   │   │   └── failed.csv
│   │   ├── track3_fakes/
│   │   │   ├── videos/                 ← in progress (not in git)
│   │   │   ├── metadata.csv
│   │   │   └── failed.csv
│   │   └── track4_fakes/               ← pending
│   │
│   └── preprocessed/                   ← cached Z_at/Z_v tensors (not in git)
│       ├── audio/
│       ├── transcripts/
│       └── features/
│           ├── z_at/
│           └── z_v/
│
├── src/
│   ├── track1/
│   │   ├── parse_cremad.py         ← CREMA-D parser + pair builder
│   │   ├── train_rvc_voices.py     ← per-actor RVC training
│   │   └── track1_generate.py      ← StyleTTS2+RVC generation
│   ├── track2/
│   │   └── track2_generate.py      ← +Wav2Lip lip correction
│   ├── track3/
│   │   ├── extract_actor_frames.py ← portrait extraction for SadTalker
│   │   └── track3_generate.py      ← +SadTalker full face synthesis
│   ├── track4/
│   │   └── track4_generate.py      ← Wav2Lip cross-speaker on MELD
│   │
│   ├── preprocessing/
│   │   ├── filters.py              ← Haar coarse filter + keyframe selection
│   │   ├── audio.py                ← Wav2Vec2 + Whisper + BERT → Z_at
│   │   ├── visual.py               ← RetinaFace + ViT → Z_v
│   │   └── pipeline.py             ← orchestrator + disk cache manager
│   │
│   ├── models/
│   │   ├── emotion_heads.py        ← EmotionHeadA (audio) + EmotionHeadB (visual)
│   │   ├── bilinear.py             ← BilinearFusion (256-dim projection + outer product)
│   │   ├── classifier.py           ← ClassifierMLP (65542→512→128→1)
│   │   └── detection_model.py      ← DeepfakeDetector (full model + freeze/unfreeze)
│   │
│   ├── training/
│   │   ├── losses.py               ← MultiTaskLoss (L_BCE + λL_CE×2)
│   │   ├── dataset.py              ← DeepfakeDataset + stratified_split
│   │   └── trainer.py              ← Trainer (Phase 1 cached + Phase 2 end-to-end)
│   │
│   ├── evaluation/
│   │   ├── metrics.py              ← DetectionMetrics (RQ1 + RQ2)
│   │   ├── ablation.py             ← Δ-removal ablation evaluator
│   │   └── ood_eval.py             ← OOD benchmark evaluator
│   │
│   └── utils/
│       ├── config.py               ← Config dataclass, loads default.yaml
│       └── logging_utils.py        ← Logger factory + TensorBoard wrapper
│
├── scripts/
│   ├── sample_by_track.py          ← split swap_pairs.csv into T1/T2/T3 manifests
│   ├── sample_meld.py              ← split MELD 50/50, build Track 4 pairs
│   ├── preprocess_all.py           ← run preprocessing on all clips → Z_at/Z_v cache
│   ├── train.py                    ← training entry point (Phase 1 / Phase 2)
│   ├── evaluate.py                 ← evaluation entry point (metrics + ablation + OOD)
│   ├── validate_generation.py      ← health check for generated clips
│   ├── migrate_stems.py            ← one-off: rename FAKE_T1_ → FAKE_T2_/T3_ (already run)
│   └── preconvert_flv.py           ← pre-convert FLVs to VideoMP4/ for problem actors
│
├── tools/
│   ├── README.md
│   ├── Applio/                     ← RVC v2 tool (not in git, clone separately)
│   ├── Wav2Lip/                    ← lip sync tool (not in git, clone separately)
│   └── SadTalker/                  ← talking head tool (not in git, clone separately)
│
├── checkpoints/                    ← saved model checkpoints (not in git)
├── logs/                           ← TensorBoard logs (not in git)
├── docs/
│   ├── system_architecture.md      ← detailed architecture spec
│   └── sys_archi_memory.md         ← implementation reference document
│
├── requirements.txt
└── .gitignore
```
