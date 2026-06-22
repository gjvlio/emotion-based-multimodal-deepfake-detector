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
  - [Track 4 — Emotion-Mismatch Lip Sync on MELD (MuseTalk)](#track-4--emotion-mismatch-lip-sync-on-meld-musetalk)
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
│  MELD ────► Track 4 (MuseTalk emotion-mismatch) ──► FAKE samples │
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
│    Z_at, Z_v ──► Compact Bilinear Pooling ──► fused (8192-dim)  │
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
| Wav2Lip | `tools/Wav2Lip/` | Track 2 |
| SadTalker | `tools/SadTalker/` | Track 3 |
| MuseTalk | `tools/MuseTalk/` | Track 4 (emotion-mismatch lip-sync) |

See [`tools/README.md`](tools/README.md) for setup and Windows patches.

---

## Datasets

| Dataset | Type | Role |
|---------|------|------|
| **CREMA-D** | 91-actor lab recordings, 6 emotions | 100% fake source — Tracks 1/2/3 |
| **MELD** | TV dialogue (Friends), ~13.7k utterances | 6,816 usable (≥2.5 s, train+dev+test) — 50% real + 50% fake → Track 4 |
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

> **MELD usable clip count:** ~13,708 annotated utterances across train/dev/test. After ≥2.5 s duration filter: **6,816 clips** (6,890 dropped — most are short conversational turns like "Yes.", "I know."). `sample_meld.py` loads all three splits and applies the filter automatically.

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

### Track 4 — Emotion-Mismatch Lip Sync on MELD (MuseTalk)
> MELD · fake-source half · 3,482 pairs (video_emotion ≠ audio_emotion, 100% mismatch rate)

```
[MELD clip A — face video] + [MELD clip B — donor audio, different emotion] → MuseTalk → fake
```

Face shows emotion A; voice (and lip movement) carries emotion B. MuseTalk (diffusion-based, 2024) generates lip movements consistent with the donor voice — mismatch is purely emotional, not kinematic. No TTS artifacts; both signals are real human speech.

**MuseTalk setup** (clone to `tools/MuseTalk/`, download weights):
- `models/musetalk/pytorch_model.bin` (3.4 GB), `musetalk.json`
- `models/dwpose/dw-ll_ucoco_384.pth` (387 MB)
- `models/face-parse-bisent/79999_iter.pth` (50 MB), `resnet18-5c106cde.pth`
- `models/sd-vae-ft-mse/config.json` + `diffusion_pytorch_model.bin` (319 MB)
- `models/whisper/pytorch_model.bin` (144 MB)

> **Note (Windows/CUDA):** DWPose mmpose replaced with `face_alignment` (standard PyPI package) to avoid mmcv/xtcocotools NumPy 2.x ABI incompatibility. Patch already applied to `tools/MuseTalk/musetalk/utils/preprocessing.py`.

```bash
# Step 1: split MELD 50/50 and build fake-source pool
#   Loads train+dev+test splits, applies ≥2.5 s filter (6,816 usable of ~13,708 total)
#   6,890 clips dropped — short conversational turns (<2.5 s)
python scripts/sample_meld.py \
  --meld_dir data/raw/MELD/MELD-RAW/MELD.Raw \
  --out_dir  data/processed/meld_manifests
# Outputs: meld_real.csv (3,334 real), meld_fake_src.csv (3,482 fake sources)

# Step 2: build emotion-mismatch pairs from fake-source pool
python scripts/sample_meld_mismatch.py
# Output: meld_mismatch_pairs.csv (3,482 pairs, 100% video_emotion ≠ audio_emotion)
# Real pool (meld_real.csv) is NEVER touched — clean 50/50 partition

# Step 3: smoke test (2 clips)
python scripts/generate_track4_musetalk.py --max_pairs 2
# Expected: 2 .mp4 files in data/synthetic/track4_fakes/, metadata.csv updated

# Step 4: generate all pairs (auto-resumes on restart, skips already-done clips)
python scripts/generate_track4_musetalk.py
# Optional: control batch size (default 500)
python scripts/generate_track4_musetalk.py --batch_size 200
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
                    Compact Bilinear Pooling
                    (Tensor Sketch → 8,192-dim)
                                 │
                                 ▼         Δ = |softmax(A) − softmax(B)|
                                 │                     │
                                 └──────[concat]───────┘
                                              │
                                        Classifier MLP
                                    (8,198 → 512 → 128 → 1)
                                              │
                                          sigmoid
                                              │
                                          P(fake)
```

**Key constraints:**
- Bilinear fusion operates on **raw embeddings only** — emotion probabilities are NOT fed into it
- Δ and bilinear output are **parallel inputs** to the classifier — not chained
- Compact Bilinear Pooling (Tensor Sketch, Fukui et al. 2016) approximates the outer product in 8,192-dim — no linear projection layers, zero learned parameters in the fusion layer

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
      ├── Fine filter — select top 8 keyframes by (confidence × sharpness × AU_saliency)
      │   AU saliency = FACS Action Unit intensity sum (py-feat); falls back to conf × sharpness
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
| Track 4 fake | 1 | audio_emotion (donor) | video_emotion (face) |

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
| Track 4 — MuseTalk MELD | MELD emotion-mismatch | 3,482 | **in progress** | 🟡 Smoke test ✅ — full generation running |

### Data Preparation

| Step | Status |
|------|--------|
| CREMA-D parsing + pair manifest | ✅ 7,441 pairs (T1: 1,452 + T2: 2,267 + T3: 3,722) |
| Track split (20/30/50%) | ✅ track1/2/3_pairs.csv |
| RVC models — all 91 actors | ✅ All trained (40 epochs each) |
| MELD split + mismatch pairs | ✅ 3,334 real + 3,482 fake pairs (≥2.5 s, train+dev+test, 6,816 usable of ~13.7k) |
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
│   │   │   ├── meld_real.csv               ← 3,334 real clips (label=0, ≥2.5 s, train+dev+test)
│   │   │   ├── meld_fake_src.csv           ← 3,482 fake-source clips (disjoint from real pool)
│   │   │   ├── meld_pairs.csv              ← cross-speaker pairs (reference only)
│   │   │   └── meld_mismatch_pairs.csv     ← 3,482 emotion-mismatch pairs → Track 4
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
│   ├── track4/                     ← (legacy; generation moved to scripts/)
│   │
│   ├── preprocessing/
│   │   ├── filters.py              ← Haar coarse filter + keyframe selection
│   │   ├── audio.py                ← Wav2Vec2 + Whisper + BERT → Z_at
│   │   ├── visual.py               ← RetinaFace + ViT → Z_v
│   │   └── pipeline.py             ← orchestrator + disk cache manager
│   │
│   ├── models/
│   │   ├── emotion_heads.py        ← EmotionHeadA (audio) + EmotionHeadB (visual)
│   │   ├── bilinear.py             ← CompactBilinearFusion (Tensor Sketch 8,192-dim CBP, Fukui et al. 2016)
│   │   ├── classifier.py           ← ClassifierMLP (8198→512→128→1)
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
│   ├── sample_meld.py              ← split MELD 50/50, build meld_pairs.csv
│   ├── sample_meld_mismatch.py     ← build emotion-mismatch pairs → Track 4
│   ├── generate_track4_musetalk.py ← Track 4 generation (resume-safe, Ctrl+C-safe)
│   ├── smoke_test_pipeline.py      ← pipeline smoke test
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
│   ├── SadTalker/                  ← talking head tool (not in git, clone separately)
│   └── MuseTalk/                   ← diffusion lip-sync tool (not in git, clone separately)
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

---

## Web Application (Detector UI)

A FastAPI service that serves the trained detector behind a single-page web UI and
**auto-equips the latest training checkpoint** — when training writes a newer
`checkpoints/full/best_phase*.pt`, the running server hot-reloads it (no restart).

```
webapp/
├── main.py            ← FastAPI app + routes + SPA shell serving
├── model_service.py   ← checkpoint hot-reload + warmup + inference
├── config.py          ← paths / device / settings (env-overridable)
├── schemas.py         ← typed JSON responses
└── static/            ← SPA frontend (index.html, css/, js/, img/, docs/)
```

### Quick start — Demo (no ML stack needed)

The demo runs the **full UI with hardcoded results** — no torch, no datasets, no GPU,
no trained model. Anyone can run it in under a minute.

> **Use Python 3.11 or 3.12** (not 3.13/3.14 — many wheels don't exist yet).
> **Do not** run the root `requirements.txt` — that's the full thesis pipeline
> (torch, RVC, faiss) and will fail to resolve. The demo only needs `webapp/requirements.txt`.

```powershell
# on the webapp branch, from the repo root
py -3.11 -m venv venv
venv\Scripts\activate

pip install -r webapp/requirements.txt      # 4 packages, a few seconds

uvicorn webapp.main:app --port 8000          # note the "8000" — --port needs a number
```

Open **<http://localhost:8000/demo>**, upload any short clip, click **Run Analysis**.
It cycles three preset results (deepfake / genuine / genuine-but-sarcastic) with a
~10–15 s staged progress screen. Live `/detect` is disabled in this mode (returns 503) —
that's expected; the demo needs no model.

### Full run — live detection

Requires the project ML stack (torch, transformers, torchaudio, openai-whisper,
insightface) installed in the venv, plus a trained checkpoint in `checkpoints/full/`.

```powershell
pip install -r webapp/requirements.txt       # web layer (ML stack must also be installed)

uvicorn webapp.main:app --port 8000                                   # CPU (~8–20 s/clip)
$env:DEEPSENTINEL_DEVICE="cuda"; uvicorn webapp.main:app --port 8000  # GPU (~3–8 s/clip)
```

Open <http://localhost:8000>. Interactive API docs at `/docs`.

### Pages

| Route | Screen |
|---|---|
| `/` | Landing |
| `/upload` | Upload a clip |
| `/analyzing` | Live pipeline progress |
| `/results` | Verdict %, sarcasm read, Δ mismatch, per-modality emotion |
| `/about/thesis` | Abstract + paper download |
| `/about/researchers` | Team profiles |

### API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | service status + equipped checkpoint |
| GET  | `/model/info` | checkpoint metadata (phase, epoch, val_loss, mtime) |
| POST | `/model/reload` | force a checkpoint re-check (normally automatic) |
| POST | `/detect` | upload a video → real/fake verdict + emotion evidence |

```bash
curl -F "file=@clip.mp4" http://localhost:8000/detect
```

### Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DEEPSENTINEL_DEVICE` | `cpu` | `cpu` or `cuda` |
| `DEEPSENTINEL_CHECKPOINT_DIR` | `checkpoints/full` | where the trainer writes checkpoints |
| `DEEPSENTINEL_WARMUP` | `1` | preload models at startup (set `0` to disable) |
| `DEEPSENTINEL_WATCH_INTERVAL_SEC` | `15` | idle checkpoint re-check cadence |
| `DEEPSENTINEL_UPLOAD_DIR` | `webapp/uploads` | uploaded videos (gitignored; not auto-deleted) |

### Notes

- **Inference fidelity:** a web upload is a raw video, so the service runs the project's
  real `PreprocessingPipeline` (Wav2Vec2 + BERT + ViT + face/ASR) → `forward_from_features`.
  This is exact for **Phase 1** checkpoints. Phase 2 (fine-tuned backbones) needs the
  end-to-end path — see the `_predict_e2e` seam in `model_service.py`.
- **Release gate:** the current model is Phase 1 (FakeAVCeleb AUC ≈ 0.50 on unseen fakes).
  The backend auto-equips Phase 2 the moment it is trained, but do not ship publicly until
  a Phase 2 checkpoint clears AUC ≥ 0.70.
