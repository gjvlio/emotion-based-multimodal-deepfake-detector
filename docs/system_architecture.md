# System Architecture — Multimodal Deepfake Generation & Detection

## Overview

System operates in two phases:

1. **Generation** — build a labeled dataset of deepfakes using 3 source datasets
2. **Detection** — train a multimodal classifier to distinguish real from fake

---

## Datasets & Their Roles

Three datasets are used, each with a distinct role:

| Dataset | Type | Role in Training |
|---------|------|-----------------|
| **CREMA-D** | Lab-recorded speech (91 actors, 6 emotions) | 100% used for deepfake generation → all become **fake** training samples |
| **MELD** | TV dialogue clips (Friends series) | 50% becomes **fake** (Track 4), 50% kept as **real** training samples |
| **CMU-MOSEI** | In-the-wild sentiment/emotion video | 100% kept as **real** training samples, no manipulation |

**Why this split?**
- Model sees fakes from lab (CREMA-D) and in-the-wild (MELD) conditions → generalises better across attack domains
- Real samples come from two different domains (MELD TV + CMU-MOSEI YouTube) → reduces domain bias
- CREMA-D originals are NOT reused as real training samples — they served as source material for generation; mixing them in as "real" would confuse the detector

---

## Phase 1 — Deepfake Generation

Four tracks, each producing progressively harder-to-detect fakes. All CREMA-D tracks share the same 3-model generation chain.

### The Core Audio Chain (Tracks 1–3)

```
[Original CREMA-D clip]
        │
        ▼
  ┌─────────────┐
  │ StyleTTS2   │  ← Text-to-speech: synthesises speech in WRONG emotion
  │ (TTS model) │    e.g., actor said "NEU" sentence → generate "ANG" version
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │   RVC v2    │  ← Voice conversion: restores original actor's vocal timbre
  │ (per-actor  │    onto synthesised audio. 91 trained models, one per actor.
  │  .pth model)│    Without this, audio sounds like generic TTS voice.
  └─────────────┘
        │
        ▼
  Emotionally mismatched audio, in actor's own voice
```

#### StyleTTS2 — Style Text-to-Speech 2
> Li et al., NeurIPS 2023

- Takes sentence text + emotion style reference → generates speech with a target emotion
- Uses style diffusion to separate *what is said* from *how it sounds emotionally*
- Output: high-quality WAV file with the wrong emotion for that actor's face

#### RVC v2 — Retrieval-based Voice Conversion
> Applio implementation

- Per-actor model trained on ~7–10 min of that actor's CREMA-D audio
- Converts any input audio → makes it sound like that specific actor
- Uses VITS architecture + speaker retrieval index + RMVPE pitch extractor
- 91 models trained (one per CREMA-D actor), ~10.7 min/actor on RTX 3060

---

### Track 1 — Audio Swap Only
> CREMA-D · 20% split

```
StyleTTS2 → RVC → ffmpeg mux into original face video
```

- Synthesised audio replaces original audio track
- Face video is unchanged — lips still move for original emotion
- Result: actor's mouth shows "happy" expression while voice sounds "angry"
- **Easiest to detect** — lip-audio mismatch is visible

---

### Track 2 — Audio Swap + Lip Correction
> CREMA-D · 30% split

```
StyleTTS2 → RVC → Wav2Lip → output
```

#### Wav2Lip
> Prajwal et al., ACM MM 2020

- GAN-based lip synchronisation model
- Takes face video + new audio → rewrites lip region of face to match speech
- Uses pretrained face detector (S3FD) + lip discriminator trained on LRS2 dataset
- Result: lips visually match the synthesised (wrong-emotion) audio

- **Harder to detect** than Track 1 — lip mismatch is corrected
- Subtle artifacts remain at lip boundary (blending seams, texture inconsistency)

---

### Track 3 — Full Face Synthesis
> CREMA-D · 50% split

```
StyleTTS2 → RVC → SadTalker → output
```

#### SadTalker
> Zhang et al., CVPR 2023

- 3D Morphable Model (3DMM)-based talking head generator
- Takes: single portrait image (middle frame of original video) + audio
- Generates: complete animated talking-head video — head pose, facial expressions, AND lip sync
- Entire face is synthesised from scratch, driven by audio — not just lip patches

- **Hardest to detect** — original face is replaced, all motion comes from the model
- Detectable artifacts: temporal flickering, unnatural eye blink patterns, skin texture inconsistencies

---

### Track 4 — Cross-Speaker Lip Sync
> MELD · 50% split

```
[MELD clip A — face video] + [MELD clip B — audio] → Wav2Lip → fake
```

- No speech synthesis at all — both face and audio are **real** MELD clips from different speakers
- Wav2Lip reanimates person A's lips to match person B's real speech
- **In-the-wild attack** — simulates putting real words in another person's mouth
- Distinct from Tracks 1–3: no TTS artifacts, both signals are authentic human speech

---

## Phase 2 — Detection Pipeline

### Audio Branch

```
Raw audio waveform
      │
      ├──────────────────────────────┐
      ▼                              ▼
 Wav2Vec 2.0                    ASR (transcription)
 (acoustic features)                 │
      │                              ▼
      │                           BERT
      │                    (semantic/linguistic features)
      │                              │
      └──────────┬───────────────────┘
                 ▼
           Audio Encoder
        (fuses both streams)
```

#### Wav2Vec 2.0
> Baevski et al., Facebook/Meta 2020

- Self-supervised model pre-trained on 960h of speech (LibriSpeech)
- Takes raw waveform → outputs dense acoustic feature vectors
- Captures: prosody, pitch contour, spectral artifacts introduced by synthesis
- Well-established baseline for audio deepfake detection; effective against StyleTTS2/RVC artifacts

#### ASR → BERT Path

- ASR (Automatic Speech Recognition) transcribes spoken audio to text
- BERT processes the transcript → semantic/linguistic embeddings
- **Purpose:** Captures *semantic-emotional incongruence* — in a deepfake, the words spoken and the emotional tone they're delivered in don't match. Example: "I'm doing fine today" delivered with rage. BERT + acoustic features together surface this inconsistency that neither stream catches alone.

---

### Visual Branch

```
Full video frames
      │
      ▼
Coarse Filtering          ← Face detection (MTCNN / RetinaFace)
(detect face bounding box)
      │
      ▼
Fine-grained Filtering    ← Facial landmark alignment
(crop + align face region)
      │
      ▼
ViT (Vision Transformer)
(patch-based attention across face frames)
      │
      ▼
Visual Encoder output
```

#### ViT — Vision Transformer
> Dosovitskiy et al., ICLR 2021

- Splits face image into fixed-size patches → treats each patch as a token
- Self-attention across patches captures global spatial relationships
- Detects: Wav2Lip seam artifacts, SadTalker texture inconsistencies, unnatural eye motion, skin blending errors
- Outperforms CNNs on deepfake detection due to global context modeling — CNNs miss long-range spatial anomalies

---

### Bilinear Fusion

```
Audio vector  a  (n-dim)
Visual vector v  (m-dim)
                  │
                  ▼
    F = a ⊗ v   (outer product)
    Result: n×m interaction matrix → flattened → Fused vector
```

- Standard feature concatenation (`[a ; v]`) treats each modality independently
- Bilinear fusion computes every pairwise interaction between audio and visual feature dimensions
- **Critical for this task:** Deepfakes are defined by *cross-modal incongruence*. If acoustic features indicate "angry" and visual features indicate "neutral," the interaction term captures that mismatch explicitly — not just each signal in isolation
- Enables detection of Track 1 fakes (lip-audio mismatch) and Track 3 fakes (synthesised face vs. RVC audio)

---

### Classifier

- MLP layer on top of fused bilinear vector
- Output: sigmoid activation → probability P(fake) ∈ [0.0, 1.0]
- Trained with Binary Cross-Entropy loss
- Label convention: fake = 1, real = 0

---

## Training Data Composition

```
                    CREMA-D (91 actors)
                           │
              ┌────────────┼────────────┐
              │            │            │
          Track 1       Track 2     Track 3
         (20% clips)  (30% clips)  (50% clips)
         StyleTTS2    StyleTTS2    StyleTTS2
         + RVC        + RVC        + RVC
                      + Wav2Lip    + SadTalker
              └────────────┼────────────┘
                           ▼
                   FAKE training samples


                      MELD (TV clips)
                           │
              ┌────────────────────────┐
              │ 50%                50% │
              ▼                     ▼  │
           Track 4              Real MELD
        (Wav2Lip               kept as-is
         cross-speaker)
              │                     │
              ▼                     ▼
         FAKE samples           REAL samples


                  CMU-MOSEI (YouTube)
                           │
                      100% kept real
                           │
                           ▼
                      REAL samples
```

| Label | Source |
|-------|--------|
| **FAKE** | CREMA-D Tracks 1 + 2 + 3, MELD Track 4 (50% of MELD) |
| **REAL** | MELD raw (50% of MELD), CMU-MOSEI (100%) |

---

## Why This Architecture Works

| Design Choice | Reason |
|---------------|--------|
| 4 attack tracks (T1→T3 escalating) | Forces detector to learn general cues, not track-specific artifacts |
| Wav2Vec + BERT dual audio stream | Wav2Vec catches synthesis artifacts; BERT catches semantic-emotional incongruence. Neither alone is sufficient |
| ViT over CNN for visual | Global attention captures long-range spatial anomalies that CNNs miss |
| Bilinear fusion over concatenation | Deepfake signal lives in cross-modal interaction, not individual modalities |
| 3 dataset sources (lab + TV + YouTube) | Prevents domain overfit; model generalises to real-world conditions |
| MELD split (50/50 real/fake) | Ensures in-the-wild real samples balance in-the-wild fake samples |
| CMU-MOSEI as pure real | Provides diverse, unconstrained real speech to anchor the real class |
