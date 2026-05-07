# Project Context — Emotion-Based Multimodal Deepfake Detection

## What this system does

This is a Grade 10 thesis project that builds a deepfake detector using **cross-modal emotional inconsistency** as the discriminative signal. The core hypothesis: deepfake generators process audio and visual modalities independently, so they fail to preserve the natural emotional coordination that exists in real human videos. Even high-quality fakes (like SadTalker, diffusion-based) cannot recreate biological cross-modal emotional coherence (Ekman & Friesen, 1969). The model exploits this by comparing the emotion expressed in the voice against the emotion shown on the face — when they disagree, the video is likely fake.

The system processes a video clip and outputs P(fake) ∈ [0, 1].

## Where we are right now

The **Deepfake Generation Module** is complete. We have a pipeline that produces forged clips from raw source data using four parallel methods (Audio-Swap, Audio-Swap + Lip Correction, Full Face Synthesis, Cross-Speaker Lip Sync). We have not yet built anything beyond generation.

## What still needs to be built

In order:
1. **Data Preprocessing Module** — turns raw video into Z_at and Z_v feature vectors
2. **Detection Module** — the inference architecture (forward pass: Z_at, Z_v → P(fake))
3. **Training Module** — multi-task loss + two-phase backpropagation
4. **Evaluation pipeline** — metrics, ablations, benchmark comparison
5. **Inference webapp** — final demo UI for defense

This document is the orientation for building items 1–4. Item 5 has a separate frontend brief.

---

## Full system architecture (4 modules)

### Module 1 — Deepfake Generation (DONE)
Produces fake clips from CREMA-D and MELD source data using four pipelines. Outputs labeled forged clips with mismatch-type metadata.

### Module 2 — Data Preprocessing (TO BUILD)
Converts raw clips (real and fake) into two feature vectors per clip:
- **Z_at** (1536-dim) — audio-text vector
- **Z_v** (768-dim) — visual vector

### Module 3 — Detection Module (TO BUILD)
The inference architecture. Takes Z_at and Z_v, runs them through emotion heads + bilinear fusion in parallel, outputs P(fake).

### Module 4 — Training Module (TO BUILD)
The learning procedure. Computes three losses (L_BCE + L_emotion_A + L_emotion_B), combines into L_total, backpropagates with a two-phase strategy.

---

## Datasets and their roles

| Dataset | Role | Sample type |
|---|---|---|
| CREMA-D | Source for fake generation only | Pipelines 1, 2, 3 fake outputs |
| MELD | Dual role: 50% real, 50% fake source | Real samples + Pipeline 4 fakes |
| CMU-MOSEI | Real-only | Real samples (100%) |

CREMA-D originals are **never used as real** — reusing them would create label ambiguity. Real-to-fake ratio enforced at 1:1.

### Track splits (within fake samples)
- Pipeline 1 (Audio-Swap): 20% of CREMA-D fakes
- Pipeline 2 (Audio-Swap + Lip Correction): 30% of CREMA-D fakes
- Pipeline 3 (Full Face Synthesis): 50% of CREMA-D fakes
- Pipeline 4 (Cross-Speaker Lip Sync): 50% of MELD samples

### Label structure per sample

| Sample type | fake_label | audio_emotion | visual_emotion |
|---|---|---|---|
| Real (MELD, CMU-MOSEI) | 0 | dataset annotation | same as audio |
| Pipeline 1 fake | 1 | target emotion | source emotion |
| Pipeline 2 fake | 1 | target emotion | target emotion |
| Pipeline 3 fake | 1 | target emotion | target emotion |
| Pipeline 4 fake | 1 | audio speaker emotion | video speaker emotion |

### Emotion label space
6-class softmax: **neutral, happy, sad, angry, fear, disgust**.
- CREMA-D natively has these 6 classes
- MELD has 7 classes (includes "surprise") — drop surprise OR map to nearest of the 6
- CMU-MOSEI uses dimensional labels — map to closest discrete class

---

## Module 2 — Data Preprocessing (build this first)

### Input
Raw video clips (real from MELD/CMU-MOSEI, fake from generation pipeline).

### Output
For each clip, two tensors:
- `Z_at` — shape `(1536,)` audio-text embedding
- `Z_v` — shape `(768,)` visual embedding

### Audio path
Two parallel sub-streams that concatenate at the end.

**Branch A — Acoustic features:**
1. Extract audio waveform from video (use `ffmpeg` or `torchaudio`)
2. Pass through pretrained **Wav2Vec 2.0** (`facebook/wav2vec2-base` from Hugging Face)
3. Mean-pool the temporal dimension to get a fixed-size 768-dim embedding

**Branch B — Linguistic features:**
1. Run audio through **ASR/Whisper** (`openai/whisper-base`) to get transcribed text
2. Tokenize with **BERT tokenizer** (`bert-base-uncased`)
3. Pass tokens through **BERT** model
4. Use the [CLS] token embedding or mean-pool for a 768-dim embedding

**Fusion:**
- Concatenate Branch A (768-dim) + Branch B (768-dim) → **Z_at** (1536-dim)

### Visual path
1. Extract frames from video (typically 30 fps, but downsample if needed)
2. **Coarse filtering** — drop frames with no detectable face (use simple haar cascade or quick face detector)
3. **RetinaFace + MobileNet** (preferred over MTCNN) — detect precise face bounding boxes
4. **Cropping & alignment** — crop face, align to standard pose (eyes level, nose centered)
5. **Fine-grained filtering** — select 4–8 highest-quality keyframes per clip based on:
   - Face confidence score from RetinaFace
   - Sharpness (Laplacian variance)
   - Expression intensity (optional: use FER score)
6. Pass each keyframe through **ViT** (`google/vit-base-patch16-224`)
7. Aggregate keyframe embeddings (mean-pool) → **Z_v** (768-dim)

### Important implementation notes
- Cache preprocessed tensors to disk (use `.pt` or `.npy` files) — preprocessing is slow, training will be re-run many times
- Use `pathlib.Path` for cross-platform paths
- Process in batches with multiprocessing to use CPU cores during I/O-heavy steps
- Save metadata alongside tensors: filename, fake_label, audio_emotion_label, visual_emotion_label, source_pipeline

### Hardware constraint
RTX 3060 12GB. Preprocessing itself is mostly CPU-bound (face detection, feature extraction can use GPU but not memory-heavy). Use `fp16` where applicable.

---

## Module 3 — Detection Module

### Architecture (forward pass only)

```
Z_at (1536)              Z_v (768)
    │                       │
    ├─→ Emotion Head A      ├─→ Emotion Head B
    │   → emotion_A (6)     │   → emotion_B (6)
    │                       │
    └──────┬──────┬─────────┘
           │      │
           ▼      ▼
       Bilinear Fusion
       (Z_at_proj ⊗ Z_v_proj)
           │
           ▼
       flatten → (256×256 = 65,536)
           │
           ▼               Δ = |emotion_A - emotion_B|  (6)
           │                        │
           └─────[concatenate]──────┘
                       │
                       ▼
              Classifier MLP
              [65,536 + 6 = 65,542 dim input]
                       │
                       ▼
                   sigmoid → P(fake)
```

### Component specs

**Emotion Head A** (audio):
```
Linear(1536 → 256) → ReLU/GELU → Dropout(0.3) → Linear(256 → 6) → Softmax
```

**Emotion Head B** (visual):
```
Linear(768 → 256) → ReLU/GELU → Dropout(0.3) → Linear(256 → 6) → Softmax
```

**Bilinear Fusion:**
- Project Z_at → 256-dim via Linear(1536 → 256)
- Project Z_v → 256-dim via Linear(768 → 256)
- Compute outer product: `Z_at_proj.unsqueeze(2) * Z_v_proj.unsqueeze(1)` → shape (B, 256, 256)
- Flatten last two dims → (B, 65536)
- This 256-dim projection is critical to avoid OOM on the 768×768 = 590K space

**Δ computation:**
```python
delta = torch.abs(emotion_A - emotion_B)  # shape (B, 6)
```

**Classifier MLP:**
```
Linear(65542 → 512) → ReLU → Dropout(0.4) → Linear(512 → 128) → ReLU → Linear(128 → 1) → Sigmoid
```

The first layer reduction from 65,542 → 512 is necessary for parameter efficiency on RTX 3060.

### Forward pass output
The Detection Module returns **three things** (needed by Training Module):
- `P(fake)` — scalar per sample
- `emotion_A` — 6-dim per sample
- `emotion_B` — 6-dim per sample

At inference time, only `P(fake)` is consumed; the emotion outputs are discarded. At training time, all three feed into the Training Module.

---

## Module 4 — Training Module

### Loss formulation

Three supervised losses, combined into one scalar:

```
L_BCE        = BCEWithLogitsLoss(P(fake), fake_label)
L_emotion_A  = CrossEntropyLoss(emotion_A_logits, audio_emotion_label)
L_emotion_B  = CrossEntropyLoss(emotion_B_logits, visual_emotion_label)

L_total = L_BCE + λ_A * L_emotion_A + λ_B * L_emotion_B
```

Suggested starting weights: `λ_A = 0.5`, `λ_B = 0.5`. Tune via validation if emotion heads underperform.

### Two-phase training strategy

**Phase 1 — Frozen backbones (warmup):**
- Freeze: Wav2Vec 2.0, BERT, ViT (set `requires_grad=False`)
- Train: Emotion Head A, Emotion Head B, Bilinear Fusion projections, Classifier MLP
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`
- Duration: ~5–10 epochs or until validation loss plateaus
- Goal: stabilize the new components before letting gradients touch pretrained weights

**Phase 2 — Full fine-tune:**
- Unfreeze all parameters
- Optimizer: AdamW, learning rate `1e-5` (much lower — protects pretrained knowledge)
- Duration: until early stopping triggers
- Goal: let backbones adapt slightly to the deepfake detection task

### Training infrastructure

**Hyperparameters:**
- Batch size: 8 (constrained by RTX 3060 12GB)
- Mixed precision: `torch.cuda.amp` with fp16
- Optimizer: AdamW
- LR scheduler: ReduceLROnPlateau on validation loss
- Early stopping: patience 5 epochs on val_loss

**Data splits:**
- Train: 70%
- Validation: 15%
- Test: 15%
- Stratify by `fake_label` AND by `source_pipeline` so all attack types are represented in each split

**Logging:**
- Track per-epoch: train_loss, val_loss, val_accuracy, val_f1, per-class emotion accuracy for both heads
- Use TensorBoard or Weights & Biases
- Save checkpoints: best val_loss, last epoch

---

## Evaluation requirements

### Primary metrics (RQ1 — detection accuracy)
- Accuracy
- Precision, Recall, F1
- AUC-ROC
- Confusion matrix (real vs fake)
- Per-pipeline breakdown (how does the model do on Pipeline 1 vs 2 vs 3 vs 4?)

### Secondary metrics (RQ2 — predictive value of Δ)
- Correlation between `||Δ||` magnitude and P(fake) confidence
- Ablation: train a version of the model with Δ removed (Classifier sees only flatten(bilinear)). Compare detection accuracy. The drop measures Δ's contribution.

### Out-of-distribution check
- Evaluate on a benchmark dataset (DFDC, FaceForensics++, or DigiFakeAV) without training on it
- Report degradation vs in-distribution accuracy
- This addresses the generalization claim in your defense

### False positive / false negative handling
- Report FP rate at threshold 0.5 and 0.65
- Implement the two-condition gate (emotion mismatch AND artifact signal) for FP mitigation if FP rate is too high
- Implement frequency-domain branch (DCT/FFT) for FN mitigation if FN rate is too high
- These are optional refinements — implement only if base model has issues

---

## File and code structure (suggested)

```
project_root/
├── data/
│   ├── raw/                      # source clips (CREMA-D, MELD, CMU-MOSEI)
│   ├── generated/                # output of Module 1 (already exists)
│   └── preprocessed/             # cached Z_at, Z_v tensors
├── src/
│   ├── preprocessing/
│   │   ├── audio.py              # Wav2Vec + Whisper + BERT pipeline
│   │   ├── visual.py             # frame extraction + RetinaFace + ViT
│   │   ├── filters.py            # coarse + fine-grained filtering
│   │   └── pipeline.py           # orchestrator that processes a clip → (Z_at, Z_v)
│   ├── models/
│   │   ├── emotion_heads.py      # Head A, Head B
│   │   ├── bilinear.py           # bilinear fusion with 256-dim projections
│   │   ├── classifier.py         # MLP classifier
│   │   └── detection_model.py    # full Detection Module (combines all above)
│   ├── training/
│   │   ├── losses.py             # L_BCE, L_emotion_A, L_emotion_B, L_total
│   │   ├── trainer.py            # two-phase training loop
│   │   └── dataset.py            # PyTorch Dataset class
│   ├── evaluation/
│   │   ├── metrics.py            # accuracy, F1, AUC, etc.
│   │   ├── ablation.py           # Δ-removed variant
│   │   └── ood_eval.py           # benchmark dataset evaluation
│   └── utils/
│       ├── config.py             # hyperparameters, paths
│       └── logging.py            # TensorBoard wrappers
├── configs/
│   └── default.yaml              # hyperparameters
├── scripts/
│   ├── preprocess_all.py         # run preprocessing on entire dataset
│   ├── train.py                  # entry point for training
│   └── evaluate.py               # entry point for evaluation
└── notebooks/                    # exploratory analysis
```

---

## Coding conventions and constraints

- Python 3.10+, PyTorch 2.x, Hugging Face Transformers
- Use `torch.nn.Module` for all model components
- Use `transformers.AutoModel` and `AutoTokenizer` for pretrained models
- Always wrap model outputs in dataclasses or named tuples for clarity (avoid bare tuples)
- Type-hint everything
- Use `logging` module, not print, for runtime info
- Save model state with `torch.save(model.state_dict(), ...)`, not the full model object
- All paths should be relative to a project root resolved via `pathlib.Path`
- Hardware target: RTX 3060 12GB, fp16, batch_size=8

---

## What success looks like

1. Preprocessing module can take any clip and produce (Z_at, Z_v) reproducibly
2. Detection Module can run forward pass on a (Z_at, Z_v) pair and output P(fake) + emotion_A + emotion_B
3. Training Module can train end-to-end with the two-phase strategy and achieve >85% val accuracy on the in-distribution test set
4. Evaluation module reports per-pipeline accuracy showing the model handles all four attack types
5. Δ-ablation shows a measurable drop in accuracy when Δ is removed (validates RQ2)
6. OOD evaluation on a benchmark dataset shows the model generalizes (degraded but reasonable accuracy)

---

## What NOT to do

- Do not use Pipeline 1/2/3 source CREMA-D clips as real samples
- Do not feed emotion probabilities into Bilinear Fusion (it operates on raw embeddings only)
- Do not skip the 256-dim projection before bilinear (will OOM)
- Do not connect Bilinear Fusion through Δ to the Classifier (they are parallel inputs)
- Do not train Phase 2 with a high learning rate (will destroy pretrained backbones)
- Do not evaluate only on the generated dataset — always include OOD benchmark check
- Do not reproduce song lyrics or copyrighted material if any clips contain them

---

## Key references for context

- Rossler et al. 2019 — FaceForensics++ (artifact-based detection benchmark we're trying to surpass)
- Ekman & Friesen 1969 — biological cross-modal coherence (theoretical anchor)
- Corvi et al. 2023 — diffusion model detection (motivates generation-method-agnostic approach)
- Jiang et al. 2020 — DeeperForensics-1.0 (benchmark for OOD evaluation)
- Dolhansky et al. 2020 — DFDC (benchmark for OOD evaluation)

---

## Defense one-liners (for context, not for code)

When asked "why generated fakes vs benchmark datasets?":
> Benchmark datasets test artifact detection, not emotional inconsistency. We need fakes where emotional mismatch is the primary discriminative feature, paired with per-modality emotion labels — neither of which existing benchmarks provide.

When asked "what makes this generation-method-agnostic?":
> Artifact-based detectors are arms-race participants — accuracy degrades on new generators. Cross-modal emotional inconsistency is invariant because deepfake pipelines process modalities independently and cannot recreate biological emotional coordination. Even diffusion methods fail this test.

When asked "why two-phase training?":
> Phase 1 protects pretrained backbones from corruption by noisy gradients from the randomly-initialized new components. Phase 2 lets the backbones fine-tune at a gentler learning rate once the rest of the network has stabilized.
