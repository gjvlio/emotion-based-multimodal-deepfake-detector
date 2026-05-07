# Thesis G10 — Session Summary (May 2026)

## Core Architectural Change

**Added multi-task emotion heads** to the original architecture.

Original: Wav2Vec + BERT + ViT → Bilinear Fusion → MLP → P(fake)

Updated:
- Audio Encoder → Emotion Head A (6-class softmax) → emotion_A
- Visual Encoder → Emotion Head B (6-class softmax) → emotion_B
- Δ = |emotion_A − emotion_B| (explicit inconsistency score)
- Bilinear Fusion operates on audio_emb ⊗ visual_emb (NOT on emotion probs)
- Classifier receives: [flatten(bilinear) ; Δ] → MLP → P(fake)
- Emotion heads go DIRECTLY to classifier (red route), NOT into bilinear fusion

**Why:** Makes the core claim ("detects cross-modal emotional inconsistency") empirically verifiable. Without it, a defender can ask "how do you know it's using emotion and not artifacts?"

---

## Training Process

Multi-task loss:
```
L_total = L_BCE(fake/real) + λ_A · L_CE(audio_emotion) + λ_B · L_CE(visual_emotion)
```

Two training phases:
1. Freeze backbones (Wav2Vec, BERT, ViT) — train heads + fusion only
2. Unfreeze all — full fine-tune at lower LR

Label sources per sample:

| Sample type | fake_label | audio_emotion | visual_emotion |
|---|---|---|---|
| Real (MELD, CMU-MOSEI) | 0 | dataset annotation | same as audio |
| Track 1 fake (CREMA-D) | 1 | target emotion | source emotion |
| Track 2 fake (CREMA-D) | 1 | target emotion | target emotion |
| Track 3 fake (CREMA-D) | 1 | target emotion | target emotion |
| Track 4 fake (MELD) | 1 | audio speaker emotion | video speaker emotion |

---

## Dataset Roles (Confirmed)

| Dataset | Real samples | Fake samples |
|---|---|---|
| CREMA-D | NONE (not reused as real) | Tracks 1, 2, 3 (20% / 30% / 50% split) |
| MELD | 50% | Track 4 (50%) |
| CMU-MOSEI | 100% | None |

CREMA-D originals NOT used as real — reusing them would introduce label ambiguity.

---

## Research Questions (Final)

**RQ1:** To what extent does the proposed emotion-based multimodal deepfake detection system accurately distinguish genuine videos from deepfake-generated content across varying generation methods and real-world recording conditions?

**RQ2:** How strongly does the degree of emotional mismatch between audio and visual modalities predict the detection confidence of the proposed framework?

---

## Hypotheses (Final)

**Hypothesis 1: On Detection Accuracy**

H0: There is no significant accuracy in the proposed emotion-based multimodal deepfake detection system in distinguishing genuine videos from deepfake-generated content across varying generation methods and real-world recording conditions.

Ha: There is a significant accuracy in the proposed emotion-based multimodal deepfake detection system in distinguishing genuine videos from deepfake-generated content across varying generation methods and real-world recording conditions.

**Hypothesis 2: On the Predictive Value of Cross-Modal Emotional Inconsistency**

H0: There is no significant relationship between the degree of emotional mismatch between audio and visual modalities and the detection confidence of the proposed framework.

Ha: There is a significant relationship between the degree of emotional mismatch between audio and visual modalities and the detection confidence of the proposed framework.

---

## Paper Sections Updated

### Introduction
- Reordered: Area (Para 1 + threats merged) → Topic (Para 2) → Challenges (Para 3) → Bridge (Para 5)
- Para 4 absorbed into Para 1
- New paragraph added on artifact-detector failure + diffusion degradation
- Citations added: Rossler et al. 2019, Corvi et al. 2023, Jiang et al. 2020, Dolhansky et al. 2020, Ekman & Friesen 1969

### Conceptual Framework
- CREMA-D: no longer "dual role" — fake-only
- MELD: dual role (50% real, 50% fake source)
- CMU-MOSEI: real-only
- WHY-focused rewrites per paragraph
- Ekman & Friesen (1969) citation added for cross-modal coherence claim

### Significance of the Study
- Overclaim removed from AI Academic Community paragraph
- WHY-sentence added to each beneficiary group
- Social Media: viral velocity argument added
- Law Enforcement: traditional forensic tools (metadata, compression) can be spoofed
- Journalists: visual verification tools cannot catch audio-visual deepfakes
- Government: current AI policy lacks empirical technical grounding

### Definition of Terms
- Updated: Deepfake (added DigiFakeAV + synthetic training data), Generative AI (broadened beyond GAN to diffusion + voice conversion), Cross-Modal Emotional Inconsistency (references emotion heads and Δ explicitly)
- Added: Multi-task Learning, Emotion Head

### Statement of the Problem
- Removed old RQs (comparison-based, methodology-answerable)
- Replaced with new RQ1 and RQ2
- Hypotheses rewritten to match new RQs in formal statistical format

### Research Instrument
- Added per-dataset dual-role clarification (CREMA-D fake-only, MELD dual, CMU-MOSEI real-only)
- Replaced "randomly swapping tracks" with accurate 4-track pipeline description
- Added DigiFakeAV to benchmark list
- Added L_CE(audio emotion) variable to variable list

### Sampling Data — Stage 2.1
- Replaced "random track swapping" with structured 4-track pipeline description
- Added 20/30/50% CREMA-D track splits
- Added 1:1 real-to-fake ratio rule
- Added explanation that CREMA-D originals excluded from real class

### Model Training (New Section)
- Written from scratch
- Covers: multi-task loss formula, two-phase training, optimizer config, early stopping, label sources table

---

## Computational Notes

- Bilinear fusion: project both embeddings to 256-dim before outer product (avoids 768×768 = 590K OOM)
- ViT: sample 4–8 keyframes per clip
- Hardware target: RTX 3060 12GB — feasible with fp16 + batch_size=8
- Face detection: RetinaFace with MobileNet backbone (recommended over MTCNN)
- Emotion head output: 6-class softmax (neutral, happy, sad, angry, fear, disgust)
- Unified emotion label space needed: CREMA-D has 6, MELD has 7 (surprise → drop or map to class 6)

---

## False Positive / False Negative Handling

**FP (real flagged as fake):**
- Two-condition gate: require emotion mismatch AND artifact signal
- Raise decision threshold from 0.50 to 0.65
- Temporal consistency: real sarcasm is brief, deepfakes are sustained

**FN (fake passes as real):**
- Frequency domain branch (DCT/FFT) addition
- Temporal flickering signal across frame sequence
- Lower threshold for uncertain zone (0.4–0.65) → human review

---

## Core Argument (For Defense)

Artifact detectors: accuracy tied to specific generator. New generator = degraded accuracy.

Emotion mismatch: generation-method-agnostic. Deepfakes necessarily decouple cross-modal coherence because pipelines process modalities independently — no joint cross-modal emotional model exists. Even diffusion models cannot recreate biological cross-modal coordination (Ekman & Friesen, 1969). The signal degrades only when AI can genuinely feel.

**One-sentence defense answer:**
> Artifact-based detectors are arms race participants. This system detects the invariant property of deepfakes — emotional incoherence across modalities — grounded in Cross-modal Consistency Theory (Ekman & Friesen, 1969).

---

## Classifications

- Computing problem: Optimization (primary), Function problem, Decision problem
- Research method: Experimental — Developmental
- Topic categories: Machine Learning (primary), Artificial Intelligence, NLP, Computer Vision

---

## Key References Added This Session

Corvi, R., Cozzolino, D., Poggi, G., Nagano, K., & Verdoliva, L. (2023). On the detection of synthetic images generated by diffusion models. *ICASSP 2023*. https://doi.org/10.1109/ICASSP49357.2023.10096609

Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., & Ferrer, C. C. (2020). *The DeepFake Detection Challenge (DFDC) dataset*. arXiv. https://arxiv.org/abs/2006.07397

Ekman, P., & Friesen, W. V. (1969). The repertoire of nonverbal behavior: Categories, origins, usage, and coding. *Semiotica*, *1*(1), 49–98. https://doi.org/10.1515/semi.1969.1.1.49

Jiang, L., Li, R., Wu, W., Qian, C., & Loy, C. C. (2020). DeeperForensics-1.0: A large-scale dataset for real-world face forgery detection. *CVPR 2020*, 2889–2898. https://doi.org/10.1109/CVPR42600.2020.00296

Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. *ICCV 2019*, 1–11. https://doi.org/10.1109/ICCV.2019.00009
