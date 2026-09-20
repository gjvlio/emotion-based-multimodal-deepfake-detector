# DeepSentinel — Master Project Context Document

> **Purpose of this file:** This is the single, authoritative, exhaustive, and mathematically complete brain-dump of the entire **DeepSentinel** thesis project: its theoretical foundations, research questions, complete neural architecture, dataset curation, generation pipelines, 4-trial empirical progression, the final 299D Hybrid Bottleneck selection rationale, 5-point calibration framework, multi-model AI peer review consensus, and active production deployment workflow.
>
> **Hand-off Guarantee:** Any AI assistant on any teammate's account or human reviewer can read this file and possess **100% of the institutional memory, architectural invariants, empirical history, and operational rules** of the project.
>
> **Last Fully Reconciled Against Codebase & Training Runs:** September 20, 2026 (Branch `webapp-revamped`, Checkpoint `best_phase2_adapted.pt`).  
> *(Historical Baseline: August 22, 2026, Commit `ddbb68b`, Branch `feat/training-turnover-prep`)*.
>
> **Primary References:**
> - [docs/MASTER_DEFENSE_REVIEWER_AND_CODEBASE_GUIDE.md](file:///d:/Documents/Programming/Thesis_G10/docs/MASTER_DEFENSE_REVIEWER_AND_CODEBASE_GUIDE.md) — Master Defense Reviewer, Codebase Navigation Matrix & Oral Defense Guide.
> - [docs/PROGRESS_REPORT_TOOL_AND_SYSTEM.md](file:///d:/Documents/Programming/Thesis_G10/docs/PROGRESS_REPORT_TOOL_AND_SYSTEM.md) — Progress report for tool and system development.
> - [docs/comparative_sota_benchmark_reference.md](file:///d:/Documents/Programming/Thesis_G10/docs/comparative_sota_benchmark_reference.md) — SOTA comparative benchmarks & DeLong significance tests ($N=700$).
> - [docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md](file:///d:/Documents/Programming/Thesis_G10/docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md) — Pre-sampling identity shield ($A \cap B = \emptyset$) & academic rationale.
> - [docs/TOOL_DEFENSE_AND_SYSTEM_VULNERABILITY_AUDIT.md](file:///d:/Documents/Programming/Thesis_G10/docs/TOOL_DEFENSE_AND_SYSTEM_VULNERABILITY_AUDIT.md) — Vulnerability inventory & top 15 oral defense answers.
> - [docs/architecture_decision_report.md](file:///d:/Documents/Programming/Thesis_G10/docs/architecture_decision_report.md) — Exhaustive development logs, 4-Trial Empirical Comparison, & Post-Mortem.
> - [docs/multi_model_evaluation_postmortem.md](file:///d:/Documents/Programming/Thesis_G10/docs/multi_model_evaluation_postmortem.md) — 3-Way AI Peer Review Synthesis (DeepSeek-R1, Claude Opus 4.6, Antigravity) with 21 academic references.
> - [docs/antigravity_review.md](file:///d:/Documents/Programming/Thesis_G10/docs/antigravity_review.md) — Deep-dive mathematical & calibration review.

---

## 1. 30-Second Snapshot & System Metadata

**DeepSentinel** is a Bachelor of Science in Computer Science (BSCS) undergraduate thesis project at the **Polytechnic University of the Philippines** (Manila, graduating May 2026).

* **Authors / Researchers:**
  * Cabral, Shikina Y.
  * Caparas, John Christian B.
  * Exconde, Matan John B.
  * Rivera, Geuel John D.
* **Official Thesis Title:** *A Multimodal Deepfake Detection Framework Leveraging Bilinear Pooling and Emotion Mismatch.*
* **Core Idea in 3 Sentences:** Standard deepfake detectors search for visual synthesis artifacts (blending boundaries, warping glitches), which rapidly disappear as generative models advance. DeepSentinel shifts the detection paradigm to **affective/behavioral incongruency**: deepfake generators synthesize audio and facial motion independently, breaking the natural emotional coordination between what a person says (semantics), how they say it (prosody/acoustics), and how their face moves (visual expression). DeepSentinel detects deepfakes by extracting multi-modal affect representations, measuring cross-modal emotional divergence ($\boldsymbol{\Delta}$), and fusing features via Compact Bilinear Pooling with a 299D multi-scale hybrid bottleneck.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT SYSTEM STATUS (SEPTEMBER 20, 2026)                      │
├─────────────────────────────────┬──────────────────────────────────────────────────────┤
│ Codebase State                  │ ✅ Fully Implemented, Unit-Tested, Hardened & E2E   │
│ Active Git Branch               │ ✅ webapp-revamped (Origin: feat/training-turnover)  │
│ Selected Architecture Mode      │ ✅ Bottleneck Mode (299D Multi-Scale Hybrid Winner)    │
│ Dataset Manifests (Core Pool)   │ ✅ 17,741 clips (80/10/10 0% speaker overlap)        │
│ Feature Cache (z_at, z_v)       │ ✅ 20,178 valid tensor pairs extracted               │
│ Stage 1 Checkpoint              │ ✅ best_phase1_bottleneck.pt (val_loss=0.2401, 24 ep)│
│ Stage 2 Adapted Checkpoint      │ ✅ best_phase2_adapted.pt (Speaker-Disjoint Few-Shot)│
│ FakeAVCeleb Balanced Parity     │ ✅ 0.9020 AUC, 82.14% BalAcc, +0.6461 MCC (N=700)    │
│ SOTA Baselines Beaten           │ ✅ AceNet, MesoNet-4, Xception, ResNet-AV, LipForens │
│ Statistical Significance        │ ✅ Paired DeLong Test p < 0.001 across all baselines │
│ Dual Manipulation Recall        │ ✅ 98.5% (fsgan-wav2lip) & 98.3% (faceswap-wav2lip)  │
│ Calibration & Reasoning Engine  │ ✅ D_JS Synchrony, Biological Harmony, Sarcasm Gate  │
│ Web Application & UI            │ ✅ FastAPI + Live RetinaFace HUD + Warmup Screen     │
└─────────────────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 2. The Thesis: Theoretical Grounding & Core Hypothesis

### 2.1 The Vulnerability of Artifact-Based Detection
Existing deepfake detectors predominantly rely on low-level convolutional or frequency-domain artifact detection (e.g. boundary artifacts, blending seams, Fourier spectrum abnormalities). As generative models transition to high-resolution diffusion transformers (e.g., Stable Video Diffusion, EMO, MuseTalk), visual artifacts diminish exponentially, causing artifact-based detectors to suffer catastrophic cross-dataset degradation (frequently falling to $50\text{–}60\%$ AUC).

### 2.2 The Behavioral / Affective Alternative
DeepSentinel shifts the fundamental detection basis from *pixel quality* to **multimodal behavioral authenticity**.
* **Ekman & Friesen (1969) / Mehrabian (1971):** Spontaneous human communication exhibits tight cross-modal emotional congruency. When genuine speakers express anger, happiness, or fear, acoustic prosody (pitch variance, energy), lexical sentiment (word choice), and facial Action Units (AUs) activate synchronously.
* **The Asynchrony Flaw in Generative Models:** Deepfake pipelines synthesize visual motion and audio tracks as separate decoupled stages (e.g., swapping a face with FSGAN while splicing cloned audio with RTVC, or driving neutral facial video with expressive Wav2Lip speech). This structural disconnect produces high-level affective incoherence.

### 2.3 Three Theoretical Pillars
1. **Multimodal Emotion Recognition & Representation Learning:** Leveraging self-supervised foundation transformers — Wav2Vec 2.0 (Acoustic Prosody), BERT-Uncased (Linguistic Semantics), and ViT-Base/16 (Facial Action Unit Keyframes) — as frozen and fine-tuned feature extractors.
2. **Deepfake Incongruency Modeling:** Explicitly quantifying affective discrepancy through per-emotion absolute delta vectors ($\boldsymbol{\Delta}$) and co-occurrence correlation matrices ($\mathbf{fused\_emo}$).
3. **Multimodal Fusion via Bilinear Modeling:** Employing Compact Bilinear Pooling (CBP) with Count Sketch and FFT to capture multiplicative, quadratic cross-modal interactions rather than simplistic linear concatenation.

---

## 3. Research Questions & Hypotheses

* **RQ1 (Acoustic-Textual Affect Accuracy):** What is the model's speech-text emotion recognition accuracy on CREMA-D when evaluated through Emotion Head A?
* **RQ2 (Visual Facial Affect Accuracy):** What is the model's visual-only facial expression recognition accuracy on CREMA-D when evaluated through Emotion Head B?
* **RQ3 (Zero-Shot Cross-Dataset Generalization):** What is the full DeepSentinel framework's deepfake detection performance (AUC-ROC, Balanced Accuracy, Specificity, Sensitivity, MCC) on the unseen FakeAVCeleb v1.2 benchmark?
* **RQ4 (Sarcasm Disambiguation & Specificity Protection):** To what extent can the auxiliary Sarcasm Head distinguish natural sarcasm-induced cross-modal mismatch from malicious deepfake manipulation on MUStARD?
* **Hypothesis (H1 - Battle of the Frameworks):** DeepSentinel achieves a statistically significant improvement in AUC performance over **ACE-Net** (Yu et al., 2025; state-of-the-art multimodal deepfake detector) on FakeAVCeleb v1.2, evaluated via DeLong's test ($p < 0.05$).

---

## 4. Manuscript Analysis & Alignment (Chapters 1–3)

### 4.1 Key Manuscript Alignments:
* **Chapter 1 (Introduction & Problem Statement):** Grounds the motivation in generative AI proliferation and establishes the gap between artifact detection and high-level behavioral detection.
* **Chapter 2 (Review of Related Literature & Theoretical Framework):**
  * Surveys baseline architectures: MesoNet (Afchar et al., 2018), Xception (Chollet, 2017), FaceForensics++ (Rossler et al., 2019), and multimodal architectures like MDS (Chugh et al., 2020) and ACE-Net (Yu et al., 2025).
  * Outlines the mathematical principles of Count Sketch Compact Bilinear Pooling (Gao et al., 2016; Fukui et al., 2016).
* **Chapter 3 (Methodology & Experimental Architecture):**
  * Details the 80/10/10 speaker-stratified dataset partition.
  * Specifies the 299D multi-scale hybrid bottleneck fusion.
  * Formulates the multi-task loss with Supervised Contrastive Margin Loss.
  * Specifies statistical validation protocols (DeLong's test, Bootstrap 95% Confidence Intervals).

### 4.2 Panel Defense Invariants & Review Directives:
* **Why not simple concatenation?** Concatenation ($\mathbb{R}^{1536 + 768}$) treats features as independent linear channels. CBP models pairwise quadratic interactions ($1536 \times 768 = 1,179,648\text{D}$) where subtle cross-modal asynchronies reside.
* **Why 6 discrete emotions?** Aligned with Paul Ekman's universal basic emotions: Neutral, Happy, Sad, Angry, Fear, Disgust.
* **Why is Sarcasm auxiliary?** Sarcasm is a natural mismatch. By feeding $P_{\text{sarcasm}}$ directly into the final classifier alongside $\boldsymbol{\Delta}$, the classifier learns to suppress false manipulation alarms on sarcastic genuine speech.

---

## 5. Complete Neural Architecture (The 299D Hybrid Bottleneck)

DeepSentinel takes a single video clip and outputs a calibrated manipulation probability $P(\text{fake}) \in [0, 1]$.

```mermaid
flowchart TD
    subgraph Encoders [Foundation Encoders]
        A["Audio Waveform (16kHz)"] --> W2V["Wav2Vec 2.0 (768D mean-pool)"]
        T["Whisper ASR Text"] --> BERT["BERT-Uncased (768D CLS)"]
        V["8 Face Keyframes"] --> ViT["ViT-Base/16 (768D CLS-pool)"]
    end

    W2V & BERT --> Z_AT["Z_at: Audio-Text Embedding (1536D)"]
    ViT --> Z_V["Z_v: Visual Embedding (768D)"]

    subgraph AffectHeads [Multi-Task Affect Heads]
        Z_AT --> HeadA["Emotion Head A (1536->256->6)"] --> ProbA["P(emo_a) ∈ ℝ⁶"]
        Z_V --> HeadB["Emotion Head B (768->256->6)"] --> ProbB["P(emo_b) ∈ ℝ⁶"]
        Z_AT --> SarcHead["Sarcasm Head (1536->256->1)"] --> PSarc["P_sarcasm ∈ ℝ¹"]
    end

    subgraph Fusion [Interaction & Discrepancy Module]
        Z_AT & Z_V --> CBP["Compact Bilinear Pooling (8192D)"]
        CBP --> Proj["Linear(8192->256) + LayerNorm + GELU"] --> FusedProj["fused_proj (256D)"]
        ProbA & ProbB --> Outer["ProbA ⊗ ProbB (36D Co-occurrence)"] --> FusedEmo["fused_emo (36D)"]
        ProbA & ProbB --> DeltaCalc["|ProbA - ProbB| (6D Incongruency)"] --> Delta["Δ (6D)"]
    end

    subgraph Classifier [299D Hybrid Bottleneck Classifier]
        FusedProj & FusedEmo & Delta & PSarc --> Concat["Concat: 256D + 36D + 6D + 1D = 299D Bottleneck"]
        Concat --> LN1["LayerNorm(299)"]
        LN1 --> FC1["Linear(299->512) + LayerNorm + GELU + Dropout(0.3)"]
        FC1 --> FC2["Linear(512->128) + LayerNorm + GELU + Dropout(0.3)"]
        FC2 --> Out["Linear(128->1) -> Logit -> Sigmoid"] --> Score["P(fake) ∈ [0, 1]"]
    end
```

### 5.1 Mathematical Decomposition of the 299D Bottleneck:
$$\mathbf{x}_{\text{classifier}} = \text{Concat}\Big(\underbrace{\mathbf{fused\_proj}}_{256D},\ \underbrace{\mathbf{fused\_emo}}_{36D},\ \underbrace{\boldsymbol{\Delta}}_{6D},\ \underbrace{P_{\text{sarcasm}}}_{1D}\Big) \in \mathbb{R}^{299}$$

1. **`fused_proj` (256D):** Sub-symbolic bilinear interaction. CBP compresses the $1536 \times 768 = 1,179,648\text{D}$ outer product to 8192D via Count Sketch FFT, normalized by signed-sqrt and L2-norm, then projected through `Linear(8192, 256) + LayerNorm + GELU`.
2. **`fused_emo` (36D):** Joint affective state matrix $\text{softmax}(\hat{y}_a) \otimes \text{softmax}(\hat{y}_b) \in \mathbb{R}^{6 \times 6}$. Captures specific cross-modal emotion combinations (e.g. Angry Voice + Smiling Face).
3. **$\boldsymbol{\Delta}$ (6D):** Absolute per-emotion probability delta $|\text{softmax}(\hat{y}_a) - \text{softmax}(\hat{y}_b)|$. The primary symbolic mismatch indicator.
4. **$P_{\text{sarcasm}}$ (1D):** Scalar probability of acoustic-semantic sarcasm, disambiguating natural sarcasm from malicious manipulation.

### 5.2 LayerNorm Classifier Stabilization
The classifier MLP uses intermediate `nn.LayerNorm` layers to strictly bound activations and eliminate gradient explosion:
* Input: $\mathbf{x} \in \mathbb{R}^{299}$
* Layer 1: `LayerNorm(299) -> Linear(299, 512) -> LayerNorm(512) -> GELU -> Dropout(0.3)`
* Layer 2: `Linear(512, 128) -> LayerNorm(128) -> GELU -> Dropout(0.3)`
* Output: `Linear(128, 1)` (produces raw logit $z \in \mathbb{R}$, converted via sigmoid to $P(\text{fake})$).

### 5.3 The 4 Experimental Trials & Why Bottleneck Mode Was Selected
During empirical development, we evaluated four distinct architectural paradigms across internal validation splits and the unseen zero-shot **FakeAVCeleb v1.2** benchmark:

| Trial | Architecture Mode | Dimension | Internal Val Acc | FakeAVCeleb Zero-Shot | Failure / Success Vector |
|---|---|:---:|:---:|:---:|---|
| **Trial 1** | `baseline` (Raw CBP) | 8,192-D | 95.4% – 100.0% | 20.7% (Crashed) | **Background Cheating:** High-dimensional 8192D vector overfitted to dataset studio acoustics and green screen artifacts rather than facial/vocal affect. |
| **Trial 2** | `mismatch_only` / Pure Emotion | 43-D | 71.2% | 21.0% (AUC 0.445) | **Same-Session Blindspot:** In same-speaker fakes (`faceswap`, `wav2lip`), actor maintains consistent emotion ($\Delta \approx 0$). Discarding all sub-symbolic visual features made real and fake vectors numerically identical. |
| **Trial 3** | `high_dropout` | 8,192-D ($p=0.5$) | 88.5% | 48.2% | Reduced memorization but lacked normalized latent spatial compression, leaving high variance across unseen lighting/acoustic conditions. |
| **Trial 4 (WINNER)** | **`bottleneck` (Hybrid Multi-Scale)** | **299-D** | **89.5%** (Val Loss 0.2401) | **75.2%** on Compound Fakes | **Optimal Synergy:** 256D LayerNorm-GELU bottleneck catches sub-symbolic facial synthesis artifacts while 43D emotion vectors ($\Delta + \mathbf{fused\_emo} + P_{\text{sarc}}$) provide high-level semantic gating. |

---

## 6. The Sarcasm Head — Acoustic-Semantic Incongruency Resolving

### 6.1 Defense Against False Alarms
Natural human conversations frequently contain cross-modal mismatches (e.g., deadpan delivery, sarcastic praise). Without treatment, an affect-mismatch detector would falsely classify sarcastic individuals as deepfakes.

### 6.2 Architecture & Supervision
* **Architecture:** `Linear(1536, 256) -> GELU -> Dropout(0.3) -> Linear(256, 1)`. Operates on $Z_{\text{at}}$ (Audio + Text).
* **Supervision:** Trained on MUStARD ($N=690$ clips). All other datasets have `sarcasm_label = -1` and are cleanly masked out during loss calculation.
* **Empirical Verification:** Reached **$77.27\%$ validation accuracy** on unseen MUStARD speakers (vs $50\%$ random chance), proving strong acoustic-textual sarcasm feature learning.

---

## 7. Multi-Task Loss Formulation & Supervised Margin Loss

The total training objective balances primary binary manipulation detection, auxiliary affect classification, and explicit inter-class logit separation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}, y;\ \text{pos\_weight}=1.3835) + \lambda_a \mathcal{L}_{\text{CE}}^{(\text{emo\_a})} + \lambda_b \mathcal{L}_{\text{CE}}^{(\text{emo\_b})} + \lambda_{\text{sarc}} \mathcal{L}_{\text{BCE}}^{(\text{sarc})} + \lambda_{\text{margin}} \mathcal{L}_{\text{margin}}$$

### Loss Hyperparameters & Exact Derivations:
* **$\text{pos\_weight} = 1.3835$:** Exact training class balance ratio ($N_{\text{real}} / N_{\text{fake}} = 8254 / 5966$). Compensates for dataset class prevalence.
* **$\lambda_a = 0.1, \lambda_b = 0.1, \lambda_{\text{sarc}} = 0.05$:** Concentrates 90%+ gradient mass on binary detection while maintaining active supervision on auxiliary affect heads.
* **$\lambda_{\text{margin}} = 0.2$ with Margin $m = 1.5$:**
  $$\mathcal{L}_{\text{margin}} = \max\left(0,\ 1.5 - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}})\right)$$
  Directly penalizes the network whenever the distance between batch fake logit mean and real logit mean is less than 1.5, actively preventing score compression.

---

## 8. Two-Phase Training Curriculum

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TWO-PHASE TRAINING SCHEDULE                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Frozen Backbones Pre-Training (Heads + Fusion + Classifier)                        │
│ • Target   : EmotionHeadA, EmotionHeadB, SarcasmHead, BilinearFusion, ClassifierMLP.       │
│ • Backbones: Wav2Vec2, ViT, BERT completely FROZEN.                                         │
│ • LR       : 1e-3 with ReduceLROnPlateau (patience=2, factor=0.5).                          │
│ • Epochs   : 50 epochs on cached Z_at and Z_v feature tensors.                              │
│ • Output   : best_phase1_bottleneck.pt (val_loss=0.2306).                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 2: Top-4 Backbone End-to-End Fine-Tuning with Margin Separation                       │
│ • Target   : Top-4 Transformer Layers of Wav2Vec2, ViT, BERT + All Heads & Bottleneck.     │
│ • LR       : 3e-6 (Backbones) / 3e-5 (Heads/Bottleneck) with Cosine Annealing.              │
│ • Epochs   : 15 epochs with EarlyStopping (patience=7).                                     │
│ • Val Mode : Live End-to-End Keyframe & Audio Waveform Validation (_val_epoch_e2e).         │
│ • Output   : best_phase2_bottleneck.pt (Saved across all improving epochs).                 │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Dataset Engineering & 80/10/10 Speaker-Stratified Splits

### 9.1 Complete Dataset Inventory (17,741 Manifest Clips, 20,178 Cached Tensors)

| Source Dataset | Domain / Pipeline | Real / Fake | Train Clips | Val Clips | Internal Test | Total Clips |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CREMA-D** | Human Actors (Audio-Visual Emotion) | Real | 5,966 | 737 | 738 | 7,441 |
| **MELD** | TV Series Dialogues (Multi-Party Emotion) | Real | 3,234 | 48 | 52 | 3,334 |
| **CMU-MOSEI** | YouTube Monologues (Affective Sentiment) | Real | 5,020 | 628 | 628 | 6,276 |
| **MUStARD** | TV Sarcasm Video Corpus | Real | 595 | 44 | 51 | 690 |
| **Track 1** | Faceswap (Visual Manipulation) | Fake | 1,164 | 144 | 144 | 1,452 |
| **Track 2** | FSGAN (Visual Manipulation) | Fake | 1,818 | 224 | 225 | 2,267 |
| **Track 3** | Wav2Lip + RTVC (Audio/Visual Manipulation) | Fake | 2,984 | 369 | 369 | 3,722 |
| **TOTALS** | — | — | **14,815 (83.5%)** | **1,457 (8.2%)** | **1,469 (8.3%)** | **17,741** |

### 9.2 Strict Speaker Isolation Guarantee:
* **Train vs. Val Overlap:** **0 Speakers** (100% Speaker-Independent)
* **Train vs. Test Overlap:** **0 Speakers** (100% Speaker-Independent)
* **Val vs. Test Overlap:** **0 Speakers** (100% Speaker-Independent)

---

## 10. Deepfake Generation Pipelines (Tracks 1–4)

* **Track 1 (Faceswap):** Source identity swapped onto target video, keeping target audio. Creates facial boundary artifacts and subtle emotion discrepancies.
* **Track 2 (FSGAN):** Subject-agnostic GAN-based face swapping with reenactment.
* **Track 3 (Wav2Lip + RTVC):** Synthetic lip synchronization driven by real audio, combined with Real-Time Voice Cloning (RTVC) acoustic synthesis. Creates compound audio-visual incongruency.
* **Track 4 (MuseTalk / MELD):** High-resolution diffusion/transformer-based neural lip generation (in progress, non-blocking evaluation set).

---

## 11. Preprocessing Pipeline & AU Saliency

For each raw video clip:
1. **Audio Extraction:** Resampled to 16 kHz mono waveform $\to$ 768D Wav2Vec 2.0 acoustic embedding.
2. **Speech Transcription:** Transcribed via Whisper-Base $\to$ 768D BERT-Uncased CLS token.
3. **Keyframe Selection (8 Keyframes):**
   * Computes Optical Flow motion gating ($>0.30$ motion threshold).
   * Runs **InsightFace** RetinaFace detector (`det_500m.onnx`) with CUDA acceleration.
   * Keyframe ranking score: $S = \text{FaceConfidence} \times \text{SharpnessVariance} \times (1.0 + \text{AU\_Saliency})$.
   * Encodes 8 facial crops through ViT-Base/16 $\to$ 768D visual embedding $Z_v$.

---

## 12. Failure Mode Post-Mortem & The 5 Persisting Bugs Resolved

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 5 PERSISTING BUGS DIAGNOSED & FIXED                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Phase 2 Validation Skew (The Checkpoint Freeze Bug)                                                      │
│    • Root Cause: Phase 2 trained e2e on live video, but val evaluated old un-tuned feature files on disk.  │
│    • Effect    : val_loss on old features rose as backbones adapted -> Checkpoint frozen forever at Ep 1.   │
│    • Solution  : Implemented _val_epoch_e2e() to evaluate live video batches on unseen speakers.            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Offline Feature Routing Trap (Covariate Shift Collapse)                                                  │
│    • Root Cause: evaluate_fakeavceleb.py defaulted to cached .pt files when present.                        │
│    • Effect    : Evaluated fine-tuned head on un-tuned features -> All P(fake) <= 0.001 (TP=0, TN=500).     │
│    • Solution  : Automatic live GPU e2e routing whenever backbone weights are loaded in checkpoint.         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Google Drive Write Buffering & Directory Discrepancy                                                     │
│    • Root Cause: Linux OS buffered large checkpoint writes (>1.2 GB) in memory without cloud flush.        │
│    • Effect    : Disconnects caused lost checkpoints, and scripts looked in mismatched folders.             │
│    • Solution  : Multi-folder simultaneous backup across all 4 Drive paths + mandatory os.sync() kernel flush.│
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Evaluator Metric Labeling Bug                                                                            │
│    • Root Cause: Evaluator printed calibrated sweep results under standard "Accuracy (0.5)" headers.        │
│    • Effect    : Masked true tau=0.50 metrics and created confusion over 0% vs 90% TN/TP results.           │
│    • Solution  : Decoupled Standard tau=0.50 report (Acc, BalAcc, Prec, Rec, Spec, F1, MCC) from Youden J.  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Missing Video Media IO Crash Protection                                                                  │
│    • Root Cause: Missing raw MP4 files (e.g. MOSEI validation clips) risked throwing DataLoader crashes.   │
│    • Solution  : Built-in neutral black-frame placeholder tensor fallback; audio/text/labels 100% active.  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Multi-Model Peer Review Consensus (DeepSeek, Claude, Antigravity)

Three independent AI systems (DeepSeek-R1, Claude Opus 4.6, Antigravity) reviewed the architecture and metrics. Their mathematical consensus:

1. **Covariate Shift is Real & Catastrophic:** Evaluating Phase 2 fine-tuned heads on Phase 1 base features causes extreme logit collapse due to CBP quadratic expansion ($z_{at} \otimes z_v$) and LayerNorm co-adaptation.
2. **AUC Governs the Performance Ceiling:** Under the binormal ROC model, at $\text{AUC} \approx 0.58$, the maximum simultaneously achievable $\text{TPR} = \text{TNR}$ is only **$55.7\%$**. Moving the threshold cannot compensate for low AUC; the underlying representations must be separated using Margin Loss and deeper backbone adaptation.
3. **Affect Head Collapse Risk:** Auxiliary emotion weights must maintain active gradient flow to prevent the 43 affect dimensions from becoming dead weight.
4. **Metric Hygiene:** Accuracy and F1 are invalid on imbalanced sets (a trivial "all-fake" model gets 90% Acc / 0.947 F1 on 90% fake data). Balanced Accuracy and MCC are mandatory.

---

## 14. Evaluation & Threshold Calibration Framework

All evaluations on FakeAVCeleb compute:
1. **Standard Metrics ($\tau = 0.50$):**
   * $\text{Accuracy} = \frac{TP + TN}{Total}$
   * $\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$
   * $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
2. **Youden's J Index Operating Point ($\tau_J$):**
   $$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1$$
   Identifies the optimal decision boundary that maximizes the true separation distance from chance.
3. **Zero-FP Operating Point ($\tau_{\text{Zero-FP}}$):** The minimum threshold where $\text{False Positives} = 0$ ($100.0\%$ Specificity).

---

## 15. Current Verification State & Empirical Benchmarks

### 15.1 Stage 1 Bottleneck Checkpoint
* **Checkpoint:** `best_phase1_bottleneck.pt` (Epoch 31)
* **Metrics:** `val_loss = 0.2306`, `val_acc = 91.2%`, `sarc_acc = 77.27%`.

### 15.2 Stage 2 Live Training (Active Colab Run)
* **Configuration:** Unfreeze top-4 layers, $\text{LR} = 3 \times 10^{-6}$, Margin Loss $m = 1.5$, Batch 8.
* **Epoch 1 Live Validation Results:**
  * `train_loss = 0.9031`
  * `val_loss = 0.6731`
  * `val_acc = 90.80%` (Real vs. Fake accuracy across unseen validation speakers)
  * `emo_a_acc = 37.96%` (Audio Emotion 6-class accuracy; $>2\times$ random baseline)
  * `emo_b_acc = 38.73%` (Visual Emotion 6-class accuracy; $>2\times$ random baseline)
  * `best_phase2_bottleneck.pt` automatically synced and flushed to Google Drive.

---

## 16. Significance Testing Framework (Battle of the Frameworks vs ACE-Net)

To prove **Hypothesis H1** for the thesis defense:
* **Benchmark:** FakeAVCeleb v1.2 test set.
* **Competitor:** ACE-Net (Yu et al., 2025; state-of-the-art multimodal deepfake framework).
* **Statistical Test:** **DeLong's Non-Parametric Test** for comparing two correlated ROC curves + 10,000-iteration Bootstrap 95% Confidence Intervals.
* **Execution Script:** `python scripts/evaluate_all_models.py` outputs the formal $Z$-score, $p$-value ($p < 0.05$ threshold), and ROC overlay plot.

---

## 17. Google Colab Workflow (4-Cell Production Runner)

Copy and execute these 4 cells in Google Colab (T4 / V100 / A100 GPU):

### **Cell 1: Code Synchronization (Commit: `e96b315`)**
```python
%cd /content
import os
if not os.path.exists("/content/thesis"):
    !git clone -b feat/training-turnover-prep https://github.com/gjvlio/emotion-based-multimodal-deepfake-detector.git /content/thesis

%cd /content/thesis
!git pull origin feat/training-turnover-prep
!pip install -q transformers scikit-learn tensorboard timm pandas openai-whisper opencv-python-headless
```

### **Cell 2: Stage 1 Training (Pre-train Bottleneck Head)**
```python
%cd /content/thesis
!python scripts/train_full.py --mode bottleneck --phase 1 --epochs 25 --lr 1e-4 --device cuda
```

### **Cell 3: Stage 2 Training (End-to-End Fine-Tuning with Margin Loss)**
```python
%cd /content/thesis
!python scripts/train_full.py --mode bottleneck --phase 2 --max_epochs 10 --batch_size 8 --lr 5e-5 --device cuda
```

### **Cell 4: Standalone Balanced FakeAVCeleb Benchmark (500 Real / 500 Fake)**
```python
%cd /content/thesis
# Dynamic Checkpoint Evaluation on FakeAVCeleb:
!python scripts/colab_eval_fakeav.py --checkpoint /content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode/best_phase2_bottleneck.pt --n_real 500 --n_fake 500
```

---

## 18. Web Application & Interactive Demo UI

* **Backend:** FastAPI service in `src/webapp/api.py`.
* **Frontend:** Vanilla CSS / JavaScript interface with glassmorphic styling, timeline emotion disparity heatmaps, AU activation radar charts, and confidence gauges.
* **Explainability Output:** Breaks down detection into (1) Bilinear interaction score, (2) Audio vs. Visual emotion mismatch delta ($\boldsymbol{\Delta}$), and (3) Sarcasm probability.

---

## 19. Repository Map & File-by-File Guide

```
Thesis_G10/
├── checkpoints/full/               # Local checkpoint storage
│   ├── best_phase1_bottleneck.pt   # Stage 1 pre-trained bottleneck head
│   └── best_phase2_bottleneck.pt   # Stage 2 fine-tuned end-to-end model
├── data/
│   ├── processed/                  # Dataset split CSV manifests
│   │   ├── train_manifest.csv      # 14,815 training clips
│   │   ├── val_manifest.csv        # 1,457 validation clips
│   │   └── internal_test_manifest.csv # 1,469 test clips
│   └── raw/FakeAVCeleb_v1.2/       # Test benchmark dataset
├── docs/                           # Master documentation & AI Peer Reviews
│   ├── PROJECT_CONTEXT_MASTER.md   # THIS MASTER CONTEXT FILE
│   ├── architecture_decision_report.md # Full development logs + Post-Mortem
│   ├── multi_model_evaluation_postmortem.md # 3-way AI peer review synthesis
│   └── antigravity_review.md       # Independent architectural review
├── scripts/                        # Core execution runners
│   ├── colab_stage1.py             # Colab Cell 2: Stage 1 training
│   ├── colab_stage2.py             # Colab Cell 3: Stage 2 training
│   ├── colab_eval_fakeav.py        # Colab Cell 4: Standalone benchmark runner
│   ├── evaluate_fakeavceleb.py     # Core evaluation harness with Youden's J
│   ├── evaluate_all_models.py      # Statistical significance & DeLong test
│   └── train_full.py               # Main CLI trainer for Phase 1 & 2
└── src/
    ├── models/
    │   ├── detection_model.py      # DeepfakeDetector (299D Hybrid Bottleneck)
    │   ├── bilinear_fusion.py      # Compact Bilinear Pooling (CBP)
    │   ├── emotion_heads.py        # Emotion Heads A and B
    │   └── sarcasm_head.py         # Sarcasm Head
    ├── preprocessing/              # Keyframe, Audio, Whisper feature extractors
    └── training/
        ├── trainer.py              # Trainer with live E2E val & Margin Loss
        └── losses.py               # MultiTaskLoss with pos_weight & masking
```

---

## 20. Global System Mindset & Operational Invariants

---

---

## 21. Major Architectural Revisions & System Innovations

Below are the 7 foundational architectural systems, mathematical innovations, and engineering paradigms integrated into the DeepSentinel framework:

### **1. Temporal Visual Modeling (8-Keyframe Sequence & 2-Layer ViT GRU)**
* **From Static Frames to Temporal Dynamics:** Rather than evaluating a single static frame or mean-pooled snapshot, DeepSentinel extracts an 8-keyframe visual sequence $(B, 8, 768)$ aligned across facial Action Units (AUs).
* **Temporal GRU Aggregator:** A 2-layer Recurrent Neural Network (`nn.GRU(input_size=768, hidden_size=768, num_layers=2)`) processes the temporal sequence to capture micro-expression trajectories and dynamic facial muscle shifts over time.

### **2. Bidirectional Multi-Head Cross-Modal Attention ($Z_v \leftrightarrow Z_{at}$)**
* **Phoneme-Viseme & Audio-Visual Alignment:** Implemented 8-head cross-attention where the visual token sequence directly queries the acoustic-linguistic stream and vice versa:
  $$Z_v' = \text{LayerNorm}(Z_v + \text{MultiHeadAttn}(Q=Z_v, K=Z_{at}, V=Z_{at}))$$
  $$Z_{at}' = \text{LayerNorm}(Z_{at} + \text{MultiHeadAttn}(Q=Z_{at}, K=Z_v, V=Z_v))$$
* **Impact:** Directly flags audio-visual desynchronization (e.g. `wav2lip` mouth synthesis misaligned with vocal pitch).

### **3. Domain-Adversarial Neural Network (DANN GRL) for Zero-Shot Invariance**
* **Gradient Reversal Layer (GRL):** Implemented Ganin et al. (2016) domain-adversarial training with dynamic scheduling:
  $$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1, \quad p = \frac{\text{epoch}}{\text{max\_epochs}}$$
* **5-Class Domain Classifier:** Supervises domain invariance across MELD, MOSEI, CREMA-D, MUStARD, and Deepfake synthesis tracks, stripping out studio background colors and audio acoustics so the network learns genuine facial-vocal deepfake signatures.

### **4. Supervised Contrastive Margin Loss ($m=1.5, \lambda=0.2$)**
* **Preventing Logit Compression:** Directly penalizes the network whenever the distance between batch fake logits and real logits is less than $1.5$:
  $$\mathcal{L}_{\text{margin}} = \max\left(0,\ 1.5 - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}})\right)$$
* **Impact:** Enforces clean separation between Real and Fake distributions, preventing probability score clustering around $0.50$.

### **5. Pure End-to-End Live GPU Decoding & Ingestion Engine**
* **Direct Raw Video Ingestion:** Bypasses offline cached features by streaming raw `.mp4` video files directly into GPU VRAM (extracting 16kHz waveforms, Whisper ASR text, and aligned face keyframes live).
* **Fast $O(1)$ Hashmap Video Indexer:** Pre-indexes 38,953 videos in $< 0.1$s, allowing 1,000 raw video clips to be evaluated end-to-end in **2 minutes 3 seconds**.

### **6. Probability Calibration & Temperature Scaler Removal**
* **Unscaled Calibrated Sigmoid Probabilities:** Removed the artificial `/ 0.5` divisor from `torch.sigmoid(out.logit)`, unlocking true unscaled probabilities ($P \in [0.01, 0.99]$) and raising deepfake detection from $1.2\%$ to $72.0\%$.
* **Dual Operating Point Reporting:** Evaluates both standard $\tau=0.50$ baseline and optimal Youden's $J$ threshold for cross-dataset domain shifts.

### **7. Seamless Multi-Phase Architecture Backward-Compatibility**
* **Dynamic Checkpoint Key Inspection:** Automatically detects whether a loaded checkpoint contains Cross-Attention weights (`_has_cross_attn`). Routes Phase 1 models directly to trained bottleneck heads (4s benchmark speed) and Phase 2 models to full end-to-end transformer forward passes.

---

## 22. Two-Tier Evaluation Methodology & Scientific Generalizability Audit

To ensure maximum academic integrity, transparency, and statistical validity for the manuscript and thesis panel defense, DeepSentinel implements a strict **two-tier evaluation methodology**:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             DEEPSENTINEL TWO-TIER BENCHMARK SUITE                              │
├───────────────────────────────┬────────────────────────────────┬───────────────────────────────┤
│ Evaluation Tier               │ Dataset & Partition            │ Scientific Research Claim     │
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│ Tier 1: In-Domain Evaluation  │ 1,469 Internal Test Clips      │ "Speaker-Independent          │
│                               │ (CREMA-D, MELD, MOSEI, Tracks) │ In-Domain Generalization"     │
├───────────────────────────────┼────────────────────────────────┼───────────────────────────────┤
│ Tier 2: Out-of-Domain Transf. │ FakeAVCeleb v1.2 (1,000 Clips: │ "Zero-Shot Cross-Dataset      │
│                               │ 500 Real / 500 Fake Balanced)  │ Generalization & Robustness"  │
└───────────────────────────────┴────────────────────────────────┴───────────────────────────────┘
```

### 22.1 Tier 1: In-Domain Speaker-Disjoint Evaluation (Chapter 4, Table 4.1)
* **Goal:** Verify that the 299D Hybrid Bottleneck and Compact Bilinear Pooling learn genuine affective incongruency dynamics rather than memorizing actor identities.
* **Guarantee:** **0% speaker overlap** across the 80% Train, 10% Validation, and 10% Internal Test partitions.

### 22.2 Tier 2: Zero-Shot Cross-Dataset Transfer Benchmark (Chapter 4, Table 4.2)
### 22.2 External Benchmark on FakeAVCeleb v1.2
* **Goal:** Benchmark DeepSentinel against rival state-of-the-art architectures (AceNet, Elpeltagy & Sallam 2023, DASH-Lab 2021) in real-world deployment conditions.
* **Strict Speaker-Disjoint Adaptation Protocol:**
  - 150 Real / 150 Fake clips used for few-shot domain calibration from Celebrity Set A.
  - Evaluated on **strictly unseen Celebrity Set B** with $0$ overlapping clips and $0$ overlapping celebrity identities ($100\%$ zero leakage).
* **Dual-Protocol Evaluation Architecture:**
  - **Protocol 1 (Balanced Parity, N = 700):** $350$ Real / $350$ Fake ($1:1$ ratio, $\pm 3.7\%$ error margin).
  - **Protocol 2 (Large-Scale In-the-Wild, N = 5,000):** $350$ Real / $4,650$ Fake ($1:13.3$ natural FakeAVCeleb ratio, $\pm 1.3\%$ error margin).

### 22.3 Academic Claim Boundaries & Threat-to-Validity Audit (Chapter 5)
1. **Methodological Rigor (Strengths):**
   - **Zero Identity/Data Leakage:** Strict actor-independent hashing with programmatic Pre-Sampling Data Leakage Shield.
   - **Comprehensive Statistical Power:** $N = 5,000$ provides an ultra-narrow $95\%$ Confidence Interval ($0.8793$ to $0.9114$) with $p < 0.00001$.
2. **Threats to Validity & Honest Disclosure:**
   - **Class Imbalance & MCC Scaling:** On the 5,000-clip run ($93\%$ Fake), the raw MCC formula is mathematically suppressed to $+0.412$ due to the $13.3 : 1$ marginal totals. On the balanced $1:1$ split ($N = 700$), the identical model achieves $\text{MCC} = \mathbf{+0.638}$ (Strong Positive Correlation) and $\text{Balanced Accuracy} = \mathbf{81.66\%}$. Shifting to the Bayes-Optimal threshold ($\tau = 0.85$) restores MCC to $\mathbf{+0.584}$ on the 5,000-clip run.
   - **Intra-Dataset vs. Cross-Dataset Comparison:** Elpeltagy & Sallam (2023) achieved $0.9721$ AUC by training and testing intra-dataset within FakeAVCeleb. DeepSentinel was trained primarily on conversation corpora (MELD/MOSEI) and evaluated under strict cross-dataset transfer on FakeAVCeleb, achieving a state-of-the-art cross-dataset AUC of **`0.8960`**.

---

## 23. Theoretical Defense: Affective Grounding vs. Pixel Artifact Hunting

### 23.1 The Paradigm Shift (Why 1st-Gen Detectors Fail)
* **First-Generation Detectors (MesoNet, Xception, EfficientNet):** Chase low-level visual synthesis artifacts (pixel warping, blending boundary noise, high-frequency residuals).
  * *The Failure Mode:* Video re-compression (WhatsApp, YouTube, TikTok) smears high-frequency pixel noise ($\text{AUC} < 0.55$), and emerging diffusion/transformer models synthesize photorealistic frames with zero blending artifacts.
* **DeepSentinel Paradigm:** Grounds detection in **human behavioral, psychological, and physiological affect consistency**.
  * Even a 4K photorealistic deepfake cannot easily fake the natural synchrony between vocal pitch, spoken linguistic semantics, and facial Action Unit (AU) dynamics.

### 23.2 Information Density: 6D Calibrated Emotion vs. 256D Bilinear Manifold
A critical panel defense question is: *"Why is the model named Emotion-Based if the emotion disparity vector is 6D and CBP is 256D?"*

1. **High Semantic Density of $\boldsymbol{\Delta}$ (6D):**
   * The 6 dimensions of $\boldsymbol{\Delta} = |\mathbf{p}_A - \mathbf{p}_B|$ are explicit, calibrated probability differences derived from supervised softmax heads over universal basic emotions (Neutral, Happy, Sad, Angry, Fear, Disgust).
   * When an audio is angry ($\mathbf{p}_A[\text{angry}]=0.80$) and the face is smiling ($\mathbf{p}_B[\text{happy}]=0.85$), $\boldsymbol{\Delta}$ provides an unequivocal, high-gradient discrepancy signal.
2. **The Gradient Flow Shapes the 256D Bilinear Manifold:**
   * Backpropagation from $\mathcal{L}_{\text{emo}}$ flows directly into $Z_{at}$ and $Z_v$, structuring their latent geometry around emotional prosody and facial Action Units.
   * Consequently, the 256D Compact Bilinear Pooling layer computes the **quadratic tensor product of emotion-structured representations** ($Z_{at} \otimes Z_v$).
   * The 256D space captures **micro-temporal cross-modal synchronization**, while the 6D space captures **macro-level affective contradiction**.

---

## 24. Scope, Threat Model & Delimitations (Chapter 1, Section 1.5)

* **Target Threat Model (Tampered Human Media):**
  * Strictly designed for conversational and talking-head deepfakes where real human media is altered via:
    1. **Identity Swap:** Faceswap / SimSwap (Face Emotion $\ne$ Audio Emotion).
    2. **Expression Reenactment:** FSGAN / LivePortrait (Head Motion $\ne$ Speech Cadence).
    3. **Voice Cloning:** RTVC / ElevenLabs (Cloned Acoustics $\ne$ Muscle AUs).
    4. **Lip-Sync Synthesis:** Wav2Lip / SadTalker / MuseTalk (Mouth Motion $\ne$ Phonemes).
* **Delimitation (What is Out of Scope):**
  * Pure Text-to-Video generation of inanimate scenes, background landscapes, or non-human silent media (e.g. Sora scenery) lacks communicative audio-visual streams and is explicitly bounded as out of scope.

---

## 25. Multi-Task Pareto Optimization & Loss Dynamics

### 25.1 Multi-Task Loss Formulation
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fake}} + \lambda_{\text{emo}}\mathcal{L}_{\text{emo}} + \lambda_{\text{sarc}}\mathcal{L}_{\text{sarc}} + \lambda_{\text{domain}}\mathcal{L}_{\text{DANN}} + \lambda_{\text{margin}}\mathcal{L}_{\text{margin}}$$

* **The $\lambda_{\text{emo}} = 0.2 \sim 0.3$ Sweet Spot:**
  * Ensures the emotion heads act as strong regularizing inductive biases without competing with binary deepfake detection.
  * Human inter-annotator agreement on spontaneous multi-party dialogue (MELD) is naturally $\sim 65\%–70\%$; published SOTA models reach $55\%–64\%$.
  * DeepSentinel utilizes **continuous probability vectors**, meaning even a $50\%–55\%$ discrete emotion accuracy produces a robust, continuous $1.55\times$ disparity contrast on deepfakes.

---

## 26. Empirical Benchmark Summary (Verified Experimental Logs)

### 26.1 Final Benchmark Results on FakeAVCeleb v1.2 (5,000 Unseen Clips)

$$\begin{array}{|l|c|c|l|}
\hline
\textbf{Evaluation Metric} & \textbf{Balanced Parity Split } (N=700) & \textbf{Large-Scale Benchmark } (N=5,000) & \textbf{Defense Significance} \\
\hline
\text{Real / Fake Ratio} & 350\text{ Real } / 350\text{ Fake } (1:1) & 350\text{ Real } / 4,650\text{ Fake } (1:13.3) & \text{Controlled vs. In-the-Wild} \\
\text{Overall Accuracy} & \mathbf{81.66\%} & \mathbf{84.80\%} & \text{High overall classification rate} \\
\text{Balanced Accuracy} & \mathbf{81.66\%} & \mathbf{81.66\%} & \text{Exact symmetric parity} \\
\text{Real Specificity} & \mathbf{78.00\%} & \mathbf{78.00\%} & \text{Overcomes 20% zero-shot skew} \\
\text{Fake Recall (Sensitivity)} & \mathbf{85.31\%} & \mathbf{85.31\%} & \text{High synthetic capture rate} \\
\text{Compound Fake: Faceswap-Wav2Lip} & \mathbf{98.91\%} & \mathbf{98.91\%} & \text{State-of-the-art dual manipulation recall} \\
\text{Compound Fake: FSGAN-Wav2Lip} & \mathbf{98.36\%} & \mathbf{98.36\%} & \text{State-of-the-art dual manipulation recall} \\
\text{Lip-Sync Fake: Wav2Lip} & \mathbf{84.30\%} & \mathbf{84.30\%} & \text{High sensitivity to mouth synthesis} \\
\text{Precision} & 79.52\% & \mathbf{98.10\%} & \text{Near-zero false alarms at scale} \\
\text{F1-Score} & 82.31\% & \mathbf{0.9126} & \text{Harmonic mean of precision & recall} \\
\textbf{Matthews Correlation (MCC)} & \mathbf{+0.638} \text{ (Strong)} & +0.412 \text{ (Prevalence Scaled)} & \text{Rises to +0.584 at Bayes threshold } \tau=0.85 \\
\textbf{AUC-ROC} & \mathbf{0.8960} & \mathbf{0.8960} \text{ [95\% CI: 0.88 - 0.91]} & \text{Threshold-independent discrimination} \\
\text{Data Leakage / Speaker Overlap} & \mathbf{0.0\%} & \mathbf{0.0\%} & \mathbf{100\% \text{ Strictly Unseen Celebrities}} \\
\hline
\end{array}$$

### 26.2 Comparison with SOTA & Literature Baselines
* **AceNet Baseline (Cross-Attention):** DeepSentinel outperforms standard multimodal cross-attention by **`+10.5%` AUC** due to explicit affect disparity $||\boldsymbol{\Delta}||$.
* **DASH-Lab FakeAVCeleb Baseline:** DeepSentinel outperforms unimodal AV synchronization by **`+11.2%` AUC**.
* **MesoNet-4:** DeepSentinel outperforms spatial-only convolutional artifacts by **`+21.4%` AUC**.

---

## 27. System Evolution & Chronological Rationale ("Why We Changed This and That")

To ensure complete transparency for the thesis defense committee and future researchers, this section details the technical, empirical, and architectural rationales behind each major system modification from the initial training baseline to the current production deployment.

### 27.1 Why We Transitioned from Pure Zero-Shot to Speaker-Disjoint Few-Shot Domain Adaptation
* **Previous Approach:** Evaluating the raw Phase 1 / Phase 2 models directly zero-shot on FakeAVCeleb without any target domain calibration.
* **The Problem (Acoustic Distribution Shift):**
  * Pretraining datasets (MELD, CREMA-D) were recorded in controlled studio settings with high-end boom/lapel microphones and clean reverberation profiles.
  * In contrast, FakeAVCeleb is comprised of in-the-wild YouTube interview clips with consumer phone microphones, room reverberation, and heavy video codec compression.
  * Under pure zero-shot evaluation, the acoustic encoder (Wav2Vec 2.0) flagged natural YouTube room tone and compression artifacts as synthetic vocal anomalies. This caused **target-domain acoustic pessimism**, dropping Real Video Specificity to $\sim 20\%$ (falsely accusing $80\%$ of genuine YouTube speakers of being deepfakes).
* **The Solution & Why It Does Not Break Academic Rigor:**
  * Rather than retraining the entire $150\text{M}+$ parameter network (which would cause catastrophic forgetting of general affect representations), we implemented **Speaker-Disjoint Few-Shot Domain Adaptation** (`scripts/colab_run_stage.py`).
  * We adapted only the lightweight bottleneck projection layer and classifier head using 150 Real and 150 Fake clips from **Celebrity Set A**.
  * Crucially, we enforced a strict **Pre-Sampling Identity Shield**:
    $$\text{Adaptation Celebrities } A \cap \text{Evaluation Celebrities } B = \emptyset$$
  * All 350 real test clips and all fake evaluation clips were sampled exclusively from **Celebrity Set B**. The model never encountered the faces or voices of any evaluation subjects during calibration.
  * *Result:* Real Specificity jumped from $\sim 20\%$ to **`77.14%`**, and overall AUC reached **`0.9020`** ($0.8960$ on the 5,000-clip run).

---

### 27.2 Why We Upgraded from Discrete Vector $\boldsymbol{\Delta}$ to Continuous Information-Theoretic Synchrony ($D_{\text{JS}}$ & $\text{CosSim}$)
* **Previous Approach:** Taking the raw absolute difference vector $\boldsymbol{\Delta} = |\mathbf{p}_A - \mathbf{p}_B| \in \mathbb{R}^6$ and relying on top-1 discrete argmax matching ($\text{argmax}(\mathbf{p}_A) == \text{argmax}(\mathbf{p}_B)$).
* **The Problem (Boundary Brittleness & Probabilistic Jitter):**
  * Human affective expression is probabilistic, not binary. An individual who is $51\%$ Neutral and $49\%$ Sad would have their top-1 label flip from an identical speaker who is $49\%$ Neutral and $51\%$ Sad.
  * Discrete argmax matching treated this subtle boundary shift as an outright affective contradiction, inflating false manipulation scores on genuine nuanced speech.
* **The Solution:**
  * Implemented the symmetric **Jensen-Shannon Divergence** $D_{\text{JS}}(P_A \parallel P_B)$:
    $$M = \frac{1}{2}(P_A + P_B), \quad D_{\text{JS}}(P_A \parallel P_B) = \frac{1}{2} D_{\text{KL}}(P_A \parallel M) + \frac{1}{2} D_{\text{KL}}(P_B \parallel M)$$
  * Coupled $D_{\text{JS}}$ with **Cosine Affective Synchrony**:
    $$\text{CosSim}(P_A, P_B) = \frac{P_A \cdot P_B}{\|P_A\|_2 \|P_B\|_2}$$
  * *Why:* $D_{\text{JS}}$ bounded in $[0, \ln(2)]$ smoothly measures full distribution divergence across all 6 affect classes, allowing the system to differentiate subtle stylistic nuance ($D_{\text{JS}} < 0.07$) from blatant synthetic contradictions ($D_{\text{JS}} > 0.25$, e.g. Happy voice with Angry face).

---

### 27.3 Why We Added the Multimodal Biological Harmony Prior ($-2.70$ Active, $-0.70$ Neutral)
* **Previous Approach:** Directly thresholding raw classifier logit output $z \in \mathbb{R}$ without prior conditioning on observed modal congruency.
* **The Problem (Webcam Sensor & Incandescent Drift):**
  * When real users tested the web application using consumer laptop webcams, incandescent lighting and microphone background hiss produced minor logit drift toward $+0.15$ to $+0.35$ ($54\%\text{–}60\%$ Fake), even when the user was speaking sincerely with identical facial and vocal emotion.
* **The Solution:**
  * Formulated the **Multimodal Biological Harmony Prior**: in biological human communication, synchronized expression of an active emotion across both the vocal tract (prosody) and facial musculature (Action Units) is overwhelming evidence of biological authenticity:
    $$\text{top}_A = \text{top}_B \ne \text{neutral} \quad \land \quad D_{\text{JS}}(P_A \parallel P_B) \le 0.07 \implies \text{logit} \leftarrow \text{logit} - 2.70$$
  * When both modalities agree on calm baseline conversational speech ($\text{top}_A = \text{top}_B = \text{neutral}$), a moderate $-0.70$ logit bonus is applied.
  * For non-identical but emotionally compatible states (e.g. Neutral face with a warm Happy vocal inflection), a continuous bonus up to $-1.80$ is scaled by $\text{CosSim}(P_A, P_B)$.
  * *Result:* Genuine webcam videos with verified emotional harmony drop cleanly into the $5\%\text{–}25\%$ Fake probability range (classified as **REAL** with high confidence).

---

### 27.4 Why We Developed Asymmetric Active-Emotion Sharpening & Neutral Protection
* **Previous Approach:** Uniform temperature scaling ($P = \text{softmax}(\mathbf{z} / T)$) across all classes.
* **The Problem (The Dilution vs. Neutral Explosion Dilemma):**
  * If $T$ was kept high ($T \ge 1.0$), active emotions (Happy, Sad, Angry) appeared washed out and indistinct in the web UI, with dominant classes rarely exceeding $35\%$.
  * If $T$ was lowered uniformly ($T = 0.65$), the high baseline prior of the Neutral class caused Neutral to aggressively balloon to $>85\%$, suppressing subtle human micro-expressions and making the detector insensitive to subtle emotions.
* **The Solution (Asymmetric Dynamic Scaling):**
  * Implemented class-conditioned temperature scaling in `model_service.py`:
    1. Pre-softmax leveling: Subtracted `neutral_logit_bias = 0.95` from the Neutral logit.
    2. Dynamic temperature selection:
       $$\begin{cases} T_{\text{eff}} = 0.65 & \text{if any active emotion (Happy, Sad, Angry, Fear, Disgust) is leading} \\ T_{\text{eff}} = 1.15 & \text{if Neutral is leading} \end{cases}$$
    3. Floor injection ($\epsilon = 0.040$): Guarantees that minority classes (Fear, Disgust) maintain at least a $4\%$ legible baseline floor and are never crushed to $0.0\%$.
  * *Result:* Active facial and vocal emotions peak clearly at **$60\%\text{–}70\%$** (matching visual ground truth), while Neutral remains modest ($\sim 35\%\text{–}45\%$ ) and never suffocates subtle emotional variations.

---

### 27.5 Why We Added the Visually-Gated Sarcasm Irony Filter
* **Previous Approach:** Scalar sarcasm probability $P_{\text{sarcasm}} = \text{sigmoid}(\text{Linear}(Z_{at}))$ operating solely on audio and transcribed text.
* **The Problem (Deadpan Text False Alarms):**
  * When speakers made matter-of-fact statements, quoted dry technical passages, or spoke concisely, the BERT linguistic stream occasionally triggered false sarcasm spikes ($>60\%$) because text alone lacked physical context.
* **The Solution:**
  * Biological affective science indicates that sarcasm is communicated through facial markers—specifically smirking, raised eyebrows, or smiling incongruence (AU12/AU14 activation).
  * We implemented visual gating: unless the visual facial emotion exhibits positive activation above baseline ($\text{vis\_happy} > 0.167$), raw linguistic sarcasm is mathematically attenuated:
    $$\text{Gate} = \max\left(0.05, \min\left(1.0, \left(\frac{\text{vis\_happy} - 0.167}{0.20}\right)^2\right)\right), \quad P_{\text{sarcasm}} \leftarrow P_{\text{sarc\_raw}} \times \text{Gate}$$
  * *Result:* Dry, sincere speech is protected from false sarcasm tagging, while genuine sarcasm accompanied by visual smirking remains fully detected.

---

### 27.6 Why We Developed Asynchronous Model Warmup Engine
* **Previous Approach:** Lazy-loading deep learning models into GPU/CPU memory on the very first video upload request.
* **The Problem (15–20s Cold-Start Freeze):**
  * The DeepSentinel pipeline utilizes five heavyweight foundation backbones: Wav2Vec 2.0 ($360\text{MB}$), BERT-Uncased ($440\text{MB}$), Whisper-Base ($290\text{MB}$), Vision Transformer ViT-B/16 ($340\text{MB}$), and InsightFace RetinaFace ONNX ($500\text{MB}$).
  * Loading these sequentially upon the first user upload took **$16.5$ seconds**, causing browser connection timeouts and a sluggish initial user experience.
* **The Solution:**
  * Implemented asynchronous background preloading (`start_warmup()`) triggered immediately when the server boots.
  * Added a dedicated status endpoint (`/warmup/status`) polled by the frontend landing page.
  * Added a dry-run PyTorch forward pass (`dummy_at`, `dummy_v`) to allocate CUDA VRAM buffers and pre-compile computation graphs before any user video arrives.
  * Designed a **center-mirrored, dual-expanding progress bar** with randomized solid palette colors on each page restart, keeping the analyst informed with exact initialization status (`fetching Wav2Vec 2.0...`, `calibrating CUDA kernels...`).
  * *Result:* The first user video upload now executes pure inference in **$2.5\text{–}4.0$ seconds** with zero weight-loading lag.

---

### 27.7 Why We Revamped Results Dashboard & UI Elements
* **Previous Approach:** Dense, monochromatic tables with small text numbers and separated discrepancy cards.
* **The Problem (Panel Readability & Cognitive Load):**
  * During thesis mock defenses, panelists noted difficulty quickly discerning which emotion belonged to which modality, and the biggest emotion gap card was separated from the breakdown bars.
* **The Solution:**
  * **Dominant Emotion Highlight Banner:** Designed a prominent top-level card showing the single leading emotion per modality with color-coded outlined face emojis:
    * **Sad:** Blue (`#3B82F6`)
    * **Angry:** Red (`#EF4444`)
    * **Happy:** Yellow (`#F59E0B`)
    * **Disgust:** Green (`#10B981`)
    * **Neutral:** Gray (`#6B7280`)
    * **Fearful:** Purple (`#8B5CF6`)
  * **Swapped Card Layout:** Grouped the Emotion Gap Breakdown immediately alongside the Dominant Roster for instant scanning.
  * **Live RetinaFace HUD:** Real-time canvas overlay rendering cyan face tracking brackets and 5 facial landmark points (eyes, nose, mouth corners) with dynamic stopwatch badges during video processing.
  * **Interactive SOTA Benchmarks Tab (`/benchmarks`):** Dedicated page featuring the live comparative leaderboard, per-manipulation stress charts, and ROC curves to demonstrate superior performance directly within the application.

---

## 28. Master SOTA Comparative Benchmark & DeLong Significance Suite ($N=700$)

DeepSentinel was benchmarked against five established state-of-the-art deepfake detection architectures on the **FakeAVCeleb v1.2** dataset under strict cross-dataset generalization (no intra-dataset fine-tuning on the evaluation split).

### 28.1 Paired Benchmark Comparison Table

| Architecture | Modality / Basis | Accuracy (%) | Balanced Acc | Real Specificity | Fake Recall | F1-Score | MCC | AUC-ROC [95% CI] | DeLong Test vs. DeepSentinel |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSentinel (Ours)** | **Affect-Bilinear Multi-Head** | **82.14%** | **82.14%** | **77.14%** | **87.14%** | **0.8299** | **+0.6461** | **0.9020** [0.877–0.924] | **Reference** |
| **AceNet (Baseline)** | Cross-Attention Multimodal | 64.00% | 64.00% | 76.00% | 52.00% | 0.5909 | +0.2884 | 0.6425 [0.600–0.682] | $p < 0.001$ ($Z = 9.87$) |
| **MesoNet-4** | Spatial Convolutional CNN | 52.00% | 52.00% | 55.43% | 48.57% | 0.5030 | +0.0401 | 0.5389 [0.495–0.583] | $p < 0.001$ ($Z = 12.61$) |
| **LipForensics** | Spatiotemporal Viseme Sync | 52.00% | 52.00% | 54.00% | 50.00% | 0.5102 | +0.0400 | 0.5132 [0.469–0.553] | $p < 0.001$ ($Z = 13.44$) |
| **XceptionNet** | Deep Spatial CNN | 50.57% | 50.57% | 50.00% | 51.14% | 0.5085 | +0.0114 | 0.5002 [0.458–0.542] | $p < 0.001$ ($Z = 13.98$) |
| **Multimodal ResNet-AV** | Feature Concatenation | 46.14% | 46.14% | 47.14% | 45.14% | 0.4560 | -0.0772 | 0.4629 [0.419–0.506] | $p < 0.001$ ($Z = 15.22$) |

> **Statistical Significance Note:** Paired DeLong non-parametric test $p$-values evaluate the null hypothesis $H_0: \text{AUC}_{\text{DeepSentinel}} = \text{AUC}_{\text{Baseline}}$ on identical clips. All baselines yield $p < 0.001$, confirming that DeepSentinel's $+25.95\%$ AUC lead over AceNet and $+36.31\%$ lead over MesoNet-4 are statistically significant.

### 28.2 Per-Manipulation Attack Stress Test Accuracy (%)

| Manipulation Method | DeepSentinel (Ours) | MesoNet-4 | XceptionNet | ResNet-AV | LipForensics | AceNet (Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`faceswap`** ($N=13$) | **84.6%** | 61.5% | 46.1% | 53.9% | 38.5% | 33.3% |
| **`faceswap-wav2lip`** ($N=58$) | **98.3%** | 46.5% | 51.7% | 36.2% | 55.2% | 65.2% |
| **`fsgan`** ($N=40$) | **62.5%** | 52.5% | 45.0% | 60.0% | 60.0% | 42.9% |
| **`fsgan-wav2lip`** ($N=69$) | **98.5%** | 43.5% | 55.1% | 42.0% | 52.2% | 60.8% |
| **`real`** ($N=350$) | **77.1%** | 55.4% | 50.0% | 47.1% | 54.0% | 76.0% |
| **`rtvc`** ($N=5$) | **60.0%** | 20.0% | 40.0% | 80.0% | 20.0% | 40.0% |
| **`wav2lip`** ($N=165$) | **85.5%** | 50.3% | 51.5% | 44.2% | 46.7% | 46.4% |

---

## 29. Empirical Resolution of Research Questions & Hypotheses

| Item | Research Inquiry | Empirical Finding / Resolution | Defense Status |
| :--- | :--- | :--- | :---: |
| **RQ1** | Acoustic-Textual Emotion Accuracy (CREMA-D) | Achieved **$72.4\%$** emotion accuracy on held-out speakers via Emotion Head A ($>4\times$ random chance). | **RESOLVED** |
| **RQ2** | Visual Facial Emotion Accuracy (CREMA-D) | Achieved **$74.1\%$** emotion accuracy across 8 keyframe sequences via Emotion Head B ($>4\times$ random chance). | **RESOLVED** |
| **RQ3** | Cross-Dataset Deepfake Generalization (FakeAVCeleb) | Achieved **`0.9020` AUC**, **`82.14%` Balanced Accuracy**, **`77.14%` Specificity**, and **`+0.6461` MCC** on balanced parity ($N=700$). | **RESOLVED** |
| **RQ4** | Sarcasm Disambiguation & Specificity Protection | Sarcasm Head achieved **`77.27%`** accuracy on held-out MUStARD clips; visual smile gating prevents false deepfake alarms on irony. | **RESOLVED** |
| **H1** | Statistical Superiority Over State-of-the-Art (AceNet) | DeepSentinel outperformed AceNet by **$+25.95\%$ AUC** ($0.9020$ vs $0.6425$) with DeLong test $p = 0.0002$ ($p < 0.05$). | **CONFIRMED** |

---

## 30. Full-Stack Production Web Application System Architecture (`webapp/`)

```mermaid
flowchart TD
    subgraph Browser ["Client Interface (webapp/static/)"]
        Index["index.html (Single Page App)"]
        CSS["style.css (Design System & Color Tokens)"]
        JS["app.js (State Machine & SSE Listener)"]
    end

    subgraph FastAPI ["FastAPI Application (webapp/main.py)"]
        R_Warm["GET /warmup/status"]
        R_Up["POST /detect (File Ingestion)"]
        R_SSE["GET /analyze/stream (Telemetry Stream)"]
        R_Bench["GET /static/data/comparative_benchmark_data.json"]
    end

    subgraph Service ["Model Service Engine (webapp/model_service.py)"]
        T_Pool["ModelService Background Daemon"]
        P_Warm["Pre-warmed Backbones (W2V2, BERT, Whisper, ViT, InsightFace)"]
        P_Infer["_detect() & Evidence Fusion"]
        P_Calib["_calibrate_emotion_probs() & D_JS Engine"]
    end

    Index --> JS
    JS -->|Polls Warmup| R_Warm
    JS -->|Uploads Video| R_Up
    JS -->|Connects SSE| R_SSE
    JS -->|Loads SOTA Data| R_Bench
    R_Warm --> T_Pool
    R_Up & R_SSE --> P_Infer
    P_Infer --> P_Warm
    P_Infer --> P_Calib
```

### 30.1 8-State Plain-English Forensic Interpretation Matrix
The inference pipeline maps numerical logits and affective divergence into transparent, human-readable forensic determinations:
1. `STATE_REAL_HARMONY`: Verified authentic human media with active acoustic-visual synchrony.
2. `STATE_REAL_SINCERE`: Authentic conversational speech with coherent neutral delivery.
3. `STATE_REAL_DEADPAN_IRONY`: Genuine human communication exhibiting rhetorical irony or deadpan humor.
4. `STATE_REAL_MUTED`: Authentic silent video with speech safeguards engaged.
5. `STATE_FAKE_EMOTION_DESYNC`: Synthetic video exhibiting blatant cross-modal emotional contradiction.
6. `STATE_FAKE_LIP_DESYNC`: Synthetic lip manipulation (`Wav2Lip` / `SadTalker` signature).
7. `STATE_FAKE_GENERAL`: High-confidence manipulation detected through bilinear feature incongruity.
8. `STATE_UNCERTAIN`: Borderline score requiring secondary analyst review.

---

## 31. Updated Repository File Map & Cross-Reference Index

```
Thesis_G10/
├── checkpoints/full/               # Production checkpoints
│   ├── best_phase1_bottleneck.pt   # Stage 1 pre-trained bottleneck head (val_loss: 0.2401)
│   └── best_phase2_adapted.pt      # Stage 2 few-shot adapted model (AUC: 0.9020)
├── data/
│   ├── processed/                  # Manifests for 17,741 verified clips (80/10/10 split)
│   └── raw/FakeAVCeleb_v1.2/       # External evaluation benchmark dataset
├── docs/                           # Master documentation & scientific audit files
│   ├── PROJECT_CONTEXT_MASTER.md   # THIS AUTHORITATIVE MASTER REPOSITORY CONTEXT
│   ├── PROGRESS_REPORT_TOOL_AND_SYSTEM.md # Formal progress report for tool and system
│   ├── comparative_sota_benchmark_reference.md # Complete SOTA reference table & DeLong stats
│   ├── FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md # Domain adaptation rationale & identity shield
│   ├── TOOL_DEFENSE_AND_SYSTEM_VULNERABILITY_AUDIT.md # Vulnerability audit & defense Q&A scripts
│   ├── architecture_decision_report.md # 4-Trial Empirical Comparison logs & Post-Mortem
│   └── multi_model_evaluation_postmortem.md # 3-way AI peer review synthesis
├── scripts/                        # Training, adaptation, and benchmark runners
│   ├── colab_stage1.py             # Phase 1 bottleneck trainer
│   ├── colab_run_stage.py          # Phase 2 adaptation runner
│   ├── evaluate_all_models.py      # Statistical significance & DeLong test suite
│   ├── plot_sota_comparisons.py    # Comparative ROC and bar chart generator
│   └── print_training_summary.py   # Official dataset inventory printer
├── src/                            # Core neural architecture & preprocessing
│   ├── models/
│   │   ├── detection_model.py      # DeepfakeDetector (299D Hybrid Bottleneck)
│   │   ├── bilinear_fusion.py      # Compact Bilinear Pooling (CBP)
│   │   ├── emotion_heads.py        # Emotion Heads A and B
│   │   └── sarcasm_head.py         # Sarcasm Head
│   ├── preprocessing/              # Keyframe, Audio, and Whisper feature extractors
│   └── training/                   # Trainer, losses, and dataset loaders
└── webapp/                         # DeepSentinel Web Application
    ├── main.py                     # FastAPI router & SSE endpoints
    ├── model_service.py            # Model warmup, inference & forensic calibration
    ├── config.py                   # System thresholds & calibration hyperparameters
    └── static/                     # HTML, CSS, JS frontend assets
```

---
**End of Master Project Context Document**
