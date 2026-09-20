# DeepSentinel: Master Defense Reviewer & Codebase Navigation Guide

> **Official Thesis Title:** *A Multimodal Deepfake Detection Framework Leveraging Bilinear Pooling and Emotion Mismatch*  
> **Institutional Affiliation:** Polytechnic University of the Philippines — Department of Computer Science (BSCS 2026)  
> **Research Team:** Cabral, Shikina Y. | Caparas, John Christian B. | Exconde, Matan John B. | Rivera, Geuel John D.  
> **Purpose of this Document:** The definitive, all-in-one oral and tool defense preparation manual. Maps every single theoretical claim, mathematical equation, empirical metric, and defense question directly to its exact source code implementation in the repository with clickable file hyperlinks.

---

## 1. The 60-Second Elevator Pitch & Core Thesis Invariants

### 1.1 The Central Thesis Statement
Standard deepfake detectors search for visual synthesis artifacts (blending seams, warping glitches, frequency anomalies). As generative architectures transition to high-resolution diffusion transformers (e.g., EMO, MuseTalk, Sora), pixel-level artifacts rapidly vanish, causing conventional detectors to suffer catastrophic cross-dataset collapse (frequently falling to $50\%\text{–}55\%$ AUC).

**DeepSentinel** shifts the detection basis from *pixel quality* to **multimodal affective and behavioral authenticity**:
* In genuine human communication, facial Action Units, vocal prosody, and linguistic semantics are biologically and evolutionarily coupled (Ekman & Friesen, 1969; Mehrabian, 1971).
* Generative pipelines synthesize audio and video in decoupled silos (e.g., swapping a face via FSGAN while splicing cloned audio via RTVC, or animating a neutral face via Wav2Lip). This decoupling produces **cross-modal affective incongruence**.
* DeepSentinel extracts tri-modal features (Wav2Vec 2.0, BERT, ViT), projects them through **Compact Bilinear Pooling** ($1,179,648\text{D} \to 8,192\text{D} \to 256\text{D}$), computes multi-task emotion discrepancy ($\boldsymbol{\Delta} \in \mathbb{R}^6$) and joint co-occurrence ($\mathbf{fused\_emo} \in \mathbb{R}^{36}$), gates natural rhetorical irony via an auxiliary **Sarcasm Head** ($P_{\text{sarc}} \in \mathbb{R}^1$), and classifies via a **299D Multi-Scale Hybrid Bottleneck**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 END-TO-END PIPELINE DATA FLOW                                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
  Raw Video (.mp4)
       │
       ├──▶ Audio (16kHz Mono) ────────▶ Wav2Vec 2.0 (768D) ──┐
       ├──▶ Audio (Whisper ASR) ───────▶ BERT-Uncased (768D) ──┼─▶ Z_at (1536D) ──┐
       └──▶ Video (RetinaFace 8 Frames) ─▶ ViT-B/16 (768D) ────┼─▶ Z_v  (768D)  ──┼─▶ CBP (8192D) ──▶ fused_proj (256D)
                                                               │                  │
                                                               ├─▶ EmotionHeadA ─▶ P_A (6D) ──┬─▶ Δ = |P_A - P_B| (6D)
                                                               ├─▶ EmotionHeadB ─▶ P_B (6D) ──┼─▶ fused_emo = P_A ⊗ P_B (36D)
                                                               └─▶ SarcasmHead  ──────────────┴─▶ P_sarc (1D)
                                                                                                        │
                                                                                      ┌─────────────────┴─────────────────┐
                                                                                      │ 299D Hybrid Bottleneck Classifier │
                                                                                      │   256D + 36D + 6D + 1D = 299D     │
                                                                                      └─────────────────┬─────────────────┘
                                                                                                        ▼
                                                                                      Raw Logit -> Harmony Calibration -> Verdict
```

### 1.2 Territory Ownership & Defense Delegation Matrix
To prevent single-point-of-failure during the oral defense, each member permanently owns a project territory (per [TEAM_ROLES.md](file:///d:/Documents/Programming/Thesis_G10/docs/TEAM_ROLES.md)):

| Researcher | Territory | Accountable Core Files | Primary Panel Inquiries |
| :--- | :--- | :--- | :--- |
| **KINA** (Cabral, S. Y.) | **Preprocessing & Vision Lead** | [`src/preprocessing/pipeline.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/pipeline.py)<br>[`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py)<br>[`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | Face detection, RetinaFace landmarks, keyframe ranking, Wav2Vec2/Whisper/BERT extraction, silent video handling. |
| **JC** (Caparas, J. C. B.) | **Training & Neural Modeling Lead** | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py)<br>[`src/models/bilinear.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py)<br>[`src/training/losses.py`](file:///d:/Documents/Programming/Thesis_G10/src/training/losses.py)<br>[`scripts/train_full.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/train_full.py) | 299D Bottleneck, Count Sketch CBP, Multi-Task Loss, Margin Loss ($m=1.5$), Sarcasm supervision, Phase 1 vs Phase 2 training. |
| **MATAN** (Exconde, M. J. B.) | **Evaluation & Statistical Benchmarks Lead** | [`src/evaluation/significance.py`](file:///d:/Documents/Programming/Thesis_G10/src/evaluation/significance.py)<br>[`scripts/evaluate_all_models.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/evaluate_all_models.py)<br>[`scripts/plot_sota_comparisons.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/plot_sota_comparisons.py) | FakeAVCeleb test protocols ($N=700$ vs $N=5000$), DeLong test ($p < 0.001$), Bootstrap 95% CIs, SOTA baselines (AceNet, MesoNet). |
| **GEL** (Rivera, G. J. D.) | **Data Integrity, WebApp & Integration Lead** | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py)<br>[`webapp/main.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/main.py)<br>[`docs/PROJECT_CONTEXT_MASTER.md`](file:///d:/Documents/Programming/Thesis_G10/docs/PROJECT_CONTEXT_MASTER.md)<br>[`scripts/print_training_summary.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/print_training_summary.py) | Dataset manifests (17,741 clips), 0% speaker leakage shield, Biological Harmony Prior, model warmup engine, live UI HUD. |

---

## 2. Master Codebase Navigation Matrix

This matrix allows any researcher to instantly answer: *"Where does that happen in the code?"*

| Stage / Concept | Mathematical Operation / Architectural Role | Primary Implementation File | Key Class / Method |
| :--- | :--- | :--- | :--- |
| **1. File Validation & Ingestion** | Format checking, aspect ratio clamping, filename sanitization, concurrency safety | [`webapp/input_validator.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/input_validator.py) | `validate_video()`<br>`validate_speech_presence()` |
| **2. Acoustic Extraction** | 16kHz mono resampling, 768D Wav2Vec 2.0 representation | [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | `extract_audio_features()`<br>`_load_wav2vec()` |
| **3. ASR Speech Transcription** | Whisper-Base speech-to-text with English enforcement | [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | `transcribe_speech()`<br>`_load_whisper()` |
| **4. Linguistic Sentiment** | 768D BERT CLS token encoding, merged to $Z_{at} \in \mathbb{R}^{1536}$ | [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) | `extract_text_features()`<br>`_load_bert()` |
| **5. Face Detection & Landmarking** | RetinaFace 5-point landmarks, Haar fallback, 1:1 square crop with +20% margin | [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py) | `detect_faces_retinaface()`<br>`crop_face_square()` |
| **6. AU Saliency & Keyframes** | Motion gating $>0.30$, Laplacian variance sharpness ranking, 8 keyframes | [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py) | `select_keyframes()`<br>`_rank_frames()` |
| **7. Visual Encoding (ViT)** | ViT-Base/16 CLS tokens over 8 keyframes $\to (B, 8, 768)$ | [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py) | `extract_visual_features()`<br>`_load_vit()` |
| **8. Cross-Modal Attention** | Bidirectional 8-head multi-head cross-attention ($Z_v \leftrightarrow Z_{at}$) | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L388-L403) | `cross_attn_v`<br>`cross_attn_at` |
| **9. Temporal GRU Modeling** | 2-layer Recurrent GRU capturing micro-expression dynamics over 8 keyframes | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L404-L407) | `vit_gru` (`nn.GRU(768, 768, 2)`) |
| **10. Compact Bilinear Pooling** | Count Sketch FFT convolution, signed-sqrt, L2-norm ($1.18\text{M} \to 8192\text{D}$) | [`src/models/bilinear.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py) | `CompactBilinearFusion`<br>`_sketch()`, `forward()` |
| **11. Sub-Symbolic Projection** | Linear projection `8192 -> 256` + LayerNorm + GELU | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L277) | `bilinear_proj`<br>`proj_ln` |
| **12. Multi-Task Emotion Heads** | Dual 6-class heads: Head A ($1536 \to 256 \to 6$), Head B ($768 \to 256 \to 6$) | [`src/models/emotion_heads.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/emotion_heads.py) | `EmotionHeadA`<br>`EmotionHeadB` |
| **13. Symbolic Emotion Incongruence** | Discrepancy $\boldsymbol{\Delta} = \|P_A - P_B\| \in \mathbb{R}^6$ & Co-occurrence $P_A \otimes P_B \in \mathbb{R}^{36}$ | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L263-L276) | `prob_a`, `prob_b`<br>`outer`, `delta` |
| **14. Sarcasm Head & Irony Filter** | Binary classification on $Z_{at}$, visually gated by excess smiling | [`src/models/sarcasm_head.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/sarcasm_head.py)<br>[`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L380-L392) | `SarcasmHead`<br>`sarc_gate` |
| **15. 299D Bottleneck Assembly** | $\text{Concat}(256\text{D}, 36\text{D}, 6\text{D}, 1\text{D}) = 299\text{D}$ | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L278) | `combined = torch.cat(...)` |
| **16. Classifier MLP** | LayerNorm(299) $\to$ Linear(512) $\to$ SE(512) $\to$ Linear(128) $\to$ Linear(1) | [`src/models/classifier.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/classifier.py) | `ClassifierMLP`<br>`SqueezeExcitation1D` |
| **17. Domain Adversarial Training** | Gradient Reversal Layer (GRL) + 5-Class Domain Classifier | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L31-L56) | `GradientReversalLayer`<br>`domain_classifier` |
| **18. Multi-Task Objective** | Focal Loss ($\gamma=2$) + Margin Loss ($m=1.5$) + Emotion CE + Sarcasm BCE | [`src/training/losses.py`](file:///d:/Documents/Programming/Thesis_G10/src/training/losses.py) | `MultiTaskLoss`<br>`forward()` |
| **19. Stage 1 Pretraining Runner** | Pretraining bottleneck on cached $Z_{at}, Z_v$ tensors (50 epochs) | [`scripts/colab_stage1.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/colab_stage1.py)<br>[`scripts/train_full.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/train_full.py) | `--mode bottleneck --phase 1` |
| **20. Few-Shot Domain Adaptation** | Speaker-disjoint tuning on Set A ($A \cap B = \emptyset$) for YouTube acoustics | [`scripts/colab_run_stage.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/colab_run_stage.py)<br>[`scripts/train_adaptation.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/train_adaptation.py) | `train_adaptation.py` |
| **21. Information-Theoretic Engine** | Symmetric Jensen-Shannon Divergence $D_{\text{JS}}(P_A \parallel P_B)$ & Cosine Synchrony | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L393-L406) | `d_js`, `cos_sim` |
| **22. Biological Harmony Prior** | Authenticity bonus: $-2.70$ (active match), $-0.70$ (neutral match) | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L412-L430) | `harmony_bonus` |
| **23. Asymmetric Sharpening** | $T=0.65$ (active emotions peak at 60–70%) vs $T=1.15$ (neutral protection) | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L332-L340) | `_calibrate_emotion_probs()` |
| **24. Silent Video Protection** | Grounding silent videos to 100% neutral voice ($\boldsymbol{\Delta}=0$, Sarcasm=0) | [`src/models/detection_model.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L254-L262)<br>[`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L367-L375) | `has_speech=False` |
| **25. DeLong Significance Testing** | Fast paired non-parametric ROC comparison algorithm vs SOTA baselines | [`src/evaluation/significance.py`](file:///d:/Documents/Programming/Thesis_G10/src/evaluation/significance.py) | `delong_test()`<br>`_fast_delong()` |
| **26. SOTA Comparative Suite** | Paired evaluations on identical 700 FakeAVCeleb clips across 6 architectures | [`scripts/export_comparative_benchmark_reference.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/export_comparative_benchmark_reference.py)<br>[`scripts/plot_sota_comparisons.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/plot_sota_comparisons.py) | `MODELS_CONFIG` |
| **27. Asynchronous Model Warmup** | Preloading 5 backbones + dry-run CUDA forward pass; mirrored progress bar | [`webapp/model_service.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L236-L283)<br>[`webapp/main.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/main.py#L170-L175) | `warmup()`<br>`warmup_status()` |
| **28. Live Telemetry HUD (SSE)** | Streaming RetinaFace bounding box, 5 landmarks, and live Whisper transcript | [`webapp/main.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/main.py#L200-L240)<br>[`webapp/static/js/app.js`](file:///d:/Documents/Programming/Thesis_G10/webapp/static/js/app.js) | `/analyze/stream`<br>`EventSource` |
| **29. Dominant Emotion Banner** | Color-coded outlined face emoji highlight card (Sad: Blue, Angry: Red, etc.) | [`webapp/static/js/app.js`](file:///d:/Documents/Programming/Thesis_G10/webapp/static/js/app.js)<br>[`webapp/static/css/style.css`](file:///d:/Documents/Programming/Thesis_G10/webapp/static/css/style.css) | `EMO_COLORS`<br>`dominant-card` |

---

## 3. Stage-by-Stage Technical Walkthrough & Mathematical Formulas

### Stage 1: Ingestion & Input Gatekeeper
* **Primary Code:** [`webapp/input_validator.py`](file:///d:/Documents/Programming/Thesis_G10/webapp/input_validator.py#L50-L130)
* **What Happens:**
  1. Checks video container integrity (`cv2.VideoCapture`).
  2. Inspects duration and enforces interactive timeline trimming (3.0s to 20.0s window).
  3. Audio decoding via `soundfile.read(dtype='float32')` (avoids Windows C++ DLL crashes in `torchaudio.load()`).
  4. Audio RMS energy check: determines `has_speech` flag to prevent microphone hiss hallucination.
  5. Filename sanitization: `f"{uuid[:8]}_{clean_name}"` prevents path traversal and concurrency collisions.

### Stage 2: Tri-Modal Feature Extraction
* **Primary Code:** [`src/preprocessing/audio.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/audio.py) & [`src/preprocessing/visual.py`](file:///d:/Documents/Programming/Thesis_G10/src/preprocessing/visual.py)
* **Acoustic Stream:**
  * Model: `facebook/wav2vec2-base` (frozen in Phase 1, unfreeze top-4 layers in Phase 2).
  * Input: 16 kHz mono raw waveform.
  * Pooling: Mean-pooled across time dimension $\to$ 768D embedding.
* **Linguistic Stream:**
  * Model 1: `openai/whisper-base` (forced English decoding `language="en"`).
  * Model 2: `bert-base-uncased`. Takes Whisper transcript tokens $\to$ 768D `[CLS]` token.
  * Fusion: Concatenated with acoustic feature:
    $$Z_{at} = [Z_{\text{audio}} \,\|\, Z_{\text{text}}] \in \mathbb{R}^{1536}$$
* **Visual Keyframe Stream:**
  * Model: `google/vit-base-patch16-224-in21k`.
  * Keyframe selection: Detects faces via **InsightFace RetinaFace** (`det_500m.onnx`), computes facial sharpness ($S = \text{Var}(\text{Laplacian})$), and ranks top 8 keyframes.
  * Temporal Modeling: Aligns 8 keyframe CLS tokens through a 2-layer Recurrent GRU (`vit_gru` in [`src/models/detection_model.py:L404`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L404)):
    $$Z_v = \text{GRU}(\text{ViT}(F_1), \dots, \text{ViT}(F_8)) \in \mathbb{R}^{768}$$

### Stage 3: Compact Bilinear Pooling (CBP)
* **Primary Code:** [`src/models/bilinear.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py#L24-L87)
* **Why Count Sketch FFT?**
  * Full outer product $Z_{at} \otimes Z_v$ produces $1536 \times 768 = 1,179,648$ dimensions, which causes catastrophic memory explosion and overfitting.
  * Compact Bilinear Pooling (Fukui et al., 2016) projects vectors into an 8,192D sketch space using Count Sketch hash tables ($h, s$), then computes polynomial convolution in the frequency domain via Fast Fourier Transform:
    $$\psi(Z_{at}, Z_v) = \text{FFT}^{-1}\Big(\text{FFT}(\text{CountSketch}(Z_{at})) \odot \text{FFT}(\text{CountSketch}(Z_v))\Big)$$
* **Signed Square-Root & $L_2$ Normalization:**
  * Essential stabilization step (preventing logit explosion):
    $$\mathbf{y} = \text{sign}(\mathbf{x}) \sqrt{|\mathbf{x}| + \epsilon}, \quad \mathbf{fused} = \frac{\mathbf{y}}{\|\mathbf{y}\|_2} \in \mathbb{R}^{8192}$$
* **Sub-Symbolic Projection:**
  * Projected through a lightweight bottleneck in [`detection_model.py:L277`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L277):
    $$\mathbf{fused\_proj} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{\text{proj}} \mathbf{fused})) \in \mathbb{R}^{256}$$

### Stage 4: Multi-Task Affect Heads & Symbolic Discrepancy
* **Primary Code:** [`src/models/emotion_heads.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/emotion_heads.py) & [`src/models/sarcasm_head.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/sarcasm_head.py)
* **Emotion Head A (Audio-Text):** $\text{Linear}(1536, 256) \to \text{GELU} \to \text{Linear}(256, 6) \implies P_A \in \mathbb{R}^6$.
* **Emotion Head B (Visual):** $\text{Linear}(768, 256) \to \text{GELU} \to \text{Linear}(256, 6) \implies P_B \in \mathbb{R}^6$.
* **Ekman 6 Emotion Space:** `[neutral, happy, sad, angry, fear, disgust]`.
* **Discrepancy Vector ($\boldsymbol{\Delta}$):**
  $$\boldsymbol{\Delta} = |P_A - P_B| \in \mathbb{R}^6$$
* **Joint Co-occurrence State Matrix ($\mathbf{fused\_emo}$):**
  $$\mathbf{fused\_emo} = \text{vec}(P_A \otimes P_B) \in \mathbb{R}^{36}$$
* **Sarcasm Head:** $\text{Linear}(1536, 256) \to \text{GELU} \to \text{Linear}(256, 1) \implies P_{\text{sarc}} \in \mathbb{R}^1$.

### Stage 5: The 299D Hybrid Bottleneck Classifier
* **Primary Code:** [`src/models/detection_model.py:L278`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L278) & [`src/models/classifier.py`](file:///d:/Documents/Programming/Thesis_G10/src/models/classifier.py)
* **Concatenation Vector:**
  $$\mathbf{x}_{299} = [\underbrace{\mathbf{fused\_proj}}_{256\text{D}} \,\|\, \underbrace{\mathbf{fused\_emo}}_{36\text{D}} \,\|\, \underbrace{\boldsymbol{\Delta}}_{6\text{D}} \,\|\, \underbrace{P_{\text{sarcasm}}}_{1\text{D}}] \in \mathbb{R}^{299}$$
* **Classifier Architecture:**
  $$\begin{aligned}
  \mathbf{h}_1 &= \text{SE}_{1\text{D}}\Big(\text{GELU}\big(\text{LayerNorm}(\mathbf{W}_1 \mathbf{x}_{299})\big)\Big) \in \mathbb{R}^{512} \\
  \mathbf{h}_2 &= \text{GELU}\big(\text{LayerNorm}(\mathbf{W}_2 \text{Dropout}(\mathbf{h}_1))\big) \in \mathbb{R}^{128} \\
  \text{raw\_logit} &= \mathbf{W}_3 \mathbf{h}_2 \in \mathbb{R}^1
  \end{aligned}$$
* **Domain-Adversarial Neural Network (DANN GRL):**
  * Gradient Reversal Layer with scheduled $\alpha(p) = \frac{2}{1 + e^{-10p}} - 1$.
  * Supervises domain invariance across MELD, MOSEI, CREMA-D, MUStARD, and fake tracks so the model strips out studio lighting and acoustics.

### Stage 6: Multi-Task Loss Objective
* **Primary Code:** [`src/training/losses.py`](file:///d:/Documents/Programming/Thesis_G10/src/training/losses.py#L33-L151)
* **Total Loss Formulation:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda_a \mathcal{L}_{\text{CE}}^{(A)} + \lambda_b \mathcal{L}_{\text{CE}}^{(B)} + \lambda_{\text{sarc}} \mathcal{L}_{\text{BCE}}^{(\text{sarc})} + \lambda_{\text{dom}} \mathcal{L}_{\text{DANN}} + \lambda_{\text{margin}} \mathcal{L}_{\text{margin}}$$
* **Exact Hyperparameters:**
  * $\mathcal{L}_{\text{det}}$: Focal Loss with $\gamma = 2.0$ and $\text{pos\_weight} = 1.3835$ (compensates for $8,254$ real vs $5,966$ fake training samples).
  * $\lambda_a = 0.1, \lambda_b = 0.1$: Auxiliary emotion regularization without degrading primary detection.
  * $\lambda_{\text{sarc}} = 0.05$: Supervised only on MUStARD clips (`sarcasm_label != -1`).
  * $\lambda_{\text{margin}} = 0.2$ with Margin $m = 1.5$:
    $$\mathcal{L}_{\text{margin}} = \max\left(0,\, 1.5 - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}})\right)$$
    Actively pushes real and fake distributions apart, preventing score clustering around $0.50$.

### Stage 7: Speaker-Disjoint Few-Shot Domain Adaptation
* **Primary Code:** [`scripts/colab_run_stage.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/colab_run_stage.py) & [`docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md`](file:///d:/Documents/Programming/Thesis_G10/docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md)
* **Why Necessary?** Studio TV dialogue (MELD, CREMA-D) has clean room acoustics. YouTube audio (FakeAVCeleb) has phone mic distortion and room echo. Pure zero-shot flagged YouTube room acoustics as synthetic, yielding $20\%$ real specificity.
* **The Pre-Sampling Identity Shield:**
  $$\text{Adaptation Celebrities } A \cap \text{Test Celebrities } B = \emptyset \quad (0\% \text{ overlap})$$
  * 150 Real / 150 Fake clips from Set A were used to calibrate only the projection layer.
  * All 350 test real clips and fakes were strictly from Set B.
  * *Outcome:* Restored Real Specificity to **`77.14%`** and established **`0.9020` AUC** without data leakage.

### Stage 8: Calibrated Forensic Reasoning Engine
* **Primary Code:** [`webapp/model_service.py:L360-L440`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L360-L440)
* **1. Continuous Jensen-Shannon Divergence ($D_{\text{JS}}$):**
  $$M = \frac{1}{2}(P_A + P_B), \quad D_{\text{JS}}(P_A \parallel P_B) = \frac{1}{2} D_{\text{KL}}(P_A \parallel M) + \frac{1}{2} D_{\text{KL}}(P_B \parallel M)$$
* **2. Multimodal Biological Harmony Prior:**
  $$\text{logit}_{\text{calib}} = \text{raw\_logit} - \text{harmony\_bonus}$$
  * Bonus = **$-2.70$** when modalities agree on active emotions ($\text{top}_A = \text{top}_B \ne \text{neutral}$) and $D_{\text{JS}} \le 0.07$.
  * Bonus = **$-0.70$** when modalities agree on calm baseline neutral speech.
  * Bonus = up to **$-1.80$** scaled continuously by $\text{CosSim}(P_A, P_B)$ for compatible active valences.
* **3. Asymmetric Active Sharpening & Neutral Protection:**
  $$\begin{cases} T = 0.65 & \text{if active emotion is leading (intensifies to 60–70%)} \\ T = 1.15 & \text{if neutral is leading (with pre-softmax } \text{neutral\_bias}=0.95\text{)} \end{cases}$$
* **4. Visually-Gated Sarcasm Filter:** Textual sarcasm is scaled by excess facial smiling:
  $$\text{Gate} = \max\left(0.05, \min\left(1.0, \left(\frac{\text{vis\_happy} - 0.167}{0.20}\right)^2\right)\right)$$

---

## 4. Master SOTA Benchmarks & Statistical Proofs ($N=700$)

All models evaluated on the identical, balanced FakeAVCeleb v1.2 test set (350 Real, 350 Fake).  
*Source Script:* [`scripts/export_comparative_benchmark_reference.py`](file:///d:/Documents/Programming/Thesis_G10/scripts/export_comparative_benchmark_reference.py) | *Reference Data:* [`comparative_sota_benchmark_reference.md`](file:///d:/Documents/Programming/Thesis_G10/docs/comparative_sota_benchmark_reference.md)

### 4.1 Master Comparative Leaderboard

| Model / Architecture | Modality / Basis | Accuracy (%) | Balanced Acc | Real Specificity | Fake Recall | F1-Score | MCC | AUC-ROC [95% CI] | Paired DeLong Test vs Ours |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSentinel (Ours)** | **Affect-Bilinear Multi-Head** | **82.14%** | **82.14%** | **77.14%** | **87.14%** | **0.8299** | **+0.6461** | **0.9020** [0.877–0.924] | **Reference** |
| **AceNet (Baseline)** | Cross-Attention Multimodal | 64.00% | 64.00% | 76.00% | 52.00% | 0.5909 | +0.2884 | 0.6425 [0.600–0.682] | $p < 0.001$ ($Z = 9.87$) |
| **MesoNet-4** | Spatial Convolutional CNN | 52.00% | 52.00% | 55.43% | 48.57% | 0.5030 | +0.0401 | 0.5389 [0.495–0.583] | $p < 0.001$ ($Z = 12.61$) |
| **LipForensics** | Spatiotemporal Viseme Sync | 52.00% | 52.00% | 54.00% | 50.00% | 0.5102 | +0.0400 | 0.5132 [0.469–0.553] | $p < 0.001$ ($Z = 13.44$) |
| **XceptionNet** | Deep Spatial CNN | 50.57% | 50.57% | 50.00% | 51.14% | 0.5085 | +0.0114 | 0.5002 [0.458–0.542] | $p < 0.001$ ($Z = 13.98$) |
| **Multimodal ResNet-AV** | Feature Concatenation | 46.14% | 46.14% | 47.14% | 45.14% | 0.4560 | -0.0772 | 0.4629 [0.419–0.506] | $p < 0.001$ ($Z = 15.22$) |

### 4.2 Per-Manipulation Attack Stress Accuracy (%)

| Manipulation Technique | DeepSentinel (Ours) | MesoNet-4 | XceptionNet | ResNet-AV | LipForensics | AceNet (Baseline) | Key Forensic Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`faceswap`** ($N=13$) | **84.6%** | 61.5% | 46.1% | 53.9% | 38.5% | 33.3% | Face emotion desync from native voice |
| **`faceswap-wav2lip`** ($N=58$) | **98.3%** | 46.5% | 51.7% | 36.2% | 55.2% | 65.2% | Dual manipulation amplifies affective gap |
| **`fsgan`** ($N=40$) | **62.5%** | 52.5% | 45.0% | 60.0% | 60.0% | 42.9% | Reenactment motion jitter |
| **`fsgan-wav2lip`** ($N=69$) | **98.5%** | 43.5% | 55.1% | 42.0% | 52.2% | 60.8% | Dual manipulation capture rate |
| **`real`** ($N=350$) | **77.1%** | 55.4% | 50.0% | 47.1% | 54.0% | 76.0% | Real video specificity protected |
| **`rtvc`** ($N=5$) | **60.0%** | 20.0% | 40.0% | 80.0% | 20.0% | 40.0% | Synthetic vocal cloning detection |
| **`wav2lip`** ($N=165$) | **85.5%** | 50.3% | 51.5% | 44.2% | 46.7% | 46.4% | Phoneme-viseme affective desynchrony |

### 4.3 Research Questions & Hypothesis H1 Verification

* **RQ1 (Speech-Text Affect Recognition):** Emotion Head A attained **$72.4\%$** accuracy on held-out CREMA-D speakers ($>4\times$ random baseline of $16.7\%$).
* **RQ2 (Visual Facial Expression Recognition):** Emotion Head B attained **$74.1\%$** accuracy across 8-keyframe visual sequences.
* **RQ3 (Zero-Shot & Adapted Cross-Dataset Generalization):** DeepSentinel attained **$0.9020$ AUC** and **$82.14\%$ Balanced Accuracy** on FakeAVCeleb v1.2.
* **RQ4 (Sarcasm Disambiguation):** Sarcasm Head attained **$77.27\%$** accuracy on held-out MUStARD speakers, preventing false alarms on deadpan delivery.
* **Hypothesis H1 (CONFIRMED):** DeepSentinel significantly outperforms state-of-the-art AceNet ($+25.95\%$ AUC lead, DeLong test $p = 0.0002 < 0.05$).

---

## 5. Panel Interrogation Playbook: Top 25 Toughest Defense Questions

### Category A: Theoretical Foundations & Motivation
#### Q1: "Why detect deepfakes through emotional inconsistency rather than visual synthesis artifacts like MesoNet, Xception, or EfficientNet?"
* **Spoken Script (30s):**  
  *"Artifact-based detectors chase transient pixel flaws like blending boundaries and warping seams. As generators adopt high-resolution diffusion transformers, these artifacts disappear, causing artifact detectors to collapse to random chance—as shown by MesoNet-4's 52% accuracy on FakeAVCeleb. In contrast, genuine human expression is biologically coordinated across facial Action Units and vocal prosody. Deepfake tools synthesize voice and face in separate pipelines, creating affective contradictions that higher rendering resolution cannot fix."*
* **Show Code:** [`docs/comparative_sota_benchmark_reference.md:L50-L62`](file:///d:/Documents/Programming/Thesis_G10/docs/comparative_sota_benchmark_reference.md#L50-L62)

#### Q2: "Why choose 6 discrete Ekman categories instead of continuous Valence-Arousal (Russell's Circumplex Model)?"
* **Spoken Script (30s):**  
  *"Two reasons: First, cross-dataset standardization—our core corpora (CREMA-D, MELD, CMU-MOSEI) provide categorical labels, whereas mapping to continuous coordinates introduces subjective interpolation error. Second, forensic explainability: in an investigative dashboard or legal audit, stating 'Voice expressed Anger (72%) while Face expressed Smiling/Happy (68%)' provides clear, actionable evidence. Continuous coordinates like V=0.21, A=0.44 lack forensic interpretability."*
* **Show Code:** [`webapp/config.py:L128-L130`](file:///d:/Documents/Programming/Thesis_G10/webapp/config.py#L128-L130)

#### Q3: "How does the model avoid accusing sarcastic or deadpan real humans of being deepfakes?"
* **Spoken Script (30s):**  
  *"Through a 3-tier defense: First, an auxiliary Sarcasm Head trained on MUStARD detects rhetorical irony. Second, this sarcasm prediction is visually gated—it requires actual facial smiling or smirking (AU12/14) to activate, preventing dry text from triggering false sarcasm. Third, our 8-state forensic matrix maps this to `STATE_REAL_DEADPAN_IRONY`, explaining to the user that deadpan delivery is natural human humor, not a deepfake."*
* **Show Code:** [`webapp/model_service.py:L380-L392`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L380-L392)

---

### Category B: Neural Architecture & Mathematics
#### Q4: "Where is the 299-dimensional vector assembled in the code?"
* **Spoken Script (20s):**  
  *"In `detection_model.py` lines 274–278. In bottleneck mode, we take the 256D projected bilinear embedding `fused_proj`, concatenate the 36D outer product `fused_emo`, the 6D discrepancy vector `delta`, and the 1D scalar `sarc`, yielding exactly 299 dimensions."*
* **Show Code:** [`src/models/detection_model.py:L274-L279`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L274-L279)

#### Q5: "Why Compact Bilinear Pooling instead of simple linear concatenation ($Z_{at} \oplus Z_v$)?"
* **Spoken Script (30s):**  
  *"Linear concatenation assumes audio and video are independent channels. Bilinear pooling calculates the full multiplicative tensor product ($1536 \times 768 = 1,179,648\text{D}$), capturing quadratic cross-modal interactions where subtle asynchronies live. Compact Bilinear Pooling uses Count Sketch hash tables and FFT convolution to approximate this 1.18-million-dimensional space in 8,192 dimensions without memory explosion."*
* **Show Code:** [`src/models/bilinear.py:L58-L86`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py#L58-L86)

#### Q6: "Why do you use signed square-root and $L_2$ normalization in Bilinear Pooling?"
* **Spoken Script (20s):**  
  *"Without signed square-root and $L_2$ normalization, the raw sketch magnitude reaches ~380. Downstream MLP logits explode into the hundreds, completely saturating the sigmoid to 0.000 or 1.000. Normalization bounds activations, allowing the classifier to produce calibrated, meaningful probabilities."*
* **Show Code:** [`src/models/bilinear.py:L84-L85`](file:///d:/Documents/Programming/Thesis_G10/src/models/bilinear.py#L84-L85)

#### Q7: "If `fused_proj` is 256D and $\boldsymbol{\Delta}$ is 6D, how do you know the model isn't ignoring the emotion mismatch?"
* **Spoken Script (30s):**  
  *"We proved this in two ways: First, backpropagation from the auxiliary emotion heads directly shapes the latent geometry of $Z_{at}$ and $Z_v$ around emotional prosody and facial Action Units, meaning the 256D CBP space is itself structured by affective representations. Second, in our ablation study (`mismatch_only` mode), classifying solely on the 6D $\boldsymbol{\Delta}$ vector and sarcasm score achieved strong standalone detection without any visual features."*
* **Show Code:** [`src/models/detection_model.py:L266-L268`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L266-L268)

---

### Category C: Dataset Integrity & Methodological Rigor
#### Q8: "How do you guarantee zero data leakage between training and testing?"
* **Spoken Script (30s):**  
  *"In our core training pool of 17,741 clips, we enforced strict speaker-disjoint hashing—zero speakers overlap across the 80% train, 10% val, and 10% test splits. In FakeAVCeleb, we enforced a Pre-Sampling Identity Shield: the 150 adaptation real clips were drawn exclusively from Celebrity Set A, while all 350 test real clips were drawn from Celebrity Set B ($A \cap B = \emptyset$). The model never saw the face or heard the voice of any evaluation celebrity during adaptation."*
* **Show Code:** [`docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md:L22-L29`](file:///d:/Documents/Programming/Thesis_G10/docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md#L22-L29)

#### Q9: "Why did you use Few-Shot Domain Adaptation instead of pure Zero-Shot?"
* **Spoken Script (30s):**  
  *"Pure zero-shot suffered from target-domain acoustic pessimism: pretraining datasets were recorded in soundproof TV studios, whereas FakeAVCeleb contains in-the-wild YouTube audio with room reverberation and phone compression. Unadapted Wav2Vec2 flagged YouTube room tone as synthetic anomalies, dropping real video specificity to ~20%. Adapting only the lightweight projection head on 150 clips from Set A calibrated the noise baseline, raising specificity to 77.14% without touching foundation representations."*
* **Show Code:** [`docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md:L40-L54`](file:///d:/Documents/Programming/Thesis_G10/docs/FEW_SHOT_ADAPTATION_AND_DEFENSE_STRATEGY.md#L40-L54)

#### Q10: "Why did your Matthews Correlation Coefficient (MCC) drop to +0.41 on the 5,000-clip test set, but reach +0.64 on the 700-clip test set?"
* **Spoken Script (30s):**  
  *"This is a known mathematical property of MCC under severe class skew. The 5,000-clip split contains 4,650 Fakes and only 350 Reals—a 13.3 to 1 imbalance (93% fake). The MCC denominator contains marginal class totals $\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$, which mathematically suppresses the scalar score even when sensitivity (85%) and specificity (78%) remain high. On the 1:1 balanced split ($N=700$), the identical model achieves MCC = +0.6461."*
* **Show Code:** [`docs/PROJECT_CONTEXT_MASTER.md:L504-L506`](file:///d:/Documents/Programming/Thesis_G10/docs/PROJECT_CONTEXT_MASTER.md#L504-L506)

---

### Category D: Forensic Calibration & Decision Making
#### Q11: "Where is the Multimodal Biological Harmony bonus calculated, and why?"
* **Spoken Script (30s):**  
  *"In `webapp/model_service.py` lines 412–430. Consumer laptop webcams introduce incandescent sensor noise that slightly pushes real clips toward the fake boundary (~54%). In affective biology, when genuine speech displays active emotional congruency (e.g. angry voice + angry face with $D_{\text{JS}} \le 0.07$), this is definitive proof of biological authenticity. We apply a -2.70 logit authenticity bonus, successfully restoring real webcam clips to high-confidence Real status."*
* **Show Code:** [`webapp/model_service.py:L412-L430`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L412-L430)

#### Q12: "How does your asymmetric emotion sharpening work, and why not use uniform temperature?"
* **Spoken Script (30s):**  
  *"Uniform low temperature caused the Neutral class to balloon to >85%, suffocating subtle emotional cues. Uniform high temperature washed out active emotions to <35%. In `model_service.py`, we subtract a 0.95 pre-softmax bias from Neutral and apply asymmetric scaling: if an active emotion is leading, we sharpen at $T=0.65$ so it peaks decisively at 60–70%; if Neutral is leading, we soften at $T=1.15$ so it stays modest and never suppresses subtle expressions."*
* **Show Code:** [`webapp/model_service.py:L332-L342`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L332-L342)

#### Q13: "What happens if an uploaded video has no speech or is completely silent?"
* **Spoken Script (30s):**  
  *"In earlier builds, microphone static caused Wav2Vec2 to hallucinate 92% Fear and 81% Sarcasm, triggering false deepfake alerts. We added a `has_speech` flag in `input_validator.py` and threaded it through the model. When speech is absent, vocal emotion is grounded to 100% Neutral, Sarcasm is set to 0.0, and $\boldsymbol{\Delta} = 0.0$. Silent videos now correctly evaluate as 93.6% Real."*
* **Show Code:** [`src/models/detection_model.py:L254-L262`](file:///d:/Documents/Programming/Thesis_G10/src/models/detection_model.py#L254-L262)<br>[`webapp/model_service.py:L367-L375`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L367-L375)

---

### Category E: Tool Implementation & Web Deployment
#### Q14: "How does the model warmup screen work, and why is it needed?"
* **Spoken Script (30s):**  
  *"DeepSentinel uses five heavy backbones totaling 1.8GB: Wav2Vec2, BERT, Whisper, ViT, and RetinaFace. Cold-loading these on the first video upload caused a 16.5-second freeze. In `model_service.py`, a background daemon asynchronously preloads all weights and executes a dry-run PyTorch CUDA forward pass during server boot. The UI displays an expanding center-mirrored progress bar with randomized solid colors, ensuring the user's first upload executes pure inference in 2.5 to 4.0 seconds."*
* **Show Code:** [`webapp/model_service.py:L236-L283`](file:///d:/Documents/Programming/Thesis_G10/webapp/model_service.py#L236-L283)

#### Q15: "What happens if a subject is wearing a face mask, sunglasses, or has a thick beard?"
* **Spoken Script (20s):**  
  *"Physical facial occlusions prevent the Vision Transformer from discerning lip and jaw Action Units (AU12/AU14/AU25). Automated pixel heuristics produced false rejections on dark lighting and beards. We resolved this by establishing physical facial occlusions as a documented operating boundary displayed directly in the web application UI and manuscript delimitations."*
* **Show Code:** [`docs/TOOL_DEFENSE_AND_SYSTEM_VULNERABILITY_AUDIT.md:L59-L60`](file:///d:/Documents/Programming/Thesis_G10/docs/TOOL_DEFENSE_AND_SYSTEM_VULNERABILITY_AUDIT.md#L59-L60)

---

## 6. Live Defense Terminal Execution Cheat-Sheet

Keep a PowerShell terminal open during the defense demo. Use these exact commands:

### 1. Show Official Preprocessed Dataset Summary (Figure 1 Verification)
```powershell
python scripts/print_training_summary.py
```
*Outputs: 17,741 verified clips, 80/10/10 split matrix, and 14,193 training clips.*

### 2. Launch the Production DeepSentinel Web Tool
```powershell
python -m uvicorn webapp.main:app --port 8000 --reload
```
*Access in browser: `http://localhost:8000` (Home & Trimmer), `/benchmarks` (SOTA Tab).*

### 3. Run DeLong Statistical Significance Test Against SOTA Baselines
```powershell
python scripts/export_comparative_benchmark_reference.py
```
*Outputs: Evaluates paired DeLong tests against AceNet, MesoNet-4, Xception, ResNet-AV, and LipForensics ($p < 0.001$).*

### 4. Regenerate Master SOTA Comparison Visualizations
```powershell
python scripts/plot_sota_comparisons.py
```
*Generates: `comparative_roc_curves.png`, `comparative_multimetric_barchart.png`, and `thesis_master_comparative_dashboard.png`.*

### 5. Run Live Inference Diagnostics on a Test Video
```powershell
python -c "from webapp.model_service import ModelService; s = ModelService(); print(s.detect('docs/genuine happy.mp4'))"
```
*Outputs: Full JSON forensic payload showing $P_A, P_B, \boldsymbol{\Delta}, D_{\text{JS}}$, and state determination.*

---

## 7. Golden Invariants for Defense Day
1. **Never guess a number:** Cite the exact figures—**$0.9020$ AUC**, **$82.14\%$ Balanced Accuracy**, **$77.14\%$ Specificity**, **$87.14\%$ Recall**, **$+0.6461$ MCC** on $N=700$.
2. **Never claim zero-shot without mentioning adaptation:** Explain that few-shot domain adaptation on Celebrity Set A calibrated YouTube acoustic shift while preserving 0% speaker leakage to Set B.
3. **Always ground explanations in affective biology:** Reiterate that generative models decouple audio-visual synthesis, making emotional incoherence an enduring, resilient signal.
4. **Point directly to code:** Use the Navigation Matrix in Section 2 to open the exact file and line when asked.
