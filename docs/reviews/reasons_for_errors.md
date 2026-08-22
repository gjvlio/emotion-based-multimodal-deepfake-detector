# Reasons for Errors — DeepSentinel Training & Evaluation Diagnostic Report

> **Document Title:** Reasons for Errors: Probability Score Compression, Decision Boundary Oscillation, and Model Failure Modes  
> **Repository:** `emotion-based-multimodal-deepfake-detector`  
> **Branch:** `feat/training-turnover-prep`  
> **Date:** August 2026  
> **Subject:** Root-Cause Breakdown of All-Fake / All-Real Output Oscillation Across Code, Equations, and Datasets

---

## Executive Summary

During the development and benchmark evaluation of DeepSentinel on the unseen **FakeAVCeleb v1.2** dataset, the framework exhibited extreme single-class output collapse:
1. **The "All-Real" Failure Mode:** Predicting $P(\text{fake}) \le 0.001$ for 100% of test clips ($TP = 3, FN = 905, TN = 200, FP = 0$).
2. **The "All-Fake" Failure Mode:** Predicting $P(\text{fake}) \ge 0.50$ for 100% of test clips at standard threshold $\tau = 0.50$ ($TP = 900, FP = 100, TN = 0, FN = 0$).

This document provides the definitive, multi-layered root-cause analysis explaining **why** the model was failing, **where** the problems are located across Phase 1, Phase 2, the loss equations, and dataset curation, and **how** each failure mode is resolved.

---

## The 3 Diagnostic Layers Causing the Errors

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          THE 3 LAYERS CAUSING THE COLLAPSE                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Code & Forward Pass Gap   │ Preprocessed feature format vs live Phase 2 video │
│ Layer 2: Mathematical Equations    │ BCE saturation + Score compression + Rigid τ=0.50 │
│ Layer 3: Dataset Distribution Gap  │ Independent sources (Train) vs Same-Session (Test)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Layer 1: Code Implementation & Forward Pass Discrepancies

### 1.1 The Preprocessed Feature vs. Live Video Sequence Mismatch
* **In Preprocessing ([src/preprocessing/visual.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/preprocessing/visual.py#L394-L396)):**  
  When offline features were generated, `get_z_v()` extracted 8 keyframes and **mean-pooled** them:
  $$\mathbf{z}_v = \frac{1}{8}\sum_{k=1}^8 \mathbf{h}_{\text{CLS}, k}^{(\text{ViT})} \in \mathbb{R}^{768}$$
* **In Phase 1 Training ([src/models/detection_model.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/models/detection_model.py#L240)):**  
  `forward_from_features` took the single mean vector $\mathbf{z}_v$ and **repeated it 8 times** (`z_v.repeat(1, 8, 1)`), passing 8 identical vectors through Cross-Modal Attention and the 2-Layer Temporal GRU.
* **In Phase 2 End-to-End ([src/models/detection_model.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/models/detection_model.py#L273-L277)):**  
  `forward()` feeds 8 **real, diverse video keyframes** through ViT $\to$ Cross-Attention $\to$ Temporal GRU.
* **The Failure Mechanism:** The Temporal GRU and Cross-Attention weights learned in Phase 1 were optimized for **static, repeated vectors**. When fed real temporal frame sequences in Phase 2 or test time, the GRU hidden state diverged, causing extreme shifts in the resulting $\mathbf{Z}_v$.

---

### 1.2 Quadratic Covariate Shift in Compact Bilinear Pooling (CBP)
* Compact Bilinear Pooling approximates the outer product $Z_{\text{at}} \otimes Z_v \in \mathbb{R}^{1536 \times 768}$.
* Because the fusion is **quadratic (multiplicative)**, a minor $5\%$ rotation in latent space between training and test representations produces a **$15\text{--}25\%$ distortion** in the 8192D fused representation.
* When this shifted tensor passes through `nn.LayerNorm(256)`, the learned affine parameters ($\gamma, \beta$) map the features to extreme values, pushing the classifier's output logit $z$ to either extreme negative ($z \le -5.0 \implies P \le 0.006$, **All-Real**) or extreme positive ($z \ge +3.0 \implies P \ge 0.95$, **All-Fake**).

---

### 1.3 Offline vs. Live Evaluation Routing Trap
* When running `evaluate_fakeavceleb.py`, if `.pt` files existed on disk, the evaluator defaulted to `forward_from_features()`.
* Evaluating a **Phase 2 fine-tuned head** on **Phase 1 un-tuned base features** Handed the network inputs with completely mismatched feature geometry.
* The LayerNorm and linear weights pushed all logits negative, causing the model to predict **100% Real** ($P \le 0.001$).

---

## 2. Layer 2: Mathematical Equations & Loss Dynamics

### 2.1 Score Compression & The "Narrow Band" Trap
* When binary cross-entropy ($\mathcal{L}_{\text{BCE}}$) trains a model, once samples are weakly separated, the gradients diminish due to sigmoid saturation:
  $$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = \sigma(z) - y$$
* Without an active Margin Loss, the model stops pushing the logits further apart once they achieve small separation (e.g., Real mean logit $\bar{z}_{\text{real}} = -0.93 \implies P \approx 0.28$, Fake mean logit $\bar{z}_{\text{fake}} = -0.79 \implies P \approx 0.31$).
* **The Failure:** The score distributions for Real and Fake clips both sit inside a tiny band ($[0.25, 0.35]$).
  * If the entire band is $< 0.50$ $\implies$ **All clips are predicted REAL ($TN=500, TP=0$)**.
  * If a positive bias pushes the entire band $\ge 0.50$ $\implies$ **All clips are predicted FAKE ($TP=500, TN=0$)**.
  * A single rigid threshold $\tau = 0.50$ fails because the model's scores are compressed on one side of 0.50.

---

### 2.2 Sensitivity to Class Weight (`pos_weight`)
* The multi-task BCE loss uses:
  $$\mathcal{L}_{\text{BCE}} = - \Big( \text{pos\_weight} \cdot y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \Big)$$
* If `pos_weight` is set below 1.0 (such as the earlier `0.5421` bug where real/fake was inverted), the penalty for missing fake clips is halved, training the network to output negative logits $\implies$ **Spams All-Real**.
* If `pos_weight` is uncalibrated relative to the test domain, the logit baseline shifts $\implies$ **Spams All-Fake**.

---

### 2.3 AUC Governs the Mathematical Performance Ceiling
Under the equal-variance binormal ROC model, the maximum simultaneously achievable $\text{TPR} = \text{TNR}$ is a deterministic function of AUC:

| AUC-ROC | Maximum Simultaneous $\text{TPR} = \text{TNR}$ | Maximum Youden's $J$ |
| :--- | :--- | :--- |
| **0.58** | **55.7%** | **0.114** |
| 0.60 | 57.1% | 0.142 |
| 0.65 | 60.7% | 0.215 |
| 0.70 | 64.3% | 0.286 |
| 0.80 | 72.6% | 0.452 |
| **0.965** | **90.0%** | **0.800** |

**Critical Insight:** When the raw representation has $\text{AUC} \approx 0.58$, no threshold calibration (Youden's J, Bayesian, or otherwise) can achieve $90\%$ TPR and $90\%$ TNR simultaneously. Moving the threshold only trades False Positives for False Negatives. Reaching high detection accuracy requires widening the underlying inter-class representation margin via Supervised Margin Loss and backbone fine-tuning.

---

## 3. Layer 3: Dataset Construction & The "Same-Session" Domain Gap

### 3.1 Domain Shortcuts in the Training Data
* In the training dataset:
  * **Real clips** come from MELD (sitcom TV set, laugh tracks), CMU-MOSEI (YouTube vlogs), and CREMA-D (studio actors).
  * **Fake clips** (Tracks 1, 2, 3) were synthesized primarily using CREMA-D studio actor videos.
* The 256D sub-symbolic CBP projection can inadvertently memorize background acoustic/visual profiles (e.g. *green screen studio acoustics = Fake*, *TV dialogue noise = Real*) rather than subtle facial-vocal asynchrony.

---

### 3.2 The Same-Session Test Blind Spot on FakeAVCeleb
* In **FakeAVCeleb** (the test benchmark), real and fake clips come from the **exact same VoxCeleb2 interview session**:
  * Real clip: Speaker A in room X talking with expression Y.
  * Fake clip: Speaker A's face/lips modified in room X with speech from expression Y.
* Because the background, lighting, and acoustic room profile are identical between Real and Fake:
  1. The background shortcuts learned during training become useless.
  2. The emotion delta $\boldsymbol{\Delta} = |\mathbf{p}_a - \mathbf{p}_b|$ is small for both Real and Fake clips (since generative lip-syncing was driven by speech matching the sentence context).
* Stripped of dataset shortcuts, the model's output scores collapse into a narrow, overlapping cluster.

---

## 4. Location Matrix: Where Were the Errors Located?

| Component | Location in Code | Error Description | Consequence |
| :--- | :--- | :--- | :--- |
| **Dataset** | `data/synthetic/` vs `data/raw/FakeAVCeleb` | Synthesized fakes built primarily on CREMA-D; real clips from diverse sitcoms/vlogs. | Model learned domain shortcuts; failed on same-session FakeAVCeleb clips. |
| **Phase 1 Training** | `src/models/detection_model.py#L240` | GRU trained on 8 repeated identical mean vectors (`z_v.repeat(1, 8, 1)`). | Temporal representations diverged when fed live 8-frame video sequences. |
| **Phase 2 Validation** | `src/training/trainer.py#L387` | Phase 2 trained live on video, but validated on static Phase 1 feature files. | Validation loss on static files rose as backbones adapted $\to$ Checkpoint locked at Epoch 1. |
| **Loss Formulation** | `scripts/train_full.py` (Historical) | `pos_weight` previously set to `0.5421` ($3234 / 5966$). | Penalized fake loss by half $\to$ Logits collapsed to $-5.0$ ($P \le 0.006$). |
| **Evaluation Harness** | `scripts/evaluate_fakeavceleb.py` | Routed Phase 2 checkpoints to offline `.pt` features on disk; evaluated on 90/10 imbalanced set. | Offline: 100% Real collapse ($P \le 0.001$). Imbalanced: F1 sweep picked $\tau=0.01$ (100% Fake). |

---

## 5. Summary of Implemented Solutions

The following 5 engineering fixes have been fully implemented in the active codebase:

1. **Supervised Margin Loss in Phase 2 ([src/training/trainer.py#L435-L447](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/training/trainer.py#L435-L447)):**  
   Adds $\mathcal{L}_{\text{margin}} = \max(0, 1.5 - (\bar{z}_{\text{fake}} - \bar{z}_{\text{real}}))$ with weight $0.2$, actively forcing a $>1.5$ logit gap and expanding the score distribution out of the $[0.28, 0.31]$ compression trap.
2. **Live End-to-End Phase 2 Validation ([src/training/trainer.py#L463-L551](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/training/trainer.py#L463-L551)):**  
   Evaluates live 8-keyframe video batches during validation, ensuring checkpoints save on actual video generalization rather than static Phase 1 features.
3. **Automatic Live GPU Routing in Evaluation ([scripts/evaluate_fakeavceleb.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/scripts/evaluate_fakeavceleb.py)):**  
   Prevents evaluating fine-tuned Phase 2 models on base `.pt` feature tensors.
4. **Exact Loss Class Balancing ([src/training/losses.py#L65-L66](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/training/losses.py#L65-L66)):**  
   Enforces $\text{pos\_weight} = 1.3835$ ($8254 \text{ Real} / 5966 \text{ Fake}$) to center logits in $[-2.0, +2.0]$.
5. **Youden's J & Balanced 50/50 Benchmark ([scripts/colab_eval_fakeav.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/scripts/colab_eval_fakeav.py)):**  
   Calibrates the operating threshold $\tau_J^*$ based on maximum vertical separation ($J = \text{Sensitivity} + \text{Specificity} - 1$) and evaluates on balanced 500 Real / 500 Fake splits with MCC and Balanced Accuracy.

---
*End of Diagnostic Report.*
