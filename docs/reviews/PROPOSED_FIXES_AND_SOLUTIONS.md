# DeepSentinel — Proposed Fixes & Engineering Remediation Guide

> **Document Title:** Proposed Fixes & Solutions for Probability Score Compression, Decision Boundary Oscillation, and Model Failure Modes  
> **Repository:** `emotion-based-multimodal-deepfake-detector`  
> **Branch:** `feat/training-turnover-prep`  
> **Date:** August 2026  
> **Authors:** Cabral, S. Y., Caparas, J. C. B., Exconde, M. J. B., Rivera, G. J. D. (Polytechnic University of the Philippines)  
> **Associated Diagnostic Document:** [docs/reasons_for_errors.md](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/docs/reasons_for_errors.md)

---

## Executive Overview

This document provides the definitive, comprehensive engineering specification of all **proposed and implemented fixes** addressing the failure modes identified during the training, evaluation, and benchmark testing of the DeepSentinel multimodal deepfake detector.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ERROR REASON ➔ SUPPOSED FIX MATRIX                                   │
├───────────────────────────────────────┬──────────────────────────────────────────────────────────────────┤
│ 1. Feature Representation Mismatch    │ ➔ Sequence Caching (8, 768) + Temporal GRU Alignment             │
│ 2. Covariate Shift on Offline Eval    │ ➔ Automatic Live GPU Routing & Post-Phase-2 Feature Re-extraction│
│ 3. Score Compression (Narrow Band)    │ ➔ Supervised Contrastive Margin Loss (m=1.5) + Focal Loss (γ=2.0)│
│ 4. Dataset Domain Shortcut Trap       │ ➔ Domain Adversarial Training (GRL) + Sub-symbolic Compression   │
│ 5. Phase 2 Checkpoint Lockout Bug     │ ➔ Live End-to-End Validation Loop (_val_epoch_e2e)               │
│ 6. Threshold & Imbalance Bias         │ ➔ Balanced 500/500 Split + Youden's J (τ_J) + Learned Temp (T*)   │
└───────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

---

## 1. Fix 1: Resolving Visual Keyframe Sequence & GRU Temporal Mismatch

### Problem Statement
In preprocessing ([src/preprocessing/visual.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/preprocessing/visual.py)), 8 keyframes were mean-pooled into a single 768D vector $\mathbf{z}_v$. In Phase 1, `forward_from_features` repeated this single vector 8 times (`z_v.repeat(1, 8, 1)`). When Phase 2 or live evaluation ran on real video, the 2-Layer Temporal GRU received 8 distinct, temporally evolving frames, causing the GRU hidden state to diverge.

### The Proposed Solutions

#### Solution 1.A: Full Keyframe Sequence Caching (8 × 768D)
Instead of caching a 1D mean-pooled vector $\mathbf{z}_v \in \mathbb{R}^{768}$, update the feature extraction pipeline to save the complete 8-frame tensor:
$$\mathbf{Z}_{v,\text{cached}} \in \mathbb{R}^{8 \times 768}$$
* **Implementation:**
  ```python
  # In src/preprocessing/visual.py
  cls_tokens = out.last_hidden_state[:, 0, :]   # (8, 768)
  torch.save(cls_tokens.cpu(), z_v_path)        # Save (8, 768) instead of mean-pooled (768,)
  ```
* **Phase 1 Forward Alignment:**
  ```python
  # In src/models/detection_model.py
  def forward_from_features(self, z_at: torch.Tensor, z_v_seq: torch.Tensor) -> DetectorOutput:
      w2v_emb = z_at[:, :768]
      bert_emb = z_at[:, 768:]
      return self._forward_impl(w2v_emb, bert_emb, z_v_seq)  # Genuine (B, 8, 768) sequence
  ```

#### Solution 1.B: Attention-Based Keyframe Pooling (Order-Invariant Architecture)
Replace the sequential GRU with a Multi-Head Attention Pooling module. This eliminates the dependency on rigid frame order and temporal repetition:
$$\mathbf{Z}_v = \text{MultiheadAttention}(\text{Query}=\mathbf{q}_{\text{learnable}},\ \text{Key}=\mathbf{Z}_{v,\text{seq}},\ \text{Value}=\mathbf{Z}_{v,\text{seq}})$$

---

## 2. Fix 2: Eliminating Covariate Shift & Offline Feature Traps

### Problem Statement
Evaluating a fine-tuned Phase 2 head on base un-tuned `.pt` features causes catastrophic logit collapse ($P(\text{fake}) \le 0.001$) because Compact Bilinear Pooling squares per-modality subspace shifts.

### The Proposed Solutions

#### Solution 2.A: Automatic Live GPU Forward-Pass Routing (Implemented)
In [scripts/evaluate_fakeavceleb.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/scripts/evaluate_fakeavceleb.py), whenever a checkpoint contains fine-tuned backbone parameters (`_backbones_loaded == True`), the evaluation harness automatically bypasses disk `.pt` files and streams raw video frames and audio directly through GPU VRAM:
```python
# Automatic routing safeguard
if is_phase2_checkpoint(checkpoint):
    print("Fine-tuned backbone detected. Routing evaluation to live end-to-end GPU decoding...")
    dataloader = FakeAVCelebLiveDataset(clips)
    model.forward(audio, input_ids, attention_mask, keyframe_pixels)
```

#### Solution 2.B: Post-Phase-2 Batch Feature Re-Extraction
If high-speed offline feature evaluation is required across large test sets ($>10,000$ clips), execute an offline feature dump using the fine-tuned backbone weights:
```bash
python scripts/colab_cache_fakeavceleb.py --checkpoint checkpoints/full/best_phase2_bottleneck.pt --output_dir data/preprocessed/features_p2
```

---

## 3. Fix 3: Widening Score Margins & Preventing Logit Compression

### Problem Statement
Standard binary cross-entropy ($\mathcal{L}_{\text{BCE}}$) gradients saturate as soon as samples are weakly separated. Real scores ($P \approx 0.28$) and Fake scores ($P \approx 0.31$) cluster within a tiny $\Delta \mu \approx 0.03$ band, causing all predictions to fall on one side of $\tau = 0.50$.

### The Proposed Solutions

#### Solution 3.A: Supervised Contrastive Margin Loss (Implemented)
Add an explicit margin penalty to Phase 2 training that penalizes the model whenever the distance between batch fake mean logit and real mean logit is $< 1.5$:

$$\mathcal{L}_{\text{margin}} = \max\left(0,\ m - (\bar{z}_{\text{fake}} - \bar{z}_{\text{real}})\right) \quad \text{where } m = 1.5,\ \lambda_{\text{margin}} = 0.2$$

* **Code Implementation ([src/training/trainer.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/training/trainer.py#L435-L447)):**
  ```python
  valid_mask = fl != -1
  fake_mask = (fl == 1) & valid_mask
  real_mask = (fl == 0) & valid_mask
  if fake_mask.any() and real_mask.any():
      mean_fake = out.logit[fake_mask].mean()
      mean_real = out.logit[real_mask].mean()
      margin_loss = torch.clamp(1.5 - (mean_fake - mean_real), min=0.0)
  else:
      margin_loss = torch.tensor(0.0, device=self.device)

  total_step_loss = loss.total + 0.2 * margin_loss
  ```
* **Effect:** Expands the score distribution from $[0.28, 0.31]$ to Real $\le 0.15$ and Fake $\ge 0.80$.

#### Solution 3.B: Focal Loss for Hard-Example Mining
Replace standard BCE with Focal Loss ($\gamma = 2.0, \alpha = 0.25$) to down-weight easy background examples and amplify gradients on ambiguous, subtle face swaps:
$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

---

## 4. Fix 4: Breaking Dataset Domain Shortcuts (Same-Session Generalization)

### Problem Statement
In training datasets, real clips come from TV series (MELD) and YouTube vlogs (CMU-MOSEI), while fake clips were synthesized on studio actors (CREMA-D). The 256D bilinear projection learned background and room acoustics as shortcuts. On FakeAVCeleb, Real and Fake clips come from the *exact same interview session*, so domain shortcuts fail and $\boldsymbol{\Delta} \approx 0$.

### The Proposed Solutions

#### Solution 4.A: Domain-Adversarial Neural Network (DANN / GRL)
Attach an auxiliary domain classifier that predicts the dataset source ($D \in \{\text{CREMA-D}, \text{MELD}, \text{MOSEI}\}$) through a Gradient Reversal Layer (GRL):
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{detection}} - \lambda_{\text{domain}} \mathcal{L}_{\text{domain}}$$
* **Mechanism:** By reversing gradients from the domain discriminator, the 256D bilinear projection is forced to discard background/studio cues and retain only manipulation-relevant affect discrepancies.

#### Solution 4.B: Sub-Symbolic Bottleneck Compression
Reduce the dimension of `fused_proj` from 256D to **64D or 32D**:
$$\mathbf{x}_{\text{classifier}} = \Big[\underbrace{\mathbf{fused\_proj}}_{32\text{D}}\ ;\ \underbrace{\mathbf{fused\_emo}}_{36\text{D}}\ ;\ \underbrace{\boldsymbol{\Delta}}_{6\text{D}}\ ;\ \underbrace{P_{\text{sarcasm}}}_{1\text{D}}\Big] \in \mathbb{R}^{75}$$
* **Mechanism:** Constraining the sub-symbolic capacity prevents the MLP from memorizing high-dimensional background patterns, forcing the model to rely on the symbolic emotion discrepancy vector ($\boldsymbol{\Delta}$).

---

## 5. Fix 5: Resolving the Phase 2 Checkpoint Lockout Bug

### Problem Statement
In Phase 2, `trainer.py` evaluated validation loss on static Phase 1 `.pt` files (`_val_epoch_cached`). As backbones adapted, validation loss on stale files increased, locking the best checkpoint at Epoch 1.

### The Implemented Solution
Implemented `_val_epoch_e2e` in [src/training/trainer.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/src/training/trainer.py#L463-L551):
* Validates live raw keyframes and audio waveforms on unseen speaker validation clips.
* Tracks genuine end-to-end generalization across all 15 epochs, saving `best_phase2_bottleneck.pt` on improving live validation loss.

---

## 6. Fix 6: Correcting Class Imbalance & Threshold Calibration

### Problem Statement
Testing on 90% fake test sets biased standard accuracy and F1 optimization toward $\tau = 0.01$, creating trivial "always fake" classifiers ($TN = 0$).

### The Implemented Solutions

#### Solution 6.A: Balanced 500 Real / 500 Fake Benchmark Harness
Implemented in [scripts/colab_eval_fakeav.py](file:///d:/Internship/emotion-based-multimodal-deepfake-detector/scripts/colab_eval_fakeav.py):
* Samples an exact 500 Real / 500 Fake balanced evaluation split.
* Mandates reporting of **Balanced Accuracy**, **Matthews Correlation Coefficient (MCC)**, and **Per-Class TPR/TNR**.

#### Solution 6.B: Youden's J Statistic Operating Point Sweep
Sweeps $\tau \in [0.01, 0.99]$ to find the optimal operating threshold $\tau_J^*$:
$$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1$$
$$\tau_J^* = \arg\max_\tau J(\tau)$$

#### Solution 6.C: Learned Temperature Scaling ($T^*$)
Fits a learned scalar temperature parameter $T^*$ on validation set Negative Log-Likelihood (NLL) before deployment:
$$P_{\text{calibrated}} = \sigma\left(\frac{z}{T^*}\right) \quad \text{where } T^* = \arg\min_T \text{NLL}(z / T, y)$$

---

## 7. Implementation & Verification Status Table

| Component | Target Problem | Fix Implemented | Status |
| :--- | :--- | :--- | :--- |
| **`trainer.py`** | Phase 2 Checkpoint Freeze | Live validation loop `_val_epoch_e2e` | ✅ Active in Branch |
| **`trainer.py`** | Logit Score Compression | Supervised Margin Loss ($m=1.5, \lambda=0.2$) | ✅ Active in Branch |
| **`losses.py`** | Class Weight Inversion | Set $\text{pos\_weight} = 1.3835$ ($8254/5966$) | ✅ Active in Branch |
| **`evaluate_fakeavceleb.py`** | Covariate Shift Collapse | Automatic live GPU routing on fine-tuned checkpoints | ✅ Active in Branch |
| **`colab_eval_fakeav.py`** | Imbalance Metric Flattery | Balanced 500/500 sampling + Youden's J sweep + MCC | ✅ Active in Branch |
| **`trainer.py`** | Google Drive Write Loss | Multi-directory backup + mandatory `os.sync()` flush | ✅ Active in Branch |

---
*End of Proposed Fixes & Engineering Remediation Guide.*
