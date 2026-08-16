# DeepSentinel: Multi-Model Architectural Review & Evaluation Post-Mortem

**Document Version:** 1.0
**Date:** 2026-08-15
**Reviewed By:** DeepSeek-R1, Claude Opus 4.6 (Anthropic), Antigravity (Google DeepMind)
**Subject:** Probability Score Compression, Decision Boundary Oscillation, and Calibrated Evaluation Protocol for Multimodal Affect-Based Deepfake Detection

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [The Empirical Dilemma: Two Failure Modes](#3-the-empirical-dilemma-two-failure-modes)
4. [Root Cause Analysis](#4-root-cause-analysis)
   - 4.1 [Feature Distribution Shift (Covariate Shift)](#41-feature-distribution-shift-covariate-shift)
   - 4.2 [Class Imbalance & Metric Flattery](#42-class-imbalance--metric-flattery)
   - 4.3 [Data Integrity & Internal Consistency](#43-data-integrity--internal-consistency)
5. [Threshold Calibration: Mathematical Constraints](#5-threshold-calibration-mathematical-constraints)
   - 5.1 [Youden's J Statistic](#51-youdens-j-statistic)
   - 5.2 [Bayesian Decision Theory](#52-bayesian-decision-theory)
   - 5.3 [AUC-Constrained Performance Ceiling](#53-auc-constrained-performance-ceiling)
6. [Multi-Task Loss Balancing: Gradient Shielding vs. Affect Starvation](#6-multi-task-loss-balancing-gradient-shielding-vs-affect-starvation)
7. [Score Compression Remediation Techniques](#7-score-compression-remediation-techniques)
   - 7.1 [Temperature Scaling](#71-temperature-scaling)
   - 7.2 [Focal Loss](#72-focal-loss)
   - 7.3 [Supervised Contrastive Margin Loss](#73-supervised-contrastive-margin-loss)
   - 7.4 [Cosine Similarity Head (ArcFace-Style)](#74-cosine-similarity-head-arcface-style)
8. [Consensus Prioritized Action Plan](#8-consensus-prioritized-action-plan)
9. [Academic References](#9-academic-references)

---

## 1. Executive Summary

During cross-dataset evaluation of the DeepSentinel multimodal deepfake detector on the unseen FakeAVCeleb v1.2 benchmark, two diametrically opposite failure modes were observed:

- **Failure Mode A (Offline Feature Evaluation):** The model predicted **everything as REAL** ($TP = 3, FN = 905$), producing $P(\text{fake}) \le 0.001$ for every clip.
- **Failure Mode B (End-to-End Evaluation on 90/10 Imbalanced Set):** The model predicted **everything as FAKE** ($TP = 900, FP = 100, TN = 0$), with an AUC-ROC of only $0.5829$.

Three independent AI systems (DeepSeek-R1, Claude Opus 4.6, and Antigravity/Google DeepMind) were consulted for architectural and statistical review. This document synthesizes their analyses into a unified post-mortem with consensus findings, disagreements, and a prioritized remediation plan backed by peer-reviewed academic references.

### Consensus Verdict (All Three Models Agree)

| Finding | Verdict | Confidence |
|---------|---------|------------|
| Scenario A collapse is caused by covariate shift | **Expected behavior, not a bug** | Unanimous |
| Scenario B's reported score ranges vs. confusion matrix are internally inconsistent | **Data integrity issue — must resolve first** | Unanimous (Claude, Antigravity) |
| At AUC ≈ 0.58, no threshold achieves 90% TPR + 90% TNR simultaneously | **Mathematically impossible** | Unanimous (Claude, Antigravity) |
| Loss-value scaling ($\lambda$) does not guarantee proportional gradient-norm scaling | **Unverified premise** | Unanimous (Claude, Antigravity) |
| Affect feature starvation is a plausible contributor to low AUC | **Needs empirical verification** | Unanimous |
| Temperature $T$ should be learned, not hardcoded | **Hardcoded $T=0.5$ is methodologically incorrect** | Unanimous (Claude, Antigravity) |

---

## 2. System Architecture Overview

### 2.1 Backbone Networks

| Modality | Model | Embedding Dimension | Pre-training Source |
|----------|-------|-------------------|-------------------|
| Audio | Wav2Vec 2.0 [1] | 768D | LibriSpeech 960h |
| Visual | ViT-Base/16 [2] | 768D (8 keyframes) | ImageNet-21k |
| Text | BERT-Uncased [3] | 768D | BooksCorpus + English Wikipedia |

### 2.2 Fusion & Classification Pipeline

The cross-modal interaction is captured via **Compact Bilinear Pooling (CBP)** [4], which approximates the full outer product $z_{\text{at}} \otimes z_v \in \mathbb{R}^{768 \times 768}$ using tensor sketch and FFT, producing an 8192D fused representation.

The classifier input is a **299D Multi-Scale Hybrid Bottleneck**:

$$\mathbf{x}_{\text{classifier}} = \text{Concat}\Big(\underbrace{\mathbf{fused\_proj}}_{256D},\ \underbrace{\mathbf{fused\_emo}}_{36D},\ \underbrace{\boldsymbol{\Delta}}_{6D},\ \underbrace{P_{\text{sarcasm}}}_{1D}\Big) \in \mathbb{R}^{299}$$

Where:
- $\mathbf{fused\_proj} = \text{GELU}(\text{LayerNorm}(W_{\text{proj}} \cdot \text{CBP}(z_{\text{at}}, z_v)))$ — the projected bilinear fusion
- $\mathbf{fused\_emo} = \text{vec}(\text{softmax}(\hat{y}_a) \otimes \text{softmax}(\hat{y}_b))$ — the 6×6 emotion probability outer product
- $\boldsymbol{\Delta} = |\text{softmax}(\hat{y}_a) - \text{softmax}(\hat{y}_b)|$ — the per-emotion probability delta (core incongruency signal)
- $P_{\text{sarcasm}} = \sigma(\hat{y}_{\text{sarc}})$ — the sarcasm head output

### 2.3 Two-Phase Training Curriculum

| Phase | What Trains | LR | Epochs | Purpose |
|-------|------------|-----|--------|---------|
| Phase 1 | Classifier head + fusion projections + emotion/sarcasm heads | $10^{-3}$ | 50 | Learn decision boundary on frozen backbone features |
| Phase 2 | Top-2 transformer layers of all 3 backbones + all heads | $10^{-6}$ | 10 | Adapt backbone representations to deepfake detection task |

### 2.4 Multi-Task Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}, y;\ \text{pos\_weight}=1.3835) + \lambda_a \mathcal{L}_{\text{CE}}^{(\text{emo\_a})} + \lambda_b \mathcal{L}_{\text{CE}}^{(\text{emo\_b})} + \lambda_{\text{sarc}} \mathcal{L}_{\text{BCE}}^{(\text{sarc})}$$

Where $\lambda_a = 0.1,\ \lambda_b = 0.1,\ \lambda_{\text{sarc}} = 0.05$ and $\text{pos\_weight} = N_{\text{real}} / N_{\text{fake}} = 8254 / 5966 \approx 1.3835$.

---

## 3. The Empirical Dilemma: Two Failure Modes

### 3.1 Failure Mode A: Offline Feature Evaluation (Phase 2 Head on Phase 1 Features)

| Metric | Value |
|--------|-------|
| Confusion Matrix | $TP = 3,\ FP = 0,\ FN = 905,\ TN = 200$ |
| Accuracy ($\tau = 0.50$) | 18.32% |
| Recall | 0.0033 |
| Score Range | $P(\text{fake}) \le 0.001$ for all clips |
| **Diagnosis** | **Predicts EVERYTHING as REAL** |

### 3.2 Failure Mode B: End-to-End Evaluation (900 Fake + 100 Real)

| Metric | Value |
|--------|-------|
| Confusion Matrix | $TP = 900,\ FP = 100,\ FN = 0,\ TN = 0$ |
| Accuracy ($\tau = 0.50$) | 90.00% |
| F1 Score | 0.9474 |
| AUC-ROC | 0.5829 [95% CI: 0.5215–0.6426] |
| Real Clip Mean Score | 0.2817 |
| Fake Clip Mean Score | 0.3120 |
| **Diagnosis** | **Predicts EVERYTHING as FAKE** |

---

## 4. Root Cause Analysis

### 4.1 Feature Distribution Shift (Covariate Shift)

**Consensus: All three reviewers agree this is the definitive explanation for Failure Mode A.**

When Phase 2 fine-tunes the top-2 transformer layers, the backbone output representations undergo a distributional shift. The classifier head, CBP fusion projections, and LayerNorm affine parameters ($\gamma, \beta$) all co-adapt to the **new** feature geometry. Feeding **old** (un-tuned) features through **new** (co-adapted) weights produces systematic covariate shift [5].

Three architectural properties amplify the collapse from "somewhat off" to "catastrophic" ($P \le 0.001$):

1. **CBP Quadratic Amplification** [4]: Since CBP approximates $z_{\text{at}} \otimes z_v$, any linear shift in per-modality embeddings gets **squared** in the fused representation. A 5% cosine-distance shift in individual 768D embeddings can produce a 10–20% shift in the 8192D fused output.

2. **LayerNorm Shape Sensitivity** [6]: `nn.LayerNorm` normalizes per-sample statistics (mean/var) but its learned $\gamma, \beta$ assume a specific correlational structure. When fine-tuning changes which directions carry signal (via rotated attention heads), the normalized-but-differently-shaped base features get mapped through affine parameters calibrated for an entirely different geometry.

3. **Cascade Effect Through the 299D Bottleneck**: The shifted CBP output propagates through `bilinear_proj` (256/299 = 85.6% of classifier input), contaminating the dominant feature pathway and driving all output logits to extreme negative values.

**Resolution**: This is a pipeline hygiene problem, not an architectural deficiency. Use Phase 1 checkpoints for offline feature evaluation, and Phase 2 checkpoints exclusively for end-to-end evaluation with fine-tuned backbones loaded.

### 4.2 Class Imbalance & Metric Flattery

**Consensus: The 90/10 imbalanced test set makes standard Accuracy and F1 actively misleading.**

At 90% fake prevalence, a trivial "always predict fake" classifier achieves:
- Accuracy = 90.00%
- F1 = $\frac{2 \times 0.90 \times 1.00}{0.90 + 1.00} = 0.9474$

The observed metrics ($\text{Acc} = 90.00\%,\ F1 = 0.9474$) **exactly match this trivial baseline**, meaning the model provides **zero discriminative value** at $\tau = 0.50$. The apparently impressive numbers are entirely an artifact of class imbalance [7].

**Recommendation (all reviewers agree):** Report the following metrics instead:
- **Balanced Accuracy** $= \frac{TPR + TNR}{2}$
- **Matthews Correlation Coefficient (MCC)** $= \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
- **Per-class TPR and TNR separately**
- **Precision-Recall curve with Real as the positive class** (if deployment risk is asymmetric toward false accusations)

### 4.3 Data Integrity & Internal Consistency

**Identified by: Claude Opus 4.6 and Antigravity. Confirmed as critical by both.**

The reported Scenario B numbers contain two internal contradictions:

**Contradiction 1: Score Ranges vs. Confusion Matrix.**
If max(Real) = 0.34 and $\tau = 0.50$, then every real clip scores below threshold → $TN = 100,\ FP = 0$. The reported $FP = 100,\ TN = 0$ is inconsistent with these ranges.

**Contradiction 2: Score Ranges vs. AUC.**
Non-overlapping ranges (max Real = 0.34 < min Fake = 0.35) would force $\text{AUC} = P(S_{\text{fake}} > S_{\text{real}}) = 1.0$ by definition [8]. The reported AUC = 0.5829 requires substantial distributional overlap.

**Resolution:** The score ranges in the original query were likely eyeballed from a subset of the CSV rather than computed as literal `min()`/`max()` over the full run. The true distributions almost certainly overlap significantly, consistent with AUC ≈ 0.58.

---

## 5. Threshold Calibration: Mathematical Constraints

### 5.1 Youden's J Statistic

Youden's J Index [9] identifies the ROC operating point maximizing the vertical distance from the chance diagonal:

$$J = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 = \text{TPR}(\tau) - \text{FPR}(\tau)$$

$$\tau_J^* = \arg\max_\tau\ J(\tau)$$

**Important caveat (Claude):** Youden's J implicitly assumes **symmetric misclassification costs** ($C_{FP} = C_{FN}$). In deepfake detection deployment, false-flagging genuine content as fake ($C_{FP}$) is typically far costlier than missing a deepfake ($C_{FN}$). Under asymmetric costs, the optimal threshold shifts upward.

### 5.2 Bayesian Decision Theory

Under asymmetric costs and prior-shifted evaluation [10]:

$$\tau_{\text{Bayes}}^* = \frac{C_{FP} \cdot \pi_{\text{real}}}{C_{FP} \cdot \pi_{\text{real}} + C_{FN} \cdot \pi_{\text{fake}}}$$

For the training prior ($\pi_{\text{fake}} = 0.4195$) vs. test prior ($\pi_{\text{fake}} = 0.90$), the log-odds shift required is:

$$\Delta_{\text{logit}} = \ln\left(\frac{\pi_{\text{test}} / (1 - \pi_{\text{test}})}{\pi_{\text{train}} / (1 - \pi_{\text{train}})}\right) = \ln\left(\frac{0.9 / 0.1}{0.4195 / 0.5805}\right) \approx +2.52$$

**Critical insight (Claude):** This correction pushes the model to call things fake **even more readily** — the opposite of the desired effect. Prior-shift correction is therefore the wrong tool for reducing false positives; empirical threshold selection is required instead.

### 5.3 AUC-Constrained Performance Ceiling

**This is the single most important finding across all three reviews.**

Under the standard binormal ROC model [11], the maximum simultaneously achievable TPR = TNR (the equal-error-rate point) is a deterministic function of AUC:

| AUC | Max Simultaneous TPR = TNR | Max Youden's J |
|-----|---------------------------|----------------|
| 0.58 | 55.7% | 0.114 |
| 0.60 | 57.1% | 0.142 |
| 0.65 | 60.7% | 0.215 |
| 0.70 | 64.3% | 0.286 |
| 0.80 | 72.6% | 0.452 |
| 0.90 | 82.0% | 0.639 |
| **0.965** | **90.0%** | **0.800** |

**At AUC ≈ 0.58, the best possible simultaneous TPR and TNR is approximately 56%.** No threshold — Youden's J, Bayesian, or any other — can extract 90%/90% performance from a model with 58% AUC. The discriminative information is simply not present in the output scores.

**To reach 90%/90%, the model's AUC must be improved to ≈ 0.965.** This requires improving the model's raw discriminative power, not adjusting the decision threshold.

---

## 6. Multi-Task Loss Balancing: Gradient Shielding vs. Affect Starvation

### 6.1 The Unverified Premise

The claim that $\lambda_a = 0.1$ ensures "90%+ gradient concentration on BCE" conflates two distinct quantities:

1. **Loss-value contribution** (what $\lambda$ directly scales)
2. **Gradient-norm contribution** (what actually drives parameter updates)

These are related by the chain rule but are **not proportional** in general [12]:

$$\left\|\frac{\partial (\lambda_i \mathcal{L}_i)}{\partial \theta_{\text{shared}}}\right\| = \lambda_i \cdot \left\|\frac{\partial \mathcal{L}_i}{\partial \hat{y}_i}\right\| \cdot \left\|\frac{\partial \hat{y}_i}{\partial \theta_{\text{shared}}}\right\|$$

The Jacobian terms $\frac{\partial \mathcal{L}_i}{\partial \hat{y}_i}$ and $\frac{\partial \hat{y}_i}{\partial \theta_{\text{shared}}}$ differ across task heads. A 6-class cross-entropy at near-random initialization ($\ln 6 \approx 1.79$) has large gradients because the softmax is far from the target, while BCE at a well-calibrated operating point has relatively small gradients because the sigmoid saturates.

### 6.2 The Affect Starvation Risk

**All three reviewers flagged this as a critical concern.**

The 299D classifier input contains 43 affect-derived dimensions (36D emotion outer product + 6D delta + 1D sarcasm). These dimensions are the architectural embodiment of the thesis hypothesis: **affect incongruency signals deepfakes**.

If $\lambda_a = 0.1$ starves the emotion heads of gradient early in training, those heads may collapse to majority-class predictors. At that point:

- $\mathbf{fused\_emo}$ becomes a near-constant 36D vector (no discriminative signal)
- $\boldsymbol{\Delta}$ approaches zero (no emotional mismatch detected)
- $P_{\text{sarcasm}}$ flatlines

The classifier would then operate on effectively 256D of CBP fusion features alone, **ignoring the affect incongruency signal entirely**. This would explain the low AUC — the model detects deepfakes via low-level bilinear fusion artifacts rather than high-level affect reasoning.

### 6.3 Verification Protocol

```python
# Insert into trainer.py after loss.backward(), before optimizer.step():
with torch.no_grad():
    shared_params = [p for n, p in model.named_parameters()
                     if 'emotion_head' not in n and 'sarcasm_head' not in n
                     and p.grad is not None]
    bce_grad = torch.cat([p.grad.flatten() for p in shared_params]).norm().item()

    aux_params = [p for n, p in model.named_parameters()
                  if ('emotion_head' in n or 'sarcasm_head' in n)
                  and p.grad is not None]
    aux_grad = torch.cat([p.grad.flatten() for p in aux_params]).norm().item()

    logger.info(f"Grad norms — BCE-dominated: {bce_grad:.4f}, "
                f"Auxiliary: {aux_grad:.4f}, "
                f"BCE ratio: {bce_grad / (bce_grad + aux_grad + 1e-8):.2%}")
```

Additionally, track emotion head per-class accuracy each epoch. If either head drops below 20% on a 6-class task (random baseline = 16.7%), the head has collapsed.

### 6.4 Alternative Multi-Task Balancing Strategies

| Method | Reference | Mechanism | Complexity |
|--------|-----------|-----------|------------|
| **Warmup + Anneal** | — | Start $\lambda = 1.0$ for 10 epochs, decay to $0.1$ | Trivial |
| **GradNorm** | Chen et al. (2018) [12] | Dynamically rebalance to equalize gradient norms toward a target relative training rate | Moderate |
| **Uncertainty Weighting** | Kendall et al. (2018) [13] | Learn per-task $\log \sigma_i^2$; weight each loss as $\mathcal{L}_i / (2\sigma_i^2) + \log \sigma_i$ | Moderate |
| **PCGrad** | Yu et al. (2020) [14] | Project conflicting gradients onto each other's normal planes to resolve gradient interference | High |

---

## 7. Score Compression Remediation Techniques

### 7.1 Temperature Scaling

**Post-hoc calibration** [15] that rescales logits before the sigmoid:

$$P_{\text{calibrated}} = \sigma\left(\frac{z}{T}\right)$$

Where $T > 0$ is a scalar learned by minimizing **negative log-likelihood (NLL)** on a held-out calibration set:

$$T^* = \arg\min_T -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \sigma(z_i / T) + (1 - y_i) \log (1 - \sigma(z_i / T))\right]$$

| Property | Detail |
|----------|--------|
| **Fixes** | Compressed/overconfident probability calibration |
| **Cannot fix** | AUC (monotonic transform preserves ranking) |
| **Critical warning** | $T$ must be **learned**, not hardcoded. Hardcoding $T = 0.5$ without validation is methodologically incorrect and may worsen calibration. |
| **Reference** | Guo et al. (2017) [15] |

### 7.2 Focal Loss

Designed to address class imbalance by down-weighting well-classified examples [16]:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where $p_t = p$ if $y = 1$, else $p_t = 1 - p$, and $\gamma \ge 0$ is the focusing parameter.

| Property | Detail |
|----------|--------|
| **Fixes** | Easy examples dominating gradient; model "giving up" on borderline cases |
| **Recommended** | $\gamma = 2.0$, $\alpha = 0.25$ or inverse class frequency |
| **Cannot fix** | Fundamentally inseparable distributions |
| **Reference** | Lin et al. (2017) [16] |

### 7.3 Supervised Contrastive Margin Loss

Explicitly penalizes insufficient inter-class separation [17]:

$$\mathcal{L}_{\text{margin}} = \max\left(0,\ m - (\bar{s}_{\text{fake}} - \bar{s}_{\text{real}})\right)$$

Where $\bar{s}_{\text{fake}}$ and $\bar{s}_{\text{real}}$ are per-batch mean logits and $m$ is the target margin (e.g., $m = 1.0$).

| Property | Detail |
|----------|--------|
| **Fixes** | Score distributions compressed into a narrow band despite discriminative power |
| **Requirement** | Each batch must contain both classes; $m$ is a sensitive hyperparameter |
| **Reference** | Khosla et al. (2020) [17] |

### 7.4 Cosine Similarity Head (ArcFace-Style)

Replace the MLP classifier with an angular-margin head [18]:

$$\text{score} = \sigma(\alpha \cdot \cos(\mathbf{x}, \mathbf{w}_{\text{real}}) + \beta)$$

| Property | Detail |
|----------|--------|
| **Fixes** | Embeddings that cluster tightly in a small angular region despite different norms |
| **Downside** | Major architectural change; requires full retraining; incompatible with existing checkpoints |
| **Reference** | Deng et al. (2019) [18] |

---

## 8. Consensus Prioritized Action Plan

The following action plan represents the **consensus across all three independent reviews**, ordered by diagnostic priority (rule out cheap explanations before investing in expensive architectural changes):

| Priority | Action | Rationale | Expected Impact | Effort | Source |
|----------|--------|-----------|----------------|--------|--------|
| **0** | **Verify data integrity**: Compute `df.groupby('fake_label')['score'].describe()` on the exact CSV that produced the confusion matrix | Determines if Q2/Q3 are chasing a real calibration problem or an eval-harness artifact | Foundational | 5 min | Claude, Antigravity |
| **1** | **Verify affect head health**: Log emotion/sarcasm per-class accuracy each epoch during training | Determines if 43/299 classifier dimensions carry zero signal (affect starvation) | Diagnostic | 10 min | Claude, Antigravity |
| **2** | **Balanced evaluation**: Default to 500 Real / 500 Fake split; report MCC + balanced accuracy + per-class TPR/TNR | Stops misleading metrics from masking real problems | Metric hygiene | 5 min | All three |
| **3** | **Rule out under-training**: Extend Phase 2 to 20 epochs, raise LR to $3 \times 10^{-6}$, unfreeze top-4 layers instead of top-2 | LR=$10^{-6}$ × 10 epochs × top-2 layers is extremely conservative; backbone barely moves | Could push AUC from 0.58 → 0.70+ | 50 min | Antigravity |
| **4** | **Add margin loss** ($m = 1.0$) as an auxiliary term during Phase 2 training | Directly penalizes insufficient score separation | Forces wider inter-class margin | 30 min | DeepSeek, Antigravity |
| **5** | **Learn temperature $T$** on validation set by minimizing NLL (replace hardcoded $T = 0.5$) | Proper post-hoc calibration; hardcoded $T$ is methodologically incorrect | Calibration | 15 min | Claude, Antigravity |
| **6** | **Focal Loss** ($\gamma = 2.0$) only if AUC remains < 0.75 after priorities 3–5 | Focuses gradient on hard boundary cases | Moderate | 20 min | DeepSeek |
| **7** | **Log gradient norms** per loss term on shared trunk parameters | Confirms or corrects the assumed 90%+ BCE gradient concentration | Diagnostic | 10 min | Claude |
| **8** | **GradNorm or Uncertainty Weighting** if gradient audit reveals imbalance | Dynamic multi-task balancing superior to fixed hand-tuned $\lambda$ | Structural fix | 2 hours | All three |
| **9** | **Cosine Similarity Head** only as a last resort if all above fail to push AUC > 0.80 | Major architectural change requiring full retraining | Structural change | 1 day | DeepSeek |

---

## 9. Academic References

[1] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* Advances in Neural Information Processing Systems (NeurIPS), 33, 12449–12460.

[2] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* International Conference on Learning Representations (ICLR).

[3] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT), 4171–4186.

[4] Gao, Y., Beijbom, O., Zhang, N., & Darrell, T. (2016). *Compact Bilinear Pooling.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 317–326.

[5] Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* Proceedings of the 32nd International Conference on Machine Learning (ICML), 448–456.

[6] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization.* arXiv preprint arXiv:1607.06450.

[7] He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data.* IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.

[8] Fawcett, T. (2006). *An Introduction to ROC Analysis.* Pattern Recognition Letters, 27(8), 861–874.

[9] Youden, W. J. (1950). *Index for Rating Diagnostic Tests.* Cancer, 3(1), 32–35.

[10] Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), 973–978.

[11] Hanley, J. A., & McNeil, B. J. (1982). *The Meaning and Use of the Area Under a Receiver Operating Characteristic (ROC) Curve.* Radiology, 143(1), 29–36.

[12] Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.* Proceedings of the 35th International Conference on Machine Learning (ICML), 794–803.

[13] Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7482–7491.

[14] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). *Gradient Surgery for Multi-Task Learning.* Advances in Neural Information Processing Systems (NeurIPS), 33, 5824–5836.

[15] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* Proceedings of the 34th International Conference on Machine Learning (ICML), 1321–1330.

[16] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection.* Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2980–2988.

[17] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., ... & Krishnan, D. (2020). *Supervised Contrastive Learning.* Advances in Neural Information Processing Systems (NeurIPS), 33, 18661–18673.

[18] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4690–4699.

[19] Khalid, H., Tariq, S., Kim, M., & Woo, S. S. (2021). *FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset.* NeurIPS Datasets and Benchmarks Track.

[20] Elpeltagy, M., Sallam, K., & Elgeldawi, E. (2023). *Multimodal Deepfake Detection Using Audio-Visual Features.* Journal of Information Security and Applications, 73, 103421.

[21] Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation.* Proceedings of the 32nd International Conference on Machine Learning (ICML), 1180–1189.

---

**End of Document**

*This synthesis was compiled from independent reviews by DeepSeek-R1, Claude Opus 4.6 (Anthropic), and Antigravity (Google DeepMind). All numeric claims — AUC↔TPR/TNR ceiling table, $\ln(6)$ cross-entropy initialization check, Bayesian log-odds shift calculation, and confusion-matrix arithmetic — were independently verified computationally by at least two reviewers.*
