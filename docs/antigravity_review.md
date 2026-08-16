# DeepSentinel Evaluation Dilemma: Independent Review (Antigravity / Claude Opus 4.6)

**Date:** 2026-08-15
**Subject:** Resolving Probability Score Compression & Decision Boundary Oscillation

---

## Preliminary Note: The Reported Numbers Are Internally Inconsistent

Before addressing the five questions, I must flag a critical data-integrity issue that undermines any threshold calibration work built on these numbers.

**You report for Scenario B:**
- Real clip scores: mean 0.2817, range 0.2022–0.3242, max 0.34
- Fake clip scores: range 0.35–0.7870
- Confusion matrix at τ=0.50: TP=900, FP=100, FN=0, TN=0
- AUC-ROC: 0.5829

**These four facts cannot all be true simultaneously:**

1. If max(Real) = 0.34 and threshold τ = 0.50, then **every real clip scores below the threshold**, so the model should predict them all as REAL → TN=100, FP=0. You report the opposite (TN=0, FP=100). This means either (a) the score ranges were not computed from the same run as the confusion matrix, or (b) there is a bug in your evaluation harness (e.g., the `pred = 1 if score >= 0.5 else 0` line is operating on raw logits rather than post-sigmoid probabilities, or labels are inverted).

2. If the score ranges genuinely don't overlap (max Real = 0.34 < min Fake = 0.35), then by the probabilistic definition of AUC:
   $$\text{AUC} = P(\text{score}_{\text{fake}} > \text{score}_{\text{real}}) = 1.0$$
   You report AUC = 0.5829, which requires **substantial overlap** between the two distributions. This directly contradicts the stated non-overlapping ranges.

**Most likely explanation:** The "score ranges" in the query were cherry-picked or eyeballed from a small subset of the CSV, not computed as the literal `min()`/`max()` over the full 1,000-clip run. The actual per-clip score distribution almost certainly has significant overlap between Real and Fake classes, which is consistent with AUC ≈ 0.58.

> **Action required before trusting any threshold derived below:** Run `df.groupby('fake_label')['score'].describe()` on the exact CSV that produced the TP/FP/FN/TN confusion matrix, and report the true min/max/mean/std for each class. All downstream calibration depends on this.

---

## Q1: Feature Distribution Shift (Scenario A Collapse)

**Is evaluating Phase 2 classifier heads on Phase 1 / base-backbone features mathematically invalid?**

**Yes, unambiguously.** This is textbook covariate shift, and the severity of the collapse ($P \le 0.001$) is expected given the architecture.

### Why the collapse is so extreme (not just "somewhat off"):

**Layer 1: Backbone weight divergence.**
Phase 2 fine-tuning updates the top-2 transformer layers of Wav2Vec2, ViT, and BERT. Even at LR=$10^{-6}$ over 10 epochs, each attention head's $W_Q, W_K, W_V$ matrices rotate the representation subspace. The classifier head's Linear weights are calibrated to the **post-rotation** feature geometry. Feeding pre-rotation features through post-rotation weights produces dot products that are systematically off-center.

**Layer 2: CBP amplification.**
Compact Bilinear Pooling approximates the outer product $z_{at} \otimes z_v$ via tensor sketch. Any shift in the individual 768D embeddings gets **squared** in the 8192D fused representation. A 5% cosine-distance shift in per-modality embeddings can produce a 10–20% shift in the fused representation, which then propagates through `bilinear_proj` (256D projection) and `proj_ln` (LayerNorm).

**Layer 3: LayerNorm co-adaptation.**
`nn.LayerNorm(512)` and `nn.LayerNorm(128)` have learned $\gamma, \beta$ parameters that assume a specific input distribution shape. When the input distribution shifts, LayerNorm normalizes the *statistics* (mean/var) but **not the correlational structure**. The learned affine transform ($\gamma \cdot \hat{x} + \beta$) maps the wrong features through the wrong scaling, driving output logits to extreme negative values.

### The correct mental model:

Think of the Phase 2 classifier head as a lock that was re-keyed during fine-tuning. The Phase 1 features are the old key — same shape, wrong grooves. The lock doesn't just fail to open; it jams hard in one direction (all-negative logits → all-REAL predictions).

### Fix:

This is not an architectural problem. It's a deployment hygiene problem with exactly two valid solutions:
1. **For offline/cached evaluation:** Use `best_phase1_bottleneck.pt` (trained on the same base-backbone features).
2. **For Phase 2 evaluation:** Run end-to-end through the fine-tuned backbones (no caching).

There is no shortcut that avoids this constraint.

---

## Q2: Class Imbalance & Threshold Squeezing (Scenario B)

**Why does τ=0.50 produce TN=0 despite Real scores being lower than Fake scores?**

Setting aside the data-integrity issue flagged above, here is what is actually happening mechanically:

### The threshold τ=0.50 is not the problem. The scores are the problem.

If AUC is truly 0.5829, then the Real and Fake score distributions **overlap massively**. The mean separation is only $\Delta\mu \approx 0.03$ (Real mean ≈ 0.28, Fake mean ≈ 0.31), with standard deviations likely ≈ 0.08–0.12 each. This means:

- Many Real clips score **above** 0.30
- Many Fake clips score **below** 0.30
- At **any** single threshold, you're cutting through the middle of both distributions

The confusion matrix TP=900, FP=100, TN=0, FN=0 at τ=0.50 would only occur if **every single clip** (Real and Fake alike) scored ≥ 0.50. This is consistent with a model that has a positive output bias, pushing all sigmoid outputs above 0.50 regardless of class — a miscalibrated decision boundary, not a class-imbalance artifact per se.

### What class imbalance actually does here:

Class imbalance doesn't change the model's per-clip output scores. It changes **which metrics flatter you.** At 90% fake prevalence:
- A trivial "always predict fake" classifier achieves 90% accuracy and F1 = 0.9474
- Your reported numbers exactly match this trivial baseline
- This means the model is providing **zero discriminative value** at τ=0.50

> **Recommendation:** Stop reporting Accuracy and F1 on imbalanced sets. Use **balanced accuracy** $(= \frac{TPR + TNR}{2})$, **MCC** (Matthews Correlation Coefficient), and **per-class TPR/TNR** as primary metrics.

---

## Q3: Optimal Threshold Calibration

**Can Youden's J at τ*≈0.34 achieve TP>90% AND TN>90%?**

**No.** This is mathematically impossible at your current AUC.

### The hard constraint:

Under the standard binormal ROC model, the maximum simultaneously achievable TPR=TNR (the "equal-error-rate" point) is a deterministic function of AUC:

| AUC | Max simultaneous TPR = TNR | Youden's J |
|-----|---------------------------|------------|
| 0.58 | 55.7% | 0.114 |
| 0.60 | 57.1% | 0.142 |
| 0.65 | 60.7% | 0.215 |
| 0.70 | 64.3% | 0.286 |
| 0.80 | 72.6% | 0.452 |
| 0.90 | 82.0% | 0.639 |
| **0.965** | **90.0%** | **0.800** |

At AUC ≈ 0.58, the best possible simultaneous TPR and TNR is roughly **56%**. No threshold — Youden's J, Bayesian, or otherwise — can extract 90%/90% performance from a model with 58% AUC. The information simply isn't in the scores.

### What Youden's J *will* give you:

The optimal $\tau_J$ will find the point of maximum vertical distance from the ROC diagonal. At AUC=0.58, expect:
- $\tau_J \approx 0.30$–$0.35$
- TPR ≈ 60%, TNR ≈ 55% (or similar asymmetric split)
- Youden's J ≈ 0.11–0.15

This is a meaningful improvement over the current "all-fake" prediction, but it's not the 90%/90% performance you're hoping for. **To reach 90%/90%, you need to improve the model's raw discriminative power (AUC), not just move the threshold.**

### The Bayesian perspective:

Bayesian Decision Theory provides the cost-optimal threshold:
$$\tau^* = \frac{C_{FP} \cdot \pi_{\text{real}}}{C_{FP} \cdot \pi_{\text{real}} + C_{FN} \cdot \pi_{\text{fake}}}$$

But this optimizes **expected cost**, not simultaneous TPR/TNR. If your deployment scenario penalizes false accusations of real content (high $C_{FP}$), the Bayesian threshold will be **higher** than Youden's J, sacrificing some recall to protect specificity. This is the right framework for deployment decisions, but it still cannot exceed the AUC ceiling.

---

## Q4: Multi-Task Loss Balancing

**Does scaling $\lambda_a=0.1, \lambda_b=0.1, \lambda_{\text{sarc}}=0.05$ guarantee 90%+ gradient concentration on BCE?**

**No. Loss-value scaling ≠ gradient-norm scaling.** This is a common and dangerous assumption.

### Why the arithmetic doesn't transfer:

The gradient of a loss term w.r.t. shared parameters depends on:
1. The loss value (scaled by λ) — **this is what you control**
2. The Jacobian of the loss w.r.t. the model output
3. The Jacobian of the model output w.r.t. the shared parameters (chain rule through the network)

Items 2 and 3 are **different for each task head**. A 6-class cross-entropy loss near its random-initialization value ($\ln 6 \approx 1.79$, matching your reported ~1.80) can have gradient norms comparable to or exceeding BCE's, even after 0.1× scaling, because:

- The softmax Jacobian distributes gradients across 6 output units
- The chain rule through the emotion head's own Linear layers compounds with the chain rule through shared fusion/backbone parameters
- BCE at a well-calibrated operating point has relatively *small* gradients (the sigmoid saturates), while cross-entropy at near-random initialization has *large* gradients (the softmax is far from the target)

### The real risk: affect feature starvation

Your 299D classifier input is:
- 256D: `fused_proj` (CBP projection)
- 36D: `fused_emo` (emotion outer product)
- 6D: `Δ` (emotion probability delta)
- 1D: `P_sarcasm`

The 43 affect-derived dimensions (36+6+1) are the architectural embodiment of your thesis hypothesis (affect incongruency signals deepfakes). If $\lambda_a=0.1$ and $\lambda_b=0.1$ are too small, the emotion heads may collapse to majority-class predictors early in training, at which point:
- `fused_emo` becomes a near-constant 36D vector (no discriminative signal)
- `Δ` becomes near-zero (no emotional mismatch to detect)
- `P_sarcasm` flatlines

The classifier would then be operating on effectively 256D of CBP fusion features, ignoring the affect incongruency signal entirely. This would explain the low AUC — the model has learned to detect deepfakes via low-level fusion artifacts alone, without the intended high-level affect reasoning.

### Verification protocol:

```python
# Add to training loop after loss.backward():
grad_bce = torch.cat([p.grad.flatten() for p in shared_params]).norm()
grad_aux = torch.cat([p.grad.flatten() for p in aux_params]).norm()
print(f"BCE grad norm: {grad_bce:.4f}, Aux grad norm: {grad_aux:.4f}, Ratio: {grad_bce/(grad_bce+grad_aux):.2%}")
```

Also track `emotion_head_a` and `emotion_head_b` per-class accuracy over training epochs. If either drops below 20% accuracy on a 6-class task (random = 16.7%), the head has collapsed.

### Better alternatives to fixed λ:

| Method | Mechanism | Complexity |
|--------|-----------|------------|
| **Warmup + anneal** | Start $\lambda_a = \lambda_b = 1.0$ for 10 epochs, then decay to 0.1 | Trivial |
| **GradNorm** (Chen et al., 2018) | Dynamically rebalance to equalize gradient norms | Moderate |
| **Uncertainty weighting** (Kendall et al., 2018) | Learn per-task $\log \sigma_i^2$; auto-downweight noisy tasks | Moderate |

---

## Q5: Preventing Score Compression

**What techniques can widen the margin between Real (≈0.28) and Fake (≈0.34)?**

### Diagnosis first: Why is the margin only ~0.06?

Before reaching for architectural fixes, consider the three most common causes of score compression in order of likelihood:

**1. Under-training (most likely).**
LR=$10^{-6}$ × 10 epochs × top-2 layers only is an extremely conservative fine-tuning schedule. The backbone is barely moving. The decision boundary may simply not have had enough gradient updates to separate the classes. Try:
- Doubling Phase 2 to 20 epochs
- Raising LR to $3 \times 10^{-6}$ with cosine annealing
- Unfreezing top-4 layers instead of top-2

**2. Domain gap (likely).**
Your training data is CREMA-D, MELD, CMU-MOSEI, and MuStARD — all real human emotion/sarcasm datasets, plus synthetic deepfake tracks. FakeAVCeleb is a completely different domain (celebrity faces, different lighting, different audio recording conditions). Some compression is inevitable when evaluating out-of-distribution.

**3. Affect feature collapse (needs verification — see Q4).**
If the emotion/sarcasm heads have collapsed, 43 of 299 input dimensions carry no signal, and the classifier is working with reduced effective dimensionality.

### Techniques (ordered by implementation effort):

**A. Temperature Scaling (Post-Hoc, Zero Retraining)**

$$P = \sigma(\text{logit} / T), \quad T \text{ fit on held-out set to minimize NLL}$$

- **What it fixes:** Compressed calibration around 0.5
- **What it cannot fix:** AUC. Temperature scaling is a monotonic transform — it cannot change the ranking of clips, only stretch/compress the probability axis. If AUC is 0.58, it stays 0.58 after temperature scaling.
- **When to use:** After all other fixes, as a final calibration step.
- **Critical note on your implementation:** You've hardcoded $T=0.5$. This is wrong. $T$ should be **learned** on a held-out calibration set by minimizing negative log-likelihood, not set by hand. A wrong $T$ can make calibration worse.

**B. Focal Loss (Moderate Retraining)**

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **What it fixes:** Focuses gradient on hard-to-classify clips near the decision boundary
- **Recommended settings:** $\gamma = 2.0$, $\alpha = 0.25$ (or set $\alpha$ = inverse class frequency)
- **When to use:** When easy examples (clearly real / clearly fake) dominate the gradient and the model "gives up" on borderline cases

**C. Margin-Based Contrastive Loss (Significant Retraining)**

$$\mathcal{L}_{\text{margin}} = \max(0, m - (s_{\text{fake}} - s_{\text{real}}))$$

- **What it fixes:** Directly penalizes insufficient separation between class logit means
- **Downside:** Requires careful batch construction (each batch needs both classes) and $m$ is a sensitive hyperparameter
- **When to use:** When the model has discriminative power (AUC > 0.70) but outputs are compressed into a narrow band

**D. Cosine Similarity Head (Major Architectural Change)**

Replace the MLP classifier with:
$$\text{score} = \sigma(\alpha \cdot \cos(\mathbf{x}, \mathbf{w}_{\text{real}}) + \beta)$$

- **What it fixes:** Forces the model to learn angularly separated embeddings rather than linearly separated logits
- **When to use:** If the 299D representations cluster tightly in a small angular region despite having different norms
- **Downside:** Requires retraining from scratch; incompatible with current checkpoint

### My recommended priority order:

1. **Verify affect head health** (Q4) — 10 minutes of logging
2. **Extend Phase 2 training** (20 epochs, LR=$3 \times 10^{-6}$, top-4 layers) — 50 minutes on T4
3. **Add margin loss** as an auxiliary term — 30 minutes of code
4. **Replace hardcoded T=0.5 with learned temperature** on validation set — 15 minutes
5. **Focal Loss** only if the above don't push AUC above 0.75

---

## Summary: Prioritized Action Table

| Priority | Action | Expected Impact | Effort |
|----------|--------|----------------|--------|
| **0** | Fix data integrity (§0): verify score ranges from the actual CSV | Determines if Q2/Q3 are real problems or eval-harness bugs | 5 min |
| **1** | Verify affect head health: log emotion/sarcasm accuracy per epoch | Determines if 43/299 dimensions are dead weight | 10 min |
| **2** | Extend Phase 2: 20 epochs, LR=$3 \times 10^{-6}$, top-4 layers | Could push AUC from 0.58 → 0.70+ if under-training is the cause | 50 min |
| **3** | Balanced evaluation: 500/500 split, report MCC + balanced accuracy | Stops misleading metrics from hiding real problems | 5 min |
| **4** | Add margin loss ($m=1.0$) to Phase 2 training | Directly forces score separation | 30 min |
| **5** | Learn temperature $T$ on validation set (not hardcoded) | Proper calibration after model is fixed | 15 min |
| **6** | Focal Loss ($\gamma=2.0$) | Focus on hard boundary cases | 20 min |

---

*This review was generated independently, treating the query document as a cold-start problem. Numeric claims (AUC↔TPR/TNR table, ln(6) cross-entropy check, Bayesian threshold formula) were verified computationally rather than estimated by hand.*
