# DeepSentinel Evaluation Review
### Independent architectural & statistical read on the Scenario A/B evaluation dilemma

---

## TL;DR

| Finding | Verdict |
|---|---|
| Scenario A (offline features) collapsing to `P(fake) ≤ 0.001` | **Expected, not a model bug.** Classic train/serve skew — Phase 2 head evaluated on Phase-1/base-backbone features. |
| Scenario B's reported numbers (score ranges, confusion matrix, AUC) | **Internally inconsistent.** They cannot all be literally true simultaneously — see §0. Resolve before trusting any threshold derived from them. |
| "Everything predicted fake" at τ=0.50 in Scenario B | Partly a genuine imbalance/calibration issue, partly (per §0) possibly an eval-harness artifact. |
| τ* ≈ 0.34 achieving TP>90% **and** TN>90% | **Not achievable** if AUC is really 0.58–0.65. That pairing requires AUC ≳ 0.965 on a well-behaved ROC curve. |
| λ-scaling (0.1/0.1/0.05) "ensuring" 90%+ gradient concentration on BCE | **Unverified premise.** Loss-value scaling ≠ gradient-norm scaling. Needs to be measured, not assumed. |
| Fastest path to fixing score compression | Rule out under-training and domain gap before reaching for focal loss / margin heads. |

---

## §0. Data Integrity Check — Read This First

Before touching the theory: **Scenario B's reported numbers can't all be literally true at once.**

**Contradiction 1 — the confusion matrix vs. the score ranges.**
You report Real clips topping out at `0.34` (max) and Fake clips starting at `0.35` (min). If that's the literal range:

- Every real clip scores ≤ 0.34, i.e. below τ=0.50 → expected result: **TN = 100, FP = 0**.
- You report the opposite: **TN = 0, FP = 100** — implying every real clip actually scored ≥ 0.50.

These cannot both be correct for the same run.

**Contradiction 2 — the score ranges vs. AUC.**
If the ranges truly don't overlap (max real 0.34 < min fake 0.35), then for *every* real/fake pair, fake > real. By definition,

$$\text{AUC} = P(\text{score}_{\text{fake}} > \text{score}_{\text{real}})$$

so non-overlapping ranges force **AUC = 1.0** exactly. You report AUC ≈ 0.58–0.65, which is only possible if the two distributions overlap *substantially* — the direct opposite of the stated ranges.

**Secondary flag — sample counts don't match between scenarios.**

| | Fake | Real | Total |
|---|---|---|---|
| Scenario A | 908 (TP+FN) | 200 (TN+FP) | 1,108 |
| Scenario B | 900 | 100 | 1,000 |

If A and B are meant to be the same held-out FakeAVCeleb split evaluated two different ways, the class counts should match exactly. They don't — worth reconciling before treating A vs. B as a clean apples-to-apples before/after comparison.

**Most likely explanations (in rough order of likelihood):**
1. The "Range" figures reported aren't the literal min/max from the *same run* that produced the τ=0.50 confusion matrix (e.g., an IQR, a different checkpoint, or a stale/cached summary).
2. **Sigmoid-omission bug** — thresholding the raw `BCEWithLogitsLoss` logit instead of `sigmoid(logit)` at eval time. This is one of the most common eval-harness bugs and would decouple the "threshold" label from what's actually being compared.
3. The confusion matrix and the distribution table were computed from different checkpoints/runs (e.g., one before and one after a fix).
4. An inverted comparison operator or mislabeled class polarity somewhere in the eval loop.

**Recommended fix (≈5 minutes):** Pull the exact per-clip score array that produced `TP=900/FP=100/FN=0/TN=0`, and recompute min/mean/max/AUC directly from *that* array — don't trust a separately-reported summary table. Everything in §2, §3, and §5 below still answers the *general* phenomenon (compressed, overlapping scores evaluated at an unvalidated threshold) regardless of which specific numbers turn out to be correct — but the specific numeric targets (e.g., τ*≈0.34) should be treated as provisional until this is resolved.

---

## §1. Feature Distribution Shift (Offline Eval Collapse)

**Is it mathematically invalid to evaluate the Phase-2 head on Phase-1/base-backbone features? Yes.**

This is standard train/serve skew, not a subtle effect.

### Mechanism

In Phase 2, the classifier head, the CBP fusion projections, and the LayerNorm affine parameters (γ, β) all keep updating *alongside* the unfrozen top-2 backbone layers. By the end of Phase 2, every downstream weight has co-adapted to whatever distribution the **fine-tuned** backbone now outputs — not the base backbone's. Feeding that head base-backbone features is handing a linear layer + LayerNorm an input it was never fit to.

Two things make the collapse this extreme rather than just "somewhat off":

- **LayerNorm normalizes per-sample** (mean/var over the feature dimension), so it won't blow up on raw scale differences the way BatchNorm would. But its learned γ, β still assume a particular *shape*/correlational structure in the normalized features. If fine-tuning changed which directions carry signal — likely, since attention layers mix information nonlinearly — the normalized-but-differently-shaped base features get mapped through weights calibrated for an entirely different shape.
- **CBP is quadratic in its inputs** (an approximation of the outer product between modality vectors via tensor sketch / FFT). Any shrinkage or rotation in the per-modality embeddings gets roughly squared in the fused representation — plausibly explaining why the collapse is as extreme as `P ≤ 0.001` rather than just moderately shifted.

### How to confirm directly

Take 10–20 clips, run them through both pipelines, and compare the raw 768D embeddings (cosine similarity is sufficient) *before* they reach the fusion module. Expect a meaningful gap.

### Fix

Not architectural — pipeline hygiene:
- Re-extract and re-cache features using the **fine-tuned (post-Phase-2)** backbone weights, or
- Keep a Phase-1-only head checkpoint paired specifically with the Phase-1 cached features, if a cached-feature eval is needed for speed.

---

## §2. Class Imbalance & Threshold Squeezing

Two distinct issues are bundled in this question — only one is actually about imbalance.

### 2a. Imbalance vs. calibration are different axes

A fixed τ=0.50 doesn't know or care about the test-set class ratio — it only reflects wherever training happened to place the decision boundary. What imbalance changes is (a) the *cost* of getting the threshold wrong, and (b) how misleading your headline metrics become.

**This is the important part:** if `TN=0` is real, that means **0% specificity** — the model never once correctly identifies a real clip — while still posting 90% accuracy and 0.9474 F1, purely because guessing "fake" for everything is a good bet when 90% of the set is fake.

> **Recommendation:** report **balanced accuracy**, **MCC (Matthews correlation coefficient)**, and **TPR/TNR separately** going forward. Accuracy and F1 will keep flattering you at this imbalance ratio.

### 2b. There's also a genuine train/test prior mismatch

`pos_weight=1.3835` was tuned against the **training** class ratio (~42% fake, from N=5,966/14,220). The FakeAVCeleb eval split is ~90% fake — nearly inverted. Assuming class-conditional feature distributions haven't *also* shifted (a big assumption given §1), the standard correction for a pure prior/label shift is an additive log-odds adjustment:

$$\text{logit}_{\text{test}}(x) = \text{logit}_{\text{train}}(x) + \ln\left(\frac{\pi_{\text{test}}/(1-\pi_{\text{test}})}{\pi_{\text{train}}/(1-\pi_{\text{train}})}\right)$$

| Quantity | Value |
|---|---|
| π_train (fake) | 0.4195 |
| π_test (fake) | 0.9000 |
| **Log-odds shift required** | **+2.52** |

A *correct* prior-shift correction would push the model to call things fake **even more readily** than it already does. This tells you prior-shift correction is the wrong tool for the symptom you actually want to fix (over-flagging real clips) — it would make it worse. What you want instead is empirical threshold selection against the eval-time distribution directly (§3).

---

## §3. Optimal Threshold Calibration (Youden's J / Bayesian Decision Theory)

### Mechanics

$$\tau^* = \arg\max_\tau \big[\text{TPR}(\tau) - \text{FPR}(\tau)\big]$$

computed across the ROC curve on a held-out validation set — the point of maximum vertical distance from the diagonal in ROC space.

### Caveats that matter here

1. **Youden's J assumes symmetric costs.** It has no notion that false-flagging real content as fake is likely far costlier (reputational/legal risk) than missing a fake in deployment. Under asymmetric costs, the Bayes-risk-optimal threshold on a properly calibrated posterior is instead:
   $$\tau^* = \frac{C_{FP}}{C_{FP}+C_{FN}}$$
   equivalently, the ROC point where the curve's local slope equals the cost-weighted prior-odds ratio. Decide explicitly which criterion you're optimizing for.

2. **Calibrate against FakeAVCeleb itself**, not the training distribution — a threshold tuned on training-distribution scores won't transfer cleanly if there's a genuine domain gap on top of §1/§2.

3. **What your AUC actually allows.** Under a simple symmetric-ROC (equal-variance binormal) model:

   | AUC | Best simultaneous TPR = TNR | Max Youden's J |
   |---|---|---|
   | 0.58 | 55.7% | 0.114 |
   | 0.60 | 57.1% | 0.142 |
   | 0.65 | 60.7% | 0.215 |
   | **~0.965 (needed for 90/90)** | **90.0%** | **0.800** |

   At AUC 0.58–0.65, the best achievable *simultaneous* TPR/TNR is roughly **56–61%** — nowhere near the TP>90%/TN>90% pairing implied by τ*≈0.34. That pairing requires an AUC around **0.965** on a well-behaved curve. If the true AUC really is ~0.6, no threshold gets you there — you need to close the domain gap (§1) or improve raw separability (§4/§5), not just move τ.

4. **Real is the minority class (10%) here.** Also inspect the **precision–recall curve with "real" as the positive class**, not just the fake-oriented ROC/J framing — if deployment risk is asymmetric toward not falsely accusing genuine content, that's the more decision-relevant curve.

---

## §4. Multi-Task Loss Balancing

### The premise needs verification, not just trust

Scaling a loss term by λ scales its contribution to the total **loss value** — it does *not* automatically translate to the same percentage of **gradient norm** on shared parameters. Those are different quantities, and the mapping between them depends on each loss's curvature at the current weights.

A 6-way softmax starting near $\ln 6 \approx 1.792$ — which matches your reported ~1.80 starting point, a sane near-random-init value — can still produce gradients comparable in magnitude to BCE's even after a 0.1× scale-down, depending on the current training regime.

> **Recommendation:** log $\lVert\nabla_{\theta_{\text{shared}}}\mathcal{L}_{\text{BCE}}\rVert$ against $\lVert\nabla_{\theta_{\text{shared}}}(\lambda_a\mathcal{L}_{\text{emo\_a}}+\lambda_b\mathcal{L}_{\text{emo\_b}}+\lambda_{\text{sarc}}\mathcal{L}_{\text{sarc}})\rVert$ on the fusion trunk periodically during training, and confirm the *realized* ratio, rather than trusting the loss-weight arithmetic to guarantee a 90/10 split.

### The bigger risk: starved auxiliary heads

The architecture's stated purpose is evaluating **audio-visual affect incongruency** — the 36D `fused_emo`, 6D Δ, and 1D sarcasm score exist specifically to carry that signal into the classifier. If λ_a=0.1, λ_b=0.1, λ_sarc=0.05 starve those heads of gradient early in training, those 43 dimensions of the 299D vector can end up carrying close to **zero** information — not merely deprioritized, but never learned in the first place. That would mean the classifier is effectively working off the 256D fusion projection alone, while still paying the architectural/compute cost of the rest.

> **Recommendation:** track auxiliary-head accuracy/F1 over training (not just loss value) to confirm they're not collapsing to a majority-class predictor.

### Better than fixed hand-tuned λ's

| Method | What it does |
|---|---|
| **GradNorm** (Chen et al., 2018) | Dynamically rebalances weights to equalize *gradient norms* (not loss values) toward a target relative training rate. |
| **Uncertainty weighting** (Kendall et al., 2018) | Learns per-task $\log\sigma_i^2$; weights each loss as $\mathcal{L}_i/(2\sigma_i^2)+\log\sigma_i$, so noisier/harder tasks are automatically down-weighted. |
| **Manual warmup + anneal** | Start auxiliary weights near parity, decay over training — lets affect features form before BCE dominates. Simplest fix if you want to stay hand-tuned. |

---

## §5. Preventing Score Compression

| Technique | Mechanism | What it fixes | What it doesn't |
|---|---|---|---|
| **Temperature scaling** | $P = \sigma(\text{logit}/T)$, T fit post-hoc on held-out data (minimize NLL/ECE) | Under-confident/compressed *calibration* around 0.5 | Cannot move AUC — it's a monotonic transform of an already-fixed ranking |
| **Focal loss** | $FL(p_t) = -\alpha(1-p_t)^\gamma \log p_t$ | Stops easy/well-classified examples from dominating gradient, forcing focus on hard boundary cases | Doesn't guarantee wider *margins* on its own — still a classification loss, not a metric loss |
| **Margin-based head** (ArcFace/CosFace, or a prototype/cosine head + contrastive/triplet loss) | Explicitly penalizes insufficient angular/embedding-space margin between classes during training | Directly targets "how far," not just "which side" | Bigger architectural change; more to validate |

**Two more mundane things to rule out first**, since they're cheaper to test than any of the above:

1. **Under-training.** LR=$10^{-6}$ over only the top-2 unfrozen layers is a conservative fine-tune. Check whether the decision boundary is genuinely under-trained (longer Phase 2, or a slightly higher LR with close monitoring) before assuming an architectural fix is needed.
2. **Domain gap.** FakeAVCeleb is out-of-domain relative to the training distribution. Some of the "compression" may be a domain gap that no amount of loss-function engineering on the training distribution will fix. A small amount of FakeAVCeleb-domain calibration/fine-tuning data would clarify this quickly.

---

## Prioritized Action Plan

| Priority | Action | Why |
|---|---|---|
| 1 | Reconcile Scenario B's numbers (§0) — recompute min/mean/max/AUC from the exact score array behind the τ=0.50 confusion matrix | Determines whether §2/§3/§5 are chasing a real calibration problem or an eval-harness bug |
| 2 | Fix the Scenario A feature-cache mismatch (§1) — re-extract features with fine-tuned backbone weights, or pair Phase-1 features with a Phase-1-only head | Currently invalidates an entire evaluation scenario outright |
| 3 | Recompute balanced accuracy, MCC, TPR/TNR (§2a) alongside accuracy/F1 | Accuracy/F1 are actively misleading at 90/10 imbalance |
| 4 | Log gradient norms per loss term on the shared trunk (§4) | Confirms or corrects the assumed 90%+ BCE gradient concentration |
| 5 | Rule out under-training and domain gap (§5) before adding focal loss / margin heads | Cheapest interventions to test first |
| 6 | Re-derive τ* via ROC/Youden's J or Bayes risk (§3) — only once scores are validated | Threshold tuning against unverified numbers wastes effort |

---

*This review covers the architecture, training setup, and both evaluation scenarios as described. Numeric claims above (log-odds shift, AUC↔Youden's J table, ln(6) check, confusion-matrix arithmetic) were verified computationally rather than estimated by hand.*
