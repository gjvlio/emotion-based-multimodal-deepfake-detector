# External AI Query: Resolving Probability Score Compression & Decision Boundary Oscillation in Multimodal Deepfake Detection

> **Purpose**: Use this standalone prompt to query external LLMs (Claude 3.5 Sonnet, GPT-4o, Gemini 1.5 Pro) for an independent architectural and statistical second opinion on our multimodal deepfake detection evaluation dilemma.

---

```markdown
### **SYSTEM & TASK CONTEXT**

We are developing a **Multimodal Deepfake Detector (DeepSentinel)** that evaluates audio-visual affect incongruency to identify deepfakes across 4 generation pipelines (Faceswap, FSGAN, Wav2Lip, RTVC).

#### **Model Architecture:**
* **Backbones**: Wav2Vec 2.0 (Audio, 768D), ViT-Base (Visual, 8 keyframes, 768D), BERT-Uncased (Text transcript, 768D).
* **Fusion Module**: Compact Bilinear Pooling (CBP) + 299D Multi-Scale Hybrid Bottleneck:
  $$\mathbf{x}_{\text{classifier}} = \text{Concat}(\mathbf{fused\_proj}_{256D}, \mathbf{fused\_emo}_{36D}, \boldsymbol{\Delta}_{6D}, P_{\text{sarcasm}, 1D}) \in \mathbb{R}^{299}$$
* **Classifier**: 2-Layer MLP with `nn.LayerNorm`, GELU activation, Dropout, and a single sigmoid output logit $P(\text{fake}) \in [0, 1]$.
* **Training Setup**:
  * **Phase 1**: Freeze backbones; pre-train classifier head & fusion projections for 50 epochs ($\text{LR}=10^{-3}$).
  * **Phase 2**: Unfreeze top-2 transformer backbone layers; fine-tune end-to-end ($\text{LR}=10^{-6}$, `freeze_layers=2`).
  * **Loss Function**: Multi-Task Loss with `BCEWithLogitsLoss`:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_a \mathcal{L}_{\text{emo\_a}} + \lambda_b \mathcal{L}_{\text{emo\_b}} + \lambda_{\text{sarc}} \mathcal{L}_{\text{sarcasm}}$$
    where `pos_weight = 1.3835` (class ratio $N_{\text{real}}/N_{\text{fake}} = 8254/5966$).

---

### **THE DILEMMA & EMPIRICAL METRIC OBSERVED**

When evaluating our trained model on the unseen **FakeAVCeleb** benchmark, we observe two extreme, opposite evaluation behaviors depending on the evaluation setup:

#### **Scenario A: Offline Feature Evaluation (Pre-extracted Features from Base Un-tuned Backbones)**
* **Setup**: Evaluated cached `.pt` feature tensors extracted prior to Phase 2 backbone fine-tuning.
* **Observed Metrics**:
  * $TP = 3, \quad FP = 0, \quad FN = 905, \quad TN = 200$
  * **Accuracy ($\tau=0.50$)**: $18.32\%$
  * **Recall**: $0.0033$ (Predicts **EVERYTHING AS REAL**; 0 True Positives on Fakes).
  * **Score Range**: $P(\text{fake}) \le 0.001$ across all clips.

#### **Scenario B: End-to-End Evaluation on Imbalanced Test Set (900 Fake + 100 Real Clips)**
* **Setup**: End-to-end forward pass loading fine-tuned backbone weights into GPU VRAM.
* **Observed Metrics at Default Threshold $\tau = 0.50$**:
  * $TP = 900, \quad FP = 100, \quad FN = 0, \quad TN = 0$
  * **Accuracy ($\tau=0.50$)**: $90.00\%$
  * **Recall**: $1.0000$ (Predicts **EVERYTHING AS FAKE**; 0 True Negatives on Real clips).
  * **F1 Score**: $0.9474$.
* **Raw Probability Distribution Inspection**:
  * Real Clips Mean Score: $0.2817$ (Range: $0.2022 - 0.3242$, Max Real: $0.34$)
  * Fake Clips Mean Score: $0.3412 - 0.7870$ (Range: $0.35 - 0.7870$)
  * **AUC-ROC**: $0.5829 - 0.6500$.

---

### **QUESTIONS FOR INDEPENDENT ARCHITECTURAL REVIEW**

1. **Feature Distribution Shift**:
   Is evaluating fine-tuned Phase 2 classifier heads using offline `.pt` features extracted by *un-tuned base backbones* mathematically invalid due to feature representation shift? Why does it collapse output probabilities to near-zero ($P \approx 0.0001$)?

2. **Class Imbalance & Threshold Squeezing**:
   When $90\%$ of the test set is Fake (900 Fake vs. 100 Real), why does default threshold $\tau = 0.50$ result in $TN = 0$ despite Real scores ($0.2817$) being lower than Fake scores ($0.3412$)?

3. **Optimal Threshold Calibration Strategy**:
   Given that Real clips score $P \in [0.20, 0.32]$ and Fake clips score $P \in [0.35, 0.78]$, how should we apply **Youden's J Statistic** ($J = \text{TPR} + \text{TNR} - 1$) or **Bayesian Decision Theory** to set a calibrated operating threshold $\tau^* \approx 0.34$ that achieves high True Negatives ($TN > 90\%$) AND high True Positives ($TP > 90\%$)?

4. **Multi-Task Loss Balancing**:
   If auxiliary 6-class emotion losses ($\mathcal{L}_{\text{emo}}$) initially produced loss values of $\sim 1.80$, how does scaling auxiliary weights to $\lambda_a = 0.1, \lambda_b = 0.1, \lambda_{\text{sarc}} = 0.05$ ensure 90%+ gradient concentration on binary fake classification ($\mathcal{L}_{\text{BCE}}$) without losing multi-modal affect features?

5. **Preventing Score Compression**:
   What additional architectural techniques (e.g. Temperature Scaling, Focal Loss, or Cosine Similarity Head) can widen the margin between Real scores ($\sim 0.28$) and Fake scores ($\sim 0.34$)?
```
