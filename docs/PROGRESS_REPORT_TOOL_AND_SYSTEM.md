# Progress in the Development of the Tool/System

The group has completed the dataset acquisition, video clip ingestion, and data preprocessing stages, and has fully implemented the training pipeline in code. The proposed deepfake detection architecture operates on the biological premise that authentic human communication maintains natural emotional coherence between voice and facial expressions, whereas deepfakes often introduce subtle cross-modal contradictions. To train and evaluate this detector, the group imported, preprocessed, and verified a primary training turnover pool of **17,741 multimodal clip pairs** across six core datasets, excluding external test sets (FakeAVCeleb), Track 4, and web application uploads.

The verified dataset inventory and its official **80% - 10% - 10% split** are structured as follows:

| Dataset / Source Pipeline | Type / Description | Total Verified Clips | Train Split (80% - Ready) |
|:---|:---|:---:|:---:|
| **Track 1** (Audio Swap: StyleTTS2 + RVC) | CREMA-D Synthetic Fake | 1,452 | 1,162 |
| **Track 2** (Lip Correction: Wav2Lip) | CREMA-D Synthetic Fake | 2,267 | 1,814 |
| **Track 3** (Full Face Synthesis: SadTalker) | CREMA-D Synthetic Fake | 3,722 | 2,978 |
| **MELD Real** (TV Multi-party Dialogues) | Authentic Real Benchmark | 3,334 | 2,667 |
| **CMU-MOSEI** (In-the-Wild YouTube Sentiment) | Authentic Real Benchmark | 6,276 | 5,021 |
| **MUStARD** (Multimodal Sarcasm & Irony) | Sarcasm Supervision / Tone | 690 | 552 |
| **TOTAL VERIFIED PREPROCESSED CLIPS** | **Core Training Turnover Pool** | **17,741** | **14,193 (80.0%)** |

> **Training Readiness Status:** Exactly **14,193 clips (80.0%)** are fully preprocessed, verified with non-empty audio-visual feature pairs, and **ready for model training**. The remaining **10.0% (1,774 clips)** and **10.0% (1,774 clips)** are partitioned strictly for validation and internal testing. External evaluation benchmarks (FakeAVCeleb), Track 4, and WebApp uploads are excluded from this training distribution to prevent data contamination.

The data preprocessing pipeline (`src/preprocessing/pipeline.py`) is now completely executed across these 17,741 clips. On the audio-linguistic stream, the pipeline extracts 16 kHz mono waveforms, derives acoustic representations using **Wav2Vec 2.0**, transcribes dialogue through OpenAI's **Whisper** model, and extracts contextual language embeddings using **BERT**. On the visual stream, the pipeline uses **InsightFace (RetinaFace)** to locate faces across video frames, track five facial landmark points, and score frame sharpness to select the clearest keyframes for facial expression analysis via a **Vision Transformer (ViT)**. Furthermore, the model training module (`src/training/trainer.py` and `scripts/train_full.py`) has been fully coded in the project pipeline, integrating Compact Bilinear Pooling, dual 6-class emotion prediction heads, the emotion discrepancy vector ($\Delta$), and a multi-task loss function combining classification and auxiliary emotion/sarcasm losses. With preprocessing finished and the training pipeline operational in code, the next immediate phase is executing full-scale model training runs and tuning decision thresholds.

On the frontend side, the web application (DeepSentinel) has been substantially revamped to serve as the user-facing evaluation tool. The interface supports video uploads of up to 10 minutes and provides an interactive filmstrip trimmer allowing users to scrub and adjust crop windows between 3 and 20 seconds. During evaluation, the interface connects to the backend via Server-Sent Events (SSE) to display a real-time face tracking HUD with RetinaFace bounding boxes and landmark points, live streaming speech transcription, and dynamic elapsed and estimated remaining timers across all pipeline steps.

---

## Suggested Figures to Include

```
+------------------------------------------------------------------------------------------+
|                                                                                          |
|                                  [ FIGURE 1 ]                                            |
|                 Terminal Execution: Preprocessed Dataset Split Table                    |
|             (Showing 17,741 Total Clips and 14,193 Clips Ready for Training)             |
|                                                                                          |
+------------------------------------------------------------------------------------------+
Figure 1. Preprocessed Dataset Inventory and Official 80-10-10 Training Split Verification

+------------------------------------------------------------------------------------------+
|                                                                                          |
|                                  [ FIGURE 2 ]                                            |
|                       Web Application Video Analysis Interface                           |
|             (Showing Live RetinaFace HUD, Streaming Transcript, & Timers)                |
|                                                                                          |
+------------------------------------------------------------------------------------------+
Figure 2. Web Application Video Analysis Interface
```

---

## Instructions on Which Screenshots to Share

### 📸 Figure 1: Preprocessed Dataset Split & 80% Readiness Table
* **What to capture:** Your terminal output running the dataset split command.
* **Exact command to run in your PowerShell terminal:**
  ```powershell
  python scripts/print_training_summary.py
  ```
* **What it outputs (matches the reference screenshot format):**
  * The formatted table displaying Track 1, Track 2, Track 3, MELD Real, CMU-MOSEI, and MUStARD.
  * The Total Verified Preprocessed Clips: **17,741**.
  * The **80% - 10% - 10%** Official Dataset Split Matrix.
  * The highlighted status: `>>> STATUS: 14,193 CLIPS (80.0%) FULLY PREPROCESSED & READY FOR TRAINING <<<`.
* **Caption:** `Figure 1. Preprocessed Dataset Inventory and Official 80-10-10 Training Split Verification`

---

### 📸 Figure 2: Web Application Video Analysis Interface
* **What to capture:** Your web browser at `http://localhost:8000` showing the video evaluation screen (the `/analyzing` view).
* **Key visual elements to show:**
  1. **Left side (Video HUD):** The video playing with the cyan **RetinaFace bounding brackets** and facial landmark points overlaid on the face, along with the top-right stopwatch badge (`⏱ 00:03.4 / ~00:07.5`).
  2. **Lower left:** The **Live Speech & Acoustic Stream** card displaying Whisper's live typed-out transcript text.
  3. **Right side:** The **Analyzing video** card displaying the 3-column timing bar (`TIME ELAPSED`, `EST. RUNTIME`, `EST. REMAINING`) and the checklist steps with green checks.
* **Caption:** `Figure 2. Web Application Video Analysis Interface`
