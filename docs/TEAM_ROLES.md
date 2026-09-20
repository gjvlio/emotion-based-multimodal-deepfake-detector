# DeepSentinel — Team Roles & Accountability

> **Purpose.** Kill the single-point-of-failure. No one person holds all the context or
> becomes the bottleneck. Each member **owns a territory** of the project end-to-end and is
> **fully accountable** for it — permanently, not on rotation.

---

## The operating principle: territorial leadership

**Leadership is permanent per territory. Only the artifacts move between territories — never
the authority.**

- Each lead owns their domain **for the whole project**. Kina is *always* the preprocessing
  authority; JC is *always* the training authority; Matan is *always* the evaluation
  authority; Gel is *always* the data/QA authority. This never hands off.
- What passes down the pipeline is the **work product** — raw clips → features → checkpoints
  → results — not the leadership. A lead does not "take over" from the previous lead; they
  **receive an artifact** and run their own territory on it.
- A lead is accountable for their territory **at all times**, even when it isn't the active
  stage. "Not the current stage" means *preparing, maintaining, documenting, and being ready*
  — never idle, never waiting to be told.
- Within their territory, a lead **directs all four members** (including the other leads,
  who act as workers on that lead's tasks). Outside their territory, they are a worker taking
  direction from that territory's lead.

**Accountability model (RACI, simplified):**
- The territory **Lead is Accountable** — the buck stops with them for that domain. If
  preprocessing is wrong, that's on Kina, regardless of who ran the cell.
- The other three are **Responsible** for the tasks their current lead assigns them.
- Gel is **Consulted/Informed** on every territory (QA + integration) but does not override a
  lead inside their own domain except at the QA gate (see below).

---

## Artifact flow (what moves, and the acceptance gate at each boundary)

```
   GEL (Data/QA)                KINA (Preprocess)        JC (Training)          MATAN (Eval)
   ─────────────                ─────────────────        ─────────────         ────────────
   raw clips + manifests  ──▶   z_at / z_v feature  ──▶  valid checkpoints ──▶  results:
   (clean, verified,            cache (complete,         (DeepSentinel        AUC, p-values,
    no leakage, labeled)        AU-off, correct)          Phase 1+2 AND         scorecard,
                                                          ACE-Net on our        sarcasm acc,
        ▲                                                 data)                 Bootstrap CI
        │                                                                          │
        └──────────────────────  QA / integration audits every boundary  ◀────────┘
```

Each arrow is a **handoff contract** with a definition-of-done. An artifact is not "handed
off" until the receiving lead confirms it meets the contract (see each role's *Delivers*
section). If it fails, it goes back to the sending lead — that's the accountability loop.

---

## GEL — Data Manager / QA / Integration

**Mission:** guarantee that the data is correct and that every territory's output integrates
into one coherent, truthful project.

**Fully accountable for:**
- **Data integrity** — every raw clip is valid (video + audio stream present, not corrupt),
  every manifest matches what's actually on disk, labels are correct, and there is **no data
  leakage** (no speaker/clip in both train and test; FakeAVCeleb never enters training).
- **The single source of truth** — `docs/PROJECT_CONTEXT_MASTER.md` stays current. When any
  lead makes a decision, Gel ensures it lands in the master doc.
- **QA gate at every boundary** — audits Kina's feature cache, JC's checkpoints, and Matan's
  result numbers before they're treated as final or put in the manuscript.
- **Code ↔ manuscript consistency** — the numbers and claims in the paper match the code and
  the real results (no stale/aspirational numbers cited).
- **Git hygiene** — branch discipline (`feat/webapp-integration` only unless told otherwise),
  commit conventions, no accidental leaks of confidential/large files.

**Owns (files/artifacts):** `data/` manifests, `configs/default.yaml`,
`docs/PROJECT_CONTEXT_MASTER.md`, the git repo state, the QA checklist.

**Directs the team in:** data validation sweeps (e.g., everyone validates a slice of clips),
manifest rebuilds, the master-doc updates.

**Delivers to Kina (definition of done):** clean raw clips + a manifest (`clip_id`,
`video_path`, labels) where every listed clip exists on disk, has audio+video, and carries a
correct label. No missing files, no duplicates.

**Maintains master-doc sections:** §7 (datasets), §12 (repo map), §13 (current state), §22
(stale-doc traps), §18 (manuscript status).

---

## KINA — Preprocessing Lead

**Mission:** turn every raw clip that needs it into a valid, consistent feature cache
(`z_at`, `z_v`), completely and correctly.

**Fully accountable for:**
- **Cache completeness** — every clip in the training manifests has both `z_at` (1536) and
  `z_v` (768) cached, or is explicitly logged as a documented failure.
- **Cache consistency** — the entire cache is **AU-OFF** (`conf × sharpness` scoring). Mixing
  AU-on and AU-off features in one training set is invalid. If AU-on is ever run (Option C),
  it goes into a *separate* store — never mixed. (See master doc §10.)
- **The Colab 4-way sharding** — the parallel preprocessing runs cleanly, shards are
  disjoint and complete (`clips[N::4]`, resume-safe), and the four shard zips merge correctly.
- **Preprocessing correctness** — the pipeline matches the spec: 25 fps, optical-flow gate
  (0.3), insightface detection (0.7), top-8 keyframes, Whisper→BERT + Wav2Vec2 mean-pool.

**Owns (files/artifacts):** `src/preprocessing/` (audio.py, visual.py, filters.py,
pipeline.py), `scripts/preprocess_all.py`, `notebooks/_gen_colab.py`,
`notebooks/colab_preprocess_person{1..4}.ipynb`, the `data/preprocessed/features/` cache.

**Directs the team in:** the parallel Colab preprocessing — **this is the one genuinely
parallel stage.** Kina assigns each of the 4 members a shard, tracks the `shard{N}_status.json`
markers, and confirms all four zips land on Drive. Also directs the **second preprocessing
pass** for ACE-Net's own feature format if JC needs it sharded.

**Receives from Gel:** clean raw clips + manifests (per the contract above). If a manifest
points to a missing/corrupt clip, it goes back to Gel — not Kina's problem to fix silently.

**Delivers to JC (definition of done):** a complete feature cache (all manifest clips have
`z_at` + `z_v`), verified AU-off, with a failure log for any clip that couldn't be processed.
Shard zips merged and integrity-checked.

**Maintains master-doc sections:** §9 (preprocessing), §10 (AU saliency), §16.2 (sharded
preprocessing).

---

## JC — Training Lead

**Mission:** produce valid, non-stale, trained checkpoints for **both frameworks** (our
DeepSentinel and the ACE-Net baseline), **both trained on our dataset**.

**Fully accountable for:**
- **DeepSentinel training** — Phase 1 (mandatory CBP-fix retrain; the old checkpoint is stale
  and invalid — master doc §11) **and** Phase 2 end-to-end fine-tune (never yet run; this is
  what buys cross-dataset generalization). Convergence, hyperparameters, checkpoint validity.
- **ACE-Net baseline training on our data** — the "Stage 0" adaptation (making the
  `Baseline_Training` repo's data loaders consume our manifests / 4-track fakes / our labels),
  then training it on our dataset so the comparison is controlled. **This is the biggest
  schedule risk — JC owns time-boxing a small spike first** (train on a tiny slice) before
  committing to the full run.
- **Checkpoint hygiene** — no stale checkpoints get passed downstream; each is tagged with
  what data + code version produced it.

**Owns (files/artifacts):** `src/training/` (trainer.py, losses.py, dataset.py),
`src/models/` (detection_model.py, bilinear.py, etc.), `scripts/train_full.py`,
`scripts/train_smoke.py`, `notebooks/colab_training.ipynb`, `checkpoints/`, **and the
`Baseline_Training` (ACE-Net) repo**.

**Directs the team in:** sharding the ACE-Net preprocessing pass (its own feature format),
monitoring long Colab training runs (checkpoint/resume across session limits), and the
two-framework training coordination.

**Receives from Kina:** the complete DeepSentinel feature cache (per contract). For ACE-Net,
receives raw clips + manifests (its preprocessing differs — coordinate with Kina on sharding).

**Delivers to Matan (definition of done):** a valid **DeepSentinel Phase-2 checkpoint** and a
valid **ACE-Net checkpoint trained on our data**, each with its training config recorded
(data used, epochs, val loss), and each confirmed to load and run a forward pass. No stale
checkpoints.

**Maintains master-doc sections:** §6 (loss + two-phase training), §6.1 (backprop), §16.3
(training), §11 (CBP fix), §8 (generation, shared with Gel where it feeds training data).

---

## MATAN — Evaluation Lead

**Mission:** produce every result number, prove it's valid, and deliver the statistics the
panel requires (a real p-value, not just a CI).

**Fully accountable for:**
- **The FakeAVCeleb benchmark** — run both frameworks on the **identical** held-out test
  sample; AUC, Accuracy, Precision, Recall, F1, per-method breakdown, Bootstrap 95% CI.
- **The significance suite (Hypothesis 1)** — **DeLong's test** (AUC p-value) + **paired
  bootstrap scorecard** (Acc/Prec/Recall/F1 p-values), comparing DeepSentinel vs ACE-Net on
  the same clips. Confirm the pairing is valid (same clips, same labels — the scripts abort
  on mismatch; Matan verifies they didn't). (Master doc §14.)
- **Sarcasm evaluation (RQ4)** — the first-ever sarcasm accuracy on the held-out MUStARD
  split, at threshold 0.5. (Master doc §5, §15.)
- **The LR baseline + Δ-ablation** — the controlled internal comparisons that prove the
  architecture and the emotion-mismatch signal contribute.
- **Number validity** — no number leaves Matan's territory unless it came from a valid
  (retrained, post-CBP-fix) checkpoint. Matan is the last line against citing dead numbers
  (e.g., the invalid 0.50 AUC).

**Owns (files/artifacts):** `src/evaluation/` (metrics.py, ablation.py, ood_eval.py,
significance.py), `scripts/evaluate.py`, `scripts/evaluate_fakeavceleb.py`,
`scripts/evaluate_sarcasm.py`, `scripts/eval_acenet_baseline.py`,
`scripts/compare_frameworks.py`, `scripts/baseline_logreg.py`,
`notebooks/evaluation.ipynb`, the `results/` directory (currently empty — Matan creates it).

**Directs the team in:** the evaluation runs (parallelize preprocessing of the FakeAVCeleb
test sample if needed), result-table construction, and sanity-checking numbers before they
reach the manuscript.

**Receives from JC:** valid checkpoints for both frameworks (per contract). A stale or
unrecorded checkpoint gets rejected back to JC.

**Delivers to Gel/manuscript (definition of done):** a results package — every metric, its
p-value/CI, the scorecard, the sarcasm number — each traceable to the checkpoint + test
sample that produced it, saved to disk (`results/*.csv`/`*.json`), not just printed.

**Maintains master-doc sections:** §14 (significance), §15 (evaluation scripts), and the
results portions of §13.

---

## Shared rules (all four)

1. **Territorial, not relayed.** Your leadership is permanent. You don't hand off authority —
   you hand off an artifact and confirm the receiver accepts it.
2. **Own your master-doc sections.** The context lives in `PROJECT_CONTEXT_MASTER.md`, not in
   one person's head. If your territory changes, you update your sections — that same day.
3. **Definition-of-done at every boundary.** Don't accept a broken artifact to be "nice."
   Reject it back to the owning lead. That's how accountability stays real.
4. **Blocked ≠ idle.** When your territory isn't the active stage, you prepare, document, and
   ready your tooling so that when the artifact arrives you run immediately.
5. **Branch = `feat/webapp-integration` only.** New branch per feature; `feat/fix/exp/ci/docs/chore`
   prefixes. No syncing `main` without explicit team agreement.
6. **Tiebreaker.** Inside a territory, that lead decides. Across territories (a decision that
   spans domains, or two leads disagree), **Gel decides** as data/QA/integration owner.
7. **Cadence.** Short sync at each handoff (artifact + "does it meet the contract?"). Don't
   wait for a weekly meeting to flag a blocker — raise it when it happens.

---

## Reality check: where the pipeline is parallel vs sequential

Territorial ownership is permanent, but the *work* still has a shape — know it so nobody
sits blocked:

- **Genuinely parallel (all 4 work at once):** preprocessing (Kina's 4-way Colab sharding);
  the ACE-Net preprocessing pass; FakeAVCeleb test-sample preprocessing; data validation
  sweeps. Kina and Matan will pull the whole team into their territory during these windows.
- **Sequential (one machine, one owner drives, others prep/assist):** training runs (JC),
  the significance computation (Matan). Here the other leads aren't idle — they're preparing
  their next parallel window or maintaining docs/tooling.
- **Continuous (all the time):** Gel's QA + integration + master-doc upkeep runs across every
  stage.

**The point:** four permanent territories, each fully owned and accountable, with the team
flowing into whichever territory is currently parallelizable — so no one is a bottleneck and
no one is idle.

---

# Appendix: Input → Process → Output (exact, per territory)

This is the streamlined contract. Each territory takes **exact input files**, runs an **exact
process**, and turns over **exact output files**. A handoff is complete only when the next
lead's *Acceptance check* passes. Paths are relative to the repo root
(`D:/Documents/Programming/Thesis_G10`) unless noted.

> ⚠️ **Save vs print.** Some scripts **write a file**; others only **print to the terminal**.
> Where a script only prints, the owner must **redirect stdout to a file** (shown with `| tee`
> / `>`), or the result is lost and the handoff fails.

---

## ① GEL — Data/QA  →  turns over to KINA

**INPUT (required, must exist):**
```
data/raw/CREMA-D/                         raw actor clips (fake-generation source)
data/raw/MELD/MELD-RAW/MELD.Raw/          raw TV clips
data/raw/CMU-MOSEI/videos/                downloaded YouTube videos
data/raw/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd
data/raw/MUStARD/raw_data/utterances_final/   690 MP4s
data/raw/MUStARD/repo/data/sarcasm_data.json  690 labels
data/synthetic/track{1,2,3,4}_fakes/videos/   generated fakes
```

**PROCESS (exact commands):**
```powershell
# 1. Segment CMU-MOSEI into 2-8s clips
python scripts/segment_cmumosei.py --min_dur 2.0 --max_dur 8.0 --workers 4

# 2. Rebuild the MOSEI manifest from the segments actually on disk
python -c "from pathlib import Path; import pandas as pd; segs=sorted(Path('data/raw/CMU-MOSEI/segments').glob('*.mp4')); pd.DataFrame([{'clip_id':v.stem,'video_path':str(v.resolve())} for v in segs]).to_csv('data/processed/mosei_manifests/mosei_real.csv', index=False); print(len(segs),'rows')"

# 3. Build MELD manifests (real half + fake-source mismatch pairs)
python scripts/sample_meld.py --meld_dir data/raw/MELD/MELD-RAW/MELD.Raw --out_dir data/processed/meld_manifests
python scripts/sample_meld_mismatch.py

# 4. Validate integrity (no corrupt/missing/streamless clips)
python scripts/validate_cmumosei_videos.py
python scripts/validate_generation.py
```

**OUTPUT (turn over to Kina — these exact files):**
```
data/processed/mosei_manifests/mosei_real.csv        (clip_id, video_path)
data/processed/meld_manifests/meld_real.csv          (3,334 real)
data/processed/meld_manifests/meld_mismatch_pairs.csv (3,482 Track-4 pairs)
data/synthetic/track{1,2,3,4}_fakes/metadata.csv     (per-track fake labels)
data/raw/MUStARD/repo/data/sarcasm_data.json         (sarcasm labels)
```

**Kina's acceptance check:** every `clip_id` in every manifest resolves to a real file on
disk that has both a video and an audio stream, and carries a valid label. If any row fails →
reject back to Gel.

---

## ② KINA — Preprocessing  →  turns over to JC

**INPUT (required, from Gel — the manifests above):** the 5 manifest/label files listed in ①'s
OUTPUT, plus the raw clips they point to.

**PROCESS (exact commands):**

*Local (small / catch-up runs):*
```powershell
python scripts/preprocess_all.py --device cuda          # AU-OFF by default. Resume-safe.
```

*Colab (the 6,277 MOSEI clips, 4-way parallel — the main run):*
```
Each member opens notebooks/colab_preprocess_person{1..4}.ipynb → T4 GPU → Run all.
It runs:  python scripts/preprocess_all.py --device cuda --num_shards 4 --shard N
Leader first uploads segments.zip to MyDrive/DeepSentinel_data/ and shares the folder.
```

**OUTPUT (turn over to JC — these exact files):**
```
data/preprocessed/features/z_at/{clip_id}.pt     (1536,) per clip
data/preprocessed/features/z_v/{clip_id}.pt      (768,)  per clip
data/preprocessed/failed_clips.txt               logged failures (if any)
# From Colab, before merge:
MyDrive/DeepSentinel_data/mosei_features_shard{0..3}.zip
MyDrive/DeepSentinel_data/shard{0..3}_status.json
```

**JC's acceptance check:** count of `z_at` == count of `z_v`, and every non-failed manifest
clip has both tensors. All features AU-off (no mixed store). Shard zips merge without
collisions. If incomplete/inconsistent → reject back to Kina.

---

## ③ JC — Training  →  turns over to MATAN

**INPUT (required):**
- *DeepSentinel:* `data/preprocessed/features/z_at/` + `z_v/` (from Kina) + `configs/default.yaml`.
- *ACE-Net:* raw clips + manifests (its preprocessing differs; coordinate a sharded pass with Kina).

**PROCESS (exact commands):**

*DeepSentinel — Phase 1 (mandatory CBP-fix retrain) + Phase 2:*
```powershell
python scripts/train_full.py --device cuda --no_track4 --patience 5 `
  --phase2_epochs 3 --phase2_batch 1 --phase2_freeze_layers 2 --no_grad_ckpt
# Phase-1-only (fast check):  python scripts/train_full.py --device cuda --no_track4 --no_phase2 --patience 10
```

*ACE-Net baseline on our data (in the separate `Baseline_Training` repo):*
```
Stage 0 (JC owns): adapt Baseline_Training/src/data loaders to read OUR manifests +
                   map our 4-track fakes -> its "fake" class. Time-box a tiny-slice spike first.
Then train Stage 1 (2 emotion classifiers) + Stage 2 (discriminator) on our dataset.
```

**OUTPUT (turn over to Matan — these exact files):**
```
checkpoints/full/best_phase1.pt      DeepSentinel Phase 1 (post-CBP-fix — NOT the old stale one)
checkpoints/full/best_phase2.pt      DeepSentinel Phase 2 (the one to benchmark)
<Baseline_Training>/checkpoints/stage2_acenet.pt   ACE-Net, trained on OUR data
+ a one-line note per checkpoint: data used, epochs, val_loss
```

**Matan's acceptance check:** each checkpoint loads (`torch.load` + `load_state_dict`) and runs
a forward pass without error; each is confirmed **post-CBP-fix** (not the stale pre-fix
checkpoint) and **trained on our data**. If stale/unrecorded → reject back to JC.

---

## ④ MATAN — Evaluation  →  turns over to GEL / manuscript

**INPUT (required, from JC):** `checkpoints/full/best_phase2.pt`,
`<Baseline_Training>/checkpoints/stage2_acenet.pt`, and the FakeAVCeleb test set at
`data/raw/FakeAVCeleb_v1.2/`.

**PROCESS (exact commands — note which SAVE vs PRINT):**
```powershell
# A. DeepSentinel on FakeAVCeleb  -> SAVES a per-clip CSV
python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/full/best_phase2.pt `
  --n_real 200 --n_fake 800 --seed 42 --save_csv results/fakeavceleb_deepsentinel.csv

# B. ACE-Net on the SAME sample  -> SAVES a per-clip CSV (match --n_real/--n_fake/--seed!)
python scripts/eval_acenet_baseline.py --n_real 200 --n_fake 800 --seed 42 `
  --save_csv results/acenet_fakeavceleb.csv

# C. Battle of frameworks: DeLong + paired-bootstrap  -> PRINTS ONLY, so redirect to a file
python scripts/compare_frameworks.py --a results/fakeavceleb_deepsentinel.csv `
  --b results/acenet_fakeavceleb.csv --name_a DeepSentinel --name_b ACE-Net `
  | tee results/scorecard.txt

# D. Sarcasm eval (RQ4)  -> PRINTS ONLY, redirect
python scripts/evaluate_sarcasm.py --checkpoint checkpoints/full/best_phase2.pt `
  | tee results/sarcasm.txt

# E. LR baseline (controlled internal comparison)  -> SAVES JSON
python scripts/baseline_logreg.py --out results/baseline_logreg.json

# F. RQ1/RQ2 metrics + Δ-ablation  -> PRINTS ONLY, redirect
python scripts/evaluate.py --checkpoint checkpoints/full/best_phase2.pt --ablation `
  | tee results/rq1_rq2_ablation.txt
```

**OUTPUT (turn over to Gel / manuscript — create the `results/` dir; these exact files):**
```
results/fakeavceleb_deepsentinel.csv    per-clip scores (DeepSentinel)
results/acenet_fakeavceleb.csv          per-clip scores (ACE-Net)
results/scorecard.txt                   DeLong p-value + Acc/Prec/Recall/F1/AUC scorecard  ← H1
results/sarcasm.txt                     MUStARD sarcasm Acc/Prec/Recall/F1/AUC @0.5        ← RQ4
results/baseline_logreg.json            LR baseline metrics
results/rq1_rq2_ablation.txt            emotion-head accuracy + Δ-ablation                 ← RQ1/RQ2
```

**Gel's acceptance check:** every number is traceable to (checkpoint + test sample) that
produced it; the two CSVs share the same `clip_id`s (paired); no number came from a stale
checkpoint. Then Gel integrates these into the manuscript result tables.

---

## One-screen summary

| Territory | INPUT (from) | PROCESS (key script) | OUTPUT (to) |
|---|---|---|---|
| **Gel** — Data/QA | raw datasets + generated fakes | `segment_cmumosei` · `sample_meld*` · validators | manifests + labels → **Kina** |
| **Kina** — Preprocess | manifests + raw clips | `preprocess_all.py` (+ 4 Colab notebooks) | `z_at/` + `z_v/` cache → **JC** |
| **JC** — Training | feature cache (+ raw for ACE-Net) | `train_full.py` (P1+P2) · ACE-Net repo | `best_phase2.pt` + ACE-Net ckpt → **Matan** |
| **Matan** — Eval | both checkpoints + FakeAVCeleb | `evaluate_fakeavceleb` · `eval_acenet_baseline` · `compare_frameworks` · `evaluate_sarcasm` · `baseline_logreg` | `results/*` → **Gel/manuscript** |

**The chain in one line:** raw clips **(Gel)** → features **(Kina)** → checkpoints **(JC)** →
result files **(Matan)** → manuscript **(Gel)**. Each arrow only completes when the receiver's
acceptance check passes.
