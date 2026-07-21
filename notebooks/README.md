# Colab Workflow — Preprocessing (4-way) + Training

Move the slow steps off the RTX 4050 and onto free Colab T4s. **Preprocessing is split
across 4 people**; training runs on one machine after the shards are merged.

```
notebooks/
├── colab_preprocess_person1.ipynb   ← Person 1  (shard 0)
├── colab_preprocess_person2.ipynb   ← Person 2  (shard 1)
├── colab_preprocess_person3.ipynb   ← Person 3  (shard 2)
├── colab_preprocess_person4.ipynb   ← Person 4  (shard 3)
├── colab_training.ipynb             ← Leader (Phase 1 + Phase 2)
├── evaluation.ipynb                 ← FakeAVCeleb benchmark
└── colab_mosei_preprocess.ipynb     ← single-person fallback (whole 6,277)
```

Every notebook clones the **`feat/webapp-integration`** branch and needs a **T4 GPU**
runtime (Runtime → Change runtime type → T4 GPU).

---

## Shared Google Drive layout

One shared folder, same for everyone:

```
MyDrive/DeepSentinel_data/
├── segments.zip                    ← leader uploads once (6,277 MOSEI clips, 6.2 GB)
├── mosei_features_shard0.zip       ← Person 1 output
├── mosei_features_shard1.zip       ← Person 2 output
├── mosei_features_shard2.zip       ← Person 3 output
├── mosei_features_shard3.zip       ← Person 4 output
├── shard{0..3}_status.json         ← per-person completion markers
├── existing_features.zip           ← leader uploads (the 14k already-cached features)
├── metadata.zip                    ← leader uploads (track/meld/mosei/mustard CSVs)
├── (phase-2 zips: audio_cache, track1-3, meld, mustard …)   ← for Phase 2 only
└── checkpoints/                    ← training outputs land here
```

The leader creates the folder, uploads `segments.zip`, and **shares it (Editor) with all 4 members**.

---

## Order of operations

### 1. Preprocess — 4 people in parallel (~10–15 min each)
Each member opens **their** notebook (`colab_preprocess_personN.ipynb`), sets T4 GPU,
and **Runtime → Run all**. It:
- extracts `segments.zip`, clones the repo
- runs only its shard: `preprocess_all.py --num_shards 4 --shard N` (every 4th clip — no overlap)
- resume-safe: if Colab disconnects, just re-run — it skips what's already done
- uploads `mosei_features_shard{N}.zip` + a `shard{N}_status.json` to Drive

**Done when** all four `mosei_features_shard*.zip` and `shard*_status.json` exist on Drive.

### 2. Prep training data (leader, on the local PC)
The exact zip commands are in **`colab_training.ipynb`** (top cell). Create and upload to the
same Drive folder:
- **Phase 1:** `existing_features.zip`, `metadata.zip`
- **Phase 2 (optional):** `audio_cache.zip`, `transcripts.zip`, `track1-3_clips.zip`, `meld_real_clips.zip`, `mustard_clips.zip`

### 3. Train (leader) — `colab_training.ipynb`
- T4 GPU → Run all. It merges the 4 shard zips automatically, builds the speaker-stratified
  split, then runs **Phase 1** (cached features, ~15–30 min).
- **Phase 2** (backbone fine-tune) runs if the Phase-2 zips are present; on T4 it uses batch 2.
- Checkpoints (`best_phase1.pt`, `best_phase2.pt`) are copied to `Drive/…/checkpoints/`.

### 4. Evaluate — `evaluation.ipynb`
Benchmark the Phase-2 checkpoint on FakeAVCeleb (release gate: **AUC ≥ 0.70**).

---

## AU saliency — these notebooks are AU-OFF (deliberate)

The Colab notebooks score keyframes by **conf × sharpness** (AU-OFF), matching the existing
14k cached features. Do **not** turn AU on for MOSEI only — mixing AU-on and AU-off features
in one training set is invalid.

Real AU saliency (`pip install py-feat`) needs an **old numpy/torch stack** that conflicts with
Colab's pre-installed environment (the reason the local `.venv-feat` exists). So the AU-on run is
a **separate, local** job, not a Colab one:

```powershell
# in the isolated .venv-feat (already set up), re-preprocess EVERYTHING AU-on into a
# separate feature store so the AU-off baseline is preserved for the ablation
.\.venv-feat\Scripts\python.exe scripts/preprocess_all.py --device cuda --use_au --au_top_k 12
```
This is the Option-C ablation path (AU-on vs AU-off). It is slow (~seconds/crop) and must cover
**all** clips, not just MOSEI. Keep the AU-off features for the baseline comparison.

## Notes
- **Why sharded, not `--max_clips`:** the shard is a *stable, disjoint* partition (`clips[i::4]`),
  so restarts never re-assign work — no gaps, no double-processing.
- **Progress:** each run prints a tqdm bar, a per-source summary, and a `failed_clips.txt`; the
  `shard{N}_status.json` gives the leader an at-a-glance completion count per person.
- **⚠️ CBP retrain:** the detection layer changed (CBP normalization), so the old
  `best_phase1.pt` is stale — this run **retrains** it. Expected.
- Regenerate the 4 shard notebooks with `python notebooks/_gen_colab.py` (edit `NUM_SHARDS`
  or `BRANCH` there if needed).
```
