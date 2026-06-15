# DeepSentinel Web Service (backend scaffolding)

FastAPI backend that serves the trained DeepSentinel deepfake detector and
**auto-equips the latest training checkpoint**. No frontend yet — this layer
exists to prove the model is wired and served correctly. UI/design comes later.

## How auto-equip works

```
trainer writes  checkpoints/full/best_phase2.pt   (or best_phase1.pt)
        │
        ▼
ModelService notices the file's (mtime, size) changed
        │
        ▼
reloads weights into the live model — next request uses the new model
```

- Checked on **every request** and by a **background watcher** every
  `DEEPSENTINEL_WATCH_INTERVAL_SEC` seconds (default 15) so idle servers stay current.
- Checkpoint preference order: `best_phase2.pt` → `best_phase1.pt`
  (Phase 2 served automatically once it exists).

## Run

```bash
# deps (project ML deps already installed in .venv)
pip install -r webapp/requirements.txt

# start (from repo root)
uvicorn webapp.main:app --port 8000
#   GPU:  DEEPSENTINEL_DEVICE=cuda uvicorn webapp.main:app --port 8000
# or:  python -m webapp
```

Interactive docs at <http://localhost:8000/docs>.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | service status + equipped checkpoint |
| GET  | `/model/info` | current checkpoint metadata (phase, epoch, val_loss, mtime) |
| POST | `/model/reload` | force a checkpoint re-check (normally automatic) |
| POST | `/detect` | upload a video → real/fake verdict + emotion evidence |

### `POST /detect` example

```bash
curl -F "file=@clip.mp4" http://localhost:8000/detect
```

```jsonc
{
  "verdict": "FAKE",
  "p_fake": 0.94,
  "threshold": 0.5,
  "audio_text_emotion": { "label": "happy", "confidence": 0.72, "distribution": {…} },
  "visual_emotion":     { "label": "angry", "confidence": 0.81, "distribution": {…} },
  "emotion_mismatch":   { "happy": 0.64, "angry": 0.68, … },
  "p_sarcasm": 0.12,
  "transcript": "…",
  "served_by": { "checkpoint": "best_phase1.pt", "phase": 1, "val_loss": 0.0372, … }
}
```

## Inference path & the Phase-2 caveat

A web upload is a **raw video**, so the service runs the project's real
`PreprocessingPipeline` (raw video → Z_at/Z_v) and calls
`detector.forward_from_features()` — the same path used in Phase-1 training.

- **Phase 1 checkpoints** (backbones frozen, never trained): feature-path
  inference is **exactly correct**.
- **Phase 2 checkpoints** (backbones fine-tuned): features must come from the
  *fine-tuned* backbones via end-to-end `forward()`. That seam is stubbed at
  `ModelService._predict_e2e` and the response carries a `note`. **Wire it before
  serving Phase 2 publicly** or feature/weight mismatch will degrade accuracy.

> ⚠️ The current on-disk model is **Phase 1 only** → FakeAVCeleb AUC ≈ 0.50
> (random on unseen fakes). The backend is correct and will auto-equip Phase 2
> the moment training produces `best_phase2.pt`, but **do not ship publicly until
> a Phase-2 checkpoint clears AUC ≥ 0.70** on the unseen benchmark.

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `DEEPSENTINEL_DEVICE` | `cpu` | `cpu` or `cuda` |
| `DEEPSENTINEL_CHECKPOINT_DIR` | `checkpoints/full` | where the trainer writes checkpoints |
| `DEEPSENTINEL_WATCH_INTERVAL_SEC` | `15` | idle re-check cadence |
| `DEEPSENTINEL_UPLOAD_DIR` | `webapp/uploads` | uploaded videos (no auto-delete) |
| `DEEPSENTINEL_PREPROCESS_CACHE_DIR` | `data/preprocessed` | feature cache |

Uploaded files are **not** auto-deleted — manage `webapp/uploads/` yourself.
