"""
smoke_test_pipeline.py — End-to-end pipeline smoke test on a small sample.

Samples N clips per source, preprocesses them, runs Phase 1 training for
2 epochs, then evaluates. Verifies the full pipeline is wired correctly
before committing to a full run.

Usage (from repo root):
    python scripts/smoke_test_pipeline.py --device cuda
    python scripts/smoke_test_pipeline.py --device cpu --n 5
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.detection_model import DeepfakeDetector
from src.preprocessing.pipeline import PreprocessingPipeline
from src.training.dataset import DeepfakeDataset
from src.training.trainer import Trainer
from src.evaluation.metrics import DetectionMetrics
from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"


def _sample_clips(csv_path: str, id_col: str, path_col: str, n: int) -> list[tuple[str, str]]:
    p = Path(csv_path)
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[df["status"] == "done"]
    df = df.dropna(subset=[id_col, path_col])
    sample = df.sample(min(n, len(df)), random_state=42)
    return [(str(row[id_col]), str(row[path_col])) for _, row in sample.iterrows()]


def stage_preprocess(cfg: Config, clips: list[tuple[str, str]], device: str, force: bool) -> int:
    log.info(f"=== STAGE 1: Preprocessing {len(clips)} clips ===")
    pipeline = PreprocessingPipeline(
        cache_dir     = cfg.paths.preprocessed_dir,
        wav2vec_model = cfg.model.wav2vec_model,
        bert_model    = cfg.model.bert_model,
        whisper_model = cfg.model.whisper_model,
        vit_model     = cfg.model.vit_model,
        face_detector = cfg.preprocessing.face_detector,
        n_keyframes   = cfg.preprocessing.n_keyframes,
        frame_size    = cfg.preprocessing.frame_size,
        max_audio_sec = cfg.preprocessing.max_audio_seconds,
        device        = device,
    )

    ok = 0
    fail = 0
    for i, (cid, vp) in enumerate(clips, 1):
        if pipeline.is_cached(cid) and not force:
            log.info(f"  [{i}/{len(clips)}] {cid} — cached, skip")
            ok += 1
            continue
        t0 = time.time()
        result = pipeline.process(cid, vp, force=force)
        elapsed = time.time() - t0
        if result is not None:
            log.info(f"  [{i}/{len(clips)}] {cid} — OK ({elapsed:.1f}s)")
            ok += 1
        else:
            log.warning(f"  [{i}/{len(clips)}] {cid} — FAILED")
            fail += 1

    status = PASS if fail == 0 else FAIL
    log.info(f"  {status} Preprocessing: {ok} OK, {fail} failed")
    return ok


def stage_train(cfg: Config, device: str) -> str | None:
    log.info("=== STAGE 2: Phase 1 Training (2 epochs) ===")
    try:
        train_ds, val_ds, _ = DeepfakeDataset.stratified_split(
            preprocessed_dir = cfg.paths.preprocessed_dir,
            train_ratio      = cfg.training.train_ratio,
            val_ratio        = cfg.training.val_ratio,
            track1_meta      = cfg.paths.track1_meta,
            track2_meta      = cfg.paths.track2_meta,
            track3_meta      = cfg.paths.track3_meta,
            track4_meta      = cfg.paths.track4_meta,
            meld_real_csv    = cfg.paths.meld_real_csv,
            mosei_real_csv   = cfg.paths.mosei_real_csv,
        )
        log.info(f"  Dataset split — train:{len(train_ds)}  val:{len(val_ds)}")

        if len(train_ds) < 2 or len(val_ds) < 1:
            log.error(f"  {FAIL} Too few cached clips for training. Run with larger --n.")
            return None

        train_loader = DataLoader(train_ds, batch_size=min(8, len(train_ds)), shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=min(8, len(val_ds)),   shuffle=False, num_workers=0)

        model = DeepfakeDetector(
            wav2vec_model = cfg.model.wav2vec_model,
            bert_model    = cfg.model.bert_model,
            vit_model     = cfg.model.vit_model,
            n_emotions    = cfg.model.n_emotions,
            proj_dim      = cfg.model.proj_dim,
            dropout_heads = cfg.model.dropout_heads,
            dropout_cls   = cfg.model.dropout_classifier,
        )

        ckpt_dir = Path(cfg.paths.checkpoints_dir) / "smoke"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model          = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            checkpoint_dir = str(ckpt_dir),
            log_dir        = cfg.paths.logs_dir,
            fp16           = cfg.training.fp16 and device == "cuda",
            lambda_a       = cfg.training.lambda_a,
            lambda_b       = cfg.training.lambda_b,
            device         = device,
        )

        p = cfg.training.phase1
        trainer.train_phase1(
            lr           = p.lr,
            weight_decay = p.weight_decay,
            max_epochs   = 2,
            patience     = 2,
        )

        ckpt_path = str(ckpt_dir / "best_phase1.pt")
        log.info(f"  {PASS} Training complete. Checkpoint: {ckpt_path}")
        return ckpt_path

    except Exception as e:
        log.error(f"  {FAIL} Training failed: {e}", exc_info=True)
        return None


@torch.no_grad()
def stage_evaluate(cfg: Config, ckpt_path: str, device: str) -> bool:
    log.info("=== STAGE 3: Evaluation ===")
    try:
        model = DeepfakeDetector(
            wav2vec_model = cfg.model.wav2vec_model,
            bert_model    = cfg.model.bert_model,
            vit_model     = cfg.model.vit_model,
            n_emotions    = cfg.model.n_emotions,
            proj_dim      = cfg.model.proj_dim,
            dropout_heads = cfg.model.dropout_heads,
            dropout_cls   = cfg.model.dropout_classifier,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"], strict=False)
        model = model.to(device)
        model.eval()

        _, _, test_ds = DeepfakeDataset.stratified_split(
            preprocessed_dir = cfg.paths.preprocessed_dir,
            train_ratio      = cfg.training.train_ratio,
            val_ratio        = cfg.training.val_ratio,
            track1_meta      = cfg.paths.track1_meta,
            track2_meta      = cfg.paths.track2_meta,
            track3_meta      = cfg.paths.track3_meta,
            track4_meta      = cfg.paths.track4_meta,
            meld_real_csv    = cfg.paths.meld_real_csv,
            mosei_real_csv   = cfg.paths.mosei_real_csv,
        )

        if len(test_ds) == 0:
            log.warning(f"  {SKIP} No test clips available (too few total clips). Skipping eval.")
            return True

        test_loader = DataLoader(test_ds, batch_size=min(8, len(test_ds)), shuffle=False, num_workers=0)

        all_logits, all_labels, all_deltas, all_pipelines = [], [], [], []
        for batch in test_loader:
            z_at = batch["z_at"].to(device)
            z_v  = batch["z_v"].to(device)
            out  = model.forward_from_features(z_at, z_v)
            all_logits.append(out.logit.cpu())
            all_labels.append(batch["fake_label"])
            delta_norm = torch.abs(
                F.softmax(out.emotion_a.cpu(), dim=-1) -
                F.softmax(out.emotion_b.cpu(), dim=-1)
            ).norm(dim=-1)
            all_deltas.append(delta_norm)
            all_pipelines.extend(batch["source_pipeline"])

        logits   = torch.cat(all_logits)
        labels   = torch.cat(all_labels)
        deltas   = torch.cat(all_deltas)

        metrics  = DetectionMetrics(threshold=0.5)
        results  = metrics.evaluate(logits, labels, deltas, all_pipelines)
        print(metrics.report(results))

        log.info(f"  {PASS} Evaluation complete.")
        return True

    except Exception as e:
        log.error(f"  {FAIL} Evaluation failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n",      type=int, default=10, help="Clips per source (default 10)")
    parser.add_argument("--force",  action="store_true", help="Re-preprocess even if cached")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    log.info(f"Smoke test | device={args.device} | n={args.n} per source")

    clips: list[tuple[str, str]] = []
    for csv_path, id_col, path_col in [
        (cfg.paths.track1_meta, "output_stem", "output_path"),
        (cfg.paths.track2_meta, "output_stem", "output_path"),
        (cfg.paths.track3_meta, "output_stem", "output_path"),
        (cfg.paths.track4_meta, "output_stem", "output_path"),
    ]:
        batch = _sample_clips(csv_path, id_col, path_col, args.n)
        log.info(f"  Sampled {len(batch)} clips from {Path(csv_path).name}")
        clips.extend(batch)

    real_batch = _sample_clips(cfg.paths.meld_real_csv, "clip_id", "video_path", args.n)
    log.info(f"  Sampled {len(real_batch)} real MELD clips")
    clips.extend(real_batch)

    random.shuffle(clips)
    log.info(f"Total clips to smoke test: {len(clips)}")

    t_start = time.time()

    ok_count = stage_preprocess(cfg, clips, args.device, args.force)
    if ok_count == 0:
        log.error(f"{FAIL} No clips preprocessed successfully. Aborting.")
        sys.exit(1)

    ckpt_path = stage_train(cfg, args.device)
    if ckpt_path is None:
        log.error(f"{FAIL} Training failed. Aborting.")
        sys.exit(1)

    eval_ok = stage_evaluate(cfg, ckpt_path, args.device)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Smoke test done in {elapsed/60:.1f} min")
    print(f"  Preprocessing : {PASS if ok_count > 0 else FAIL}")
    print(f"  Training      : {PASS if ckpt_path else FAIL}")
    print(f"  Evaluation    : {PASS if eval_ok else FAIL}")
    print(f"{'='*50}")

    if not eval_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
