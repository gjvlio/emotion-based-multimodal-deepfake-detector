"""
evaluate_fakeavceleb.py — Benchmark trained detector on FakeAVCeleb v1.2.

Input  : MP4 videos in FakeAVCeleb_v1.2/ (test-only, never trained on)
Output : AUC-ROC, Accuracy, F1, per-method breakdown, Bootstrap 95% CI

Pipeline per clip (internal, user doesn't manage this):
    MP4 → Wav2Vec2 + BERT → Z_at (1536)
          ViT keyframes    → Z_v  (768)
          → DeepfakeDetector.forward_from_features() → P(fake)

Features are cached after first run — re-running reuses cache (fast).

Usage:
    python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/smoke/best_phase1.pt
    python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/full/best_phase1.pt
    python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/smoke/best_phase1.pt --n_real 200 --n_fake 800
    python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/smoke/best_phase1.pt --no_cache
    python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/smoke/best_phase1.pt --no_hard
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.detection_model import DeepfakeDetector
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).resolve().parents[1]
FAV_ROOT   = REPO_ROOT / "data/raw/FakeAVCeleb_v1.2"
META_CSV   = FAV_ROOT  / "meta_data.csv"
REAL_TYPE  = "RealVideo-RealAudio"
SECTION    = "=" * 60


def _section(title: str) -> None:
    print(f"\n{SECTION}\n  {title}\n{SECTION}")


# Compound fakes manipulate BOTH audio and video → smaller audio-visual mismatch (Δ) → hardest for our detector.
# Single-modality fakes create real Z_at/Z_v divergence → easier.
_HARD_METHODS  = {"faceswap-wav2lip", "fsgan-wav2lip"}  # compound: both modalities fake
_MED_METHODS   = {"wav2lip"}                             # lip sync only — medium
_EASY_METHODS  = {"faceswap", "fsgan", "rtvc"}           # single-modality — easiest

# Target fake budget proportions for hard mode
_HARD_FRAC = 0.40   # 40% compound
_MED_FRAC  = 0.40   # 40% wav2lip
_EASY_FRAC = 0.20   # 20% single-modality


def load_clips(n_real: int, n_fake: int, seed: int = 42, hard: bool = True) -> list[dict]:
    """
    Stratified random sample from meta_data.csv.
    hard=True: over-samples compound fakes (hardest for audio-visual mismatch detector).
      40% compound (faceswap-wav2lip + fsgan-wav2lip), 40% wav2lip, 20% other.
    hard=False: uniform random sample from all fake methods.
    """
    import random
    rng = random.Random(seed)

    real_pool: list[dict] = []
    fake_by_tier: dict[str, list[dict]] = {"hard": [], "med": [], "easy": []}
    missing = 0

    with open(META_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat      = row.get("type",   "").strip()
            race     = row.get("race",   "").strip()
            gender   = row.get("gender", "").strip()
            source   = row.get("source", "").strip()
            filename = row.get("path",   "").strip()
            method   = row.get("method", "real").strip()

            if not all([cat, race, gender, source, filename]):
                continue

            video_path = FAV_ROOT / cat / race / gender / source / filename
            if not video_path.exists():
                missing += 1
                continue

            entry = {
                "clip_id":    f"fav_{source}_{Path(filename).stem}",
                "video_path": str(video_path),
                "fake_label": 0 if cat == REAL_TYPE else 1,
                "method":     method,
                "type":       cat,
                "speaker_id": source,
            }
            if cat == REAL_TYPE:
                real_pool.append(entry)
            elif method in _HARD_METHODS:
                fake_by_tier["hard"].append(entry)
            elif method in _MED_METHODS:
                fake_by_tier["med"].append(entry)
            else:
                fake_by_tier["easy"].append(entry)

    if missing:
        log.warning(f"{missing} metadata rows skipped — video not found on disk")

    for pool in [real_pool, *fake_by_tier.values()]:
        rng.shuffle(pool)

    sampled_real = real_pool[:n_real]
    if len(sampled_real) < n_real:
        log.warning(f"Requested {n_real} real clips but only {len(sampled_real)} available")

    if hard:
        n_h = min(int(n_fake * _HARD_FRAC), len(fake_by_tier["hard"]))
        n_m = min(int(n_fake * _MED_FRAC),  len(fake_by_tier["med"]))
        n_e = min(n_fake - n_h - n_m,       len(fake_by_tier["easy"]))
        sampled_fake = (fake_by_tier["hard"][:n_h]
                        + fake_by_tier["med"][:n_m]
                        + fake_by_tier["easy"][:n_e])
        # fill shortfall from any tier
        if len(sampled_fake) < n_fake:
            used = set(c["clip_id"] for c in sampled_fake)
            all_fake = [c for tier in fake_by_tier.values() for c in tier if c["clip_id"] not in used]
            rng.shuffle(all_fake)
            sampled_fake += all_fake[:n_fake - len(sampled_fake)]
        print(f"  Hard sampling  : {n_h} compound + {n_m} wav2lip + {n_e} single-mod")
    else:
        all_fake = [c for tier in fake_by_tier.values() for c in tier]
        rng.shuffle(all_fake)
        sampled_fake = all_fake[:n_fake]

    if len(sampled_fake) < n_fake:
        log.warning(f"Requested {n_fake} fake clips but only {len(sampled_fake)} available")

    combined = sampled_real + sampled_fake
    rng.shuffle(combined)
    return combined


def run_inference(
    clips:    list[dict],
    pipeline: PreprocessingPipeline,
    model:    DeepfakeDetector,
    device:   str,
    no_cache: bool,
) -> list[dict]:
    """
    For each clip: extract features (or load from cache) → run detector.
    Returns list of {clip_id, fake_label, method, type, score, pred}.
    """
    results = []
    model.eval()

    pbar = tqdm(clips, desc="Evaluating", unit="clip", dynamic_ncols=True)
    for c in pbar:
        # Feature extraction (cached after first run)
        feats = pipeline.process(c["clip_id"], c["video_path"], force=no_cache)
        if feats is None:
            log.warning(f"Feature extraction failed: {c['clip_id']} — skipping")
            continue

        z_at = feats.z_at.unsqueeze(0).to(device)   # (1, 1536)
        z_v  = feats.z_v.unsqueeze(0).to(device)    # (1, 768)

        with torch.no_grad():
            out   = model.forward_from_features(z_at, z_v)
            score = torch.sigmoid(out.logit).item()     # P(fake) ∈ [0, 1]
            pred  = 1 if score >= 0.5 else 0

        results.append({
            "clip_id":    c["clip_id"],
            "fake_label": c["fake_label"],
            "method":     c["method"],
            "type":       c["type"],
            "score":      score,
            "pred":       pred,
        })

        pbar.set_postfix(score=f"{score:.3f}")

    return results


def compute_metrics(results: list[dict]) -> dict:
    labels = [r["fake_label"] for r in results]
    scores = [r["score"]      for r in results]
    preds  = [r["pred"]       for r in results]

    tp = sum(1 for r in results if r["pred"] == 1 and r["fake_label"] == 1)
    fp = sum(1 for r in results if r["pred"] == 1 and r["fake_label"] == 0)
    fn = sum(1 for r in results if r["pred"] == 0 and r["fake_label"] == 1)
    tn = sum(1 for r in results if r["pred"] == 0 and r["fake_label"] == 0)

    acc  = (tp + tn) / max(len(results), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)

    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) == 2:
            auc = roc_auc_score(labels, scores)
    except ImportError:
        pass

    # Bootstrap 95% CI on AUC
    auc_lo = auc_hi = None
    if auc is not None:
        import numpy as np
        rng = np.random.default_rng(42)
        n   = len(results)
        boot_aucs = []
        for _ in range(10_000):
            idx    = rng.integers(0, n, size=n)
            b_lab  = [labels[i] for i in idx]
            b_scr  = [scores[i] for i in idx]
            if len(set(b_lab)) == 2:
                try:
                    boot_aucs.append(roc_auc_score(b_lab, b_scr))
                except Exception:
                    pass
        if boot_aucs:
            auc_lo, auc_hi = float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))

    return dict(
        total=len(results), tp=tp, fp=fp, fn=fn, tn=tn,
        acc=acc, prec=prec, rec=rec, f1=f1,
        auc=auc, auc_lo=auc_lo, auc_hi=auc_hi,
    )


def per_method_breakdown(results: list[dict]) -> None:
    from collections import defaultdict
    by_method: dict[str, list] = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)

    print(f"\n  {'Method':<25} {'N':>6}  {'Acc':>6}  {'AUC':>6}")
    print(f"  {'-'*50}")
    for method, recs in sorted(by_method.items()):
        labs  = [r["fake_label"] for r in recs]
        scrs  = [r["score"]      for r in recs]
        preds = [r["pred"]       for r in recs]
        a     = sum(p == l for p, l in zip(preds, labs)) / max(len(recs), 1)
        auc_m = None
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(labs)) == 2:
                auc_m = roc_auc_score(labs, scrs)
        except Exception:
            pass
        auc_str = f"{auc_m:.4f}" if auc_m is not None else "  N/A"
        print(f"  {method:<25} {len(recs):>6}  {a:.4f}  {auc_str}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark detector on FakeAVCeleb v1.2")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained checkpoint (e.g. checkpoints/smoke/best_phase1.pt)")
    parser.add_argument("--config",     default=None)
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device",     default=_default_device)
    parser.add_argument("--n_real",     type=int, default=200,
                        help="Number of real clips to sample (default 200)")
    parser.add_argument("--n_fake",     type=int, default=800,
                        help="Number of fake clips to sample (default 800)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for clip sampling")
    parser.add_argument("--no_cache",   action="store_true",
                        help="Ignore feature cache and reprocess all clips")
    parser.add_argument("--no_hard",    action="store_true",
                        help="Disable hard method stratification (uniform random sample)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return
    if not META_CSV.exists():
        print(f"ERROR: FakeAVCeleb metadata not found: {META_CSV}")
        return

    _section("FakeAVCeleb v1.2 — Cross-Dataset Benchmark")
    hard = not args.no_hard
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Device       : {args.device}")
    print(f"  Sample       : {args.n_real} real + {args.n_fake} fake = {args.n_real + args.n_fake} clips")
    print(f"  Fake ratio   : {args.n_fake / (args.n_real + args.n_fake):.0%}  (harder for detector)")
    print(f"  Hard mode    : {'ON — compound fakes over-sampled (hardest for Δ signal)' if hard else 'OFF — uniform random'}")
    print(f"  Seed         : {args.seed}")
    print(f"  Cache        : {'disabled' if args.no_cache else 'enabled (fast on re-run)'}")
    print(f"  NOTE         : FakeAVCeleb is TEST-ONLY — model was never trained on it.")

    # ── Load clips ─────────────────────────────────────────────────────────────
    _section("Sampling FakeAVCeleb clips")
    clips = load_clips(args.n_real, args.n_fake, seed=args.seed, hard=hard)
    real_n = sum(1 for c in clips if c["fake_label"] == 0)
    fake_n = sum(1 for c in clips if c["fake_label"] == 1)
    print(f"  Sampled      : {len(clips)} clips")
    print(f"    Real       : {real_n}")
    print(f"    Fake       : {fake_n}")

    # ── Load model ─────────────────────────────────────────────────────────────
    _section("Loading trained model")
    model = DeepfakeDetector().to(args.device)
    ckpt  = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', '?')}")

    # ── Load preprocessing pipeline ────────────────────────────────────────────
    _section("Initializing preprocessing pipeline")
    cfg = Config.from_yaml(args.config)
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
        device        = args.device,
    )
    cached = sum(1 for c in clips if pipeline.is_cached(c["clip_id"]))
    to_process = len(clips) - cached
    print(f"  Already cached : {cached}/{len(clips)} clips (run instantly)")
    print(f"  Need extraction: {to_process} clips (~{to_process * 9 // 60}min on GPU)")

    # ── Run inference ──────────────────────────────────────────────────────────
    _section("Running inference (MP4 → features → P(fake))")
    results = run_inference(clips, pipeline, model, args.device, args.no_cache)
    print(f"\n  Evaluated : {len(results)} clips  ({len(clips) - len(results)} failed/skipped)")

    if not results:
        print("  No results — check video paths and checkpoint.")
        return

    # ── Metrics ────────────────────────────────────────────────────────────────
    _section("Results")
    m = compute_metrics(results)

    print(f"  Clips evaluated  : {m['total']}")
    print(f"  Accuracy         : {m['acc']:.4f}")
    print(f"  Precision        : {m['prec']:.4f}")
    print(f"  Recall           : {m['rec']:.4f}")
    print(f"  F1               : {m['f1']:.4f}")
    print(f"  TP/FP/FN/TN      : {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")

    if m["auc"] is not None:
        ci_str = (f"  [95% CI: {m['auc_lo']:.4f}–{m['auc_hi']:.4f}]"
                  if m["auc_lo"] is not None else "")
        print(f"  AUC-ROC          : {m['auc']:.4f}{ci_str}")
        # Elpeltagy et al. (2023) — multimodal (whole videos) AUROC: 97.21%
        ELPELTAGY_AUC = 0.9721
        delta = m["auc"] - ELPELTAGY_AUC
        sign  = "+" if delta >= 0 else ""
        print(f"  vs Elpeltagy 2023: {ELPELTAGY_AUC:.4f}  (ours {sign}{delta:.4f})")
    else:
        print("  AUC-ROC          : N/A (need both classes in evaluated set)")

    _section("Per-method breakdown")
    per_method_breakdown(results)

    print(f"\n  NOTE: Smoke checkpoint trained on ~1k clips. Full dataset will yield higher AUC.")
    print(f"  Rival comparison requires DeLong's test — run evaluation notebook for full stats.\n")


if __name__ == "__main__":
    main()
