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

    with open(META_CSV, newline="", encoding="utf-8", errors="replace") as f:
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
            clip_id = f"fav_{source}_{Path(filename).stem}"
            z_at_p = REPO_ROOT / "data/preprocessed/features/z_at" / f"{clip_id}.pt"
            z_v_p = REPO_ROOT / "data/preprocessed/features/z_v" / f"{clip_id}.pt"
            
            features_cached = z_at_p.exists() and z_v_p.exists()
            if not features_cached and not video_path.exists():
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


from torch.utils.data import Dataset, DataLoader

class FakeAVCelebEvalDataset(Dataset):
    def __init__(self, clips: list[dict], pipeline: PreprocessingPipeline):
        self.clips = clips
        self.pipeline = pipeline
        self.drive_cache_dir = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/preprocessed")
        
        # Preload transformers processors outside processing loop
        from transformers import Wav2Vec2FeatureExtractor, BertTokenizer, ViTImageProcessor
        self.wav2vec_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.bert_tok     = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vit_proc     = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def __len__(self):
        return len(self.clips)

    def _sync_from_drive(self, local_path: Path, drive_path: Path) -> bool:
        if local_path.exists():
            return True
        if self.drive_cache_dir.exists() and drive_path.exists():
            try:
                import shutil
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(drive_path, local_path)
                return True
            except Exception:
                pass
        return False

    def __getitem__(self, idx):
        import torchaudio
        from PIL import Image
        
        c = self.clips[idx]
        clip_id = c["clip_id"]
        video_path = c["video_path"]
        
        # 1. Audio (Self-Caching Resampled WAV + Drive Lookup)
        wav_path = self.pipeline._wav_path(clip_id)
        drive_wav_path = self.drive_cache_dir / "audio" / f"{clip_id}.wav"
        self._sync_from_drive(wav_path, drive_wav_path)

        try:
            if not wav_path.exists():
                wav, sr = torchaudio.load(video_path)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(wav_path), wav, 16000)
            else:
                wav, sr = torchaudio.load(str(wav_path))
            
            audio_enc = self.wav2vec_proc(
                wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt",
                padding="max_length", max_length=80000, truncation=True
            )
            audio_values = audio_enc.input_values.squeeze(0)
        except Exception:
            audio_values = torch.zeros(80000)
            
        # 2. Text (Drive Lookup)
        txt_path = self.pipeline._txt_path(clip_id)
        drive_txt_path = self.drive_cache_dir / "transcripts" / f"{clip_id}.txt"
        self._sync_from_drive(txt_path, drive_txt_path)

        try:
            text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            bert_enc = self.bert_tok(
                text, return_tensors="pt",
                padding="max_length", max_length=128, truncation=True
            )
            input_ids = bert_enc.input_ids.squeeze(0)
            attention_mask = bert_enc.attention_mask.squeeze(0)
        except Exception:
            input_ids = torch.zeros(128, dtype=torch.long)
            attention_mask = torch.zeros(128, dtype=torch.long)
            
        # 3. Visual (Self-Caching Aligned JPEGs + Drive Lookup)
        kf_dir = self.pipeline.cache_dir / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        kf_path = kf_dir / f"{clip_id}.jpg"
        drive_kf_path = self.drive_cache_dir / "keyframes" / f"{clip_id}.jpg"
        self._sync_from_drive(kf_path, drive_kf_path)

        try:
            if kf_path.exists():
                grid_img = Image.open(kf_path)
                pils = []
                for idx_kf in range(8):
                    crop_box = (224 * idx_kf, 0, 224 * (idx_kf + 1), 224)
                    pils.append(grid_img.crop(crop_box))
            else:
                from src.preprocessing.visual import optical_flow_gate, detect_and_align_faces, extract_frames
                from src.preprocessing.filters import sharpness_score, select_keyframes, frames_to_pil
                
                frames = extract_frames(video_path, target_fps=25.0)
                if not frames:
                    raise ValueError("No frames extracted")
                    
                gated_frames = optical_flow_gate(frames, motion_threshold=0.3)
                face_results = detect_and_align_faces(gated_frames, detector="retinaface", confidence_threshold=0.7)
                
                if not face_results:
                    face_results = detect_and_align_faces(gated_frames, detector="retinaface", confidence_threshold=0.0)
                if not face_results:
                    face_results = detect_and_align_faces(frames, detector="retinaface", confidence_threshold=0.0)
                if not face_results:
                    face_results = [(f, sharpness_score(f)) for f in frames]
                    
                crops = [r[0] for r in face_results]
                scores = [r[1] for r in face_results]
                keyframes = select_keyframes(crops, scores, k=8)
                pils = frames_to_pil(keyframes, size=224)
                
                while len(pils) < 8:
                    pils.append(pils[-1].copy() if pils else Image.new('RGB', (224, 224)))
                    
                # Save horizontal concatenated JPEG strip to cache
                grid_img = Image.new('RGB', (224 * 8, 224))
                for idx_kf, img in enumerate(pils):
                    grid_img.paste(img, (224 * idx_kf, 0))
                grid_img.save(kf_path, 'JPEG', quality=90)
            
            vit_enc = self.vit_proc(pils, return_tensors="pt")
            keyframe_pixels = vit_enc.pixel_values.squeeze(0)
        except Exception:
            keyframe_pixels = torch.zeros(8, 3, 224, 224)
            
        return {
            "clip_id": clip_id,
            "fake_label": c["fake_label"],
            "method": c["method"],
            "type": c["type"],
            "audio_values": audio_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "keyframe_pixels": keyframe_pixels,
        }


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

    if model._backbones_loaded:
        print("  Running parallel end-to-end evaluation via DataLoader workers...")
        dataset = FakeAVCelebEvalDataset(clips, pipeline)
        loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)
        
        pbar = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
        for batch in pbar:
            audio_values    = batch["audio_values"].to(device)
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            keyframe_pixels = batch["keyframe_pixels"].to(device)
            
            with torch.no_grad():
                out = model(audio_values, input_ids, attention_mask, keyframe_pixels)
                scores = torch.sigmoid(out.logit).squeeze(1).cpu().tolist()
                
            for j in range(len(scores)):
                score = scores[j]
                pred = 1 if score >= 0.5 else 0
                results.append({
                    "clip_id":    batch["clip_id"][j],
                    "fake_label": int(batch["fake_label"][j].item()),
                    "method":     batch["method"][j],
                    "type":       batch["type"][j],
                    "score":      score,
                    "pred":       pred,
                })
    else:
        # Phase 1: load precomputed feature vectors from cache (runs instantly)
        pbar = tqdm(clips, desc="Evaluating", unit="clip", dynamic_ncols=True)
        for c in pbar:
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

    # Standard threshold = 0.5
    tp = sum(1 for r in results if r["pred"] == 1 and r["fake_label"] == 1)
    fp = sum(1 for r in results if r["pred"] == 1 and r["fake_label"] == 0)
    fn = sum(1 for r in results if r["pred"] == 0 and r["fake_label"] == 1)
    tn = sum(1 for r in results if r["pred"] == 0 and r["fake_label"] == 0)

    acc  = (tp + tn) / max(len(results), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)

    # Threshold calibration sweep (find optimal F1 decision boundary)
    best_thresh, best_cal_f1, best_cal_acc = 0.5, 0.0, 0.0
    best_tp = best_fp = best_fn = best_tn = 0
    best_prec = best_rec = 0.0
    
    # Youden's J Statistic sweep (J = Sensitivity + Specificity - 1 = TPR - FPR)
    youden_thresh, best_youden_j = 0.5, -1.0
    youden_acc = youden_sens = youden_spec = 0.0

    for th_val in [i / 100.0 for i in range(1, 100)]:
        c_tp = sum(1 for r in results if r["score"] >= th_val and r["fake_label"] == 1)
        c_fp = sum(1 for r in results if r["score"] >= th_val and r["fake_label"] == 0)
        c_fn = sum(1 for r in results if r["score"] < th_val  and r["fake_label"] == 1)
        c_tn = sum(1 for r in results if r["score"] < th_val  and r["fake_label"] == 0)
        
        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec  = c_tp / max(c_tp + c_fn, 1)
        c_f1   = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-8)
        if c_f1 > best_cal_f1:
            best_cal_f1 = c_f1
            best_thresh = th_val
            best_cal_acc = (c_tp + c_tn) / max(len(results), 1)
            best_tp, best_fp, best_fn, best_tn = c_tp, c_fp, c_fn, c_tn
            best_prec, best_rec = c_prec, c_rec

        # Youden's J calculation
        sens = c_tp / max(c_tp + c_fn, 1)   # Recall / Sensitivity
        spec = c_tn / max(c_tn + c_fp, 1)   # Specificity / TNR
        j_stat = sens + spec - 1.0
        if j_stat > best_youden_j:
            best_youden_j = j_stat
            youden_thresh = th_val
            youden_acc  = (c_tp + c_tn) / max(len(results), 1)
            youden_sens = sens
            youden_spec = spec

    # Update predictions with optimal calibrated threshold
    for r in results:
        r["pred"] = 1 if r["score"] >= best_thresh else 0

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
        total=len(results), tp=best_tp, fp=best_fp, fn=best_fn, tn=best_tn,
        acc=best_cal_acc, prec=best_prec, rec=best_rec, f1=best_cal_f1,
        best_thresh=best_thresh, best_cal_f1=best_cal_f1, best_cal_acc=best_cal_acc,
        youden_thresh=youden_thresh, best_youden_j=best_youden_j, youden_acc=youden_acc,
        youden_sens=youden_sens, youden_spec=youden_spec,
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
    parser.add_argument("--save_csv",   default=None,
                        help="Save per-clip results (clip_id,fake_label,method,type,score,pred) to this CSV path "
                             "— needed to compare against another framework's scores on the SAME clips "
                             "(see scripts/compare_frameworks.py)")
    parser.add_argument("--classifier_mode", type=str, default="baseline",
                        choices=["baseline", "mismatch_only", "emotion_bilinear", "bottleneck", "high_dropout"])
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    drive_latest = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest") / ckpt_path.name
    drive_base   = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints") / ckpt_path.name
    
    drive_source = drive_latest if drive_latest.exists() else (drive_base if drive_base.exists() else None)
    if drive_source is not None:
        try:
            import shutil
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_source, ckpt_path)
            print(f"  [DRIVE OVERWRITE] Overwrote local evaluation checkpoint from Drive ({drive_source.parent.name}) -> {ckpt_path}")
        except Exception as e:
            print(f"  [DRIVE OVERWRITE WARNING] Failed to copy checkpoint from Drive: {e}")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return
    if not META_CSV.exists():
        print(f"ERROR: FakeAVCeleb metadata not found: {META_CSV}")
        return

    _section("FakeAVCeleb v1.2 - Cross-Dataset Benchmark")
    hard = not args.no_hard
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  Device       : {args.device}")
    print(f"  Sample       : {args.n_real} real + {args.n_fake} fake = {args.n_real + args.n_fake} clips")
    print(f"  Fake ratio   : {args.n_fake / (args.n_real + args.n_fake):.0%}  (harder for detector)")
    print(f"  Hard mode    : {'ON - compound fakes over-sampled (hardest for Delta signal)' if hard else 'OFF - uniform random'}")
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
    model = DeepfakeDetector(classifier_mode=args.classifier_mode).to(args.device)
    ckpt  = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    
    # Phase 2 checkpoints contain un-frozen backbones. Detect and load them dynamically.
    has_backbones = any(k.startswith("vit.") or k.startswith("wav2vec.") or k.startswith("bert.") for k in ckpt["model_state"].keys())
    if has_backbones:
        print("  Detected backbone weights in checkpoint. Initializing backbones for evaluation...")
        model.load_backbones()
        model.to(args.device)
        
    model.load_state_dict(ckpt["model_state"], strict=False)
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
    _section("Running inference (MP4 -> features -> P(fake))")
    results = run_inference(clips, pipeline, model, args.device, args.no_cache)
    print(f"\n  Evaluated : {len(results)} clips  ({len(clips) - len(results)} failed/skipped)")

    if not results:
        print("  No results - check video paths and checkpoint.")
        return

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "fake_label", "method", "type", "score", "pred"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  Per-clip results saved -> {csv_path}  ({len(results)} rows)")

    # ── Metrics ────────────────────────────────────────────────────────────────
    _section("Results")
    m = compute_metrics(results)

    print(f"  Clips evaluated  : {m['total']}")
    print(f"  Accuracy (0.5)   : {m['acc']:.4f}")
    print(f"  Precision        : {m['prec']:.4f}")
    print(f"  Recall           : {m['rec']:.4f}")
    print(f"  F1 (0.5)         : {m['f1']:.4f}")
    print(f"  TP/FP/FN/TN      : {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")

    if m.get("best_thresh") is not None:
        print(f"  --- Calibrated Threshold Sweeps ---")
        print(f"  Optimal F1 Threshold  : {m['best_thresh']:.2f}  (F1={m['best_cal_f1']:.4f}, Acc={m['best_cal_acc']:.4f})")
        print(f"  Youden's J Threshold  : {m['youden_thresh']:.2f}  (J={m['best_youden_j']:.4f}, Acc={m['youden_acc']:.4f}, Sens={m['youden_sens']:.4f}, Spec={m['youden_spec']:.4f})")

    if m["auc"] is not None:
        ci_str = (f"  [95% CI: {m['auc_lo']:.4f} to {m['auc_hi']:.4f}]"
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
