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
import os
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


def load_clips(n_real: int = 200, n_fake: int = 800, seed: int = 42, hard: bool = True, ignore_missing: bool = False) -> list[dict]:
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

    meta_candidates = [
        FAV_ROOT / "meta_data.csv",
        REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/meta_data.csv",
        REPO_ROOT / "data/FakeAVCeleb_v1.2/meta_data.csv",
        REPO_ROOT / "data/raw/FakeAVCeleb/meta_data.csv",
        REPO_ROOT / "data/FakeAVCeleb/meta_data.csv",
        REPO_ROOT / "data/raw/meta_data.csv",
        REPO_ROOT / "data/meta_data.csv",
        REPO_ROOT / "data/preprocessed/fakeavceleb_cached_manifest.csv",
    ]
    csv_file_to_use = None
    for c in meta_candidates:
        if c.exists():
            csv_file_to_use = c
            break

    if csv_file_to_use is None or not csv_file_to_use.exists():
        log.error("Neither meta_data.csv nor fakeavceleb_cached_manifest.csv found.")
        return []

    print(f"  [Dataset] Using metadata CSV: {csv_file_to_use}")

    fav_data_dirs = [
        FAV_ROOT,
        FAV_ROOT / "FakeAVCeleb",
        FAV_ROOT / "FakeAVCeleb_v1.2",
        FAV_ROOT / "FakeAVCeleb_v1.2" / "FakeAVCeleb",
        REPO_ROOT / "data/raw/FakeAVCeleb_v1.2",
        REPO_ROOT / "data/raw/FakeAVCeleb_v1.2/FakeAVCeleb",
        REPO_ROOT / "data/raw/FakeAVCeleb",
        REPO_ROOT / "data/raw",
        REPO_ROOT / "data/FakeAVCeleb_v1.2",
        REPO_ROOT / "data/FakeAVCeleb_v1.2/FakeAVCeleb",
        REPO_ROOT / "data/FakeAVCeleb",
        REPO_ROOT / "data",
    ]
    active_fav_roots = [d for d in fav_data_dirs if d.exists() and any((d / t).exists() for t in [
        "RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"
    ])]
    if not active_fav_roots:
        active_fav_roots = fav_data_dirs
    else:
        print(f"  [Dataset] Discovered active FakeAVCeleb video root: {active_fav_roots[0]}")

    print("  [Dataset] Building fast MP4 video hashmap index from disk...")
    mp4_index = {}
    for p in REPO_ROOT.glob("data/**/*.mp4"):
        mp4_index[p.name] = p
        mp4_index[f"{p.parent.name}/{p.name}"] = p
    print(f"  [Dataset] Indexed {len(mp4_index):,} MP4 video files ready on local SSD.")

    with open(csv_file_to_use, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cat      = row.get("type",   "").strip()
            source   = row.get("source", row.get("speaker_id", "")).strip()
            filename = row.get("path",   "").strip()
            method   = row.get("method", "real").strip()
            clip_id  = row.get("clip_id", "").strip() or f"fav_{source}_{Path(filename).stem if filename else 'clip'}"
            race     = row.get("race",   "").strip()
            gender   = row.get("gender", "").strip()
            rel_v_path = row.get("video_path", "").strip()

            video_path = None
            if rel_v_path:
                for base in [REPO_ROOT, FAV_ROOT]:
                    p = base / rel_v_path
                    if p.is_file():
                        video_path = p
                        break

            # 1. Look up via active directory hierarchy
            if video_path is None and filename:
                for root in active_fav_roots:
                    cand = root / cat / race / gender / source / filename
                    if cand.is_file():
                        video_path = cand
                        break

            # 2. Fast hashmap index lookup
            if video_path is None and filename:
                video_path = mp4_index.get(f"{source}/{filename}") or mp4_index.get(filename)

            # 3. Clip_id fallback extraction
            if video_path is None and clip_id:
                parts = clip_id.split("_")
                if len(parts) >= 3:
                    spk = parts[1]
                    fname = f"{parts[2]}.mp4"
                    video_path = mp4_index.get(f"{spk}/{fname}") or mp4_index.get(fname)

            z_at_p = REPO_ROOT / "data/preprocessed/features/z_at" / f"{clip_id}.pt"
            z_v_p  = REPO_ROOT / "data/preprocessed/features/z_v"  / f"{clip_id}.pt"
            features_cached = z_at_p.is_file() and z_v_p.is_file()

            if not ignore_missing and not features_cached and video_path is None:
                missing += 1
                continue

            entry = {
                "clip_id":    clip_id,
                "video_path": str(video_path) if video_path is not None else "",
                "fake_label": 0 if cat == REAL_TYPE else 1,
                "method":     method,
                "type":       cat,
                "speaker_id": source,
                "cached":     features_cached,
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

    sampled_real = real_pool[:min(n_real, len(real_pool))]
    if len(sampled_real) < n_real:
        log.info(f"Sampled all {len(sampled_real):,} available real clips from FakeAVCeleb.")

    n_fake_target = n_fake
    if hard:
        n_h = min(int(n_fake_target * _HARD_FRAC), len(fake_by_tier["hard"]))
        n_m = min(int(n_fake_target * _MED_FRAC),  len(fake_by_tier["med"]))
        n_e = min(n_fake_target - n_h - n_m,       len(fake_by_tier["easy"]))
        sampled_fake = (fake_by_tier["hard"][:n_h]
                        + fake_by_tier["med"][:n_m]
                        + fake_by_tier["easy"][:n_e])
        if len(sampled_fake) < n_fake_target:
            used = set(c["clip_id"] for c in sampled_fake)
            all_fake = [c for tier in fake_by_tier.values() for c in tier if c["clip_id"] not in used]
            rng.shuffle(all_fake)
            sampled_fake += all_fake[:n_fake_target - len(sampled_fake)]
        print(f"  Hard sampling  : {n_h} compound + {n_m} wav2lip + {n_e} single-mod")
    else:
        all_fake = [c for tier in fake_by_tier.values() for c in tier]
        rng.shuffle(all_fake)
        sampled_fake = all_fake[:n_fake_target]

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
        import cv2
        from PIL import Image
        
        c = self.clips[idx]
        clip_id = c["clip_id"]
        video_path = c["video_path"]
        
        # 1. Audio (Fast resampled WAV extraction)
        wav_path = self.pipeline._wav_path(clip_id)
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
            
        # 2. Text (Fast transcript check)
        txt_path = self.pipeline._txt_path(clip_id)
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
            
        # 3. Visual (Fast Keyframe Extraction: 16 Uniform Temporal Samples)
        kf_dir = self.pipeline.cache_dir / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        kf_path = kf_dir / f"{clip_id}.jpg"

        try:
            if kf_path.exists():
                grid_img = Image.open(kf_path)
                pils = []
                for idx_kf in range(8):
                    crop_box = (224 * idx_kf, 0, 224 * (idx_kf + 1), 224)
                    pils.append(grid_img.crop(crop_box))
            else:
                from src.preprocessing.visual import detect_and_align_faces
                from src.preprocessing.filters import sharpness_score, select_keyframes, frames_to_pil
                
                cap = cv2.VideoCapture(video_path)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_indices = [int(i * total_f / 16) for i in range(16)] if total_f > 16 else list(range(max(total_f, 1)))
                
                frames = []
                for f_idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()

                if not frames:
                    raise ValueError("No frames read from video")

                face_results = detect_and_align_faces(frames, detector="retinaface", confidence_threshold=0.5)
                if not face_results:
                    face_results = [(f, sharpness_score(f)) for f in frames]
                    
                crops = [r[0] for r in face_results]
                scores = [r[1] for r in face_results]
                keyframes = select_keyframes(crops, scores, k=8)
                pils = frames_to_pil(keyframes, size=224)
                
                while len(pils) < 8:
                    pils.append(pils[-1].copy() if pils else Image.new('RGB', (224, 224)))
                    
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
    clips: list[dict],
    pipeline: PreprocessingPipeline,
    model: DeepfakeDetector,
    device: str,
    no_cache: bool,
    force_features: bool = False,
    cached_only: bool = False,
) -> list[dict]:
    """
    For each clip: extract features (or load from cache) → run detector.
    Returns list of {clip_id, fake_label, method, type, score, pred}.
    """
    results = []
    model.eval()

    all_cached = all(c.get("cached", False) for c in clips) and not no_cache
    if model._backbones_loaded and not force_features and not cached_only:
        n_workers = 0 if os.name == "nt" else 2
        dataset = FakeAVCelebEvalDataset(clips, pipeline)
        loader = DataLoader(dataset, batch_size=8, num_workers=n_workers, pin_memory=(device != "cpu"))
        
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
        # Load precomputed feature vectors from cache (runs in ~1 second)
        pbar = tqdm(clips, desc="Evaluating (from cached features)", unit="clip", dynamic_ncols=True)
        for c in pbar:
            z_at_path = REPO_ROOT / "data/preprocessed/features/z_at" / f"{c['clip_id']}.pt"
            z_v_path  = REPO_ROOT / "data/preprocessed/features/z_v"  / f"{c['clip_id']}.pt"
            
            if z_at_path.exists() and z_v_path.exists():
                z_at = torch.load(z_at_path, map_location=device, weights_only=True).unsqueeze(0)
                z_v  = torch.load(z_v_path,  map_location=device, weights_only=True).unsqueeze(0)
            else:
                feats = pipeline.process(c["clip_id"], c["video_path"], force=no_cache)
                if feats is None:
                    log.warning(f"Feature extraction failed: {c['clip_id']} — skipping")
                    continue
                z_at = feats.z_at.unsqueeze(0).to(device)
                z_v  = feats.z_v.unsqueeze(0).to(device)

            with torch.no_grad():
                out   = model.forward_from_features(z_at, z_v)
                score = torch.sigmoid(out.logit).item()     # P(fake) in [0, 1]
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
    import numpy as np
    labels = [r["fake_label"] for r in results]
    scores = [r["score"]      for r in results]

    # Standard metrics at tau = 0.50
    std_p = (np.array(scores) >= 0.5).astype(int)
    y_true = np.array(labels)
    tp_50 = int(((std_p == 1) & (y_true == 1)).sum())
    fp_50 = int(((std_p == 1) & (y_true == 0)).sum())
    fn_50 = int(((std_p == 0) & (y_true == 1)).sum())
    tn_50 = int(((std_p == 0) & (y_true == 0)).sum())
    
    acc_50 = (tp_50 + tn_50) / max(len(results), 1)
    prec_50 = tp_50 / max(tp_50 + fp_50, 1)
    rec_50  = tp_50 / max(tp_50 + fn_50, 1)
    spec_50 = tn_50 / max(tn_50 + fp_50, 1)
    f1_50   = 2 * prec_50 * rec_50 / max(prec_50 + rec_50, 1e-8)
    bal_acc_50 = (rec_50 + spec_50) / 2.0
    
    mcc_denom = np.sqrt(float((tp_50 + fp_50) * (tp_50 + fn_50) * (tn_50 + fp_50) * (tn_50 + fn_50)))
    mcc_50 = ((tp_50 * tn_50) - (fp_50 * fn_50)) / max(mcc_denom, 1e-8)

    # Threshold calibration sweep (find optimal F1 decision boundary)
    best_cal_f1   = -1.0
    best_thresh   = 0.5
    best_tp, best_fp, best_fn, best_tn = tp_50, fp_50, fn_50, tn_50
    best_cal_acc, best_prec, best_rec = acc_50, prec_50, rec_50

    best_youden_j = -2.0
    youden_thresh = 0.5
    youden_acc, youden_sens, youden_spec, youden_bal_acc = acc_50, rec_50, spec_50, bal_acc_50
    youden_tp, youden_fp, youden_fn, youden_tn = tp_50, fp_50, fn_50, tn_50

    zero_fp_thresh, zero_fp_sens = None, 0.0

    for th in range(1, 100):
        th_val = th / 100.0
        p_th = (np.array(scores) >= th_val).astype(int)
        c_tp = int(((p_th == 1) & (y_true == 1)).sum())
        c_fp = int(((p_th == 1) & (y_true == 0)).sum())
        c_fn = int(((p_th == 0) & (y_true == 1)).sum())
        c_tn = int(((p_th == 0) & (y_true == 0)).sum())

        if c_fp == 0 and (zero_fp_thresh is None or th_val < zero_fp_thresh):
            zero_fp_thresh = th_val
            zero_fp_sens   = c_tp / max(c_tp + c_fn, 1)

        c_prec = c_tp / max(c_tp + c_fp, 1)
        c_rec  = c_tp / max(c_tp + c_fn, 1)
        c_f1   = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-8)
        if c_f1 > best_cal_f1:
            best_cal_f1 = c_f1
            best_thresh = th_val
            best_cal_acc = (c_tp + c_tn) / max(len(results), 1)
            best_tp, best_fp, best_fn, best_tn = c_tp, c_fp, c_fn, c_tn
            best_prec, best_rec = c_prec, c_rec

        # Youden's J calculation (Sensitivity + Specificity - 1)
        sens = c_tp / max(c_tp + c_fn, 1)
        spec = c_tn / max(c_tn + c_fp, 1)
        j_stat = sens + spec - 1.0
        if j_stat > best_youden_j:
            best_youden_j = j_stat
            youden_thresh = th_val
            youden_acc    = (c_tp + c_tn) / max(len(results), 1)
            youden_sens   = sens
            youden_spec   = spec
            youden_bal_acc = (sens + spec) / 2.0
            youden_tp, youden_fp, youden_fn, youden_tn = c_tp, c_fp, c_fn, c_tn

    # Update predictions with optimal Youden's J calibrated threshold
    for r in results:
        r["pred"] = 1 if r["score"] >= youden_thresh else 0

    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) == 2:
            auc = float(roc_auc_score(labels, scores))
    except ImportError:
        pass

    # Bootstrap 95% CI on AUC
    auc_lo = auc_hi = None
    if auc is not None:
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
        total=len(results),
        tp_50=tp_50, fp_50=fp_50, fn_50=fn_50, tn_50=tn_50,
        acc_50=acc_50, prec_50=prec_50, rec_50=rec_50, spec_50=spec_50,
        f1_50=f1_50, bal_acc_50=bal_acc_50, mcc_50=mcc_50,
        best_thresh=best_thresh, best_cal_f1=best_cal_f1, best_cal_acc=best_cal_acc,
        youden_thresh=youden_thresh, best_youden_j=best_youden_j, youden_acc=youden_acc,
        youden_sens=youden_sens, youden_spec=youden_spec, youden_bal_acc=youden_bal_acc,
        youden_tp=youden_tp, youden_fp=youden_fp, youden_fn=youden_fn, youden_tn=youden_tn,
        zero_fp_thresh=zero_fp_thresh, zero_fp_sens=zero_fp_sens,
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


def ensure_preprocessed_features():
    """Checks local and Google Drive preprocessed feature tensors (z_at and z_v) and syncs if needed."""
    z_at_dir = REPO_ROOT / "data/preprocessed/features/z_at"
    z_v_dir  = REPO_ROOT / "data/preprocessed/features/z_v"
    z_at_dir.mkdir(parents=True, exist_ok=True)
    z_v_dir.mkdir(parents=True, exist_ok=True)

    drive_base = Path("/content/drive/MyDrive/THESIS_MOTHERFILE")
    drive_z_at = drive_base / "preprocessed/features/z_at"
    drive_z_v  = drive_base / "preprocessed/features/z_v"

    # 1. Check local features_drive folder (downloaded from Drive)
    local_drive_at = REPO_ROOT / "data/preprocessed/features_drive/z_at"
    local_drive_v  = REPO_ROOT / "data/preprocessed/features_drive/z_v"
    if local_drive_at.exists() and local_drive_v.exists():
        import shutil
        fd_at_names    = set(p.name for p in local_drive_at.glob("*.pt"))
        local_at_names = set(p.name for p in z_at_dir.glob("*.pt"))
        missing_fd     = fd_at_names - local_at_names
        if missing_fd:
            print(f"  [LOCAL FEATURES_DRIVE SYNC] Merging {len(missing_fd):,} missing feature tensors from features_drive folder...")
            synced_fd = 0
            for name in missing_fd:
                src_at = local_drive_at / name
                src_v  = local_drive_v / name
                dst_at = z_at_dir / name
                dst_v  = z_v_dir / name
                shutil.copy2(src_at, dst_at)
                synced_fd += 1
                if src_v.exists():
                    shutil.copy2(src_v, dst_v)
            print(f"  [LOCAL FEATURES_DRIVE SYNC] Fast-merged {synced_fd:,} feature tensor pairs into data/preprocessed/features.")

    # 2. Check mounted Google Drive folder for preprocessed.zip (bulk fast extraction)
    if drive_base.exists() and (not any(z_at_dir.glob("*.pt"))):
        zip_path = None
        for root, _, files in os.walk(drive_base):
            for f in files:
                if "preprocessed" in f.lower() and f.endswith(".zip"):
                    zip_path = Path(root) / f
                    break
            if zip_path:
                break

        if zip_path:
            size_mb = zip_path.stat().st_size / 1e6
            print(f"  [AUTO-EXTRACT] Fast-extracting {zip_path.name} ({size_mb:.1f} MB) to local SSD...")
            import shutil
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(REPO_ROOT / "data")
            print("  [AUTO-EXTRACT] Done! Preprocessed feature cache extracted.\n")

    local_at_files = set(p.stem for p in z_at_dir.glob("*.pt"))
    local_v_files  = set(p.stem for p in z_v_dir.glob("*.pt"))
    valid_pairs    = local_at_files.intersection(local_v_files)

    print(f"\n[CACHE CHECK] Preprocessed Features Cache: {len(valid_pairs):,} valid pairs ready on disk.")

    if not valid_pairs and drive_base.exists():
        print("  [AUTO-CHECK] Local features missing. Searching Google Drive for preprocessed.zip...")
        zip_path = None
        for root, _, files in os.walk(drive_base):
            for f in files:
                if "preprocessed" in f.lower() and f.endswith(".zip"):
                    zip_path = Path(root) / f
                    break
            if zip_path:
                break

        if zip_path:
            size_mb = zip_path.stat().st_size / 1e6
            print(f"  [AUTO-EXTRACT] Extracting {zip_path.name} ({size_mb:.1f} MB) to local SSD...")
            import shutil
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                namelist = zf.namelist()
                has_data_prefix = any(name.startswith("data/") for name in namelist)
                has_prep_prefix = any(name.startswith("preprocessed/") for name in namelist)
                
                dest = REPO_ROOT if has_data_prefix else (REPO_ROOT / "data" if has_prep_prefix else REPO_ROOT / "data/preprocessed")
                dest.mkdir(parents=True, exist_ok=True)
                zf.extractall(dest)

            nested = REPO_ROOT / "data/preprocessed/preprocessed"
            if nested.exists() and nested.is_dir():
                for item in nested.iterdir():
                    target = REPO_ROOT / "data/preprocessed" / item.name
                    if not target.exists():
                        shutil.move(str(item), str(target))
            print("  [AUTO-EXTRACT] Done! Preprocessed feature cache extracted.\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSentinel on FakeAVCeleb dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/full/best_phase2_emotion_bilinear.pt")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device",     default=_default_device)
    parser.add_argument("--n_real",     type=int, default=500,
                        help="Number of real clips to sample (default 500)")
    parser.add_argument("--n_fake",     type=int, default=500,
                        help="Number of fake clips to sample (default 500)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for clip sampling")
    parser.add_argument("--no_cache",   action="store_true",
                        help="Ignore feature cache and reprocess all clips")
    parser.add_argument("--no_hard",    action="store_true",
                        help="Disable hard method stratification (uniform random sample)")
    parser.add_argument("--save_csv",   default=None,
                        help="Save per-clip results (clip_id,fake_label,method,type,score,pred) to this CSV path")
    parser.add_argument("--classifier_mode", type=str, default="baseline",
                        choices=["baseline", "mismatch_only", "emotion_bilinear", "bottleneck", "high_dropout"])
    parser.add_argument("--cached_only", action="store_true",
                        help="Evaluate ONLY clips that have pre-extracted .pt feature tensors ready on disk (instant <2s run)")
    parser.add_argument("--force_features", action="store_true",
                        help="Force feature-based forward pass (forward_from_features), bypassing raw video decoding & face detector")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    drive_mode_p2 = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode/best_phase2_bottleneck.pt")
    drive_mode_p1 = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest/bottleneck_mode/best_phase1_bottleneck.pt")
    drive_latest  = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints/latest") / ckpt_path.name
    drive_base    = Path("/content/drive/MyDrive/THESIS_MOTHERFILE/checkpoints") / ckpt_path.name
    
    drive_source = None
    if ckpt_path.name == "best_phase2_bottleneck.pt" and drive_mode_p2.exists():
        drive_source = drive_mode_p2
    elif ckpt_path.name == "best_phase1_bottleneck.pt" and drive_mode_p1.exists():
        drive_source = drive_mode_p1
    elif drive_mode_p2.exists():
        drive_source = drive_mode_p2
    elif drive_latest.exists():
        drive_source = drive_latest
    elif drive_base.exists():
        drive_source = drive_base

    # Only fetch from Drive if local checkpoint is missing or 0-byte corrupted
    if (not ckpt_path.exists() or ckpt_path.stat().st_size < 1000000) and drive_source is not None and drive_source.exists():
        try:
            import shutil
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_source, ckpt_path)
            print(f"  [DRIVE FETCH] Loaded evaluation checkpoint from Drive ({drive_source.relative_to(Path('/content/drive/MyDrive'))}) -> {ckpt_path}")
        except Exception as e:
            print(f"  [DRIVE FETCH WARNING] Failed to copy checkpoint from Drive: {e}")

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return
    cached_manifest = REPO_ROOT / "data/preprocessed/fakeavceleb_cached_manifest.csv"
    if not META_CSV.exists() and not cached_manifest.exists():
        print(f"ERROR: Neither FakeAVCeleb metadata ({META_CSV}) nor cached manifest ({cached_manifest}) found.")
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
    has_backbones = any(
        k.startswith("vit.") or k.startswith("_vit.") or
        k.startswith("wav2vec.") or k.startswith("wav2vec2.") or k.startswith("_wav2vec.") or
        k.startswith("bert.") or k.startswith("_bert.")
        for k in ckpt["model_state"].keys()
    )
    if has_backbones and not args.force_features and not args.cached_only:
        print("  [HuggingFace] Detected Stage 2 backbone weights in checkpoint.")
        print("  [HuggingFace] Loading ViT, Wav2Vec2, and BERT backbones into GPU VRAM...")
        model.load_backbones()
        model.to(args.device)
        print("  [HuggingFace] Backbone weights loaded successfully.")
    else:
        # We need the feature cache for Phase 1 / offline cached evaluation
        ensure_preprocessed_features()
        
    model.load_state_dict(ckpt["model_state"], strict=False)
    model._has_cross_attn = any(k.startswith("cross_attn_") for k in ckpt["model_state"].keys())
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', '?')}")

    # ── Load preprocessing pipeline ────────────────────────────────────────────
    _section("Initializing preprocessing pipeline")
    print("  [Pipeline] Connecting feature cache and checking preprocessed feature tensors...")
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
    if args.cached_only:
        clips = [c for c in clips if pipeline.is_cached(c["clip_id"])]
        print(f"  [CACHED ONLY] Filtered evaluation set to {len(clips):,} cached feature clips ready on disk.")

    # ── Run inference ──────────────────────────────────────────────────────────
    _section("Running inference (MP4 -> features -> P(fake))")
    results = run_inference(
        clips, pipeline, model, args.device, args.no_cache,
        force_features=args.force_features, cached_only=args.cached_only
    )
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

    print(f"  Clips evaluated        : {m['total']}")
    print(f"  --- Standard Metrics (tau = 0.50) ---")
    print(f"  Accuracy (0.5)         : {m['acc_50']:.4f}")
    print(f"  Balanced Accuracy (0.5): {m['bal_acc_50']:.4f}")
    print(f"  Precision (0.5)        : {m['prec_50']:.4f}")
    print(f"  Recall (0.5)           : {m['rec_50']:.4f}")
    print(f"  Specificity (0.5)      : {m['spec_50']:.4f}")
    print(f"  F1-Score (0.5)         : {m['f1_50']:.4f}")
    print(f"  MCC (0.5)              : {m['mcc_50']:.4f}")
    print(f"  TP/FP/FN/TN (0.5)      : {m['tp_50']}/{m['fp_50']}/{m['fn_50']}/{m['tn_50']}")

    if m.get("best_thresh") is not None:
        print(f"\n  --- Calibrated Threshold Sweeps ---")
        print(f"  Optimal F1 Threshold   : {m['best_thresh']:.2f}  (F1={m['best_cal_f1']:.4f}, Acc={m['best_cal_acc']:.4f})")
        print(f"  Youden's J Threshold   : {m['youden_thresh']:.2f}  (J={m['best_youden_j']:.4f}, BalAcc={m['youden_bal_acc']:.4f}, Sens={m['youden_sens']:.4f}, Spec={m['youden_spec']:.4f})")
        print(f"  Youden's TP/FP/FN/TN   : {m['youden_tp']}/{m['youden_fp']}/{m['youden_fn']}/{m['youden_tn']}")
        if m.get("zero_fp_thresh") is not None:
            print(f"  Zero-FP Operating Pt   : {m['zero_fp_thresh']:.2f}  (Specificity=100.0%, FP=0, Recall={m['zero_fp_sens']:.4f})")

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
