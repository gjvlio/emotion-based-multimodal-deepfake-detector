"""
dataset.py — PyTorch Dataset for deepfake detection training.

Loads cached (Z_at, Z_v) feature tensors + labels from all 4 tracks + real sources.

Emotion label space (6 classes):
    0 = neutral   1 = happy   2 = sad
    3 = angry     4 = fear    5 = disgust

Visual emotion assignment per track (per sys_archi_memory.md):
    Track 1:  visual_emotion = source face emotion  (video_emotion column)
    Track 2:  visual_emotion = target audio emotion (audio_emotion column — lips corrected)
    Track 3:  visual_emotion = target audio emotion (audio_emotion column — face synthesised)
    Track 4:  visual_emotion = video speaker emotion (video_emotion column)
    Real:     visual_emotion = audio_emotion (same annotation)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# ── Emotion label maps ────────────────────────────────────────────────────────

EMOTION_TO_IDX: Dict[str, int] = {
    # CREMA-D codes
    "NEU": 0, "neutral": 0,
    "HAP": 1, "happy": 1,    "joy": 1,
    "SAD": 2, "sad": 2,      "sadness": 2,
    "ANG": 3, "angry": 3,    "anger": 3,
    # MELD extras
    "FEA": 4, "fear": 4,     "fearful": 4,
    "DIS": 5, "disgust": 5,  "disgusted": 5,
    # MELD surprise → neutral (spec: "drop surprise OR map to nearest")
    "surprise": 0,
    # CMU-MOSEI variants
    "frustrated": 3,
    "excited": 1,
}

UNKNOWN_EMOTION = -1   # masked in CrossEntropyLoss


def _emo(code: Optional[str]) -> int:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return UNKNOWN_EMOTION
    return EMOTION_TO_IDX.get(str(code).strip(), UNKNOWN_EMOTION)


def _speaker_from_row(row: dict) -> str:
    """Extract speaker ID from a metadata row, trying common column names."""
    for col in ("actor_id", "video_speaker", "speaker", "speaker_id"):
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            s = str(val).strip()
            if s:
                return s
    return "unknown"


# ── Record builder ────────────────────────────────────────────────────────────

def _build_records(
    meta_csv: str | Path,
    source_pipeline: str,
    preprocessed_dir: Path,
) -> List[dict]:
    """Parse one metadata CSV into a flat list of sample records."""
    p = Path(meta_csv)
    if not p.exists():
        log.warning(f"Metadata CSV not found, skipping: {p}")
        return []
    df = pd.read_csv(p)
    records = []
    for _, row in df.iterrows():
        clip_id = str(row.get("output_stem") or row.get("clip_id") or "")
        if not clip_id:
            continue

        z_at_path = preprocessed_dir / "features" / "z_at" / f"{clip_id}.pt"
        z_v_path  = preprocessed_dir / "features" / "z_v"  / f"{clip_id}.pt"
        if not z_at_path.exists() or not z_v_path.exists():
            continue  # not yet preprocessed — skip silently

        fake_label = int(row.get("label", 1))
        aud_emo    = _emo(row.get("audio_emotion"))
        vid_emo_raw = row.get("video_emotion")

        # Visual emotion per spec
        if source_pipeline in ("track2", "track3"):
            vis_emo = aud_emo  # face was synthesised to match audio
        else:
            vis_emo = _emo(vid_emo_raw)

        records.append({
            "clip_id":           clip_id,
            "z_at_path":         str(z_at_path),
            "z_v_path":          str(z_v_path),
            "fake_label":        fake_label,
            "audio_emotion":     aud_emo,
            "visual_emotion":    vis_emo,
            "source_pipeline":   source_pipeline,
            "speaker_id":        _speaker_from_row(row.to_dict()),
        })
    return records


def _build_real_records(
    real_csv: str | Path,
    source_name: str,
    preprocessed_dir: Path,
    clip_id_col: str = "clip_id",
    emotion_col: str = "emotion",
    speaker_col: str = "speaker",
) -> List[dict]:
    p = Path(real_csv)
    if not p.exists():
        log.warning(f"Real CSV not found, skipping: {p}")
        return []
    df = pd.read_csv(p)
    records = []
    for _, row in df.iterrows():
        clip_id = str(row.get(clip_id_col, ""))
        if not clip_id:
            continue
        z_at_path = preprocessed_dir / "features" / "z_at" / f"{clip_id}.pt"
        z_v_path  = preprocessed_dir / "features" / "z_v"  / f"{clip_id}.pt"
        if not z_at_path.exists() or not z_v_path.exists():
            continue
        emo = _emo(row.get(emotion_col))

        # Speaker: try named column first, fall back to first segment of clip_id
        row_dict = row.to_dict()
        spk = _speaker_from_row({**row_dict, "speaker": row_dict.get(speaker_col, "")})
        if spk == "unknown":
            spk = clip_id.split("_")[0]

        records.append({
            "clip_id":           clip_id,
            "z_at_path":         str(z_at_path),
            "z_v_path":          str(z_v_path),
            "fake_label":        0,
            "audio_emotion":     emo,
            "visual_emotion":    emo,
            "source_pipeline":   source_name,
            "speaker_id":        spk,
        })
    return records


# ── Main Dataset ──────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Loads cached (Z_at, Z_v) tensors from all tracks + real sources.

    Each item is a dict:
        z_at            (1536,) float32
        z_v             (768,)  float32
        fake_label      scalar int  0/1
        audio_emotion   scalar int  0-5 or -1
        visual_emotion  scalar int  0-5 or -1
        source_pipeline str
        clip_id         str
        speaker_id      str
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        track1_meta:  Optional[str | Path] = None,
        track2_meta:  Optional[str | Path] = None,
        track3_meta:  Optional[str | Path] = None,
        track4_meta:  Optional[str | Path] = None,
        meld_real_csv: Optional[str | Path] = None,
        mosei_real_csv: Optional[str | Path] = None,
        indices: Optional[List[int]] = None,   # for train/val/test splits
    ):
        self.preprocessed_dir = Path(preprocessed_dir)
        all_records: List[dict] = []

        # Fake samples
        for csv, name in [
            (track1_meta,  "track1"),
            (track2_meta,  "track2"),
            (track3_meta,  "track3"),
            (track4_meta,  "track4"),
        ]:
            if csv:
                all_records += _build_records(csv, name, self.preprocessed_dir)

        # Real samples — MELD
        if meld_real_csv:
            all_records += _build_real_records(
                meld_real_csv, "meld_real", self.preprocessed_dir,
                clip_id_col="clip_id", emotion_col="emotion", speaker_col="speaker",
            )

        # Real samples — CMU-MOSEI
        if mosei_real_csv:
            all_records += _build_real_records(
                mosei_real_csv, "mosei_real", self.preprocessed_dir,
                clip_id_col="clip_id", emotion_col="emotion", speaker_col="speaker",
            )

        self._records = all_records if indices is None else [all_records[i] for i in indices]
        log.info(f"Dataset: {len(self._records)} samples loaded.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        z_at = torch.load(r["z_at_path"], weights_only=True).float()
        z_v  = torch.load(r["z_v_path"],  weights_only=True).float()
        return {
            "z_at":            z_at,
            "z_v":             z_v,
            "fake_label":      torch.tensor(r["fake_label"],     dtype=torch.long),
            "audio_emotion":   torch.tensor(r["audio_emotion"],  dtype=torch.long),
            "visual_emotion":  torch.tensor(r["visual_emotion"], dtype=torch.long),
            "source_pipeline": r["source_pipeline"],
            "clip_id":         r["clip_id"],
            "speaker_id":      r["speaker_id"],
        }

    # ── Speaker-independent stratified split ──────────────────────────────────

    @classmethod
    def stratified_split(
        cls,
        preprocessed_dir: str | Path,
        train_ratio: float = 0.80,
        val_ratio:   float = 0.10,
        seed:        int   = 42,
        **kwargs,
    ) -> Tuple["DeepfakeDataset", "DeepfakeDataset", "DeepfakeDataset"]:
        """
        Build the full dataset, then split by speaker to prevent speaker leakage.
        Speakers are assigned to train/val/test as whole groups — no speaker
        appears in more than one split. Proportions apply to the number of
        speakers, not clips (clip counts may vary slightly from ratios).
        Returns (train_ds, val_ds, test_ds).
        """
        full = cls(preprocessed_dir=preprocessed_dir, **kwargs)
        n = len(full._records)
        if n == 0:
            raise ValueError("Dataset is empty — run preprocess_all.py first.")

        rng = np.random.default_rng(seed)

        # Group record indices by speaker_id
        speaker_to_indices: dict[str, list] = defaultdict(list)
        for i, r in enumerate(full._records):
            speaker_to_indices[r["speaker_id"]].append(i)

        speakers = list(speaker_to_indices.keys())
        rng.shuffle(speakers)

        n_spk   = len(speakers)
        n_train = int(n_spk * train_ratio)
        n_val   = int(n_spk * val_ratio)

        train_speakers = set(speakers[:n_train])
        val_speakers   = set(speakers[n_train:n_train + n_val])
        # remaining speakers → test

        train_idx, val_idx, test_idx = [], [], []
        for spk, idxs in speaker_to_indices.items():
            if spk in train_speakers:
                train_idx.extend(idxs)
            elif spk in val_speakers:
                val_idx.extend(idxs)
            else:
                test_idx.extend(idxs)

        log.info(
            f"Speaker-independent split — "
            f"speakers: {n_train}/{n_val}/{n_spk - n_train - n_val} "
            f"(train/val/test) | "
            f"clips: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
        )

        def _subset(indices):
            ds = cls.__new__(cls)
            ds.preprocessed_dir = full.preprocessed_dir
            ds._records = [full._records[i] for i in indices]
            return ds

        return _subset(train_idx), _subset(val_idx), _subset(test_idx)
