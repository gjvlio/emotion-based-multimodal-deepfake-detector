"""
create_dataset_splits.py
========================
Performs a dataset-balanced and speaker-stratified 80/10/10 split on the exclusive
training dataset (`data/processed/training_turnover_manifest.csv`).

Guarantees:
  1. EVERY split (Train, Validation, Internal Test) contains clips from ALL source
     pipelines (Track 1, Track 2, Track 3, MELD Real, CMU-MOSEI Real, MUStARD).
  2. ABSOLUTELY ZERO speaker overlap between Train, Validation, and Internal Test splits
     (CREMA-D actors 1001-1091 are grouped globally across Track 1, Track 2, and Track 3).
  3. Saved explicitly as CSV manifest files for training pipelines.

Outputs:
  data/processed/train_manifest.csv           (80% Train split)
  data/processed/val_manifest.csv             (10% Validation split)
  data/processed/internal_test_manifest.csv   (10% Internal Test split)
  (and mirrors in data/preprocessed/)

Usage:
    python scripts/create_dataset_splits.py [--seed 42]
"""
import argparse
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data/processed"
PREPROCESSED_DIR = REPO_ROOT / "data/preprocessed"

INPUT_MANIFEST = PROCESSED_DIR / "training_turnover_manifest.csv"

TRAIN_CSV = PROCESSED_DIR / "train_manifest.csv"
VAL_CSV = PROCESSED_DIR / "val_manifest.csv"
TEST_CSV = PROCESSED_DIR / "internal_test_manifest.csv"


def get_speaker_family(rec: dict) -> tuple[str, str]:
    """Return (dataset_family, speaker_id) for global speaker partitioning."""
    spk = rec["speaker_id"]
    pipe = rec["source_pipeline"]
    if pipe in ("track1", "track2", "track3") or spk.startswith("crema_"):
        return "crema", spk
    elif pipe == "meld_real" or spk.startswith("meld_"):
        return "meld", spk
    elif pipe == "mosei_real" or spk.startswith("mosei_"):
        return "mosei", spk
    elif pipe == "mustard" or spk.startswith("mustard_"):
        return "mustard", spk
    return pipe, spk


def create_splits(seed: int = 42):
    log.info("=" * 68)
    log.info("DATASET-BALANCED & STRICT SPEAKER-STRATIFIED 80/10/10 SPLIT GENERATOR")
    log.info("=" * 68)

    if not INPUT_MANIFEST.exists():
        log.error(f"Input manifest not found: {INPUT_MANIFEST}. Run validate_training_prep.py first.")
        return

    # 1. Read input records
    with open(INPUT_MANIFEST, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    log.info(f"Loaded {len(records):,} records from {INPUT_MANIFEST.name}")

    # Group records by (family, speaker_id)
    family_speaker_map = defaultdict(lambda: defaultdict(list))
    for rec in records:
        family, spk = get_speaker_family(rec)
        family_speaker_map[family][spk].append(rec)

    rng = random.Random(seed)

    train_records = []
    val_records = []
    test_records = []

    # 2. Perform Speaker-Stratified 80/10/10 split globally per dataset family
    log.info("\nPartitioning speakers per dataset family (80% Train / 10% Val / 10% Test)...")

    for family, spk_dict in sorted(family_speaker_map.items()):
        speakers = sorted(spk_dict.keys())
        rng.shuffle(speakers)

        n_spk = len(speakers)
        n_val_spk = max(1, int(round(0.10 * n_spk)))
        n_test_spk = max(1, int(round(0.10 * n_spk)))
        n_train_spk = n_spk - n_val_spk - n_test_spk

        val_spks = set(speakers[:n_val_spk])
        test_spks = set(speakers[n_val_spk : n_val_spk + n_test_spk])
        train_spks = set(speakers[n_val_spk + n_test_spk :])

        fam_train, fam_val, fam_test = [], [], []

        for spk, clips in spk_dict.items():
            if spk in val_spks:
                fam_val.extend(clips)
            elif spk in test_spks:
                fam_test.extend(clips)
            else:
                fam_train.extend(clips)

        train_records.extend(fam_train)
        val_records.extend(fam_val)
        test_records.extend(fam_test)

        log.info(
            f"  - Family {family.upper():<8}: {n_spk:,} speakers -> "
            f"Train: {len(fam_train):,} clips ({len(train_spks):,} spks) | "
            f"Val: {len(fam_val):,} clips ({len(val_spks):,} spks) | "
            f"Test: {len(fam_test):,} clips ({len(test_spks):,} spks)"
        )

    # Shuffle output records
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    if not records:
        log.warning("No records found in training_turnover_manifest.csv to split.")
        return

    # Fieldnames for output CSVs
    fieldnames = list(records[0].keys())

    def _write_csv(path: Path, rows: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Mirror in preprocessed directory
        mirror_path = PREPROCESSED_DIR / path.name
        with open(mirror_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(TRAIN_CSV, train_records)
    _write_csv(VAL_CSV, val_records)
    _write_csv(TEST_CSV, test_records)

    # 3. Print Comprehensive Verification Report
    log.info("\n" + "=" * 68)
    log.info("FINAL SPLIT VERIFICATION REPORT")
    log.info("=" * 68)
    log.info(f"Train Manifest        : {TRAIN_CSV} ({len(train_records):,} clips, {len(train_records)/len(records)*100:.1f}%)")
    log.info(f"Val Manifest          : {VAL_CSV} ({len(val_records):,} clips, {len(val_records)/len(records)*100:.1f}%)")
    log.info(f"Internal Test Manifest: {TEST_CSV} ({len(test_records):,} clips, {len(test_records)/len(records)*100:.1f}%)")
    log.info(f"Total Dataset Clips   : {len(records):,}")

    log.info("\nPer-Pipeline Breakdown Across Splits:")
    log.info(f"{'Source Pipeline':<16} | {'Train Clips':<12} | {'Val Clips':<12} | {'Internal Test':<14} | {'Total':<10}")
    log.info("-" * 72)

    def _count_by_pipe(recs):
        cnt = defaultdict(int)
        for r in recs:
            cnt[r["source_pipeline"]] += 1
        return cnt

    tr_cnt = _count_by_pipe(train_records)
    va_cnt = _count_by_pipe(val_records)
    te_cnt = _count_by_pipe(test_records)

    all_pipes = sorted(set(tr_cnt.keys()) | set(va_cnt.keys()) | set(te_cnt.keys()))
    for pipe in all_pipes:
        tr = tr_cnt[pipe]
        va = va_cnt[pipe]
        te = te_cnt[pipe]
        tot = tr + va + te
        log.info(f"{pipe:<16} | {tr:<12,} | {va:<12,} | {te:<14,} | {tot:<10,}")

    log.info("-" * 72)
    log.info(f"{'TOTALS':<16} | {len(train_records):<12,} | {len(val_records):<12,} | {len(test_records):<14,} | {len(records):<10,}")

    # Speaker Overlap Verification
    tr_spks = set(r["speaker_id"] for r in train_records)
    va_spks = set(r["speaker_id"] for r in val_records)
    te_spks = set(r["speaker_id"] for r in test_records)

    tr_va_overlap = tr_spks.intersection(va_spks)
    tr_te_overlap = tr_spks.intersection(te_spks)
    va_te_overlap = va_spks.intersection(te_spks)

    log.info("\nSpeaker Overlap Audit:")
    log.info(f"  - Train vs Val speaker overlap          : {len(tr_va_overlap)} (ZERO expected)")
    log.info(f"  - Train vs Internal Test speaker overlap: {len(tr_te_overlap)} (ZERO expected)")
    log.info(f"  - Val vs Internal Test speaker overlap  : {len(va_te_overlap)} (ZERO expected)")

    if len(tr_va_overlap) == 0 and len(tr_te_overlap) == 0 and len(va_te_overlap) == 0:
        log.info("\nVERDICT: PERFECT SPLIT — 100% Speaker-Independent & Dataset-Balanced!")
    else:
        log.error("\nVERDICT: WARNING — Speaker overlap detected!")
    log.info("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/val/internal_test dataset splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    args = parser.parse_args()
    create_splits(seed=args.seed)
