"""
validate_and_retry.py
=====================
Post-generation validation for Tracks 1, 2, and 3 (part 1 = first 930 clips).

For each track:
  1. Load pairs CSV (Track 3 limited to first 930 for part-1 validation)
  2. Identify every stem not in metadata.csv (not generated)
  3. Identify every stem whose MP4 is missing or < MIN_BYTES (corrupt/empty)
  4. Delete bad MP4s
  5. Write a retry CSV for each track that has gaps
  6. Print a strict count summary — pass only if done == expected

Usage:
    python scripts/validate_and_retry.py
    python scripts/validate_and_retry.py --delete_bad     # actually delete bad MP4s
    python scripts/validate_and_retry.py --write_retry    # write retry CSVs

Retry CSVs go to data/processed/track1_manifests/trackN_retry.csv.
Feed them into the generator the same way as normal pairs CSVs:

    python src/track1/track1_generate.py \\
        --pairs_csv data/processed/track1_manifests/track1_retry.csv ...

    python src/track2/track2_generate.py \\
        --pairs_csv data/processed/track1_manifests/track2_retry.csv ...

    python src/track3/track3_generate.py \\
        --pairs_csv data/processed/track1_manifests/track3_retry.csv ...
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]

TRACKS = {
    1: {
        "pairs_csv":  REPO_ROOT / "data/processed/track1_manifests/track1_pairs.csv",
        "meta_csv":   REPO_ROOT / "data/synthetic/track1_fakes/metadata.csv",
        "failed_csv": REPO_ROOT / "data/synthetic/track1_fakes/failed.csv",
        "retry_csv":  REPO_ROOT / "data/processed/track1_manifests/track1_retry.csv",
        "max_clips":  None,  # all
    },
    2: {
        "pairs_csv":  REPO_ROOT / "data/processed/track1_manifests/track2_pairs.csv",
        "meta_csv":   REPO_ROOT / "data/synthetic/track2_fakes/metadata.csv",
        "failed_csv": REPO_ROOT / "data/synthetic/track2_fakes/failed.csv",
        "retry_csv":  REPO_ROOT / "data/processed/track1_manifests/track2_retry.csv",
        "max_clips":  None,
    },
    3: {
        "pairs_csv":  REPO_ROOT / "data/processed/track1_manifests/track3_pairs.csv",
        "meta_csv":   REPO_ROOT / "data/synthetic/track3_fakes/metadata.csv",
        "failed_csv": REPO_ROOT / "data/synthetic/track3_fakes/failed.csv",
        "retry_csv":  REPO_ROOT / "data/processed/track1_manifests/track3_retry.csv",
        "max_clips":  930,   # part 1 only
    },
}

# Per-track minimum MP4 size (bytes). SadTalker 256px outputs are legitimately
# small (~25-50 KB); Wav2Lip and ffmpeg-mux outputs are larger (~50-170 KB).
MIN_BYTES = {
    1: 20_000,   # ffmpeg mux — should be large, but be conservative
    2: 20_000,   # Wav2Lip — observed min ~50 KB; 20 KB catches corrupt/empty
    3:  5_000,   # SadTalker 256px — observed range 25-50 KB; 5 KB catches empty
}


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_track(track_id: int, cfg: dict, delete_bad: bool, write_retry: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"TRACK {track_id}")
    print(f"{'='*60}")

    pairs = pd.read_csv(cfg["pairs_csv"])
    if cfg["max_clips"]:
        pairs = pairs.head(cfg["max_clips"])
    expected = len(pairs)
    print(f"  Expected clips : {expected}")

    meta = pd.read_csv(cfg["meta_csv"]) if cfg["meta_csv"].exists() else pd.DataFrame()
    print(f"  In metadata    : {len(meta)}")

    done_stems = set(meta["output_stem"]) if not meta.empty else set()
    all_stems  = set(pairs["output_stem"])

    # 1. Stems missing from metadata entirely
    not_in_meta = all_stems - done_stems
    print(f"  Not in meta    : {len(not_in_meta)}")

    # 2. Stems in metadata but MP4 missing or corrupt
    bad_mp4: list[str] = []
    if not meta.empty:
        for _, row in meta.iterrows():
            if row["output_stem"] not in all_stems:
                continue
            p = Path(row["output_path"])
            if not p.exists():
                bad_mp4.append(row["output_stem"])
                print(f"    MISSING MP4 : {p.name}")
            elif p.stat().st_size < MIN_BYTES[track_id]:
                bad_mp4.append(row["output_stem"])
                print(f"    TINY MP4    : {p.name}  ({p.stat().st_size} bytes)")

    print(f"  Bad MP4s       : {len(bad_mp4)}")

    # 3. Total needing regeneration
    needs_regen = not_in_meta | set(bad_mp4)
    print(f"  Need regen     : {len(needs_regen)}")

    # 4. Delete bad MP4s
    if bad_mp4 and delete_bad:
        bad_stems = set(bad_mp4)
        for _, row in meta.iterrows():
            if row["output_stem"] in bad_stems:
                p = Path(row["output_path"])
                if p.exists():
                    p.unlink()
                    print(f"    DELETED     : {p.name}")

    # 5. Write retry CSV
    if needs_regen and write_retry:
        retry_df = pairs[pairs["output_stem"].isin(needs_regen)].copy()
        retry_df.to_csv(cfg["retry_csv"], index=False)
        print(f"  Retry CSV      : {cfg['retry_csv']}  ({len(retry_df)} rows)")

    # 6. Strict pass/fail
    actually_done = expected - len(needs_regen)
    passed = len(needs_regen) == 0
    status = "PASS" if passed else f"FAIL  ({len(needs_regen)} clips missing)"
    print(f"  Done / Expected: {actually_done} / {expected}  ->  {status}")

    if needs_regen:
        print(f"\n  Stems needing regen:")
        for s in sorted(needs_regen):
            print(f"    {s}")

    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Track 1/2/3-part1 generation completeness"
    )
    parser.add_argument("--delete_bad",  action="store_true",
                        help="Delete bad/corrupt MP4s found during validation")
    parser.add_argument("--write_retry", action="store_true",
                        help="Write retry CSVs for clips needing regeneration")
    parser.add_argument("--tracks", nargs="+", type=int, default=[1, 2, 3],
                        help="Which tracks to validate (default: 1 2 3)")
    args = parser.parse_args()

    results = {}
    for tid in args.tracks:
        if tid not in TRACKS:
            print(f"Unknown track {tid}, skipping.")
            continue
        results[tid] = validate_track(tid, TRACKS[tid], args.delete_bad, args.write_retry)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for tid, passed in results.items():
        label = "PASS" if passed else "FAIL"
        print(f"  Track {tid}: {label}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tracks complete and consistent.")
    else:
        print("\nAction needed -- run with --delete_bad --write_retry then rerun generators.")
        sys.exit(1)


if __name__ == "__main__":
    main()
