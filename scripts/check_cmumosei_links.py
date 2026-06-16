"""
check_cmumosei_links.py — Validate CMU-MOSEI YouTube links via oEmbed endpoint.

No API key needed. Parallelized with threads.
HTTP 200 / 401 (age-restricted) = active (yt-dlp can handle both).
Everything else = dead/private/unavailable.

Usage:
    python scripts/check_cmumosei_links.py [--workers 32] [--timeout 10]

Reads:  data/raw/CMU-MOSEI/yt_ids.txt
Writes:
    data/raw/CMU-MOSEI/yt_ids_active.txt    -- ready for yt-dlp --batch-file
    data/raw/CMU-MOSEI/yt_ids_dead.txt      -- unavailable/private/deleted
    data/raw/CMU-MOSEI/yt_ids_check.csv     -- full status log
"""

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
IDS_FILE  = REPO_ROOT / "data/raw/CMU-MOSEI/yt_ids.txt"
ACTIVE    = REPO_ROOT / "data/raw/CMU-MOSEI/yt_ids_active.txt"
DEAD      = REPO_ROOT / "data/raw/CMU-MOSEI/yt_ids_dead.txt"
LOG       = REPO_ROOT / "data/raw/CMU-MOSEI/yt_ids_check.csv"

OEMBED = "https://www.youtube.com/oembed"


def check_url(url: str, timeout: int) -> tuple[str, int, str]:
    """Return (url, http_status, status_label)."""
    vid = url.split("v=")[-1]
    try:
        r = requests.get(
            OEMBED,
            params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"},
            timeout=timeout,
            allow_redirects=True,
        )
        code = r.status_code
        if code == 200:
            label = "active"
        elif code == 401:
            label = "age_restricted"   # yt-dlp can still download
        elif code == 404:
            label = "unavailable"
        elif code == 403:
            label = "private"
        else:
            label = f"http_{code}"
        return url, code, label
    except requests.exceptions.Timeout:
        return url, -1, "timeout"
    except Exception as e:
        return url, -2, f"error:{type(e).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32, help="Parallel threads")
    parser.add_argument("--timeout", type=int, default=10, help="Per-request timeout (s)")
    args = parser.parse_args()

    urls = [line.strip() for line in IDS_FILE.read_text().splitlines() if line.strip()]
    print(f"Checking {len(urls)} URLs with {args.workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_url, u, args.timeout): u for u in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="url"):
            results.append(fut.result())

    active_labels = {"active", "age_restricted"}
    active = [r for r in results if r[2] in active_labels]
    dead   = [r for r in results if r[2] not in active_labels]

    ACTIVE.write_text("\n".join(r[0] for r in active) + "\n")
    DEAD.write_text("\n".join(r[0] for r in dead) + "\n")

    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "http_status", "label"])
        w.writerows(sorted(results, key=lambda x: x[2]))

    from collections import Counter
    counts = Counter(r[2] for r in results)
    print(f"\nResults:")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {n}")
    print(f"\nActive (downloadable): {len(active)}")
    print(f"Dead/unavailable:      {len(dead)}")
    print(f"\nWrote:")
    print(f"  {ACTIVE}")
    print(f"  {DEAD}")
    print(f"  {LOG}")


if __name__ == "__main__":
    main()
