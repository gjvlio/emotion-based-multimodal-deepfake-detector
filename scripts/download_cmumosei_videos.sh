#!/bin/bash
# Download CMU-MOSEI raw videos from YouTube.
# Skips unavailable/private/deleted videos automatically.
# Resume-safe: completed IDs tracked in archive.txt.
# Run from repo root.
#
# Batch mode (25% each, share one archive.txt):
#   bash scripts/download_cmumosei_videos.sh --batch 1   # IDs 1-547
#   bash scripts/download_cmumosei_videos.sh --batch 2   # IDs 548-1094
#   bash scripts/download_cmumosei_videos.sh --batch 3   # IDs 1095-1641
#   bash scripts/download_cmumosei_videos.sh --batch 4   # IDs 1642-2187
#   bash scripts/download_cmumosei_videos.sh             # all IDs

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDS_FILE="$REPO_ROOT/data/raw/CMU-MOSEI/yt_ids_active.txt"
OUT_DIR="$REPO_ROOT/data/raw/CMU-MOSEI/videos"
ARCHIVE="$OUT_DIR/archive.txt"
COOKIES="$REPO_ROOT/cookies.txt"
BATCH_NUM=""

# Parse --batch argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)
            BATCH_NUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"; exit 1
            ;;
    esac
done

mkdir -p "$OUT_DIR"

TOTAL_IDS=$(wc -l < "$IDS_FILE")

# Select batch slice
if [[ -n "$BATCH_NUM" ]]; then
    BATCH_SIZE=$(( (TOTAL_IDS + 3) / 4 ))   # ceil(total/4)
    START=$(( (BATCH_NUM - 1) * BATCH_SIZE + 1 ))
    END=$(( BATCH_NUM * BATCH_SIZE ))
    [[ $END -gt $TOTAL_IDS ]] && END=$TOTAL_IDS

    BATCH_IDS="$OUT_DIR/batch${BATCH_NUM}_ids.txt"
    sed -n "${START},${END}p" "$IDS_FILE" > "$BATCH_IDS"
    ACTIVE_IDS="$BATCH_IDS"
    BATCH_TOTAL=$(wc -l < "$BATCH_IDS")
    echo "=== Batch ${BATCH_NUM}/4 (IDs ${START}-${END}, ${BATCH_TOTAL} total) ==="
else
    ACTIVE_IDS="$IDS_FILE"
    echo "=== CMU-MOSEI YouTube download (all IDs) ==="
fi

DONE=$(wc -l < "$ARCHIVE" 2>/dev/null || echo 0)
ACTIVE_TOTAL=$(wc -l < "$ACTIVE_IDS")
echo "IDs file: $ACTIVE_IDS"
echo "Archive:  $ARCHIVE (shared across all batches)"
echo "Output:   $OUT_DIR"
echo "Total in batch: $ACTIVE_TOTAL | Archive so far: $DONE"
echo ""

COOKIE_ARGS=()
if [[ -f "$COOKIES" ]]; then
    COOKIE_ARGS=(--cookies "$COOKIES")
    echo "Using cookies: $COOKIES"
fi

yt-dlp \
    --batch-file "$ACTIVE_IDS" \
    --output "$OUT_DIR/%(id)s.%(ext)s" \
    --format "bestvideo[height<=720]+bestaudio/best[height<=720]/best" \
    --merge-output-format mp4 \
    --download-archive "$ARCHIVE" \
    --ignore-errors \
    --no-warnings \
    --continue \
    --concurrent-fragments 2 \
    --retries 5 \
    --fragment-retries 5 \
    --sleep-interval 3 \
    --max-sleep-interval 10 \
    --sleep-requests 1 \
    --no-write-thumbnail \
    --progress \
    "${COOKIE_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/download_batch${BATCH_NUM:-all}.log"

echo ""
echo "=== Done. Videos in $OUT_DIR ==="
TOTAL=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l)
DONE_FINAL=$(wc -l < "$ARCHIVE" 2>/dev/null || echo 0)
echo "MP4 files: $TOTAL | Archive entries: $DONE_FINAL"
