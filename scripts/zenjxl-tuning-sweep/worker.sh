#!/usr/bin/env bash
# W44-212 fleet worker: process one chunk of cell specs.
#
# Inputs:
#   $1 — sweep id
#   $2 — local path to chunks-in-flight/<id>.json (NDJSON: one cell per line)
#   $3 — worker id (host-pid)
#
# For each cell:
#   1. Pull source image from R2 (cached in /corpus/)
#   2. Pull params blob from R2 (cached in /sweep-state/params/) if specified
#   3. Run zenjxl-tuning-runner --cell <json> --output /tmp/c-<chunk_claim>.parquet
#   4. Upload the Parquet to s3://<bucket>/<sweep>/cells/<worker>-<chunk_claim>.parquet
#   5. Log JSON summary line to s3://<bucket>/<sweep>/logs/<worker>-<chunk>.ndjson
set -euo pipefail

SWEEP_ID="$1"
CHUNK_FILE="$2"
WORKER_ID="$3"

SWEEP_BUCKET="${W44_212_SWEEP_BUCKET:-zen-tuning-ephemeral}"
CORPUS_BUCKET="${W44_212_CORPUS_BUCKET:-zen-corpus}"
OUTPUT_PREFIX="${W44_212_OUTPUT_PREFIX:-cells}"

mkdir -p /corpus /sweep-state/params /sweep-output

CHUNK_ID="$(basename "$CHUNK_FILE" .json)"
SUMMARY_LOG=/sweep-output/${CHUNK_ID}.ndjson
: > "$SUMMARY_LOG"

CELL_COUNT=0
OK_COUNT=0
FAIL_COUNT=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    CELL_COUNT=$((CELL_COUNT + 1))

    # Parse out the image_path + chunk_claim_id + (optional) params_blob_path
    IMAGE_PATH=$(echo "$line" | jq -r '.image_path')
    IMAGE_SHA=$(echo "$line" | jq -r '.image_sha256 // empty')
    CHUNK_CLAIM=$(echo "$line" | jq -r '.chunk_claim_id')
    PARAMS_BLOB=$(echo "$line" | jq -r '.params_blob_path // empty')

    # Pull source if not cached. We use the image_path as the local
    # path AND assume image_path is also the R2 key under the corpus
    # bucket (e.g. "cid22/1418519.png"). Sweeps that need a different
    # layout should set image_path to the desired LOCAL path and
    # store the R2 key elsewhere.
    if [[ ! -f "$IMAGE_PATH" ]]; then
        # Derive R2 key from the absolute path (strip leading slash).
        R2_KEY="${IMAGE_PATH#/}"
        # Try corpus bucket; fall back to the sweep bucket
        # (per-sweep image staging).
        if ! s5cmd cp "s3://$CORPUS_BUCKET/$R2_KEY" "$IMAGE_PATH" 2>/dev/null; then
            s5cmd cp "s3://$SWEEP_BUCKET/$SWEEP_ID/corpus/$R2_KEY" "$IMAGE_PATH" 2>&1 || {
                echo "[worker] FAIL: cannot fetch image $IMAGE_PATH" >&2
                echo "{\"chunk_claim_id\":\"$CHUNK_CLAIM\",\"status\":\"err\",\"error\":\"image_fetch_failed\"}" >> "$SUMMARY_LOG"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            }
        fi
    fi

    # Pull params blob if specified and not cached.
    if [[ -n "$PARAMS_BLOB" && ! -f "$PARAMS_BLOB" ]]; then
        R2_KEY="${PARAMS_BLOB#/}"
        s5cmd cp "s3://$SWEEP_BUCKET/$SWEEP_ID/params/$(basename "$PARAMS_BLOB")" "$PARAMS_BLOB" 2>&1 || \
            s5cmd cp "s3://$SWEEP_BUCKET/$SWEEP_ID/$R2_KEY" "$PARAMS_BLOB" 2>&1 || {
                echo "[worker] FAIL: cannot fetch params $PARAMS_BLOB" >&2
                echo "{\"chunk_claim_id\":\"$CHUNK_CLAIM\",\"status\":\"err\",\"error\":\"params_fetch_failed\"}" >> "$SUMMARY_LOG"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            }
    fi

    # Run the cell.
    OUT_PARQUET=/sweep-output/${WORKER_ID}-${CHUNK_CLAIM}.parquet
    RESULT=$(zenjxl-tuning-runner --cell "$line" --output "$OUT_PARQUET" 2>>"$SUMMARY_LOG" || true)
    STATUS=$(echo "$RESULT" | jq -r '.status // "unknown"')
    echo "$RESULT" >> "$SUMMARY_LOG"

    if [[ "$STATUS" == "ok" && -f "$OUT_PARQUET" ]]; then
        # Upload to R2. Atomic single-object PUT.
        s5cmd cp "$OUT_PARQUET" \
            "s3://$SWEEP_BUCKET/$SWEEP_ID/$OUTPUT_PREFIX/${WORKER_ID}-${CHUNK_CLAIM}.parquet" \
            >>"$SUMMARY_LOG" 2>&1 || {
                echo "[worker] WARN: upload failed for $CHUNK_CLAIM" >&2
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            }
        rm -f "$OUT_PARQUET"
        OK_COUNT=$((OK_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done < "$CHUNK_FILE"

echo "[worker] chunk=$CHUNK_ID cells=$CELL_COUNT ok=$OK_COUNT fail=$FAIL_COUNT"

# Upload the chunk summary log
s5cmd cp "$SUMMARY_LOG" \
    "s3://$SWEEP_BUCKET/$SWEEP_ID/logs/${WORKER_ID}-${CHUNK_ID}.ndjson" \
    2>&1 || echo "[worker] WARN: summary upload failed"
