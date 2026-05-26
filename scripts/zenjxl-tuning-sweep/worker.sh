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
#   5. (W44-PHASE4-S1g) Upload the per-cell encoded-JXL + diffmap
#      artifacts that the W44-PHASE4-M1 runner stages under
#      $W44_PHASE4_M1_ARTIFACTS_DIR/{jxl,diffmap}/<sha[0..2]>/<sha>.<ext>
#      to s3://<bucket>/<sweep>/artifacts/{jxl,diffmap}/<sha[0..2]>/<sha>.<ext>
#   6. Log JSON summary line to s3://<bucket>/<sweep>/logs/<worker>-<chunk>.ndjson
#
# Per /home/lilith/.claude/CLAUDE.md §4 ("Always persist encoded
# variants when encoding is expensive — NO EXCEPTIONS", hard rule
# added 2026-05-24 after the W44-PHASE4-S1 incident discarded ~$30 of
# encoded bytes), the three W44_PHASE4_M1_* flags are exported
# default-ON below. Set W44_PHASE4_M1_ARTIFACTS_DISABLE=1 to turn all
# three OFF — escape hatch for cells where bytes-persistence is
# genuinely not desired (integration tests, smoke runs).
set -euo pipefail

# Shared R2 helpers: handles env hydration + endpoint when worker.sh
# is invoked outside the launcher bootstrap (e.g. local-replay runs).
# In the production vast.ai bootstrap path the launcher pre-exports
# AWS_*/S3_ENDPOINT_URL, so zen_r2_init is a no-op there — the
# `_ZEN_R2_LIB_INITIALIZED` guard short-circuits the re-source.
# Lib lives at /usr/local/lib/zen-r2-lib.sh in the production image,
# and next to the script for local invocations.
if [[ -r /usr/local/lib/zen-r2-lib.sh ]]; then
    # shellcheck source=/dev/null
    source /usr/local/lib/zen-r2-lib.sh
else
    # shellcheck source=lib/zen-r2-lib.sh
    source "$(dirname "$0")/lib/zen-r2-lib.sh"
fi
# Best-effort: in production env is fully pre-staged; if zen_r2_init
# fails locally we don't want to block the worker's existing s5cmd
# call sites that rely on the ambient AWS_* env. So tolerate it.
zen_r2_init 2>/dev/null || true

SWEEP_ID="$1"
CHUNK_FILE="$2"
WORKER_ID="$3"

SWEEP_BUCKET="${W44_212_SWEEP_BUCKET:-zen-tuning-ephemeral}"
CORPUS_BUCKET="${W44_212_CORPUS_BUCKET:-zen-corpus}"
OUTPUT_PREFIX="${W44_212_OUTPUT_PREFIX:-cells}"

# ── W44-PHASE4-S1g: M1 artifact-persistence env flags ────────────────
# Per /home/lilith/.claude/CLAUDE.md §4 (ML Data Pipeline Discipline:
# "Always persist encoded variants when encoding is expensive — NO
# EXCEPTIONS"), all three M1 artifact flags default ON. The runner
# stages content-addressed files under $W44_PHASE4_M1_ARTIFACTS_DIR
# (default <output_parquet>/../artifacts/ = /sweep-output/artifacts/);
# the upload step below ships them to R2 under the per-sweep prefix.
#
# Escape hatches:
#   W44_PHASE4_M1_ARTIFACTS_DISABLE=1   turns ALL three OFF in one shot
#   W44_PHASE4_M1_SAVE_ENCODED=0        per-flag override (wins)
#   W44_PHASE4_M1_SAVE_DIFFMAP=0        per-flag override
#   W44_PHASE4_M1_COMPUTE_MULTIMETRIC=0 per-flag override
if [[ "${W44_PHASE4_M1_ARTIFACTS_DISABLE:-0}" == "1" ]]; then
    export W44_PHASE4_M1_SAVE_ENCODED="${W44_PHASE4_M1_SAVE_ENCODED:-0}"
    export W44_PHASE4_M1_SAVE_DIFFMAP="${W44_PHASE4_M1_SAVE_DIFFMAP:-0}"
    export W44_PHASE4_M1_COMPUTE_MULTIMETRIC="${W44_PHASE4_M1_COMPUTE_MULTIMETRIC:-0}"
else
    export W44_PHASE4_M1_SAVE_ENCODED="${W44_PHASE4_M1_SAVE_ENCODED:-1}"
    export W44_PHASE4_M1_SAVE_DIFFMAP="${W44_PHASE4_M1_SAVE_DIFFMAP:-1}"
    export W44_PHASE4_M1_COMPUTE_MULTIMETRIC="${W44_PHASE4_M1_COMPUTE_MULTIMETRIC:-1}"
fi
# Local stage dir for content-addressed artifacts; the runner derives
# the same default (<output_parquet>/../artifacts/) when this is unset,
# so the two views agree.
export W44_PHASE4_M1_ARTIFACTS_DIR="${W44_PHASE4_M1_ARTIFACTS_DIR:-/sweep-output/artifacts}"

ARTIFACTS_DIR="$W44_PHASE4_M1_ARTIFACTS_DIR"
ARTIFACT_R2_PREFIX="s3://$SWEEP_BUCKET/$SWEEP_ID/artifacts"

# ── W44-PHASE4-S1h: image fetch retry helper ─────────────────────────
# Pre-W44-S1h: a single s5cmd cp invocation per source. If R2 returned
# a transient 5xx, ECONNRESET, or DNS hiccup, the cell was instantly
# marked image_fetch_failed with no retry. This was paired with a
# pre-flight gap (4 images never uploaded — see launcher), but even
# AFTER the launcher pre-flight catches that root cause, real-world
# R2 has occasional transient failures (1-2 % of fetches in our
# experience across W44-216..S1). 3 retries with exponential backoff
# (0.5s → 1s → 2s) brings effective fetch reliability to ~99.999 %.
#
# Returns 0 on success, non-zero on final failure. Errors go to stderr
# so the caller's `|| { ... }` block fires correctly on hard failure.
fetch_with_retry() {
    local src="$1"
    local dst="$2"
    local label="${3:-fetch}"
    local max_attempts=3
    local delay=0.5
    local attempt
    for (( attempt = 1; attempt <= max_attempts; attempt++ )); do
        if s5cmd cp "$src" "$dst" 2>/dev/null; then
            return 0
        fi
        if (( attempt < max_attempts )); then
            echo "[worker] WARN: $label attempt ${attempt}/${max_attempts} failed for $src; retrying in ${delay}s" >&2
            sleep "$delay"
            delay=$(echo "$delay * 2" | bc -l)
        fi
    done
    echo "[worker] ERR: $label failed all ${max_attempts} attempts for $src" >&2
    return 1
}

mkdir -p /corpus /sweep-state/params /sweep-output "$ARTIFACTS_DIR"

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
        # W44-PHASE4-S1h: 3-retry exp-backoff on each candidate path.
        # Try corpus bucket; fall back to the sweep bucket
        # (per-sweep image staging).
        if ! fetch_with_retry "s3://$CORPUS_BUCKET/$R2_KEY" "$IMAGE_PATH" "image-corpus"; then
            fetch_with_retry "s3://$SWEEP_BUCKET/$SWEEP_ID/corpus/$R2_KEY" "$IMAGE_PATH" "image-sweep" || {
                echo "[worker] FAIL: cannot fetch image $IMAGE_PATH (both buckets, 3 retries each)" >&2
                echo "{\"chunk_claim_id\":\"$CHUNK_CLAIM\",\"status\":\"err\",\"error\":\"image_fetch_failed\"}" >> "$SUMMARY_LOG"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            }
        fi
    fi

    # Pull params blob if specified and not cached.
    if [[ -n "$PARAMS_BLOB" && ! -f "$PARAMS_BLOB" ]]; then
        R2_KEY="${PARAMS_BLOB#/}"
        # W44-PHASE4-S1h: 3-retry exp-backoff on each candidate path.
        if ! fetch_with_retry "s3://$SWEEP_BUCKET/$SWEEP_ID/params/$(basename "$PARAMS_BLOB")" "$PARAMS_BLOB" "params-flat"; then
            fetch_with_retry "s3://$SWEEP_BUCKET/$SWEEP_ID/$R2_KEY" "$PARAMS_BLOB" "params-fullpath" || {
                echo "[worker] FAIL: cannot fetch params $PARAMS_BLOB (both paths, 3 retries each)" >&2
                echo "{\"chunk_claim_id\":\"$CHUNK_CLAIM\",\"status\":\"err\",\"error\":\"params_fetch_failed\"}" >> "$SUMMARY_LOG"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                continue
            }
        fi
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

# ── W44-PHASE4-S1g: post-chunk artifact upload to R2 ─────────────────
# The runner staged encoded JXL bytes under
#   $ARTIFACTS_DIR/jxl/<sha[0..2]>/<sha>.jxl
# and diffmap blobs under
#   $ARTIFACTS_DIR/diffmap/<sha[0..2]>/<sha>.bin
# Content-addressed names mean re-uploading the same sha is a no-op
# with --no-clobber. s5cmd cp handles the wildcard subtree + parallel
# uploads internally; one invocation per artifact type. After upload
# we rm -rf the local subtree so the pod disk doesn't fill up across
# hundreds of chunks. Failed uploads still leave the local copy gone,
# but the failure is captured in the summary log AND the cell parquet
# (which is uploaded separately above) carries the sha256 / r2_key
# pointer — so any future re-encode can verify the artifact's missing
# and re-stage it.
JXL_COUNT=0
DIFFMAP_COUNT=0
if [[ -d "$ARTIFACTS_DIR/jxl" ]]; then
    # find -name '*.jxl' restricts to staged artifacts (skips .tmp).
    JXL_COUNT=$(find "$ARTIFACTS_DIR/jxl" -type f -name '*.jxl' 2>/dev/null | wc -l)
    if (( JXL_COUNT > 0 )); then
        # Trailing slash on the destination tells s5cmd to preserve the
        # source dir structure (so <sha[0..2]>/<sha>.jxl lands at the
        # same path on R2). --no-clobber skips already-uploaded shas.
        s5cmd cp --no-clobber \
            "$ARTIFACTS_DIR/jxl/*" \
            "$ARTIFACT_R2_PREFIX/jxl/" \
            >>"$SUMMARY_LOG" 2>&1 || \
            echo "[worker] WARN: artifact jxl upload had errors for chunk=$CHUNK_ID (see summary log)" >&2
    fi
fi
if [[ -d "$ARTIFACTS_DIR/diffmap" ]]; then
    DIFFMAP_COUNT=$(find "$ARTIFACTS_DIR/diffmap" -type f -name '*.bin' 2>/dev/null | wc -l)
    if (( DIFFMAP_COUNT > 0 )); then
        s5cmd cp --no-clobber \
            "$ARTIFACTS_DIR/diffmap/*" \
            "$ARTIFACT_R2_PREFIX/diffmap/" \
            >>"$SUMMARY_LOG" 2>&1 || \
            echo "[worker] WARN: artifact diffmap upload had errors for chunk=$CHUNK_ID (see summary log)" >&2
    fi
fi
echo "[upload] chunk=$CHUNK_ID: $JXL_COUNT jxl + $DIFFMAP_COUNT diffmap artifacts uploaded"
# Clean up the local artifact subtree so this pod doesn't run out of
# disk across hundreds of chunks. Re-created by mkdir -p at the top of
# the next worker.sh invocation. The artifact bytes are on R2 (or the
# upload failed and the failure is captured in the summary log).
rm -rf "$ARTIFACTS_DIR/jxl" "$ARTIFACTS_DIR/diffmap"

echo "[worker] chunk=$CHUNK_ID cells=$CELL_COUNT ok=$OK_COUNT fail=$FAIL_COUNT jxl=$JXL_COUNT diffmap=$DIFFMAP_COUNT"

# Upload the chunk summary log
s5cmd cp "$SUMMARY_LOG" \
    "s3://$SWEEP_BUCKET/$SWEEP_ID/logs/${WORKER_ID}-${CHUNK_ID}.ndjson" \
    2>&1 || echo "[worker] WARN: summary upload failed"
