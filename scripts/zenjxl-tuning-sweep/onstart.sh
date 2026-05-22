#!/usr/bin/env bash
# W44-212 fleet onstart: hydrate env from /proc/1/environ, verify
# baked tools exist, loop pulling chunks from R2 and dispatching to
# worker.sh per cell.
#
# Mirrors the zenmetrics v26 onstart pattern (no apt at runtime,
# every binary pre-baked).
set -euo pipefail

# ─── Hydrate env from /proc/1/environ (vast.ai injection point) ─────
# vast.ai sets env vars on the container's PID 1 (the entrypoint
# script) rather than via docker -e. Pull them into our scope.
if [[ -f /proc/1/environ ]]; then
    while IFS='=' read -r -d '' k v; do
        # Only export the W44_212_* vars we care about; ignore the
        # rest (vast.ai also injects ~50 metadata vars we don't need).
        case "$k" in
            W44_212_*|R2_*|AWS_*|RUNNER_*|SWEEP_*) export "$k=$v" ;;
        esac
    done < /proc/1/environ
fi

# ─── Verify baked tools (fail loud if missing) ──────────────────────
for tool in zenjxl-tuning-runner s5cmd jq; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[w44-212-onstart] FATAL: missing baked tool $tool — image is broken, rebuild." >&2
        exit 64
    fi
done
# zen-metrics is optional (CPU-fallback path works without it)
if command -v zen-metrics >/dev/null 2>&1; then
    echo "[w44-212-onstart] zen-metrics found: $(zen-metrics --version 2>&1 | head -1)"
else
    echo "[w44-212-onstart] WARN: zen-metrics not found; cells will use CPU metric fallback (cvvdp = null)."
fi

# ─── Heartbeat ──────────────────────────────────────────────────────
HEARTBEAT_KEY="${W44_212_SWEEP_BUCKET:-zen-tuning-ephemeral}/heartbeat/$(hostname)-$(date -u +%s).txt"
echo "boot: $(date -u +%FT%TZ) host=$(hostname) commit=${W44_212_RUNNER_COMMIT:-unknown}" | \
    s5cmd pipe "s3://$HEARTBEAT_KEY" 2>/dev/null || echo "[w44-212-onstart] heartbeat upload skipped (no R2 creds yet)"

# ─── Main loop ──────────────────────────────────────────────────────
SWEEP_ID="${W44_212_SWEEP_ID:?missing W44_212_SWEEP_ID env}"
SWEEP_BUCKET="${W44_212_SWEEP_BUCKET:-zen-tuning-ephemeral}"
CHUNK_PREFIX="${W44_212_CHUNK_QUEUE_PREFIX:-chunks}"
WORKER_ID="${W44_212_WORKER_ID:-$(hostname)-$$}"

echo "[w44-212-onstart] sweep=$SWEEP_ID worker=$WORKER_ID bucket=$SWEEP_BUCKET starting loop"

# Each "chunk" is a JSON file containing N cell specs. The launcher
# generates them ahead of time and writes them under
# s3://<bucket>/<sweep_id>/<chunk_prefix>/<chunk_id>.json. The worker
# atomically claims by renaming to chunks-in-flight/<chunk_id>.json
# (s5cmd mv is atomic on R2). If the rename fails another worker
# beat us — try the next chunk. Same pattern as zenmetrics v26.
while true; do
    LIST=$(s5cmd ls "s3://$SWEEP_BUCKET/$SWEEP_ID/$CHUNK_PREFIX/*.json" 2>/dev/null \
           | awk '{print $NF}' \
           | shuf \
           | head -32) || LIST=""
    if [[ -z "$LIST" ]]; then
        echo "[w44-212-onstart] no chunks available; sleeping 60s"
        sleep 60
        continue
    fi
    CLAIMED=""
    for chunk in $LIST; do
        chunk_id="${chunk%.json}"
        if s5cmd mv \
            "s3://$SWEEP_BUCKET/$SWEEP_ID/$CHUNK_PREFIX/$chunk" \
            "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/$chunk" \
            2>/dev/null; then
            CLAIMED="$chunk"
            break
        fi
    done
    if [[ -z "$CLAIMED" ]]; then
        echo "[w44-212-onstart] failed to claim any chunk; sleeping 30s"
        sleep 30
        continue
    fi
    echo "[w44-212-onstart] claimed chunk=$CLAIMED"
    # Pull chunk JSON down and hand to worker.sh
    LOCAL=/sweep-state/$CLAIMED
    mkdir -p /sweep-state
    s5cmd cp "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/$CLAIMED" "$LOCAL" 2>&1
    /usr/local/bin/worker.sh "$SWEEP_ID" "$LOCAL" "$WORKER_ID" || \
        echo "[w44-212-onstart] worker.sh exited non-zero for $CLAIMED; continuing"
    # Mark chunk done by moving to chunks-done/ (audit trail)
    s5cmd mv \
        "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/$CLAIMED" \
        "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-done/$CLAIMED" \
        2>&1 || echo "[w44-212-onstart] WARN: failed to mark $CLAIMED done"
done
