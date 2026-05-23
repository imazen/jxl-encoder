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
BASE_WORKER_ID="${W44_212_WORKER_ID:-$(hostname)-$$}"

# W44-229k (2026-05-23): N-worker fork per pod.
#
# Pre-W44-229k: a single worker.sh loop processed one cell at a time
# sequentially. On big-core pods (28–128 cores) this left CPU util at
# 1–4 % and produced ~30 cells/min/pod. Each cell is a JXL encode →
# decode → metric, dominated by the rust encoder (which uses some
# rayon internally but mostly runs serial). The runner is single-cell
# per invocation, so process-grain parallelism is the simplest win.
#
# Pod sizing: workers do encode → decode → optional GPU score →
# s5cmd upload. They share one GPU when zen-metrics is loaded, but
# most cells in W44-229 score on CPU. Per-worker memory footprint is
# small (~200 MB resident). We cap at 8 workers to limit upload
# bandwidth contention and avoid GPU OOM, and also cap at nproc so
# small pods don't oversubscribe their cores.
#
# Override via W44_229K_WORKERS_PER_POD env var; default cap=8.
WORKERS_PER_POD="${W44_229K_WORKERS_PER_POD:-0}"
if [[ "$WORKERS_PER_POD" == "0" || -z "$WORKERS_PER_POD" ]]; then
    _nproc=$(nproc 2>/dev/null || echo 1)
    if (( _nproc > 8 )); then
        WORKERS_PER_POD=8
    elif (( _nproc < 1 )); then
        WORKERS_PER_POD=1
    else
        WORKERS_PER_POD=$_nproc
    fi
fi
# W44-229L1 (2026-05-23): per-worker RAYON_NUM_THREADS.
#
# The jxl-encoder crate is built with the `parallel` feature (see
# zenjxl-tuning-runner/Cargo.toml line 51). At runtime it uses the
# ambient rayon pool for tree-learning, group fan-out, and per-iter
# transform_and_quantize_into. Without RAYON_NUM_THREADS set, rayon
# defaults to nproc threads per process — combined with the W44-229k
# N-worker fork this yields N × nproc threads = catastrophic
# oversubscription (e.g. 8 workers × 28 cores = 224 threads on a
# 28-core box).
#
# Compute per-worker thread budget = floor(nproc / WORKERS_PER_POD),
# clamped to >=1. This keeps total threads bounded by nproc while
# letting each worker's encoder internally parallelise.
#
# On a 28-core / 8-worker pod: 3 rayon threads/worker × 8 workers = 24
# threads, leaves headroom for s5cmd uploads + zen-metrics GPU calls.
# On a 4-core / 4-worker pod: 1 thread/worker (sequential encoder).
_pod_nproc=$(nproc 2>/dev/null || echo 1)
_rayon_per_worker=$(( _pod_nproc / WORKERS_PER_POD ))
if (( _rayon_per_worker < 1 )); then _rayon_per_worker=1; fi
export RAYON_NUM_THREADS="${W44_229L1_RAYON_NUM_THREADS:-$_rayon_per_worker}"

echo "[w44-229L1-onstart] sweep=$SWEEP_ID base_worker_id=$BASE_WORKER_ID bucket=$SWEEP_BUCKET workers_per_pod=$WORKERS_PER_POD nproc=$_pod_nproc rayon_num_threads=$RAYON_NUM_THREADS"

# W44-216: graceful exit on chunks-empty.
EMPTY_POLL_BUDGET="${W44_216_EMPTY_POLL_BUDGET:-2}"
EMPTY_POLL_SLEEP_S="${W44_216_EMPTY_POLL_SLEEP_S:-30}"

# The per-worker chunk loop. Forked N times below. Each fork has its
# own WORKER_ID (BASE-w0 / -w1 / ...). The s5cmd-mv chunk claim is
# atomic on R2 so two workers racing on the same chunk-id end up with
# exactly one winner; the loser sees mv exit non-zero and tries the
# next candidate from its shuffled batch.
worker_loop() {
    local worker_idx="$1"
    local WORKER_ID="${BASE_WORKER_ID}-w${worker_idx}"
    local empty_polls=0
    local _w219_s5out="/tmp/w219_s5cmd_out_w${worker_idx}.txt"
    echo "[w44-229k-worker-$worker_idx] starting WORKER_ID=$WORKER_ID"

    while true; do
        # W44-219 fix (2026-05-22): two-part fix:
        # (1) split s5cmd output into a file (the one-shot pipe lost
        #     output on big input — reproduced on pod 37399636)
        # (2) replace `shuf | head -32` with `shuf -n 32`. The original
        #     pipe failed under `set -o pipefail` (set on line 8): when
        #     head -32 takes its 32 lines and closes the pipe, shuf gets
        #     SIGPIPE on its remaining writes → exits non-zero → pipefail
        #     fires → `$()` returns failure → `|| LIST=""` triggers →
        #     LIST is empty even though all 4793 chunks were listed.
        #     `shuf -n 32` samples 32 inside shuf without needing head.
        # W44-229k: each worker gets its own tmp file to avoid two
        # workers stomping the same path mid-write.
        s5cmd ls "s3://$SWEEP_BUCKET/$SWEEP_ID/$CHUNK_PREFIX/*.json" \
            > "$_w219_s5out" 2>/dev/null || true
        local LIST
        LIST=$(awk '{print $NF}' < "$_w219_s5out" | shuf -n 32) || LIST=""
        if [[ -z "$LIST" ]]; then
            empty_polls=$((empty_polls + 1))
            echo "[w44-229k-worker-$worker_idx] no chunks available (empty_poll=${empty_polls}/${EMPTY_POLL_BUDGET})"
            if (( empty_polls >= EMPTY_POLL_BUDGET )); then
                echo "[w44-229k-worker-$worker_idx] queue drained; writing worker-done marker"
                local DONE_MARKER="s3://$SWEEP_BUCKET/$SWEEP_ID/worker-done/${WORKER_ID}.txt"
                echo "drained: $(date -u +%FT%TZ) host=$(hostname) worker=$WORKER_ID reason=chunks-empty" \
                    | s5cmd pipe "$DONE_MARKER" 2>&1 || \
                    echo "[w44-229k-worker-$worker_idx] WARN: failed to write done marker $DONE_MARKER"
                echo "[w44-229k-worker-$worker_idx] exiting cleanly"
                return 0
            fi
            # W44-229k: stagger sleep by worker_idx so they don't all
            # wake at the same moment and slam R2 with simultaneous
            # `s5cmd ls` calls.
            sleep $(( EMPTY_POLL_SLEEP_S + worker_idx ))
            continue
        fi
        local CLAIMED=""
        local chunk chunk_id
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
            empty_polls=$((empty_polls + 1))
            echo "[w44-229k-worker-$worker_idx] failed to claim any chunk (empty_poll=${empty_polls}/${EMPTY_POLL_BUDGET})"
            if (( empty_polls >= EMPTY_POLL_BUDGET )); then
                echo "[w44-229k-worker-$worker_idx] claim contention drained; writing worker-done marker"
                local DONE_MARKER="s3://$SWEEP_BUCKET/$SWEEP_ID/worker-done/${WORKER_ID}.txt"
                echo "drained: $(date -u +%FT%TZ) host=$(hostname) worker=$WORKER_ID reason=claim-contention" \
                    | s5cmd pipe "$DONE_MARKER" 2>&1 || \
                    echo "[w44-229k-worker-$worker_idx] WARN: failed to write done marker $DONE_MARKER"
                echo "[w44-229k-worker-$worker_idx] exiting cleanly"
                return 0
            fi
            sleep $(( EMPTY_POLL_SLEEP_S + worker_idx ))
            continue
        fi
        # Reset the empty-poll counter — we got real work.
        empty_polls=0
        echo "[w44-229k-worker-$worker_idx] claimed chunk=$CLAIMED"
        # Pull chunk JSON down and hand to worker.sh
        local LOCAL="/sweep-state/$CLAIMED"
        mkdir -p /sweep-state
        s5cmd cp "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/$CLAIMED" "$LOCAL" 2>&1
        /usr/local/bin/worker.sh "$SWEEP_ID" "$LOCAL" "$WORKER_ID" || \
            echo "[w44-229k-worker-$worker_idx] worker.sh exited non-zero for $CLAIMED; continuing"
        # Mark chunk done by moving to chunks-done/ (audit trail)
        s5cmd mv \
            "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/$CLAIMED" \
            "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-done/$CLAIMED" \
            2>&1 || echo "[w44-229k-worker-$worker_idx] WARN: failed to mark $CLAIMED done"
    done
}

# Fork the N worker loops in parallel, wait for all of them. Each loop
# returns 0 cleanly on drained-queue.
worker_pids=()
for ((i=0; i<WORKERS_PER_POD; i++)); do
    worker_loop "$i" &
    worker_pids+=($!)
    # Brief stagger between worker startups so the initial s5cmd ls
    # calls don't all hit R2 at the same millisecond. 250 ms × N is
    # negligible vs the 4 h sweep wall.
    sleep 0.25
done
echo "[w44-229k-onstart] forked ${#worker_pids[@]} worker loops: pids=${worker_pids[*]}"

# Wait for all workers; exit success if any of them returned cleanly.
# Bash `wait $pid` returns the exit status of that pid; if any worker
# crashes the rest keep going.
for pid in "${worker_pids[@]}"; do
    wait "$pid" || echo "[w44-229k-onstart] worker pid=$pid exited non-zero (other workers continue)"
done
echo "[w44-229k-onstart] all worker loops exited; pod done"
