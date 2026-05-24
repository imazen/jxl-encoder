#!/usr/bin/env bash
# janitor_w44_phase4_s2.sh — autonomous lifecycle manager for
# claude-w44-phase4-s2-* pods. Forked from janitor_w44_phase4_s1.sh;
# the W44-229j MAX(cpu_util, gpu_util) idle-heuristic fix is BAKED-IN
# per memory/w44_229f_sweep_finalize_2026-05-23.md DO-NOT list #1.
#
# Per the 2026-05-22 tag-then-manage rule:
# (~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
#  incident_killed_other_user_vastai_pod_2026-05-22.md)
#
# Differences from janitor_w44_phase4_s1.sh:
#   - Default LABEL_PREFIX is `claude-w44-phase4-s2-`
#   - Default SWEEP_ID is `w44-phase4-s2-c2-validate`
#   - COST_CAP_USD lowered to $5 (S2 scoped to ~$10; janitor cap is half
#     to leave headroom for any required finalize re-runs)
#
# Reconstructed 2026-05-24 by W44-AUDIT-1 — the S2 sweep finished cleanly,
# but the janitor script wasn't committed. This file mirrors the
# behavior used in production.
#
# Stop pods whose label starts with the prefix when they are idle
# (MAX(cpu_util, gpu_util) < 1.0% per W44-229j fix) AND either:
#     a) the per-worker `worker-done/<host>-*.txt` marker exists in R2, OR
#     b) the pod has been idle for ≥ IDLE_TIMEOUT_S (default 300s)
# NEVER touches pods whose label does NOT start with `claude-`.
#
# Usage:
#   bash janitor_w44_phase4_s2.sh [LABEL_PREFIX] [SWEEP_ID]
set -euo pipefail

LABEL_PREFIX="${1:-${LABEL_PREFIX:-claude-w44-phase4-s2-}}"
SWEEP_ID="${2:-${SWEEP_ID:-w44-phase4-s2-c2-validate}}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-300}"
IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-300}"
COST_CAP_USD="${COST_CAP_USD:-5.00}"
SWEEP_BUCKET="${SWEEP_BUCKET:-zen-tuning-ephemeral}"

# Hard safety check.
if [[ "$LABEL_PREFIX" != claude-* ]]; then
    echo "FATAL: LABEL_PREFIX must start with 'claude-' (got '$LABEL_PREFIX')" >&2
    exit 1
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

declare -A idle_since
total_stopped=0
total_cost_estimate=0
loop_count=0

echo "[janitor] starting; label_prefix=$LABEL_PREFIX sweep_id=$SWEEP_ID poll=${POLL_INTERVAL_S}s idle_timeout=${IDLE_TIMEOUT_S}s cost_cap=\$${COST_CAP_USD}"

trap "echo '[janitor] caught SIGINT; exiting'; exit 0" INT TERM

while true; do
    loop_count=$((loop_count + 1))
    now_s=$(date -u +%s)

    # 0. Rescue any chunks-in-flight whose age >= 600s.
    if (( loop_count % 2 == 1 )); then
        STALE_CHUNKS=$(AWS_PROFILE=r2 aws s3 ls --endpoint-url="$R2_ENDPOINT" \
            "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/" 2>/dev/null | python3 -c "
import sys, datetime
now = datetime.datetime.utcnow()
for line in sys.stdin:
    parts = line.split()
    if len(parts) < 4: continue
    ts = parts[0] + ' ' + parts[1]
    try:
        dt = datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        age = (now - dt).total_seconds()
        if age >= 600:
            print(parts[3])
    except Exception:
        pass
" || echo "")
        if [[ -n "$STALE_CHUNKS" ]]; then
            for c in $STALE_CHUNKS; do
                echo "[janitor] RESCUING stale in-flight chunk $c"
                AWS_PROFILE=r2 aws s3 mv --endpoint-url="$R2_ENDPOINT" \
                    "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks-in-flight/$c" \
                    "s3://$SWEEP_BUCKET/$SWEEP_ID/chunks/$c" 2>&1 | tail -1
            done
        fi
    fi

    # 1. Fetch instances
    INSTANCES_JSON=$(vastai show instances --raw 2>&1 | grep -v "^DEPRECATED" || echo "[]")
    if ! echo "$INSTANCES_JSON" | python3 -c "import json,sys; json.loads(sys.stdin.read())" 2>/dev/null; then
        echo "[janitor] loop $loop_count: vastai output not JSON; skipping"
        sleep "$POLL_INTERVAL_S"
        continue
    fi

    # 2. Filter to my-tagged
    MY_PODS=$(echo "$INSTANCES_JSON" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
prefix = '$LABEL_PREFIX'
out = []
for i in data:
    label = i.get('label') or ''
    if not label.startswith(prefix):
        continue
    out.append({
        'id': i.get('id'),
        'label': label,
        'status': i.get('actual_status', '?'),
        'gpu_util': float(i.get('gpu_util', 0) or 0),
        'cpu_util': float(i.get('cpu_util', 0) or 0),
        'dph': float(i.get('dph_total', 0) or 0),
        'duration_s': float(i.get('duration', 0) or 0),
    })
print(json.dumps(out))
")
    POD_COUNT=$(echo "$MY_PODS" | python3 -c "import json,sys; print(len(json.loads(sys.stdin.read())))")

    HOURLY_BURN=$(echo "$MY_PODS" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
print(sum(p['dph'] for p in data if p['status'] == 'running'))
")

    echo "[janitor] loop=$loop_count pods=$POD_COUNT burn=\$${HOURLY_BURN}/hr stopped_total=$total_stopped"

    if [[ "$POD_COUNT" == "0" ]]; then
        if (( loop_count > 1 )); then
            echo "[janitor] no my-tagged pods remain; exiting"
            exit 0
        fi
    fi

    # 3. Cost cap enforcement
    cap_breach=$(python3 -c "print(1 if $total_cost_estimate >= $COST_CAP_USD else 0)")
    if [[ "$cap_breach" == "1" ]]; then
        echo "[janitor] COST CAP \$${COST_CAP_USD} BREACHED; stopping ALL my-tagged pods"
        echo "$MY_PODS" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
for p in data:
    if p['status'] == 'running':
        print(p['id'])
" | while read -r pid; do
            echo "[janitor] STOPPING pod $pid (cost cap)"
            vastai stop instance "$pid" 2>&1 | head -3 || echo "[janitor] WARN: stop $pid failed"
            total_stopped=$((total_stopped + 1))
        done
        exit 2
    fi

    # 4. Per-pod evaluation (W44-229j MAX(cpu,gpu) baked in)
    POD_LIST=$(echo "$MY_PODS" | python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
for p in data:
    cpu = p.get('cpu_util') or 0.0
    gpu = p.get('gpu_util') or 0.0
    try:
        cpu = float(cpu); gpu = float(gpu); util = max(cpu, gpu)
    except (TypeError, ValueError):
        util = 0.0
    print(f\"{p['id']}\t{p['label']}\t{p['status']}\t{util}\t{p['dph']}\")
")
    HAS_MARKER=$( (AWS_PROFILE=r2 aws s3 ls --endpoint-url="$R2_ENDPOINT" "s3://$SWEEP_BUCKET/$SWEEP_ID/worker-done/" 2>/dev/null || true) | wc -l)
    while IFS=$'\t' read -r pid label status util dph; do
        [[ -z "$pid" ]] && continue
        if [[ "$status" != "running" ]]; then
            unset 'idle_since[$pid]' 2>/dev/null || true
            continue
        fi
        is_idle=$(python3 -c "
try:
    print(1 if float('$util') < 1.0 else 0)
except Exception:
    print(1)
")
        if [[ "$is_idle" == "1" ]]; then
            if [[ -z "${idle_since[$pid]:-}" ]]; then
                idle_since[$pid]=$now_s
                echo "[janitor]   pod $pid ($label) idle=${util}% — marking idle_since=$now_s"
                continue
            fi
            idle_for=$((now_s - ${idle_since[$pid]}))
        else
            unset 'idle_since[$pid]' 2>/dev/null || true
            idle_for=0
        fi

        should_stop=0
        reason=""
        if [[ "$HAS_MARKER" -gt 0 ]] && [[ "$idle_for" -ge 60 ]]; then
            should_stop=1
            reason="worker-done marker present + idle ${idle_for}s"
        elif [[ "$idle_for" -ge "$IDLE_TIMEOUT_S" ]]; then
            should_stop=1
            reason="idle ${idle_for}s ≥ timeout ${IDLE_TIMEOUT_S}s"
        fi

        if [[ "$should_stop" == "1" ]]; then
            echo "[janitor]   STOPPING pod $pid ($label) — $reason"
            vastai stop instance "$pid" 2>&1 | head -3 || echo "[janitor]   WARN: stop $pid failed"
            total_stopped=$((total_stopped + 1))
            unset 'idle_since[$pid]' 2>/dev/null || true
        else
            echo "[janitor]   keeping pod $pid ($label) — util=${util}% idle_for=${idle_for}s"
        fi
    done < <(echo "$POD_LIST")

    est_increment=$(python3 -c "print($HOURLY_BURN * $POLL_INTERVAL_S / 3600)")
    total_cost_estimate=$(python3 -c "print(round($total_cost_estimate + $est_increment, 4))")

    sleep "$POLL_INTERVAL_S"
done
