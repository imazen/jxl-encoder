#!/usr/bin/env bash
# launch_w44_219_fleet.sh — W44-219 densify-sweep N-box launcher with
# mandatory `claude-w44-219-*` label per the 2026-05-22 tag-then-manage
# rule (see ~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
# incident_killed_other_user_vastai_pod_2026-05-22.md).
#
# Forked from launch_w44_216_fleet.sh. Differences:
#   - SWEEP_ID defaults to `w44-219-densify`
#   - LABEL_PREFIX defaults to `claude-w44-219-fullgrid`
#   - Image still ghcr.io/lilith/zenjxl-tuning-sweep:v2-w44-216 (Docker
#     image is sweep-agnostic — W44-219 chunks just live under a
#     different R2 prefix).
#   - Hard rule from W44-216: refuse non-claude-* LABEL_PREFIX.
#
# Usage:
#   SWEEP_ID=w44-219-densify BOXES=20 \
#     bash scripts/zenjxl-tuning-sweep/launch_w44_219_fleet.sh
#
# Env knobs (with defaults):
#   SWEEP_ID         w44-219-densify
#   BOXES            5
#   IMAGE            ghcr.io/lilith/zenjxl-tuning-sweep:v2-w44-216
#   MAX_DPH          0.25     (cap per-instance hourly cost USD)
#   MIN_GPU_RAM_MB   10000
#   MIN_DISK_GB      30
#   GHCR_USER        lilith
#   LABEL_PREFIX     claude-w44-219-fullgrid

set -euo pipefail

SWEEP_ID="${SWEEP_ID:-w44-219-densify}"
BOXES="${BOXES:-5}"
IMAGE="${IMAGE:-ghcr.io/lilith/zenjxl-tuning-sweep:v2-w44-216}"
MAX_DPH="${MAX_DPH:-0.25}"
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-10000}"
MIN_DISK_GB="${MIN_DISK_GB:-30}"
GHCR_USER="${GHCR_USER:-lilith}"
LABEL_PREFIX="${LABEL_PREFIX:-claude-w44-219-fullgrid}"

# Hard rule: refuse to launch if the label prefix isn't a `claude-*` prefix.
if [[ "$LABEL_PREFIX" != claude-* ]]; then
    echo "ERROR: LABEL_PREFIX must start with 'claude-' (got '$LABEL_PREFIX')" >&2
    echo "  See 2026-05-22 tag-then-manage rule. The label IS the authorization." >&2
    exit 1
fi

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

GHCR_TOKEN="$(gh auth token)"
[[ -n "$GHCR_TOKEN" ]] || { echo "ERROR: gh auth token empty" >&2; exit 1; }

# ── pre-flight: chunks must exist in R2 ─────────────────────────────
CHUNKS_AVAIL=$(AWS_PROFILE=r2 aws s3 ls --endpoint-url="$R2_ENDPOINT" \
    "s3://zen-tuning-ephemeral/${SWEEP_ID}/chunks/" 2>/dev/null | wc -l)
if (( CHUNKS_AVAIL == 0 )); then
    echo "ERROR: no chunks at s3://zen-tuning-ephemeral/${SWEEP_ID}/chunks/" >&2
    echo "  run build_stage_b_chunks.py + aws s3 cp first" >&2
    exit 1
fi
echo "[w44-216-fleet] $CHUNKS_AVAIL chunks queued for sweep=$SWEEP_ID"

# ── pick the cheapest viable offers ─────────────────────────────────
QUERY="rentable=true reliability>0.99 dph_total<${MAX_DPH} cpu_cores>=4 cpu_ram>=8 disk_space>${MIN_DISK_GB} gpu_total_ram>=$((MIN_GPU_RAM_MB / 1024)) gpu_frac>=1.0 cuda_max_good>=12.0 driver_version>=555.0.0 dlperf>=12 num_gpus=1"
echo "[w44-216-fleet] querying ${BOXES} offers"
echo "  $QUERY"

OFFER_IDS=$(vastai search offers "$QUERY" --order 'dph_total' --raw \
    | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
if isinstance(d, dict) and 'offers' in d: d = d['offers']
if not d: raise SystemExit('no offers match query')
n = int('$BOXES')
for o in d[:n]:
    print(o['id'])
")
[[ -z "$OFFER_IDS" ]] && { echo "ERROR: no offers matched" >&2; exit 1; }

# Build the bootstrap script ONCE, reuse across all boxes
BOOTSTRAP_SCRIPT=$(cat <<EOF
set -e
mkdir -p /root/.aws
cat > /root/.aws/credentials <<CREDS
[default]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
CREDS
export S3_ENDPOINT_URL="${R2_ENDPOINT}"
export AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION=auto
echo "[bootstrap] env hydrated"

# W44-219 HOT-FIX: live-patch /usr/local/bin/onstart.sh in the baked
# image. The original onstart pipeline
#   LIST=\$(s5cmd ls ... | awk | shuf | head -32)
# silently returns 0 lines on some pods when the s5cmd output is large
# (~4793 chunks for W44-219), likely a SIGPIPE-vs-go-runtime
# interaction on the head -32 closing the pipe before s5cmd finishes
# writing. Reproduced on smoke pod 37399636 (2026-05-22): both bash
# stages succeeded individually but the one-shot pipe returned empty.
# File-redirect equivalent (s5cmd > file then awk < file | ...) works.
# Patched here so all W44-219 fleet pods get the fix without an image
# rebuild. Drop this block once a v3 image with the fix is pushed.
python3 - <<'PYEOF'
import re, sys
from pathlib import Path
p = Path("/usr/local/bin/onstart.sh")
src = p.read_text()
old = '''    LIST=\$(s5cmd ls "s3://\$SWEEP_BUCKET/\$SWEEP_ID/\$CHUNK_PREFIX/*.json" 2>/dev/null \\\\
           | awk '{print \$NF}' \\\\
           | shuf \\\\
           | head -32) || LIST=""
'''
new = '''    # W44-219 hot-fix: split pipe → file redirects (some pods fail the
    # one-shot pipe with SIGPIPE on big input, returning empty LIST).
    _w219_s5out=/tmp/w219_s5cmd_out.txt
    s5cmd ls "s3://\$SWEEP_BUCKET/\$SWEEP_ID/\$CHUNK_PREFIX/*.json" \\\\
        > "\$_w219_s5out" 2>/dev/null || true
    LIST=\$(awk '{print \$NF}' < "\$_w219_s5out" | shuf | head -32) || LIST=""
'''
if old not in src:
    print("[bootstrap] WARN: W44-219 hot-fix target not found; skipping", file=sys.stderr)
else:
    p.write_text(src.replace(old, new))
    print("[bootstrap] W44-219 hot-fix applied to /usr/local/bin/onstart.sh")
PYEOF

exec /usr/local/bin/onstart.sh
EOF
)
BOOTSTRAP_B64=$(echo "$BOOTSTRAP_SCRIPT" | base64 -w0)
ONSTART_CMD="bash -c \"echo ${BOOTSTRAP_B64} | base64 -d > /tmp/bs.sh && chmod +x /tmp/bs.sh && exec /tmp/bs.sh\""

LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"

ENV_STR="-e R2_ACCOUNT_ID=${R2_ACCOUNT_ID}"
ENV_STR+=" -e R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
ENV_STR+=" -e R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
ENV_STR+=" -e AWS_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}"
ENV_STR+=" -e AWS_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}"
ENV_STR+=" -e AWS_DEFAULT_REGION=auto"
ENV_STR+=" -e S3_ENDPOINT_URL=${R2_ENDPOINT}"
ENV_STR+=" -e W44_212_SWEEP_ID=${SWEEP_ID}"
ENV_STR+=" -e W44_212_SWEEP_BUCKET=zen-tuning-ephemeral"
ENV_STR+=" -e W44_212_CORPUS_BUCKET=zen-tuning-ephemeral"
# W44-219: increased graceful-exit budget from 2 polls × 30s = 60s to
# 10 polls × 60s = 10 min. Rationale: the W44-219 smoke pod
# (37398803, 2026-05-22) exited cleanly after 2 empty polls with the
# 4793-chunk queue PRESENT in R2 — likely a cold-listing race or
# transient s5cmd error. W44-216 never hit this code path because its
# queue never drained. Defensive widening so a single transient list
# failure doesn't kill the worker. Knob still adjustable via env at
# launcher invocation time.
ENV_STR+=" -e W44_216_EMPTY_POLL_BUDGET=${W44_216_EMPTY_POLL_BUDGET:-10}"
ENV_STR+=" -e W44_216_EMPTY_POLL_SLEEP_S=${W44_216_EMPTY_POLL_SLEEP_S:-60}"

LAUNCHED=()
N=0
for offer_id in $OFFER_IDS; do
    N=$((N+1))
    LABEL="${LABEL_PREFIX}-$(printf '%03d' $N)"
    echo "[w44-216-fleet] launching box ${N}/${BOXES} (offer $offer_id, label $LABEL)"
    OUT=$(vastai create instance "$offer_id" \
        --image "$IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "$ONSTART_CMD" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        --raw 2>&1) || { echo "[w44-216-fleet] WARN: create failed: $OUT" >&2; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "[w44-216-fleet] WARN: no id in $OUT" >&2; continue; }
    LAUNCHED+=("$ID")
    vastai start instance "$ID" 2>&1 | head -1 || true
done

echo
echo "[w44-216-fleet] launched ${#LAUNCHED[@]} of ${BOXES} requested boxes"
echo "  ids: ${LAUNCHED[*]}"
echo "  label prefix: $LABEL_PREFIX"
echo
echo "Monitor:"
echo "  vastai show instances | grep ${LABEL_PREFIX}"
echo "  aws --profile r2 s3 ls --endpoint-url=${R2_ENDPOINT} s3://zen-tuning-ephemeral/${SWEEP_ID}/cells/ | wc -l"
echo "  aws --profile r2 s3 ls --endpoint-url=${R2_ENDPOINT} s3://zen-tuning-ephemeral/${SWEEP_ID}/chunks-done/ | wc -l"
echo "  aws --profile r2 s3 ls --endpoint-url=${R2_ENDPOINT} s3://zen-tuning-ephemeral/${SWEEP_ID}/worker-done/ | wc -l"
echo
echo "Run janitor:"
echo "  bash scripts/zenjxl-tuning-sweep/janitor_w44_216.sh ${LABEL_PREFIX}"
