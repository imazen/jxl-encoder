#!/usr/bin/env bash
# launch_w44_229_fleet.sh — W44-229 Tier-2 5-knob validation sweep
# launcher. Forked from launch_w44_216_fleet.sh; per the 2026-05-22
# tag-then-manage rule every instance gets a `claude-w44-229-*` label.
# Per Rule 9 (research_methodology_9_rules_2026-05-22.md) default is
# interruptible (~50-70% cheaper than on-demand).
#
# This sweep validates the W44-222 5-knob Tier2Knobs expander on
# out-of-distribution data + produces Tier-3 MLP training labels in
# one shot. See:
#   - W44-229 chunk spec
#   - memory/phase_b_tier2_complete_2026-05-23.md (Option 3)
#   - scripts/zenjxl-tuning-sweep/build_w44_229_chunks.py
#
# Usage:
#   # Smoke test (1 box, on-demand):
#   INTERRUPTIBLE=0 BOXES=1 \
#     bash scripts/zenjxl-tuning-sweep/launch_w44_229_fleet.sh
#
#   # Production fleet:
#   BOXES=8 BID_PRICE=0.07 \
#     bash scripts/zenjxl-tuning-sweep/launch_w44_229_fleet.sh
#
# Env knobs (with defaults):
#   SWEEP_ID         w44-229-tier2-knob-validation
#   BOXES            8
#   IMAGE            ghcr.io/lilith/zenjxl-tuning-sweep:v2-w44-216
#                    (sweep-agnostic image; W44-229 just uses a
#                     different R2 prefix)
#   MAX_DPH          0.20     (cap per-instance hourly cost USD)
#   MIN_GPU_RAM_MB   10000
#   MIN_DISK_GB      30
#   GHCR_USER        lilith
#   LABEL_PREFIX     claude-w44-229-tier2
#   INTERRUPTIBLE    1        (Rule 9 default)
#   BID_PRICE        0.07     ($/hr cap on interruptible bid)
set -euo pipefail

SWEEP_ID="${SWEEP_ID:-w44-229-tier2-knob-validation}"
BOXES="${BOXES:-8}"
IMAGE="${IMAGE:-ghcr.io/lilith/zenjxl-tuning-sweep:v2-w44-216}"
MAX_DPH="${MAX_DPH:-0.20}"
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-10000}"
MIN_DISK_GB="${MIN_DISK_GB:-30}"
GHCR_USER="${GHCR_USER:-lilith}"
LABEL_PREFIX="${LABEL_PREFIX:-claude-w44-229-tier2}"

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
    echo "  Run build_w44_229_chunks.py + sync params+chunks to R2 first." >&2
    exit 1
fi
echo "[w44-229-fleet] $CHUNKS_AVAIL chunks queued for sweep=$SWEEP_ID"

# ── pick the cheapest viable offers ─────────────────────────────────
QUERY="rentable=true reliability>0.99 dph_total<${MAX_DPH} cpu_cores>=4 cpu_ram>=8 disk_space>${MIN_DISK_GB} gpu_total_ram>=$((MIN_GPU_RAM_MB / 1024)) gpu_frac>=1.0 cuda_max_good>=12.0 driver_version>=555.0.0 dlperf>=12 num_gpus=1"
echo "[w44-229-fleet] querying ${BOXES} offers"
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
echo "[bootstrap] env hydrated, exec'ing onstart.sh"
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
ENV_STR+=" -e W44_216_EMPTY_POLL_BUDGET=2"
ENV_STR+=" -e W44_216_EMPTY_POLL_SLEEP_S=30"

# Per research methodology Rule 9 (research_methodology_9_rules_2026-05-22.md):
# default to interruptible (~50-70% cheaper). Janitor + chunk-rescue handles
# the eviction risk we were already taking. Set INTERRUPTIBLE=0 to opt out
# (e.g. smoke runs that must definitely complete in 5 minutes).
INTERRUPTIBLE="${INTERRUPTIBLE:-1}"
BID_PRICE="${BID_PRICE:-0.07}"  # $/hr cap; tuned for W44-229's ~$5 budget on ~4h sweep
BID_ARGS=()
if [[ "$INTERRUPTIBLE" == "1" ]]; then
    BID_ARGS=(--bid_price "$BID_PRICE")
    echo "[w44-229-fleet] interruptible mode: bid_price=\$${BID_PRICE}/hr"
else
    echo "[w44-229-fleet] on-demand mode (no --bid_price)"
fi

LAUNCHED=()
N=0
for offer_id in $OFFER_IDS; do
    N=$((N+1))
    LABEL="${LABEL_PREFIX}-$(printf '%03d' $N)"
    echo "[w44-229-fleet] launching box ${N}/${BOXES} (offer $offer_id, label $LABEL)"
    OUT=$(vastai create instance "$offer_id" \
        --image "$IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "$ONSTART_CMD" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        "${BID_ARGS[@]}" \
        --raw 2>&1) || { echo "[w44-229-fleet] WARN: create failed: $OUT" >&2; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "[w44-229-fleet] WARN: no id in $OUT" >&2; continue; }
    LAUNCHED+=("$ID")
    vastai start instance "$ID" 2>&1 | head -1 || true
done

echo
echo "[w44-229-fleet] launched ${#LAUNCHED[@]} of ${BOXES} requested boxes"
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
echo "  bash scripts/zenjxl-tuning-sweep/janitor_w44_229.sh ${LABEL_PREFIX} ${SWEEP_ID}"
