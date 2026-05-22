#!/usr/bin/env bash
# launch_w44_215_smoke.sh — Stage A single-box smoke launcher.
#
# Modelled on zenmetrics scripts/sweep/launch_single_instance.sh.
# Pushes onstart-bootstrap that writes ~/.aws/credentials for R2 inside
# the vast.ai box, then exec's the baked /usr/local/bin/onstart.sh.
#
# Required env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
# VAST_API_KEY (or vastai config in place).
#
# Required CLI: vastai, gh, jq.

set -euo pipefail

SWEEP_ID="${SWEEP_ID:-w44-215-smoke-1box}"
IMAGE="${IMAGE:-ghcr.io/lilith/zenjxl-tuning-sweep:v2-f941d190-fixed}"
MAX_DPH="${MAX_DPH:-0.20}"
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-10000}"
MIN_DISK_GB="${MIN_DISK_GB:-30}"
GHCR_USER="${GHCR_USER:-lilith}"

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID missing}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY missing}"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

GHCR_TOKEN="$(gh auth token)"
[[ -n "$GHCR_TOKEN" ]] || { echo "ERROR: gh auth token empty" >&2; exit 1; }

# ── pick the cheapest viable single-GPU offer with a reasonable driver ──
# (CUDA 12.0+, ≥ 555 driver per zenmetrics launch_single_instance.sh
# rationale; 12+ GB VRAM for the GPU metric calls.)
QUERY="rentable=true reliability>0.99 dph_total<${MAX_DPH} cpu_cores>=4 cpu_ram>=8 disk_space>${MIN_DISK_GB} gpu_total_ram>=$((MIN_GPU_RAM_MB / 1024)) gpu_frac>=1.0 cuda_max_good>=12.0 driver_version>=555.0.0 dlperf>=12 num_gpus=1"
echo "[w44-215-smoke] querying offers"
echo "  $QUERY"
OFFER_ID=$(vastai search offers "$QUERY" --order 'dph_total' --raw \
    | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
if isinstance(d, dict) and 'offers' in d: d = d['offers']
if not d: raise SystemExit('no offers match query')
print(d[0]['id'])
")
echo "  picked offer $OFFER_ID"

# ── ENV passed to the container ─────────────────────────────────────
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
ENV_STR+=" -e W44_212_RUNNER_COMMIT=f941d190"

# ── boot-time bootstrap: write s5cmd-compatible AWS creds + exec onstart.sh
# We base64-encode the script to dodge vast.ai's onstart-cmd quoting layer
# (single + double quotes nested with $-expansion broke W44-215 V1 launch
# attempt — bash reported `unexpected EOF` because vast.ai split on the
# inner quote).
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

LABEL="${SWEEP_ID}-single"
LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"

# Build the onstart-cmd as a single line that:
#   1. decodes the base64 bootstrap
#   2. saves to /tmp/bs.sh
#   3. chmod + exec it
# Each step uses only simple double-quotes so vast.ai's CLI tokenizer
# doesn't choke.
ONSTART_CMD="bash -c \"echo ${BOOTSTRAP_B64} | base64 -d > /tmp/bs.sh && chmod +x /tmp/bs.sh && exec /tmp/bs.sh\""

echo "[w44-215-smoke] creating instance"
echo "  IMAGE:  $IMAGE"
echo "  LABEL:  $LABEL"
echo
OUT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --login "$LOGIN_STR" \
    --onstart-cmd "$ONSTART_CMD" \
    --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
    --raw 2>&1)
echo "$OUT" | head -10
ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
[[ -z "$ID" ]] && { echo "ERROR: launch failed:" >&2; echo "$OUT" >&2; exit 1; }

echo
echo "[w44-215-smoke] launched instance $ID (offer $OFFER_ID, label $LABEL)"

# vast.ai ssh-runtype instances may need explicit start.
echo "[w44-215-smoke] starting instance $ID"
vastai start instance "$ID" 2>&1 | head -2 || true

echo
echo "Monitor commands:"
echo "  vast status:         vastai show instances | grep $LABEL"
echo "  follow logs:         vastai logs $ID --tail"
echo "  gpu util:            vastai execute $ID 'nvidia-smi -l 2 --query-gpu=utilization.gpu,memory.used --format=csv,noheader' | head -20"
echo "  parquets uploaded:   aws --profile r2 s3 ls --endpoint-url=$R2_ENDPOINT s3://zen-tuning-ephemeral/$SWEEP_ID/cells/ | wc -l"
echo "  log tail:            aws --profile r2 s3 cp --endpoint-url=$R2_ENDPOINT s3://zen-tuning-ephemeral/$SWEEP_ID/logs/ - --recursive"
echo "  heartbeat:           aws --profile r2 s3 ls --endpoint-url=$R2_ENDPOINT s3://zen-tuning-ephemeral/heartbeat/"
echo "  destroy when done:   vastai destroy instance $ID"
