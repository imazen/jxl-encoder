#!/usr/bin/env bash
# W44-212 fleet launcher: spawn N vast.ai instances for one sweep.
#
# Prerequisites:
#   - $VAST_API_KEY set (export VAST_API_KEY=...)
#   - $R2_ACCESS_KEY_ID + $R2_SECRET_ACCESS_KEY + $R2_ACCOUNT_ID set
#   - sweep chunks already pushed to s3://$SWEEP_BUCKET/$SWEEP_ID/chunks/
#   - corpus pushed to s3://$CORPUS_BUCKET/ (per-image path matching
#     image_path in the cell specs)
#   - docker image built + pushed:
#       ghcr.io/imazen/zenjxl-tuning-sweep:v1-<commit>
#
# Usage:
#   ./launch_fleet.sh \
#     --sweep-id W44-XYZ-mysweep \
#     --num-instances 8 \
#     [--gpu-type "RTX 3090"] \
#     [--max-bid 0.20] \
#     [--image ghcr.io/imazen/zenjxl-tuning-sweep:v1-abc1234]
set -euo pipefail

SWEEP_ID=""
NUM=4
GPU_TYPE=""
MAX_BID="0.30"
IMAGE="${W44_212_IMAGE:-ghcr.io/imazen/zenjxl-tuning-sweep:v1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep-id) SWEEP_ID="$2"; shift 2 ;;
        --num-instances) NUM="$2"; shift 2 ;;
        --gpu-type) GPU_TYPE="$2"; shift 2 ;;
        --max-bid) MAX_BID="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$SWEEP_ID" ]] && { echo "--sweep-id required" >&2; exit 1; }
[[ -z "${VAST_API_KEY:-}" ]] && { echo "VAST_API_KEY not set" >&2; exit 1; }
[[ -z "${R2_ACCESS_KEY_ID:-}" ]] && { echo "R2_ACCESS_KEY_ID not set" >&2; exit 1; }

# Build the env-vars block passed to each vast.ai instance.
ENV_VARS="-e W44_212_SWEEP_ID=$SWEEP_ID"
ENV_VARS+=" -e W44_212_SWEEP_BUCKET=${W44_212_SWEEP_BUCKET:-zen-tuning-ephemeral}"
ENV_VARS+=" -e W44_212_CORPUS_BUCKET=${W44_212_CORPUS_BUCKET:-zen-corpus}"
ENV_VARS+=" -e W44_212_OUTPUT_PREFIX=${W44_212_OUTPUT_PREFIX:-cells}"
ENV_VARS+=" -e AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID"
ENV_VARS+=" -e AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY"
ENV_VARS+=" -e AWS_DEFAULT_REGION=auto"
ENV_VARS+=" -e S3_ENDPOINT_URL=https://${R2_ACCOUNT_ID:?}.r2.cloudflarestorage.com"

# Use vastai CLI to spawn. The user must already have it installed
# (`pip install vastai`).
SEARCH_QUERY="reliability > 0.95 disk_space > 30 inet_down > 50"
if [[ -n "$GPU_TYPE" ]]; then
    SEARCH_QUERY+=" gpu_name=\"$GPU_TYPE\""
fi

echo "[w44-212-launch] image=$IMAGE sweep=$SWEEP_ID instances=$NUM"
echo "[w44-212-launch] vast query: $SEARCH_QUERY"

# Pick the top N offers by $/hr.
OFFERS=$(vastai search offers --raw "$SEARCH_QUERY" --order "dph_total" --limit "$NUM" 2>/dev/null)
if [[ -z "$OFFERS" ]]; then
    echo "[w44-212-launch] no offers found matching query" >&2
    exit 2
fi

ID_LIST=$(echo "$OFFERS" | jq -r '.[].id' | head -n "$NUM")
COUNT=0
for offer_id in $ID_LIST; do
    COUNT=$((COUNT + 1))
    LABEL="w44-212-$SWEEP_ID-$COUNT"
    echo "[w44-212-launch] creating instance $COUNT id=$offer_id label=$LABEL"
    vastai create instance "$offer_id" \
        --image "$IMAGE" \
        --label "$LABEL" \
        --price "$MAX_BID" \
        --disk 20 \
        --env "$ENV_VARS" \
        --onstart-cmd "/usr/local/bin/onstart.sh" || \
        echo "[w44-212-launch] WARN: failed to create $COUNT (continuing)"
done

echo "[w44-212-launch] launched $COUNT instances. Monitor:"
echo "  vastai show instances | grep w44-212-$SWEEP_ID"
echo "  s5cmd ls s3://${W44_212_SWEEP_BUCKET:-zen-tuning-ephemeral}/$SWEEP_ID/cells/ | wc -l"
