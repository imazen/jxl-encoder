#!/usr/bin/env bash
# launch_w44_phase4_s2_fleet.sh — W44-PHASE4-S2-c2-validate fleet
# launcher. Forked from launch_w44_phase4_s1_fleet.sh; the only
# functional differences are SWEEP_ID, LABEL_PREFIX, IMAGE tag, and
# the cost cap (S2 was scoped to ~$10, much tighter than S1's $30).
#
# Purpose: validate the W44-PHASE4-S2-refit-c2 per-stratum lookup
# (commit cc081ff5) on a representative 9-image subset (~9.3K cells).
# See scripts/zenjxl-tuning-sweep/build_w44_phase4_s2_chunks.py for
# the corpus + LHS design details.
#
# Per the 2026-05-22 tag-then-manage rule every instance gets a
# `claude-w44-phase4-s2-*` label. Per research methodology Rule 9
# default is interruptible (~50-70% cheaper than on-demand).
#
# Reconstructed 2026-05-24 by W44-AUDIT-1 — the script was missing from
# the repo despite the S2 sweep having completed. The S2 sweep on R2 at
# s3://zen-tuning-ephemeral/w44-phase4-s2-c2-validate/ is the canonical
# record; this script reproduces the launcher used at that time.
#
# Reference docs:
#   - benchmarks/sweeps/w44-phase4-s2-c2-validate/ (results)
#   - scripts/zenjxl-tuning-sweep/build_w44_phase4_s2_chunks.py
#   - memory/research_methodology_9_rules_2026-05-22.md (Rules 4 + 9)
#
# Usage:
#   # Smoke test (1 box, on-demand):
#   INTERRUPTIBLE=0 BOXES=1 \
#     bash scripts/zenjxl-tuning-sweep/launch_w44_phase4_s2_fleet.sh
#
#   # Production fleet (interruptible):
#   BOXES=3 BID_PRICE=0.07 \
#     bash scripts/zenjxl-tuning-sweep/launch_w44_phase4_s2_fleet.sh
#
# Env knobs (with defaults):
#   SWEEP_ID         w44-phase4-s2-c2-validate
#   BOXES            3        (tighter $10 budget vs S1's 5)
#   IMAGE            ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2-bdd5f4fb
#                    (built from origin/main bdd5f4fb — post-S2-refit-c2
#                     stack + B5b iter-0 divergence detector)
#   MAX_DPH          0.20     (cap per-instance hourly cost USD)
#   MIN_GPU_RAM_MB   10000
#   MIN_DISK_GB      30
#   GHCR_USER        lilith
#   LABEL_PREFIX     claude-w44-phase4-s2
#   INTERRUPTIBLE    1        (Rule 9 default)
#   BID_PRICE        0.07     ($/hr cap on interruptible bid)
set -euo pipefail

SWEEP_ID="${SWEEP_ID:-w44-phase4-s2-c2-validate}"
BOXES="${BOXES:-3}"
# IMPORTANT: this image MUST contain a zenjxl-tuning-runner binary
# built from origin/main bdd5f4fb or newer (post-S2-refit-c2 + B5b).
# Build via scripts/zenjxl-tuning-sweep/build_and_push_image.sh.
IMAGE="${IMAGE:-ghcr.io/lilith/zenjxl-tuning-sweep:v3-schema-v2-bdd5f4fb}"
MAX_DPH="${MAX_DPH:-0.20}"
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-10000}"
MIN_DISK_GB="${MIN_DISK_GB:-30}"
GHCR_USER="${GHCR_USER:-lilith}"
LABEL_PREFIX="${LABEL_PREFIX:-claude-w44-phase4-s2}"

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
    echo "  Run build_w44_phase4_s2_chunks.py + sync params+chunks to R2 first." >&2
    exit 1
fi
echo "[phase4-s2-fleet] $CHUNKS_AVAIL chunks queued for sweep=$SWEEP_ID"

# ── W44-PHASE4-S1h pre-flight: verify every image in chunks_manifest.tsv ──
# exists in the corpus bucket BEFORE we burn $ on vast.ai boxes. The S1h
# postmortem (CLAUDE.md Investigation Notes) documents why this matters —
# S1 lost 4,128 cells / ~$3 to a silent corpus-staging gap.
W44_212_CORPUS_BUCKET="${W44_212_CORPUS_BUCKET:-zen-tuning-ephemeral}"
MANIFEST_LOCAL="/tmp/${SWEEP_ID}.chunks_manifest.tsv"
echo "[phase4-s2-fleet] pre-flight: verifying corpus images present in s3://${W44_212_CORPUS_BUCKET}/"
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "s3://zen-tuning-ephemeral/${SWEEP_ID}/chunks_manifest.tsv" \
    "$MANIFEST_LOCAL" 2>/dev/null || {
        echo "ERROR: cannot fetch chunks_manifest.tsv from s3://zen-tuning-ephemeral/${SWEEP_ID}/" >&2
        exit 1
    }
UNIQUE_IMAGES=$(awk -F'\t' 'NR>1{print $2}' "$MANIFEST_LOCAL" | sort -u)
N_IMAGES=$(echo "$UNIQUE_IMAGES" | wc -l)
echo "[phase4-s2-fleet] $N_IMAGES unique images referenced in manifest"

MISSING_IMAGES=()
for img_path in $UNIQUE_IMAGES; do
    R2_KEY="${img_path#/}"
    if ! AWS_PROFILE=r2 aws s3api head-object \
            --endpoint-url="$R2_ENDPOINT" \
            --bucket "$W44_212_CORPUS_BUCKET" \
            --key "$R2_KEY" >/dev/null 2>&1; then
        MISSING_IMAGES+=("$img_path")
    fi
done

if (( ${#MISSING_IMAGES[@]} > 0 )); then
    echo "" >&2
    echo "ERROR: ${#MISSING_IMAGES[@]} corpus image(s) missing from s3://${W44_212_CORPUS_BUCKET}/:" >&2
    for img in "${MISSING_IMAGES[@]}"; do
        echo "  MISSING: ${img}" >&2
    done
    echo "" >&2
    echo "Upload them BEFORE launching the fleet." >&2
    exit 2
fi
echo "[phase4-s2-fleet] pre-flight OK: all $N_IMAGES images reachable"
rm -f "$MANIFEST_LOCAL"

# ── pick the cheapest viable offers ─────────────────────────────────
QUERY="rentable=true reliability>0.96 dph_total<${MAX_DPH} cpu_cores>=8 cpu_cores<=24 cpu_ram>=12 disk_space>${MIN_DISK_GB} gpu_total_ram>=$((MIN_GPU_RAM_MB / 1024)) gpu_frac>=1.0 cuda_max_good>=12.0 dlperf>=10 num_gpus=1 inet_down>=50"
echo "[phase4-s2-fleet] querying ${BOXES} offers"
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

INTERRUPTIBLE="${INTERRUPTIBLE:-1}"
BID_PRICE="${BID_PRICE:-0.07}"
BID_ARGS=()
if [[ "$INTERRUPTIBLE" == "1" ]]; then
    BID_ARGS=(--bid_price "$BID_PRICE")
    echo "[phase4-s2-fleet] interruptible mode: bid_price=\$${BID_PRICE}/hr"
else
    echo "[phase4-s2-fleet] on-demand mode (no --bid_price)"
fi

LAUNCHED=()
N=0
for offer_id in $OFFER_IDS; do
    N=$((N+1))
    LABEL="${LABEL_PREFIX}-$(printf '%03d' $N)"
    echo "[phase4-s2-fleet] launching box ${N}/${BOXES} (offer $offer_id, label $LABEL)"
    OUT=$(vastai create instance "$offer_id" \
        --image "$IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "$ONSTART_CMD" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        "${BID_ARGS[@]}" \
        --raw 2>&1) || { echo "[phase4-s2-fleet] WARN: create failed: $OUT" >&2; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "[phase4-s2-fleet] WARN: no id in $OUT" >&2; continue; }
    LAUNCHED+=("$ID")
    vastai start instance "$ID" 2>&1 | head -1 || true
done

echo
echo "[phase4-s2-fleet] launched ${#LAUNCHED[@]} of ${BOXES} requested boxes"
echo "  ids: ${LAUNCHED[*]}"
echo "  label prefix: $LABEL_PREFIX"
echo
echo "Monitor:"
echo "  vastai show instances | grep ${LABEL_PREFIX}"
echo "  aws --profile r2 s3 ls --endpoint-url=${R2_ENDPOINT} s3://zen-tuning-ephemeral/${SWEEP_ID}/cells/ | wc -l"
echo
echo "Run janitor:"
echo "  bash scripts/zenjxl-tuning-sweep/janitor_w44_phase4_s2.sh ${LABEL_PREFIX} ${SWEEP_ID}"
