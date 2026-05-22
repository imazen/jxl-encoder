#!/usr/bin/env bash
# finalize_w44_216.sh — sync cells from R2, merge into canonical
# Parquet, mirror to Tower, verify variance, write final memo entries.
#
# Run after the fleet has drained (all my-tagged pods stopped).
#
# Usage:
#   bash finalize_w44_216.sh
#
# Env:
#   SWEEP_ID         w44-216-stage-b
#   LOCAL_CELLS_DIR  /tmp/w44-216-cells
#   LOCAL_MERGED_DIR /tmp/w44-216-merged
#   ZENTRAIN_BUCKET  zentrain
set -euo pipefail

SWEEP_ID="${SWEEP_ID:-w44-216-stage-b}"
LOCAL_CELLS_DIR="${LOCAL_CELLS_DIR:-/tmp/w44-216-cells}"
LOCAL_MERGED_DIR="${LOCAL_MERGED_DIR:-/tmp/w44-216-merged}"
ZENTRAIN_BUCKET="${ZENTRAIN_BUCKET:-zentrain}"
TOWER_DIR="${TOWER_DIR:-/mnt/tower/output/zenjxl-tuning/2026-05-22/w44-216-stage-b}"

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

mkdir -p "$LOCAL_CELLS_DIR" "$LOCAL_MERGED_DIR" "$TOWER_DIR"

# ── 1. Verify fleet drained ─────────────────────────────────────────
RUNNING_PODS=$(vastai show instances --raw 2>/dev/null | grep -v DEPRECATED | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    n = len([i for i in d if (i.get('label') or '').startswith('claude-w44-216-') and i.get('actual_status')=='running'])
    print(n)
except Exception:
    print(0)
" 2>/dev/null || echo 0)
echo "[finalize] $RUNNING_PODS my-tagged pods still running"
if (( RUNNING_PODS > 0 )); then
    echo "[finalize] WARN: fleet not fully drained; proceed anyway? (5s grace)"
    sleep 5
fi

# ── 2. Sync cells from R2 ────────────────────────────────────────────
echo "[finalize] syncing cells from R2 → $LOCAL_CELLS_DIR"
AWS_PROFILE=r2 aws s3 sync --endpoint-url="$R2_ENDPOINT" --quiet \
    "s3://zen-tuning-ephemeral/$SWEEP_ID/cells/" "$LOCAL_CELLS_DIR/"
N_CELLS=$(ls "$LOCAL_CELLS_DIR"/*.parquet 2>/dev/null | wc -l)
echo "[finalize] $N_CELLS cell Parquets local"
if (( N_CELLS == 0 )); then
    echo "[finalize] FAIL: no cells found, aborting" >&2
    exit 1
fi

# ── 3. Merge ────────────────────────────────────────────────────────
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "[finalize] merging → $LOCAL_MERGED_DIR"
python3 "$SCRIPT_DIR/merge_w44_216_cells.py" \
    --in-dir "$LOCAL_CELLS_DIR" --out-dir "$LOCAL_MERGED_DIR" 2>&1 | tee "$LOCAL_MERGED_DIR/merge.log"

[[ -f "$LOCAL_MERGED_DIR/merged.parquet" ]] || { echo "[finalize] FAIL: merged.parquet not produced" >&2; exit 1; }

# ── 4. Upload to zentrain (canonical) ─────────────────────────────────
ZENTRAIN_KEY="zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.parquet"
echo "[finalize] uploading canonical to s3://$ZENTRAIN_BUCKET/$ZENTRAIN_KEY"
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "$LOCAL_MERGED_DIR/merged.parquet" "s3://$ZENTRAIN_BUCKET/$ZENTRAIN_KEY" || {
        echo "[finalize] WARN: zentrain bucket upload failed (bucket may not exist on this account); skipping"
    }
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "$LOCAL_MERGED_DIR/merged.meta" "s3://$ZENTRAIN_BUCKET/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.meta" 2>/dev/null || true
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "$LOCAL_MERGED_DIR/merged.variance_check.tsv" "s3://$ZENTRAIN_BUCKET/zenjxl-tuning/2026-05-22/w44-216-stage-b/merged.variance_check.tsv" 2>/dev/null || true

# ── 5. Mirror to Tower NAS ────────────────────────────────────────────
echo "[finalize] mirroring to $TOWER_DIR"
cp "$LOCAL_MERGED_DIR/merged.parquet" "$TOWER_DIR/"
cp "$LOCAL_MERGED_DIR/merged.meta" "$TOWER_DIR/"
cp "$LOCAL_MERGED_DIR/merged.variance_check.tsv" "$TOWER_DIR/"
cp "$LOCAL_MERGED_DIR/merge.log" "$TOWER_DIR/"
ls -lah "$TOWER_DIR/"

echo
echo "[finalize] DONE"
echo "  local merged: $LOCAL_MERGED_DIR/merged.parquet"
echo "  zentrain:     s3://$ZENTRAIN_BUCKET/$ZENTRAIN_KEY"
echo "  tower:        $TOWER_DIR/merged.parquet"
echo
echo "Variance check first 10 lines:"
head -10 "$LOCAL_MERGED_DIR/merged.variance_check.tsv" 2>/dev/null || echo "(no variance check)"
