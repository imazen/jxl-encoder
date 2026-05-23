#!/usr/bin/env bash
# finalize_w44_229.sh — sync cells from R2, merge into canonical
# Parquet, mirror to zentrain (R2) + Tower NAS, verify variance.
#
# Run after the fleet has drained (all claude-w44-229* pods drained).
#
# Usage:
#   R2_ACCOUNT_ID=<id> bash finalize_w44_229.sh
#
# Env:
#   SWEEP_ID         w44-229-tier2-knob-validation
#   LOCAL_CELLS_DIR  /tmp/w44-229-cells
#   LOCAL_MERGED_DIR /tmp/w44-229-merged
#   ZENTRAIN_BUCKET  zentrain
#   TOWER_DIR        /mnt/tower/output/zenjxl-tuning/2026-05-23/w44-229-tier2-knob-validation
set -euo pipefail

SWEEP_ID="${SWEEP_ID:-w44-229-tier2-knob-validation}"
LOCAL_CELLS_DIR="${LOCAL_CELLS_DIR:-/tmp/w44-229-cells}"
LOCAL_MERGED_DIR="${LOCAL_MERGED_DIR:-/tmp/w44-229-merged}"
ZENTRAIN_BUCKET="${ZENTRAIN_BUCKET:-zentrain}"
TOWER_DIR="${TOWER_DIR:-/mnt/tower/output/zenjxl-tuning/2026-05-23/w44-229-tier2-knob-validation}"

: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing}"
R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

mkdir -p "$LOCAL_CELLS_DIR" "$LOCAL_MERGED_DIR" "$TOWER_DIR"

# ── 1. Sync cells from R2 ────────────────────────────────────────────
echo "[finalize] syncing cells from R2 → $LOCAL_CELLS_DIR"
AWS_PROFILE=r2 aws s3 sync --endpoint-url="$R2_ENDPOINT" --quiet \
    "s3://zen-tuning-ephemeral/$SWEEP_ID/cells/" "$LOCAL_CELLS_DIR/"
N_CELLS=$(ls "$LOCAL_CELLS_DIR"/*.parquet 2>/dev/null | wc -l)
echo "[finalize] $N_CELLS cell Parquets local"
if (( N_CELLS == 0 )); then
    echo "[finalize] FAIL: no cells found, aborting" >&2
    exit 1
fi

# ── 2. Merge ────────────────────────────────────────────────────────
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo "[finalize] merging → $LOCAL_MERGED_DIR"
python3 "$SCRIPT_DIR/merge_w44_229_cells.py" \
    --in-dir "$LOCAL_CELLS_DIR" --out-dir "$LOCAL_MERGED_DIR" \
    --sweep-id "$SWEEP_ID" 2>&1 | tee "$LOCAL_MERGED_DIR/merge.log"

[[ -f "$LOCAL_MERGED_DIR/merged.parquet" ]] || { echo "[finalize] FAIL: merged.parquet not produced" >&2; exit 1; }

# ── 3. Upload to zentrain (canonical R2) ─────────────────────────────
ZENTRAIN_KEY="zenjxl-tuning/2026-05-23/w44-229-tier2-knob-validation/merged.parquet"
echo "[finalize] uploading canonical to s3://$ZENTRAIN_BUCKET/$ZENTRAIN_KEY"
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "$LOCAL_MERGED_DIR/merged.parquet" "s3://$ZENTRAIN_BUCKET/$ZENTRAIN_KEY" || {
        echo "[finalize] WARN: zentrain bucket upload failed (bucket may not exist on this account); continuing"
    }
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "$LOCAL_MERGED_DIR/merged.meta" "s3://$ZENTRAIN_BUCKET/zenjxl-tuning/2026-05-23/w44-229-tier2-knob-validation/merged.meta" 2>/dev/null || true
AWS_PROFILE=r2 aws s3 cp --endpoint-url="$R2_ENDPOINT" \
    "$LOCAL_MERGED_DIR/merged.variance_check.tsv" "s3://$ZENTRAIN_BUCKET/zenjxl-tuning/2026-05-23/w44-229-tier2-knob-validation/merged.variance_check.tsv" 2>/dev/null || true

# ── 4. Mirror to Tower NAS ────────────────────────────────────────────
echo "[finalize] mirroring to $TOWER_DIR"
cp "$LOCAL_MERGED_DIR/merged.parquet" "$TOWER_DIR/"
cp "$LOCAL_MERGED_DIR/merged.meta" "$TOWER_DIR/"
cp "$LOCAL_MERGED_DIR/merged.variance_check.tsv" "$TOWER_DIR/" 2>/dev/null || true
cp "$LOCAL_MERGED_DIR/merge.log" "$TOWER_DIR/"
ls -lah "$TOWER_DIR/"

# ── 5. SHA256 cross-check (local vs Tower) ──────────────────────────
LOCAL_SHA=$(sha256sum "$LOCAL_MERGED_DIR/merged.parquet" | awk '{print $1}')
TOWER_SHA=$(sha256sum "$TOWER_DIR/merged.parquet" | awk '{print $1}')
echo "[finalize] sha256 local : $LOCAL_SHA"
echo "[finalize] sha256 tower : $TOWER_SHA"
if [[ "$LOCAL_SHA" != "$TOWER_SHA" ]]; then
    echo "[finalize] FAIL: sha256 mismatch local vs Tower" >&2
    exit 1
fi
# Verify R2 zentrain copy if the upload succeeded
R2_ETAG=$(AWS_PROFILE=r2 aws s3api head-object --endpoint-url="$R2_ENDPOINT" \
    --bucket "$ZENTRAIN_BUCKET" --key "$ZENTRAIN_KEY" \
    --query 'ContentLength' --output text 2>/dev/null || echo "missing")
LOCAL_SIZE=$(stat -c%s "$LOCAL_MERGED_DIR/merged.parquet")
echo "[finalize] zentrain size : $R2_ETAG (local: $LOCAL_SIZE)"

echo
echo "[finalize] DONE"
echo "  local merged: $LOCAL_MERGED_DIR/merged.parquet"
echo "  zentrain:     s3://$ZENTRAIN_BUCKET/$ZENTRAIN_KEY"
echo "  tower:        $TOWER_DIR/merged.parquet"
echo "  sha256:       $LOCAL_SHA"
echo
echo "Variance check first 10 lines:"
head -10 "$LOCAL_MERGED_DIR/merged.variance_check.tsv" 2>/dev/null || echo "(no variance check)"
