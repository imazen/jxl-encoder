#!/usr/bin/env bash
#
# HDR drift bench (W19-2): run the existing hdr_rd_sweep_vs_cjxl harness
# twice — once against cjxl_OLD (d2c7032) and once against cjxl_NEW (4279d48)
# — then diff the two TSVs into a third "drift" TSV.

set -euo pipefail

: "${CJXL_OLD:=/tmp/cjxl_old_d2c7032}"
: "${CJXL_NEW:=/tmp/cjxl_new_4279d48}"
: "${REPO:=/home/lilith/work/zen/jxl-encoder}"
: "${OUT_DIR:?must set OUT_DIR}"

mkdir -p "$OUT_DIR"

echo "=== HDR sweep against OLD cjxl ===" >&2
CJXL="$CJXL_OLD" cargo run -q --release --manifest-path "$REPO/Cargo.toml" \
    -p jxl-encoder --example hdr_rd_sweep_vs_cjxl 2>&1 | tail -20

# the example writes hdr_rd_sweep_<UTC>.tsv into REPO/benchmarks/
old_tsv=$(ls -t "$REPO/benchmarks/"hdr_rd_sweep_*.tsv | head -1)
echo "OLD tsv: $old_tsv" >&2
cp "$old_tsv" "$OUT_DIR/hdr_old.tsv"

echo "=== HDR sweep against NEW cjxl ===" >&2
CJXL="$CJXL_NEW" cargo run -q --release --manifest-path "$REPO/Cargo.toml" \
    -p jxl-encoder --example hdr_rd_sweep_vs_cjxl 2>&1 | tail -20

new_tsv=$(ls -t "$REPO/benchmarks/"hdr_rd_sweep_*.tsv | head -1)
echo "NEW tsv: $new_tsv" >&2
cp "$new_tsv" "$OUT_DIR/hdr_new.tsv"

# diff: layout, distance, ours_bytes (should be identical),
# cjxl_old_bytes, cjxl_new_bytes, delta_pct
DRIFT="$OUT_DIR/hdr_drift.tsv"
awk -F'\t' '
NR==FNR {
    if (FNR==1) next
    key=$1"\t"$2
    old_ours[key]=$6
    old_cjxl[key]=$7
    next
}
FNR==1 { print "layout\tdistance\tours_bytes\tcjxl_old_bytes\tcjxl_new_bytes\tdelta_new_vs_old_pct"; next }
{
    key=$1"\t"$2
    new_ours=$6
    new_cjxl=$7
    delta = (new_cjxl - old_cjxl[key]) / old_cjxl[key] * 100.0
    printf "%s\t%s\t%s\t%s\t%s\t%+.2f\n", $1, $2, new_ours, old_cjxl[key], new_cjxl, delta
}' "$OUT_DIR/hdr_old.tsv" "$OUT_DIR/hdr_new.tsv" > "$DRIFT"

echo "wrote: $DRIFT" >&2
cat "$DRIFT"
