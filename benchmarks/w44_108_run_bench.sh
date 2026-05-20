#!/bin/bash
# W44-108 bench reproducer.
# Runs the 18-cell target sweep + 24-cell spot-check.
# Pre-req: built ledger binary at $LEDGER (see CARGO_TARGET_DIR below).
set -e

CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/work/zen/jxl-encoder-shared-target}
cargo build -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' --example cjxl_parity_ledger
LEDGER=$CARGO_TARGET_DIR/release/examples/cjxl_parity_ledger

REPO_ROOT=$(jj workspace list 2>/dev/null | head -1 | awk '{print $2}')
[ -n "$REPO_ROOT" ] || REPO_ROOT=$(pwd)
OUT_TARGET="$REPO_ROOT/benchmarks/w44_108_recover_d2d3_wins_2026-05-20.tsv"
OUT_SPOT="$REPO_ROOT/benchmarks/w44_108_spotcheck_post_fix_2026-05-20.tsv"
rm -f "$OUT_TARGET" "$OUT_SPOT"

# 18 target cells (sacrificed wins + preserved cells + W44-105 control)
declare -a TARGETS=(
    "codec_wiki.png 8 2.0"
    "codec_wiki.png 8 2.5"
    "codec_wiki.png 8 3.0"
    "terminal.png 8 2.0"
    "terminal.png 8 2.5"
    "terminal.png 8 3.0"
    "imac_g3.png 8 3.0"
    "terminal.png 9 2.5"
    "terminal.png 8 4.0"
    "terminal.png 8 5.0"
    "terminal.png 8 6.0"
    "terminal.png 9 4.0"
    "terminal.png 9 5.0"
    "terminal.png 9 6.0"
    "codec_wiki.png 8 4.0"
    "codec_wiki.png 8 5.0"
)
for cell in "${TARGETS[@]}"; do
    read -r img eff dist <<<"$cell"
    "$LEDGER" --update --image "$img" --effort "$eff" --distance "$dist" --output "$OUT_TARGET" 2>&1 | tail -1
done

# 24-cell spot-check (photos + low-d screenshots — gate cannot fire)
declare -a SPOT=(
    "1418519.png 8 2.5" "1418519.png 9 2.5" "1418519.png 8 3.0"
    "1189261.png 8 2.5" "1189261.png 9 3.0"
    "1025469.png 8 2.5" "1025469.png 9 2.0"
    "1420710.png 8 2.5" "1420710.png 8 3.0" "1420710.png 9 3.0"
    "1531677.png 8 2.0" "1531677.png 9 2.5" "1531677.png 8 3.0"
    "1044329.png 8 2.0" "1044329.png 9 3.0"
    "1624487.png 8 3.0"
    "1418519.png 8 4.0" "1418519.png 9 5.0" "1189261.png 8 4.0"
    "1025469.png 9 4.0"
    "1531677.png 8 4.0" "1044329.png 8 5.0"
    "1624487.png 9 4.0"
    "terminal.png 8 0.5" "terminal.png 9 0.5" "codec_wiki.png 9 0.4"
    "imac_g3.png 8 0.6" "imac_g3.png 9 0.5"
)
for cell in "${SPOT[@]}"; do
    read -r img eff dist <<<"$cell"
    "$LEDGER" --update --image "$img" --effort "$eff" --distance "$dist" --output "$OUT_SPOT" 2>&1 | tail -1
done

echo "DONE. Target: $OUT_TARGET ($(wc -l < $OUT_TARGET) rows)"
echo "      Spot:   $OUT_SPOT ($(wc -l < $OUT_SPOT) rows)"
