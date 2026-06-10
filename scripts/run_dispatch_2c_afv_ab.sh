#!/usr/bin/env bash
# Issue #43 chunk 2c — production-context paired A/B driver.
#
# Runs the `dispatch_2c_afv_screenshot_ab --bench` grid alternately with
# arm A (env-disabled gate = pre-chunk-2c production path) and arm B
# (gate active), N passes, sample-major interleaved at process level.
# The encoder is deterministic per (cell, arm); wall is min-per-(cell,arm)
# in the analyzer.
#
# Usage: scripts/run_dispatch_2c_afv_ab.sh <output.tsv> [n_samples=3]
set -euo pipefail
OUT="${1:?usage: run_dispatch_2c_afv_ab.sh <output.tsv> [n_samples]}"
N="${2:-3}"
BIN=./target/release/examples/dispatch_2c_afv_screenshot_ab
[ -x "$BIN" ] || {
    echo "build first: cargo build -p jxl-encoder --release \\" >&2
    echo "  --features '__expert parallel butteraugli-loop ssim2-loop' \\" >&2
    echo "  --example dispatch_2c_afv_screenshot_ab" >&2
    exit 2
}
DUMP_DIR="${DUMP_DIR:-/mnt/v/output/jxl-encoder/dispatch_2c_afv_2026-06-10}"
for s in $(seq 1 "$N"); do
    DUMP_ARGS=()
    # Persist bitstreams on the first pass only (deterministic per
    # (cell, arm) — later passes are byte-identical).
    [ "$s" = "1" ] && DUMP_ARGS=(--dump "$DUMP_DIR")
    echo "=== pass $s/$N arm A (gate off) ===" >&2
    JXL_DISPATCH_AFV_SCREENSHOT_DISABLE=1 "$BIN" --bench --output "$OUT" --sample "$s" "${DUMP_ARGS[@]}"
    echo "=== pass $s/$N arm B (gate on) ===" >&2
    env -u JXL_DISPATCH_AFV_SCREENSHOT_DISABLE "$BIN" --bench --output "$OUT" --sample "$s" "${DUMP_ARGS[@]}"
done
echo "done: $OUT (bitstreams in $DUMP_DIR)" >&2
