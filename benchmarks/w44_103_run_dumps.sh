#!/usr/bin/env bash
# W44-103: per-cell strategy dump runner.
#
# Re-runs ours and cjxl encodes in separate processes per cell so that the
# W44-76 dump infrastructure (initializes once per process) writes a fresh
# TSV for each (effort, distance) cell.
#
# Output: /tmp/w44_103_dumps/e{effort}_d{distance}_{ours|cjxl}/per_block_*.tsv

set -euo pipefail

TERMINAL=/home/lilith/work/codec-corpus/gb82-sc/terminal.png
DISTANCE=4
CJXL_RS=/home/lilith/work/zen/jxl-encoder-shared-target/release/cjxl-rs
CJXL_PATCHED=/home/lilith/work/jxl-efforts/libjxl--w44-76-per-block-debug-dump/build/tools/cjxl

mkdir -p /tmp/w44_103_dumps

for effort in 5 6 7 8 9; do
  for side in ours cjxl; do
    dump_dir=/tmp/w44_103_dumps/e${effort}_d${DISTANCE}_${side}
    rm -rf "$dump_dir"
    mkdir -p "$dump_dir"
    out=/tmp/w44_103_${side}_e${effort}_d${DISTANCE}.jxl
    rm -f "$out"
    if [ "$side" = ours ]; then
      JXL_W44_76_PER_BLOCK_DUMP="$dump_dir" "$CJXL_RS" \
        --effort $effort --distance $DISTANCE --threads 1 \
        "$TERMINAL" "$out" 2>&1 | tail -2 | sed "s/^/  [ours e$effort] /"
    else
      JXL_W44_76_PER_BLOCK_DUMP="$dump_dir" "$CJXL_PATCHED" \
        "$TERMINAL" "$out" -e $effort -d $DISTANCE --num_threads 1 --quiet 2>&1 \
        | tail -2 | sed "s/^/  [cjxl e$effort] /"
    fi
    sz=$(wc -c < "$out" 2>/dev/null || echo "FAIL")
    dump_sz=$(du -sh "$dump_dir"/*.tsv 2>/dev/null | awk '{print $1}' || echo "MISSING")
    echo "  -> $side e$effort d$DISTANCE: bytes=$sz dump=$dump_sz"
  done
done
