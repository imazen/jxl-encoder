#!/usr/bin/env bash
# W44-155: per-cell strategy dump runner (1420710 e5 d=5/6 — diagnose
# why d=6 doesn't close under W44-154 B=1.22 while d=5 does).
#
# Re-runs ours and cjxl encodes in separate processes per cell so that the
# W44-76 dump infrastructure (initializes once per process) writes a fresh
# TSV for each (effort, distance) cell.
#
# Output: /tmp/w44_155_dumps/e{effort}_d{distance}_{ours|cjxl}/per_block_*.tsv

set -euo pipefail

IMG=/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1420710.png
CJXL_RS=/home/lilith/work/zen/jxl-encoder-shared-target/release/cjxl-rs
CJXL_PATCHED=/home/lilith/work/jxl-efforts/libjxl--w44-76-per-block-debug-dump/build/tools/cjxl

mkdir -p /tmp/w44_155_dumps

EFFORT=5
for distance in 5 6; do
  for side in ours cjxl; do
    dump_dir=/tmp/w44_155_dumps/e${EFFORT}_d${distance}_${side}
    rm -rf "$dump_dir"
    mkdir -p "$dump_dir"
    out=/tmp/w44_155_${side}_e${EFFORT}_d${distance}.jxl
    rm -f "$out"
    if [ "$side" = ours ]; then
      JXL_W44_76_PER_BLOCK_DUMP="$dump_dir" "$CJXL_RS" \
        --effort $EFFORT --distance $distance --threads 1 \
        "$IMG" "$out" 2>&1 | tail -2 | sed "s/^/  [ours e$EFFORT d$distance] /"
    else
      JXL_W44_76_PER_BLOCK_DUMP="$dump_dir" "$CJXL_PATCHED" \
        "$IMG" "$out" -e $EFFORT -d $distance --num_threads 1 --quiet 2>&1 \
        | tail -2 | sed "s/^/  [cjxl e$EFFORT d$distance] /"
    fi
    sz=$(wc -c < "$out" 2>/dev/null || echo "FAIL")
    dump_sz=$(du -sh "$dump_dir"/*.tsv 2>/dev/null | awk '{print $1}' || echo "MISSING")
    echo "  -> $side e$EFFORT d$distance: bytes=$sz dump=$dump_sz"
  done
done
