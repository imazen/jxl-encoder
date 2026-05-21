#!/usr/bin/env bash
# W44-147: per-cell strategy dump runner for photo cluster (top 5 SSIM2 deficits).
#
# Mirrors W44-103 / W44-121 / W44-144 methodology. Re-runs ours and cjxl
# encodes in separate processes per cell so that the W44-76 dump
# infrastructure (initializes once per process) writes a fresh TSV for
# each (image, effort, distance) cell.
#
# Output: /tmp/w44_147_dumps/{img}_e{effort}_d{distance}_{ours|cjxl}/per_block_*.tsv

set -euo pipefail

CORPUS=/home/lilith/work/codec-corpus/CID22/CID22-512/validation
CJXL_RS=/home/lilith/work/zen/jxl-encoder-shared-target/release/cjxl-rs
CJXL_PATCHED=/home/lilith/work/jxl-efforts/libjxl--w44-76-per-block-debug-dump/build/tools/cjxl

rm -rf /tmp/w44_147_dumps
mkdir -p /tmp/w44_147_dumps

# Top 5 photo cells by SSIM2 deficit, w/ mix of efforts:
#   #1 1418519 e8 d=5     -> -2.57 (buttloop fires)
#   #2 1531677 e7 d=5     -> -2.00 AT-PARITY bytes (most actionable)
#   #3 1420710 e8 d=6     -> -2.13 (buttloop)
#   #4 1531677 e5 d=3     -> -1.32 AT-PARITY bytes, low-effort path
#   #5 1418519 e7 d=5     -> -2.39 (no buttloop, e7 cost-model)
#
# Format: img:effort:distance
CELLS=(
  "1418519.png:8:5"
  "1531677.png:7:5"
  "1420710.png:8:6"
  "1531677.png:5:3"
  "1418519.png:7:5"
)

for cell in "${CELLS[@]}"; do
  IFS=':' read -r img eff dist <<< "$cell"
  base="${img%.png}"
  src="$CORPUS/$img"
  if [ ! -f "$src" ]; then
    echo "MISSING source: $src" >&2
    continue
  fi
  for side in ours cjxl; do
    dump_dir=/tmp/w44_147_dumps/${base}_e${eff}_d${dist}_${side}
    mkdir -p "$dump_dir"
    out=/tmp/w44_147_${base}_${side}_e${eff}_d${dist}.jxl
    rm -f "$out"
    if [ "$side" = ours ]; then
      JXL_W44_76_PER_BLOCK_DUMP="$dump_dir" "$CJXL_RS" \
        --effort $eff --distance $dist --threads 1 \
        "$src" "$out" 2>&1 | tail -2 | sed "s/^/  [ours $base e$eff d$dist] /"
    else
      JXL_W44_76_PER_BLOCK_DUMP="$dump_dir" "$CJXL_PATCHED" \
        "$src" "$out" -e $eff -d $dist --num_threads 1 --quiet 2>&1 \
        | tail -2 | sed "s/^/  [cjxl $base e$eff d$dist] /" || true
    fi
    sz=$(wc -c < "$out" 2>/dev/null || echo "FAIL")
    dump_sz=$(du -sh "$dump_dir"/*.tsv 2>/dev/null | awk '{print $1}' || echo "MISSING")
    echo "  -> $side $base e$eff d$dist: bytes=$sz dump=$dump_sz"
  done
done
