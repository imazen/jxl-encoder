#!/bin/bash
# Test each AC strategy individually to find which ones produce garbage
set -euo pipefail

CJXL_RS="${JXL_CLI_PATH:-/home/lilith/work/jxl-encoder-rs/target/release/cjxl-rs}"
DJXL="${DJXL_PATH:-/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl}"
BFLY="${BUTTERAUGLI_PATH:-/home/lilith/work/butteraugli/target/release/butteraugli}"
OUTDIR="${JXL_ENCODER_OUTPUT_DIR:-/mnt/v/output/jxl-encoder-rs}/bisect_strategy"
mkdir -p "$OUTDIR"

CORPUS="${CODEC_CORPUS_DIR:-/home/lilith/work/codec-corpus}"
IMG="${CORPUS}/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
DIST="${1:-2.0}"

echo "Testing forced strategies on 02809272 at d=$DIST"
printf "%-20s %8s %8s\n" "Strategy" "Size" "Bfly"
echo "-------------------------------------------"

# Strategies: 0=DCT8, 1=IDENTITY, 2=DCT2X2, 3=DCT4X4, 4=DCT16X8, 5=DCT8X16,
# 6=DCT16X16, 7=DCT32X16, 8=DCT16X32, 9=DCT32X32, 10=DCT4X8, 11=DCT8X4,
# 12=AFV0, 13=AFV1, 14=AFV2, 15=AFV3, 16=DCT64X64, 17=DCT64X32, 18=DCT32X64

for strat in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
    out="$OUTDIR/strat${strat}_d${DIST}.jxl"
    dec="$OUTDIR/strat${strat}_d${DIST}.png"

    if "$CJXL_RS" "$IMG" "$out" -d "$DIST" --force-strategy "$strat" 2>/dev/null; then
        if "$DJXL" "$out" "$dec" 2>/dev/null; then
            size=$(stat -c%s "$out")
            bfly=$("$BFLY" --quiet "$IMG" "$dec" 2>/dev/null || echo "ERR")
            printf "%-20s %8d %8s\n" "strat$strat" "$size" "$bfly"
        else
            printf "%-20s %8s %8s\n" "strat$strat" "DECODE" "FAIL"
        fi
    else
        printf "%-20s %8s %8s\n" "strat$strat" "ENCODE" "FAIL"
    fi
done
