#!/bin/bash
# Lossless compression comparison: cjxl-rs vs cjxl at effort 7
# Outputs CSV to stdout for analysis

set -euo pipefail

CJXL_RS="${JXL_CLI_PATH:-./target/release/cjxl-rs}"
CJXL="${CJXL_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/cjxl}"
DJXL="${DJXL_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/djxl}"
OUTDIR="${JXL_ENCODER_OUTPUT_DIR:-/mnt/v/output/jxl-encoder-rs}/lossless-parity"

# Extra flags for cjxl-rs (passed as arguments to this script)
EXTRA_FLAGS="${@}"

CORPUS="${CODEC_CORPUS_DIR:-$HOME/work/codec-corpus}"
CLIC_DIR="${CORPUS}/clic2025-1024"
SC_DIR="${CORPUS}/gb82-sc"

# Pick 8 CLIC photos (first 8 non-pareto files)
CLIC_IMAGES=$(ls "$CLIC_DIR"/*.png | grep -v pareto | head -8)

# Pick key screenshots
SC_IMAGES="$SC_DIR/imac_dark.png $SC_DIR/windows.png $SC_DIR/codec_wiki.png $SC_DIR/terminal.png"

echo "image,source_bytes,cjxl_e7_bytes,cjxl_rs_bytes,ratio_vs_cjxl,pct_larger"

for img in $CLIC_IMAGES $SC_IMAGES; do
    name=$(basename "$img" .png)
    src_size=$(stat -c%s "$img")

    # Encode with cjxl e7 lossless
    cjxl_out="$OUTDIR/${name}_cjxl_e7.jxl"
    $CJXL "$img" "$cjxl_out" -d 0 -e 7 --quiet 2>/dev/null || true
    cjxl_size=$(stat -c%s "$cjxl_out" 2>/dev/null || echo 0)

    # Encode with our encoder (default effort 7 lossless)
    rs_out="$OUTDIR/${name}_cjxl_rs.jxl"
    $CJXL_RS --lossless "$img" "$rs_out" $EXTRA_FLAGS 2>/dev/null || true
    rs_size=$(stat -c%s "$rs_out" 2>/dev/null || echo 0)

    if [ "$cjxl_size" -gt 0 ] && [ "$rs_size" -gt 0 ]; then
        ratio=$(echo "scale=4; $rs_size / $cjxl_size" | bc)
        pct=$(echo "scale=1; ($rs_size - $cjxl_size) * 100 / $cjxl_size" | bc)
        echo "$name,$src_size,$cjxl_size,$rs_size,$ratio,$pct"
    else
        echo "$name,$src_size,$cjxl_size,$rs_size,ERROR,ERROR"
    fi
done
