#!/bin/bash
# Verify the strategy disable fix on all previously-catastrophic images
set -euo pipefail

CJXL_RS="${JXL_CLI_PATH:-/home/lilith/work/jxl-encoder-rs/target/release/cjxl-rs}"
DJXL="${DJXL_PATH:-/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl}"
BFLY="${BUTTERAUGLI_PATH:-/home/lilith/work/butteraugli/target/release/butteraugli}"
OUTDIR="${JXL_ENCODER_OUTPUT_DIR:-/mnt/v/output/jxl-encoder-rs}/verify_fix"
mkdir -p "$OUTDIR"

CORPUS="${CODEC_CORPUS_DIR:-/home/lilith/work/codec-corpus}"

printf "%-10s %6s %10s %10s\n" "Image" "Dist" "Size" "Bfly"
echo "==========================================="

for img_path in \
    "${CORPUS}/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png" \
    "${CORPUS}/clic2025-1024/50fe4c3d47d864858e1aaa60fecef5c453b4e18d2b368718eeb5c1e249e0c902.png" \
    "${CORPUS}/clic2025-1024/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png" \
    "${CORPUS}/clic2025-1024/870516c65d81fb9267de6865964083a9.png" \
    "${CORPUS}/clic2025-1024/bb7344a2ba499b2d48b891abee1b903dc17d265437ac57028b5999b6cd5bcdc4.png" \
    "${CORPUS}/clic2025-1024/1b4ad095795ac552b38a21d51be7bfaee8e7d0a70619d84767814321df4ed062.png" \
    "${CORPUS}/clic2025-1024/a36713f1943dac6bc74dea50cadaee6f.png" \
    "${CORPUS}/clic2025-1024/d1a9be98d1936065967adac50a6fb750.png" \
    "${CORPUS}/clic2025-1024/ddcd24d99f48eaa369207882a6f37831.png"; do

    short=$(basename "$img_path" .png | cut -c1-8)

    for dist in 0.5 1.0 2.0 3.0; do
        out="$OUTDIR/${short}_d${dist}.jxl"
        dec="$OUTDIR/${short}_d${dist}.png"

        "$CJXL_RS" "$img_path" "$out" -d "$dist" 2>/dev/null
        "$DJXL" "$out" "$dec" 2>/dev/null

        size=$(stat -c%s "$out")
        bfly=$("$BFLY" --quiet "$img_path" "$dec" 2>/dev/null || echo "ERR")

        printf "%-10s %6s %10d %10s\n" "$short" "$dist" "$size" "$bfly"
    done
done
