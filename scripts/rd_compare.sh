#!/bin/bash
# Compare jxl-encoder (cjxl-rs) vs libjxl (cjxl) across distances and efforts
# Measures file size and butteraugli score for each combination

set -euo pipefail

CJXL_RS="/home/lilith/work/jxl-encoder-rs/target/release/cjxl-rs"
CJXL="/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl"
DJXL="/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl"
BFLY="/home/lilith/work/butteraugli/target/release/butteraugli"
OUTDIR="/mnt/v/output/jxl-encoder-rs/comparison"

DISTANCES="0.5 1.0 2.0 3.0"
EFFORTS="1 3 5 7"

IMAGES=(
    "/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/1b4ad095795ac552b38a21d51be7bfaee8e7d0a70619d84767814321df4ed062.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/50fe4c3d47d864858e1aaa60fecef5c453b4e18d2b368718eeb5c1e249e0c902.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/870516c65d81fb9267de6865964083a9.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/a36713f1943dac6bc74dea50cadaee6f.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/bb7344a2ba499b2d48b891abee1b903dc17d265437ac57028b5999b6cd5bcdc4.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/d1a9be98d1936065967adac50a6fb750.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/ddcd24d99f48eaa369207882a6f37831.png"
    "/home/lilith/work/codec-corpus/clic2025-1024/e0d8e29cadfc99663c7d1a4a5afe20c454ec54d0d873776ec397c59405c74790.png"
)

mkdir -p "$OUTDIR"

# CSV header
echo "image,distance,encoder,effort,size_bytes,butteraugli"

for img in "${IMAGES[@]}"; do
    name=$(basename "$img" .png)
    short="${name:0:8}"

    for dist in $DISTANCES; do
        # --- Our encoder ---
        rs_out="$OUTDIR/${short}_d${dist}_rs.jxl"
        rs_dec="$OUTDIR/${short}_d${dist}_rs.png"

        "$CJXL_RS" "$img" "$rs_out" -d "$dist" 2>/dev/null
        "$DJXL" "$rs_out" "$rs_dec" 2>/dev/null

        rs_size=$(stat -c%s "$rs_out")
        rs_bfly=$("$BFLY" --quiet "$img" "$rs_dec" 2>/dev/null || echo "ERR")

        echo "${short},${dist},cjxl-rs,-,${rs_size},${rs_bfly}"

        # --- cjxl at each effort ---
        for effort in $EFFORTS; do
            c_out="$OUTDIR/${short}_d${dist}_e${effort}.jxl"
            c_dec="$OUTDIR/${short}_d${dist}_e${effort}.png"

            "$CJXL" "$img" "$c_out" -d "$dist" -e "$effort" 2>/dev/null
            "$DJXL" "$c_out" "$c_dec" 2>/dev/null

            c_size=$(stat -c%s "$c_out")
            c_bfly=$("$BFLY" --quiet "$img" "$c_dec" 2>/dev/null || echo "ERR")

            echo "${short},${dist},cjxl,${effort},${c_size},${c_bfly}"
        done
    done
done
