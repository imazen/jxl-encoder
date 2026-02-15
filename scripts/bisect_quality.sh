#!/bin/bash
# Bisect which feature causes catastrophic butteraugli at d>=2.0
# Tests one image at d=2.0 with various features disabled

set -euo pipefail

CJXL_RS="/home/lilith/work/jxl-encoder-rs/target/release/cjxl-rs"
DJXL="/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl"
BFLY="/home/lilith/work/butteraugli/target/release/butteraugli"
OUTDIR="/mnt/v/output/jxl-encoder-rs/bisect"
mkdir -p "$OUTDIR"

IMG="/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
DIST="${1:-2.0}"

echo "Testing image 02809272 at d=$DIST"
echo "==========================================="
printf "%-40s %8s %8s\n" "Configuration" "Size" "Bfly"
echo "-------------------------------------------"

run_test() {
    local label="$1"
    shift
    local out="$OUTDIR/${label}_d${DIST}.jxl"
    local dec="$OUTDIR/${label}_d${DIST}.png"

    "$CJXL_RS" "$IMG" "$out" -d "$DIST" "$@" 2>/dev/null
    "$DJXL" "$out" "$dec" 2>/dev/null

    local size=$(stat -c%s "$out")
    local bfly=$("$BFLY" --quiet "$IMG" "$dec" 2>/dev/null || echo "ERR")

    printf "%-40s %8d %8s\n" "$label" "$size" "$bfly"
}

# Baseline (all defaults)
run_test "defaults"

# Disable butteraugli loop
run_test "no-butteraugli" --no-butteraugli

# Disable gaborish
run_test "no-gaborish" --no-gaborish

# Disable pixel-domain loss
run_test "no-pixel-domain" --no-pixel-domain-loss

# Disable error diffusion
run_test "no-error-diffusion" --no-error-diffusion

# DCT8 only (disable large strategies)
run_test "dct8-only" --dct8-only

# No custom orders
run_test "no-custom-orders" --no-custom-orders

# No ANS (use Huffman)
run_test "no-ans" --no-ans

# Combinations
run_test "dct8+no-bfly" --dct8-only --no-butteraugli
run_test "dct8+no-gab" --dct8-only --no-gaborish
run_test "dct8+no-pixel" --dct8-only --no-pixel-domain-loss
run_test "no-bfly+no-gab" --no-butteraugli --no-gaborish
run_test "no-bfly+no-pixel" --no-butteraugli --no-pixel-domain-loss
run_test "minimal" --dct8-only --no-butteraugli --no-gaborish --no-pixel-domain-loss --no-error-diffusion --no-custom-orders

echo ""
echo "Testing 3 more catastrophic images at d=$DIST..."
echo "==========================================="

for img_path in \
    "/home/lilith/work/codec-corpus/clic2025-1024/50fe4c3d47d864858e1aaa60fecef5c453b4e18d2b368718eeb5c1e249e0c902.png" \
    "/home/lilith/work/codec-corpus/clic2025-1024/bb7344a2ba499b2d48b891abee1b903dc17d265437ac57028b5999b6cd5bcdc4.png" \
    "/home/lilith/work/codec-corpus/clic2025-1024/870516c65d81fb9267de6865964083a9.png"; do

    short=$(basename "$img_path" .png | cut -c1-8)
    echo ""
    echo "Image: $short"
    printf "%-40s %8s %8s\n" "Configuration" "Size" "Bfly"
    echo "-------------------------------------------"

    for config_label in "defaults" "no-butteraugli" "dct8-only" "dct8+no-bfly"; do
        out="$OUTDIR/${short}_${config_label}_d${DIST}.jxl"
        dec="$OUTDIR/${short}_${config_label}_d${DIST}.png"

        case "$config_label" in
            "defaults") flags="" ;;
            "no-butteraugli") flags="--no-butteraugli" ;;
            "dct8-only") flags="--dct8-only" ;;
            "dct8+no-bfly") flags="--dct8-only --no-butteraugli" ;;
        esac

        "$CJXL_RS" "$img_path" "$out" -d "$DIST" $flags 2>/dev/null
        "$DJXL" "$out" "$dec" 2>/dev/null

        size=$(stat -c%s "$out")
        bfly=$("$BFLY" --quiet "$img_path" "$dec" 2>/dev/null || echo "ERR")

        printf "%-40s %8d %8s\n" "$config_label" "$size" "$bfly"
    done
done
