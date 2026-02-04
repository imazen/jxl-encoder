#!/bin/bash
set -e

CWEBP=~/work/libwebp/examples/cwebp
DWEBP=~/work/libwebp/examples/dwebp
SSIM2=~/work/fast-ssim2/target/release/fast-ssim2-cli
CJXL=~/work/jxl-encoder-rs/target/debug/cjxl-rs
DJXL=~/work/jxl-efforts/libjxl/build/tools/djxl
IMGDIR=~/work/codec-corpus/CID22/CID22-512/validation
TMPDIR=/tmp/rd_compare_$$
mkdir -p "$TMPDIR"

JXL_DISTS=(0.5 1.0 2.0 4.0)
WEBP_QS=(95 85 70 40)

echo "=== cjxl-rs vs libwebp 1.6.0 — CID22 validation (41 images, 512x512) ==="
echo ""

for di in 0 1 2 3; do
    jxl_d=${JXL_DISTS[$di]}
    webp_q=${WEBP_QS[$di]}

    jxl_total_size=0
    webp_total_size=0
    count=0
    jxl_ssim2_sum="0"
    webp_ssim2_sum="0"

    for img in "$IMGDIR"/*.png; do
        name=$(basename "$img" .png)

        # JXL encode + decode
        "$CJXL" "$img" "$TMPDIR/${name}.jxl" -d "$jxl_d" >/dev/null 2>&1
        "$DJXL" "$TMPDIR/${name}.jxl" "$TMPDIR/${name}_jxl.png" >/dev/null 2>&1
        jxl_size=$(stat -c%s "$TMPDIR/${name}.jxl")
        jxl_s=$("$SSIM2" image "$img" "$TMPDIR/${name}_jxl.png" 2>/dev/null | sed 's/Score: //')

        # WebP encode + decode
        "$CWEBP" -q "$webp_q" "$img" -o "$TMPDIR/${name}.webp" >/dev/null 2>&1
        "$DWEBP" "$TMPDIR/${name}.webp" -o "$TMPDIR/${name}_webp.png" >/dev/null 2>&1
        webp_size=$(stat -c%s "$TMPDIR/${name}.webp")
        webp_s=$("$SSIM2" image "$img" "$TMPDIR/${name}_webp.png" 2>/dev/null | sed 's/Score: //')

        jxl_total_size=$((jxl_total_size + jxl_size))
        webp_total_size=$((webp_total_size + webp_size))
        jxl_ssim2_sum=$(echo "$jxl_ssim2_sum + $jxl_s" | bc -l)
        webp_ssim2_sum=$(echo "$webp_ssim2_sum + $webp_s" | bc -l)
        count=$((count + 1))

        rm -f "$TMPDIR/${name}.jxl" "$TMPDIR/${name}_jxl.png" "$TMPDIR/${name}.webp" "$TMPDIR/${name}_webp.png"
    done

    jxl_avg_ssim2=$(echo "scale=2; $jxl_ssim2_sum / $count" | bc)
    webp_avg_ssim2=$(echo "scale=2; $webp_ssim2_sum / $count" | bc)
    jxl_avg_kb=$(echo "scale=1; $jxl_total_size / $count / 1024" | bc)
    webp_avg_kb=$(echo "scale=1; $webp_total_size / $count / 1024" | bc)
    size_pct=$(echo "scale=1; ($jxl_total_size - $webp_total_size) * 100 / $webp_total_size" | bc)

    printf "jxl d=%-3s  %5s KB  SSIM2 %5s  |  webp q=%-2s  %5s KB  SSIM2 %5s  |  size %+5s%%  ssim2 %+5s\n" \
        "$jxl_d" "$jxl_avg_kb" "$jxl_avg_ssim2" "$webp_q" "$webp_avg_kb" "$webp_avg_ssim2" \
        "$size_pct" "$(echo "$jxl_avg_ssim2 - $webp_avg_ssim2" | bc)"
done

rm -rf "$TMPDIR"
