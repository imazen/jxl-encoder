#!/bin/bash
# Benchmark comparison: cjxl-rs vs libjxl (cjxl)
# Usage: ./scripts/benchmark_comparison.sh [num_images] [distances...]
# Example: ./scripts/benchmark_comparison.sh 10 1.0 2.0 4.0

set -e

# Tools
CJXL_RS="./target/release/cjxl-rs"
CJXL=~/work/jxl-efforts/libjxl/build/tools/cjxl
DJXL=~/work/jxl-efforts/libjxl/build/tools/djxl
SSIM2=~/work/jxl-efforts/libjxl/build/tools/ssimulacra2

# Corpus paths
CLIC_DIR=~/work/codec-corpus/clic2025/final-test
CID_DIR=~/work/codec-corpus/CID22/CID22-512/validation

# Output directory
OUT_DIR=/tmp/jxl_benchmark_$$
mkdir -p "$OUT_DIR"

# Parameters
NUM_IMAGES=${1:-5}
shift || true
DISTANCES="${@:-1.0 2.0 4.0}"

echo "=== JXL Encoder Benchmark ==="
echo "cjxl-rs: $CJXL_RS"
echo "cjxl:    $CJXL"
echo "Output:  $OUT_DIR"
echo "Images:  $NUM_IMAGES from each corpus"
echo "Distances: $DISTANCES"
echo ""

# Collect images
CLIC_IMAGES=($(ls "$CLIC_DIR"/*.png | head -n "$NUM_IMAGES"))
CID_IMAGES=($(ls "$CID_DIR"/*.png | head -n "$NUM_IMAGES"))

# Results file
RESULTS="$OUT_DIR/results.csv"
echo "corpus,image,distance,encoder,size_bytes,ssim2,encode_ms" > "$RESULTS"

encode_and_measure() {
    local corpus=$1
    local img=$2
    local dist=$3
    local encoder=$4
    local encoder_cmd=$5
    local out_jxl="$OUT_DIR/$(basename "$img" .png)_${encoder}_d${dist}.jxl"
    local out_png="$OUT_DIR/$(basename "$img" .png)_${encoder}_d${dist}_decoded.png"

    # Encode with timing
    local start_ms=$(date +%s%3N)
    if [[ "$encoder" == "cjxl-rs" ]]; then
        $encoder_cmd -d "$dist" "$img" "$out_jxl" 2>/dev/null
    else
        $encoder_cmd -d "$dist" "$img" "$out_jxl" 2>/dev/null
    fi
    local end_ms=$(date +%s%3N)
    local encode_ms=$((end_ms - start_ms))

    # Get file size
    local size=$(stat -c%s "$out_jxl")

    # Decode and measure SSIM2
    $DJXL "$out_jxl" "$out_png" 2>/dev/null
    local ssim2=$($SSIM2 "$img" "$out_png" 2>/dev/null | grep -oP '[\d.]+$' || echo "ERROR")

    # Clean up decoded PNG
    rm -f "$out_png"

    echo "$corpus,$(basename "$img"),${dist},${encoder},${size},${ssim2},${encode_ms}" >> "$RESULTS"
    echo "  $encoder: ${size} bytes, SSIM2=${ssim2}, ${encode_ms}ms"
}

# Process CLIC images
echo "=== CLIC 2025 (${#CLIC_IMAGES[@]} images) ==="
for img in "${CLIC_IMAGES[@]}"; do
    echo ""
    echo "Processing: $(basename "$img")"
    dims=$(file "$img" | grep -oP '\d+ x \d+')
    echo "  Dimensions: $dims"

    for dist in $DISTANCES; do
        echo "  Distance $dist:"
        encode_and_measure "clic" "$img" "$dist" "cjxl" "$CJXL"
        encode_and_measure "clic" "$img" "$dist" "cjxl-rs" "$CJXL_RS"
    done
done

# Process CID images
echo ""
echo "=== CID22-512 (${#CID_IMAGES[@]} images) ==="
for img in "${CID_IMAGES[@]}"; do
    echo ""
    echo "Processing: $(basename "$img")"

    for dist in $DISTANCES; do
        echo "  Distance $dist:"
        encode_and_measure "cid" "$img" "$dist" "cjxl" "$CJXL"
        encode_and_measure "cid" "$img" "$dist" "cjxl-rs" "$CJXL_RS"
    done
done

echo ""
echo "=== Summary ==="
echo ""

# Generate summary
python3 << 'PYEOF'
import csv
import sys
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))

with open("'$RESULTS'") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['corpus'], row['distance'])
        results[key][row['encoder']].append({
            'size': int(row['size_bytes']),
            'ssim2': float(row['ssim2']) if row['ssim2'] != 'ERROR' else None,
            'time': int(row['encode_ms'])
        })

print(f"{'Corpus':<8} {'Dist':<6} {'Metric':<12} {'cjxl':<12} {'cjxl-rs':<12} {'Diff':<12}")
print("-" * 62)

for (corpus, dist), encoders in sorted(results.items()):
    cjxl_data = encoders.get('cjxl', [])
    rs_data = encoders.get('cjxl-rs', [])

    if not cjxl_data or not rs_data:
        continue

    # Average sizes
    cjxl_size = sum(d['size'] for d in cjxl_data) / len(cjxl_data)
    rs_size = sum(d['size'] for d in rs_data) / len(rs_data)
    size_diff = (rs_size - cjxl_size) / cjxl_size * 100

    # Average SSIM2
    cjxl_ssim = [d['ssim2'] for d in cjxl_data if d['ssim2'] is not None]
    rs_ssim = [d['ssim2'] for d in rs_data if d['ssim2'] is not None]
    cjxl_ssim_avg = sum(cjxl_ssim) / len(cjxl_ssim) if cjxl_ssim else 0
    rs_ssim_avg = sum(rs_ssim) / len(rs_ssim) if rs_ssim else 0
    ssim_diff = rs_ssim_avg - cjxl_ssim_avg

    # Average time
    cjxl_time = sum(d['time'] for d in cjxl_data) / len(cjxl_data)
    rs_time = sum(d['time'] for d in rs_data) / len(rs_data)
    time_ratio = rs_time / cjxl_time if cjxl_time > 0 else 0

    print(f"{corpus:<8} {dist:<6} {'Size (B)':<12} {cjxl_size:<12.0f} {rs_size:<12.0f} {size_diff:+.1f}%")
    print(f"{'':<8} {'':<6} {'SSIM2':<12} {cjxl_ssim_avg:<12.2f} {rs_ssim_avg:<12.2f} {ssim_diff:+.2f}")
    print(f"{'':<8} {'':<6} {'Time (ms)':<12} {cjxl_time:<12.0f} {rs_time:<12.0f} {time_ratio:.1f}x")
    print()

PYEOF

echo "Raw results: $RESULTS"
echo "JXL files: $OUT_DIR/*.jxl"
