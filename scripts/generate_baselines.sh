#!/bin/bash
# Generate baseline metrics from libjxl for parity testing

set -e

CJXL=~/work/jxl-efforts/libjxl/build/tools/cjxl
DJXL=~/work/jxl-efforts/libjxl/build/tools/djxl
SSIMULACRA2=~/work/jpegli-rs/internal/jpegli-cpp/jpegli-rs/ssimulacra2-fork/ssimulacra2
BUTTERAUGLI=~/work/jpegli-rs/internal/jpegli-cpp/jpegli-rs/butteraugli/butteraugli
DSSIM=~/.cargo/bin/dssim
CORPUS=~/work/codec-corpus/kodak
OUTPUT=~/work/jxl-encoder-rs/test_baselines

mkdir -p "$OUTPUT/jxl" "$OUTPUT/decoded"

echo "image,distance,effort,file_size,ssimulacra2,butteraugli,dssim" > "$OUTPUT/baselines.csv"

# Test configurations
DISTANCES="0.0 1.0 2.0"
EFFORTS="3 7"

# Use a subset of Kodak images for quick testing
IMAGES="10.png 11.png 12.png 13.png 14.png"

for img in $IMAGES; do
    src="$CORPUS/$img"
    base="${img%.png}"

    if [ ! -f "$src" ]; then
        echo "Skipping $img - not found"
        continue
    fi

    echo "Processing $img..."

    for dist in $DISTANCES; do
        for effort in $EFFORTS; do
            jxl_file="$OUTPUT/jxl/${base}_d${dist}_e${effort}.jxl"
            dec_file="$OUTPUT/decoded/${base}_d${dist}_e${effort}.png"

            # Encode
            $CJXL "$src" "$jxl_file" -d "$dist" -e "$effort" --quiet 2>/dev/null

            # Get file size
            file_size=$(stat -c%s "$jxl_file")

            # Decode
            $DJXL "$jxl_file" "$dec_file" --quiet 2>/dev/null

            # Compute metrics
            if [ "$dist" = "0.0" ]; then
                # Lossless - should be perfect
                ssim2="100.0"
                butter="0.0"
                dssim_val="0.0"
            else
                # Compute SSIMULACRA2
                if [ -x "$SSIMULACRA2" ]; then
                    ssim2=$($SSIMULACRA2 "$src" "$dec_file" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1) || ssim2="N/A"
                else
                    ssim2="N/A"
                fi

                # Compute Butteraugli
                if [ -x "$BUTTERAUGLI" ]; then
                    butter=$($BUTTERAUGLI "$src" "$dec_file" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1) || butter="N/A"
                else
                    butter="N/A"
                fi

                # Compute DSSIM
                if [ -x "$DSSIM" ]; then
                    dssim_val=$($DSSIM "$src" "$dec_file" 2>/dev/null | awk '{print $1}') || dssim_val="N/A"
                else
                    dssim_val="N/A"
                fi
            fi

            echo "$base,$dist,$effort,$file_size,$ssim2,$butter,$dssim_val" >> "$OUTPUT/baselines.csv"
            echo "  d=$dist e=$effort: ${file_size} bytes, ssim2=$ssim2, butter=$butter, dssim=$dssim_val"
        done
    done
done

echo ""
echo "Baselines written to $OUTPUT/baselines.csv"
cat "$OUTPUT/baselines.csv"
