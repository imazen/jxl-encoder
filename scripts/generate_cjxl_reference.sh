#!/bin/bash
# Generate cjxl reference metrics CSV.
# Encodes images x 9 distances x 4 efforts, measures ssimulacra2 + butteraugli.
# Pre-converts to PNM so timing excludes PNG decode overhead.
# Uses fast-ssim2 (Rust) for SSIMULACRA2 measurement.
# Resumes from partial runs: skips rows already present in the CSV.
#
# Usage: bash scripts/generate_cjxl_reference.sh
# Output: reference/cjxl_reference.csv

set -euo pipefail

CJXL="${CJXL_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/cjxl}"
DJXL="${DJXL_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/djxl}"
SSIM2="${SSIMULACRA2_PATH:-$HOME/work/fast-ssim2/target/release/fast-ssim2-cli}"
BFLY="${BUTTERAUGLI_MAIN_PATH:-$HOME/work/jxl-efforts/libjxl/build/tools/butteraugli_main}"

CACHE_DIR="${JXL_ENCODER_OUTPUT_DIR:-/mnt/v/output/jxl-encoder-rs}/cjxl-reference"
STRIP_DIR="/tmp/cjxl-ref-stripped"
PNM_DIR="/tmp/cjxl-ref-pnm"
DECODED_DIR="/tmp/cjxl-ref-decoded"
TIMING_FILE="/tmp/cjxl-ref-timing.txt"

OUTCSV="reference/cjxl_reference.csv"

DISTANCES="0.25 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0"
EFFORTS="5 6 7 8"

# Corpora
CORPUS="${CODEC_CORPUS_DIR:-$HOME/work/codec-corpus}"
declare -A CORPUS_DIRS
CORPUS_DIRS[clic2025]="${CORPUS}/clic2025-1024"
CORPUS_DIRS[cid22]="${CORPUS}/CID22/CID22-512/validation"
CORPUS_DIRS[gb82-sc]="${CORPUS}/gb82-sc"
CORPUS_DIRS[cid22-train]="${CORPUS}/CID22/CID22-512/training"
CORPUS_DIRS[frymire]="$HOME/work/jxl-encoder-rs/jxl_encoder/tests/images"

CORPORA="clic2025 cid22 gb82-sc cid22-train frymire"

# Verify tools exist
for tool in "$CJXL" "$DJXL" "$SSIM2" "$BFLY"; do
    if [ ! -x "$tool" ]; then
        echo "ERROR: tool not found: $tool" >&2
        exit 1
    fi
done

mkdir -p "$CACHE_DIR" "$STRIP_DIR" "$PNM_DIR" "$DECODED_DIR" "$(dirname "$OUTCSV")"

# Get cjxl version
CJXL_VERSION=$("$CJXL" --version 2>&1 | head -1 | awk '{print $2}')

# Count total work items
total=0
for corpus in $CORPORA; do
    for img in "${CORPUS_DIRS[$corpus]}"/*.png; do
        for d in $DISTANCES; do
            for e in $EFFORTS; do
                total=$((total + 1))
            done
        done
    done
done
echo "Total encodes: $total (cjxl $CJXL_VERSION)" >&2

# Load already-done keys for resume
declare -A DONE
if [ -f "$OUTCSV" ]; then
    while IFS=, read -r c i w h d e rest; do
        [[ "$c" == "#"* || "$c" == "corpus" ]] && continue
        DONE["${c}_${i}_${d}_${e}"]=1
    done < "$OUTCSV"
    echo "Resuming: ${#DONE[@]} rows already present" >&2
else
    # Write header for new file
    echo "# cjxl $CJXL_VERSION, generated $(date +%Y-%m-%d)" > "$OUTCSV"
    echo "corpus,image,width,height,distance,effort,size_bytes,ssimulacra2,butteraugli,enc_wall_s,enc_user_s,enc_sys_s" >> "$OUTCSV"
fi

n=0
skipped=0
for corpus in $CORPORA; do
    for img in "${CORPUS_DIRS[$corpus]}"/*.png; do
        name=$(basename "$img" .png)
        short="${name:0:8}"

        # Get dimensions
        dims=$(identify -format "%w %h" "$img")
        width=${dims%% *}
        height=${dims##* }

        # Strip PNG metadata once (for metric comparison)
        stripped="$STRIP_DIR/${corpus}_${short}.png"
        if [ ! -f "$stripped" ]; then
            convert "$img" -background white -flatten -strip "$stripped"
        fi

        # Convert to PNM once (for encode input — excludes PNG decode from timing)
        pnm="$PNM_DIR/${corpus}_${short}.pnm"
        if [ ! -f "$pnm" ]; then
            convert "$img" -background white -flatten -strip "$pnm"
        fi

        for d in $DISTANCES; do
            for e in $EFFORTS; do
                n=$((n + 1))
                key="${corpus}_${short}_${d}_${e}"

                # Resume support: skip already-done rows
                if [[ -v "DONE[$key]" ]]; then
                    skipped=$((skipped + 1))
                    continue
                fi

                printf "\r[%d/%d] %s/%s d=%s e%s  " "$n" "$total" "$corpus" "$short" "$d" "$e" >&2

                jxl="$CACHE_DIR/${corpus}_${short}_d${d}_e${e}.jxl"

                # Encode with timing
                /usr/bin/time -f "%e %U %S" -o "$TIMING_FILE" \
                    "$CJXL" "$pnm" "$jxl" -d "$d" -e "$e" 2>/dev/null

                read -r wall user sys < "$TIMING_FILE"

                size=$(stat -c%s "$jxl")

                # Decode JXL to PNG for fast-ssim2 (can't read JXL directly)
                decoded="$DECODED_DIR/${corpus}_${short}_d${d}_e${e}.png"
                "$DJXL" "$jxl" "$decoded" 2>/dev/null

                # Strip decoded PNG to remove color metadata (prevents TF mismatch:
                # cjxl declares gamma(0.4545), stripped source assumes sRGB — comparing
                # two stripped PNGs ensures both are linearized with the same sRGB TF)
                decoded_stripped="${decoded%.png}_stripped.png"
                convert "$decoded" -strip "$decoded_stripped"

                # ssimulacra2 via fast-ssim2: "Score: 86.28667631"
                ssim2=$("$SSIM2" image "$stripped" "$decoded_stripped" 2>/dev/null | grep -oP '[\d.]+')

                # butteraugli: compare stripped source vs stripped decoded (not JXL directly!)
                bfly=$("$BFLY" "$stripped" "$decoded_stripped" 2>/dev/null | head -1 | tr -d '[:space:]')

                # Clean up decoded PNGs
                rm -f "$decoded" "$decoded_stripped"

                echo "${corpus},${short},${width},${height},${d},${e},${size},${ssim2},${bfly},${wall},${user},${sys}" >> "$OUTCSV"
            done
        done
    done
done

echo "" >&2
done_count=${#DONE[@]:-0}
echo "Done. $((n - skipped)) new rows written ($skipped skipped). Total in CSV: $((n - skipped + done_count))" >&2
echo "Output: $OUTCSV" >&2
