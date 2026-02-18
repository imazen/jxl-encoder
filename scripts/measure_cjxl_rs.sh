#!/bin/bash
# Measure cjxl-rs against the same images/distances as the cjxl reference CSV.
# Pre-converts to PNM so timing excludes PNG decode overhead.
# Uses fast-ssim2 (Rust) for SSIMULACRA2 measurement.
#
# Usage: bash scripts/measure_cjxl_rs.sh [--quick]
# Output: reference/cjxl_rs_latest.csv
#
# --quick: only first 10 CLIC + all gb82-sc + first 5 CID22 (~25 images x 9 distances)

set -euo pipefail

CJXL_RS="$(pwd)/target/release/cjxl-rs"
DJXL="$HOME/work/jxl-efforts/libjxl/build/tools/djxl"
SSIM2="$HOME/work/fast-ssim2/target/release/fast-ssim2-cli"
BFLY="$HOME/work/jxl-efforts/libjxl/build/tools/butteraugli_main"

CACHE_DIR="/mnt/v/output/jxl-encoder-rs/cjxl-rs-latest"
STRIP_DIR="/tmp/cjxl-ref-stripped"
PNM_DIR="/tmp/cjxl-ref-pnm"
DECODED_DIR="/tmp/cjxl-rs-decoded"
TIMING_FILE="/tmp/cjxl-rs-timing.txt"

REF_CSV="reference/cjxl_reference.csv"
OUTCSV="reference/cjxl_rs_latest.csv"

DISTANCES="0.25 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0"

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=1
fi

# Verify tools
if [ ! -x "$CJXL_RS" ]; then
    echo "ERROR: cjxl-rs not found at $CJXL_RS" >&2
    echo "Run: cargo build --release -p jxl-encoder-cli" >&2
    exit 1
fi
for tool in "$DJXL" "$SSIM2" "$BFLY"; do
    if [ ! -x "$tool" ]; then
        echo "ERROR: tool not found: $tool" >&2
        exit 1
    fi
done
if [ ! -f "$REF_CSV" ]; then
    echo "ERROR: reference CSV not found: $REF_CSV" >&2
    echo "Run: just generate-reference" >&2
    exit 1
fi

# Corpora
declare -A CORPUS_DIRS
CORPUS_DIRS[clic2025]="$HOME/work/codec-corpus/clic2025-1024"
CORPUS_DIRS[cid22]="$HOME/work/codec-corpus/CID22/CID22-512/validation"
CORPUS_DIRS[gb82-sc]="$HOME/work/codec-corpus/gb82-sc"

mkdir -p "$CACHE_DIR" "$STRIP_DIR" "$PNM_DIR" "$DECODED_DIR"

# Extract unique (corpus, image, width, height) from reference CSV
declare -A IMAGES  # key=corpus_image, value="corpus image width height full_path"
while IFS=, read -r corpus image width height rest; do
    [[ "$corpus" == "#"* || "$corpus" == "corpus" ]] && continue
    key="${corpus}_${image}"
    if [[ ! -v "IMAGES[$key]" ]]; then
        # Find the full path
        dir="${CORPUS_DIRS[$corpus]}"
        full_path=$(ls "${dir}/${image}"*.png 2>/dev/null | head -1)
        if [ -n "$full_path" ]; then
            IMAGES["$key"]="${corpus} ${image} ${width} ${height} ${full_path}"
        fi
    fi
done < "$REF_CSV"

echo "Found ${#IMAGES[@]} unique images in reference CSV" >&2

# Apply --quick filter
declare -A SELECTED
clic_count=0
cid_count=0
for key in $(echo "${!IMAGES[@]}" | tr ' ' '\n' | sort); do
    IFS=' ' read -r corpus image width height full_path <<< "${IMAGES[$key]}"
    if [ "$QUICK" -eq 1 ]; then
        case "$corpus" in
            clic2025)
                if [ "$clic_count" -ge 10 ]; then continue; fi
                clic_count=$((clic_count + 1))
                ;;
            cid22)
                if [ "$cid_count" -ge 5 ]; then continue; fi
                cid_count=$((cid_count + 1))
                ;;
            gb82-sc) ;;  # always include all
        esac
    fi
    SELECTED["$key"]="${IMAGES[$key]}"
done

num_images=${#SELECTED[@]}
num_distances=$(echo $DISTANCES | wc -w)
total=$((num_images * num_distances))
echo "Encoding $num_images images x $num_distances distances = $total encodes" >&2

# Load already-done keys for resume
declare -A DONE
if [ -f "$OUTCSV" ]; then
    while IFS=, read -r c i w h d rest; do
        [[ "$c" == "#"* || "$c" == "corpus" ]] && continue
        DONE["${c}_${i}_${d}"]=1
    done < "$OUTCSV"
    echo "Resuming: ${#DONE[@]} rows already present" >&2
else
    CJXL_RS_VERSION=$("$CJXL_RS" --version 2>&1 | head -1 || echo "unknown")
    echo "# cjxl-rs $CJXL_RS_VERSION, generated $(date +%Y-%m-%d)" > "$OUTCSV"
    echo "corpus,image,width,height,distance,size_bytes,ssimulacra2,butteraugli,enc_wall_s,enc_user_s,enc_sys_s" >> "$OUTCSV"
fi

n=0
skipped=0
for key in $(echo "${!SELECTED[@]}" | tr ' ' '\n' | sort); do
    IFS=' ' read -r corpus image width height full_path <<< "${SELECTED[$key]}"

    # Ensure stripped PNG exists (for metric comparison)
    stripped="$STRIP_DIR/${corpus}_${image}.png"
    if [ ! -f "$stripped" ]; then
        convert "$full_path" -background white -flatten -strip "$stripped"
    fi

    # Ensure PNM exists (for encode input)
    pnm="$PNM_DIR/${corpus}_${image}.pnm"
    if [ ! -f "$pnm" ]; then
        convert "$full_path" -background white -flatten -strip "$pnm"
    fi

    for d in $DISTANCES; do
        n=$((n + 1))
        rkey="${corpus}_${image}_${d}"

        if [[ -v "DONE[$rkey]" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        printf "\r[%d/%d] %s/%s d=%s  " "$n" "$total" "$corpus" "$image" "$d" >&2

        jxl="$CACHE_DIR/${corpus}_${image}_d${d}.jxl"

        # Encode with timing (uses default effort)
        /usr/bin/time -f "%e %U %S" -o "$TIMING_FILE" \
            "$CJXL_RS" "$pnm" "$jxl" -d "$d" --quiet 2>/dev/null

        read -r wall user sys < "$TIMING_FILE"

        size=$(stat -c%s "$jxl")

        # Decode JXL to PNG for fast-ssim2 (can't read JXL directly)
        decoded="$DECODED_DIR/${corpus}_${image}_d${d}.png"
        "$DJXL" "$jxl" "$decoded" 2>/dev/null

        # ssimulacra2 via fast-ssim2
        ssim2=$("$SSIM2" image "$stripped" "$decoded" 2>/dev/null | grep -oP '[\d.]+')

        # butteraugli: first line is max distance
        bfly=$("$BFLY" "$stripped" "$jxl" 2>/dev/null | head -1 | tr -d '[:space:]')

        # Clean up decoded PNG
        rm -f "$decoded"

        echo "${corpus},${image},${width},${height},${d},${size},${ssim2},${bfly},${wall},${user},${sys}" >> "$OUTCSV"
    done
done

echo "" >&2
echo "Done. $((n - skipped)) new rows ($skipped skipped)." >&2
echo "Output: $OUTCSV" >&2
