#!/bin/bash
# Generate cjxl lossless reference CSV.
# Encodes images × 5 efforts (lossless, d=0), measures file size and timing.
# Decodes each JXL and hashes the raw pixel data (SHA-256) for accuracy verification.
# Pre-converts to PNM so timing excludes PNG decode overhead.
# Resumes from partial runs: skips rows already present in the CSV.
#
# Usage: bash scripts/generate_cjxl_lossless_reference.sh
# Output: reference/cjxl_lossless_reference.csv

set -euo pipefail

CJXL="$HOME/work/jxl-efforts/libjxl/build/tools/cjxl"
DJXL="$HOME/work/jxl-efforts/libjxl/build/tools/djxl"

PNM_DIR="/tmp/cjxl-lossless-ref-pnm"
DECODED_DIR="/tmp/cjxl-lossless-ref-decoded"
CACHE_DIR="/mnt/v/output/jxl-encoder-rs/cjxl-lossless-reference"
TIMING_FILE="/tmp/cjxl-lossless-ref-timing.txt"

OUTCSV="reference/cjxl_lossless_reference.csv"

EFFORTS="5 6 7 8 9"

# Corpora
declare -A CORPUS_DIRS
CORPUS_DIRS[cid22]="$HOME/work/codec-corpus/CID22/CID22-512/validation"
CORPUS_DIRS[cid22-train]="$HOME/work/codec-corpus/CID22/CID22-512/training"
CORPUS_DIRS[gb82-sc]="$HOME/work/codec-corpus/gb82-sc"
CORPUS_DIRS[frymire]="$HOME/work/jxl-encoder-rs/jxl_encoder/tests/images"

CORPORA="cid22 cid22-train gb82-sc frymire"

# cid22-train: only first 10 images
CID22_TRAIN_LIMIT=10

# Verify tools exist
for tool in "$CJXL" "$DJXL"; do
    if [ ! -x "$tool" ]; then
        echo "ERROR: tool not found: $tool" >&2
        exit 1
    fi
done

mkdir -p "$PNM_DIR" "$DECODED_DIR" "$CACHE_DIR" "$(dirname "$OUTCSV")"

# Get cjxl version
CJXL_VERSION=$("$CJXL" --version 2>&1 | head -1 | awk '{print $2}')

# Hash raw pixel data from a PNM file (skip header, hash only pixel bytes)
hash_pnm_pixels() {
    local pnm="$1"
    # P6 header: "P6\n<w> <h>\n<maxval>\n" — find byte offset after 3rd newline
    local offset
    offset=$(python3 -c "
data = open('$pnm', 'rb').read(200)
nl = 0
for i, b in enumerate(data):
    if b == 10:
        nl += 1
        if nl == 3:
            print(i + 1)
            break
")
    tail -c +"$((offset + 1))" "$pnm" | sha256sum | cut -d' ' -f1
}

# Build image list per corpus
declare -A CORPUS_IMAGES
for corpus in $CORPORA; do
    imgs=()
    count=0
    for img in "${CORPUS_DIRS[$corpus]}"/*.png; do
        [ -f "$img" ] || continue
        # frymire: only frymire-srgb.png
        if [ "$corpus" = "frymire" ]; then
            case "$(basename "$img")" in
                frymire-srgb.png) ;;
                *) continue ;;
            esac
        fi
        imgs+=("$img")
        count=$((count + 1))
        # cid22-train: limit to first N
        if [ "$corpus" = "cid22-train" ] && [ "$count" -ge "$CID22_TRAIN_LIMIT" ]; then
            break
        fi
    done
    CORPUS_IMAGES[$corpus]="${imgs[*]}"
done

# Count total work items
total=0
for corpus in $CORPORA; do
    for img in ${CORPUS_IMAGES[$corpus]}; do
        for e in $EFFORTS; do
            total=$((total + 1))
        done
    done
done
echo "Total encodes: $total (cjxl $CJXL_VERSION, lossless)" >&2

# Load already-done keys for resume
declare -A DONE
if [ -f "$OUTCSV" ]; then
    while IFS=, read -r c i w h e rest; do
        [[ "$c" == "#"* || "$c" == "corpus" ]] && continue
        DONE["${c}_${i}_${e}"]=1
    done < "$OUTCSV"
    echo "Resuming: ${#DONE[@]} rows already present" >&2
else
    # Write header for new file
    echo "# cjxl $CJXL_VERSION lossless, generated $(date +%Y-%m-%d)" > "$OUTCSV"
    echo "corpus,image,width,height,effort,size_bytes,pixel_sha256,enc_wall_s,enc_user_s,enc_sys_s" >> "$OUTCSV"
fi

n=0
skipped=0
for corpus in $CORPORA; do
    for img in ${CORPUS_IMAGES[$corpus]}; do
        name=$(basename "$img" .png)
        short="${name:0:8}"

        # Get dimensions
        dims=$(identify -format "%w %h" "$img")
        width=${dims%% *}
        height=${dims##* }

        # Convert to PNM once (excludes PNG decode from timing)
        pnm="$PNM_DIR/${corpus}_${short}.pnm"
        if [ ! -f "$pnm" ]; then
            convert "$img" -strip "$pnm"
        fi

        for e in $EFFORTS; do
            n=$((n + 1))
            key="${corpus}_${short}_${e}"

            # Resume support: skip already-done rows
            if [[ -v "DONE[$key]" ]]; then
                skipped=$((skipped + 1))
                continue
            fi

            printf "\r[%d/%d] %s/%s e%s  " "$n" "$total" "$corpus" "$short" "$e" >&2

            jxl="$CACHE_DIR/${corpus}_${short}_e${e}.jxl"

            # Encode lossless with timing
            /usr/bin/time -f "%e %U %S" -o "$TIMING_FILE" \
                "$CJXL" "$pnm" "$jxl" -d 0 -e "$e" 2>/dev/null

            read -r wall user sys < "$TIMING_FILE"

            size=$(stat -c%s "$jxl")

            # Decode JXL to PNM and hash pixel data
            decoded="$DECODED_DIR/${corpus}_${short}_e${e}.pnm"
            "$DJXL" "$jxl" "$decoded" 2>/dev/null
            pixel_hash=$(hash_pnm_pixels "$decoded")
            rm -f "$decoded"

            echo "${corpus},${short},${width},${height},${e},${size},${pixel_hash},${wall},${user},${sys}" >> "$OUTCSV"
        done
    done
done

echo "" >&2
done_count=0; [[ ${#DONE[@]} -gt 0 ]] 2>/dev/null && done_count=${#DONE[@]}
echo "Done. $((n - skipped)) new rows written ($skipped skipped). Total in CSV: $((n - skipped + done_count))" >&2
echo "Output: $OUTCSV" >&2
