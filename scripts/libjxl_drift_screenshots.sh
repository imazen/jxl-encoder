#!/usr/bin/env bash
#
# Screenshot drift bench (W19-2): the libjxl HEAD diff contains
# - e39a6aa "Disable global palette for progressive lossless"
# - b3510d1 "Revise MA tree check to align with new buffering logic"
# - 032d39a "Default to buffering level 2"
# - acc28c0 "Streaming encode without streaming output"
# - 1389871 "Loosen buffering check"
# These could plausibly impact lossless on screenshots (palette+tree-heavy).
# Run cjxl_old vs cjxl_new on the gb82-sc corpus at e7 to detect drift.

set -euo pipefail

: "${CJXL_OLD:=/tmp/cjxl_old_d2c7032}"
: "${CJXL_NEW:=/tmp/cjxl_new_4279d48}"
: "${OUT_TSV:?must set OUT_TSV}"

SCRATCH=/tmp/drift_bench_screenshots
mkdir -p "$SCRATCH"

IMAGES=(
    /home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png
    /home/lilith/work/codec-corpus/gb82-sc/imac_dark.png
    /home/lilith/work/codec-corpus/gb82-sc/imac_g3.png
    /home/lilith/work/codec-corpus/gb82-sc/terminal.png
    /home/lilith/work/codec-corpus/gb82-sc/windows95.png
)

printf "image\tencoder\tbytes\n" > "$OUT_TSV"

for src in "${IMAGES[@]}"; do
    stem=$(basename "$src" .png)
    echo "=== $stem ===" >&2

    out="$SCRATCH/${stem}_old.jxl"
    "$CJXL_OLD" -d 0 -e 7 --quiet "$src" "$out" >/dev/null 2>&1
    bytes_old=$(stat -c %s "$out")
    printf "%s\tcjxl_old\t%s\n" "$stem" "$bytes_old" >> "$OUT_TSV"

    out="$SCRATCH/${stem}_new.jxl"
    "$CJXL_NEW" -d 0 -e 7 --quiet "$src" "$out" >/dev/null 2>&1
    bytes_new=$(stat -c %s "$out")
    printf "%s\tcjxl_new\t%s\n" "$stem" "$bytes_new" >> "$OUT_TSV"

    delta_pct=$(awk -v o="$bytes_old" -v n="$bytes_new" 'BEGIN { printf "%+.2f", (n-o)/o*100 }')
    echo "  old=$bytes_old new=$bytes_new delta=${delta_pct}%" >&2
done

echo "wrote: $OUT_TSV" >&2
