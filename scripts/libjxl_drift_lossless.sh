#!/usr/bin/env bash
#
# Lossless drift bench (W19-2): 5 CLIC photos × {cjxl_old e7, cjxl_new e7, ours e7}.
# Bytes-only (lossless = pixel-exact, no metric needed).

set -euo pipefail

: "${CJXL_OLD:=/tmp/cjxl_old_d2c7032}"
: "${CJXL_NEW:=/tmp/cjxl_new_4279d48}"
: "${CJXL_RS:=/home/lilith/work/zen/jxl-encoder/target/release/cjxl-rs}"
: "${OUT_TSV:?must set OUT_TSV}"

SCRATCH=/tmp/drift_bench_lossless
mkdir -p "$SCRATCH"

for tool in "$CJXL_OLD" "$CJXL_NEW" "$CJXL_RS"; do
    if [[ ! -x "$tool" ]]; then
        echo "ERROR: missing or non-exec tool: $tool" >&2
        exit 1
    fi
done

IMAGES=(
    /home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png
    /home/lilith/work/codec-corpus/clic2025-1024/0369d229ba4c9965d5caeb38c359a027a810968eee930b81520b604e76b4df14.png
    /home/lilith/work/codec-corpus/clic2025-1024/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png
    /home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png
    /home/lilith/work/codec-corpus/clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png
)
EFFORT=7

printf "image\tencoder\tbytes\n" > "$OUT_TSV"

for src in "${IMAGES[@]}"; do
    stem=$(basename "$src" .png)
    short=${stem:0:8}
    echo "=== $short ===" >&2

    # cjxl-rs --lossless
    out="$SCRATCH/${short}_ours.jxl"
    "$CJXL_RS" --lossless -e "$EFFORT" "$src" "$out" >/dev/null 2>&1
    bytes=$(stat -c %s "$out")
    printf "%s\tours\t%s\n" "$short" "$bytes" >> "$OUT_TSV"
    echo "  ours:     $bytes" >&2

    # cjxl_old -d 0 (lossless)
    out="$SCRATCH/${short}_old.jxl"
    "$CJXL_OLD" -d 0 -e "$EFFORT" --quiet "$src" "$out" >/dev/null 2>&1
    bytes=$(stat -c %s "$out")
    printf "%s\tcjxl_old\t%s\n" "$short" "$bytes" >> "$OUT_TSV"
    echo "  cjxl_old: $bytes" >&2

    # cjxl_new -d 0 (lossless)
    out="$SCRATCH/${short}_new.jxl"
    "$CJXL_NEW" -d 0 -e "$EFFORT" --quiet "$src" "$out" >/dev/null 2>&1
    bytes=$(stat -c %s "$out")
    printf "%s\tcjxl_new\t%s\n" "$short" "$bytes" >> "$OUT_TSV"
    echo "  cjxl_new: $bytes" >&2
done

echo "wrote: $OUT_TSV" >&2
