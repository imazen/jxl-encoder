#!/usr/bin/env bash
#
# libjxl drift bench (W19-2): compare OLD vs NEW cjxl bytes + butteraugli on
# 5 CLIC photos × 4 distances, plus our cjxl-rs as the third arm. Reports
# rows to a TSV under benchmarks/.
#
# Inputs (env overridable):
#   CJXL_OLD  — path to old cjxl (default /tmp/cjxl_old_d2c7032)
#   CJXL_NEW  — path to new cjxl (default /tmp/cjxl_new_4279d48)
#   CJXL_RS   — path to our encoder (default repo target/release/cjxl-rs)
#   DJXL      — path to djxl     (default libjxl build)
#   BFLY      — path to rust butteraugli (default ~/.cargo/bin/butteraugli)
#   OUT_TSV   — output TSV path  (required)
#
# Each image is encoded at d=0.5, 1.0, 2.0, 5.0 at default effort 7. The decoded
# PNG is compared against the source PNG via Rust butteraugli (metadata-immune,
# CLAUDE.md compliant).

set -euo pipefail

: "${CJXL_OLD:=/tmp/cjxl_old_d2c7032}"
: "${CJXL_NEW:=/tmp/cjxl_new_4279d48}"
: "${CJXL_RS:=/home/lilith/work/zen/jxl-encoder/target/release/cjxl-rs}"
: "${DJXL:=/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl}"
: "${BFLY:=/home/lilith/.cargo/bin/butteraugli}"
: "${OUT_TSV:?must set OUT_TSV}"

SCRATCH=/tmp/drift_bench
mkdir -p "$SCRATCH"

for tool in "$CJXL_OLD" "$CJXL_NEW" "$CJXL_RS" "$DJXL" "$BFLY"; do
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
DISTANCES=(0.5 1.0 2.0 5.0)
EFFORT=7

# TSV header
printf "image\tdistance\tencoder\tbytes\tbutteraugli\n" > "$OUT_TSV"

for src in "${IMAGES[@]}"; do
    stem=$(basename "$src" .png)
    short=${stem:0:8}
    echo "=== $short ===" >&2
    for d in "${DISTANCES[@]}"; do
        echo "  d=$d" >&2

        # ours: cjxl-rs
        ours_jxl="$SCRATCH/${short}_d${d}_ours.jxl"
        ours_png="$SCRATCH/${short}_d${d}_ours.png"
        "$CJXL_RS" -d "$d" -e "$EFFORT" "$src" "$ours_jxl" >/dev/null 2>&1
        "$DJXL" --quiet "$ours_jxl" "$ours_png" >/dev/null 2>&1
        ours_bytes=$(stat -c %s "$ours_jxl")
        ours_bfly=$("$BFLY" --format score "$src" "$ours_png" 2>/dev/null)
        printf "%s\t%s\tours\t%s\t%s\n" "$short" "$d" "$ours_bytes" "$ours_bfly" >> "$OUT_TSV"

        # cjxl_old
        old_jxl="$SCRATCH/${short}_d${d}_old.jxl"
        old_png="$SCRATCH/${short}_d${d}_old.png"
        "$CJXL_OLD" -d "$d" -e "$EFFORT" --quiet "$src" "$old_jxl" >/dev/null 2>&1
        "$DJXL" --quiet "$old_jxl" "$old_png" >/dev/null 2>&1
        old_bytes=$(stat -c %s "$old_jxl")
        old_bfly=$("$BFLY" --format score "$src" "$old_png" 2>/dev/null)
        printf "%s\t%s\tcjxl_old\t%s\t%s\n" "$short" "$d" "$old_bytes" "$old_bfly" >> "$OUT_TSV"

        # cjxl_new
        new_jxl="$SCRATCH/${short}_d${d}_new.jxl"
        new_png="$SCRATCH/${short}_d${d}_new.png"
        "$CJXL_NEW" -d "$d" -e "$EFFORT" --quiet "$src" "$new_jxl" >/dev/null 2>&1
        "$DJXL" --quiet "$new_jxl" "$new_png" >/dev/null 2>&1
        new_bytes=$(stat -c %s "$new_jxl")
        new_bfly=$("$BFLY" --format score "$src" "$new_png" 2>/dev/null)
        printf "%s\t%s\tcjxl_new\t%s\t%s\n" "$short" "$d" "$new_bytes" "$new_bfly" >> "$OUT_TSV"
    done
done

echo "wrote: $OUT_TSV" >&2
