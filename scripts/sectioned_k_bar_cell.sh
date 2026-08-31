#!/usr/bin/env bash
# sectioned_k_bar_cell.sh — the #99 item-1 WALL BAR cell, measured with
# repeats, for the sectioned content-adaptive predictor selector.
#
# The bar is "sectioned lossless wall <= 1.3x cjxl v0.12 on the same input".
# The reference cell (issue #99 / `benchmarks/jxl_sectioned_prune_k_2026-08-28`)
# is the imazen-26 1403 photo cropped TOP-LEFT to 3840x2160, at e7 and e9,
# threads 1 and 8.
#
# Two wall conventions are reported because they differ materially at t=8:
#   cjxl_ms  = whole process, INCLUDING PNG decode (the convention #99 used)
#   ours_ms  = encode only (the probe harness times `encode()`)
# At t=1 the PNG load is noise; at t=8 it is ~10 % of cjxl's number, so the
# process-wall ratio flatters us. Both are in the output.
#
# Repeats matter: t=8 cells vary run to run with worker scheduling (measured
# up to 15 % on this cell), t=1 cells are stable to ~2 %. MIN over repeats is
# reported alongside the median.
#
# Usage: sectioned_k_bar_cell.sh <out.tsv> <bar_crop.png> [reps]
set -uo pipefail
OUT="${1:?usage: sectioned_k_bar_cell.sh <out.tsv> <bar.png> [reps]}"
IMG="${2:?bar crop png}"
REPS="${3:-5}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="$ROOT/target/release/examples/sectioned_k_corpus"
[ -x "$PROBE" ] || { echo "build first: cargo build --release -p jxl-encoder --example sectioned_k_corpus --features 'std parallel profile-phases'" >&2; exit 2; }

{
  printf '# sectioned_k_bar_cell.sh  commit=%s  host=%s  date=%s  reps=%s\n' \
    "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REPS"
  printf '# img=%s sha256=%s cjxl=%s\n' "$IMG" \
    "$(shasum -a 256 "$IMG" | cut -d' ' -f1)" "$(cjxl --version 2>&1 | head -1)"
  printf 'encoder\teffort\tthreads\tarm\trep\tbytes\twall_ms\n'
} > "$OUT"

for E in 7 9; do
  for T in 1 8; do
    # cjxl: --num_threads=0 is single-threaded in v0.12
    CT=$T; [ "$T" = "1" ] && CT=0
    for r in $(seq 1 "$REPS"); do
      S=$( { /usr/bin/time -p nice -n 19 cjxl -d 0 -e "$E" --num_threads="$CT" \
             "$IMG" "$HOME/tmp/t3gate/_bar.jxl" ; } 2>&1 | awk '/^real/{print $2}' )
      B=$(stat -f%z "$HOME/tmp/t3gate/_bar.jxl" 2>/dev/null || echo 0)
      printf 'cjxl\t%s\t%s\tprocess\t%s\t%s\t%s\n' "$E" "$T" "$r" "$B" \
        "$(awk -v s="$S" 'BEGIN{printf "%.1f", s*1000}')" >> "$OUT"
    done
    # INTERLEAVED: both arms inside one process per rep, alternating, so a
    # thermal or scheduler drift over the run biases both arms equally. Five
    # reps of arm A followed by five of arm B does NOT do that — measured on
    # this cell at t=8, where the block-ordered form put the two arms on
    # opposite sides of a drift and inverted the sign of the difference.
    for r in $(seq 1 "$REPS"); do
      nice -n 19 "$PROBE" phases "$IMG" "$E" "$T" k8 default 2>/dev/null \
        | awk -v e="$E" -v t="$T" -v r="$r" \
            '/^== /{gsub(",","",$3); a=$2; sub(":","",a); printf "ours\t%s\t%s\t%s\t%s\t%s\t%s\n", e,t,a,r,$3,$5}' >> "$OUT"
    done
  done
done
echo "wrote $OUT"
