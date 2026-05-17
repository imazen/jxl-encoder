#!/usr/bin/env bash
# Paired interleaved A/B benchmark for the small-image parallel-tree-learning
# fallback (audit `rejected_optimizations_conditional_value_2026-05-17.md`
# items #9 + #10).
#
# Compares the OPT-IN fallback (`--small-image-fallback`, cache bypass
# for <1 MP at e<=7) against the default behaviour (cache always on).
# Same binary, only the flag flips between iterations so thermal/turbo
# bias is paired.
#
# Output: TSV with columns
#   image  variant  effort  threads  iter  time_ms  bytes
#
# Variants:
#   default      = small-image fallback OFF — default (cache always on)
#   fallback     = `--small-image-fallback` — opt-in, cache bypassed for <1 MP at e<=7
#
# Interleave order per (image,effort,threads) cycle:
#   default -> fallback (then repeat)
#
# Usage:
#   SAMPLES=10 ./bench_small_image_fallback_inline.sh > out.tsv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS="${RS:-$REPO_ROOT/target/release/cjxl-rs}"
SAMPLES="${SAMPLES:-10}"
THREADS="${THREADS:-8}"
OUT_DIR="${OUT_DIR:-/tmp/small_image_fallback_bench}"
mkdir -p "$OUT_DIR"

IMG_S="${IMG_S:-/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png}"
IMG_M="${IMG_M:-/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png}"
IMG_L="${IMG_L:-/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png}"

declare -a IMAGES=(
  "small_0.26MP=$IMG_S"
  "medium_1.05MP=$IMG_M"
  "large_4.19MP=$IMG_L"
)
EFFORTS=(${EFFORTS:-7 8 9})

iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
refresh_marker() {
  echo "$(iso_now) claude-small-fallback bench: $1" > "$REPO_ROOT/.workongoing"
}

time_cmd() {
  local out_path="$1"; shift
  local start_ns end_ns
  start_ns=$(date +%s%N)
  "$@" > /dev/null 2>&1
  end_ns=$(date +%s%N)
  local elapsed_ms
  elapsed_ms=$(awk "BEGIN { printf \"%.2f\", ($end_ns - $start_ns) / 1e6 }")
  local bytes
  bytes=$(stat -c '%s' "$out_path" 2>/dev/null || echo 0)
  echo "$elapsed_ms $bytes"
}

# Header
echo -e "image\tvariant\teffort\tthreads\titer\ttime_ms\tbytes"

for cell in "${IMAGES[@]}"; do
  label="${cell%%=*}"
  path="${cell#*=}"
  for effort in "${EFFORTS[@]}"; do
    refresh_marker "$label e$effort"

    # Pre-warm filesystem / shared libs (both variants)
    RAYON_NUM_THREADS=$THREADS "$RS" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
    RAYON_NUM_THREADS=$THREADS "$RS" "$path" "$OUT_DIR/warmup_fb.jxl" -e "$effort" --lossless --small-image-fallback > /dev/null 2>&1 || true

    for ((s=1; s<=SAMPLES; s++)); do
      # Default (fallback OFF — cache always on)
      out="$OUT_DIR/${label}_e${effort}_default_${s}.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out" "$RS" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tdefault\t$effort\t$THREADS\t$s\t$elapsed\t$bytes"

      # Opt-in fallback (cache bypassed for <1 MP at e<=7)
      out="$OUT_DIR/${label}_e${effort}_fallback_${s}.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out" "$RS" "$path" "$out" -e "$effort" --lossless --small-image-fallback)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tfallback\t$effort\t$THREADS\t$s\t$elapsed\t$bytes"
    done
  done
done

refresh_marker "bench done"
