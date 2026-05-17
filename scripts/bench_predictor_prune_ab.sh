#!/usr/bin/env bash
# Paired interleaved A/B bench for `predictor_prune` integration into
# `find_best_predictor` (issue #23 chunk 2).
#
# A = baseline main (c579cbd: chunk-1 primitive shipped, NOT wired)
# B = this branch  (chunk-2: lb-skip wired into both sequential cfg paths)
#
# Both binaries built with --features parallel-tree-learning so the only
# difference is the find_best_predictor body (sequential branches only;
# parallel branch deferred to chunk 3 per the chunk-1 commit plan).
#
# Output: TSV with columns
#   image  variant  effort  threads  iter  time_ms  bytes
#
# Within each (image, effort, threads) cell we alternate A/B per iter and
# repeat SAMPLES times. Both encoders read the same input on the same machine
# in adjacent iterations so thermal/turbo bias is paired out.
#
# Usage:
#   SAMPLES=8 THREADS=8 ./bench_predictor_prune_ab.sh > out.tsv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS_BASE="${RS_BASE:-/home/lilith/work/zen/jxl-encoder--baseline-main/target/release/cjxl-rs}"
RS_NEW="${RS_NEW:-$REPO_ROOT/target/release/cjxl-rs}"
SAMPLES="${SAMPLES:-8}"
THREADS="${THREADS:-8}"
OUT_DIR="${OUT_DIR:-/tmp/predictor_prune_ab}"
mkdir -p "$OUT_DIR"

IMG_S="/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png"
IMG_M="/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
IMG_L="/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png"

declare -a IMAGES=(
  "small_0.26MP=$IMG_S"
  "medium_1.05MP=$IMG_M"
  "large_4.19MP=$IMG_L"
)
EFFORTS=(7 8 9)

iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
refresh_marker() {
  echo "$(iso_now) claude-issue23-chunk2-resume bench: $1" > "$REPO_ROOT/.workongoing"
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

    # Pre-warm both binaries (fs cache, cold-start cost, etc.)
    RAYON_NUM_THREADS=$THREADS "$RS_BASE" "$path" "$OUT_DIR/warmup_base.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
    RAYON_NUM_THREADS=$THREADS "$RS_NEW"  "$path" "$OUT_DIR/warmup_new.jxl"  -e "$effort" --lossless > /dev/null 2>&1 || true

    for ((s=1; s<=SAMPLES; s++)); do
      # Alternate A then B per iter so consecutive samples interleave cleanly.
      out="$OUT_DIR/${label}_e${effort}_base_${s}.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out" "$RS_BASE" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tA_base\t$effort\t$THREADS\t$s\t$elapsed\t$bytes"

      out="$OUT_DIR/${label}_e${effort}_new_${s}.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out" "$RS_NEW" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tB_new\t$effort\t$THREADS\t$s\t$elapsed\t$bytes"

      refresh_marker "$label e$effort iter $s/$SAMPLES"
    done
  done
done

refresh_marker "bench complete"
