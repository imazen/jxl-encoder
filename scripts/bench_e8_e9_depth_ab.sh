#!/usr/bin/env bash
# Paired A/B bench: pre vs post chunk-2 (parallel-tree-learning effort tuning).
#
# Output: TSV with columns
#   image  binary  effort  threads  iter  time_ms  bytes
#
# `binary` is one of {cjxl, pre, post}. cjxl serves as a noise anchor — its
# timing should be roughly constant across (pre,post) cycles.
#
# Interleave order per (image, effort): cjxl -> pre -> post repeated SAMPLES
# times to keep thermal/turbo bias paired across compared cells.
#
# Usage:
#   PRE_BIN=/tmp/baseline_target_e9/release/cjxl-rs \
#   POST_BIN=$REPO_ROOT/target/release/cjxl-rs \
#   SAMPLES=10 ./bench_e8_e9_depth_ab.sh > out.tsv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CJXL="${CJXL:-/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl}"
PRE_BIN="${PRE_BIN:?PRE_BIN must be set}"
POST_BIN="${POST_BIN:?POST_BIN must be set}"
SAMPLES="${SAMPLES:-7}"
THREADS="${THREADS:-8}"
OUT_DIR="${OUT_DIR:-/tmp/e9_chunk2_ab}"
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
  echo "$(iso_now) claude-e9-depth-restart bench: $1" > "$REPO_ROOT/.workongoing"
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
echo -e "image\tbinary\teffort\tthreads\titer\ttime_ms\tbytes"

for cell in "${IMAGES[@]}"; do
  label="${cell%%=*}"
  path="${cell#*=}"
  for effort in "${EFFORTS[@]}"; do
    refresh_marker "$label e$effort"

    # Warm shared libs / fs cache once per cell
    "$PRE_BIN"  "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
    "$POST_BIN" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
    "$CJXL"     "$path" "$OUT_DIR/warmup.jxl" -e "$effort" -d 0      > /dev/null 2>&1 || true

    for ((s=1; s<=SAMPLES; s++)); do
      out="$OUT_DIR/${label}_e${effort}_cjxl.jxl"
      r=$(time_cmd "$out" "$CJXL" "$path" "$out" -e "$effort" -d 0)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tcjxl\t$effort\t32\t$s\t$elapsed\t$bytes"

      out="$OUT_DIR/${label}_e${effort}_pre.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out" "$PRE_BIN" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tpre\t$effort\t$THREADS\t$s\t$elapsed\t$bytes"

      out="$OUT_DIR/${label}_e${effort}_post.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out" "$POST_BIN" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tpost\t$effort\t$THREADS\t$s\t$elapsed\t$bytes"

      refresh_marker "$label e$effort iter $s/$SAMPLES"
    done
  done
done

refresh_marker "bench complete"
