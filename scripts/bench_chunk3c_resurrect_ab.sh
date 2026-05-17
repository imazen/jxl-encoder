#!/usr/bin/env bash
# Paired interleaved A/B benchmark for issue #40 chunk-3c resurrection.
#
# Compares BASELINE (props-swapping path, JXL_DISABLE_CHUNK3C=1) vs NEW
# (chunk-3c props-swap skipped on lossless main path, env unset). Same
# binary, same compile flags, env var flipped per invocation. Order per
# (image,effort,thread) cell: BASELINE -> NEW -> BASELINE -> NEW ...
#
# Output: CSV with columns
#   image,effort,threads,variant,sample,bytes,ms
#
# Usage:
#   SAMPLES=10 ./bench_chunk3c_resurrect_ab.sh > out.csv
#
# Requires `target/release/cjxl-rs` already built.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RS="${RS:-$REPO_ROOT/target/release/cjxl-rs}"
SAMPLES="${SAMPLES:-10}"
OUT_DIR="${OUT_DIR:-/tmp/chunk3c_resurrect_bench}"
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
THREADS=(8)

iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
refresh_marker() {
  echo "$(iso_now) claude-chunk3c-resurrect bench: $1" > "$REPO_ROOT/.workongoing"
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
echo "image,effort,threads,variant,sample,bytes,ms"

for cell in "${IMAGES[@]}"; do
  label="${cell%%=*}"
  path="${cell#*=}"
  for effort in "${EFFORTS[@]}"; do
    for nt in "${THREADS[@]}"; do
      refresh_marker "$label e$effort ${nt}T"

      # Pre-warm filesystem / shared libs (both variants once)
      JXL_DISABLE_CHUNK3C=1 RAYON_NUM_THREADS="$nt" "$RS" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
      RAYON_NUM_THREADS="$nt" "$RS" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true

      for ((s=1; s<=SAMPLES; s++)); do
        # BASELINE (props swap retained)
        out="$OUT_DIR/${label}_e${effort}_${nt}t_baseline.jxl"
        r=$(JXL_DISABLE_CHUNK3C=1 RAYON_NUM_THREADS="$nt" time_cmd "$out" "$RS" "$path" "$out" -e "$effort" --lossless)
        elapsed="${r% *}"; bytes="${r#* }"
        echo "$label,$effort,$nt,BASELINE,$s,$bytes,$elapsed"

        # NEW (chunk-3c skip enabled)
        out="$OUT_DIR/${label}_e${effort}_${nt}t_new.jxl"
        r=$(RAYON_NUM_THREADS="$nt" time_cmd "$out" "$RS" "$path" "$out" -e "$effort" --lossless)
        elapsed="${r% *}"; bytes="${r#* }"
        echo "$label,$effort,$nt,NEW,$s,$bytes,$elapsed"

        refresh_marker "$label e$effort ${nt}T iter $s/$SAMPLES"
      done
    done
  done
done

refresh_marker "bench complete"
