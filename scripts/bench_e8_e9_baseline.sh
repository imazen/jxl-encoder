#!/usr/bin/env bash
# Paired interleaved benchmark for lossless e7/e8/e9 across cjxl and cjxl-rs.
#
# Output: TSV with columns
#   image  binary  effort  threads  iter  time_ms  bytes
#
# Interleave order per (image,effort,threads): cjxl -> rs-1t -> rs-8t
# repeated `SAMPLES` times. Within an outer cycle, we also alternate effort
# to keep thermal/turbo bias paired across compared cells.
#
# Usage:
#   SAMPLES=10 ./bench_e8_e9_baseline.sh > out.tsv

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CJXL="${CJXL:-/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl}"
RS="${RS:-$REPO_ROOT/target/release/cjxl-rs}"
SAMPLES="${SAMPLES:-10}"
OUT_DIR="${OUT_DIR:-/tmp/e8_e9_bench}"
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
  echo "$(iso_now) claude-e8-e9-baseline bench: $1" > "$REPO_ROOT/.workongoing"
}

time_cmd() {
  # Use bash builtin `time` to measure wall-clock in milliseconds (via /proc).
  local out_path="$1"; shift
  local start_ns end_ns
  start_ns=$(date +%s%N)
  "$@" > /dev/null 2>&1
  end_ns=$(date +%s%N)
  local elapsed_ms=$(awk "BEGIN { printf \"%.2f\", ($end_ns - $start_ns) / 1e6 }")
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

    # Pre-warm filesystem / shared libs
    "$RS" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
    "$CJXL" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" -d 0 > /dev/null 2>&1 || true

    for ((s=1; s<=SAMPLES; s++)); do
      # Order: cjxl_32t -> rs_1t -> rs_8t. cjxl always uses --num_threads=-1 (default 32).
      out="$OUT_DIR/${label}_e${effort}_cjxl.jxl"
      r=$(time_cmd "$out" "$CJXL" "$path" "$out" -e "$effort" -d 0)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\tcjxl\t$effort\t32\t$s\t$elapsed\t$bytes"

      out="$OUT_DIR/${label}_e${effort}_rs1.jxl"
      r=$(RAYON_NUM_THREADS=1 time_cmd "$out" "$RS" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\trs-1t\t$effort\t1\t$s\t$elapsed\t$bytes"

      out="$OUT_DIR/${label}_e${effort}_rs8.jxl"
      r=$(RAYON_NUM_THREADS=8 time_cmd "$out" "$RS" "$path" "$out" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\trs-8t\t$effort\t8\t$s\t$elapsed\t$bytes"

      refresh_marker "$label e$effort iter $s/$SAMPLES"
    done
  done
done

refresh_marker "bench complete"
