#!/usr/bin/env bash
# Paired A/B: --smart-fanout off vs on (same binary).
#
# Output: TSV with columns
#   image effort threads variant iter time_ms bytes
#
# Usage:
#   SAMPLES=7 ./bench_smart_fanout_inline.sh > out.tsv

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$REPO_ROOT/target/release/cjxl-rs}"
SAMPLES="${SAMPLES:-7}"
THREADS="${THREADS:-8}"
OUT_DIR="${OUT_DIR:-/tmp/smart_fanout_ab}"
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
  echo "$(iso_now) claude-smart-fanout-ab bench: $1" > "$REPO_ROOT/.workongoing"
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

echo -e "image\teffort\tthreads\tvariant\titer\ttime_ms\tbytes"
for cell in "${IMAGES[@]}"; do
  label="${cell%%=*}"
  path="${cell#*=}"
  for effort in "${EFFORTS[@]}"; do
    refresh_marker "$label e$effort"
    # Warm shared libs / fs cache
    "$BIN" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless > /dev/null 2>&1 || true
    "$BIN" "$path" "$OUT_DIR/warmup.jxl" -e "$effort" --lossless --smart-fanout > /dev/null 2>&1 || true

    for ((s=1; s<=SAMPLES; s++)); do
      # Interleave: pre then post each iter to pair thermal/turbo
      out_pre="$OUT_DIR/${label}_e${effort}_pre.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out_pre" "$BIN" "$path" "$out_pre" -e "$effort" --lossless)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\t$effort\t$THREADS\tpre\t$s\t$elapsed\t$bytes"

      out_post="$OUT_DIR/${label}_e${effort}_post.jxl"
      r=$(RAYON_NUM_THREADS=$THREADS time_cmd "$out_post" "$BIN" "$path" "$out_post" -e "$effort" --lossless --smart-fanout)
      elapsed="${r% *}"; bytes="${r#* }"
      echo -e "$label\t$effort\t$THREADS\tpost\t$s\t$elapsed\t$bytes"

      refresh_marker "$label e$effort iter $s/$SAMPLES"
    done

    # Verify byte-identical
    sha_pre=$(sha256sum "$out_pre" 2>/dev/null | cut -d' ' -f1)
    sha_post=$(sha256sum "$out_post" 2>/dev/null | cut -d' ' -f1)
    if [ "$sha_pre" != "$sha_post" ]; then
      echo "# WARN: $label e$effort bytes differ pre=$sha_pre post=$sha_post" >&2
    else
      echo "# OK: $label e$effort byte-identical sha=$sha_pre" >&2
    fi
  done
done
