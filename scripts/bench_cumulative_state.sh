#!/usr/bin/env bash
# Paired A/B bench across a broader 20-image corpus.
#
# Cells:
#   - 5 small  (~0.26 MP photos, CID22-512)
#   - 5 medium (~1.05 MP photos, clic2025-1024)
#   - 5 large  (~2.79-4.19 MP photos, clic2025/final-test)
#   - 5 screenshots / edge cases (gb82-sc)
#
# For each image x effort in {7,8,9}, capture:
#   - cjxl reference (effort 7/8/9, lossless, 8T)
#   - cjxl-rs --no-smart-fanout (current default, 8T)
#   - cjxl-rs --smart-fanout    (opt-in candidate,  8T)
#
# Output: TSV with columns
#   image effort threads encoder variant iter time_ms bytes
#
# Usage:
#   SAMPLES=5 ./bench_cumulative_state.sh > out.tsv
#
# Variants:
#   variant=cjxl                — libjxl cjxl baseline
#   variant=rs_smart_off        — cjxl-rs current default
#   variant=rs_smart_on         — cjxl-rs with --smart-fanout

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$REPO_ROOT/target/release/cjxl-rs}"
CJXL="${CJXL:-/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl}"
SAMPLES="${SAMPLES:-5}"
THREADS="${THREADS:-8}"
OUT_DIR="${OUT_DIR:-/tmp/cumulative_bench}"
mkdir -p "$OUT_DIR"

# 20 image corpus: 5 small + 5 medium + 5 large + 5 screenshots
# Names use bucket_size_label format (no spaces) for easy grouping.
declare -a IMAGES=(
  # small (CID22-512, ~0.26 MP photos)
  "small_S1_258947=/home/lilith/work/codec-corpus/CID22/CID22-512/training/258947.png"
  "small_S2_3705529=/home/lilith/work/codec-corpus/CID22/CID22-512/training/3705529.png"
  "small_S3_580612=/home/lilith/work/codec-corpus/CID22/CID22-512/training/580612.png"
  "small_S4_208560=/home/lilith/work/codec-corpus/CID22/CID22-512/training/208560.png"
  "small_S5_459728=/home/lilith/work/codec-corpus/CID22/CID22-512/training/459728.png"

  # medium (clic2025-1024, 1.05 MP photos)
  "medium_M1_8426ed=/home/lilith/work/codec-corpus/clic2025-1024/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png"
  "medium_M2_02809272=/home/lilith/work/codec-corpus/clic2025-1024/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
  "medium_M3_5dbdb989=/home/lilith/work/codec-corpus/clic2025-1024/5dbdb989cf026f7ab3dd7167d7fc0fe2.png"
  "medium_M4_e0d8e29c=/home/lilith/work/codec-corpus/clic2025-1024/e0d8e29cadfc99663c7d1a4a5afe20c454ec54d0d873776ec397c59405c74790.png"
  "medium_M5_097cb426=/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png"

  # large (clic2025/final-test, 2.79-4.19 MP photos)
  "large_L1_8426ed=/home/lilith/work/codec-corpus/clic2025/final-test/8426ed2245c791232862b0a0b2a62a1f17031e8e6e38921fe939df0b3a05ac41.png"
  "large_L2_02809272=/home/lilith/work/codec-corpus/clic2025/final-test/02809272b4ca9b08af45771501b741296187c7e26907efb44abbbfcb6cd804f7.png"
  "large_L3_2684452d=/home/lilith/work/codec-corpus/clic2025/final-test/2684452db505ddbbb53f42a3f3bcfe86fdd0d6d8d98c029db4b4c6fc1f55b750.png"
  "large_L4_07b9f93f=/home/lilith/work/codec-corpus/clic2025/final-test/07b9f93f170a0381836bdf301280a5b80b2c4be6e66f793a3c335dc200fb4e5b.png"
  "large_L5_1cba10ad=/home/lilith/work/codec-corpus/clic2025/final-test/1cba10ad9bb4ced57e42f7656c5f2a58d32dc6bad084957d2f8d1c78e0fcd224.png"

  # screenshots / edge cases (gb82-sc, mix of sizes)
  "scrn_T1_windows95=/home/lilith/work/codec-corpus/gb82-sc/windows95.png"
  "scrn_T2_terminal=/home/lilith/work/codec-corpus/gb82-sc/terminal.png"
  "scrn_T3_imac_dark=/home/lilith/work/codec-corpus/gb82-sc/imac_dark.png"
  "scrn_T4_codec_wiki=/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png"
  "scrn_T5_imessage=/home/lilith/work/codec-corpus/gb82-sc/imessage.png"
)

EFFORTS=(7 8 9)

iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
refresh_marker() {
  echo "$(iso_now) claude-cumulative-bench bench: $1" > "$REPO_ROOT/.workongoing"
}

# Wait for system load to drop below threshold.
wait_for_load() {
  local max="${1:-2.0}"
  local sleep_s=15
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    local load1
    load1=$(awk '{print $1}' /proc/loadavg)
    if awk "BEGIN { exit !($load1 < $max) }"; then
      return 0
    fi
    echo "# load $load1 >= $max, sleeping ${sleep_s}s" >&2
    sleep $sleep_s
  done
  echo "# WARN: load never dropped below $max, proceeding anyway" >&2
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

echo -e "image\teffort\tthreads\tencoder\tvariant\titer\ttime_ms\tbytes\tload1"

for cell in "${IMAGES[@]}"; do
  label="${cell%%=*}"
  path="${cell#*=}"
  if [ ! -f "$path" ]; then
    echo "# WARN: missing $path" >&2
    continue
  fi
  for effort in "${EFFORTS[@]}"; do
    refresh_marker "$label e$effort"
    wait_for_load "${LOAD_GATE:-5.0}"
    # Warm shared libs / fs cache for each variant.
    out_warm_cjxl="$OUT_DIR/${label}_e${effort}_cjxl_warm.jxl"
    out_warm_off="$OUT_DIR/${label}_e${effort}_off_warm.jxl"
    out_warm_on="$OUT_DIR/${label}_e${effort}_on_warm.jxl"
    "$CJXL" "$path" "$out_warm_cjxl" -e "$effort" -d 0 --num_threads "$THREADS" > /dev/null 2>&1 || true
    RAYON_NUM_THREADS=$THREADS "$BIN" "$path" "$out_warm_off" -e "$effort" --lossless > /dev/null 2>&1 || true
    RAYON_NUM_THREADS=$THREADS "$BIN" "$path" "$out_warm_on"  -e "$effort" --lossless --smart-fanout > /dev/null 2>&1 || true

    for ((s=1; s<=SAMPLES; s++)); do
      load1=$(awk '{print $1}' /proc/loadavg)

      # cjxl reference
      out_cjxl="$OUT_DIR/${label}_e${effort}_cjxl.jxl"
      r=$( time_cmd "$out_cjxl" "$CJXL" "$path" "$out_cjxl" -e "$effort" -d 0 --num_threads "$THREADS" )
      ems="${r% *}"; b="${r#* }"
      echo -e "$label\t$effort\t$THREADS\tcjxl\tref\t$s\t$ems\t$b\t$load1"

      # cjxl-rs smart-fanout OFF
      out_off="$OUT_DIR/${label}_e${effort}_off.jxl"
      r=$( RAYON_NUM_THREADS=$THREADS time_cmd "$out_off" "$BIN" "$path" "$out_off" -e "$effort" --lossless )
      ems="${r% *}"; b="${r#* }"
      echo -e "$label\t$effort\t$THREADS\trs\tsmart_off\t$s\t$ems\t$b\t$load1"

      # cjxl-rs smart-fanout ON
      out_on="$OUT_DIR/${label}_e${effort}_on.jxl"
      r=$( RAYON_NUM_THREADS=$THREADS time_cmd "$out_on" "$BIN" "$path" "$out_on" -e "$effort" --lossless --smart-fanout )
      ems="${r% *}"; b="${r#* }"
      echo -e "$label\t$effort\t$THREADS\trs\tsmart_on\t$s\t$ems\t$b\t$load1"

      refresh_marker "$label e$effort iter $s/$SAMPLES"
    done

    # Verify byte-identical between smart-off / smart-on (bitstream-equivalent claim).
    sha_off=$(sha256sum "$out_off" 2>/dev/null | cut -d' ' -f1)
    sha_on=$(sha256sum "$out_on" 2>/dev/null | cut -d' ' -f1)
    if [ "$sha_off" != "$sha_on" ]; then
      echo "# WARN: $label e$effort smart-off vs smart-on differ off=$sha_off on=$sha_on" >&2
    else
      echo "# OK: $label e$effort smart-off==smart-on sha=$sha_off" >&2
    fi
  done
done
