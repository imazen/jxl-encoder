#!/usr/bin/env bash
# mem_grid_sweep.sh — peak-RSS grid over zensysbench corpus PPMs for the
# 2026-08-01 memory-model recalibration (size x effort x threads).
#
# Each cell is its own process (RSS high-water is per-process), run under
# run-heavy's cgroup memory cap so a blown cell OOM-kills its own scope
# (rc=137 — recorded as a data point), never the box. /usr/bin/time -v
# supplies OS max RSS; the probe prints VmHWM + the encoder's own
# MemoryBudget peak in the same row.
#
# Usage: scripts/mem_grid_sweep.sh <out.tsv> [mem-cap, default 44G]
# Corpus: ~/work/zen/zensysbench/corpus-cache/photo_{mp12,mp20,mp108}.ppm
# Probe:  target/release/examples/mem_grid_probe (build with
#         cargo build --release -p jxl-encoder --features parallel --example mem_grid_probe)

set -uo pipefail

OUT="${1:?usage: mem_grid_sweep.sh <out.tsv> [mem-cap]}"
MEM="${2:-44G}"
CORPUS="$HOME/work/zen/zensysbench/corpus-cache"
PROBE="target/release/examples/mem_grid_probe"
RUNHEAVY="$HOME/work/zen/scripts/run-heavy"
LOGDIR="$HOME/tmp/memgrid"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$LOGDIR"

[ -x "$PROBE" ] || { echo "probe not built: $PROBE" >&2; exit 2; }

printf 'img\tmode\teffort\tdistance\tthreads\trc\tok\ttime_max_rss_kb\tvmhwm_peak_kb\tdelta_kb\tbudget_peak_kb\twall_ms\tout_bytes\test_typ_kb\test_max_kb\terr\n' > "$OUT"

refresh_marker() {
  printf '%s claude-jxlmem-agent mem grid sweep: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "$REPO_ROOT/.workongoing"
}

cell() {
  local img="$1" mode="$2" effort="$3" dist="$4" threads="$5"
  local tag="${img}_${mode}_e${effort}_d${dist}_t${threads}"
  local log="$LOGDIR/${tag}.log"
  refresh_marker "$tag"
  echo "=== cell $tag ($(date -u +%H:%M:%SZ)) ===" | tee -a "$LOGDIR/driver.log"
  "$RUNHEAVY" --mem "$MEM" --jobs 12 -- /usr/bin/time -v \
    "$PROBE" "$CORPUS/photo_${img}.ppm" "$mode" "$effort" "$dist" "$threads" max \
    > "$log" 2>&1
  local rc=$?
  # Parse the probe line (may be absent on cgroup OOM kill).
  local line
  line=$(grep -o 'ok=[01].*$' "$log" | head -1)
  local probe_row
  probe_row=$(grep '^w=[0-9]' "$log" | head -1)
  get() { echo "$probe_row" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2; }
  local maxrss
  maxrss=$(grep 'Maximum resident set size' "$log" | grep -o '[0-9]*' | tail -1)
  local ok vh dk bp wm ob et em err
  ok=$(get ok); vh=$(get vmhwm_peak_kb); dk=$(get delta_kb); bp=$(get budget_peak_kb)
  wm=$(get wall_ms); ob=$(get out_bytes); et=$(get est_typ_kb); em=$(get est_max_kb)
  err=$(echo "$probe_row" | grep -o 'err=.*$' | head -1 | cut -c5- | tr '\t' ' ')
  [ "$rc" -eq 137 ] && err="cgroup-oom-killed(${MEM})"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$img" "$mode" "$effort" "$dist" "$threads" "$rc" "${ok:-0}" "${maxrss:-0}" \
    "${vh:-0}" "${dk:-0}" "${bp:-0}" "${wm:-0}" "${ob:-0}" "${et:-0}" "${em:-0}" \
    "${err:--}" >> "$OUT"
  echo "  rc=$rc maxrss_kb=${maxrss:-0} budget_peak_kb=${bp:-0} ($line)" | tee -a "$LOGDIR/driver.log"
}

# ── Grid (smallest first; mp108 last) ───────────────────────────────────────
# lossy quality axis check at t=1 (d1.75 ≈ q75, d6.0 ≈ q30)
for e in 5 7; do cell mp12 lossy "$e" 1.75 1; done
# lossy main grid: e{5,7} x d6.0 x t{1,2,4,8,16} x {mp12,mp20}
for img in mp12 mp20; do
  for e in 5 7; do
    for t in 1 2 4 8 16; do cell "$img" lossy "$e" 6.0 "$t"; done
  done
done
# lossless spot cells (thread-invariance + size check): e{5,7} x t{1,8} x mp12
for e in 5 7; do
  for t in 1 8; do cell mp12 lossless "$e" 0 "$t"; done
done
# 108 MP: lossy e{5,7} x d6.0 x t{1,2,4}; plus e7 d1.75 t1 confirm
for e in 5 7; do
  for t in 1 2 4; do cell mp108 lossy "$e" 6.0 "$t"; done
done
cell mp108 lossy 7 1.75 1

echo "grid complete -> $OUT" | tee -a "$LOGDIR/driver.log"
