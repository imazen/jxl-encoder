#!/usr/bin/env bash
# mem_sectioned_sweep.sh — allocator-agnostic peak_live grid for the
# lossless SECTIONED local-tree mode (imazen/jxl-encoder#96), the input to
# the estimator's sectioned arm (`heuristics::estimate_encode_sectioned`).
#
# Axes (per the source-informing sweep discipline): size (tiny -> full,
# real-content CROPS of the same source, never resampled) x effort {7, 9}
# (the tree-learning band; sectioned only engages at e >= 7) x threads
# {1, 4, 8, 12} (each in-flight worker learns one group's tree, so the
# per-thread term is the axis the whole-image model never had) x content
# {photo, screen}. Global-tree cells at t=1 on the larger sizes ride along
# for context (they are the band the existing 4K/12 MP pins cover).
#
# Each cell is its own process (peak_live is a per-process high-water mark
# reset right before the encode). Probe:
#   nice -n 19 cargo build -j 4 -p jxl-encoder-cli --release --example mem_probe
#
# Usage: scripts/mem_sectioned_sweep.sh <out.tsv> <photo.png> <screen.png> [screen2.png]
#   photo.png  >= 4000x3000 (imazen-26 png-v3 camera photo)
#   screen.png >= 1313x8008 (qoi-benchmark screenshot_web/reddit.com.png)
# Output columns: content src crop w h mode effort threads tree rc bytes
#   wall_ms live_pre_kb peak_live_kb marginal_live_kb allocs est_typ_kb est_max_kb

set -uo pipefail

OUT="${1:?usage: mem_sectioned_sweep.sh <out.tsv> <photo.png> <screen.png> [screen2.png]}"
PHOTO="${2:?photo png}"
SCREEN="${3:?screen png}"
SCREEN2="${4:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="$REPO_ROOT/target/release/examples/mem_probe"
LOGDIR="$HOME/tmp/memsectioned"
mkdir -p "$LOGDIR"

[ -x "$PROBE" ] || { echo "probe not built: $PROBE" >&2; exit 2; }

# Resumable: an existing OUT is appended to and cells already recorded in
# it (same content/crop/effort/threads/tree, rc=0) are skipped, so a killed
# run continues where it stopped.
if [ ! -s "$OUT" ]; then
  {
    printf '# mem_sectioned_sweep.sh  commit=%s  host=%s  date=%s\n' \
      "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
      "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'content\tsrc\tcrop\tw\th\tmode\teffort\tthreads\ttree\trc\tbytes\twall_ms\tlive_pre_kb\tpeak_live_kb\tmarginal_live_kb\tallocs\test_typ_kb\test_max_kb\n'
  } > "$OUT"
fi

refresh_marker() {
  printf '%s claude-fixer-jxl-encoder issue96 sectioned mem sweep: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "$REPO_ROOT/.workongoing"
}

cell() {
  local content="$1" src="$2" crop="$3" effort="$4" threads="$5" tree="$6"
  local tag="${content}_${crop}_e${effort}_t${threads}_${tree}"
  local log="$LOGDIR/${tag}.log"
  if awk -F'\t' -v c="$content" -v cr="$crop" -v e="$effort" -v t="$threads" -v tr="$tree" \
       '$1==c && $3==cr && $7==e && $8==t && $9==tr && $10=="0" {found=1} END {exit !found}' "$OUT"; then
    echo "--- skip $tag (recorded)" | tee -a "$LOGDIR/driver.log"
    return 0
  fi
  refresh_marker "$tag"
  echo "=== cell $tag ($(date -u +%H:%M:%SZ)) ===" | tee -a "$LOGDIR/driver.log"
  local envcrop=()
  [ "$crop" != "full" ] && envcrop=(env "MEM_PROBE_CROP=$crop")
  "${envcrop[@]}" nice -n 19 "$PROBE" "$src" lossless "$effort" 0 8 rgb "$threads" "$tree" \
    > "$log" 2>&1
  local rc=$?
  local row
  row=$(grep '^delta_kb=' "$log" | head -1)
  get() { echo "$row" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2; }
  local w h
  if [ "$crop" = "full" ]; then
    read -r w h < <(python3 -c "import struct,sys;d=open(sys.argv[1],'rb').read(24);print(*struct.unpack('>II',d[16:24]))" "$src")
  else
    w="${crop%x*}"; h="${crop#*x}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\tlossless\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$content" "$(basename "$src")" "$crop" "$w" "$h" "$effort" "$threads" "$tree" "$rc" \
    "$(get bytes)" "$(get wall_ms)" "$(get live_pre_kb)" "$(get peak_live_kb)" \
    "$(get marginal_live_kb)" "$(get allocs)" "$(get est_typ_kb)" "$(get est_max_kb)" >> "$OUT"
  echo "  rc=$rc peak_live_kb=$(get peak_live_kb) marginal_kb=$(get marginal_live_kb) wall_ms=$(get wall_ms) bytes=$(get bytes)" \
    | tee -a "$LOGDIR/driver.log"
}

# ── Sectioned grid (smallest first) ─────────────────────────────────────────
for crop in 64x64 256x256 1024x1024 2048x2048 3840x2160 full; do
  for e in 7 9; do
    for t in 1 4 8 12; do cell photo "$PHOTO" "$crop" "$e" "$t" sectioned; done
  done
done
for crop in 64x64 256x256 1024x1024 1313x2048 1313x4096 full; do
  for e in 7 9; do
    for t in 1 4 8 12; do cell screen "$SCREEN" "$crop" "$e" "$t" sectioned; done
  done
done
if [ -n "$SCREEN2" ]; then
  for e in 7 9; do
    for t in 1 4 8 12; do cell screen2 "$SCREEN2" full "$e" "$t" sectioned; done
  done
fi

# ── Global-tree context cells at t=1 (the band the existing pins cover) ────
for crop in 1024x1024 2048x2048 3840x2160 full; do
  for e in 7 9; do cell photo "$PHOTO" "$crop" "$e" 1 global; done
done
for crop in 1024x1024 1313x4096 full; do
  for e in 7 9; do cell screen "$SCREEN" "$crop" "$e" 1 global; done
done
# Global at t=8 on the two full images: is the whole-image band really
# thread-invariant with parallel-tree-learning on?
for e in 7 9; do
  cell photo "$PHOTO" full "$e" 8 global
  cell screen "$SCREEN" full "$e" 8 global
done

# ── Stage 2: alpha term + the t=1 excess boundary ────────────────────────
# rgba: the probe's alpha plane is the source GREEN channel (worst-case
# entropy) — the sectioned alpha per-pixel term. t=2 at 12 MP: whether the
# threads=1 pre-tree excess is a 1-worker-only artifact.
cell_layout() {
  local content="$1" src="$2" crop="$3" effort="$4" threads="$5" tree="$6" layout="$7"
  local tag="${content}_${crop}_e${effort}_t${threads}_${tree}_${layout}"
  local log="$LOGDIR/${tag}.log"
  if awk -F'\t' -v c="${content}-${layout}" -v cr="$crop" -v e="$effort" -v t="$threads" -v tr="$tree" \
       '$1==c && $3==cr && $7==e && $8==t && $9==tr && $10=="0" {found=1} END {exit !found}' "$OUT"; then
    echo "--- skip $tag (recorded)" | tee -a "$LOGDIR/driver.log"
    return 0
  fi
  refresh_marker "$tag"
  echo "=== cell $tag ($(date -u +%H:%M:%SZ)) ===" | tee -a "$LOGDIR/driver.log"
  local envcrop=()
  [ "$crop" != "full" ] && envcrop=(env "MEM_PROBE_CROP=$crop")
  "${envcrop[@]}" nice -n 19 "$PROBE" "$src" lossless "$effort" 0 8 "$layout" "$threads" "$tree" \
    > "$log" 2>&1
  local rc=$?
  local row
  row=$(grep '^delta_kb=' "$log" | head -1)
  get() { echo "$row" | grep -o "$1=[^ ]*" | head -1 | cut -d= -f2; }
  local w h
  if [ "$crop" = "full" ]; then
    read -r w h < <(python3 -c "import struct,sys;d=open(sys.argv[1],'rb').read(24);print(*struct.unpack('>II',d[16:24]))" "$src")
  else
    w="${crop%x*}"; h="${crop#*x}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\tlossless\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${content}-${layout}" "$(basename "$src")" "$crop" "$w" "$h" "$effort" "$threads" "$tree" "$rc" \
    "$(get bytes)" "$(get wall_ms)" "$(get live_pre_kb)" "$(get peak_live_kb)" \
    "$(get marginal_live_kb)" "$(get allocs)" "$(get est_typ_kb)" "$(get est_max_kb)" >> "$OUT"
  echo "  rc=$rc peak_live_kb=$(get peak_live_kb) marginal_kb=$(get marginal_live_kb) wall_ms=$(get wall_ms) bytes=$(get bytes)" \
    | tee -a "$LOGDIR/driver.log"
}
for crop in 1024x1024 3840x2160; do
  for e in 7 9; do
    for t in 1 8; do cell_layout photo "$PHOTO" "$crop" "$e" "$t" sectioned rgba; done
  done
done
cell_layout screen "$SCREEN" 1313x4096 7 8 sectioned rgba
cell_layout photo "$PHOTO" full 7 2 sectioned rgb

echo "grid complete -> $OUT" | tee -a "$LOGDIR/driver.log"
