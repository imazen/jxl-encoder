#!/usr/bin/env bash
# mem_sectioned_threads_sweep.sh — THREAD-DENSE peak_live grid for the
# lossless SECTIONED local-tree mode, the calibration input for
# `heuristics::estimate_encode_sectioned`'s per-worker term (issue #99
# item 3 / CLAUDE.md T5).
#
# Why a separate script from `mem_sectioned_sweep.sh`: that one samples
# threads {1, 4, 8, 12}. Four points cannot tell a SUM-shaped model
# (floor + g·(t−1), what the estimator had) from a MAX-shaped one
# (max(floor, tree-learn(t))) — the two agree wherever one term dominates
# and differ only through the crossover, which lands between t=1 and t=4
# on 1–4 MP content and above t=4 on ≥ 5 MP content. Threads
# {1,2,3,4,6,8,12} brackets the crossover on every size in the grid.
#
# Axes (source-informing sweep discipline): size tiny→large as REAL crops
# (never resampled) × effort {7,9} (the tree-learning band) × threads
# {1,2,3,4,6,8,12} × content {photo, screen-web, screen-palette}.
#
# The small-crop ladder 64² / 256² / 512×256 / 512² / 768² is deliberate: at
# the default 256-pixel modular group dimension those are 1 / 1 / 2 / 4 / 9
# groups, which is the only place the estimate's `min(threads, groups + 1)`
# in-flight clamp is exercised (every ≥ 2 MP crop has ≥ 31 groups, so the
# clamp is inert there and cannot be validated by the large cells).
#
# `REPEATS=n` re-runs every cell n times (the tree-learn-bound cells vary
# ±8-12 % run to run with worker scheduling; the estimator must cover the
# MAXIMUM, so a single pass under-states the requirement).
#
# Probe: nice -n 19 cargo build -j 4 -p jxl-encoder-cli --release --example mem_probe
# Usage: scripts/mem_sectioned_threads_sweep.sh <out.tsv> <photo.png> <reddit.png> <imac.png>
# Columns: content src crop w h mode effort threads tree rc bytes wall_ms
#          live_pre_kb peak_live_kb marginal_live_kb allocs est_typ_kb est_max_kb rep

set -uo pipefail

OUT="${1:?usage: mem_sectioned_threads_sweep.sh <out.tsv> <photo.png> <reddit.png> <imac.png>}"
PHOTO="${2:?photo png}"
REDDIT="${3:?reddit png}"
IMAC="${4:?imac png}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="$REPO_ROOT/target/release/examples/mem_probe"
LOGDIR="$HOME/tmp/memsectthreads"
mkdir -p "$LOGDIR"

[ -x "$PROBE" ] || { echo "probe not built: $PROBE" >&2; exit 2; }

REPEATS="${REPEATS:-1}"

if [ ! -s "$OUT" ]; then
  {
    printf '# mem_sectioned_threads_sweep.sh  commit=%s  host=%s  date=%s\n' \
      "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
      "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'content\tsrc\tcrop\tw\th\tmode\teffort\tthreads\ttree\trc\tbytes\twall_ms\tlive_pre_kb\tpeak_live_kb\tmarginal_live_kb\tallocs\test_typ_kb\test_max_kb\trep\n'
  } > "$OUT"
fi

refresh_marker() {
  printf '%s claude-T5-estimator-accuracy sectioned thread-dense mem sweep: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "$REPO_ROOT/.workongoing"
}

cell() {
  local content="$1" src="$2" crop="$3" effort="$4" threads="$5" layout="${6:-rgb}"
  local key="$content"
  [ "$layout" != "rgb" ] && key="${content}-${layout}"
  local rep
  for rep in $(seq 1 "$REPEATS"); do
    local tag="${content}_${crop}_e${effort}_t${threads}_${layout}_r${rep}"
    local log="$LOGDIR/${tag}.log"
    if awk -F'\t' -v c="$key" -v cr="$crop" -v e="$effort" -v t="$threads" -v rp="$rep" \
         '$1==c && $3==cr && $7==e && $8==t && $9=="sectioned" && $10=="0" && $19==rp {found=1} END {exit !found}' "$OUT"; then
      echo "--- skip $tag (recorded)" | tee -a "$LOGDIR/driver.log"
      continue
    fi
    refresh_marker "$tag"
    echo "=== cell $tag ($(date -u +%H:%M:%SZ)) ===" | tee -a "$LOGDIR/driver.log"
    local envcrop=()
    [ "$crop" != "full" ] && envcrop=(env "MEM_PROBE_CROP=$crop")
    "${envcrop[@]}" nice -n 19 "$PROBE" "$src" lossless "$effort" 0 8 "$layout" "$threads" sectioned \
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
    printf '%s\t%s\t%s\t%s\t%s\tlossless\t%s\t%s\tsectioned\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$key" "$(basename "$src")" "$crop" "$w" "$h" "$effort" "$threads" "$rc" \
      "$(get bytes)" "$(get wall_ms)" "$(get live_pre_kb)" "$(get peak_live_kb)" \
      "$(get marginal_live_kb)" "$(get allocs)" "$(get est_typ_kb)" "$(get est_max_kb)" "$rep" >> "$OUT"
    echo "  rc=$rc rep=$rep peak_live_kb=$(get peak_live_kb) marginal_kb=$(get marginal_live_kb) wall_ms=$(get wall_ms) bytes=$(get bytes)" \
      | tee -a "$LOGDIR/driver.log"
  done
}

THREADS_DENSE="1 2 3 4 6 8 12"
THREADS_TINY="1 2 4 8 12"

# ── photo (imazen-26 1403, 4000x3000) crops, tiny -> large ─────────────
# The 1/1/2/4/9-group ladder: the only cells where the in-flight clamp bites.
for e in 7 9; do
  for t in $THREADS_TINY; do cell photo "$PHOTO" 64x64   "$e" "$t"; done
  for t in $THREADS_TINY; do cell photo "$PHOTO" 256x256 "$e" "$t"; done
  for t in $THREADS_TINY; do cell photo "$PHOTO" 512x256 "$e" "$t"; done
  for t in $THREADS_TINY; do cell photo "$PHOTO" 512x512 "$e" "$t"; done
  for t in $THREADS_TINY; do cell photo "$PHOTO" 768x768 "$e" "$t"; done
done
# Screen content on the same ladder: a single group of palette / web content
# learns a much cheaper tree than a photo group, so these are the cells that
# say whether the (content-blind) per-worker envelope still covers.
for e in 7 9; do
  for t in 1 2 12; do
    cell imac   "$IMAC"   256x256 "$e" "$t"
    cell reddit "$REDDIT" 256x256 "$e" "$t"
    cell imac   "$IMAC"   512x512 "$e" "$t"
    cell imac   "$IMAC"   768x768 "$e" "$t"
  done
done
for e in 7 9; do
  for t in $THREADS_DENSE; do cell photo "$PHOTO" 1024x1024 "$e" "$t"; done
done
for e in 7 9; do
  for t in $THREADS_DENSE; do cell photo "$PHOTO" 2048x2048 "$e" "$t"; done
done
for e in 7 9; do
  for t in $THREADS_DENSE; do cell photo "$PHOTO" 3840x2160 "$e" "$t"; done
done
for e in 7 9; do
  for t in $THREADS_DENSE; do cell photo "$PHOTO" full "$e" "$t"; done
done

# ── screen-palette (gb82-sc imac_dark 2940x1912): the cell issue #99
#    item 3 is about (palette + ChannelCompact + patches) ──────────────
for e in 7 9; do
  for t in $THREADS_DENSE; do cell imac "$IMAC" full "$e" "$t"; done
done

# ── screen-web (qoi reddit.com 1313x8008) + crop ──────────────────────
for e in 7 9; do
  for t in $THREADS_DENSE; do cell reddit "$REDDIT" 1313x4096 "$e" "$t"; done
done
for e in 7 9; do
  for t in $THREADS_DENSE; do cell reddit "$REDDIT" full "$e" "$t"; done
done

# ── rgba arm (alpha := source green): the alpha per-worker factor ──────
for e in 7 9; do
  for t in 1 2 4 8 12; do cell photo "$PHOTO" 1024x1024 "$e" "$t" rgba; done
done
for e in 7 9; do
  for t in 1 2 4 8 12; do cell photo "$PHOTO" 3840x2160 "$e" "$t" rgba; done
done

echo "grid complete -> $OUT" | tee -a "$LOGDIR/driver.log"
