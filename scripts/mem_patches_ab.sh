#!/usr/bin/env bash
# mem_patches_ab.sh — MEM_PROBE_PATCHES=1|0 A/B + re-pin grid for the
# lossless patches-phase lifetime measurement (imazen/jxl-encoder#96 residual,
# issue #99 follow-ups). Produced benchmarks/jxl_sectioned_patches_lifetime_
# 2026-08-30.tsv (run once per phase: before a change, and after — merge the
# TSVs with a phase column, or diff the peaks directly).
#
# Grid: sectioned tree at photo crops 512^2..full x e{7,9} x t1 (+t4 spots on
# the full frames), imac_dark + reddit.com full frames, patches forced 1 then
# 0; a global-tree e{5..9} t1 arm on the two screens (default patches) covers
# the whole-image band cells; rgba spot cells match the heuristics pins.
# Peak attribution for any surprising cell: JXL_ALLOC_SITES=1 (see mem_probe).
#
# Probe build:
#   nice -n 19 cargo build -j 4 -p jxl-encoder-cli --release --example mem_probe
#
# Usage: scripts/mem_patches_ab.sh <out.tsv> <photo.png> <imac.png> <reddit.png>
#   photo.png  >= 4000x3000 (imazen-26 1403 nature camera photo, sdr PNG)
#   imac.png   gb82-sc imac_dark.png (2940x1912, palette + patches screenshot)
#   reddit.png qoi-benchmark screenshot_web/reddit.com.png (1313x8008)
# Output columns: content crop channels tree effort threads patches rc bytes
#   wall_ms live_pre_kb peak_live_kb marginal_live_kb allocs
# Resumable: recorded rc=0 cells are skipped on re-run.
set -uo pipefail
OUT="${1:?usage: mem_patches_ab.sh <out.tsv> <photo.png> <imac.png> <reddit.png>}"
PHOTO="${2:?photo png}"
IMAC="${3:?imac_dark png}"
REDDIT="${4:?reddit png}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBE="$REPO_ROOT/target/release/examples/mem_probe"
[ -x "$PROBE" ] || { echo "probe not built: $PROBE" >&2; exit 2; }

if [ ! -s "$OUT" ]; then
  {
    printf '# mem_patches_ab.sh  commit=%s  host=%s  date=%s\n' \
      "$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
      "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'content\tcrop\tchannels\ttree\teffort\tthreads\tpatches\trc\tbytes\twall_ms\tlive_pre_kb\tpeak_live_kb\tmarginal_live_kb\tallocs\n'
  } > "$OUT"
fi

refresh_marker() {
  printf '%s mem_patches_ab issue99: %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "$REPO_ROOT/.workongoing"
}

cell() { # content src crop channels effort threads tree patches(1|0|default)
  local content=$1 src=$2 crop=$3 ch=$4 e=$5 t=$6 tree=$7 p=$8
  if awk -F'\t' -v c="$content" -v cr="$crop" -v ch="$ch" -v tr="$tree" \
       -v e="$e" -v t="$t" -v p="$p" \
    '$1==c&&$2==cr&&$3==ch&&$4==tr&&$5==e&&$6==t&&$7==p&&$8=="0"&&$9!=""{f=1}END{exit !f}' "$OUT"; then
    echo "skip $content $crop $ch $tree e$e t$t p$p"; return; fi
  refresh_marker "$content $crop $tree e$e t$t p$p"
  local -a envargs=()
  [ "$crop" != full ] && envargs+=("MEM_PROBE_CROP=$crop")
  [ "$p" != default ] && envargs+=("MEM_PROBE_PATCHES=$p")
  local line rc
  line=$(cd "$REPO_ROOT" && env -u MEM_PROBE_CROP -u MEM_PROBE_PATCHES "${envargs[@]}" \
    nice -n 19 "$PROBE" "$src" lossless "$e" 0 8 "$ch" "$t" "$tree" 2>&1 \
    | grep '^delta_kb=' | tail -1)
  [ -n "$line" ]; rc=$?
  get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$content" "$crop" "$ch" "$tree" "$e" "$t" "$p" "$rc" \
    "$(get bytes)" "$(get wall_ms)" "$(get live_pre_kb)" "$(get peak_live_kb)" \
    "$(get marginal_live_kb)" "$(get allocs)" >> "$OUT"
  echo "done $content $crop $ch $tree e$e t$t p$p peak=$(get peak_live_kb) bytes=$(get bytes)"
}

# ── patches A/B, sectioned (the lifetime measurement) ──
for p in 1 0; do
  for e in 7 9; do
    for crop in 512x512 1024x1024 2048x2048 3840x2160 full; do
      cell photo "$PHOTO" "$crop" rgb "$e" 1 sectioned "$p"
    done
    cell imac "$IMAC" full rgb "$e" 1 sectioned "$p"
    cell reddit "$REDDIT" full rgb "$e" 1 sectioned "$p"
  done
  cell photo "$PHOTO" full rgb 7 4 sectioned "$p"
  cell imac "$IMAC" full rgb 7 4 sectioned "$p"
  cell reddit "$REDDIT" full rgb 7 4 sectioned "$p"
  cell photo "$PHOTO" full rgb 7 1 global "$p"
  cell imac "$IMAC" full rgb 7 1 global "$p"
  cell reddit "$REDDIT" full rgb 7 1 global "$p"
done

# ── heuristics-pin cells (default patches) ──
for e in 7 9; do for t in 1 4 12; do cell imac "$IMAC" full rgb "$e" "$t" sectioned default; done; done
cell reddit "$REDDIT" full rgb 7 1 sectioned default
cell reddit "$REDDIT" full rgb 9 12 sectioned default
cell reddit "$REDDIT" 1313x4096 rgb 9 12 sectioned default
cell reddit "$REDDIT" 1313x4096 rgba 7 8 sectioned default
cell reddit "$REDDIT" 256x256 rgb 9 12 sectioned default
cell photo "$PHOTO" 1024x1024 rgba 7 1 sectioned default
cell photo "$PHOTO" 3840x2160 rgba 7 8 sectioned default
for e in 5 6 7 8 9; do
  cell imac "$IMAC" full rgb "$e" 1 global default
  cell reddit "$REDDIT" full rgb "$e" 1 global default
done
cell reddit "$REDDIT" 1313x4096 rgb 7 1 global default
cell reddit "$REDDIT" 1313x4096 rgb 9 1 global default
echo ALL-DONE
