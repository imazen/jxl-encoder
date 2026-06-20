#!/usr/bin/env bash
# vCPU resource sweep — measure real peak heap / peak RSS / marginal working
# set + wall time across (size x path x effort x THREAD-COUNT) for the jxl
# encoder, alongside the heuristics::estimate_encode prediction (emitted by
# the driver in the same row). This ADDS the thread/vCPU axis that
# estimate_encode does not model (calibrated threads=1).
#
# Two runs per cell:
#   clean    -> driver's own VmHWM delta (marginal WS), VmHWM peak (whole-proc
#               peak RSS, no instrumentation), wall/user/sys, est_*, bytes
#   heaptrack-> allocator PEAK_HEAP (true heap high-water)
#
# One process per cell (RSS/VmHWM high-water is per-process). The whole sweep
# runs under run-heavy (nice+ionice+cgroup cap); cells are serial within.
#
# Usage: scripts/vcpu_resource_sweep.sh <driver_bin> <img_dir> <out.tsv>
#   img_dir must contain pre-generated downscale-only PNGs named <label>.png
#   where <label> encodes the square size (see IMAGES below).
set -uo pipefail

DRIVER="${1:?driver bin}"
IMGDIR="${2:?image dir}"
OUT="${3:?out tsv}"
HT_DIR="${HT_DIR:-/tmp/jxl_vcpu_heaptrack}"
mkdir -p "$HT_DIR"

# WSL2 glibc arena policy swings single-sample RSS +/-120 MB; pin it (per the
# zenjpeg/issue#64 memory-A/B note) so peaks are comparable across cells.
export GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072

# (label content_class) — square PNGs at these sizes must exist in IMGDIR.
IMAGES=( "256:photo" "1024:photo" "2048:photo" )
# (path effort distance) cells. lossless e7 = MA tree-learning (per-worker
# SplitWorkspace -> the gamma*threads memory term); lossy e9 = buttloop.
CELLS=( "lossless 7 0.0" "lossy 9 2.0" )
THREADS=( 1 2 4 8 16 28 )
# heaptrack (allocator PEAK_HEAP) is ~2-10x slower than a clean run; restrict
# its (expensive) arm to a thread subset — the clean run's VmHWM peak already
# gives peak RSS at every thread point, heaptrack is the allocator cross-check.
HT_THREADS="${HT_THREADS:-1 8 28}"
DEPTH=8
ALPHA=rgb

# heaptrack_print summary -> peak heap KB and peak RSS KB.
parse_ht() { # $1 = .zst file
  heaptrack_print "$1" 2>/dev/null | python3 -c '
import sys,re
ph=pr=0
# heaptrack_print uses single-letter IEC units with no space: 25.05M = 25.05 MiB.
def kb(v,u):
    f={"B":1/1024,"K":1,"M":1024,"G":1024*1024}.get(u[0].upper(),0); return f*float(v)
for ln in sys.stdin:
    m=re.search(r"peak heap memory consumption:\s*([\d.]+)\s*([KMGB])",ln)
    if m: ph=kb(m.group(1),m.group(2))
    m=re.search(r"peak RSS[^:]*:\s*([\d.]+)\s*([KMGB])",ln)
    if m: pr=kb(m.group(1),m.group(2))
print(f"{int(ph)} {int(pr)}")'
}

getf() { sed -n "s/.*\b$2=\([^ ]*\).*/\1/p" <<<"$1"; }

echo -e "codec\tcontent_class\tsrc\twidth\theight\tpixels\tpath\teffort\tthreads\test_min_kb\test_typ_kb\test_max_kb\test_time_ms\tmeas_peak_heap_kb\tmeas_peak_rss_kb\tmeas_vmhwm_kb\tmeas_delta_kb\tmeas_wall_ms\tmeas_user_ms\tmeas_sys_ms\tbytes\tok" > "$OUT"

total=$(( ${#IMAGES[@]} * ${#CELLS[@]} * ${#THREADS[@]} )); i=0
for spec in "${IMAGES[@]}"; do
  label="${spec%%:*}"; cls="${spec##*:}"; png="$IMGDIR/${label}.png"
  [[ -f "$png" ]] || { echo "MISSING $png — skipping" >&2; continue; }
  for cell in "${CELLS[@]}"; do
    read -r path effort dist <<<"$cell"
    for t in "${THREADS[@]}"; do
      i=$((i+1))
      # refresh marker so a concurrent agent sees this repo is busy
      printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "claude-resource-harness" \
        "vcpu sweep cell $i/$total ${label} ${path} e${effort} t${t}" > .workongoing 2>/dev/null || true
      echo "[$i/$total] ${label}^2 ${path} e${effort} t${t}" >&2

      # --- clean run: time + VmHWM peak/delta + est_* ---
      line=$("$DRIVER" "$png" "$path" "$effort" "$dist" "$DEPTH" "$ALPHA" "$t" 2>/dev/null)
      [[ -z "$line" ]] && { echo "  FAIL clean run" >&2; continue; }
      delta=$(getf "$line" delta_kb);   vmhwm=$(getf "$line" peak_kb)
      wall=$(getf "$line" wall_ms);     user=$(getf "$line" user_ms)
      sys=$(getf "$line" sys_ms);       bytes=$(getf "$line" bytes)
      emin=$(getf "$line" est_min_kb);  etyp=$(getf "$line" est_typ_kb)
      emax=$(getf "$line" est_max_kb);  etime=$(getf "$line" est_time_ms)

      # --- heaptrack run: PEAK_HEAP / PEAK_RSS (thread subset only) ---
      ph=""; pr=""
      if [[ " $HT_THREADS " == *" $t "* ]]; then
        htf="$HT_DIR/${label}_${path}_e${effort}_t${t}"
        rm -f "${htf}.zst"
        heaptrack -o "$htf" "$DRIVER" "$png" "$path" "$effort" "$dist" "$DEPTH" "$ALPHA" "$t" >/dev/null 2>&1
        read -r ph pr < <(parse_ht "${htf}.zst")
      fi

      px=$((label*label))
      echo -e "jxl-encoder\t${cls}\t${label}.png\t${label}\t${label}\t${px}\t${path}\t${effort}\t${t}\t${emin}\t${etyp}\t${emax}\t${etime}\t${ph}\t${pr}\t${vmhwm}\t${delta}\t${wall}\t${user}\t${sys}\t${bytes}\t1" >> "$OUT"
    done
  done
done
echo "wrote $OUT ($(( $(wc -l < "$OUT") - 1 )) rows)" >&2
