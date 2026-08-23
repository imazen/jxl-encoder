#!/usr/bin/env bash
# Three-arm A/B over a list of PNGs: ours-default vs ours-with-env vs cjxl,
# djxl-decoded, scored with the decoder-independent `ssim2_png_pair`.
#
# Promoted from the 2026-08-21 plots/NOAA ad-hoc loops, with the discipline
# those loops lacked: every encode/decode is rc-checked and a failed arm is
# recorded as `fail` (bytes=NA) — never read from a stale temp file (the
# 5314 row in noaa_lossless_refresh_2026-08-21.tsv was exactly that bug).
#
# usage:
#   ab_env_cells.sh <out.tsv> "<ENV=VAL ...>" <effort> "<d1 d2 ...>" <png>...
#   e.g. ab_env_cells.sh ~/tmp/ab.tsv "JXL_W44_184_FORCE_LIBJXL_NEWTON=1" 7 "1.0 2.25 3.5" imgs/*.png
# Lossless: pass effort with a leading "L" (e.g. L5) — uses -q 100, one arm
# per encoder, ssim2 skipped (bytes only, pixel-exactness is the decoder's job).
#
# env: CJXL (cjxl binary), DJXL (djxl binary), CJXLRS (ours; default target/release/cjxl-rs),
#      SSIM2_PAIR (default target/release/examples/ssim2_png_pair)
set -u
out=$1; envspec=$2; effort=$3; dists=$4; shift 4
CJXL=${CJXL:-$HOME/work/jxl-efforts/libjxl/build/tools/cjxl}
DJXL=${DJXL:-$HOME/work/jxl-efforts/libjxl/build/tools/djxl}
CJXLRS=${CJXLRS:-./target/release/cjxl-rs}
SSIM2_PAIR=${SSIM2_PAIR:-./target/release/examples/ssim2_png_pair}
W=${AB_WORK:-$HOME/tmp/ab_env_cells.$$}; mkdir -p "$W"
lossless=0; case "$effort" in L*) lossless=1; effort=${effort#L};; esac
echo -e "image\tarm\teffort\tdistance\tbytes\tssim2\tstatus" > "$out"
row() { echo -e "$1\t$2\t$effort\t$3\t${4:-NA}\t${5:-NA}\t$6" >> "$out"; }
for f in "$@"; do
  id=$(basename "$f" | cut -d_ -f1)
  if [ $lossless = 1 ]; then dlist="lossless"; else dlist=$dists; fi
  for d in $dlist; do
    for arm in def env cjxl; do
      rm -f "$W/$arm.jxl" "$W/$arm.png"
      case $arm in
        def)  if [ $lossless = 1 ]; then nice -n19 "$CJXLRS" "$f" "$W/def.jxl" -q 100 -e "$effort" >/dev/null 2>&1; else nice -n19 "$CJXLRS" "$f" "$W/def.jxl" -d "$d" -e "$effort" >/dev/null 2>&1; fi ;;
        env)  [ -z "$envspec" ] && continue
              if [ $lossless = 1 ]; then env $envspec nice -n19 "$CJXLRS" "$f" "$W/env.jxl" -q 100 -e "$effort" >/dev/null 2>&1; else env $envspec nice -n19 "$CJXLRS" "$f" "$W/env.jxl" -d "$d" -e "$effort" >/dev/null 2>&1; fi ;;
        cjxl) if [ $lossless = 1 ]; then nice -n19 "$CJXL" "$f" "$W/cjxl.jxl" -q 100 -e "$effort" >/dev/null 2>&1; else nice -n19 "$CJXL" "$f" "$W/cjxl.jxl" -d "$d" -e "$effort" >/dev/null 2>&1; fi ;;
      esac
      if [ ! -s "$W/$arm.jxl" ]; then row "$id" "$arm" "$d" NA NA encode_fail; continue; fi
      bytes=$(stat -c%s "$W/$arm.jxl" 2>/dev/null || stat -f%z "$W/$arm.jxl")
      if [ $lossless = 1 ]; then row "$id" "$arm" "$d" "$bytes" NA ok; continue; fi
      if ! "$DJXL" "$W/$arm.jxl" "$W/$arm.png" >/dev/null 2>&1 || [ ! -s "$W/$arm.png" ]; then row "$id" "$arm" "$d" "$bytes" NA decode_fail; continue; fi
      s=$("$SSIM2_PAIR" "$f" "x=$W/$arm.png" 2>/dev/null | cut -f2 | cut -d= -f2)
      if [ -z "$s" ]; then row "$id" "$arm" "$d" "$bytes" NA score_fail; else row "$id" "$arm" "$d" "$bytes" "$s" ok; fi
    done
  done
  echo "PROG $id" >&2
done
echo "AB-DONE $out" >&2
