#!/usr/bin/env bash
# W44-105 bench: pair our cjxl-rs (with the buttloop qf seed-scale fix) against cjxl reference
# on 3 cell groups:
#   1. terminal × {e7, e8, e9} × {d=2, 3, 4, 5, 6}        = 15 cells (PRIMARY)
#   2. 10 gb82-sc screenshots × {e8, e9} × {d=2, 4}        = 40 cells (CLASS generalization)
#   3. 30 spot-check FIXED photo cells (no regression)
set -euo pipefail
WS=/home/lilith/work/zen/jxl-encoder--w44-105-buttloop-qac-fix
CJXL_OURS=$HOME/work/zen/jxl-encoder-shared-target/release/cjxl-rs
CJXL_REF=/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl
DJXL=/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl
SSIM2=$HOME/work/zen/jxl-encoder-shared-target/release/fast-ssim2-cli
BFLY=/home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main

CORPUS_SCREEN=/home/lilith/work/codec-corpus/gb82-sc
CORPUS_PHOTO=/home/lilith/work/codec-corpus/CID22/CID22-512/validation

OUT_TSV=$WS/benchmarks/w44_105_buttloop_qac_fix_2026-05-20.tsv
LOG=/tmp/w44_105_bench.log

mkdir -p /tmp/w44_105

echo "# W44-105 buttloop qf seed-scale fix vs baseline (this commit) and cjxl reference" > $OUT_TSV
echo "# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) by $USER" >> $OUT_TSV
echo "# WS: $WS" >> $OUT_TSV
echo "# Branch info: $(cd $WS && jj log -r @ --no-graph -T 'change_id ++ " " ++ description.first_line()' 2>&1 | head -1)" >> $OUT_TSV
printf "cell_group\timage\teffort\tdistance\tours_bytes\tours_ssim2\tours_bfly\tcjxl_bytes\tcjxl_ssim2\tcjxl_bfly\tdelta_bytes_pct\tdelta_ssim2\tdelta_bfly_pct\n" >> $OUT_TSV

bench_cell() {
  local group=$1 img=$2 effort=$3 distance=$4
  local src="$img"
  local stem=$(basename "$img" .png)
  local ours_jxl=/tmp/w44_105/ours_${stem}_e${effort}_d${distance}.jxl
  local cjxl_jxl=/tmp/w44_105/cjxl_${stem}_e${effort}_d${distance}.jxl
  local ours_png=/tmp/w44_105/ours_${stem}_e${effort}_d${distance}.png
  local cjxl_png=/tmp/w44_105/cjxl_${stem}_e${effort}_d${distance}.png

  # Encode
  $CJXL_OURS -e $effort -d $distance "$src" "$ours_jxl" >/dev/null 2>&1 || { echo "FAIL ours $group $img e$effort d$distance" >> $LOG; return; }
  $CJXL_REF -e $effort -d $distance "$src" "$cjxl_jxl" >/dev/null 2>&1 || { echo "FAIL cjxl $group $img e$effort d$distance" >> $LOG; return; }
  # Decode
  $DJXL "$ours_jxl" "$ours_png" 2>/dev/null
  $DJXL "$cjxl_jxl" "$cjxl_png" 2>/dev/null
  # Metrics
  local ours_b=$(stat -c%s "$ours_jxl")
  local cjxl_b=$(stat -c%s "$cjxl_jxl")
  local ours_s=$($SSIM2 image "$src" "$ours_png" 2>&1 | tail -1 | sed 's/Score: //' | tr -d '[:space:]')
  local cjxl_s=$($SSIM2 image "$src" "$cjxl_png" 2>&1 | tail -1 | sed 's/Score: //' | tr -d '[:space:]')
  local ours_bf=$($BFLY "$src" "$ours_png" 2>&1 | head -1 | tr -d '[:space:]')
  local cjxl_bf=$($BFLY "$src" "$cjxl_png" 2>&1 | head -1 | tr -d '[:space:]')
  # Delta calculations
  local dbytes=$(awk "BEGIN { if ($cjxl_b > 0) printf \"%.2f\", ($ours_b - $cjxl_b) / $cjxl_b * 100; else print \"NaN\" }")
  local dssim2=$(awk "BEGIN { printf \"%.2f\", $ours_s - $cjxl_s }")
  local dbfly=$(awk "BEGIN { if ($cjxl_bf > 0) printf \"%.2f\", ($ours_bf - $cjxl_bf) / $cjxl_bf * 100; else print \"NaN\" }")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$group" "$stem" "$effort" "$distance" "$ours_b" "$ours_s" "$ours_bf" "$cjxl_b" "$cjxl_s" "$cjxl_bf" "$dbytes" "$dssim2" "$dbfly" >> $OUT_TSV
  echo "$group $stem e$effort d$distance ours=${ours_b}B/${ours_s}/${ours_bf}  cjxl=${cjxl_b}B/${cjxl_s}/${cjxl_bf}  Δ%bytes=$dbytes Δssim2=$dssim2 Δ%bfly=$dbfly"
}

# Group 1: terminal primary target
for e in 7 8 9; do
  for d in 2 3 4 5 6; do
    bench_cell "terminal" "$CORPUS_SCREEN/terminal.png" $e $d
  done
done

# Group 2: 10 gb82-sc screenshots at e8/e9, d=2 and d=4
SCREENS="codec_wiki gmessages graph gui imac_dark imac_g3 imessage terminal windows windows95"
for img in $SCREENS; do
  for e in 8 9; do
    for d in 2 4; do
      bench_cell "screen" "$CORPUS_SCREEN/${img}.png" $e $d
    done
  done
done

# Group 3: 30 spot-check FIXED photo cells (no regression). Pick stratified images at varied d.
if [ -d "$CORPUS_PHOTO" ]; then
  # Pick first 10 photos × 3 distances (d=1, d=2, d=4)
  PHOTOS=$(ls $CORPUS_PHOTO/*.png 2>/dev/null | head -10)
  for img in $PHOTOS; do
    for e in 8 9; do
      for d in 2 4; do
        bench_cell "photo" "$img" $e $d
      done
    done
  done
fi

echo "DONE $OUT_TSV"
