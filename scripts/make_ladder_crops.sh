#!/usr/bin/env bash
# Regenerate the exact inputs behind every 2026-09-10 ladder / phase-scaling
# measurement (benchmarks/ladder_vs_cjxl*_2026-09-10.*,
# benchmarks/e5_t8_phase_scaling_2026-09-10.md, the *_parallel_ab_* records).
#
# Before this script existed those inputs lived only in ~/tmp/chroma on one Mac,
# and no benchmark record named the source files — so the grid could not be
# reproduced anywhere else. Provenance was recovered from the session transcript
# and this script was checked to reproduce the original files BYTE-FOR-BYTE
# (see docs/PERFORMANCE.md, "Reproducing").
#
# Sources: imazen-26, branch variant/png-v3 (the SDR renders), checked out at
# $IMAZEN26_PNG (default ~/work/zen/imazen-26-png-v3/png-v3). Pull LFS content
# for these four files first if the checkout is sparse.
#
# Crops are CENTRE crops (`-gravity center`) with metadata STRIPPED: an iCCP
# chunk changes how cjxl reads the input, which silently changes the reference
# arm (see the iCCP trap in CLAUDE.md). plots and webshot have no 2048 crop —
# their sources are smaller than 2048 on one axis, and nothing is upscaled.
#
# Naming matches what scripts/ladder_vs_cjxl.py and the A/B harnesses were run
# on: the 1024 crop is the BARE name (nature.png), the others are suffixed.
#
# Usage: scripts/make_ladder_crops.sh [out_dir]      (default ~/tmp/chroma)
set -euo pipefail
B=${IMAZEN26_PNG:-$HOME/work/zen/imazen-26-png-v3/png-v3}
OUT=${1:-$HOME/tmp/chroma}
mkdir -p "$OUT"
command -v magick >/dev/null || { echo "needs ImageMagick 7 (magick)" >&2; exit 2; }

declare -a IMGS=(
  "plots:$B/7000-lilith-plots/polygons/7078_plots_line-polygons-00038-s335b5033_1024x1024.sdr.png"
  "webshot:$B/8100-lilith-web-screenshots/1920x1080/8210_web-screenshots_usgs-main-home_dpr1_page2_1920x1080.sdr.png"
  "nature:$B/1400-lilith-nature/1518_nature_pink-flowers-and-bee_colorado_ip8plus_iso40-f2p8_img-2464_3024x4032.sdr.png"
  "brochure:$B/5000-national-park-service-brochures/grayscale/5058_nps_shen-bigmeadows-roadtrail_gray_p02_2550x3300.sdr.png"
)

crop() { # src size dst
  nice -n 19 magick "$1" -gravity center -crop "${2}x${2}+0+0" +repage -strip "$3"
}

for entry in "${IMGS[@]}"; do
  tag=${entry%%:*}; src=${entry#*:}
  [ -f "$src" ] || { echo "missing source: $src" >&2; exit 3; }
  crop "$src" 1024 "$OUT/$tag.png"
  for n in 64 256; do crop "$src" "$n" "$OUT/${tag}_${n}.png"; done
done
for tag in nature brochure; do
  src=$(printf '%s\n' "${IMGS[@]}" | sed -n "s/^$tag://p")
  crop "$src" 2048 "$OUT/${tag}_2048.png"
done
echo "wrote crops to $OUT"
