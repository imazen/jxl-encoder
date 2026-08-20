#!/bin/bash
# Fetch the imazen-26 gate-cell images into $CODEC_CORPUS_DIR.
#
# The imazen-26 corpus images are NOT in the codec-corpus git repo (only
# manifests are tracked) — the canonical public mirror is R2 over plain
# HTTPS (imazen-26/ACCESS.md). Every file is sha256-pinned here; a hash
# mismatch fails loudly (R2 drift or truncation must never silently
# change gate pixels). Used by .github/workflows/nightly.yml AND local
# gate runs. All five are 8-bit RGB (color-type 2) PNGs with only
# IHDR/IDAT/IEND(+pHYs) chunks — safe for cjxl 0.12 and loaded
# identically by the gate's to_rgb8() path (verified 2026-08-20).
set -euo pipefail
DIR="${CODEC_CORPUS_DIR:?set CODEC_CORPUS_DIR}"
BASE="https://codec-corpus.r2.imazen.org/imazen-26-unprocessed"

fetch() { # <sha256> <rel-path-under-imazen-26>
  local sha="$1" rel="$2" dst="$DIR/imazen-26/$2"
  if [ -f "$dst" ] && echo "$sha  $dst" | sha256sum -c --quiet - 2>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  curl -sfL "$BASE/$rel" -o "$dst"
  echo "$sha  $dst" | sha256sum -c --quiet -
}

fetch ff4cd87728467925a9ec9013d9b71b6fc5458acee2692ce3087fc7d3b20b1373 \
  "5300-noaa-hurricane-documents/5308_noaa_nhc-al022024-beryl_p01_2550x3300.png"
fetch 786737436f55090e58ba85b6a92fdac02dfe6cacb8718d62a7c1faf4f3a66daf \
  "7000-lilith-plots/aliased-polygons/7026_plots_line-00081-s68404bb1_1024x1024.png"
fetch 97cb57c03b01bd634c4129fb42cd8f4be1afe6de3d5b50b076cc23a163734694 \
  "8100-lilith-web-screenshots/1440x900/8106_web-screenshots_climate-news_dpr1_page1_1440x900.png"
fetch bd161a7d566c80d14230bbf4346f413210d93e9f4183fe204d4cb9be4d6e1c62 \
  "9226-lilith-ai-products/grocery/9678_gen_products-grocery_cereal-box-coral_p0439_1024x1536.png"
fetch e0825408d8b880ea80385d9d28af238f636b9f997b33770d9473ddc0bccec22c \
  "6800-ia-scans-manuscript-text/6824_scans-text_redoute-fr-description_p0052_2415x3528.png"
echo "imazen-26 gate files present + sha-verified in $DIR/imazen-26/"
