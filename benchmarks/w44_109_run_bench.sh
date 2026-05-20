#!/bin/bash
# W44-109 bench reproducer.
# Runs the 15-cell terminal target sweep + e8/e9 preservation + codec_wiki +
# photo spot-check + imac_g3 bonus survey.
# Pre-req: built ledger binary at $LEDGER.
set -e

CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/work/zen/jxl-encoder-shared-target}
cargo build -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' --example cjxl_parity_ledger
LEDGER=$CARGO_TARGET_DIR/release/examples/cjxl_parity_ledger

REPO_ROOT=$(jj workspace list 2>/dev/null | head -1 | awk '{print $2}')
[ -n "$REPO_ROOT" ] || REPO_ROOT=$(pwd)
DATE=2026-05-20
OUT_TERMINAL_15="$REPO_ROOT/benchmarks/w44_109_terminal_15_${DATE}.tsv"
OUT_TERMINAL_E89="$REPO_ROOT/benchmarks/w44_109_terminal_e8e9_preservation_${DATE}.tsv"
OUT_CODEC_WIKI="$REPO_ROOT/benchmarks/w44_109_codec_wiki_${DATE}.tsv"
OUT_PHOTOS="$REPO_ROOT/benchmarks/w44_109_photos_spotcheck_${DATE}.tsv"
OUT_IMAC_G3="$REPO_ROOT/benchmarks/w44_109_imac_g3_${DATE}.tsv"

rm -f "$OUT_TERMINAL_15" "$OUT_TERMINAL_E89" "$OUT_CODEC_WIKI" "$OUT_PHOTOS" "$OUT_IMAC_G3"

# 15 terminal cells — primary gate cells (e ∈ {5,6,7} × d ∈ {2,3,4,5,6}).
for eff in 5 6 7; do
    for dist in 2.0 3.0 4.0 5.0 6.0; do
        "$LEDGER" --update --image terminal.png --effort $eff --distance $dist --output "$OUT_TERMINAL_15" 2>&1 | tail -1
    done
done

# 10 terminal cells at e8/e9 — preservation check (W44-105/108 must still work).
for eff in 8 9; do
    for dist in 2.0 3.0 4.0 5.0 6.0; do
        "$LEDGER" --update --image terminal.png --effort $eff --distance $dist --output "$OUT_TERMINAL_E89" 2>&1 | tail -1
    done
done

# 8 codec_wiki cells — W44-107/108 gate-logic preservation + bonus d=4 sweep.
for eff in 5 6 7 8; do
    for dist in 3.0 4.0; do
        "$LEDGER" --update --image codec_wiki.png --effort $eff --distance $dist --output "$OUT_CODEC_WIKI" 2>&1 | tail -1
    done
done

# 45 photo cells — spot-check (5 images × 3 efforts × 3 distances).
for img in 1418519.png 1189261.png 1025469.png 1420710.png 1531677.png; do
    for eff in 5 7 8; do
        for dist in 1.0 2.0 4.0; do
            "$LEDGER" --update --image $img --effort $eff --distance $dist --output "$OUT_PHOTOS" 2>&1 | tail -1
        done
    done
done

# 10 imac_g3 cells — bonus screenshot-class survey across efforts.
# (e8/e9 at d=4/6 currently fail due to a pre-existing OOM/budget issue
#  on 2940×1912 imac_g3.png — not a W44-109 regression. Skipped here.)
for eff in 5 6 7; do
    for dist in 3.0 4.0 6.0; do
        "$LEDGER" --update --image imac_g3.png --effort $eff --distance $dist --output "$OUT_IMAC_G3" 2>&1 | tail -1
    done
done
"$LEDGER" --update --image imac_g3.png --effort 8 --distance 3.0 --output "$OUT_IMAC_G3" 2>&1 | tail -1

echo "DONE."
echo "  terminal-15:  $(wc -l < $OUT_TERMINAL_15) rows"
echo "  terminal-e89: $(wc -l < $OUT_TERMINAL_E89) rows"
echo "  codec_wiki:   $(wc -l < $OUT_CODEC_WIKI) rows"
echo "  photos:       $(wc -l < $OUT_PHOTOS) rows"
echo "  imac_g3:      $(wc -l < $OUT_IMAC_G3) rows"
