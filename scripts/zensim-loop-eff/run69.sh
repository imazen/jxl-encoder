#!/bin/bash
# #69 loop-steering matrix runner — rescued from scratch (~/tmp/attrmap-69/)
# into the repo 2026-07-31 (measurement tooling must not live only in
# scratch). Study doc: benchmarks/zensim_attr_loop69_2026-07-29.md.
# The corpus TSV (path\tname\tclass rows) is the #69 9-ref fixture set; the
# efficiency-study runner (run_eff.sh, same dir) regenerates it — point
# CORPUS at that file or your own.
set -u
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxleff-target}/release/examples/zensim_diffmap_rd}
CORPUS=${CORPUS:-$HOME/tmp/diffmap-eff/corpus9.tsv}
OUT=${OUT:-$HOME/tmp/attrmap-69/ab}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
BLIN=${B_BAKE:-$HOME/work/zen/zensim/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin}
ARMS=baseline,abs,attr,h1-signed,h2-ctrl,h3-mag
mkdir -p "$OUT"
LOG=$OUT/run69.log
echo "# 69 matrix $(date -u +%FT%TZ)" > "$LOG"
nice -n19 ionice -c3 "$BIN" --corpus-file "$CORPUS" --zensim-targets 70,80,88 \
  --arms "$ARMS" --bake "$V47" --iters 6 --label v47A --out-dir "$OUT" >> "$LOG" 2>&1
echo "v47A done $(date -u +%FT%TZ)" >> "$LOG"
nice -n19 ionice -c3 "$BIN" --corpus-file "$CORPUS" --zensim-targets 70,80,88 \
  --arms "$ARMS" --bake "$BLIN" --iters 6 --label shippedB --out-dir "$OUT" >> "$LOG" 2>&1
echo "69 MATRIX DONE $(date -u +%FT%TZ)" >> "$LOG"
