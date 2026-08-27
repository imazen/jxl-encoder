#!/bin/bash
# RD-vs-independent-judges A/B (registration: benchmarks/zensim_loop_rd_independent_judges_2026-08-27.md).
# Arm R = shipped defaults; arm N = ZENSIM_FACTOR_MAX=1.0 (redistribution structurally dead).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlloop-target}/release/examples/zensim_diffmap_rd}
OUT=${RD_OUT:-/mnt/v/output/jxl-encoder/rd-judges-2026-08-27}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
CORPUS=${CORPUS:-/mnt/v/output/jxl-encoder/s4-iter1-eps-2026-08-27/corpus9.tsv}
RUN="nice -n19 ionice -c3"
LOG=$OUT/run.log
mkdir -p "$OUT/R" "$OUT/N"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
while IFS=$'\t' read -r path name class; do
  one=$OUT/one_corpus.tsv
  printf '%s\t%s\t%s\n' "$path" "$name" "$class" > "$one"
  for t in 70 80 88; do
    for arm in R N; do
      if [ "$arm" = N ]; then FM=1.0; else FM=1.15; fi
      lbl=${name}_t${t}_${arm}
      env JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_EMIT_BEST=1 ZENSIM_FACTOR_MAX=$FM \
        $RUN "$BIN" --corpus-file "$one" --zensim-targets "$t" \
        --arms baseline --bake "$V47" --iters 3 --label "$lbl" \
        --out-dir "$OUT/$arm" >> "$LOG" 2>&1
      f=$OUT/$arm/target_ab_$lbl.tsv
      [ -s "$f" ] || { say "MISSING OUTPUT $f — run void"; exit 1; }
      say "cell $lbl done"
    done
  done
done < "$CORPUS"
say "all cells done"
