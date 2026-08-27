#!/bin/bash
# S4 iter-1 elasticity census (registration: benchmarks/s4_iter1_eps_wave_2026-08-27.md).
# Arm A = control (ctrl_exp 1.0 everywhere). Arm B = per-(image,t) ctrl_exp from
# the frozen table (t70 stays 1.0 per registration). 9 refs x t{70,80,88} x k2
# emit-best, v47 bake, baseline arm, TOL=-1, secant ON (defaults).
# Zero loop-code change: JXL_ZENSIM_CTRL_EXP is env, read per encode.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlloop-target}/release/examples/zensim_diffmap_rd}
S4=${S4_OUT:-/mnt/v/output/jxl-encoder/s4-iter1-eps-2026-08-27}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
TAB=$S4/ctrl_exp_table.tsv
CORPUS=$S4/corpus9.tsv
RUN="nice -n19 ionice -c3"
LOG=$S4/census.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
exp_for() { # exp_for <name> <t>  -> table exp or 1.0 (loud on miss for t80/88)
  local e
  e=$(awk -F'\t' -v n="$1" -v t="$2" '$1==n && $2==t {print $3}' "$TAB")
  if [ -z "$e" ]; then
    if [ "$2" != 70 ]; then say "TABLE MISS $1 t$2 -> exp 1.0 (control fallback)"; fi
    e=1.0
  fi
  echo "$e"
}
mkdir -p "$S4/A" "$S4/B"
: > "$S4/cells_A.tsv"; : > "$S4/cells_B.tsv"
first=1
while IFS=$'\t' read -r path name class; do
  for t in 70 80 88; do
    one=$S4/one_corpus.tsv
    printf '%s\t%s\t%s\n' "$path" "$name" "$class" > "$one"
    for arm in A B; do
      if [ "$arm" = A ] || [ "$t" = 70 ]; then EXPV=1.0; else EXPV=$(exp_for "$name" "$t"); fi
      lbl=${name}_t${t}_${arm}
      env JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_EMIT_BEST=1 JXL_ZENSIM_CTRL_EXP=$EXPV \
        $RUN "$BIN" --corpus-file "$one" --zensim-targets "$t" \
        --arms baseline --bake "$V47" --iters 2 --label "$lbl" \
        --out-dir "$S4/$arm" >> "$LOG" 2>&1
      f=$S4/$arm/target_ab_$lbl.tsv
      [ -s "$f" ] || { say "MISSING OUTPUT $f — census void"; exit 1; }
      if [ $first = 1 ]; then head -1 "$f" | sed 's/^/exp\t/' | tee -a "$S4/cells_A.tsv" >> "$S4/cells_B.tsv"; first=0; fi
      tail -n +2 "$f" | sed "s/^/$EXPV\t/" >> "$S4/cells_$arm.tsv"
      say "cell $lbl exp=$EXPV done"
    done
  done
done < "$CORPUS"
say "census complete: $(($(wc -l < "$S4/cells_A.tsv")-1))+$(($(wc -l < "$S4/cells_B.tsv")-1)) cells"
