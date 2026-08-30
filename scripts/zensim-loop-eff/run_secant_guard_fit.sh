#!/bin/bash
# Fit the diffmap secant's guard thresholds (T2, 2026-08-30).
#
# Two guards, both swept by this one runner so the arms are directly comparable:
#   AXIS=dlnl -> JXL_ZENSIM_SECANT_MIN_DLNL, the min |Δln L| registered in
#                benchmarks/zensim_secant_2026-08-25.md and shipped with a
#                GUESSED 1e-3 (the NUMERATOR of ε̂ = Δln L / Δln S).
#   AXIS=eps  -> JXL_ZENSIM_SECANT_MIN_EPS, the min |ε̂| trust region (the
#                DENOMINATOR of the step (ln L_t − ln L)/ε̂, i.e. what actually
#                sets how far the controller travels).
# Each threshold is swept as a controller hyperparameter over a log grid and
# judged on the instrument's own decoded columns, so a shipped default is fitted
# rather than assumed. Protocol + results:
#   benchmarks/zensim_secant_min_dlnl_2026-08-30.md
#
# Arms: $THRESHOLDS on the chosen axis (0 = that guard OFF) plus a
# JXL_ZENSIM_SECANT=0 power-law control, x K in $KS, emit-best, on a 9-ref
# x 3-target grid (27 cells per arm). The C bake + h3-mag, matching
# benchmarks/zensim_secant_2026-08-25.md.
#
# Build first (from repo root):
#   CARGO_TARGET_DIR=$HOME/tmp/jxlloop-target nice -n19 cargo build --release \
#     -j 4 -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop ssim2-loop parallel"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlloop-target}/release/examples/zensim_diffmap_rd}
CORPUS=${CORPUS_TSV:-$HOME/tmp/t2sec/corpus9.tsv}
CBAKE=${CBAKE_BAKE:-$HOME/work/zen/zensim/zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin}
THRESHOLDS=${THRESHOLDS:-"0 1e-4 1e-3 3e-3 6e-3 1e-2 2e-2 3e-2 6e-2 1e-1 2e-1 3e-1 6e-1"}
KS=${KS:-"2 3"}
TARGETS=${TARGETS:-70,80,88}
AXIS=${AXIS:-dlnl}
case "$AXIS" in
  dlnl) GUARD_VAR=JXL_ZENSIM_SECANT_MIN_DLNL ;;
  eps)  GUARD_VAR=JXL_ZENSIM_SECANT_MIN_EPS ;;
  *)    echo "STOP: AXIS must be dlnl or eps (got '$AXIS')"; exit 1 ;;
esac
# emit-best (the shipped default) or emit-last. emit-last is the arm the
# 2026-08-25 note said the un-guarded overshoot actually hurt, so a guard
# confirmation must cover it.
EMIT=${EMIT:-best}
case "$EMIT" in
  best) EMIT_ENV=(JXL_ZENSIM_EMIT_BEST=1) ;;
  last) EMIT_ENV=() ;;
  *)    echo "STOP: EMIT must be best or last (got '$EMIT')"; exit 1 ;;
esac
TAG=${TAG:-$AXIS}
OUT=${TS_OUT:-$HOME/tmp/t2sec/sweep-$TAG}
RUN="nice -n19"

[ -f "$BIN" ]    || { echo "STOP: harness missing at $BIN"; exit 1; }
[ -f "$CBAKE" ]  || { echo "STOP: C bake missing at $CBAKE"; exit 1; }
[ -f "$CORPUS" ] || { echo "STOP: corpus missing at $CORPUS"; exit 1; }

# The harness APPENDS to the trace/probe files, so a rerun on a populated dir
# doubles every engagement count. Preserve any prior run, then start clean.
[ -d "$OUT" ] && mv "$OUT" "$OUT.bak.$(date -u +%s)"
mkdir -p "$OUT"
LOG=$OUT/run.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
NCELLS=$(( $(wc -l < "$CORPUS") * ($(tr ',' '\n' <<<"$TARGETS" | wc -l)) ))
say "corpus=$CORPUS cells/arm=$NCELLS axis=$AXIS ($GUARD_VAR) emit=$EMIT thresholds='$THRESHOLDS' ks='$KS'"

run_arm() { # run_arm <label> <K> <env...>
  local lbl=$1 K=$2; shift 2
  say "arm $lbl K=$K env: $*"
  env "$@" ${EMIT_ENV[@]+"${EMIT_ENV[@]}"} JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_TRACE="$OUT/trace_$lbl.tsv" \
    JXL_ZENSIM_ATTR_PROBE="$OUT/probe_$lbl.tsv" \
    JXL_ZENSIM_SECANT_TRACE="$OUT/sec_$lbl.tsv" \
    $RUN "$BIN" --corpus-file "$CORPUS" --zensim-targets "$TARGETS" \
      --arms h3-mag --bake "$CBAKE" --iters "$K" --label "$lbl" --out-dir "$OUT" \
      >> "$LOG" 2>&1
}

for K in $KS; do
  run_arm "ctrl_k${K}" "$K" JXL_ZENSIM_SECANT=0
  for th in $THRESHOLDS; do
    run_arm "g${th}_k${K}" "$K" JXL_ZENSIM_SECANT=1 "$GUARD_VAR=$th"
  done
done

# Engagement gates: h3-mag steers 1..K => probe exactly NCELLS*K lines; the
# per-compare trace exactly NCELLS*(K+1) rows; the controller trace NCELLS*K
# rows (one controller step per steered iterate). A silent fall-through would
# make the whole sweep a null comparison, so gate it.
fail=0
for K in $KS; do
  for lbl in "ctrl_k${K}" $(for th in $THRESHOLDS; do echo "g${th}_k${K}"; done); do
    n=$(wc -l < "$OUT/probe_$lbl.tsv" 2>/dev/null || echo 0)
    tn=$(wc -l < "$OUT/trace_$lbl.tsv" 2>/dev/null || echo 0)
    sn=$(wc -l < "$OUT/sec_$lbl.tsv" 2>/dev/null || echo 0)
    say "ENGAGE $lbl probe=$n/$((NCELLS * K)) trace=$tn/$((NCELLS * (K + 1))) sec=$sn/$((NCELLS * K))"
    [ "$n" -eq "$((NCELLS * K))" ] || fail=1
    [ "$tn" -eq "$((NCELLS * (K + 1)))" ] || fail=1
    [ "$sn" -eq "$((NCELLS * K))" ] || fail=1
  done
done
[ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }

# Cells TSV in the committed 23shot shape (verdict_23shot_cells.py reads it).
BD=${TS_BD:-$REPO/benchmarks}
CELLS=$BD/zensim_secant_${TAG}_cells_2026-08-30.tsv
{
  printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
  for f in "$OUT"/target_ab_*.tsv; do
    [ -f "$f" ] || continue
    run=$(basename "$f" .tsv); run=${run#target_ab_}
    awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
  done
} > "$CELLS"
wc -l "$CELLS" | tee -a "$LOG"
# Controller traces: one committed TSV with a header + a `run` column.
STRACE=$BD/zensim_secant_${TAG}_ctrltrace_2026-08-30.tsv
{
  printf 'run\ttrace_id\titer\tln_L\td_ln_L\td_ln_S\teps_hat\tused\tg_pow\tg_raw\tg\n'
  for f in "$OUT"/sec_*.tsv; do
    [ -f "$f" ] || continue
    run=$(basename "$f" .tsv); run=${run#sec_}
    awk -F'\t' -v r="$run" '{ print r "\t" $0 }' "$f"
  done
} > "$STRACE"
wc -l "$STRACE" | tee -a "$LOG"

python3 "$REPO/scripts/zensim-loop-eff/verdict_23shot_cells.py" "$CELLS" \
  | tee "$OUT/verdict.txt" | tee -a "$LOG"
say "done"
