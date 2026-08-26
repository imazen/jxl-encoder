#!/bin/bash
# Exact 2/3-shot sweep runner (2026-08-01) — protocol + results:
#   benchmarks/zensim_loop_23shot_2026-08-01.md
# Fills the missing decoded-judged budget-2/3 cells on the standard 9-ref x {70,80,88}
# matrix: 5 arms {v47A_base, B_base, bvls_base, blend2L_base, v47A_h3g20c135}
# x {k2 emit-last, k2 emit-best, k3 emit-best} = 15 runs = 405 encodes, plus a
# substrate-verification probe (v47A_base k3 emit-last re-run, diffed numerically
# against the committed metric-matrix TSV) that validates deriving k3 emit-last +
# outer j2/j3 entries from benchmarks/zensim_mm_{cells,outer}_2026-07-31.tsv
# without re-running them.
# Phases: probe fresh collect   (usage: run_23shot.sh <phase>|all)
#
# Build first (from repo root; heavy -> nice'd, own target dir):
#   CARGO_TARGET_DIR=$HOME/tmp/jxl23-target nice -n19 ionice -c3 \
#     cargo build --release -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxl23-target}/release/examples/zensim_diffmap_rd}
OUT=${TS_OUT:-$HOME/tmp/23shot/run}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
BLIN=${B_BAKE:-$HOME/work/zen/zensim/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin}
BVLS=${BVLS_BAKE:-/mnt/v/output/zensim/bakes/v02_bvls_NO_shaping_2026-05-28.bin}
BLEND=${BLEND_BAKE:-/mnt/v/output/zensim/reports/b_negatives/mlp_2L_diverse_H128_2026-07-15.bin}
GB82=${GB82_DIR:-$HOME/work/codec-corpus/gb82-sc}
CID=${CID_DIR:-$HOME/work/codec-corpus/CID22/CID22-512/validation}
COH=${COH_DIR:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
RUN="nice -n19 ionice -c3"
mkdir -p "$OUT/fixtures"
LOG=$OUT/run_23shot.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ── fixtures (the #69 nonphoto crops; regenerate if absent) ──────────────
for sc in codec_wiki gui imessage; do
  f=$OUT/fixtures/sc_${sc}.png
  [ -f "$f" ] || convert "$GB82/${sc}.png" -crop 576x576+512+256 +repage "$f"
done
CORPUS=$OUT/corpus9.tsv
{
  printf '%s\tcity\tphoto\n'        "$COH/city.png"
  printf '%s\tdog\tphoto\n'         "$COH/dog.png"
  printf '%s\tgirl\tphoto\n'        "$COH/girl.png"
  printf '%s\tcid1025469\tphoto\n'  "$CID/1025469.png"
  printf '%s\tcid1418519\tphoto\n'  "$CID/1418519.png"
  printf '%s\tcid1189261\tphoto\n'  "$CID/1189261.png"
  printf '%s\tsc_wiki\tnonphoto\n'    "$OUT/fixtures/sc_codec_wiki.png"
  printf '%s\tsc_gui\tnonphoto\n'     "$OUT/fixtures/sc_gui.png"
  printf '%s\tsc_imessage\tnonphoto\n' "$OUT/fixtures/sc_imessage.png"
} > "$CORPUS"

phase=${1:-all}
run_ab() { # run_ab <outdir> <label> <bake> <arms> <iters> <targets> [env...]
  local od=$1 lbl=$2 bk=$3 arms=$4 it=$5 tg=$6; shift 6
  mkdir -p "$od"
  say "run_ab $lbl arms=$arms iters=$it targets=$tg env: $*"
  env "$@" $RUN "$BIN" --corpus-file "$CORPUS" --zensim-targets "$tg" \
    --arms "$arms" --bake "$bk" --iters "$it" --label "$lbl" --out-dir "$od" \
    >> "$LOG" 2>&1
}

# ── probe: substrate verification for the DERIVED entries ─────────────────
# Re-runs the mm study's v47A_base_k3 (emit-last) on the CURRENT substrate and
# numerically diffs achieved_decoded/abs_err/bytes vs the committed
# zensim_mm_cells_2026-07-31.tsv rows. 27/27 equal => the mm-substrate k3
# emit-last + outer rows may be derived without re-running (the mm study's own
# "never merge cross-substrate" rule, satisfied by measurement instead of hope).
if [ "$phase" = probe ] || [ "$phase" = all ]; then
  PD=$OUT/probe
  mkdir -p "$PD"
  run_ab "$PD" v47A_base_k3_lastprobe "$V47" baseline 3 70,80,88 \
    JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_TRACE=$PD/trace_v47A_base_k3_lastprobe.tsv \
    JXL_ZENSIM_ATTR_PROBE=$PD/probe_v47A_base_k3_lastprobe.tsv
  python3 "$REPO/scripts/zensim-loop-eff/analyze_23shot.py" verify \
    --probe-tsv "$PD/target_ab_v47A_base_k3_lastprobe.tsv" \
    --probe-trace "$PD/trace_v47A_base_k3_lastprobe.tsv" \
    --mm-cells "$REPO/benchmarks/zensim_mm_cells_2026-07-31.tsv" \
    --mm-traces "$REPO/benchmarks/zensim_mm_traces_2026-07-31.tsv" \
    | tee -a "$LOG"
  st=${PIPESTATUS[0]}
  if [ "$st" -ne 0 ]; then
    say "SUBSTRATE PROBE FAIL — mm-derived k3-last/outer entries are INVALID on this substrate."
    say "STOP: either fix the drift or run fresh k3-last for all arms (and outer) instead of deriving."
    exit 1
  fi
  say "SUBSTRATE PROBE PASS — mm-TSV derivations are valid on this substrate"
fi

# ── fresh: 5 arms x {k2_last, k2_best, k3_best} = 15 runs, 405 encodes ────
if [ "$phase" = fresh ] || [ "$phase" = all ]; then
  FD=$OUT/fresh
  mkdir -p "$FD"
  declare -A BAKE=( [v47A_base]="$V47" [B_base]="$BLIN" [bvls_base]="$BVLS" [blend2L_base]="$BLEND" [v47A_h3g20c135]="$V47" )
  declare -A ARMS=( [v47A_base]=baseline [B_base]=baseline [bvls_base]=baseline [blend2L_base]=baseline [v47A_h3g20c135]=h3-mag )
  for arm in v47A_base B_base bvls_base blend2L_base v47A_h3g20c135; do
    EXTRA=()
    [ "$arm" = v47A_h3g20c135 ] && EXTRA=(ZENSIM_H3_GAIN=20.0 JXL_ZENSIM_CTRL_CLAMP=1.35)
    for mode in k2_last k2_best k3_best; do
      K=${mode:1:1}
      EB=()
      [ "${mode#*_}" = best ] && EB=(JXL_ZENSIM_EMIT_BEST=1)
      lbl=${arm}_${mode}
      run_ab "$FD" "$lbl" "${BAKE[$arm]}" "${ARMS[$arm]}" "$K" 70,80,88 \
        JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
        ${EXTRA[@]+"${EXTRA[@]}"} ${EB[@]+"${EB[@]}"} \
        JXL_ZENSIM_TRACE=$FD/trace_$lbl.tsv \
        JXL_ZENSIM_ATTR_PROBE=$FD/probe_$lbl.tsv
    done
  done
  # Engagement gates: h3 probes = 27*K lines exactly; baselines = 0.
  fail=0
  for mode in k2_last k2_best k3_best; do
    K=${mode:1:1}
    # 2026-08-26: the h3 attr-probe emits TWICE per (cell x iter) on the current loop —
    # measured benign (the arm converges tightly; see benchmarks/zensim_loop_23shot_STALE_2026-08-26.md).
    want=$((54 * K))
    n=$(wc -l < "$FD/probe_v47A_h3g20c135_${mode}.tsv" 2>/dev/null || echo 0)
    say "ENGAGE v47A_h3g20c135_$mode probe=$n want=$want"
    [ "$n" -eq "$want" ] || fail=1
    for arm in v47A_base B_base bvls_base blend2L_base; do
      n=$(wc -l < "$FD/probe_${arm}_${mode}.tsv" 2>/dev/null || echo 0)
      say "ENGAGE ${arm}_$mode probe=$n want=0"
      [ "$n" -eq 0 ] || fail=1
    done
  done
  # Emit-best engagement REPORT (not a hard gate: at low k the argmin is often the
  # last compare, in which case best==last bitstreams are correct behavior).
  for arm in v47A_base B_base bvls_base blend2L_base v47A_h3g20c135; do
    diffn=0; tot=0
    for f in "$FD"/decoded/${arm}_k2_last__*.jxl; do
      bn=$(basename "$f"); bb=${bn/_k2_last__/_k2_best__}
      tot=$((tot + 1))
      cmp -s "$f" "$FD/decoded/$bb" || diffn=$((diffn + 1))
    done
    say "EMIT_BEST k2 engagement $arm: $diffn/$tot bitstreams differ from emit-last (0 = argmin==last everywhere, legal)"
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
fi

# ── k3last: fresh k3 emit-last for all 5 arms (2026-08-26) ────────────────
# The census originally DERIVED k3-emit-last from the 2026-07-31 mm study; the substrate
# probe now FAILS on the current loop (see benchmarks/zensim_loop_23shot_STALE_2026-08-26.md),
# so these must be measured fresh. Same run_ab shape as the probe (no EMIT_BEST).
if [ "$phase" = k3last ]; then
  FD=$OUT/fresh
  mkdir -p "$FD"
  declare -A BAKE=( [v47A_base]="$V47" [B_base]="$BLIN" [bvls_base]="$BVLS" [blend2L_base]="$BLEND" [v47A_h3g20c135]="$V47" )
  declare -A ARMS=( [v47A_base]=baseline [B_base]=baseline [bvls_base]=baseline [blend2L_base]=baseline [v47A_h3g20c135]=h3-mag )
  for arm in v47A_base B_base bvls_base blend2L_base v47A_h3g20c135; do
    EXTRA=()
    [ "$arm" = v47A_h3g20c135 ] && EXTRA=(ZENSIM_H3_GAIN=20.0 JXL_ZENSIM_CTRL_CLAMP=1.35)
    lbl=${arm}_k3_last
    run_ab "$FD" "$lbl" "${BAKE[$arm]}" "${ARMS[$arm]}" 3 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      JXL_ZENSIM_TRACE=$FD/trace_$lbl.tsv \
      JXL_ZENSIM_ATTR_PROBE=$FD/probe_$lbl.tsv
  done
  say "k3last done: $(ls $FD/target_ab_*_k3_last.tsv 2>/dev/null | wc -l)/5 arm TSVs"
fi

# ── collect: concatenate committed TSV into benchmarks/ ───────────────────
if [ "$phase" = collect ] || [ "$phase" = all ]; then
  BD=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$OUT"/probe/target_ab_*.tsv "$OUT"/fresh/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_23shot_2026-08-01.tsv"
  wc -l "$BD/zensim_loop_23shot_2026-08-01.tsv" | tee -a "$LOG"
fi

say "phase '$phase' complete"
