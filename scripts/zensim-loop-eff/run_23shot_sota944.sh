#!/bin/bash
# SOTA-944 candidate 2/3-shot runner (2026-08-05) — protocol + results:
#   benchmarks/zensim_loop_23shot_sota944_2026-08-05.md
# Adds W10L9_s4003_packed (944-class PRUNED bake) as arm `W10L9_base` to the
# 2026-08-01 panel: {k2 emit-last, k2 emit-best, k3 emit-last, k3 emit-best}
# fresh (k3-last has no mm rows to derive from), controls carried behind the
# same substrate probe as the 2026-08-01 study (which doubles as the
# R0-identity gate for the folded-class integration changes).
# Phases: probe fresh collect h3own h3ownsp gainsweep secant s3gain s3s1
#   (usage: run_23shot_sota944.sh <phase>|all)
# `h3ownsp` (campaign appendix P lever 2, 2026-08-05): G-P5 — h3-mag +
# JXL_ZENSIM_SINGLEPASS=1 (stale-map single-pass) vs the committed fresh
# h3own rows. `gainsweep` (appendix P lever 3): ZENSIM_H3_GAIN ∈ {5,20,40}
# × k3 emit-best on the fresh h3own arm.
# `h3own` (campaign appendix N.4, 2026-08-05): the candidate's OWN-map H3
# magnitude-steering arm through the FUSED folded-944 compare — vs the
# CARRIED W10L9_base rows (same cells, same stats owner). Runs after `probe`
# passes on the current substrate.
#
# Build first (from repo root; heavy -> nice'd, own target dir):
#   CARGO_TARGET_DIR=$HOME/tmp/jxlloop-target nice -n19 ionice -c3 \
#     cargo build --release -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlloop-target}/release/examples/zensim_diffmap_rd}
OUT=${TS_OUT:-$HOME/tmp/jxlloop/run}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
CAND=${CAND_BAKE:-/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin}
GB82=${GB82_DIR:-$HOME/work/codec-corpus/gb82-sc}
CID=${CID_DIR:-$HOME/work/codec-corpus/CID22/CID22-512/validation}
COH=${COH_DIR:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
RUN="nice -n19 ionice -c3"
mkdir -p "$OUT/fixtures"
LOG=$OUT/run_23shot_sota944.log
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

# ── probe: substrate verification (carried controls) + R0-identity gate ───
# Re-runs the mm study's v47A_base_k3 (emit-last) on the CURRENT substrate
# (which includes the folded-class integration changes) and numerically diffs
# achieved_decoded/abs_err/bytes + the 108 trace compares vs the committed
# zensim_mm_cells/traces 2026-07-31 TSVs. 27/27 + 108/108 equal =>
# (i) the 2026-08-01 + mm rows are valid to carry, and (ii) the integration
# changes did NOT alter the 372-class loop.
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
    say "SUBSTRATE PROBE FAIL — carried controls INVALID on this substrate."
    say "STOP: fresh re-runs of all control arms required (and outer cannot be derived)."
    exit 1
  fi
  say "SUBSTRATE PROBE PASS — carried controls valid; 372-class loop unchanged by the integration"
fi

# ── fresh: candidate x {k2_last, k2_best, k3_last, k3_best} = 108 encodes ─
if [ "$phase" = fresh ] || [ "$phase" = all ]; then
  FD=$OUT/fresh
  mkdir -p "$FD"
  for mode in k2_last k2_best k3_last k3_best; do
    K=${mode:1:1}
    EB=()
    [ "${mode#*_}" = best ] && EB=(JXL_ZENSIM_EMIT_BEST=1)
    lbl=W10L9_base_${mode}
    run_ab "$FD" "$lbl" "$CAND" baseline "$K" 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      ${EB[@]+"${EB[@]}"} \
      JXL_ZENSIM_TRACE=$FD/trace_$lbl.tsv \
      JXL_ZENSIM_ATTR_PROBE=$FD/probe_$lbl.tsv
  done
  # Engagement gates: baseline-mode candidate => probe files 0 lines; trace
  # files exactly 27*(K+1) compare rows.
  fail=0
  for mode in k2_last k2_best k3_last k3_best; do
    K=${mode:1:1}
    n=$(wc -l < "$FD/probe_W10L9_base_${mode}.tsv" 2>/dev/null || echo 0)
    say "ENGAGE W10L9_base_$mode probe=$n want=0"
    [ "$n" -eq 0 ] || fail=1
    tn=$(wc -l < "$FD/trace_W10L9_base_${mode}.tsv" 2>/dev/null || echo 0)
    want=$((27 * (K + 1)))
    say "TRACE  W10L9_base_$mode rows=$tn want=$want"
    [ "$tn" -eq "$want" ] || fail=1
  done
  # Emit-best k2 + k3 A/B REPORT (0 diffs = argmin==last everywhere, legal).
  for K in 2 3; do
    diffn=0; tot=0
    for f in "$FD"/decoded/W10L9_base_k${K}_last__*.jxl; do
      [ -f "$f" ] || continue
      bn=$(basename "$f"); bb=${bn/_k${K}_last__/_k${K}_best__}
      tot=$((tot + 1))
      cmp -s "$f" "$FD/decoded/$bb" || diffn=$((diffn + 1))
    done
    say "EMIT_BEST k$K engagement W10L9_base: $diffn/$tot bitstreams differ from emit-last"
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
fi

# ── h3own: candidate + OWN-map H3 magnitude steering through the FUSED
#    folded-944 compare (campaign appendix N.4) x {k2,k3} x {last,best} ────
if [ "$phase" = h3own ]; then
  HD=$OUT/h3own
  mkdir -p "$HD"
  for mode in k2_last k2_best k3_last k3_best; do
    K=${mode:1:1}
    EB=()
    [ "${mode#*_}" = best ] && EB=(JXL_ZENSIM_EMIT_BEST=1)
    lbl=W10L9_h3own_${mode}
    run_ab "$HD" "$lbl" "$CAND" h3-mag "$K" 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      ${EB[@]+"${EB[@]}"} \
      JXL_ZENSIM_TRACE=$HD/trace_$lbl.tsv \
      JXL_ZENSIM_ATTR_PROBE=$HD/probe_$lbl.tsv
  done
  # Engagement gates: h3 steers iterations 1..K => probe files exactly
  # 27*K attr_iter lines; trace files exactly 27*(K+1) compare rows.
  fail=0
  for mode in k2_last k2_best k3_last k3_best; do
    K=${mode:1:1}
    n=$(wc -l < "$HD/probe_W10L9_h3own_${mode}.tsv" 2>/dev/null || echo 0)
    # 2026-08-26 MEASURED on the current loop: these own-map h3 arms emit exactly 27*K
    # probe lines (NO doubling here — the 2x is recipe-dependent: present in the generic
    # v47A_h3g20c135 + exp100 recipes, absent in these phases).
    want=$((27 * K))
    say "ENGAGE W10L9_h3own_$mode probe=$n want=$want"
    [ "$n" -eq "$want" ] || fail=1
    tn=$(wc -l < "$HD/trace_W10L9_h3own_${mode}.tsv" 2>/dev/null || echo 0)
    wantt=$((27 * (K + 1)))
    say "TRACE  W10L9_h3own_$mode rows=$tn want=$wantt"
    [ "$tn" -eq "$wantt" ] || fail=1
  done
  for K in 2 3; do
    diffn=0; tot=0
    for f in "$HD"/decoded/W10L9_h3own_k${K}_last__*.jxl; do
      [ -f "$f" ] || continue
      bn=$(basename "$f"); bb=${bn/_k${K}_last__/_k${K}_best__}
      tot=$((tot + 1))
      cmp -s "$f" "$HD/decoded/$bb" || diffn=$((diffn + 1))
    done
    say "EMIT_BEST k$K engagement W10L9_h3own: $diffn/$tot bitstreams differ from emit-last"
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
  # Committed cells TSV for the h3own study.
  BD=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$HD"/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_h3own_sota944_2026-08-05.tsv"
  wc -l "$BD/zensim_loop_h3own_sota944_2026-08-05.tsv" | tee -a "$LOG"
fi

# ── h3ownsp: appendix P lever 2 gate G-P5 — the same h3own grid with
#    JXL_ZENSIM_SINGLEPASS=1 (stale-map single-pass: first steered
#    iteration fused + map cached, later iterations score-only extraction
#    steering with the cached map). A/B vs the COMMITTED fresh h3own rows.
if [ "$phase" = h3ownsp ]; then
  HD=$OUT/h3ownsp
  mkdir -p "$HD"
  for mode in k2_last k2_best k3_last k3_best; do
    K=${mode:1:1}
    EB=()
    [ "${mode#*_}" = best ] && EB=(JXL_ZENSIM_EMIT_BEST=1)
    lbl=W10L9_h3ownsp_${mode}
    run_ab "$HD" "$lbl" "$CAND" h3-mag "$K" 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_SINGLEPASS=1 \
      ${EB[@]+"${EB[@]}"} \
      JXL_ZENSIM_TRACE=$HD/trace_$lbl.tsv \
      JXL_ZENSIM_ATTR_PROBE=$HD/probe_$lbl.tsv
  done
  # Engagement gates: identical to the h3own phase — the cheap path still
  # steers (and probes) every iteration 1..K.
  fail=0
  for mode in k2_last k2_best k3_last k3_best; do
    K=${mode:1:1}
    n=$(wc -l < "$HD/probe_W10L9_h3ownsp_${mode}.tsv" 2>/dev/null || echo 0)
    # 2026-08-26 MEASURED on the current loop: these own-map h3 arms emit exactly 27*K
    # probe lines (NO doubling here — the 2x is recipe-dependent: present in the generic
    # v47A_h3g20c135 + exp100 recipes, absent in these phases).
    want=$((27 * K))
    say "ENGAGE W10L9_h3ownsp_$mode probe=$n want=$want"
    [ "$n" -eq "$want" ] || fail=1
    tn=$(wc -l < "$HD/trace_W10L9_h3ownsp_${mode}.tsv" 2>/dev/null || echo 0)
    wantt=$((27 * (K + 1)))
    say "TRACE  W10L9_h3ownsp_$mode rows=$tn want=$wantt"
    [ "$tn" -eq "$wantt" ] || fail=1
  done
  for K in 2 3; do
    diffn=0; tot=0
    for f in "$HD"/decoded/W10L9_h3ownsp_k${K}_last__*.jxl; do
      [ -f "$f" ] || continue
      bn=$(basename "$f"); bb=${bn/_k${K}_last__/_k${K}_best__}
      tot=$((tot + 1))
      cmp -s "$f" "$HD/decoded/$bb" || diffn=$((diffn + 1))
    done
    say "EMIT_BEST k$K engagement W10L9_h3ownsp: $diffn/$tot bitstreams differ from emit-last"
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
  BD=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$HD"/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_h3ownsp_sota944_2026-08-05.tsv"
  wc -l "$BD/zensim_loop_h3ownsp_sota944_2026-08-05.tsv" | tee -a "$LOG"
fi

# ── gainsweep: appendix P lever 3 — ZENSIM_H3_GAIN ∈ {5,20,40} × k3
#    emit-best on the FRESH fused h3own arm (gain 10 = the committed h3own
#    k3_best rows; registered as a sweep, no default change without the
#    curve).
if [ "$phase" = gainsweep ]; then
  GD=$OUT/gainsweep
  mkdir -p "$GD"
  for gain in 5 20 40; do
    lbl=W10L9_h3own_g${gain}_k3_best
    run_ab "$GD" "$lbl" "$CAND" h3-mag 3 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_EMIT_BEST=1 \
      ZENSIM_H3_GAIN=$gain \
      JXL_ZENSIM_TRACE=$GD/trace_$lbl.tsv \
      JXL_ZENSIM_ATTR_PROBE=$GD/probe_$lbl.tsv
  done
  fail=0
  for gain in 5 20 40; do
    n=$(wc -l < "$GD/probe_W10L9_h3own_g${gain}_k3_best.tsv" 2>/dev/null || echo 0)
    say "ENGAGE W10L9_h3own_g${gain}_k3_best probe=$n want=81"
    [ "$n" -eq 81 ] || fail=1
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
  BD=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$GD"/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_h3gain_sota944_2026-08-05.tsv"
  wc -l "$BD/zensim_loop_h3gain_sota944_2026-08-05.tsv" | tee -a "$LOG"
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
  } > "$BD/zensim_loop_23shot_sota944_2026-08-05.tsv"
  wc -l "$BD/zensim_loop_23shot_sota944_2026-08-05.tsv" | tee -a "$LOG"
fi

say "phase '$phase' complete"

# ── secant: the registered DECODED-JUDGED A/B for the guarded diffmap secant
#    (benchmarks/zensim_secant_2026-08-25.md "Next" #1). Frontier C bake
#    (Profile C, c_sdr_mlp944_corrmix) + h3-mag, K in {2,3} x {last,best} x
#    JXL_ZENSIM_SECANT in {0,1}. The 08-25 tables were INTERNAL-score; this
#    phase judges on the instrument's achieved_decoded/abs_err columns.
if [ "$phase" = secant ]; then
  SD=$OUT/secant
  # The harness APPENDS to JXL_ZENSIM_TRACE/ATTR_PROBE files, so a rerun on a
  # populated dir doubles every gate count (measured 2026-08-27: probe 108
  # want 54). Preserve any prior run, then start clean.
  [ -d "$SD" ] && mv "$SD" "$SD.bak.$(date -u +%s)"
  mkdir -p "$SD"
  CBAKE=${CBAKE_BAKE:-$HOME/work/zen/zensim/zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin}
  [ -f "$CBAKE" ] || { say "STOP: C bake missing at $CBAKE"; exit 1; }
  for sec in 0 1; do
    for mode in k2_last k2_best k3_last k3_best; do
      K=${mode:1:1}
      EB=()
      [ "${mode#*_}" = best ] && EB=(JXL_ZENSIM_EMIT_BEST=1)
      lbl=C944_sec${sec}_${mode}
      run_ab "$SD" "$lbl" "$CBAKE" h3-mag "$K" 70,80,88 \
        JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_SECANT=$sec \
        ${EB[@]+"${EB[@]}"} \
        JXL_ZENSIM_TRACE=$SD/trace_$lbl.tsv \
        JXL_ZENSIM_ATTR_PROBE=$SD/probe_$lbl.tsv
    done
  done
  # Engagement gates: h3-mag steers 1..K => probe exactly 27*K lines; trace
  # exactly 27*(K+1) compare rows — for EVERY arm.
  fail=0
  for sec in 0 1; do
    for mode in k2_last k2_best k3_last k3_best; do
      K=${mode:1:1}
      n=$(wc -l < "$SD/probe_C944_sec${sec}_${mode}.tsv" 2>/dev/null || echo 0)
      want=$((27 * K))
      say "ENGAGE C944_sec${sec}_$mode probe=$n want=$want"
      [ "$n" -eq "$want" ] || fail=1
      tn=$(wc -l < "$SD/trace_C944_sec${sec}_${mode}.tsv" 2>/dev/null || echo 0)
      wantt=$((27 * (K + 1)))
      say "TRACE  C944_sec${sec}_$mode rows=$tn want=$wantt"
      [ "$tn" -eq "$wantt" ] || fail=1
    done
  done
  # Secant-engagement: sec1 must CHANGE bitstreams vs sec0 somewhere at each K
  # (the 08-25 smoke's divergence proof, now as a gate — a silent fall-through
  # to the power law would make the whole A/B a null comparison).
  for K in 2 3; do
    diffn=0; tot=0
    for f in "$SD"/decoded/C944_sec0_k${K}_last__*.jxl; do
      [ -f "$f" ] || continue
      bn=$(basename "$f"); bb=${bn/_sec0_/_sec1_}
      tot=$((tot + 1))
      cmp -s "$f" "$SD/decoded/$bb" || diffn=$((diffn + 1))
    done
    say "SECANT-ENGAGE k$K: $diffn/$tot bitstreams differ sec0-vs-sec1"
    [ "$diffn" -ge 1 ] || fail=1
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
  # Committed cells TSV.
  BD=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$SD"/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_secant_decoded_2026-08-26.tsv"
  wc -l "$BD/zensim_loop_secant_decoded_2026-08-26.tsv" | tee -a "$LOG"
  # Decoded verdict (derived from the instrument's OWN achieved_decoded/abs_err
  # columns — census within +-2.0 + median |err| + total bytes per arm).
  python3 "$REPO/scripts/zensim-loop-eff/verdict_23shot_cells.py" "$BD/zensim_loop_secant_decoded_2026-08-26.tsv" | tee "$SD/verdict.txt" | tee -a "$LOG"
fi

# ── s3gain: plan §5 arm S3 (per-tile secant gain, ZENSIM_H3_GAIN_MODE=
#    tile-secant) vs fixed gain, SAME substrate (fresh controls — the secant
#    phase's sec0 rows are a different binary). Global secant OFF in both
#    arms to isolate the gain axis. K in {2,3}, emit-best (the S3 endpoint
#    reads census + bytes; per-cell rows feed the rate-matched analysis).
if [ "$phase" = s3gain ]; then
  SD=$OUT/s3gain
  # The harness APPENDS to JXL_ZENSIM_TRACE/ATTR_PROBE files, so a rerun on a
  # populated dir doubles every gate count (measured 2026-08-27: probe 108
  # want 54). Preserve any prior run, then start clean.
  [ -d "$SD" ] && mv "$SD" "$SD.bak.$(date -u +%s)"
  mkdir -p "$SD"
  CBAKE=${CBAKE_BAKE:-$HOME/work/zen/zensim/zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin}
  [ -f "$CBAKE" ] || { say "STOP: C bake missing at $CBAKE"; exit 1; }
  for gm in fixed tilesec; do
    GME=()
    [ "$gm" = tilesec ] && GME=(ZENSIM_H3_GAIN_MODE=tile-secant)
    for K in 2 3; do
      lbl=C944_${gm}_k${K}_best
      run_ab "$SD" "$lbl" "$CBAKE" h3-mag "$K" 70,80,88 \
        JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_SECANT=0 \
        JXL_ZENSIM_EMIT_BEST=1 \
        ${GME[@]+"${GME[@]}"} \
        JXL_ZENSIM_TRACE=$SD/trace_$lbl.tsv \
        JXL_ZENSIM_ATTR_PROBE=$SD/probe_$lbl.tsv
    done
  done
  fail=0
  for gm in fixed tilesec; do
    for K in 2 3; do
      n=$(wc -l < "$SD/probe_C944_${gm}_k${K}_best.tsv" 2>/dev/null || echo 0)
      want=$((27 * K))
      say "ENGAGE C944_${gm}_k${K}_best probe=$n want=$want"
      [ "$n" -eq "$want" ] || fail=1
      tn=$(wc -l < "$SD/trace_C944_${gm}_k${K}_best.tsv" 2>/dev/null || echo 0)
      wantt=$((27 * (K + 1)))
      say "TRACE  C944_${gm}_k${K}_best rows=$tn want=$wantt"
      [ "$tn" -eq "$wantt" ] || fail=1
    done
  done
  # S3-engagement, structural (measured 2026-08-26): the per-tile gain first
  # DIFFERS from fixed at the 2nd steered iterate, and a redistribution only
  # reaches a bitstream via the NEXT encode — so at K=2 (steered iters 1,2;
  # no encode after iter 2) tile-secant CANNOT change any emitted bitstream:
  # k2 must be IDENTICAL (0 diffs — a free structural identity control), and
  # k3 must diverge somewhere (first run: k2 0/27, k3 25/27).
  for K in 2 3; do
    diffn=0; tot=0
    for f in "$SD"/decoded/C944_fixed_k${K}_best__*.jxl; do
      [ -f "$f" ] || continue
      bn=$(basename "$f"); bb=${bn/_fixed_/_tilesec_}
      tot=$((tot + 1))
      cmp -s "$f" "$SD/decoded/$bb" || diffn=$((diffn + 1))
    done
    say "S3-ENGAGE k$K: $diffn/$tot bitstreams differ fixed-vs-tilesec"
    if [ "$K" -eq 2 ]; then
      [ "$diffn" -eq 0 ] || { say "k2 must be identical by construction"; fail=1; }
    else
      [ "$diffn" -ge 1 ] || fail=1
    fi
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
  BD=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$SD"/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_s3gain_decoded_2026-08-26.tsv"
  wc -l "$BD/zensim_loop_s3gain_decoded_2026-08-26.tsv" | tee -a "$LOG"
  python3 "$REPO/scripts/zensim-loop-eff/verdict_23shot_cells.py" "$BD/zensim_loop_s3gain_decoded_2026-08-26.tsv" | tee "$SD/verdict.txt" | tee -a "$LOG"
fi


# ── s3s1: the registered S3xS1 COMPOSITION (global guarded secant + per-tile
#    secant gain together), k3 emit-best only (S3 is structurally inert at k2;
#    the k2 composition is S1 alone). Controls: the s3gain phase's fixed/
#    tilesec k3 rows — SAME harness binary (this phase refuses to run if the
#    s3gain TSV is absent, so the comparison is never cross-substrate).
if [ "$phase" = s3s1 ]; then
  SD=$OUT/s3s1
  [ -d "$SD" ] && mv "$SD" "$SD.bak.$(date -u +%s)"
  mkdir -p "$SD"
  BD=$REPO/benchmarks
  CTRL_TSV=$BD/zensim_loop_s3gain_decoded_2026-08-26.tsv
  [ -f "$CTRL_TSV" ] || { say "STOP: run the s3gain phase first (same-substrate controls)"; exit 1; }
  CBAKE=${CBAKE_BAKE:-$HOME/work/zen/zensim/zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin}
  [ -f "$CBAKE" ] || { say "STOP: C bake missing at $CBAKE"; exit 1; }
  for gm in fixed tilesec; do
    GME=()
    [ "$gm" = tilesec ] && GME=(ZENSIM_H3_GAIN_MODE=tile-secant)
    lbl=C944_sec1${gm}_k3_best
    run_ab "$SD" "$lbl" "$CBAKE" h3-mag 3 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_SECANT=1 \
      JXL_ZENSIM_EMIT_BEST=1 \
      ${GME[@]+"${GME[@]}"} \
      JXL_ZENSIM_TRACE=$SD/trace_$lbl.tsv \
      JXL_ZENSIM_ATTR_PROBE=$SD/probe_$lbl.tsv
  done
  fail=0
  for gm in fixed tilesec; do
    n=$(wc -l < "$SD/probe_C944_sec1${gm}_k3_best.tsv" 2>/dev/null || echo 0)
    say "ENGAGE C944_sec1${gm}_k3_best probe=$n want=81"
    [ "$n" -eq 81 ] || fail=1
    tn=$(wc -l < "$SD/trace_C944_sec1${gm}_k3_best.tsv" 2>/dev/null || echo 0)
    say "TRACE  C944_sec1${gm}_k3_best rows=$tn want=108"
    [ "$tn" -eq 108 ] || fail=1
  done
  # Composition engagement: the tile gain must still change bitstreams under
  # the global secant at k3.
  diffn=0; tot=0
  for f in "$SD"/decoded/C944_sec1fixed_k3_best__*.jxl; do
    [ -f "$f" ] || continue
    bn=$(basename "$f"); bb=${bn/_sec1fixed_/_sec1tilesec_}
    tot=$((tot + 1))
    cmp -s "$f" "$SD/decoded/$bb" || diffn=$((diffn + 1))
  done
  say "S3S1-ENGAGE k3: $diffn/$tot bitstreams differ sec1fixed-vs-sec1tilesec"
  [ "$diffn" -ge 1 ] || fail=1
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$SD"/target_ab_*.tsv; do
      [ -f "$f" ] || continue
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_loop_s3s1_decoded_2026-08-27.tsv"
  wc -l "$BD/zensim_loop_s3s1_decoded_2026-08-27.tsv" | tee -a "$LOG"
  python3 "$REPO/scripts/zensim-loop-eff/verdict_23shot_cells.py" "$BD/zensim_loop_s3s1_decoded_2026-08-27.tsv" "$CTRL_TSV" | tee "$SD/verdict.txt" | tee -a "$LOG"
fi
