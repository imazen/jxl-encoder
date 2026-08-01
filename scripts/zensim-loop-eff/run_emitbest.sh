#!/bin/bash
# Emit-best A/B runner (2026-07-31) — protocol:
#   benchmarks/zensim_emit_best_2026-07-31.md (frozen pre-registration)
# Phases: gate0 (R0a main-vs-new byte identity, env unset)
#         last  (4 emit-last runs: {base,h3g20c135} x {k6,k12})
#         gate1 (R0b set-but-last-best identity + R0c engagement, single cells)
#         best  (4 emit-best runs)
#         collect (concatenate committed TSVs into benchmarks/)
# Usage: run_emitbest.sh <phase>|all
#
# Build first (from repo root; heavy -> nice'd, own target dir):
#   CARGO_TARGET_DIR=$HOME/tmp/jxlesp-target nice -n19 ionice -c3 \
#     cargo build --release -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel"
# MAIN_BIN = the pre-change binary (built at main before the edit) for R0a.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlesp-target}/release/examples/zensim_diffmap_rd}
MAIN_BIN=${MAIN_BIN:-$HOME/tmp/emit-singlepass/zensim_diffmap_rd_MAIN}
OUT=${EB_OUT:-$HOME/tmp/emit-singlepass/run}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
GB82=${GB82_DIR:-$HOME/work/codec-corpus/gb82-sc}
CID=${CID_DIR:-$HOME/work/codec-corpus/CID22/CID22-512/validation}
COH=${COH_DIR:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
RUN="nice -n19 ionice -c3"
mkdir -p "$OUT/fixtures"
LOG=$OUT/run_emitbest.log
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
CORPUS1=$OUT/corpus1.tsv   # R0a identity gate cell (city)
grep -P '\tcity\t' "$CORPUS" > "$CORPUS1"

phase=${1:-all}
run_ab() { # run_ab <outdir> <label> <arms> <iters> <targets> [env...]
  local od=$1 lbl=$2 arms=$3 it=$4 tg=$5; shift 5
  mkdir -p "$od"
  say "run_ab $lbl arms=$arms iters=$it targets=$tg env: $*"
  env "$@" $RUN "$BIN" --corpus-file "$CORPUS_ACTIVE" --zensim-targets "$tg" \
    --arms "$arms" --bake "$V47" --iters "$it" --label "$lbl" --out-dir "$od" \
    >> "$LOG" 2>&1
}

CORPUS_ACTIVE=$CORPUS

# ── gate0: R0a — env unset, new binary == MAIN (pre-change) binary ───────
if [ "$phase" = gate0 ] || [ "$phase" = all ]; then
  CORPUS_ACTIVE=$CORPUS1
  [ -x "$MAIN_BIN" ] || { say "R0a: MAIN_BIN $MAIN_BIN missing — STOP"; exit 1; }
  mkdir -p "$OUT/gate0"
  say "R0a: MAIN binary encode (env unset)"
  ZDR_SAVE=1 env JXL_SAVE_BITSTREAM=1 $RUN "$MAIN_BIN" --corpus-file "$CORPUS1" \
    --zensim-targets 80 --arms baseline --bake "$V47" --iters 6 \
    --label r0a_main --out-dir "$OUT/gate0" >> "$LOG" 2>&1
  say "R0a: NEW binary encode (env unset)"
  run_ab "$OUT/gate0" r0a_new baseline 6 80 JXL_SAVE_BITSTREAM=1
  a=$(sha256sum "$OUT"/gate0/decoded/r0a_main__city__t80__baseline.jxl | cut -d' ' -f1)
  b=$(sha256sum "$OUT"/gate0/decoded/r0a_new__city__t80__baseline.jxl | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then
    say "R0a IDENTITY PASS ($a)"
  else
    say "R0a IDENTITY FAIL: main=$a new=$b — STOP, fix before measuring"
    exit 1
  fi
  CORPUS_ACTIVE=$CORPUS
fi

# ── last: the 4 emit-last runs (TRACE + probes + bitstreams) ─────────────
if [ "$phase" = last ] || [ "$phase" = all ]; then
  LD=$OUT/last
  mkdir -p "$LD"
  for K in 6 12; do
    run_ab "$LD" v47A_base_k${K}_last baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      JXL_ZENSIM_TRACE=$LD/trace_v47A_base_k${K}_last.tsv \
      JXL_ZENSIM_ATTR_PROBE=$LD/probe_v47A_base_k${K}_last.tsv
    run_ab "$LD" v47A_h3g20c135_k${K}_last h3-mag $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 ZENSIM_H3_GAIN=20 \
      JXL_ZENSIM_TRACE=$LD/trace_v47A_h3g20c135_k${K}_last.tsv \
      JXL_ZENSIM_ATTR_PROBE=$LD/probe_v47A_h3g20c135_k${K}_last.tsv
  done
  fail=0
  for K in 6 12; do
    want=$((27 * K))
    n=$(wc -l < "$LD/probe_v47A_h3g20c135_k${K}_last.tsv" 2>/dev/null || echo 0)
    say "ENGAGE h3 k=$K probe=$n want=$want"
    [ "$n" -eq "$want" ] || fail=1
    n=$(wc -l < "$LD/probe_v47A_base_k${K}_last.tsv" 2>/dev/null || echo 0)
    say "ENGAGE base k=$K probe=$n want=0"
    [ "$n" -eq 0 ] || fail=1
  done
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
fi

# ── gate1: R0b (set-but-last-best identity) + R0c (engagement) ───────────
if [ "$phase" = gate1 ] || [ "$phase" = all ]; then
  G1=$OUT/gate1
  mkdir -p "$G1"
  # Pick cells from the emit-last traces: one argmin==last, one argmin<last.
  python3 "$REPO/scripts/zensim-loop-eff/analyze_emitbest.py" pick \
    --run-dir "$OUT" > "$G1/picks.tsv" || { say "gate1 pick FAIL"; exit 1; }
  say "gate1 picks: $(cat "$G1/picks.tsv" | tr '\n' ' ; ')"
  while IFS=$'\t' read -r kind lbl name t arm k; do
    [ -n "$kind" ] || continue
    grep -P "\t$name\t" "$CORPUS" > "$G1/corpus_$kind.tsv"
    CORPUS_ACTIVE=$G1/corpus_$kind.tsv
    ENVS=(JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_EMIT_BEST=1)
    [ "$arm" = h3-mag ] && ENVS+=(ZENSIM_H3_GAIN=20)
    run_ab "$G1" "g1_$kind" "$arm" "$k" "$t" "${ENVS[@]}"
    ref=$OUT/last/decoded/${lbl}__${name}__t${t}__${arm}.jxl
    new=$G1/decoded/g1_${kind}__${name}__t${t}__${arm}.jxl
    if [ "$kind" = lastbest ]; then
      if cmp -s "$ref" "$new"; then say "R0b IDENTITY PASS ($name t$t $arm k$k: emit-best == emit-last when last is best)"
      else say "R0b IDENTITY FAIL ($name t$t $arm k$k) — STOP"; exit 1; fi
    else
      if cmp -s "$ref" "$new"; then say "R0c ENGAGEMENT FAIL ($name t$t $arm k$k: bitstream unchanged) — STOP"; exit 1
      else say "R0c ENGAGEMENT PASS ($name t$t $arm k$k: bitstream differs)"; fi
    fi
  done < "$G1/picks.tsv"
  CORPUS_ACTIVE=$CORPUS
fi

# ── best: the 4 emit-best runs ───────────────────────────────────────────
if [ "$phase" = best ] || [ "$phase" = all ]; then
  BD=$OUT/best
  mkdir -p "$BD"
  for K in 6 12; do
    run_ab "$BD" v47A_base_k${K}_best baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_EMIT_BEST=1 \
      JXL_ZENSIM_TRACE=$BD/trace_v47A_base_k${K}_best.tsv \
      JXL_ZENSIM_ATTR_PROBE=$BD/probe_v47A_base_k${K}_best.tsv
    run_ab "$BD" v47A_h3g20c135_k${K}_best h3-mag $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_EMIT_BEST=1 \
      ZENSIM_H3_GAIN=20 \
      JXL_ZENSIM_TRACE=$BD/trace_v47A_h3g20c135_k${K}_best.tsv \
      JXL_ZENSIM_ATTR_PROBE=$BD/probe_v47A_h3g20c135_k${K}_best.tsv
  done
fi

# ── collect: concatenate committed TSVs into benchmarks/ ─────────────────
if [ "$phase" = collect ] || [ "$phase" = all ]; then
  BDIR=$REPO/benchmarks
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$OUT"/last/target_ab_*.tsv "$OUT"/best/target_ab_*.tsv; do
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BDIR/zensim_emitbest_cells_2026-07-31.tsv"
  {
    printf 'trace_id\titer\tscore\tqf_mean\tqf_min\tqf_max\titer_ms\n'
    cat "$OUT"/last/trace_*.tsv "$OUT"/best/trace_*.tsv 2>/dev/null
  } > "$BDIR/zensim_emitbest_traces_2026-07-31.tsv"
  wc -l "$BDIR"/zensim_emitbest_{cells,traces}_2026-07-31.tsv | tee -a "$LOG"
fi

say "phase '$phase' complete"
