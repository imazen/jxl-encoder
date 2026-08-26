#!/bin/bash
# BEATS-BUTTER consolidation runner (2026-08-07) — protocol + gates:
#   benchmarks/zensim_loop_beatbutter_2026-08-07.md  (registered before runs)
# Phases: bingate clampsweep collect   (usage: run_beatbutter.sh <phase>|all)
#
# BINGATE: exp100 recipe (h3-mag, candidate bake, CTRL_EXP=1.00) x {k2,k3}
#   x {ZENSIM_ATTR_BIN=1, ZENSIM_ATTR_BIN=8}. G-BB1: bin=1 must reproduce the
#   committed exp100 census (k3 20/27 med 0.564; k2 17/27 med 1.395).
# CLAMPSWEEP: exp100 (bin 8) x CTRL_CLAMP {1.60,2.00,2.50} x k3.
#
# Build first (from repo root; own target dir, nice'd):
#   CARGO_TARGET_DIR=$HOME/tmp/jxlloop-target nice -n19 ionice -c3 \
#     cargo build --release -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlloop-target}/release/examples/zensim_diffmap_rd}
OUT=${BB_OUT:-$HOME/tmp/jxlloop/beatbutter}
CAND=${CAND_BAKE:-/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin}
GB82=${GB82_DIR:-$HOME/work/codec-corpus/gb82-sc}
CID=${CID_DIR:-$HOME/work/codec-corpus/CID22/CID22-512/validation}
COH=${COH_DIR:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
RUN="nice -n19 ionice -c3"
mkdir -p "$OUT/fixtures"
LOG=$OUT/run_beatbutter.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

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
run_ab() { # run_ab <outdir> <label> <arms> <iters> [env...]
  local od=$1 lbl=$2 arms=$3 it=$4; shift 4
  mkdir -p "$od"
  say "run_ab $lbl arms=$arms iters=$it env: $*"
  env "$@" $RUN "$BIN" --corpus-file "$CORPUS" --zensim-targets 70,80,88 \
    --arms "$arms" --bake "$CAND" --iters "$it" --label "$lbl" --out-dir "$od" \
    >> "$LOG" 2>&1
}
COMMON=(JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_EMIT_BEST=1 JXL_ZENSIM_CTRL_EXP=1.00)

if [ "$phase" = bingate ] || [ "$phase" = all ]; then
  D=$OUT/bingate; mkdir -p "$D"
  for k in 3 2; do
    for b in 1 8; do
      run_ab "$D" "exp100_bin${b}_k${k}" h3-mag "$k" "${COMMON[@]}" \
        ZENSIM_ATTR_BIN=$b \
        JXL_ZENSIM_TRACE=$D/trace_exp100_bin${b}_k${k}.tsv
    done
  done
  say "bingate encodes done"
fi

if [ "$phase" = clampsweep ] || [ "$phase" = all ]; then
  D=$OUT/clampsweep; mkdir -p "$D"
  for cl in 1.60 2.00 2.50; do
    run_ab "$D" "exp100_cl${cl}_k3" h3-mag 3 "${COMMON[@]}" \
      ZENSIM_ATTR_BIN=8 JXL_ZENSIM_CTRL_CLAMP=$cl \
      JXL_ZENSIM_TRACE=$D/trace_exp100_cl${cl}_k3.tsv
  done
  say "clampsweep encodes done"
fi

if [ "$phase" = h3ctrl2fresh ]; then
  # 2026-08-26: fresh re-measure of the ADOPTED frontier arm (summary key W10L9_h3ctrl2 =
  # exp100 + CTRL_CLAMP 2.00 + ATTR_BIN 8) on the current secant-on-default substrate —
  # the 08-07 rows are old-substrate (see zensim_loop_23shot_STALE_2026-08-26.md).
  D=$OUT/h3ctrl2fresh; mkdir -p "$D"
  for k in 2 3; do
    run_ab "$D" "W10L9_h3ctrl2_k${k}_best" h3-mag "$k" "${COMMON[@]}" \
      ZENSIM_ATTR_BIN=8 JXL_ZENSIM_CTRL_CLAMP=2.00 \
      JXL_ZENSIM_TRACE=$D/trace_h3ctrl2_k${k}_best.tsv
    run_ab "$D" "W10L9_h3ctrl2_k${k}_last" h3-mag "$k" \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_CTRL_EXP=1.00 \
      ZENSIM_ATTR_BIN=8 JXL_ZENSIM_CTRL_CLAMP=2.00 \
      JXL_ZENSIM_TRACE=$D/trace_h3ctrl2_k${k}_last.tsv
  done
  say "h3ctrl2fresh done: $(ls $D/target_ab_W10L9_h3ctrl2_*.tsv 2>/dev/null | wc -l)/4 TSVs"
fi

say "phase(s) '$phase' complete; collect via analyze_23shot.cells_stats over $OUT/*/*.tsv"
touch "$OUT/PHASE_${phase}.done"
