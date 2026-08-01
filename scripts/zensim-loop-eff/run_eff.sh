#!/bin/bash
# Efficiency-study runner (2026-07-31) — protocol:
#   benchmarks/zensim_diffmap_efficiency_2026-07-31.md  (frozen pre-registration)
# Phases: r0 (identity gate) r1 (main matrix) r2 (tolerance) r3 (extended)
#         r4 (budget caps) r5 (bytes targeting). Usage: run_eff.sh <phase>|all
#
# Build first (from repo root; heavy → nice'd, own target dir):
#   CARGO_TARGET_DIR=$HOME/tmp/jxleff-target nice -n19 ionice -c3 \
#     cargo build --release -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxleff-target}/release/examples/zensim_diffmap_rd}
OUT=${EFF_OUT:-$HOME/tmp/diffmap-eff}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
BLIN=${B_BAKE:-$HOME/work/zen/zensim/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin}
GB82=${GB82_DIR:-$HOME/work/codec-corpus/gb82-sc}
CID=${CID_DIR:-$HOME/work/codec-corpus/CID22/CID22-512/validation}
COH=${COH_DIR:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
RUN="nice -n19 ionice -c3"
mkdir -p "$OUT/fixtures"
LOG=$OUT/run_eff.log
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
CORPUS3=$OUT/corpus3.tsv   # E6 subsample — FROZEN in the registration
grep -E $'\t(city|cid1025469|sc_wiki)\t' "$CORPUS" > "$CORPUS3"
CORPUS1=$OUT/corpus1.tsv   # R0 identity gate cell
grep -P '\tcity\t' "$CORPUS" > "$CORPUS1"

phase=${1:-all}
run_ab() { # run_ab <outdir> <label> <bake> <arms> <iters> <targets> [env...]
  local od=$1 lbl=$2 bk=$3 arms=$4 it=$5 tg=$6; shift 6
  mkdir -p "$od"
  say "run_ab $lbl arms=$arms iters=$it targets=$tg env: $*"
  env "$@" $RUN "$BIN" --corpus-file "$CORPUS_ACTIVE" --zensim-targets "$tg" \
    --arms "$arms" --bake "$bk" --iters "$it" --label "$lbl" --out-dir "$od" \
    >> "$LOG" 2>&1
}

CORPUS_ACTIVE=$CORPUS

# ── R0: default-unchanged + determinism identity gate ────────────────────
if [ "$phase" = r0 ] || [ "$phase" = all ]; then
  CORPUS_ACTIVE=$CORPUS1
  run_ab "$OUT/r0" r0a "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1
  run_ab "$OUT/r0" r0b "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_TRACE=$OUT/r0/trace_r0b.tsv
  run_ab "$OUT/r0" r0c "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_QF_GLOBAL_SCALE=1.0
  a=$(sha256sum "$OUT"/r0/decoded/r0a__city__t80__baseline.jxl | cut -d' ' -f1)
  b=$(sha256sum "$OUT"/r0/decoded/r0b__city__t80__baseline.jxl | cut -d' ' -f1)
  c=$(sha256sum "$OUT"/r0/decoded/r0c__city__t80__baseline.jxl | cut -d' ' -f1)
  if [ "$a" = "$b" ] && [ "$a" = "$c" ]; then
    say "R0 IDENTITY GATE PASS ($a)"
  else
    say "R0 IDENTITY GATE FAIL: a=$a b=$b c=$c — STOP, fix before measuring"
    exit 1
  fi
  CORPUS_ACTIVE=$CORPUS
fi

# ── R1: main matrix, no early stop, iters=6, TRACE + engagement probes ───
if [ "$phase" = r1 ] || [ "$phase" = all ]; then
  run_ab "$OUT/r1" v47A_base_r1 "$V47" baseline 6 70,80,88 \
    JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_TRACE=$OUT/r1/trace_v47A_base.tsv \
    JXL_ZENSIM_ATTR_PROBE=$OUT/r1/probe_v47A_base.tsv
  run_ab "$OUT/r1" v47A_h3_r1 "$V47" h3-mag 6 70,80,88 \
    JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_TRACE=$OUT/r1/trace_v47A_h3.tsv \
    JXL_ZENSIM_ATTR_PROBE=$OUT/r1/probe_v47A_h3.tsv
  run_ab "$OUT/r1" B_base_r1 "$BLIN" baseline 6 70,80,88 \
    JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_TRACE=$OUT/r1/trace_B_base.tsv \
    JXL_ZENSIM_ATTR_PROBE=$OUT/r1/probe_B_base.tsv
  pb=$(wc -l < "$OUT/r1/probe_v47A_base.tsv" 2>/dev/null || echo 0)
  ph=$(wc -l < "$OUT/r1/probe_v47A_h3.tsv" 2>/dev/null || echo 0)
  pB=$(wc -l < "$OUT/r1/probe_B_base.tsv" 2>/dev/null || echo 0)
  say "R1 ENGAGEMENT: probe lines base=$pb h3=$ph B=$pB (h3 must be >0; both baselines must be 0)"
  if [ "$ph" -eq 0 ] || [ "$pb" -ne 0 ] || [ "$pB" -ne 0 ]; then
    say "R1 ENGAGEMENT FAIL — arm did not engage as registered; STOP"
    exit 1
  fi
fi

# ── R2: tolerance runs (E3) ──────────────────────────────────────────────
if [ "$phase" = r2 ] || [ "$phase" = all ]; then
  for tol in 0.25 0.5 1.0 2.0; do
    tt=${tol/./}
    run_ab "$OUT/r2_$tt" v47A_base_tol$tt "$V47" baseline 6 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=$tol JXL_ZENSIM_TRACE=$OUT/r2_$tt/trace_v47A_base.tsv
    run_ab "$OUT/r2_$tt" v47A_h3_tol$tt "$V47" h3-mag 6 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=$tol JXL_ZENSIM_TRACE=$OUT/r2_$tt/trace_v47A_h3.tsv
    run_ab "$OUT/r2_$tt" B_base_tol$tt "$BLIN" baseline 6 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=$tol JXL_ZENSIM_TRACE=$OUT/r2_$tt/trace_B_base.tsv
  done
fi

# ── R3: extended budget iters=12 (E5; also E6 k=12) ──────────────────────
if [ "$phase" = r3 ] || [ "$phase" = all ]; then
  run_ab "$OUT/r3" v47A_base_x12 "$V47" baseline 12 70,80,88 \
    JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_TRACE=$OUT/r3/trace_v47A_base_x12.tsv
  run_ab "$OUT/r3" v47A_h3_x12 "$V47" h3-mag 12 70,80,88 \
    JXL_ZENSIM_TARGET_TOL=-1 JXL_ZENSIM_TRACE=$OUT/r3/trace_v47A_h3_x12.tsv
fi

# ── R4: budget-capped k ∈ {1,2,4,8} on the frozen 3-ref subsample (E6) ───
if [ "$phase" = r4 ] || [ "$phase" = all ]; then
  CORPUS_ACTIVE=$CORPUS3
  for k in 1 2 4 8; do
    run_ab "$OUT/r4" v47A_base_k$k "$V47" baseline "$k" 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1
    run_ab "$OUT/r4" v47A_h3_k$k "$V47" h3-mag "$k" 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1
  done
  CORPUS_ACTIVE=$CORPUS
fi

# ── R5: bytes-target outer loop (E7) — targets = R1 v47A baseline bytes ──
if [ "$phase" = r5 ] || [ "$phase" = all ]; then
  BT=$OUT/r5/bytes_targets.tsv
  mkdir -p "$OUT/r5"
  awk -F'\t' 'NR==FNR { p[$2]=$1; next } FNR>1 { print p[$1] "\t" $1 "\t" $2 "\t" $3 "\t" $11 }' \
    "$CORPUS" "$OUT/r1/target_ab_v47A_base_r1.tsv" > "$BT"
  say "R5 bytes targets: $(wc -l < "$BT") rows"
  mkdir -p "$OUT/r5"
  say "run_bytes_target v47A baseline iters=6 outer=8"
  $RUN "$BIN" --bytes-targets-file "$BT" --bake "$V47" --iters 6 \
    --label v47A_bytes --out-dir "$OUT/r5" >> "$LOG" 2>&1
fi

say "phase '$phase' complete"
