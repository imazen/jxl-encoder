#!/bin/bash
# Metric-matrix study runner (2026-07-31) — protocol:
#   benchmarks/zensim_loop_metric_matrix_2026-07-31.md  (frozen pre-registration)
# Phases: gate0 (R0 identity for JXL_ZENSIM_CTRL_CLAMP) inner (10 arms × k3/k6)
#         outer (zensimA + ssim2 decoded-judged controllers) xmetric (batch
#         ssim2 over inner emissions) collect (concatenate committed TSVs).
# Usage: run_mm.sh <phase>|all
#
# Build first (from repo root; heavy → nice'd, own target dir):
#   CARGO_TARGET_DIR=$HOME/tmp/jxlmm-target nice -n19 ionice -c3 \
#     cargo build --release -p jxl-encoder --example zensim_diffmap_rd \
#     --features "__expert butteraugli-loop zensim-loop cvvdp-loop-cpu ssim2-loop parallel"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BIN=${ZDR_BIN:-${CARGO_TARGET_DIR:-$HOME/tmp/jxlmm-target}/release/examples/zensim_diffmap_rd}
OUT=${MM_OUT:-$HOME/tmp/metric-matrix}
V47=${V47_BAKE:-$HOME/work/zen/zensim/zensim/weights/v47_strict_qat_native_2026-05-27.bin}
BLIN=${B_BAKE:-$HOME/work/zen/zensim/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin}
BVLS=${BVLS_BAKE:-/mnt/v/output/zensim/bakes/v02_bvls_NO_shaping_2026-05-28.bin}
BLEND=${BLEND_BAKE:-/mnt/v/output/zensim/reports/b_negatives/mlp_2L_diverse_H128_2026-07-15.bin}
SSIM2_BIN=${SSIM2_BIN:-$HOME/work/zen/zenmetrics/target/release/zenmetrics}
GB82=${GB82_DIR:-$HOME/work/codec-corpus/gb82-sc}
CID=${CID_DIR:-$HOME/work/codec-corpus/CID22/CID22-512/validation}
COH=${COH_DIR:-/mnt/v/output/zensim/diffmap-coherence-2026-07-18}
RUN="nice -n19 ionice -c3"
mkdir -p "$OUT/fixtures"
LOG=$OUT/run_mm.log
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

# ── gate0: default-unchanged byte-identity for the new CTRL_CLAMP knob ───
if [ "$phase" = gate0 ] || [ "$phase" = all ]; then
  CORPUS_ACTIVE=$CORPUS1
  run_ab "$OUT/gate0" g0a "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1
  run_ab "$OUT/gate0" g0b "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_TRACE=$OUT/gate0/trace_g0b.tsv
  run_ab "$OUT/gate0" g0c "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_CTRL_CLAMP=1.35
  run_ab "$OUT/gate0" g0d "$V47" baseline 6 80 JXL_SAVE_BITSTREAM=1 \
    JXL_ZENSIM_QF_GLOBAL_SCALE=1.0
  a=$(sha256sum "$OUT"/gate0/decoded/g0a__city__t80__baseline.jxl | cut -d' ' -f1)
  b=$(sha256sum "$OUT"/gate0/decoded/g0b__city__t80__baseline.jxl | cut -d' ' -f1)
  c=$(sha256sum "$OUT"/gate0/decoded/g0c__city__t80__baseline.jxl | cut -d' ' -f1)
  d=$(sha256sum "$OUT"/gate0/decoded/g0d__city__t80__baseline.jxl | cut -d' ' -f1)
  if [ "$a" = "$b" ] && [ "$a" = "$c" ] && [ "$a" = "$d" ]; then
    say "R0 IDENTITY GATE PASS ($a)"
  else
    say "R0 IDENTITY GATE FAIL: a=$a b=$b c=$c d=$d — STOP, fix before measuring"
    exit 1
  fi
  CORPUS_ACTIVE=$CORPUS
fi

# ── inner: 10 arms × k ∈ {3, 6}, tol = −1, TRACE + probes + bitstreams ───
if [ "$phase" = inner ] || [ "$phase" = all ]; then
  ID=$OUT/inner
  mkdir -p "$ID"
  for K in 3 6; do
    run_ab "$ID" v47A_base_k$K "$V47" baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      JXL_ZENSIM_TRACE=$ID/trace_v47A_base_k$K.tsv \
      JXL_ZENSIM_ATTR_PROBE=$ID/probe_v47A_base_k$K.tsv
    run_ab "$ID" v47A_basec160_k$K "$V47" baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 JXL_ZENSIM_CTRL_CLAMP=1.6 \
      JXL_ZENSIM_TRACE=$ID/trace_v47A_basec160_k$K.tsv \
      JXL_ZENSIM_ATTR_PROBE=$ID/probe_v47A_basec160_k$K.tsv
    run_ab "$ID" B_base_k$K "$BLIN" baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      JXL_ZENSIM_TRACE=$ID/trace_B_base_k$K.tsv \
      JXL_ZENSIM_ATTR_PROBE=$ID/probe_B_base_k$K.tsv
    run_ab "$ID" latest_base_k$K "profile:latest" baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      JXL_ZENSIM_TRACE=$ID/trace_latest_base_k$K.tsv \
      JXL_ZENSIM_ATTR_PROBE=$ID/probe_latest_base_k$K.tsv
    run_ab "$ID" bvls_base_k$K "$BVLS" baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      JXL_ZENSIM_TRACE=$ID/trace_bvls_base_k$K.tsv \
      JXL_ZENSIM_ATTR_PROBE=$ID/probe_bvls_base_k$K.tsv
    run_ab "$ID" blend2L_base_k$K "$BLEND" baseline $K 70,80,88 \
      JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
      JXL_ZENSIM_TRACE=$ID/trace_blend2L_base_k$K.tsv \
      JXL_ZENSIM_ATTR_PROBE=$ID/probe_blend2L_base_k$K.tsv
    for GC in "10.0 1.35" "10.0 1.6" "20.0 1.35" "20.0 1.6"; do
      set -- $GC
      gl=${1%%.*}; cl=${2/./}
      lbl=v47A_h3g${gl}c${cl}_k$K
      run_ab "$ID" "$lbl" "$V47" h3-mag $K 70,80,88 \
        JXL_ZENSIM_TARGET_TOL=-1 JXL_SAVE_BITSTREAM=1 \
        ZENSIM_H3_GAIN=$1 JXL_ZENSIM_CTRL_CLAMP=$2 \
        JXL_ZENSIM_TRACE=$ID/trace_$lbl.tsv \
        JXL_ZENSIM_ATTR_PROBE=$ID/probe_$lbl.tsv
    done
  done
  # Engagement gates: h3 probes = 27*K lines exactly; baselines = 0.
  fail=0
  for K in 3 6; do
    want=$((27 * K))
    for lbl in v47A_h3g10c135 v47A_h3g10c16 v47A_h3g20c135 v47A_h3g20c16; do
      n=$(wc -l < "$ID/probe_${lbl}_k$K.tsv" 2>/dev/null || echo 0)
      say "ENGAGE $lbl k=$K probe=$n want=$want"
      [ "$n" -eq "$want" ] || fail=1
    done
    for lbl in v47A_base v47A_basec160 B_base latest_base bvls_base blend2L_base; do
      n=$(wc -l < "$ID/probe_${lbl}_k$K.tsv" 2>/dev/null || echo 0)
      say "ENGAGE $lbl k=$K probe=$n want=0"
      [ "$n" -eq 0 ] || fail=1
    done
  done
  # Clamp engagement: basec160 must diverge from base on >=1 cell (bytes col).
  if cmp -s <(cut -f11 "$ID/target_ab_v47A_base_k3.tsv") \
            <(cut -f11 "$ID/target_ab_v47A_basec160_k3.tsv"); then
    say "ENGAGE FAIL: CTRL_CLAMP=1.6 produced identical bytes to 1.35 on all 27 cells"
    fail=1
  else
    say "ENGAGE clamp1.6 diverges from 1.35 (ok)"
  fi
  # Mount-equivalence control: latest vs shippedB per-cell bitstream shas.
  eq=0; tot=0
  for f in "$ID"/decoded/B_base_k3__*.jxl; do
    bn=$(basename "$f"); ln=${bn/B_base_k3/latest_base_k3}
    tot=$((tot + 1))
    cmp -s "$f" "$ID/decoded/$ln" && eq=$((eq + 1))
  done
  say "MOUNT-EQUIV latest==shippedB bitstreams: $eq/$tot identical (expected all)"
  [ "$fail" -eq 0 ] || { say "ENGAGEMENT GATE FAIL — STOP"; exit 1; }
fi

# ── outer: decoded-judged controllers, 4 encodes/cell, inner iters=1 ─────
if [ "$phase" = outer ] || [ "$phase" = all ]; then
  OD=$OUT/outer
  mkdir -p "$OD"
  say "outer zensimA (judge-driven) 27 cells x 4 encodes"
  env JXL_ZENSIM_TRACE=$OD/trace_outer_zensimA.tsv $RUN "$BIN" \
    --corpus-file "$CORPUS" --zensim-targets 70,80,88 \
    --score-targets-outer zensim --bake "$V47" --iters 1 \
    --ssim2-bin "$SSIM2_BIN" --label outer_zensimA --out-dir "$OD" \
    >> "$LOG" 2>&1
  say "outer ssim2 (zenmetrics-driven) 27 cells x 4 encodes"
  env JXL_ZENSIM_TRACE=$OD/trace_outer_ssim2.tsv $RUN "$BIN" \
    --corpus-file "$CORPUS" --zensim-targets 70,80,88 \
    --score-targets-outer ssim2 --bake "$V47" --iters 1 \
    --ssim2-bin "$SSIM2_BIN" --label outer_ssim2 --out-dir "$OD" \
    >> "$LOG" 2>&1
  n1=$(grep -c nan "$OD/score_outer_outer_ssim2.tsv" || true)
  say "outer done (ssim2 rows containing 'nan': $n1 — recorded nulls if any)"
fi

# ── xmetric: batch ssim2 over ALL inner emissions (k3 + k6) ──────────────
if [ "$phase" = xmetric ] || [ "$phase" = all ]; then
  ID=$OUT/inner
  P=$OUT/xmetric_pairs.tsv
  printf 'ref_path\tdist_path\n' > "$P"
  for f in "$ID"/decoded/*.png; do
    bn=$(basename "$f" .png)
    name=$(echo "$bn" | awk -F'__' '{print $2}')
    printf '%s\t%s\n' "$ID/ref/$name.png" "$f" >> "$P"
  done
  say "xmetric: $(($(wc -l < "$P") - 1)) pairs -> zenmetrics ssim2"
  $RUN "$SSIM2_BIN" batch --metric ssim2 --pairs "$P" \
    --output "$OUT/xmetric_ssim2_raw.tsv" >> "$LOG" 2>&1
  say "xmetric done"
fi

# ── collect: concatenate committed TSVs into benchmarks/ ─────────────────
if [ "$phase" = collect ] || [ "$phase" = all ]; then
  ID=$OUT/inner
  BD=$REPO/benchmarks
  # cells: run label + the target_ab schema
  {
    printf 'run\timage\tclass\ttarget\tarm\tbake\tseed_d\tachieved_inloop\titers_used\tachieved_decoded\tabs_err\tbytes\tencode_ms\tloop_ms\tms_per_compare\n'
    for f in "$ID"/target_ab_*.tsv; do
      run=$(basename "$f" .tsv); run=${run#target_ab_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_mm_cells_2026-07-31.tsv"
  # traces: all inner + outer trace files (trace_id embeds run|name|class|t|arm)
  {
    printf 'trace_id\titer\tscore\tqf_mean\tqf_min\tqf_max\titer_ms\n'
    cat "$ID"/trace_*.tsv "$OUT"/outer/trace_*.tsv 2>/dev/null
  } > "$BD/zensim_mm_traces_2026-07-31.tsv"
  # outer: run label + the score_outer schema
  {
    printf 'run\timage\tclass\ttarget\tmetric\touter_iter\tqf_scale\tbytes\tjudged\tzensimA\tssim2\tencode_ms\tjudge_ms\tssim2_ms\n'
    for f in "$OUT"/outer/score_outer_*.tsv; do
      run=$(basename "$f" .tsv); run=${run#score_outer_}
      awk -F'\t' -v r="$run" 'NR>1 { print r "\t" $0 }' "$f"
    done
  } > "$BD/zensim_mm_outer_2026-07-31.tsv"
  # xmetric: parse run/image/target/arm out of the decoded filename
  {
    printf 'run\timage\ttarget\tarm\tssim2\n'
    awk -F'\t' 'NR>1 {
      n = split($2, seg, "/"); bn = seg[n]; sub(/\.png$/, "", bn)
      split(bn, p, "__"); t = p[3]; sub(/^t/, "", t)
      print p[1] "\t" p[2] "\t" t "\t" p[4] "\t" $3
    }' "$OUT/xmetric_ssim2_raw.tsv"
  } > "$BD/zensim_mm_xmetric_2026-07-31.tsv"
  wc -l "$BD"/zensim_mm_{cells,traces,outer,xmetric}_2026-07-31.tsv | tee -a "$LOG"
fi

say "phase '$phase' complete"
