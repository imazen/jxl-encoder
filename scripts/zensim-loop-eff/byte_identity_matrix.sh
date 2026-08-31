#!/usr/bin/env bash
# Copyright (c) Imazen LLC and the JPEG XL Project Authors.
# Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
#
# Byte-identity matrix for the zensim loop's configuration surface.
#
# Runs `examples/zensim_config_byte_identity` once per env arm — ONE ARM PER
# PROCESS, because `JXL_ZENSIM_RD_PROFILE` resolves through a process-wide
# `OnceLock` — and concatenates the per-cell SHA256 lines into one TSV. It also
# captures the four instrumentation sinks and normalises the wall-clock columns
# out of them, so their COLUMN SHAPES and deterministic values are diffable too.
#
# Acceptance for a behaviour-preserving refactor of `vardct/zensim_loop.rs`:
# run this on both sides of the change and `diff -r` the two output dirs. Empty
# diff = byte identity of the bitstreams AND of the traces, with and without
# each env var set. A non-empty diff means the refactor changed behaviour —
# find out why before shipping it.
#
#   ./byte_identity_matrix.sh <out_dir> [corpus_dir] [limit]
#
# Build the binary first (ONE cargo at a time on the shared box):
#   nice -n 19 cargo build --release -p jxl-encoder --features zensim-loop \
#     --example zensim_config_byte_identity -j 4
set -euo pipefail

OUTDIR="${1:?usage: byte_identity_matrix.sh <out_dir> [corpus_dir] [limit]}"
CORPUS="${2:-$HOME/work/codec-corpus/CID22/CID22-512/validation}"
LIMIT="${3:-4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${REPO_ROOT}/target/release/examples/zensim_config_byte_identity"
[ -x "$BIN" ] || { echo "missing $BIN — build it first (see header)" >&2; exit 1; }

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
HASHES="${OUTDIR}/hashes.tsv"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# Every env name the loop reads. Cleared for every arm so a stray value in the
# caller's shell cannot silently contaminate the matrix.
ZENSIM_ENV=(
  ZENSIM_MASKING ZENSIM_SQRT ZENSIM_HF ZENSIM_EDGE_MSE ZENSIM_NORM
  ZENSIM_SPATIAL_W ZENSIM_RATIO_MAX ZENSIM_ALPHA ZENSIM_FACTOR_MAX
  ZENSIM_H3_GAIN ZENSIM_H3_GAIN_MODE ZENSIM_ATTR_BIN
  JXL_ZENSIM_RD_PROFILE JXL_ZENSIM_MAP_BAKE JXL_ZENSIM_MODEL_MAP
  JXL_ZENSIM_SINGLEPASS JXL_ZENSIM_MAP_EMA JXL_ZENSIM_QF_GLOBAL_SCALE
  JXL_ZENSIM_TARGET_SCORE JXL_ZENSIM_TARGET_TOL JXL_ZENSIM_EMIT_BEST
  JXL_ZENSIM_CTRL_CLAMP JXL_ZENSIM_CTRL_EXP JXL_ZENSIM_S4_EPS
  JXL_ZENSIM_SECANT JXL_ZENSIM_SECANT_MIN_DLNL JXL_ZENSIM_SECANT_MIN_EPS
  JXL_ZENSIM_SECANT_TRACE JXL_ZENSIM_RD_STATS JXL_ZENSIM_TRACE
  JXL_ZENSIM_TRACE_ID JXL_ZENSIM_ATTR_PROBE JXL_ZENSIM_FDPROBE
)
UNSET=()
for v in "${ZENSIM_ENV[@]}"; do UNSET+=(-u "$v"); done

: > "$HASHES"
run_arm() {
  local label="$1"; shift
  echo "  arm: ${label}" >&2
  env "${UNSET[@]}" "$@" "$BIN" --corpus "$CORPUS" --limit "$LIMIT" --label "$label" \
    | grep -v '^#' >> "$HASHES"
}

echo "== bitstream hashes ==" >&2

# 1. The shipped default: no env at all. This is the arm that must reproduce
#    the pre-change bitstream byte for byte.
run_arm default

# 2. Knobs that act WITHOUT a score target (redistribution + the pre-scale).
run_arm qf_scale_1p1      JXL_ZENSIM_QF_GLOBAL_SCALE=1.1
run_arm alpha_0p5         ZENSIM_ALPHA=0.5
run_arm norm_4            ZENSIM_NORM=4.0
run_arm factor_max_1p4    ZENSIM_FACTOR_MAX=1.4
run_arm spatial_w_1       ZENSIM_SPATIAL_W=1.0
run_arm masking_none      ZENSIM_MASKING=none

# 3. The damped controller. Everything in this block is INERT without
#    `JXL_ZENSIM_TARGET_SCORE` (the whole controller sits inside
#    `if let Some(tgt) = target_native`), so each arm carries a target.
run_arm target_92         JXL_ZENSIM_TARGET_SCORE=92
run_arm t92_secant_off    JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_SECANT=0
run_arm t92_min_eps_0     JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_SECANT_MIN_EPS=0
run_arm t92_min_eps_0p60  JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_SECANT_MIN_EPS=0.60
run_arm t92_min_dlnl_0p3  JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_SECANT_MIN_DLNL=0.3
run_arm t92_exp_0p6       JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_CTRL_EXP=0.6
run_arm t92_exp_1p5       JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_CTRL_EXP=1.5
run_arm t92_clamp_1p35    JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_CTRL_CLAMP=1.35
run_arm t92_tol_8         JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_TARGET_TOL=8.0
run_arm t92_emit_best     JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_EMIT_BEST=1
run_arm t92_s4_off        JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_S4_EPS=0

# 4. Rejected / malformed values must fall back to the shipped default — the
#    validation filters are behaviour, not decoration. Each of these must equal
#    the `default` arm cell-for-cell WITHIN a single run, as well as across the
#    refactor.
run_arm bad_clamp_0p5     JXL_ZENSIM_CTRL_CLAMP=0.5
run_arm bad_exp_9         JXL_ZENSIM_CTRL_EXP=9
run_arm bad_attr_bin_0    ZENSIM_ATTR_BIN=0
run_arm qf_scale_1p0      JXL_ZENSIM_QF_GLOBAL_SCALE=1.0
run_arm garbage_norm      ZENSIM_NORM=not-a-number
run_arm emit_best_notgt   JXL_ZENSIM_EMIT_BEST=1

# 5. Instrumentation sinks. Column SHAPES are load-bearing:
#      JXL_ZENSIM_TRACE        7 cols  (numerically diffed by the substrate
#                                       probe in analyze_23shot.py verify —
#                                       must not gain columns)
#      JXL_ZENSIM_SECANT_TRACE 10 cols (a SEPARATE file for exactly that reason)
#      JXL_ZENSIM_RD_STATS     4 cols
#      JXL_ZENSIM_ATTR_PROBE   4 cols  (asserted by tests/zensim_attr_smoke.rs)
#    Wall-clock columns are normalised out so the rest is diffable.
echo "== instrumentation sinks ==" >&2
TR="${SCRATCH}/trace.tsv"; ST="${SCRATCH}/secant.tsv"
RS="${SCRATCH}/stats.tsv"; AP="${SCRATCH}/attr.tsv"

echo "  sink: traces (target arm)" >&2
env "${UNSET[@]}" \
  JXL_ZENSIM_TARGET_SCORE=92 JXL_ZENSIM_TRACE="$TR" JXL_ZENSIM_TRACE_ID=bid \
  JXL_ZENSIM_SECANT_TRACE="$ST" JXL_ZENSIM_RD_STATS="$RS" \
  "$BIN" --corpus "$CORPUS" --limit 2 --label traces >/dev/null

echo "  sink: attr probe (h3-mag arm, profile b)" >&2
env "${UNSET[@]}" \
  JXL_ZENSIM_RD_PROFILE=b JXL_ZENSIM_MODEL_MAP=h3-mag JXL_ZENSIM_ATTR_PROBE="$AP" \
  "$BIN" --corpus "$CORPUS" --limit 1 --label attrprobe >/dev/null

col_shape() { awk -F'\t' '{print NF}' "$1" | sort -u | tr '\n' ',' ; }
{
  echo "sink	lines	distinct_column_counts"
  echo "JXL_ZENSIM_TRACE	$(wc -l < "$TR" | tr -d ' ')	$(col_shape "$TR")"
  echo "JXL_ZENSIM_SECANT_TRACE	$(wc -l < "$ST" | tr -d ' ')	$(col_shape "$ST")"
  echo "JXL_ZENSIM_RD_STATS	$(wc -l < "$RS" | tr -d ' ')	$(col_shape "$RS")"
  echo "JXL_ZENSIM_ATTR_PROBE	$(wc -l < "$AP" | tr -d ' ')	$(col_shape "$AP")"
} > "${OUTDIR}/sink_shapes.tsv"

# Deterministic columns only: TRACE drops iter_ms (col 7); RD_STATS drops the
# total and per-iter millisecond columns (3,4). SECANT_TRACE and ATTR_PROBE
# carry no timing at all.
cut -f1-6 "$TR" > "${OUTDIR}/trace_deterministic.tsv"
cp "$ST"        "${OUTDIR}/secant_trace.tsv"
cut -f1-2 "$RS" > "${OUTDIR}/stats_deterministic.tsv"
cp "$AP"        "${OUTDIR}/attr_probe.tsv"

echo "wrote $(grep -c . "$HASHES") bitstream cells + 4 sinks to $OUTDIR" >&2
