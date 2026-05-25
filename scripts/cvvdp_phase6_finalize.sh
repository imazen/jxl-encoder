#!/bin/bash
# cvvdp-fork Phase 6 finalize helper.
# Run after all 4 backend sweeps (B, B_GPU, C_GPU, C_CPU) are populated
# in benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv.
#
# Steps:
# 1. Clean malformed rows from the TSV (NF<13 → drop).
# 2. Run the Pareto analyzer.
# 3. Run the multi-decoder spotcheck (acceptance gate (e)).
# 4. Print the verdict line for inclusion in the decision memo.

set -euo pipefail

cd "$(dirname "$0")/.."

TSV="benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv"
DATE="2026-05-24"

echo "=== Phase 6 finalize ==="

# Clean malformed rows
echo "[1/4] Cleaning TSV (filter rows with NF=13)..."
TMP=$(mktemp)
awk -F'\t' 'NR==1 || NF == 13' "$TSV" > "$TMP"
ROW_PRE=$(wc -l < "$TSV")
ROW_POST=$(wc -l < "$TMP")
echo "       before=$ROW_PRE after=$ROW_POST"
if [ "$ROW_PRE" != "$ROW_POST" ]; then
    cp "$TMP" "$TSV"
    echo "       cleaned ($((ROW_PRE - ROW_POST)) rows dropped)"
fi
rm -f "$TMP"

# Per-backend cell counts
echo "[2/4] Backend coverage:"
awk -F'\t' 'NR>1 {c[$5]++} END {for(b in c) printf "       %s: %d\n", b, c[b]}' "$TSV" | sort

# Pareto analysis
echo "[3/4] Running cvvdp_pareto_analysis.py..."
python3 scripts/cvvdp_pareto_analysis.py "$TSV" "$DATE"

# Multi-decoder spotcheck
echo "[4/4] Multi-decoder spotcheck (acceptance gate (e))..."
if CUDA_PATH=/usr/local/cuda cargo run --release -p jxl-encoder \
    --features "__expert butteraugli-loop gpu-butteraugli cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel" \
    --example cvvdp_phase6_decoder_spotcheck 2>&1 | tail -30; then
    echo "       SPOTCHECK PASS"
else
    echo "       SPOTCHECK FAIL — see benchmarks/cvvdp_phase6_decoder_spotcheck_2026-05-24.tsv"
    echo "       This triggers REVERT per RFC §5.4 + Phase 6 brief gate (e)."
fi

echo "=== verdict ==="
grep -A 5 "^## VERDICT" "scripts/cvvdp_pareto_analysis_${DATE}.meta" | head -10
