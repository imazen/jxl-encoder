#!/usr/bin/env bash
# W44-PHASE4-S2-refit-c2: per-knob ablation audit across all 8 strata.
#
# For each stratum:
#   1. baseline: all knobs at default (k1=0.5, k2=1.0, k3=1.0, k4=3.5, k5=0.0)
#   2. full S2-refit: stratum-specific tuple from the lookup table
#   3-7. one ablation per knob: S2-refit tuple but with k_i reset to default
#
# 7 encodes × 8 strata = 56 encodes. ~1-2s each = ~2-3 min total.
#
# Output: TSV at benchmarks/w44_phase4_s2_refit_c2_audit_2026-05-24.tsv
set -euo pipefail

ROOT="/home/lilith/work/zen/jxl-encoder"
BIN="/home/lilith/work/zen/jxl-encoder-shared-target/release/examples/w44_phase4_s2_refit_c2_ablate"
OUT="${ROOT}/benchmarks/w44_phase4_s2_refit_c2_audit_2026-05-24.tsv"
CORPUS_CID22="/home/lilith/work/codec-corpus/CID22/CID22-512/validation"
CORPUS_GB82="/home/lilith/work/codec-corpus/gb82-sc"

# Wipe & start fresh
rm -f "${OUT}"

# Defaults (production defaults; expansion → RuntimeTuning::default() → no install)
D_K1=0.5
D_K2=1.0
D_K3=1.0
D_K4=3.5
D_K5=0.0

# Per-stratum S2-refit tuples (from src/tuning.rs default_for_stratum, 2026-05-24)
declare -A S2_K1 S2_K2 S2_K3 S2_K4 S2_K5
# screen/very_high (terminal e8 d=4 is the W44-105 SHIP cell)
S2_K1["screen/very_high"]=0.0
S2_K2["screen/very_high"]=0.0
S2_K3["screen/very_high"]=0.5
S2_K4["screen/very_high"]=2.1666666666666665
S2_K5["screen/very_high"]=-0.3333333333333334
# screen/high
S2_K1["screen/high"]=0.0
S2_K2["screen/high"]=0.3333333333333333
S2_K3["screen/high"]=0.5
S2_K4["screen/high"]=2.1666666666666665
S2_K5["screen/high"]=-0.6666666666666667
# screen/mid
S2_K1["screen/mid"]=0.16666666666666666
S2_K2["screen/mid"]=0.3333333333333333
S2_K3["screen/mid"]=0.5
S2_K4["screen/mid"]=4.166666666666666
S2_K5["screen/mid"]=-1.0
# screen/low
S2_K1["screen/low"]=0.0
S2_K2["screen/low"]=0.3333333333333333
S2_K3["screen/low"]=0.5
S2_K4["screen/low"]=2.1666666666666665
S2_K5["screen/low"]=-1.0
# photo/very_high
S2_K1["photo/very_high"]=0.0
S2_K2["photo/very_high"]=0.0
S2_K3["photo/very_high"]=0.5
S2_K4["photo/very_high"]=2.833333333333333
S2_K5["photo/very_high"]=0.33333333333333326
# photo/high
S2_K1["photo/high"]=0.0
S2_K2["photo/high"]=0.0
S2_K3["photo/high"]=0.5
S2_K4["photo/high"]=3.5
S2_K5["photo/high"]=-0.3333333333333334
# photo/mid
S2_K1["photo/mid"]=0.0
S2_K2["photo/mid"]=0.0
S2_K3["photo/mid"]=0.5
S2_K4["photo/mid"]=1.5
S2_K5["photo/mid"]=-1.0
# photo/low
S2_K1["photo/low"]=0.0
S2_K2["photo/low"]=0.0
S2_K3["photo/low"]=0.5
S2_K4["photo/low"]=1.5
S2_K5["photo/low"]=-1.0

# Cell selection per stratum: (image_path, distance, class)
# screen cells: terminal/graph from gb82-sc; photo: CID22 validation
declare -A CELL_IMG CELL_DIST CELL_CLASS
CELL_IMG["screen/very_high"]="${CORPUS_GB82}/terminal.png"  # W44-105 SHIP cell
CELL_DIST["screen/very_high"]=4.0
CELL_CLASS["screen/very_high"]=screen
CELL_IMG["screen/high"]="${CORPUS_GB82}/graph.png"
CELL_DIST["screen/high"]=3.0
CELL_CLASS["screen/high"]=screen
CELL_IMG["screen/mid"]="${CORPUS_GB82}/graph.png"
CELL_DIST["screen/mid"]=1.5
CELL_CLASS["screen/mid"]=screen
CELL_IMG["screen/low"]="${CORPUS_GB82}/graph.png"
CELL_DIST["screen/low"]=0.7
CELL_CLASS["screen/low"]=screen
CELL_IMG["photo/very_high"]="${CORPUS_CID22}/1531677.png"
CELL_DIST["photo/very_high"]=5.0
CELL_CLASS["photo/very_high"]=photo
CELL_IMG["photo/high"]="${CORPUS_CID22}/1189261.png"
CELL_DIST["photo/high"]=3.0
CELL_CLASS["photo/high"]=photo
CELL_IMG["photo/mid"]="${CORPUS_CID22}/1025469.png"
CELL_DIST["photo/mid"]=1.5
CELL_CLASS["photo/mid"]=photo
CELL_IMG["photo/low"]="${CORPUS_CID22}/1418519.png"
CELL_DIST["photo/low"]=0.7
CELL_CLASS["photo/low"]=photo

STRATA=("screen/very_high" "screen/high" "screen/mid" "screen/low"
        "photo/very_high" "photo/high" "photo/mid" "photo/low")

EFFORT=8

# Refresh marker periodically
refresh_marker() {
    date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && \
        printf '%s %s %s\n' "$(cat /tmp/ts)" "claude-w44-phase4-s2-refit-c2" "ablating: $1" \
        > "${ROOT}/.workongoing"
}

run_cell() {
    local stratum="$1" label="$2" k1="$3" k2="$4" k3="$5" k4="$6" k5="$7"
    local img="${CELL_IMG[${stratum}]}"
    local dist="${CELL_DIST[${stratum}]}"
    local cls="${CELL_CLASS[${stratum}]}"
    echo "[${stratum}] ${label}: k=(${k1}, ${k2}, ${k3}, ${k4}, ${k5})" >&2
    "${BIN}" \
        --image "${img}" \
        --effort ${EFFORT} --distance "${dist}" --class "${cls}" \
        --stratum-name "${stratum}" --knob-label "${label}" \
        --k1 "${k1}" --k2 "${k2}" --k3 "${k3}" --k4 "${k4}" --k5 "${k5}" \
        --append "${OUT}" > /dev/null
}

for stratum in "${STRATA[@]}"; do
    refresh_marker "${stratum}"
    s1=${S2_K1[${stratum}]}; s2=${S2_K2[${stratum}]}; s3=${S2_K3[${stratum}]}
    s4=${S2_K4[${stratum}]}; s5=${S2_K5[${stratum}]}

    # baseline (all defaults)
    run_cell "${stratum}" "baseline_defaults" "${D_K1}" "${D_K2}" "${D_K3}" "${D_K4}" "${D_K5}"
    # full S2-refit
    run_cell "${stratum}" "full_s2refit" "${s1}" "${s2}" "${s3}" "${s4}" "${s5}"
    # ablations: knob k_i reset to default, others at S2-refit
    run_cell "${stratum}" "k1_default" "${D_K1}" "${s2}" "${s3}" "${s4}" "${s5}"
    run_cell "${stratum}" "k2_default" "${s1}" "${D_K2}" "${s3}" "${s4}" "${s5}"
    run_cell "${stratum}" "k3_default" "${s1}" "${s2}" "${D_K3}" "${s4}" "${s5}"
    run_cell "${stratum}" "k4_default" "${s1}" "${s2}" "${s3}" "${D_K4}" "${s5}"
    run_cell "${stratum}" "k5_default" "${s1}" "${s2}" "${s3}" "${s4}" "${D_K5}"
done

echo "DONE. Output: ${OUT}" >&2
wc -l "${OUT}"
