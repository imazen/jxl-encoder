"""W44-221 Phase 3 prep: do the W44-218 ridge directions span the PC1-PC4 basis?

For each W44-218 knob, compute the "delta direction" as the param vector
moves over knob range [knob_min, knob_max] around default. Compare against
the PC1-PC4 right-singular-vectors from Phase 2b.

Outputs:
- phase3_ridge_alignment.log
- phase3_ridge_directions.tsv
"""
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("/tmp/w44-221")
LOG_PATH = OUT_DIR / "phase3_ridge_alignment.log"
LOG_HANDLE = LOG_PATH.open("w")


def log(msg):
    print(msg)
    LOG_HANDLE.write(msg + "\n")
    LOG_HANDLE.flush()


# Load Phase 2b basis
arrs = np.load(OUT_DIR / "phase2b_arrays.npz")
Vt = arrs["Vt"]  # [6 × 6], rows = PCs
S = arrs["S"]
explained_var = (S ** 2) / (S ** 2).sum()
log("Phase 2b PC variance fractions: " + ", ".join(f"PC{k+1}:{explained_var[k]:.3f}" for k in range(6)))

# W44-218 defaults
P_DEFAULT = np.array([85.0, 95.0, 4.0, 3.5, 2.0, 3.0])
PARAM_BOUNDS = {
    "p1": (40.50, 192.86),
    "p2": ( 75.63, 108.15),
    "p3": (  1.15,   7.89),
    "p4": (  1.71,   5.33),
    "p5": (  1.19,   3.80),
    "p6": (  1.64,   5.41),
}
# Same normalization the gradient SVD used: divide by W44-216 LHS range
# (i.e. we measure "knob direction" in normalized-param-space).
RANGES = np.array([h - l for (l, h) in PARAM_BOUNDS.values()])


# W44-218 ridge fns reimplemented
def p1_p2_ridge(s):
    P1_RIDGE_MAX, P2_RIDGE_MAX = 192.86, 108.15
    p1_unclamped = 85.0 + (P1_RIDGE_MAX - 85.0) * (1.0 - 2.0 * s)
    p2_unclamped = 95.0 + (P2_RIDGE_MAX - 95.0) * (1.0 - 2.0 * s)
    p1_lo = max(0.0, 2.0 * 85.0 - P1_RIDGE_MAX)
    p2_lo = max(0.0, 2.0 * 95.0 - P2_RIDGE_MAX)
    p1 = max(p1_lo, min(P1_RIDGE_MAX, p1_unclamped))
    p2 = max(p2_lo, min(P2_RIDGE_MAX, p2_unclamped))
    return p1, p2

def p3_p6_lift(a):
    P3_P6_SAT = 0.7
    a_eff = a if a <= 1.0 else 1.0 + (a - 1.0) * P3_P6_SAT
    return 4.0 * a_eff, 3.0 * a_eff

def p5_p6_lift(k):
    P5_P6_SAT = 0.8
    k_eff = k if k <= 1.0 else 1.0 + (k - 1.0) * P5_P6_SAT
    return 2.0 * k_eff, 3.0 * k_eff


# Compute ridge directions: for each W44-218 knob, central-difference around
# its default value to get a tangent direction in 6-param space.
KNOBS = [
    ("smoothness_bias",                  0.5, 0.45, 0.55, lambda s: build_p_smoothness(s)),
    ("screenshot_quant_aggressiveness",  1.0, 0.95, 1.05, lambda a: build_p_screen_aggr(a)),
    ("screen_quant_lift",                1.0, 0.95, 1.05, lambda k: build_p_screen_quant_lift(k)),
    ("buttloop_screen_d_gate",           3.5, 3.4,  3.6,  lambda d: build_p_buttloop_gate(d)),
]


def build_p_smoothness(s):
    p1, p2 = p1_p2_ridge(s)
    return np.array([p1, p2, 4.0, 3.5, 2.0, 3.0])


def build_p_screen_aggr(a):
    p3, p6 = p3_p6_lift(a)
    return np.array([85.0, 95.0, p3, 3.5, 2.0, p6])


def build_p_screen_quant_lift(k):
    p5, p6 = p5_p6_lift(k)
    return np.array([85.0, 95.0, 4.0, 3.5, p5, p6])


def build_p_buttloop_gate(d):
    return np.array([85.0, 95.0, 4.0, max(1.5, min(5.5, d)), 2.0, 3.0])


log("\nW44-218 ridge tangent directions (normalized) — central diff at default:")
ridge_dirs = []
for name, default, lo, hi, build_fn in KNOBS:
    p_lo = build_fn(lo)
    p_hi = build_fn(hi)
    delta = (p_hi - p_lo) / RANGES  # normalized
    norm = np.linalg.norm(delta)
    delta_unit = delta / (norm + 1e-12)
    ridge_dirs.append((name, delta_unit, norm))
    log(f"  {name:>32s}  raw_delta={delta}, ‖Δ‖={norm:.4f}")
    log(f"  {' ':>32s}  unit_vec=" + ", ".join(f"{v:+.3f}" for v in delta_unit))

# Compare to PC1..PC6 via inner products (cosine similarity)
log("\nCosine similarity between W44-218 ridge directions and PC1-PC6:")
log(f"  {'ridge':>32s}  " + "  ".join(f"PC{k+1:>2d}" for k in range(6)))
for name, dir_unit, _ in ridge_dirs:
    cos = np.array([abs(np.dot(dir_unit, Vt[k])) for k in range(6)])
    row = "  ".join(f"{c:+.3f}" for c in cos)
    log(f"  {name:>32s}  {row}")

# Are the 4 ridges linearly independent?
ridge_mat = np.array([d[1] for d in ridge_dirs])  # [4 × 6]
log(f"\n  Ridge-matrix rank: {np.linalg.matrix_rank(ridge_mat)} (4 rows)")

# Span analysis: what fraction of PC1+PC2+PC3+PC4 variance lies in the ridge span?
# Project each PC into the ridge subspace and report residual norm.
Q, _ = np.linalg.qr(ridge_mat.T)  # Q columns are orthonormal basis for ridge span
ridge_proj_norms = np.zeros(6)
for k in range(6):
    proj = Q @ (Q.T @ Vt[k])
    ridge_proj_norms[k] = np.linalg.norm(proj)

log(f"\n  PC projection norms onto W44-218 ridge span (4-dim):")
for k in range(6):
    log(f"    PC{k+1}: ‖proj‖ = {ridge_proj_norms[k]:.4f}  (1.0 = fully in span; lost: {(1 - ridge_proj_norms[k]**2)*100:.1f}%)")

ridge_coverage = sum(explained_var[k] * ridge_proj_norms[k]**2 for k in range(6))
log(f"\n  Total fraction of joint-gradient variance reachable by 4 W44-218 ridges: {ridge_coverage:.4f}")
log(f"  Remaining variance: {1 - ridge_coverage:.4f}")

# Now: project EACH ridge onto PC1..PC4 to see if they're nearly aligned.
log(f"\n  Ridge → PC1..PC4 projection (decomposition of each ridge in PC basis):")
log(f"  {'ridge':>32s}  " + "  ".join(f"PC{k+1:>2d}" for k in range(4)) + "  ‖proj‖²")
for name, dir_unit, _ in ridge_dirs:
    coeffs = np.array([np.dot(dir_unit, Vt[k]) for k in range(4)])
    norm = np.sum(coeffs ** 2)
    row = "  ".join(f"{c:+.3f}" for c in coeffs)
    log(f"  {name:>32s}  {row}  {norm:.4f}")

import csv

with (OUT_DIR / "phase3_ridge_directions.tsv").open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["ridge_name", "p1", "p2", "p3", "p4", "p5", "p6"] +
               [f"PC{k+1}_cos" for k in range(6)] +
               [f"PC{k+1}_proj" for k in range(4)])
    for name, dir_unit, _ in ridge_dirs:
        cos = [abs(np.dot(dir_unit, Vt[k])) for k in range(6)]
        proj = [np.dot(dir_unit, Vt[k]) for k in range(4)]
        w.writerow([name] + [f"{v:.6f}" for v in dir_unit] +
                   [f"{c:.6f}" for c in cos] + [f"{p:.6f}" for p in proj])

log(f"\n  Wrote: phase3_ridge_directions.tsv")
LOG_HANDLE.close()
