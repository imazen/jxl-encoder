#!/usr/bin/env python3
"""W44-218 Phase 4: derive Tier-2 ridge parameterizations from corpus
GEOMETRY (not response fitting).

W44-217 found that with only 13 LHS param blobs, per-pair response fits
have test R² ≤ 0.1 — the corpus is too sparse to identify coupling
coefficients individually. However, the corpus IS dense enough to
identify the EMPIRICAL RANGES each param swept, and to characterize the
shape of the joint param distribution.

This script derives 1-D Tier-2 ridges through the 6-D param space that:
1. Pass through the production defaults at k = k_default
2. Cover the empirical range observed in the W44-216 LHS sweep
3. Respect the coupling class observed in W44-217 (synergistic /
   suppressive / gated)

The ridges become the implementations of the skeleton fns in
src/tuning.rs::coupling. Defaults round-trip byte-for-byte.

The ridge parameterizations:

(p1, p2) ridge — smoothness_bias s ∈ [0, 1]:
  p1(s) = 85 + (P1_MAX - 85) * (1 - 2*s)            # higher s = stricter photo admit
  p2(s) = 95 + (P2_MAX - 95) * (1 - 2*s)            # higher s = lower screen threshold
  Defaults at s=0.5 → (85, 95).
  Both move together (positive correlation in W44-216 LHS top blobs).

(p3, p6) ridge — screenshot_quant_aggressiveness a ∈ [0, 2]:
  joint lift k = a (so a=1 = production, a=0 = nullify both, a=2 = max in LHS)
  with soft cap: effective_k = a if a < 1.0 else 1 + (a-1) * tanh((a-1)/0.5)
  p3(a) = 4 * effective_k
  p6(a) = 3 * effective_k
  Defaults at a=1.0 → (4, 3). Cap kicks in past a=1.2.
  Single 1-D ridge along the (p3/4, p6/3) diagonal.

(p4) — buttloop_screen_d_gate d ∈ [1.5, 5.0]:
  p4(d) = d
  Default at d=3.5 → 3.5. Direct expose.

(p5, p6) ridge — screen_quant_lift k ∈ [0.5, 2.0]:
  joint diagonal: p5(k) = 2.0 * k_capped, p6(k) = 3.0 * k_capped
  k_capped = k if k < 1.0 else 1 + (k-1) * 0.8       # 20% saturation cap
  Default at k=1.0 → (2, 3).
  Shares p6 with (p3, p6) ridge — coordinated at expand step.

(p4, p5) ridge — adaptive_quant_aggressiveness applied through screen_quant_lift,
  buttloop_screen_d_gate exposed directly.

Outputs:
- /tmp/w44-218/ridges.json — calibrated ridge constants
- /tmp/w44-218/ridge_coverage.tsv — for each ridge knob value, the
  resulting (p1..p6) and how many W44-216 LHS blobs land within ε of
  the curve.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULTS = {
    'p1': 85.0,
    'p2': 95.0,
    'p3': 4.0,
    'p4': 3.5,
    'p5': 2.0,
    'p6': 3.0,
}
OUT_DIR = '/tmp/w44-218'


def load_corpus_blob_params():
    df = pq.read_table('/tmp/w44-217/corpus_prepped.parquet').to_pandas()
    df = df[df['strategy'] == 'zenjxl']
    blobs = df[['params_blob_sha256',
                'p1_smart_zenjxl_photo_mask_p25_min',
                'p2_screenshot_median_threshold',
                'p3_buttloop_default_screenshot_qf_seed_scale',
                'p4_buttloop_qf_seed_scale_min_distance',
                'p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6',
                'p6_adaptive_quant_screenshot_qf_seed_scale_e7']].drop_duplicates().rename(columns={
        'p1_smart_zenjxl_photo_mask_p25_min': 'p1',
        'p2_screenshot_median_threshold': 'p2',
        'p3_buttloop_default_screenshot_qf_seed_scale': 'p3',
        'p4_buttloop_qf_seed_scale_min_distance': 'p4',
        'p5_adaptive_quant_screenshot_qf_seed_scale_e5_e6': 'p5',
        'p6_adaptive_quant_screenshot_qf_seed_scale_e7': 'p6',
    })
    return blobs.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Ridge parameterizations
# ─────────────────────────────────────────────────────────────────────


def p1_p2_ridge(smoothness_bias: float, p1_max: float, p2_max: float):
    """smoothness_bias ∈ [0, 1] → (p1, p2)

    At s=0.5 → defaults (85, 95).
    At s=0.0 → (P1_MAX, P2_MAX) — least smoothness, admit more screen content.
    At s=1.0 → (P1_MIN, P2_MIN) — most smoothness, fewer screen admissions.
    Both move together (W44-217 finds them positively correlated in top blobs).
    """
    s = float(smoothness_bias)
    p1_min = 2 * DEFAULTS['p1'] - p1_max
    p2_min = 2 * DEFAULTS['p2'] - p2_max
    p1 = DEFAULTS['p1'] + (p1_max - DEFAULTS['p1']) * (1.0 - 2.0 * s)
    p2 = DEFAULTS['p2'] + (p2_max - DEFAULTS['p2']) * (1.0 - 2.0 * s)
    p1 = max(p1_min, min(p1_max, p1))
    p2 = max(p2_min, min(p2_max, p2))
    return p1, p2


def screen_quant_lift(k: float, sat_strength: float = 0.8):
    """screen_quant_lift k ∈ [0.5, 2.0] → (p5, p6)

    Diagonal through default with soft cap above 1.0:
        k_eff = k if k <= 1.0 else 1 + (k - 1) * sat_strength
    Default at k=1.0 → (2.0, 3.0).
    """
    k = float(k)
    if k <= 1.0:
        k_eff = k
    else:
        k_eff = 1.0 + (k - 1.0) * sat_strength
    p5 = DEFAULTS['p5'] * k_eff
    p6 = DEFAULTS['p6'] * k_eff
    return p5, p6


def screenshot_quant_aggressiveness(a: float, sat_strength: float = 0.7):
    """screenshot_quant_aggressiveness a ∈ [0, 2] → (p3, p6)

    Joint lift on (p3, p6) — both target the same screen-class qac field.
    Stronger saturation than (p5, p6) because (p3, p6) is the FULL
    multiplicative lift at e7+. Cap kicks in past a=1.0.
    Default at a=1.0 → (4.0, 3.0).
    """
    a = float(a)
    if a <= 1.0:
        a_eff = a
    else:
        a_eff = 1.0 + (a - 1.0) * sat_strength
    p3 = DEFAULTS['p3'] * a_eff
    p6 = DEFAULTS['p6'] * a_eff
    return p3, p6


def buttloop_screen_d_gate(d: float):
    """buttloop_screen_d_gate d ∈ [1.5, 5.0] → p4

    Direct mapping. Default at d=3.5 → 3.5. Cover the W44-216 LHS range
    [1.71, 5.33].
    """
    return float(d)


def p4_p5_dispatch(d_gate: float, lift_k: float):
    """p4 from buttloop_screen_d_gate (direct), p5 from screen_quant_lift.

    Both are exposed simultaneously — the (p4, p5) coupling is GATED-by-p4
    and the user controls both knobs. Returns (p4, p5).
    """
    p4 = buttloop_screen_d_gate(d_gate)
    p5, _ = screen_quant_lift(lift_k)
    return p4, p5


def p4_p6_synergy(d_gate: float, lift_k: float):
    """p4 from buttloop_screen_d_gate (direct), p6 from screen_quant_lift.

    Same family as p4_p5. Returns (p4, p6).
    """
    p4 = buttloop_screen_d_gate(d_gate)
    _, p6 = screen_quant_lift(lift_k)
    return p4, p6


def p3_p4_photo_high_d(d_gate: float, a: float):
    """p3 from screenshot_quant_aggressiveness, p4 from buttloop_screen_d_gate.

    Mechanism: SYNERGISTIC at class=photo/very_high (W44-217 found
    +0.151 cross_norm on log_bytes). Both buttloop layers compose at
    photos that fall onto the W44-176 terminal-class path.
    Defaults at (d=3.5, a=1.0) → (4.0, 3.5).
    """
    p3, _ = screenshot_quant_aggressiveness(a)
    p4 = buttloop_screen_d_gate(d_gate)
    return p3, p4


def p1_p3_mutually_exclusive(smoothness: float, aggressiveness: float,
                              p1_max: float = None, p2_max: float = None):
    """p1 from smoothness_bias ridge, p3 from screenshot_quant_aggressiveness.

    Per W44-217: STRUCTURALLY MUTUALLY EXCLUSIVE — never co-fire.
    Tier-2 exposes them as two independent knobs.
    Defaults at (s=0.5, a=1.0) → (85, 4.0).
    """
    if p1_max is None:
        p1_max = 192.86
    if p2_max is None:
        p2_max = 108.15
    p1, _ = p1_p2_ridge(smoothness, p1_max, p2_max)
    p3, _ = screenshot_quant_aggressiveness(aggressiveness)
    return p1, p3


# ─────────────────────────────────────────────────────────────────────
# Calibrate the bounds from the W44-216 LHS empirical range
# ─────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    blobs = load_corpus_blob_params()
    print(f"Loaded {len(blobs)} unique param blobs")
    print(blobs.describe())

    p1_max = float(blobs['p1'].max())
    p2_max = float(blobs['p2'].max())

    # Calibrate (p1, p2) ridge bounds
    print("\n=== (p1, p2) ridge sample ===")
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p1, p2 = p1_p2_ridge(s, p1_max, p2_max)
        print(f"  s={s:.2f} → p1={p1:7.2f} p2={p2:7.2f}")

    # Calibrate (p5, p6) ridge
    print("\n=== screen_quant_lift sample ===")
    for k in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        p5, p6 = screen_quant_lift(k)
        print(f"  k={k:.2f} → p5={p5:.3f} p6={p6:.3f}")

    # Calibrate (p3, p6) ridge
    print("\n=== screenshot_quant_aggressiveness sample ===")
    for a in [0.0, 0.5, 1.0, 1.5, 2.0]:
        p3, p6 = screenshot_quant_aggressiveness(a)
        print(f"  a={a:.2f} → p3={p3:.3f} p6={p6:.3f}")

    # Verify roundtrip at defaults
    print("\n=== DEFAULT roundtrip check ===")
    p1, p2 = p1_p2_ridge(0.5, p1_max, p2_max)
    p3, p6_a = screenshot_quant_aggressiveness(1.0)
    p4 = buttloop_screen_d_gate(3.5)
    p5, p6_k = screen_quant_lift(1.0)
    # p6 should match between (p3, p6) and (p5, p6) ridges at defaults
    print(f"  p1={p1:.4f} (expected 85.0)")
    print(f"  p2={p2:.4f} (expected 95.0)")
    print(f"  p3={p3:.4f} (expected 4.0)")
    print(f"  p4={p4:.4f} (expected 3.5)")
    print(f"  p5={p5:.4f} (expected 2.0)")
    print(f"  p6 (from p5_p6_ridge) = {p6_k:.4f} (expected 3.0)")
    print(f"  p6 (from p3_p6_ridge) = {p6_a:.4f} (expected 3.0)")
    assert abs(p1 - 85.0) < 1e-4
    assert abs(p2 - 95.0) < 1e-4
    assert abs(p3 - 4.0) < 1e-4
    assert abs(p4 - 3.5) < 1e-4
    assert abs(p5 - 2.0) < 1e-4
    assert abs(p6_k - 3.0) < 1e-4
    assert abs(p6_a - 3.0) < 1e-4
    print("  ✓ All defaults round-trip cleanly")

    # Coverage analysis: how many LHS blobs land near each ridge?
    print("\n=== Coverage of LHS blobs against ridges ===")

    def dist_to_curve(blobs, ridge_fn, knob_range, columns):
        """For each blob, find the closest point on the ridge curve."""
        best_d = []
        best_k = []
        for _, row in blobs.iterrows():
            target = np.array([row[c] for c in columns])
            best = None; best_kk = None
            for k in knob_range:
                pred = np.array(list(ridge_fn(k)))
                # normalize by default to get relative distance
                norms = np.array([DEFAULTS[c] for c in columns])
                rel = (target - pred) / norms
                d = float(np.linalg.norm(rel))
                if best is None or d < best:
                    best = d; best_kk = k
            best_d.append(best); best_k.append(best_kk)
        return best_d, best_k

    # (p1, p2)
    knob_s = np.linspace(-2.0, 2.0, 401)
    p1p2_d, p1p2_k = dist_to_curve(blobs,
        lambda s: p1_p2_ridge(s, p1_max, p2_max), knob_s, ['p1', 'p2'])
    print(f"  (p1, p2) ridge: median blob distance = {np.median(p1p2_d):.3f}, max = {max(p1p2_d):.3f}")

    # (p3, p6)
    knob_a = np.linspace(-2.0, 4.0, 601)
    p3p6_d, p3p6_k = dist_to_curve(blobs,
        screenshot_quant_aggressiveness, knob_a, ['p3', 'p6'])
    print(f"  (p3, p6) ridge: median blob distance = {np.median(p3p6_d):.3f}, max = {max(p3p6_d):.3f}")

    # (p5, p6)
    knob_k = np.linspace(0.0, 4.0, 401)
    p5p6_d, p5p6_k = dist_to_curve(blobs,
        screen_quant_lift, knob_k, ['p5', 'p6'])
    print(f"  (p5, p6) ridge: median blob distance = {np.median(p5p6_d):.3f}, max = {max(p5p6_d):.3f}")

    # Persist ridge metadata
    ridge_meta = {
        'defaults': DEFAULTS,
        'p1_max_empirical': p1_max,
        'p2_max_empirical': p2_max,
        'p1_p2_smoothness_bias': {
            'knob_range': [0.0, 1.0],
            'knob_default': 0.5,
            'p1_max': p1_max,
            'p2_max': p2_max,
            'p1_min': 2 * DEFAULTS['p1'] - p1_max,
            'p2_min': 2 * DEFAULTS['p2'] - p2_max,
            'mechanism': 'SHARED-DISCRIMINATOR: linear ridge through (85, 95), positive slope (both move together)',
            'coverage_median': float(np.median(p1p2_d)),
            'coverage_max': float(max(p1p2_d)),
        },
        'screenshot_quant_aggressiveness': {
            'knob_range': [0.0, 2.0],
            'knob_default': 1.0,
            'saturation_strength': 0.7,
            'mechanism': 'SUPPRESSIVE/SATURATION: joint lift on (p3, p6), soft cap at ~6× combined',
            'coverage_median': float(np.median(p3p6_d)),
            'coverage_max': float(max(p3p6_d)),
        },
        'screen_quant_lift': {
            'knob_range': [0.5, 2.0],
            'knob_default': 1.0,
            'saturation_strength': 0.8,
            'mechanism': 'MULTIPLICATIVE-with-saturation: diagonal (k*2.0, k*3.0)',
            'coverage_median': float(np.median(p5p6_d)),
            'coverage_max': float(max(p5p6_d)),
        },
        'buttloop_screen_d_gate': {
            'knob_range': [1.5, 5.0],
            'knob_default': 3.5,
            'empirical_range': [float(blobs['p4'].min()), float(blobs['p4'].max())],
            'mechanism': 'GATED: direct distance threshold for buttloop screen lift',
        },
    }
    with open(f'{OUT_DIR}/ridges.json', 'w') as f:
        json.dump(ridge_meta, f, indent=2)
    print(f"\nWrote {OUT_DIR}/ridges.json")

    # Tabulate coverage
    cov_rows = []
    for ridge_name, knobs, fn, cols in [
        ('p1_p2_smoothness_bias', np.linspace(0.0, 1.0, 11),
         lambda s: p1_p2_ridge(s, p1_max, p2_max), ['p1','p2']),
        ('screen_quant_lift', np.linspace(0.5, 2.0, 16),
         screen_quant_lift, ['p5','p6']),
        ('screenshot_quant_aggressiveness', np.linspace(0.0, 2.0, 21),
         screenshot_quant_aggressiveness, ['p3','p6']),
        ('buttloop_screen_d_gate', np.linspace(1.5, 5.0, 8),
         lambda d: (buttloop_screen_d_gate(d),), ['p4']),
    ]:
        for k in knobs:
            pred = list(fn(k))
            cov_rows.append({'ridge': ridge_name, 'knob': float(k),
                             **{c: pred[i] for i, c in enumerate(cols)}})
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(f'{OUT_DIR}/ridge_coverage.tsv', sep='\t', index=False)
    print(f"Wrote {OUT_DIR}/ridge_coverage.tsv")


if __name__ == '__main__':
    main()
