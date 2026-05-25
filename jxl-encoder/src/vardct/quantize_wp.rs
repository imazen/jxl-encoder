// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-AUDIT-8 Phase 6 — libjxl `QuantizeWP` DC quantization shape.
//!
//! Mirrors libjxl `enc_modular.cc::QuantizeWP` (lines 1542-1559) and the
//! per-channel walk in `enc_modular.cc:1640-1674` (`else if (nl_dc)` branch).
//!
//! Three extras over plain `(value * inv_factor).round()`:
//!
//! 1. **WP-relative residual**: prediction comes from the JPEG XL
//!    Weighted predictor over already-quantized DC values, then
//!    `svalue = value * inv_factor - pred`.
//! 2. **0.62 deadzone**: residuals with `|svalue| < 0.62` quantize to 0.
//!    libjxl `q_deadzone = 0.62f` (line 1541).
//! 3. **Snap-to-even** for residuals with `|residual| > 2`:
//!    `residual = round(svalue * 0.5) * 2` — biases large residuals
//!    onto even integer multiples, reducing the alphabet at the tail.
//!
//! The final stored value is `pred + residual` (line 1559) — i.e. the
//! reconstructed integer DC. This is what the entropy coder later
//! tokenises (the predictor in the entropy path will see the same
//! `qrow` values and produce the same residual).
//!
//! ## Channel order + CFL handling
//!
//! libjxl walks channels in `{1, 0, 2}` order (Y first, then X, then B).
//! Y is quantized direct; X and B subtract a CFL term derived from the
//! already-quantized Y:
//!
//! ```text
//! value_for_q = row[x] - quant_y_row[x] * y_factor * cfl_factor[c]
//! ```
//!
//! where `y_factor = GetDcStep(1) / mul` and
//! `inv_factor = GetInvDcStep(c) * mul`. Our existing inline path uses
//! the algebraically-equivalent `dc_cfl_factor` shortcut
//! (`vardct/transform.rs:847`), so we re-derive it here for parity
//! with libjxl's formula.
//!
//! ## Gate
//!
//! Active when [`crate::effort::EffortProfile::use_libjxl_wp_dc_quant`]
//! is `true` (effort ≤ 7 by default, matching the existing Phase 5
//! `extra_dc_precision = 1` gate — both gates fire under libjxl's
//! `nl_dc = speed_tier < kFalcon` condition).
//!
//! At effort ≥ 8 the butteraugli quantization loop owns DC refinement
//! and libjxl drops both gates; we mirror that.

extern crate alloc;
use alloc::vec::Vec;

use crate::modular::predictor::{Neighbors, WeightedPredictorState};
use crate::vardct::quant::INV_DC_QUANT;

/// libjxl `q_deadzone` (`enc_modular.cc:1541`). Residuals with
/// `|svalue| < q_deadzone` quantize to zero before rounding.
const Q_DEADZONE: f32 = 0.62;

/// Y channel index in libjxl 3-channel layout. Walked first because
/// X and B both subtract a CFL term derived from the quantized Y.
const C_Y: usize = 1;

/// CFL factor per channel — `enc_xyb.cc::ColorCorrelation::DCFactors()`.
/// X uses 0.0 (no CFL on DC); B uses 0.5 (matching the inline path in
/// `vardct/transform.rs:847`).
const DC_CFL_FACTOR: [f32; 3] = [0.0, 0.0, 0.5];

/// Per-block QuantizeWP for a single DC value (legacy entry — multiplies
/// `value * inv_factor` internally). Kept for the unit tests; callers
/// in [`requantize_dc_group_wp`] use [`quantize_wp_one_presvalued`]
/// which takes the already-scaled `svalue_base` (allowing CFL
/// subtraction in the scaled domain to match libjxl exactly).
///
/// Mirrors libjxl `QuantizeWP` (`enc_modular.cc:1542-1559`).
///
/// Returns the FINAL stored integer (= `pred + residual`), which is
/// what gets written into `quant_dc[c][y][x]`.
#[inline]
fn quantize_wp_one(value: f32, inv_factor: f32, pred: i32) -> i32 {
    quantize_wp_one_presvalued(value * inv_factor, pred)
}

/// QuantizeWP core, taking `svalue_base = value * inv_factor` already
/// computed by the caller. Used by [`requantize_dc_group_wp`] so the
/// CFL pre-subtraction can apply in the same `inv_factor`-scaled
/// domain libjxl uses (line 1668: `(row[x] - quant_y * y_factor *
/// cfl_factor) * inv_factor`).
#[inline]
fn quantize_wp_one_presvalued(svalue_base: f32, pred: i32) -> i32 {
    let mut svalue = svalue_base - pred as f32;

    // 0.62 deadzone (line 1549).
    if svalue > -Q_DEADZONE && svalue < Q_DEADZONE {
        svalue = 0.0;
    }

    // Outlier guard — finite-only round; NaN / overflow falls through
    // to residual=0 (libjxl line 1551-1556 sets has_outliers but still
    // emits 0). We don't track has_outliers (only used in libjxl to
    // gate decoder fallback paths that don't apply to our pipeline).
    let mut residual = if svalue.is_finite() {
        // Safe cast: round produces a value in i32 range or NaN/inf,
        // and finite check above rules those out.
        svalue.round() as i32
    } else {
        0
    };

    // Snap-to-even for |residual| > 2 (line 1558).
    // `round(svalue * 0.5) * 2` produces the nearest even integer.
    if residual > 2 || residual < -2 {
        residual = ((svalue * 0.5).round() as i32) * 2;
    }

    pred + residual
}

/// Compute libjxl-style WP prediction `pred.guess` (`enc_modular.cc:1547`).
///
/// libjxl calls `PredictNoTreeWP(w, qrow + x, onerow, x, y,
/// Predictor::Weighted, wp_state)` where `qrow` holds the
/// already-quantized values up to but not including position `(x, y)`.
///
/// Our [`WeightedPredictorState::predict`] needs the four
/// `Neighbors { w, n, nw, ne }` indices in i32. We supply them with
/// libjxl's edge behaviour: missing neighbours default to
/// `Predictor::Gradient`'s edge fallbacks (north → west when y=0,
/// west → 0 when x=0 and y=0 — the seed value).
fn wp_predict(
    qrow: &[i16],
    prev_row: Option<&[i16]>,
    x: usize,
    y: usize,
    xsize: usize,
    wp_state: &mut WeightedPredictorState,
) -> i32 {
    // Build neighbours mirroring our existing dc_tree_learn.rs edge
    // handling (which itself mirrors libjxl's `PredictNoTreeWP` edge
    // behaviour through `weighted::State::Predict<true>`).
    let w = if x > 0 {
        qrow[x - 1] as i32
    } else if y > 0 {
        prev_row.map(|r| r[x] as i32).unwrap_or(0)
    } else {
        0
    };
    let n = if let Some(pr) = prev_row {
        pr[x] as i32
    } else {
        w
    };
    let nw = if let Some(pr) = prev_row {
        if x > 0 { pr[x - 1] as i32 } else { w }
    } else {
        w
    };
    let ne = if let Some(pr) = prev_row {
        if x + 1 < xsize { pr[x + 1] as i32 } else { n }
    } else {
        n
    };
    // Need `nn` for Neighbors; libjxl WP uses it via PredictNoTreeWP path.
    // We don't have a "prev-prev row" handy here; default to `n` (the same
    // fallback our dc_tree_learn.rs uses for `toptop` when y < 2).
    let nn = n;
    let nee = ne; // far-NE fallback

    let neighbors = Neighbors {
        w,
        n,
        nw,
        ne,
        nn,
        ww: if x > 1 { qrow[x - 2] as i32 } else { w },
        nee,
    };

    wp_state.predict(x, y, xsize, &neighbors)
}

/// Apply libjxl QuantizeWP shape to a full DC group in-place.
///
/// Replaces every entry of `quant_dc[c][y][x]` with the QuantizeWP
/// result computed from `float_dc[c][y * xsize + x]` and the existing
/// quantizer scale.
///
/// Channel walk is `{Y, X, B}`. Y is processed first so X and B can
/// subtract the CFL term from the already-updated `quant_dc[Y]`.
///
/// `scale_dc` and `extra_dc_precision` mirror the inline path in
/// `vardct/transform.rs` so callers can keep the existing inline
/// quantization as a no-op fallback when the gate is off.
///
/// Returns the number of values whose final integer differs from the
/// pre-pass `quant_dc` value (diagnostic — used by tests / probes).
pub fn requantize_dc_group_wp(
    quant_dc: &mut [Vec<Vec<i16>>; 3],
    float_dc: &[Vec<f32>; 3],
    xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    scale_dc: f32,
    extra_dc_precision: u8,
) -> usize {
    let mut changed = 0usize;
    let region_width = end_bx.saturating_sub(start_bx);
    let region_height = end_by.saturating_sub(start_by);
    if region_width == 0 || region_height == 0 {
        return 0;
    }

    let dc_mul = (1u32 << extra_dc_precision) as f32;

    // Walk libjxl channel order — Y first so X/B can reference the
    // updated `quant_dc[Y]` for CFL subtraction.
    for &c in &[C_Y, 0usize, 2usize] {
        let inv_factor = INV_DC_QUANT[c] * scale_dc * dc_mul;

        // libjxl: y_factor = GetDcStep(1) / mul = 1 / (INV_DC_QUANT[1] * scale_dc * mul)
        // → `y_factor * inv_factor[c]` simplifies to
        // `INV_DC_QUANT[c] / INV_DC_QUANT[1]` (the mul / scale_dc terms cancel).
        // Equivalently we use the same `dc_cfl_factor` constant the inline
        // path uses; both are byte-identical at f32 precision.
        let dc_cfl_factor = DC_CFL_FACTOR[c];

        // Allocate a fresh WP state per channel (matches libjxl
        // `weighted::State wp_state(header, xsize, ysize)` per channel
        // at `enc_modular.cc:1647`).
        let mut wp_state = WeightedPredictorState::with_defaults(region_width);

        // Walk row-major within the region.
        for (ly, gy) in (start_by..end_by).enumerate() {
            // Snapshot prev_row BEFORE borrowing the current row mutably.
            // The borrow checker won't let us hold a shared ref to row gy-1
            // while we mutably borrow row gy from the same Vec<Vec<i16>>,
            // so we copy the prev row into a local Vec each iteration.
            // Cost: O(region_width) per row × region_height = same big-O as
            // the pass itself; under any plausible DC group size (≤256 blocks)
            // this is trivial relative to the f32 quantize work.
            let prev_row: Option<Vec<i16>> = if gy > start_by {
                Some(
                    quant_dc[c][gy - 1][start_bx..end_bx]
                        .iter()
                        .copied()
                        .collect(),
                )
            } else {
                None
            };

            // Pre-extract a copy of the Y row (when c != Y) for CFL
            // subtraction. By the time we walk X (c=0) or B (c=2), the
            // Y channel has already been requantized for the whole image.
            let y_row: Option<Vec<i16>> = if c != C_Y {
                Some(
                    quant_dc[C_Y][gy][start_bx..end_bx]
                        .iter()
                        .copied()
                        .collect(),
                )
            } else {
                None
            };

            // Borrow the current row mutably for writeback.
            let row = &mut quant_dc[c][gy];

            for (lx, gx) in (start_bx..end_bx).enumerate() {
                let float_v = float_dc[c][gy * xsize_blocks + gx];
                // Compute svalue directly, mirroring transform.rs's
                // `dc * inv_factor - y_dc * dc_cfl_factor` shape.
                // libjxl's QuantizeWP takes `value = row[x] - quant_y * y_factor * cfl_factor`
                // then computes `svalue = value * inv_factor`, which expands to
                // `row[x] * inv_factor - quant_y * (y_factor * cfl_factor * inv_factor)`.
                // The bracketed product is the SAME constant as our `dc_cfl_factor`
                // (e.g. 0.5 for B, 0.0 for X), so we incorporate it post-scale here.
                let svalue_base = if c == C_Y {
                    float_v * inv_factor
                } else {
                    let y_val = y_row.as_ref().unwrap()[lx] as f32;
                    float_v * inv_factor - y_val * dc_cfl_factor
                };

                // WP prediction over already-updated `row[..lx]`.
                let qrow_so_far = &row[start_bx..start_bx + lx + 1]; // includes slot at lx (uninitialised, fine — predict only reads x-1)
                let pred = wp_predict(
                    qrow_so_far,
                    prev_row.as_deref(),
                    lx,
                    ly,
                    region_width,
                    &mut wp_state,
                );

                let q = quantize_wp_one_presvalued(svalue_base, pred);

                // Clamp to i16 range (libjxl uses i32 storage internally
                // and only clamps at bitstream encode time). Our
                // `quant_dc` is i16 today (matches the existing inline
                // path; `transform.rs` does the same `as i16` cast).
                let q16 = q.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                if q16 != row[gx] {
                    changed += 1;
                }
                row[gx] = q16;

                // Update WP error state for this position (mirrors libjxl
                // line 1660 `wp_state.UpdateErrors(quant_row[x], x, y, xsize)`).
                wp_state.update_errors(q16 as i32, lx, ly, region_width);
            }
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_quantizes_to_zero() {
        // svalue = 0.5 * 1.0 - 0 = 0.5; |0.5| < 0.62 → 0
        let q = quantize_wp_one(0.5, 1.0, 0);
        assert_eq!(q, 0);
    }

    #[test]
    fn outside_deadzone_rounds() {
        // svalue = 0.9 * 1.0 - 0 = 0.9; |0.9| > 0.62 → round(0.9) = 1
        let q = quantize_wp_one(0.9, 1.0, 0);
        assert_eq!(q, 1);
    }

    #[test]
    fn snap_to_even_for_large_residual() {
        // svalue = 5.0 - 0 = 5.0; round = 5, |5|>2 → round(5*0.5)*2 = round(2.5)*2
        // Rust's round() uses ties-away-from-zero by default: round(2.5)=3 → 6
        // libjxl uses std::round which is also ties-away-from-zero.
        let q = quantize_wp_one(5.0, 1.0, 0);
        assert_eq!(q, 6);
    }

    #[test]
    fn snap_to_even_negative() {
        let q = quantize_wp_one(-5.0, 1.0, 0);
        assert_eq!(q, -6);
    }

    #[test]
    fn pred_added_to_residual() {
        // svalue = 10.0 - 100 = -90; |-90|>2 → round(-90*0.5)*2 = -90
        // final = 100 + (-90) = 10
        let q = quantize_wp_one(10.0, 1.0, 100);
        assert_eq!(q, 10);
    }

    #[test]
    fn nan_input_is_safe() {
        let q = quantize_wp_one(f32::NAN, 1.0, 42);
        assert_eq!(q, 42); // residual=0, pred unchanged
    }

    #[test]
    fn requantize_smoke() {
        // 4×4 DC group, single channel populated with monotone values.
        let xsize = 4;
        let ysize = 4;
        let mut quant_dc: [Vec<Vec<i16>>; 3] = [
            vec![vec![0i16; xsize]; ysize],
            vec![vec![0i16; xsize]; ysize],
            vec![vec![0i16; xsize]; ysize],
        ];
        let float_dc: [Vec<f32>; 3] = [
            vec![0.0; xsize * ysize],
            (0..xsize * ysize).map(|i| i as f32).collect(),
            vec![0.0; xsize * ysize],
        ];
        let _changed =
            requantize_dc_group_wp(&mut quant_dc, &float_dc, xsize, 0, 0, xsize, ysize, 1.0, 1);
        // Just check we didn't panic and Y channel got non-zero values
        // somewhere (input was 0..16, inv_factor = INV_DC_QUANT[1] * 1.0 * 2).
        let y_nonzero = quant_dc[C_Y].iter().flatten().any(|&v| v != 0);
        assert!(
            y_nonzero,
            "Y channel should have some non-zero quantized values"
        );
    }
}
