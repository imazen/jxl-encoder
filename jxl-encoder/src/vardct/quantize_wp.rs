// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Prediction-relative DC quantization: libjxl `QuantizeWP` / `QuantizeGradient`.
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
//! value_for_q = row[x] - quant_y_row[x] * (y_factor * cfl_factor[c])
//! ```
//!
//! where `y_factor = GetDcStep(1) / mul` and
//! `inv_factor = GetInvDcStep(c) * mul`. Strict parity keeps this source-domain
//! subtraction order. Normal mode retains the calibrated post-scale shortcut;
//! cancelling these factors can change f32 rounding at quantization boundaries.
//!
//! Strict DC parity selects Gradient prediction when faster decoding is set,
//! matching libjxl `AddVarDCTDC`. Normal policies retain Weighted prediction.
//! Both apply the same deadzone and snap-to-even rule.
//!
//! ## Gate
//!
//! Active when [`crate::effort::EffortProfile::use_libjxl_wp_dc_quant`]
//! is `true` (effort ≥ 4 under strict parity, matching libjxl's
//! `nl_dc = speed_tier < kFalcon` condition — the same condition that
//! sets `extra_dc_precision = 1`).
//!
//! Corrected 2026-09-22: this pass also runs at effort ≥ 8. libjxl's
//! `else if (nl_dc)` arm in `AddVarDCTDC` has no further speed-tier
//! gate — the butteraugli loop refines `global_scale`/`quant_dc`
//! *scales*, then `QuantizeWP` rewrites the DC *values* so the shipped
//! modular stream is self-consistent under the Weighted predictor the
//! DC stream uses (`kWPOnly` learned tree at e8+, `kWPFixedDC` at
//! `tier ≥ kSquirrel`). An earlier comment claiming "e8+ no-op" was
//! wrong; the call site was already correct.

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
/// in [`requantize_dc_group`] use [`quantize_wp_one_presvalued`]
/// which takes the already-scaled `svalue_base` (allowing CFL
/// subtraction before prediction-relative residual quantization).
///
/// Mirrors libjxl `QuantizeWP` (`enc_modular.cc:1542-1559`).
///
/// Returns the FINAL stored integer (= `pred + residual`), which is
/// what gets written into `quant_dc[c][y][x]`.
#[cfg(test)]
fn quantize_wp_one(value: f32, inv_factor: f32, pred: i32) -> i32 {
    quantize_wp_one_presvalued(value * inv_factor, pred)
}

/// QuantizeWP core, taking `svalue_base = value * inv_factor` already
/// computed by the caller. Used by [`requantize_dc_group`] so the
/// caller can preserve the reference operation order before entering this core.
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
    if !(-2..=2).contains(&residual) {
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
fn dc_predict(
    qrow: &[i32],
    prev_row: Option<&[i32]>,
    x: usize,
    y: usize,
    xsize: usize,
    wp_state: Option<&mut WeightedPredictorState>,
) -> i32 {
    // Build neighbours mirroring our existing dc_tree_learn.rs edge
    // handling (which itself mirrors libjxl's `PredictNoTreeWP` edge
    // behaviour through `weighted::State::Predict<true>`).
    let w = if x > 0 {
        qrow[x - 1]
    } else if y > 0 {
        prev_row.map(|r| r[x]).unwrap_or(0)
    } else {
        0
    };
    let n = if let Some(pr) = prev_row { pr[x] } else { w };
    let nw = if let Some(pr) = prev_row {
        if x > 0 { pr[x - 1] } else { w }
    } else {
        w
    };
    let ne = if let Some(pr) = prev_row {
        if x + 1 < xsize { pr[x + 1] } else { n }
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
        ww: if x > 1 { qrow[x - 2] } else { w },
        nee,
    };

    match wp_state {
        Some(state) => state.predict(x, y, xsize, &neighbors),
        None => crate::modular::predictor::clamped_gradient(n, w, nw),
    }
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
/// `reference_y_factor` selects reference-domain CfL subtraction. It is the
/// wire quantizer's `GetDcStep(Y) / mul`, computed without inverting a rounded
/// inverse step. `None` retains the calibrated normal scaled-domain arithmetic.
///
/// Returns the number of values whose final integer differs from the
/// pre-pass `quant_dc` value (diagnostic — used by tests / probes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn requantize_dc_group(
    quant_dc: &mut [Vec<Vec<i32>>; 3],
    float_dc: &[Vec<f32>; 3],
    xsize_blocks: usize,
    start_bx: usize,
    start_by: usize,
    end_bx: usize,
    end_by: usize,
    scale_dc: f32,
    extra_dc_precision: u8,
    gradient: bool,
    reference_y_factor: Option<f32>,
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

        // Normal mode retains its calibrated post-scale CfL factor. Strict
        // parity subtracts quantized Y in the source domain before scaling;
        // cancelling factors changes rounding at quantization boundaries.
        let dc_cfl_factor = DC_CFL_FACTOR[c];

        // Allocate a fresh WP state per channel (matches libjxl
        // `weighted::State wp_state(header, xsize, ysize)` per channel
        // at `enc_modular.cc:1647`).
        let mut wp_state = (!gradient).then(|| WeightedPredictorState::with_defaults(region_width));

        // Walk row-major within the region.
        for (ly, gy) in (start_by..end_by).enumerate() {
            // Snapshot prev_row BEFORE borrowing the current row mutably.
            // The borrow checker won't let us hold a shared ref to row gy-1
            // while we mutably borrow row gy from the same Vec<Vec<i32>>,
            // so we copy the prev row into a local Vec each iteration.
            // Cost: O(region_width) per row × region_height = same big-O as
            // the pass itself; under any plausible DC group size (≤256 blocks)
            // this is trivial relative to the f32 quantize work.
            let prev_row: Option<Vec<i32>> = if gy > start_by {
                Some(quant_dc[c][gy - 1][start_bx..end_bx].to_vec())
            } else {
                None
            };

            // Pre-extract a copy of the Y row (when c != Y) for CFL
            // subtraction. By the time we walk X (c=0) or B (c=2), the
            // Y channel has already been requantized for the whole image.
            let y_row: Option<Vec<i32>> = if c != C_Y {
                Some(quant_dc[C_Y][gy][start_bx..end_bx].to_vec())
            } else {
                None
            };

            // Borrow the current row mutably for writeback.
            let row = &mut quant_dc[c][gy];

            for (lx, gx) in (start_bx..end_bx).enumerate() {
                let float_v = float_dc[c][gy * xsize_blocks + gx];
                let svalue_base = if c == C_Y {
                    float_v * inv_factor
                } else {
                    let y_val = y_row.as_ref().unwrap()[lx] as f32;
                    if let Some(y_factor) = reference_y_factor {
                        let cfl_factor = if c == 2 { 1.0 } else { 0.0 };
                        (float_v - y_val * (y_factor * cfl_factor)) * inv_factor
                    } else {
                        float_v * inv_factor - y_val * dc_cfl_factor
                    }
                };

                // WP prediction over already-updated `row[..lx]`.
                let qrow_so_far = &row[start_bx..start_bx + lx + 1]; // includes slot at lx (uninitialised, fine — predict only reads x-1)
                let pred = dc_predict(
                    qrow_so_far,
                    prev_row.as_deref(),
                    lx,
                    ly,
                    region_width,
                    wp_state.as_mut(),
                );

                let q = quantize_wp_one_presvalued(svalue_base, pred);

                // Store the quantized DC directly as i32 — matching libjxl,
                // which keeps DC in i32 storage and only clamps at bitstream
                // encode time. `quant_dc` was i16 until issue #18: at fine
                // butteraugli distances (d ≲ 0.02) the DC inv_factor grows
                // large enough that legitimate DC coefficients exceed
                // ±32767, and the old `as i16` saturation corrupted the
                // low-frequency image. i32 removes the saturation; the DC
                // token encoder (`dc_coding::write_dc_tokens_region`) already
                // packs i32 residuals, so in-range DC stays byte-identical.
                let qv = q;
                if qv != row[gx] {
                    changed += 1;
                }
                row[gx] = qv;

                // Update WP error state for this position (mirrors libjxl
                // line 1660 `wp_state.UpdateErrors(quant_row[x], x, y, xsize)`).
                if let Some(state) = &mut wp_state {
                    state.update_errors(qv, lx, ly, region_width);
                }
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
        let mut quant_dc: [Vec<Vec<i32>>; 3] = [
            vec![vec![0i32; xsize]; ysize],
            vec![vec![0i32; xsize]; ysize],
            vec![vec![0i32; xsize]; ysize],
        ];
        let float_dc: [Vec<f32>; 3] = [
            vec![0.0; xsize * ysize],
            (0..xsize * ysize).map(|i| i as f32).collect(),
            vec![0.0; xsize * ysize],
        ];
        let _changed = requantize_dc_group(
            &mut quant_dc,
            &float_dc,
            xsize,
            0,
            0,
            xsize,
            ysize,
            1.0,
            1,
            false,
            None,
        );
        // Just check we didn't panic and Y channel got non-zero values
        // somewhere (input was 0..16, inv_factor = INV_DC_QUANT[1] * 1.0 * 2).
        let y_nonzero = quant_dc[C_Y].iter().flatten().any(|&v| v != 0);
        assert!(
            y_nonzero,
            "Y channel should have some non-zero quantized values"
        );
    }

    #[test]
    fn gradient_dc_matches_unmodified_libjxl_object() {
        let fixture = include_str!("quantize_gradient_reference.txt");
        let mut cases = 0;
        let mut separates_weighted = false;
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let fields: Vec<_> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 4);
            let width: usize = fields[0].parse().unwrap();
            let height: usize = fields[1].parse().unwrap();
            let inputs: Vec<_> = fields[2]
                .split(',')
                .map(|bits| f32::from_bits(u32::from_str_radix(bits, 16).unwrap()))
                .collect();
            let expected: Vec<i32> = fields[3]
                .split(',')
                .map(|value| value.parse().unwrap())
                .collect();
            assert_eq!(inputs.len(), width * height);
            assert_eq!(expected.len(), inputs.len());
            for offset in [0, 1] {
                let stride = width + 2 * offset;
                let rows = height + 2 * offset;
                let mut quantized = core::array::from_fn(|_| vec![vec![123456; stride]; rows]);
                let mut floats = core::array::from_fn(|_| vec![0.0; stride * rows]);
                for y in 0..height {
                    for x in 0..width {
                        floats[C_Y][(y + offset) * stride + x + offset] =
                            inputs[y * width + x] / INV_DC_QUANT[C_Y];
                    }
                }
                requantize_dc_group(
                    &mut quantized,
                    &floats,
                    stride,
                    offset,
                    offset,
                    offset + width,
                    offset + height,
                    1.0,
                    0,
                    true,
                    None,
                );
                for y in 0..height {
                    assert_eq!(
                        &quantized[C_Y][y + offset][offset..offset + width],
                        &expected[y * width..(y + 1) * width],
                        "{width}x{height} offset {offset}, row {y}"
                    );
                }
                let mut weighted = quantized.clone();
                requantize_dc_group(
                    &mut weighted,
                    &floats,
                    stride,
                    offset,
                    offset,
                    offset + width,
                    offset + height,
                    1.0,
                    0,
                    false,
                    None,
                );
                separates_weighted |= weighted[C_Y] != quantized[C_Y];
                for channel in &quantized {
                    for (y, row) in channel.iter().enumerate() {
                        for (x, &value) in row.iter().enumerate() {
                            if x < offset
                                || x >= offset + width
                                || y < offset
                                || y >= offset + height
                            {
                                assert_eq!(value, 123456, "outside DC region {x},{y}");
                            }
                        }
                    }
                }
            }
            cases += 1;
        }
        assert_eq!(cases, 6);
        assert!(
            separates_weighted,
            "oracle inputs must distinguish the two predictors"
        );
    }
    #[test]
    fn dc_cfl_quantization_matches_reference_operation_order() {
        use crate::vardct::{encoder::VarDctEncoder, frame::DistanceParams};
        let mut encoder = VarDctEncoder::new(1.0);
        encoder.resolved_improvements.dc_encode_libjxl_parity = true;
        let fixture = include_str!("quantize_cfl_reference.txt");
        let mut cases = 0;
        let mut separating_cases = 0;
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let f: Vec<_> = line.split_whitespace().collect();
            let global_scale: f32 = f[0].parse().unwrap();
            let quant_dc: f32 = f[1].parse().unwrap();
            let precision: u8 = f[2].parse().unwrap();
            let bits = |i: usize| u32::from_str_radix(f[i], 16).unwrap();
            let y = f32::from_bits(bits(3));
            let b = f32::from_bits(bits(4));
            let expected_y: i32 = f[5].parse().unwrap();
            let expected_b: i32 = f[6].parse().unwrap();
            let legacy_b: i32 = f[7].parse().unwrap();
            let scale = global_scale / 65536.0;
            let scale_dc = scale * quant_dc;
            let mul = (1u32 << precision) as f32;
            let mut params = DistanceParams::compute(1.0);
            params.global_scale = global_scale as i32;
            params.quant_dc = quant_dc as i32;
            params.scale = scale;
            params.inv_scale = 1.0 / scale;
            params.scale_dc = scale_dc;
            params.extra_dc_precision = precision;
            let y_factor = encoder.reference_dc_y_factor(&params).unwrap();
            assert_eq!(y_factor.to_bits(), bits(9), "Y step: {line}");
            assert_eq!((INV_DC_QUANT[C_Y] * scale_dc * mul).to_bits(), bits(8));
            separating_cases += usize::from(expected_b != legacy_b);
            for gradient in [false, true] {
                for offset in [0, 2] {
                    let stride = 1 + 2 * offset;
                    for reference in [false, true] {
                        let mut quantized =
                            core::array::from_fn(|_| vec![vec![123456; stride]; stride]);
                        let mut floats = core::array::from_fn(|_| vec![f32::NAN; stride * stride]);
                        floats[0][offset * stride + offset] = 0.0;
                        floats[C_Y][offset * stride + offset] = y;
                        floats[2][offset * stride + offset] = b;
                        requantize_dc_group(
                            &mut quantized,
                            &floats,
                            stride,
                            offset,
                            offset,
                            offset + 1,
                            offset + 1,
                            scale_dc,
                            precision,
                            gradient,
                            reference.then_some(y_factor),
                        );
                        assert_eq!(quantized[C_Y][offset][offset], expected_y, "Y: {line}");
                        assert_eq!(
                            quantized[2][offset][offset],
                            if reference { expected_b } else { legacy_b },
                            "reference={reference}, gradient={gradient}: {line}"
                        );
                        for channel in &quantized {
                            for (row, values) in channel.iter().enumerate() {
                                for (col, &value) in values.iter().enumerate() {
                                    if row != offset || col != offset {
                                        assert_eq!(value, 123456);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            cases += 1;
        }
        assert_eq!(cases, 10);
        assert_eq!(separating_cases, 5);
    }
}
