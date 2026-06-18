// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Spline encoding for JPEG XL.
//!
//! Splines are parametric Gaussian-blurred curves overlaid additively onto
//! decoded images. They efficiently encode thin features (power lines,
//! horizons, etc.) that VarDCT handles poorly. The encoder quantizes
//! splines, subtracts them from XYB, and encodes the residual via VarDCT.
//! The decoder adds splines back after VarDCT reconstruction.

use core::f32::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

use super::common::pack_signed;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::encode::{
    build_entropy_code_ans_with_options, write_entropy_code_ans, write_tokens_ans,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;

/// Round f32 → usize, clamped to `[0, cap]`. Panics on non-finite input.
///
/// Spline rendering operates on Catmull-Rom-interpolated control points
/// and arc-length parameterization — both finite by construction on
/// finite input. Non-finite here is always an upstream bug.
#[inline]
fn finite_round_to_usize(v: f32, cap: usize) -> usize {
    assert!(
        v.is_finite(),
        "splines::finite_round_to_usize: non-finite input {v} \
         (upstream spline parameter should be finite — check Catmull-Rom \
         interpolation / arc-length parameterization)"
    );
    let r = v.round();
    if r <= 0.0 {
        0
    } else if r >= cap as f32 {
        cap
    } else {
        r as usize
    }
}

/// Round f32 → i64, clamped to `[lo, hi]`. Panics on non-finite input.
#[inline]
fn finite_round_to_i64(v: f32, lo: i64, hi: i64) -> i64 {
    assert!(
        v.is_finite(),
        "splines::finite_round_to_i64: non-finite input {v}"
    );
    let r = v.round();
    if r <= lo as f32 {
        lo
    } else if r >= hi as f32 {
        hi
    } else {
        r as i64
    }
}

// ── Public types ────────────────────────────────────────────────────────────

/// A control point on a spline curve.
#[derive(Clone, Copy, Debug, Default)]
pub struct SplinePoint {
    /// X coordinate in image space.
    pub x: f32,
    /// Y coordinate in image space.
    pub y: f32,
}

impl SplinePoint {
    /// Create a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn abs(&self) -> f32 {
        self.x.hypot(self.y)
    }
}

impl core::ops::Add for SplinePoint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl core::ops::Sub for SplinePoint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl core::ops::Mul<f32> for SplinePoint {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl core::ops::Div<f32> for SplinePoint {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        Self {
            x: self.x * inv,
            y: self.y * inv,
        }
    }
}

/// A spline with control points, color DCT coefficients, and sigma DCT.
///
/// Control points define the curve path. The 32-element DCT arrays define
/// how color intensity and Gaussian width vary along the curve.
#[derive(Clone, Debug)]
pub struct Spline {
    /// Control points of the spline (at least 1).
    pub control_points: Vec<SplinePoint>,
    /// Color DCT coefficients: `[channel][coeff]` for X, Y, B channels.
    pub color_dct: [[f32; 32]; 3],
    /// Sigma (Gaussian width) DCT coefficients.
    pub sigma_dct: [f32; 32],
}

// ── Internal types ──────────────────────────────────────────────────────────

/// Quantized spline (delta-of-deltas control points, integer DCT coefficients).
struct QuantizedSpline {
    /// Double-delta-encoded control points (excluding the starting point).
    control_points: Vec<(i64, i64)>,
    /// Quantized color DCT: `[channel][coeff]`.
    color_dct: [[i32; 32]; 3],
    /// Quantized sigma DCT.
    sigma_dct: [i32; 32],
}

/// A single rendered segment of a spline (one sample point along the curve).
#[derive(Clone, Copy, Debug, Default)]
struct SplineSegment {
    center_x: f32,
    center_y: f32,
    maximum_distance: f32,
    inv_sigma: f32,
    sigma_over_4_times_intensity: f32,
    color: [f32; 3],
}

/// Fully prepared spline data ready for subtraction/addition and encoding.
pub(crate) struct SplinesData {
    /// Quantization adjustment parameter.
    quantization_adjustment: i32,
    /// Original splines (for encoding).
    splines: Vec<Spline>,
    /// Quantized splines (for bitstream encoding).
    quantized: Vec<QuantizedSpline>,
    /// Rendered segments for pixel operations.
    segments: Vec<SplineSegment>,
    /// Indices into `segments` sorted by y coordinate.
    segment_indices: Vec<usize>,
    /// Prefix-sum index: `segment_y_start[y]` is the start index in
    /// `segment_indices` for row y. Length = image_height + 1.
    segment_y_start: Vec<usize>,
}

// ── Constants ───────────────────────────────────────────────────────────────

/// Channel weights for quantization: [X, Y, B, sigma].
pub(crate) const CHANNEL_WEIGHT: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];

/// Number of entropy contexts for spline encoding.
pub(crate) const NUM_SPLINE_CONTEXTS: usize = 6;

/// Target rendering distance between sample points along the curve.
pub(crate) const DESIRED_RENDERING_DISTANCE: f32 = 1.0;

/// 1 / (2 * sqrt(2)), used in Gaussian splatting.
pub(crate) const ONE_OVER_2S2: f32 = 0.353_553_38;

/// Exponent for maximum_distance computation (fast mode, matches jxl-rs default).
pub(crate) const DISTANCE_EXP: f32 = 3.0;

/// Number of sub-points per Catmull-Rom segment.
pub(crate) const NUM_POINTS_PER_SEGMENT: usize = 16;

// ── Fast math ───────────────────────────────────────────────────────────────

/// Fast error function approximation (max error ~6e-4).
/// Ported from jxl-rs `fast_math.rs`.
#[inline]
fn fast_erf(x: f32) -> f32 {
    let absx = x.abs();
    let d1 = absx * 7.77394369e-02 + 2.05260015e-04;
    let d2 = d1 * absx + 2.32120216e-01;
    let d3 = d2 * absx + 2.77820801e-01;
    let d4 = d3 * absx + 1.0;
    let d5 = d4 * d4;
    let inv = 1.0 / d5;
    (-inv * inv + 1.0).copysign(x)
}

/// Fast cosine approximation (max error ~1e-4).
/// Ported from jxl-rs `fast_math.rs`.
#[inline]
fn fast_cos(x: f32) -> f32 {
    let pi2 = PI * 2.0;
    let pi2_inv = 0.5 / PI;
    let npi2 = (x * pi2_inv).floor() * pi2;
    let xmodpi2 = x - npi2;
    let x_pi = xmodpi2.min(pi2 - xmodpi2);
    let above_pihalf = x_pi >= PI / 2.0;
    let x_pihalf = if above_pihalf { PI - x_pi } else { x_pi };
    let xs = x_pihalf * 0.25;
    let x2 = xs * xs;
    let x4 = x2 * x2;
    let cosx_prescaling = x4 * 0.06960438 + (x2 * -0.84087373 + 1.68179268);
    let cosx_scale1 = cosx_prescaling * cosx_prescaling - SQRT_2;
    let cosx_scale2 = cosx_scale1 * cosx_scale1 - 1.0;
    if above_pihalf {
        -cosx_scale2
    } else {
        cosx_scale2
    }
}

// ── Continuous IDCT ─────────────────────────────────────────────────────────

/// Precomputed cosines for continuous IDCT at a given t value.
/// Computed once per sample point and reused for all 4 DCT evaluations.
struct PrecomputedCosines([f32; 32]);

impl PrecomputedCosines {
    #[inline]
    fn new(t: f32) -> Self {
        let tandhalf = t + 0.5;
        Self(core::array::from_fn(|i| {
            fast_cos(PI / 32.0 * i as f32 * tandhalf)
        }))
    }
}

/// Evaluate continuous IDCT with precomputed cosines.
#[inline]
fn continuous_idct(dct: &[f32; 32], precomputed: &PrecomputedCosines) -> f32 {
    dct.iter()
        .zip(precomputed.0.iter())
        .map(|(&c, &cos)| c * cos)
        .sum::<f32>()
        * SQRT_2
}

// ── Catmull-Rom interpolation ───────────────────────────────────────────────

/// Centripetal Catmull-Rom spline interpolation.
/// Ported from libjxl `splines.cc:294-336` / jxl-rs `spline.rs`.
fn draw_centripetal_catmull_rom(points: &[SplinePoint]) -> Vec<SplinePoint> {
    if points.is_empty() {
        return vec![];
    }
    if points.len() == 1 {
        return vec![points[0]];
    }

    // Extend endpoints by reflection.
    let first_extra = points[0] + (points[0] - points[1]);
    let last_extra =
        points[points.len() - 1] + (points[points.len() - 1] - points[points.len() - 2]);

    let extended: Vec<SplinePoint> = core::iter::once(first_extra)
        .chain(points.iter().copied())
        .chain(core::iter::once(last_extra))
        .collect();

    // Compute centripetal distances between consecutive extended points.
    let mut dists = Vec::with_capacity(extended.len());
    for i in 0..extended.len() - 1 {
        dists.push((extended[i + 1] - extended[i]).abs().sqrt());
    }
    // dists[i] = sqrt(|extended[i+1] - extended[i]|), length = extended.len() - 1

    let num_windows = extended.len() - 3; // = points.len() - 1
    let mut result = Vec::with_capacity(num_windows * NUM_POINTS_PER_SEGMENT + 1);

    for w in 0..num_windows {
        // Window: extended[w], extended[w+1], extended[w+2], extended[w+3]
        // Distances: dists[w], dists[w+1], dists[w+2]
        let p = [
            extended[w],
            extended[w + 1],
            extended[w + 2],
            extended[w + 3],
        ];
        let d = [dists[w], dists[w + 1], dists[w + 2]];

        let mut t = [0.0f32; 4];
        t[1] = t[0] + d[0];
        t[2] = t[1] + d[1];
        t[3] = t[2] + d[2];

        // First point of this segment
        result.push(p[1]);

        for i in 1..NUM_POINTS_PER_SEGMENT {
            let tt = d[0] + (i as f32 / NUM_POINTS_PER_SEGMENT as f32) * d[1];

            // Three-level interpolation
            let mut a = [SplinePoint::default(); 3];
            for k in 0..3 {
                a[k] = p[k] + (p[k + 1] - p[k]) * ((tt - t[k]) / d[k]);
            }
            let mut b = [SplinePoint::default(); 2];
            for k in 0..2 {
                b[k] = a[k] + (a[k + 1] - a[k]) * ((tt - t[k]) / (d[k] + d[k + 1]));
            }
            let point = b[0] + (b[1] - b[0]) * ((tt - t[1]) / d[1]);
            result.push(point);
        }
    }
    // Add the final point
    result.push(points[points.len() - 1]);
    result
}

// ── Equal-distance resampling ───────────────────────────────────────────────

/// Walk curve at uniform intervals, collecting (point, multiplier) pairs.
/// Ported from libjxl `splines.cc:344-375` / jxl-rs `spline.rs`.
fn for_each_equally_spaced_point(
    points: &[SplinePoint],
    desired_distance: f32,
) -> Vec<(SplinePoint, f32)> {
    if points.is_empty() {
        return vec![];
    }
    let mut result = Vec::new();
    result.push((points[0], desired_distance));
    if points.len() == 1 {
        return result;
    }

    let mut accumulated_distance = 0.0f32;
    for index in 0..points.len() - 1 {
        let mut current = points[index];
        let next = points[index + 1];
        let segment = next - current;
        let segment_length = segment.abs();
        if segment_length < 1e-10 {
            continue;
        }
        let unit_step = segment / segment_length;
        if accumulated_distance + segment_length >= desired_distance {
            current = current + unit_step * (desired_distance - accumulated_distance);
            result.push((current, desired_distance));
            accumulated_distance -= desired_distance;
        }
        accumulated_distance += segment_length;
        while accumulated_distance >= desired_distance {
            current = current + unit_step * desired_distance;
            result.push((current, desired_distance));
            accumulated_distance -= desired_distance;
        }
    }
    result.push((points[points.len() - 1], accumulated_distance));
    result
}

// ── Quantization ────────────────────────────────────────────────────────────

/// Compute inverse adjusted quantization factor.
fn inv_adjusted_quant(adjustment: i32) -> f32 {
    if adjustment >= 0 {
        1.0 / (1.0 + 0.125 * adjustment as f32)
    } else {
        1.0 - 0.125 * adjustment as f32
    }
}

/// Compute adjusted quantization factor (inverse of inv_adjusted_quant).
fn adjusted_quant(adjustment: i32) -> f32 {
    if adjustment >= 0 {
        1.0 + 0.125 * adjustment as f32
    } else {
        1.0 / (1.0 - 0.125 * adjustment as f32)
    }
}

impl QuantizedSpline {
    /// Quantize a spline. Ported from libjxl `QuantizedSpline::Create()`.
    ///
    /// Process order: Y (channel 1) first for CfL decorrelation, then X (0), B (2).
    fn from_spline(
        spline: &Spline,
        quantization_adjustment: i32,
        y_to_x: f32,
        y_to_b: f32,
    ) -> Self {
        let quant = adjusted_quant(quantization_adjustment);

        // Quantize control points: delta-of-deltas encoding.
        // Starting point is encoded separately; here we encode the second-order
        // differences of the remaining points.
        let mut control_points = Vec::new();
        if spline.control_points.len() > 1 {
            let pts = &spline.control_points;
            let mut prev_delta_x = 0i64;
            let mut prev_delta_y = 0i64;
            let mut prev_x = pts[0].x.round() as i64;
            let mut prev_y = pts[0].y.round() as i64;

            for p in pts.iter().skip(1) {
                let cur_x = p.x.round() as i64;
                let cur_y = p.y.round() as i64;
                let delta_x = cur_x - prev_x;
                let delta_y = cur_y - prev_y;
                let dd_x = delta_x - prev_delta_x;
                let dd_y = delta_y - prev_delta_y;
                control_points.push((dd_x, dd_y));
                prev_delta_x = delta_x;
                prev_delta_y = delta_y;
                prev_x = cur_x;
                prev_y = cur_y;
            }
        }

        // Quantize Y channel first (channel 1) for CfL reference.
        let mut quantized_color = [[0i32; 32]; 3];
        for (i, qc) in quantized_color[1].iter_mut().enumerate() {
            let dct_factor = if i == 0 { SQRT_2 } else { 1.0 };
            *qc = (spline.color_dct[1][i] * dct_factor * quant / CHANNEL_WEIGHT[1]).round() as i32;
        }

        // Dequantize Y for CfL decorrelation reference.
        let inv_quant = inv_adjusted_quant(quantization_adjustment);
        let mut restored_y = [0.0f32; 32];
        for (i, ry) in restored_y.iter_mut().enumerate() {
            let inv_dct_factor = if i == 0 { FRAC_1_SQRT_2 } else { 1.0 };
            *ry = quantized_color[1][i] as f32 * inv_dct_factor * CHANNEL_WEIGHT[1] * inv_quant;
        }

        // Quantize X (channel 0) and B (channel 2) with CfL decorrelation.
        for c in [0, 2] {
            let cfl_factor = if c == 0 { y_to_x } else { y_to_b };
            for (i, qc) in quantized_color[c].iter_mut().enumerate() {
                let dct_factor = if i == 0 { SQRT_2 } else { 1.0 };
                let decorrelated = spline.color_dct[c][i] - cfl_factor * restored_y[i];
                *qc = (decorrelated * dct_factor * quant / CHANNEL_WEIGHT[c]).round() as i32;
            }
        }

        // Quantize sigma DCT.
        let mut quantized_sigma = [0i32; 32];
        for (i, qs) in quantized_sigma.iter_mut().enumerate() {
            let dct_factor = if i == 0 { SQRT_2 } else { 1.0 };
            *qs = (spline.sigma_dct[i] * dct_factor * quant / CHANNEL_WEIGHT[3]).round() as i32;
        }

        Self {
            control_points,
            color_dct: quantized_color,
            sigma_dct: quantized_sigma,
        }
    }

    /// Dequantize back to floating-point spline (for rendering).
    /// This matches what the decoder will reconstruct.
    fn dequantize(
        &self,
        starting_point: SplinePoint,
        quantization_adjustment: i32,
        y_to_x: f32,
        y_to_b: f32,
    ) -> DequantizedSpline {
        let inv_quant = inv_adjusted_quant(quantization_adjustment);

        // Reconstruct control points from delta-of-deltas.
        let mut control_points = Vec::with_capacity(self.control_points.len() + 1);
        let sp_x = starting_point.x.round() as i64;
        let sp_y = starting_point.y.round() as i64;
        control_points.push(SplinePoint::new(sp_x as f32, sp_y as f32));

        let mut cur_x = sp_x;
        let mut cur_y = sp_y;
        let mut delta_x = 0i64;
        let mut delta_y = 0i64;
        for &(dd_x, dd_y) in &self.control_points {
            delta_x += dd_x;
            delta_y += dd_y;
            cur_x += delta_x;
            cur_y += delta_y;
            control_points.push(SplinePoint::new(cur_x as f32, cur_y as f32));
        }

        // Dequantize color DCTs.
        let mut color_dct = [[0.0f32; 32]; 3];
        for (c, (out_ch, in_ch)) in color_dct.iter_mut().zip(self.color_dct.iter()).enumerate() {
            for (i, (out, &inp)) in out_ch.iter_mut().zip(in_ch.iter()).enumerate() {
                let inv_dct_factor = if i == 0 { FRAC_1_SQRT_2 } else { 1.0 };
                *out = inp as f32 * inv_dct_factor * CHANNEL_WEIGHT[c] * inv_quant;
            }
        }
        // Apply CfL: add Y contribution to X and B.
        // Index-based loop required: simultaneous mutable access to channels 0/2
        // while reading channel 1 of the same array.
        #[allow(clippy::needless_range_loop)]
        for i in 0..32 {
            color_dct[0][i] += y_to_x * color_dct[1][i];
            color_dct[2][i] += y_to_b * color_dct[1][i];
        }

        // Dequantize sigma DCT.
        let mut sigma_dct = [0.0f32; 32];
        for (i, (out, &inp)) in sigma_dct.iter_mut().zip(self.sigma_dct.iter()).enumerate() {
            let inv_dct_factor = if i == 0 { FRAC_1_SQRT_2 } else { 1.0 };
            *out = inp as f32 * inv_dct_factor * CHANNEL_WEIGHT[3] * inv_quant;
        }

        DequantizedSpline {
            control_points,
            color_dct,
            sigma_dct,
        }
    }
}

/// Intermediate dequantized spline used for rendering.
struct DequantizedSpline {
    control_points: Vec<SplinePoint>,
    color_dct: [[f32; 32]; 3],
    sigma_dct: [f32; 32],
}

// ── Segment generation ──────────────────────────────────────────────────────

/// Create a segment from a sample point along the spline.
fn make_segment(
    center: &SplinePoint,
    intensity: f32,
    color: [f32; 3],
    sigma: f32,
) -> Option<SplineSegment> {
    if sigma.is_infinite() || sigma == 0.0 || (1.0 / sigma).is_infinite() || intensity.is_infinite()
    {
        return None;
    }
    let max_color = [0.01, color[0].abs(), color[1].abs(), color[2].abs()]
        .iter()
        .copied()
        .map(|c| (c * intensity).abs())
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    let max_distance =
        (-2.0 * sigma * sigma * (0.1f32.ln() * DISTANCE_EXP - max_color.ln())).sqrt();
    if max_distance.is_nan() || max_distance <= 0.0 {
        return None;
    }
    Some(SplineSegment {
        center_x: center.x,
        center_y: center.y,
        color,
        inv_sigma: 1.0 / sigma,
        sigma_over_4_times_intensity: 0.25 * sigma * intensity,
        maximum_distance: max_distance,
    })
}

/// Generate segments from a dequantized spline.
fn generate_segments(spline: &DequantizedSpline) -> Vec<SplineSegment> {
    let intermediate = draw_centripetal_catmull_rom(&spline.control_points);
    let points_to_draw = for_each_equally_spaced_point(&intermediate, DESIRED_RENDERING_DISTANCE);
    if points_to_draw.len() < 2 {
        return vec![];
    }

    let length = (points_to_draw.len() as isize - 2) as f32 * DESIRED_RENDERING_DISTANCE
        + points_to_draw[points_to_draw.len() - 1].1;
    if length <= 0.0 {
        return vec![];
    }

    let inv_length = 1.0 / length;
    let mut segments = Vec::new();

    for (point_index, (point, multiplier)) in points_to_draw.iter().enumerate() {
        let progress = (point_index as f32 * DESIRED_RENDERING_DISTANCE * inv_length).min(1.0);
        let t = 31.0 * progress;

        let precomputed = PrecomputedCosines::new(t);
        let mut color = [0.0f32; 3];
        for (c, coeffs) in spline.color_dct.iter().enumerate() {
            color[c] = continuous_idct(coeffs, &precomputed);
        }
        let sigma = continuous_idct(&spline.sigma_dct, &precomputed);

        if let Some(seg) = make_segment(point, *multiplier, color, sigma) {
            segments.push(seg);
        }
    }
    segments
}

// ── Gaussian splatting (add/subtract) ───────────────────────────────────────

/// Apply a segment to a single pixel.
#[inline]
fn apply_segment_at(
    planes: &mut [Vec<f32>; 3],
    stride: usize,
    x: usize,
    y: usize,
    segment: &SplineSegment,
    add: bool,
) {
    let dx = x as f32 - segment.center_x;
    let dy = y as f32 - segment.center_y;
    let distance = (dx * dx + dy * dy).sqrt();
    let one_dim = fast_erf((distance * 0.5 + ONE_OVER_2S2) * segment.inv_sigma)
        - fast_erf((distance * 0.5 - ONE_OVER_2S2) * segment.inv_sigma);
    let local_intensity = segment.sigma_over_4_times_intensity * one_dim * one_dim;

    let idx = y * stride + x;
    let sign = if add { 1.0 } else { -1.0 };
    for (plane, &color) in planes.iter_mut().zip(segment.color.iter()) {
        plane[idx] += sign * color * local_intensity;
    }
}

/// Apply all spline segments to XYB planes (add or subtract).
fn apply_splines(
    planes: &mut [Vec<f32>; 3],
    stride: usize,
    width: usize,
    height: usize,
    data: &SplinesData,
    add: bool,
) {
    for y in 0..height {
        let first = data.segment_y_start[y];
        let last = data.segment_y_start[y + 1];
        for seg_idx_pos in first..last {
            let segment = &data.segments[data.segment_indices[seg_idx_pos]];
            let x0 = finite_round_to_usize(segment.center_x - segment.maximum_distance, width);
            let x1_raw = finite_round_to_usize(segment.center_x + segment.maximum_distance, width);
            let x1 = x1_raw.saturating_add(1).min(width);
            for x in x0..x1 {
                apply_segment_at(planes, stride, x, y, segment, add);
            }
        }
    }
}

/// Subtract splines from XYB planes (encoder side: before VarDCT).
pub(crate) fn subtract_splines(
    planes: &mut [Vec<f32>; 3],
    stride: usize,
    width: usize,
    height: usize,
    data: &SplinesData,
) {
    apply_splines(planes, stride, width, height, data, false);
}

/// Add splines to XYB planes (reconstruction: after VarDCT decode, for butteraugli).
#[allow(dead_code)]
pub(crate) fn add_splines(
    planes: &mut [Vec<f32>; 3],
    stride: usize,
    width: usize,
    height: usize,
    data: &SplinesData,
) {
    apply_splines(planes, stride, width, height, data, true);
}

// ── Auto-detection ──────────────────────────────────────────────────────────

/// Tunable detection thresholds. All values were picked to ride on the
/// conservative side of the cost-benefit gate so the default-config
/// hash-lock invariants stay intact on photo content.
///
/// W44-211: `pub(crate)` to allow `crate::tuning::splines` to re-export.
pub(crate) mod detect_params {
    /// Minimum gradient magnitude (in linearised Y intensity) for a
    /// pixel to be a ridge candidate. The Y channel after XYB
    /// conversion has roughly `[-0.5, +0.5]` range; 0.15 picks up only
    /// high-contrast features (power lines on sky, glyph edges on UI).
    pub const MIN_GRAD_MAG: f32 = 0.15;

    /// Minimum Hessian eigenvalue ratio (`λ_large / λ_small`) for a
    /// pixel to be classified as line-like rather than blob/corner.
    /// Real ridges have one eigenvalue ≫ the other; isotropic
    /// gradients (corners, noise) have a small ratio.
    pub const MIN_EIG_RATIO: f32 = 5.0;

    /// Minimum polyline length (in ridge pixels). Splines shorter than
    /// this never carry their per-spline overhead (control points +
    /// 32-coeff DCTs).
    pub const MIN_POLYLINE_LEN: usize = 32;

    /// Maximum polyline length we trace before splitting. Longer
    /// ridges are sliced into multiple splines because the
    /// 32-coefficient DCT fit loses fidelity past a few hundred
    /// pixels. (libjxl `splines.h:53` caps each spline's segment
    /// budget at 64 sampled points; our 1024 here is the
    /// _polyline-trace_ cap before split — actual per-spline samples
    /// are decimated by `for_each_equally_spaced_point` to
    /// roughly that budget.)
    pub const MAX_POLYLINE_LEN: usize = 1024;

    /// Target number of Catmull-Rom control points per spline.
    /// libjxl's `enc_splines.cc` notes 4–8 is the sweet spot for the
    /// 32-coeff color/sigma DCT to remain a good fit.
    pub const TARGET_CONTROL_POINTS: usize = 8;

    /// Maximum number of splines per image. Beyond this the encoder
    /// overhead dominates and the cost gate would reject them anyway.
    pub const MAX_SPLINES: usize = 64;

    /// Sigma (Gaussian width, in pixels) we initialize each spline
    /// with when the per-CP Hessian fit isn't available (zero-curvature
    /// region, fall-back path). Real thin features are ~1.0–1.5 pixels
    /// wide post-XYB after the implicit gaborish lowpass.
    pub const INIT_SIGMA: f32 = 1.0;

    /// Lower clamp on per-control-point Hessian-derived sigma. A super-
    /// thin (1-px-wide) ridge would push sigma well below 1.0 and the
    /// resulting Gaussian would be too narrow for the renderer's
    /// `maximum_distance` (which is the [-3σ, +3σ] envelope at
    /// `DISTANCE_EXP=3`). Clamp keeps the rendered radius ≥ 1 px.
    pub const SIGMA_MIN: f32 = 0.6;

    /// Upper clamp on per-control-point Hessian-derived sigma. Beyond
    /// this the spline is too wide for thin-feature rendering and the
    /// VarDCT residual is probably cheaper. Anchored at 4 px — wider
    /// features want patches, not splines.
    pub const SIGMA_MAX: f32 = 4.0;

    /// Cost-benefit safety margin for the chunk-3 trial-encode gate.
    /// Realised VarDCT bytes saved (from an energy-reduction proxy)
    /// must exceed measured spline encoded bytes by this factor for
    /// the spline to be admitted. Mirrors the 2× margin used by
    /// patches (`vardct/patches.rs:281`).
    pub const COST_BENEFIT_MARGIN: f32 = 2.0;

    /// Chunk-6 false-positive suppression on textured photo content.
    /// Minimum spline-bbox span (max of bbox-width / bbox-height,
    /// in pixels) as a fraction of the larger image dimension. A
    /// candidate whose bbox doesn't span at least this fraction of
    /// the long image dimension is rejected.
    ///
    /// Rationale (see `benchmarks/auto_splines_bench_2026-05-17_chunk6_fp.tsv`):
    /// every false-positive admit in a 42-image CID22 sweep at d=1.0
    /// e7/e8 had a bbox span of at most ~510 pixels on 512×512 photos
    /// (a span/long_dim ratio of <= 0.99). Genuine thin features
    /// (power lines, hair strands, image-spanning ridges) almost
    /// always cross most of the image in their dominant orientation
    /// — they cover the image's long dimension nearly edge-to-edge.
    /// At a 1.0 threshold the gate admits only candidates that span
    /// the FULL image long dimension, which preserves the existing
    /// chunk-3 power-line-on-textured-background test image
    /// (1024×256, wire spans ≈ 1018 pixels = 0.99 of image width =
    /// 1024) while rejecting every observed textured-photo FP.
    ///
    /// 1.0 is a strict cutoff — a candidate must span at least the
    /// image's long dimension. This is intentional: post chunk-5 the
    /// detector's only known "real" win case is full-image-spanning
    /// power-line synthetics, and a shorter ridge fragment is more
    /// efficiently handled by VarDCT than by emitting a Gaussian
    /// splat over a sub-image bbox where the splat-vs-VarDCT cost
    /// gap narrows.
    pub const MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM: f32 = 1.0;
}

/// Automatic spline detection from XYB pixel planes.
///
/// **Chunk 2: full detection pipeline** —
///   1. Sobel gradient magnitude on Y plane (post-patches, pre-gaborish).
///   2. Non-max suppression along the gradient direction (1D NMS).
///   3. Hessian eigenvalue ratio test (`λ1/λ2 > MIN_EIG_RATIO`) to
///      keep line-like ridges and reject corners / isotropic noise.
///   4. 8-connected polyline tracing along ridge pixels, biased to
///      continue in the previous trace direction (suppresses branches).
///   5. Centripetal-Catmull-Rom-aware subsampling to ~8 control points.
///   6. Per-curve 32-coefficient DCT fit for color (X/Y/B) and sigma.
///   7. Per-spline cost-benefit gate: estimated VarDCT bytes saved
///      (modulated by distance, same shape as `patches::is_cost_effective`)
///      must exceed estimated spline encoded bytes by
///      `COST_BENEFIT_MARGIN`.
///
/// libjxl's `FindSplines` (`lib/jxl/enc_splines.cc:104-107`) is itself
/// a stub. Upstream the encoder relies on the (also-stub-in-libjxl)
/// manual API for splines, and `cparams.custom_splines.HasAny()` is the
/// only path that puts a non-empty `Splines` into the bitstream. So
/// this chunk is novel territory above libjxl's reference encoder.
///
/// # Parameters
/// - `xyb_x` / `xyb_y` / `xyb_b`: post-XYB-conversion planes
///   (post-patches-subtract, pre-gaborish — mirroring libjxl's
///   `FindSplines(*opsin)` call site after
///   `PatchDictionaryEncoder::SubtractFrom`)
/// - `width` / `height`: image dimensions in pixels
/// - `stride`: row stride of the plane buffers
///
/// # Returns
/// `Vec<Spline>` of detected splines. May be empty (e.g. photo content,
/// or any content where the cost gate rejects every candidate). When
/// non-empty, every entry is guaranteed to have at least
/// `MIN_POLYLINE_LEN` pixels of supporting ridge evidence and to have
/// passed the cost-benefit gate.
///
/// Convenience wrapper around [`find_splines_at_distance`] with the
/// cost-benefit gate scaled for `distance = 1.0` — used by unit tests
/// that don't have a real `distance` parameter to pass.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_splines(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Vec<Spline> {
    find_splines_at_distance(xyb_x, xyb_y, xyb_b, width, height, stride, 1.0)
}

/// Empirically-derived median-mask1x1 threshold above which an image is
/// classified as "screenshot-like" content. Mirrors the value used by
/// the GPU encoder's AFV cost-grid gate (W7-3,
/// `jxl-encoder-gpu/src/lossy_encoder.rs::SCREENSHOT_MEDIAN_MASK_THRESHOLD`)
/// where the same threshold was validated on 16 CLIC photos (all
/// median ≤ 87) and 10 gb82-sc screenshots (9 of 10 median = 100.01).
///
/// Used by [`looks_like_screenshot`] to skip the auto-splines pipeline
/// on content where the bbox-area-linear energy-drop proxy structurally
/// over-claims VarDCT byte savings on long bright ridges (table
/// borders, wallpaper edges) — see chunk-4 bench notes in
/// `effort.rs::auto_splines_default` for the underlying regression.
pub(crate) const SCREENSHOT_MEDIAN_MASK_THRESHOLD: f32 = 95.0;

/// Content discriminator: classify an XYB Y-plane as "screenshot-like"
/// based on the median per-8x8-block mean of [`compute_mask1x1`]. The
/// 1x1 Laplacian masking field is large in flat regions and small in
/// textured regions, so the per-block-mean median is a cheap proxy for
/// "what fraction of this image is flat".
///
/// Returns `true` when the median per-block mean exceeds
/// [`SCREENSHOT_MEDIAN_MASK_THRESHOLD`]. Callers should skip the
/// auto-splines pipeline on `true` to avoid the long-bright-ridge
/// over-claim documented in chunk-4
/// (`benchmarks/auto_splines_bench_2026-05-17_chunk4.tsv`).
///
/// Cost: one [`compute_mask1x1`] pass (~SIMD per-pixel + 5x5 blur over
/// `width * height` f32s) plus a per-block-mean reduction and a partial
/// sort over the `(width / 8) * (height / 8)` block-mean vector. Cheap
/// relative to the full splines pipeline that follows (Sobel + NMS +
/// Hessian + polyline tracing + per-spline DCT fit + trial-encode gate),
/// so paying it up front is a net win whenever the gate fires.
pub(crate) fn looks_like_screenshot(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<bool> {
    if width < 8 || height < 8 {
        // Below one full 8x8 block — there's no meaningful median to take,
        // and `find_splines_at_distance` already short-circuits at
        // `width < 16 || height < 16` anyway. Treat as "not screenshot".
        return Ok(false);
    }

    // Re-pack the (possibly strided) Y plane into a contiguous buffer
    // because `compute_mask1x1` assumes `stride == width`. The mask1x1
    // computation expects row-major contiguous input — see callers in
    // `vardct/encoder.rs:1218` which pass `padded_width` directly as
    // both width AND stride. The strided repack is dimension-driven, so
    // route it through the runtime fallible-alloc policy; byte-identical
    // when infallible.
    let y_contig: alloc::vec::Vec<f32> = if stride == width {
        // SAFETY: stride == width means buffer is already contiguous.
        // Slice up to width*height to drop any trailing slop.
        xyb_y[..width * height].to_vec()
    } else {
        let mut buf = crate::budget::vec_with_capacity_fallible(
            budget.is_some_and(|b| b.is_fallible()),
            width * height,
        )?;
        for y in 0..height {
            let row_start = y * stride;
            buf.extend_from_slice(&xyb_y[row_start..row_start + width]);
        }
        buf
    };

    let mask1x1 = super::adaptive_quant::compute_mask1x1(&y_contig, width, height);

    // Per-block mean of mask1x1, matching the GPU encoder's
    // `compute_block_mask_means` (jxl-encoder-gpu/src/lossy_encoder.rs).
    let blocks_per_row = width / 8;
    let blocks_per_col = height / 8;
    if blocks_per_row == 0 || blocks_per_col == 0 {
        return Ok(false);
    }
    let n_blocks = blocks_per_row * blocks_per_col;
    let mut block_means: alloc::vec::Vec<f32> = alloc::vec::Vec::with_capacity(n_blocks);
    for by in 0..blocks_per_col {
        for bx in 0..blocks_per_row {
            let mut sum: f64 = 0.0;
            let mut count: usize = 0;
            for dy in 0..8 {
                let y = by * 8 + dy;
                if y >= height {
                    break;
                }
                for dx in 0..8 {
                    let x = bx * 8 + dx;
                    if x >= width {
                        break;
                    }
                    sum += mask1x1[y * width + x] as f64;
                    count += 1;
                }
            }
            let mean = if count > 0 {
                (sum / count as f64) as f32
            } else {
                1.0
            };
            block_means.push(mean);
        }
    }

    block_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let median = block_means[block_means.len() / 2];
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "looks_like_screenshot: w={width} h={height} n_blocks={} median(mask1x1)={median:.3} threshold={SCREENSHOT_MEDIAN_MASK_THRESHOLD} → {}",
        block_means.len(),
        if median > SCREENSHOT_MEDIAN_MASK_THRESHOLD {
            "SCREENSHOT (skip splines)"
        } else {
            "non-screenshot"
        }
    );
    Ok(median > SCREENSHOT_MEDIAN_MASK_THRESHOLD)
}

/// Distance-aware variant of [`find_splines`]. Internal entry used by
/// the encoder when it knows the configured `distance` parameter so the
/// cost-benefit gate can scale VarDCT-bytes-saved estimates correctly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_splines_at_distance(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    distance: f32,
) -> Vec<Spline> {
    // Skip tiny images: spline overhead dominates pixel savings.
    if width < 16 || height < 16 {
        return Vec::new();
    }

    // (1) Sobel gradient + magnitude.
    let (gx, gy, gmag) = sobel_xy(xyb_y, width, height, stride);

    // (2) Non-max suppression along the gradient direction.
    let nms_mask = nms_along_gradient(&gx, &gy, &gmag, width, height);

    // (3) Hessian eigenvalue ratio test — confirm each NMS pixel is
    //     line-like, not blob-like.
    let ridge_mask = hessian_ridge_filter(xyb_y, &nms_mask, width, height, stride);

    // Early-out: photos typically have very few ridge pixels after
    // these three filters.
    let ridge_pixel_count = ridge_mask.iter().filter(|&&b| b).count();
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "find_splines: nms={} ridge={} (w={} h={})",
        nms_mask.iter().filter(|&&b| b).count(),
        ridge_pixel_count,
        width,
        height
    );
    if ridge_pixel_count < detect_params::MIN_POLYLINE_LEN {
        return Vec::new();
    }

    // (4) 8-connected polyline tracing.
    let polylines = trace_polylines(&ridge_mask, &gmag, width, height);
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "find_splines: polylines={} max_len={}",
        polylines.len(),
        polylines.iter().map(|p| p.len()).max().unwrap_or(0)
    );
    if polylines.is_empty() {
        return Vec::new();
    }

    // (5)+(6) Subsample to Catmull-Rom control points + fit per-channel DCTs.
    let mut candidates: Vec<Spline> = polylines
        .iter()
        .take(detect_params::MAX_SPLINES * 2)
        .filter_map(|poly| {
            let cps = subsample_polyline(poly, detect_params::TARGET_CONTROL_POINTS);
            if cps.len() < 2 {
                return None;
            }
            let (color_dct, sigma_dct) =
                fit_curve_dcts(&cps, xyb_x, xyb_y, xyb_b, width, height, stride);
            Some(Spline {
                control_points: cps,
                color_dct,
                sigma_dct,
            })
        })
        .collect();

    if candidates.is_empty() {
        return Vec::new();
    }

    // (6.5) Deduplicate near-coincident candidates (chunk-3 addition).
    // The 8-connected polyline tracer happily emits both sides of a
    // long ridge as separate seeds, so we end up with two splines
    // tracing essentially the same line (offset by 1-2 pixels). The
    // second one buys nothing — its energy reduction is ~0 because the
    // first already subtracted the ridge intensity — but it still
    // costs ~95 bytes to encode. Drop any candidate whose start AND
    // end control points are both within `DUP_RADIUS_PX` of an
    // existing kept candidate.
    const DUP_RADIUS_PX: f32 = 4.0;
    let mut deduped: Vec<Spline> = Vec::with_capacity(candidates.len());
    'outer: for cand in &candidates {
        let cs = cand.control_points.first().copied();
        let ce = cand.control_points.last().copied();
        for kept in &deduped {
            let ks = kept.control_points.first().copied();
            let ke = kept.control_points.last().copied();
            if let (Some(cs), Some(ce), Some(ks), Some(ke)) = (cs, ce, ks, ke) {
                let near_start = (cs - ks).abs() < DUP_RADIUS_PX;
                let near_end = (ce - ke).abs() < DUP_RADIUS_PX;
                let near_start_rev = (cs - ke).abs() < DUP_RADIUS_PX;
                let near_end_rev = (ce - ks).abs() < DUP_RADIUS_PX;
                if (near_start && near_end) || (near_start_rev && near_end_rev) {
                    continue 'outer;
                }
            }
        }
        deduped.push(cand.clone());
    }
    candidates = deduped;

    // (7) Cost-benefit gate (chunk 3): trial-encode each candidate's
    //     splines section and compare the realised encoded bytes
    //     against a measured-residual-energy savings proxy. Splines
    //     that don't clear the `COST_BENEFIT_MARGIN` factor get dropped.
    #[cfg(feature = "debug-tokens")]
    let before = candidates.len();
    candidates.retain(|s| {
        spline_passes_trial_encode_gate(s, xyb_x, xyb_y, xyb_b, width, height, stride, distance)
    });
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "find_splines: candidates_before_gate={} after={}",
        before,
        candidates.len()
    );

    if candidates.len() > detect_params::MAX_SPLINES {
        candidates.truncate(detect_params::MAX_SPLINES);
    }

    candidates
}

// ── (1) Sobel 3x3 ───────────────────────────────────────────────────────────

/// 3x3 Sobel filter on a single plane. Returns `(gx, gy, magnitude)`,
/// each of length `width * height` (contiguous, NO stride — internal
/// layout). Border pixels are zero.
fn sobel_xy(
    plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = width * height;
    let mut gx = vec![0.0f32; n];
    let mut gy = vec![0.0f32; n];
    let mut gmag = vec![0.0f32; n];

    if width < 3 || height < 3 {
        return (gx, gy, gmag);
    }

    // Sobel kernels:
    //   Gx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
    //   Gy = [[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]
    for y in 1..height - 1 {
        let row_m = (y - 1) * stride;
        let row_0 = y * stride;
        let row_p = (y + 1) * stride;
        let out_row = y * width;
        for x in 1..width - 1 {
            let nw = plane[row_m + x - 1];
            let n_ = plane[row_m + x];
            let ne = plane[row_m + x + 1];
            let w_ = plane[row_0 + x - 1];
            let e_ = plane[row_0 + x + 1];
            let sw = plane[row_p + x - 1];
            let s_ = plane[row_p + x];
            let se = plane[row_p + x + 1];

            let gxv = (ne + 2.0 * e_ + se) - (nw + 2.0 * w_ + sw);
            let gyv = (sw + 2.0 * s_ + se) - (nw + 2.0 * n_ + ne);
            let idx = out_row + x;
            gx[idx] = gxv;
            gy[idx] = gyv;
            gmag[idx] = (gxv * gxv + gyv * gyv).sqrt();
        }
    }

    (gx, gy, gmag)
}

// ── (2) Non-max suppression ─────────────────────────────────────────────────

/// Canny-style 1D non-max suppression along the gradient direction.
/// Output is a boolean mask (`width * height`, contiguous) marking
/// pixels that are local maxima of `|∇|` along their gradient
/// orientation AND above `MIN_GRAD_MAG`.
fn nms_along_gradient(
    gx: &[f32],
    gy: &[f32],
    gmag: &[f32],
    width: usize,
    height: usize,
) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    if width < 3 || height < 3 {
        return mask;
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let m = gmag[idx];
            if m < detect_params::MIN_GRAD_MAG {
                continue;
            }
            let gxv = gx[idx];
            let gyv = gy[idx];
            // Quantize gradient direction to one of 4 bins (0°, 45°, 90°, 135°).
            let abs_gx = gxv.abs();
            let abs_gy = gyv.abs();
            let (a_idx, b_idx) = if abs_gx > 2.0 * abs_gy {
                // ~horizontal gradient → compare with left/right
                (idx - 1, idx + 1)
            } else if abs_gy > 2.0 * abs_gx {
                // ~vertical gradient → compare with up/down
                (idx - width, idx + width)
            } else if gxv.signum() == gyv.signum() {
                // diagonal NE-SW
                (idx - width - 1, idx + width + 1)
            } else {
                // diagonal NW-SE
                (idx - width + 1, idx + width - 1)
            };
            if m >= gmag[a_idx] && m >= gmag[b_idx] {
                mask[idx] = true;
            }
        }
    }
    mask
}

// ── (3) Hessian eigenvalue ratio test ───────────────────────────────────────

/// Ridge filter via the 2x2 image-Hessian eigenvalue ratio.
///
/// For each NMS pixel, compute `Ixx`, `Iyy`, `Ixy` by finite
/// differences on the input plane, then derive the two eigenvalues
///     `λ_{1,2} = (tr/2) ± sqrt((tr/2)^2 - det)`,
///     `det = Ixx*Iyy - Ixy^2`, `tr = Ixx + Iyy`.
/// A "line-like" feature has one large `|λ|` and one small `|λ|`.
/// We require `|λ_large| / |λ_small| >= MIN_EIG_RATIO` AND
/// `|λ_large| >= MIN_GRAD_MAG` (so a tiny gradient doesn't sneak in
/// because its tiny noise neighbour is smaller).
///
/// This is the same family as Frangi's vesselness — we just keep it
/// to the eigenvalue magnitudes since orientation is already locked
/// in by NMS, and we don't need scale-space here (post-XYB the
/// gaborish lowpass acts as a single-scale prefilter).
fn hessian_ridge_filter(
    plane: &[f32],
    nms_mask: &[bool],
    width: usize,
    height: usize,
    stride: usize,
) -> Vec<bool> {
    let mut out = vec![false; width * height];
    if width < 3 || height < 3 {
        return out;
    }
    let r = detect_params::MIN_EIG_RATIO;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mask_idx = y * width + x;
            if !nms_mask[mask_idx] {
                continue;
            }
            let center = plane[y * stride + x];
            let left = plane[y * stride + x - 1];
            let right = plane[y * stride + x + 1];
            let up = plane[(y - 1) * stride + x];
            let down = plane[(y + 1) * stride + x];
            let nw = plane[(y - 1) * stride + x - 1];
            let ne = plane[(y - 1) * stride + x + 1];
            let sw = plane[(y + 1) * stride + x - 1];
            let se = plane[(y + 1) * stride + x + 1];

            let ixx = right - 2.0 * center + left;
            let iyy = down - 2.0 * center + up;
            let ixy = 0.25 * (se - sw - ne + nw);

            // 2x2 symmetric Hessian eigenvalues.
            let tr = ixx + iyy;
            let det = ixx * iyy - ixy * ixy;
            let disc = (tr * 0.5).powi(2) - det;
            if disc < 0.0 {
                continue; // complex eigenvalues → not a ridge
            }
            let sqrt_disc = disc.sqrt();
            let l1 = tr * 0.5 + sqrt_disc;
            let l2 = tr * 0.5 - sqrt_disc;
            let abs1 = l1.abs();
            let abs2 = l2.abs();
            let (big, small) = if abs1 >= abs2 {
                (abs1, abs2)
            } else {
                (abs2, abs1)
            };
            // Require the larger eigenvalue's magnitude to be at least
            // MIN_GRAD_MAG and the small/big ratio to be small enough
            // that the ridge is truly line-like.
            if big < detect_params::MIN_GRAD_MAG {
                continue;
            }
            if small * r > big {
                continue; // ratio too low → blob/corner, not ridge
            }
            out[mask_idx] = true;
        }
    }
    out
}

// ── (4) 8-connected polyline tracing ────────────────────────────────────────

/// Trace polylines along ridge pixels using a direction-biased
/// 8-connected walk. Each pixel is visited at most once.
fn trace_polylines(
    ridge_mask: &[bool],
    gmag: &[f32],
    width: usize,
    height: usize,
) -> Vec<Vec<(i32, i32)>> {
    let mut visited = vec![false; width * height];
    // Seed order: strongest |∇| first → strong ridges win priority over
    // weaker parallel ones, and never have to fight for pixels.
    let mut seeds: Vec<(usize, f32)> = (0..width * height)
        .filter(|&i| ridge_mask[i] && !visited[i])
        .map(|i| (i, gmag[i]))
        .collect();
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut polylines: Vec<Vec<(i32, i32)>> = Vec::new();

    // 8-neighbor offsets in scan order.
    const NB: [(i32, i32); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];

    for (seed_idx, _) in seeds {
        if visited[seed_idx] {
            continue;
        }
        let sx = (seed_idx % width) as i32;
        let sy = (seed_idx / width) as i32;

        // Walk forward from seed.
        let forward = walk_ridge(
            ridge_mask,
            &mut visited,
            gmag,
            width,
            height,
            sx,
            sy,
            None,
            &NB,
        );
        // Walk backward (away from seed in the opposite initial direction)
        // by starting again, but on the already-existing reverse list.
        // Simpler: re-seed and walk in the OPPOSITE initial direction.
        // We accomplish this by re-marking the seed unvisited then walking
        // with a `forbidden_first` direction equal to the forward's first
        // direction, so the second walk picks the other side.
        if visited[seed_idx] {
            // we need to keep the seed marked so the loop terminates, but
            // we still want the chain to extend backwards. Solution: pop
            // the seed off the forward chain temporarily by reversing
            // through it manually using the un-set-then-walk trick below.
        }
        let mut chain: Vec<(i32, i32)> = forward;

        // Reverse-walk attempt: temporarily un-visit the seed, then
        // re-walk in the direction opposite to forward[1] (if any).
        if chain.len() >= 2 {
            let (x1, y1) = chain[1];
            let (sx2, sy2) = chain[0];
            let forbidden = (x1 - sx2, y1 - sy2);
            visited[seed_idx] = false;
            let backward = walk_ridge(
                ridge_mask,
                &mut visited,
                gmag,
                width,
                height,
                sx,
                sy,
                Some(forbidden),
                &NB,
            );
            // backward starts at seed and walks the OTHER direction.
            // Prepend it (reversed, minus duplicated seed) to chain.
            if backward.len() > 1 {
                let mut full = Vec::with_capacity(backward.len() + chain.len() - 1);
                for &p in backward.iter().skip(1).rev() {
                    full.push(p);
                }
                full.extend(chain.iter().copied());
                chain = full;
            }
        }

        if chain.len() >= detect_params::MIN_POLYLINE_LEN {
            // Split overly-long chains so the per-spline DCT fit stays
            // accurate.
            for piece in chain.chunks(detect_params::MAX_POLYLINE_LEN) {
                if piece.len() >= detect_params::MIN_POLYLINE_LEN {
                    polylines.push(piece.to_vec());
                }
            }
            if polylines.len() >= detect_params::MAX_SPLINES * 2 {
                break;
            }
        }
    }

    polylines
}

/// Walk a ridge starting at `(sx, sy)`. Greedy 8-connected — at each
/// step, pick the unvisited neighbor with the greatest gradient
/// magnitude, biased toward continuing in the previous direction.
/// Returns the visited path (including the seed).
#[allow(clippy::too_many_arguments)]
fn walk_ridge(
    ridge_mask: &[bool],
    visited: &mut [bool],
    gmag: &[f32],
    width: usize,
    height: usize,
    sx: i32,
    sy: i32,
    forbidden_first: Option<(i32, i32)>,
    nb: &[(i32, i32); 8],
) -> Vec<(i32, i32)> {
    let mut path = Vec::new();
    let mut x = sx;
    let mut y = sy;
    let mut prev_dir: Option<(i32, i32)> = None;
    for step in 0..detect_params::MAX_POLYLINE_LEN {
        let idx = (y as usize) * width + (x as usize);
        if visited[idx] {
            break;
        }
        visited[idx] = true;
        path.push((x, y));

        // Pick the strongest unvisited ridge neighbor.
        let mut best: Option<((i32, i32), f32)> = None;
        for &(dx, dy) in nb {
            // First step: respect `forbidden_first` so the seed can be
            // walked in BOTH directions across two calls.
            if step == 0 && forbidden_first.is_some_and(|fd| (dx, dy) == fd) {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let nidx = (ny as usize) * width + (nx as usize);
            if visited[nidx] || !ridge_mask[nidx] {
                continue;
            }
            // Score: |∇| magnitude * directional-continuation bias.
            let mut score = gmag[nidx];
            if let Some((pdx, pdy)) = prev_dir {
                let dot = (dx * pdx + dy * pdy) as f32;
                // Strongly favour continuing forward (dot=+2 for same dir,
                // +1 for adjacent, 0 for perpendicular, negative for reverse).
                score *= 1.0 + 0.25 * dot;
            }
            match best {
                None => best = Some(((dx, dy), score)),
                Some((_, bs)) if score > bs => best = Some(((dx, dy), score)),
                _ => {}
            }
        }
        match best {
            Some(((dx, dy), _)) => {
                x += dx;
                y += dy;
                prev_dir = Some((dx, dy));
            }
            None => break,
        }
    }
    path
}

// ── (5) Polyline → Catmull-Rom control points ───────────────────────────────

/// Subsample a polyline to ~`target` control points via arc-length
/// uniform spacing. The first and last polyline pixels always become
/// the first and last control points.
fn subsample_polyline(polyline: &[(i32, i32)], target: usize) -> Vec<SplinePoint> {
    if polyline.len() < 2 {
        return polyline
            .iter()
            .map(|&(x, y)| SplinePoint::new(x as f32, y as f32))
            .collect();
    }
    let target = target.max(2);
    if polyline.len() <= target {
        return polyline
            .iter()
            .map(|&(x, y)| SplinePoint::new(x as f32, y as f32))
            .collect();
    }
    // Compute cumulative arc length.
    let mut cum = Vec::with_capacity(polyline.len());
    cum.push(0.0f32);
    for w in polyline.windows(2) {
        let dx = (w[1].0 - w[0].0) as f32;
        let dy = (w[1].1 - w[0].1) as f32;
        cum.push(cum.last().unwrap() + (dx * dx + dy * dy).sqrt());
    }
    let total = *cum.last().unwrap();
    if total < 1e-3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(target);
    out.push(SplinePoint::new(polyline[0].0 as f32, polyline[0].1 as f32));
    for k in 1..target - 1 {
        let target_dist = total * (k as f32) / ((target - 1) as f32);
        // Linear search (target is small — usually 8).
        let mut i = 1;
        while i < cum.len() && cum[i] < target_dist {
            i += 1;
        }
        if i >= cum.len() {
            i = cum.len() - 1;
        }
        let (x0, y0) = polyline[i - 1];
        let (x1, y1) = polyline[i];
        let seg_len = cum[i] - cum[i - 1];
        let t = if seg_len > 1e-6 {
            (target_dist - cum[i - 1]) / seg_len
        } else {
            0.0
        };
        let x = (x0 as f32) + t * ((x1 - x0) as f32);
        let y = (y0 as f32) + t * ((y1 - y0) as f32);
        out.push(SplinePoint::new(x, y));
    }
    let last = polyline[polyline.len() - 1];
    out.push(SplinePoint::new(last.0 as f32, last.1 as f32));
    out
}

// ── (6) Per-curve color/sigma DCT fit ───────────────────────────────────────

/// Bilinear lookup of one channel at fractional coordinates `(fx, fy)`.
/// Clamps to image bounds (so callers can pass arc-length samples
/// without worrying about edge handling).
///
/// Chunk-3 fidelity refinement (was nearest-pixel in chunk 2): along a
/// thin ridge the sample line rarely sits on integer pixels, so
/// nearest-pixel sampling under-represents the ridge intensity by up to
/// 50% (one pixel left/right of the true peak). Bilinear catches the
/// true sub-pixel intensity, which the IDCT then reproduces — closing
/// the residual gap that left chunk 2 net-cost on the 1024×256
/// power-line synthetic.
#[inline]
fn bilinear_sample(
    plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    fx: f32,
    fy: f32,
) -> f32 {
    let cx = fx.clamp(0.0, width.saturating_sub(1) as f32);
    let cy = fy.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;
    let v00 = plane[y0 * stride + x0];
    let v10 = plane[y0 * stride + x1];
    let v01 = plane[y1 * stride + x0];
    let v11 = plane[y1 * stride + x1];
    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;
    v0 * (1.0 - ty) + v1 * ty
}

/// Compute the 2x2 image-Hessian at fractional position `(fx, fy)` on
/// the supplied plane and return `|λ_small|` (magnitude of the smaller
/// eigenvalue of the Hessian). For a ridge pixel this is small (low
/// curvature *along* the ridge); for a thin sharp ridge `|λ_large|` is
/// large *across* the ridge.
///
/// Used by chunk-3 to derive per-control-point sigma: ridge width
/// scales roughly as `1 / sqrt(|λ_large|)` for a Gaussian-shaped
/// profile (second derivative of a Gaussian is the Gaussian times a
/// quadratic). We return `|λ_small|` rather than `|λ_large|` because
/// the chunk-3 sigma fit uses the SAME mapping (`width ∝ 1/sqrt`)
/// regardless of which axis we treat as "across" — `|λ_large|` is the
/// across-ridge curvature, which is what governs visible thinness.
#[inline]
fn hessian_lambda_large(
    plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    fx: f32,
    fy: f32,
) -> f32 {
    let cx = fx.clamp(1.0, (width.saturating_sub(2)) as f32);
    let cy = fy.clamp(1.0, (height.saturating_sub(2)) as f32);
    // Round to nearest integer pixel for finite-difference Hessian; the
    // sigma fit is downstream-DCTed and inherently smooth so subpixel
    // jitter in the Hessian itself doesn't hurt.
    let x = cx.round() as usize;
    let y = cy.round() as usize;
    let center = plane[y * stride + x];
    let left = plane[y * stride + x - 1];
    let right = plane[y * stride + x + 1];
    let up = plane[(y - 1) * stride + x];
    let down = plane[(y + 1) * stride + x];
    let nw = plane[(y - 1) * stride + x - 1];
    let ne = plane[(y - 1) * stride + x + 1];
    let sw = plane[(y + 1) * stride + x - 1];
    let se = plane[(y + 1) * stride + x + 1];
    let ixx = right - 2.0 * center + left;
    let iyy = down - 2.0 * center + up;
    let ixy = 0.25 * (se - sw - ne + nw);
    let tr = ixx + iyy;
    let det = ixx * iyy - ixy * ixy;
    let disc = ((tr * 0.5).powi(2) - det).max(0.0);
    let sqrt_disc = disc.sqrt();
    let l1 = (tr * 0.5 + sqrt_disc).abs();
    let l2 = (tr * 0.5 - sqrt_disc).abs();
    l1.max(l2)
}

/// Sample the three XYB channels along the dequantized-rendering of
/// the control points (one sample per arc-length unit), fit each
/// channel as a 32-coefficient DCT-II over the curve length, then do
/// the same for `sigma` derived per-sample from the Hessian's larger
/// eigenvalue (chunk-3: was DC-only sigma in chunk 2).
///
/// The DCT-II we use here is the same orthonormal basis that the
/// decoder's continuous-IDCT (`continuous_idct`) consumes — so
/// `IDCT(DCT(samples)) ≈ samples`.
///
/// **Chunk-3 fidelity changes**:
/// - Bilinear (not nearest-pixel) sampling at each arc-length point.
/// - Per-sample sigma from `1 / sqrt(|λ_large|)` of the local Hessian,
///   clamped to `[SIGMA_MIN, SIGMA_MAX]` and DCT-fitted alongside the
///   colour channels. A sharper ridge gets smaller sigma (tighter
///   Gaussian), a softer ridge gets larger sigma.
fn fit_curve_dcts(
    control_points: &[SplinePoint],
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> ([[f32; 32]; 3], [f32; 32]) {
    let mut color_dct = [[0.0f32; 32]; 3];
    let mut sigma_dct = [0.0f32; 32];

    // Walk the polyline at unit spacing and sample the XYB planes.
    let rendered = draw_centripetal_catmull_rom(control_points);
    let samples = for_each_equally_spaced_point(&rendered, DESIRED_RENDERING_DISTANCE);
    if samples.len() < 2 {
        sigma_dct[0] = detect_params::INIT_SIGMA / SQRT_2;
        return (color_dct, sigma_dct);
    }

    let n = samples.len();
    let mut sx = vec![0.0f32; n];
    let mut sy = vec![0.0f32; n];
    let mut sb = vec![0.0f32; n];
    let mut ssigma = vec![0.0f32; n];
    for (i, (p, _)) in samples.iter().enumerate() {
        // (chunk-3) Bilinear colour sampling.
        sx[i] = bilinear_sample(xyb_x, width, height, stride, p.x, p.y);
        sy[i] = bilinear_sample(xyb_y, width, height, stride, p.x, p.y);
        sb[i] = bilinear_sample(xyb_b, width, height, stride, p.x, p.y);
        // (chunk-3) Per-sample sigma from the Y-plane Hessian.
        // Line-width ∝ 1/sqrt(λ_large): the second derivative of a 1D
        // Gaussian of width σ at its peak is `-1/σ²`, so for a unit-
        // amplitude ridge the visible-thinness curvature is roughly
        // `1/σ²`. Inverting: `σ ≈ 1/sqrt(λ_large)`. The renderer's
        // `INIT_SIGMA = 1.0` constant is the photo-noise floor; for
        // very sharp ridges (`λ_large > 1`) the formula returns sub-
        // unit sigma, which we clamp to `SIGMA_MIN` so the renderer's
        // `maximum_distance` halo stays ≥ 1 px.
        let lam = hessian_lambda_large(xyb_y, width, height, stride, p.x, p.y);
        let sigma_raw = if lam > 1e-6 {
            1.0 / lam.sqrt()
        } else {
            detect_params::SIGMA_MAX
        };
        ssigma[i] = sigma_raw.clamp(detect_params::SIGMA_MIN, detect_params::SIGMA_MAX);
    }

    // DCT-II fit: for each k in 0..32,
    //     coeff_k = (1/N) * sum_i sample_i * cos(pi * k * (i + 0.5) / N)
    // The decoder's continuous IDCT scales by SQRT_2 and uses a
    // 0..31 range over [0,N), so we mirror that convention. We use
    // the same fit for all four DCTs (3 colors + sigma).
    let inv_n = 1.0 / n as f32;
    // The k loop indexes four parallel sums; `for (k, ...)` enumerate
    // doesn't compose with that without extra zips.
    #[allow(clippy::needless_range_loop)]
    for k in 0..32 {
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cb = 0.0f32;
        let mut csigma = 0.0f32;
        for i in 0..n {
            let progress = (i as f32 + 0.5) * inv_n; // in [0, 1]
            let t = 31.0 * progress;
            let arg = PI / 32.0 * k as f32 * (t + 0.5);
            let c = fast_cos(arg);
            cx += sx[i] * c;
            cy += sy[i] * c;
            cb += sb[i] * c;
            csigma += ssigma[i] * c;
        }
        // Match the IDCT scaling: IDCT multiplies the sum by SQRT_2,
        // and our DCT-II convention divides by N. We absorb the
        // SQRT_2 here so IDCT recovers the original samples.
        let scale = inv_n / SQRT_2;
        color_dct[0][k] = cx * scale;
        color_dct[1][k] = cy * scale;
        color_dct[2][k] = cb * scale;
        sigma_dct[k] = csigma * scale;
    }

    (color_dct, sigma_dct)
}

// ── (7) Cost-benefit gate ───────────────────────────────────────────────────

/// Chunk-3 cost-benefit gate.
///
/// This replaces chunk-2's purely analytical estimate with a
/// **trial-encode** of the candidate's splines section, mirroring
/// `vardct/patches::trial_encode_ref_frame_bytes`. The encoded byte
/// count is exact (modulo the per-image fixed overhead which we
/// amortize across the whole vec elsewhere).
///
/// For the savings side we measure the **realised XYB residual energy
/// reduction** in the spline's bounding box after applying just this
/// candidate, then map it to bytes via a per-distance constant. The
/// constant `BYTES_PER_ENERGY_UNIT` was anchored empirically: on the
/// 1024x256 power-line synthetic at d=1.0, the chunk-3 detector
/// produces a candidate whose energy reduction predicts ~200 saved
/// bytes (matching the realised VarDCT residual drop on that image).
///
/// The `COST_BENEFIT_MARGIN` (2×) safety factor keeps photo-noise
/// candidates rejected — realistic photo noise produces small, weakly-
/// correlated residual changes, so the energy reduction stays below
/// the encoded-bytes overhead.
#[allow(clippy::too_many_arguments)]
fn spline_passes_trial_encode_gate(
    spline: &Spline,
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    distance: f32,
) -> bool {
    let num_cps = spline.control_points.len();
    if num_cps < 2 {
        return false;
    }

    // ── Trial-encode this single spline into a fresh BitWriter.
    let trial_data = SplinesData::from_splines(
        alloc::vec![spline.clone()],
        0,   // quantization_adjustment
        0.0, // y_to_x (default DC CfL)
        1.0, // y_to_b (default DC CfL)
        width,
        height,
    );
    let mut writer = BitWriter::new();
    let encoded_bytes = if encode_splines_section(&trial_data, &mut writer).is_ok() {
        writer.zero_pad_to_byte();
        writer.bytes_written()
    } else {
        return false; // Encode failure → safe fallback.
    };

    // ── Measure realised XYB residual energy reduction in the bbox.
    // Energy is per-channel sum-of-squares weighted by CHANNEL_WEIGHT
    // (same weights the encoder uses to quantize, so it's a reasonable
    // proxy for "bits per channel" magnitude).
    let bbox = spline_bbox(spline, width, height);

    // ── Chunk-6 false-positive suppression on textured photo content.
    // Reject any candidate whose bbox doesn't span at least
    // `MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM` of the image's long dimension.
    // The L2-energy proxy used below cannot reliably tell a true thin
    // feature (low-baseline bbox + high-contrast ridge) from a sub-
    // image ridge segment riding across textured content (where the
    // splat-subtract drops raw L2 by 5-15% via partial texture
    // averaging without reducing actual VarDCT residual cost). The
    // observed FPs cluster at bbox spans <= 0.99 × image long dim;
    // genuine wins span the full long dim almost edge-to-edge. See
    // `benchmarks/auto_splines_bench_2026-05-17_chunk6_fp.tsv` for the
    // per-bbox distribution and 4 of 42 CID22 photo regressions
    // (+0.05% to +1.19% file size) that this gate closes.
    let bbox_span = (bbox.2 - bbox.0).max(bbox.3 - bbox.1) as f32;
    let image_long_dim = width.max(height) as f32;
    let min_span = detect_params::MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM * image_long_dim;
    if bbox_span < min_span {
        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "spline_passes_trial_encode_gate: cps={} encoded={} bbox=({},{},{},{}) \
             bbox_span={} image_long_dim={} pass=false \
             (chunk6 bbox-span floor {:.3}*long_dim={} not met)",
            num_cps,
            encoded_bytes,
            bbox.0,
            bbox.1,
            bbox.2,
            bbox.3,
            bbox_span as usize,
            image_long_dim as usize,
            detect_params::MIN_BBOX_SPAN_OF_IMAGE_LONG_DIM,
            min_span as usize,
        );
        return false;
    }

    let energy_before = bbox_energy(xyb_x, xyb_y, xyb_b, stride, bbox);
    // Apply the spline subtraction to a local copy of the bbox region.
    let energy_after = bbox_energy_after_subtract(xyb_x, xyb_y, xyb_b, stride, bbox, &trial_data);
    let energy_drop = (energy_before - energy_after).max(0.0);

    // Distance-aware bytes-per-energy mapping. At higher distances the
    // VarDCT residual is already coarsely quantized, so a given energy
    // drop saves fewer bytes.
    //
    // Recalibration (chunk 4, 2026-05-17): the original `50.0` anchor
    // was derived from a stale comment that estimated `energy_drop ≈ 2-4`
    // for the 1024x256 power-line synthetic. The chunk-3 detector
    // actually measures `energy_drop ≈ 500` for the same synthetic, so
    // the realised bytes-per-energy ratio is closer to `0.07-0.15`
    // (138 bytes saved on 4-line synth / 4 splines / ~533 e_drop each
    // ≈ 0.065; per-spline encoded ≈ 39 bytes, so a 2× safety margin
    // demands `bytes_saved_proxy >= 78`, which lands right at the
    // ratio×e_drop threshold).
    //
    // The previous `50.0` value over-claimed savings by ~770× on
    // synthetics (and worse on screenshots, where ridge density is
    // high but DCT8 already captures sharp text edges so the spline
    // overlay does not actually reduce VarDCT bytes — and in fact
    // regresses screenshot encodes by 3-8% at e7/e8/e9).
    //
    // The new anchor `0.20` is the geomean-fit per-spline ratio across
    // the synthetic power-line / multi-line bench plus three CLIC2025
    // photos that the detector falsely admitted under the old constant.
    // It admits the multi-line synthetics (target wins) while rejecting
    // every screenshot/photo spline in the chunk-3 bench. See
    // `benchmarks/auto_splines_bench_2026-05-17_chunk4.tsv`.
    const BYTES_PER_ENERGY_UNIT_AT_D1: f32 = 0.20;
    let bytes_saved_proxy =
        (energy_drop * BYTES_PER_ENERGY_UNIT_AT_D1 / distance.max(1.0)) as usize;

    let pass =
        (bytes_saved_proxy as f32) >= detect_params::COST_BENEFIT_MARGIN * (encoded_bytes as f32);
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "spline_passes_trial_encode_gate: cps={} encoded={} bbox=({},{},{},{}) \
         e_before={:.3} e_after={:.3} e_drop={:.3} bytes_saved_proxy={} pass={}",
        num_cps,
        encoded_bytes,
        bbox.0,
        bbox.1,
        bbox.2,
        bbox.3,
        energy_before,
        energy_after,
        energy_drop,
        bytes_saved_proxy,
        pass
    );
    pass
}

/// Conservative bounding box for a single spline, in image pixels.
/// Returns `(x0, y0, x1, y1)` where `x1`/`y1` are exclusive.
///
/// We use control-point extents plus a `±SIGMA_MAX * DISTANCE_EXP`
/// halo so the Gaussian splat is fully covered. This is the same
/// halo size `apply_splines` uses for per-segment x-extent.
fn spline_bbox(spline: &Spline, width: usize, height: usize) -> (usize, usize, usize, usize) {
    let halo = detect_params::SIGMA_MAX * DISTANCE_EXP;
    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for cp in &spline.control_points {
        min_x = min_x.min(cp.x);
        min_y = min_y.min(cp.y);
        max_x = max_x.max(cp.x);
        max_y = max_y.max(cp.y);
    }
    let x0 = ((min_x - halo).max(0.0) as usize).min(width);
    let y0 = ((min_y - halo).max(0.0) as usize).min(height);
    let x1 = (((max_x + halo).ceil() as usize) + 1).min(width);
    let y1 = (((max_y + halo).ceil() as usize) + 1).min(height);
    (x0, y0, x1, y1)
}

/// Weighted sum-of-squares energy over the bbox region.
/// Weights mirror `CHANNEL_WEIGHT[0..3]` so the proxy aligns with
/// the encoder's quantization-channel sensitivity.
fn bbox_energy(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    bbox: (usize, usize, usize, usize),
) -> f32 {
    let (x0, y0, x1, y1) = bbox;
    let wx = CHANNEL_WEIGHT[0] * CHANNEL_WEIGHT[0];
    let wy = CHANNEL_WEIGHT[1] * CHANNEL_WEIGHT[1];
    let wb = CHANNEL_WEIGHT[2] * CHANNEL_WEIGHT[2];
    let mut e = 0.0f32;
    for y in y0..y1 {
        let row = y * stride;
        for x in x0..x1 {
            let i = row + x;
            let vx = xyb_x[i];
            let vy = xyb_y[i];
            let vb = xyb_b[i];
            e += wx * vx * vx + wy * vy * vy + wb * vb * vb;
        }
    }
    // Normalise by Y-channel weight so the magnitudes are O(1) and
    // the `BYTES_PER_ENERGY_UNIT_AT_D1` constant stays interpretable.
    // (wy is the dominant term; dividing by wy keeps `energy_drop`
    // roughly in the per-pixel-variance scale.)
    e / wy
}

/// `bbox_energy` after applying the spline data's subtraction to a
/// local copy of the bbox region (so the input XYB is left untouched).
fn bbox_energy_after_subtract(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    bbox: (usize, usize, usize, usize),
    data: &SplinesData,
) -> f32 {
    let (x0, y0, x1, y1) = bbox;
    let w = x1.saturating_sub(x0);
    let h = y1.saturating_sub(y0);
    if w == 0 || h == 0 {
        return 0.0;
    }
    // Copy bbox slice into a contiguous local buffer for each channel.
    let mut buf_x = alloc::vec![0.0f32; w * h];
    let mut buf_y = alloc::vec![0.0f32; w * h];
    let mut buf_b = alloc::vec![0.0f32; w * h];
    for ly in 0..h {
        let src_row = (y0 + ly) * stride + x0;
        let dst_row = ly * w;
        buf_x[dst_row..dst_row + w].copy_from_slice(&xyb_x[src_row..src_row + w]);
        buf_y[dst_row..dst_row + w].copy_from_slice(&xyb_y[src_row..src_row + w]);
        buf_b[dst_row..dst_row + w].copy_from_slice(&xyb_b[src_row..src_row + w]);
    }
    // Apply spline segments restricted to the bbox. We can't reuse
    // `apply_splines` directly (it walks `segment_y_start` over the
    // full image height) — but we can walk segments that intersect
    // `[y0, y1)` and call `apply_segment_at` on the local buffers
    // with a shifted (cx, cy).
    let mut planes = [buf_x, buf_y, buf_b];
    for y in y0..y1 {
        let first = data.segment_y_start[y];
        let last = data.segment_y_start[y + 1];
        for seg_idx_pos in first..last {
            let segment = &data.segments[data.segment_indices[seg_idx_pos]];
            let sx0 = finite_round_to_usize(segment.center_x - segment.maximum_distance, x1);
            let sx1_raw = finite_round_to_usize(segment.center_x + segment.maximum_distance, x1);
            let sx1 = sx1_raw.saturating_add(1).min(x1);
            let sx0 = sx0.max(x0);
            // Shift segment center into local-bbox coords.
            let local_seg = SplineSegment {
                center_x: segment.center_x - x0 as f32,
                center_y: segment.center_y - y0 as f32,
                maximum_distance: segment.maximum_distance,
                inv_sigma: segment.inv_sigma,
                sigma_over_4_times_intensity: segment.sigma_over_4_times_intensity,
                color: segment.color,
            };
            for x in sx0..sx1 {
                apply_segment_at(&mut planes, w, x - x0, y - y0, &local_seg, false);
            }
        }
    }
    let wx = CHANNEL_WEIGHT[0] * CHANNEL_WEIGHT[0];
    let wy = CHANNEL_WEIGHT[1] * CHANNEL_WEIGHT[1];
    let wb = CHANNEL_WEIGHT[2] * CHANNEL_WEIGHT[2];
    let mut e = 0.0f32;
    for ((&vx, &vy), &vb) in planes[0]
        .iter()
        .zip(planes[1].iter())
        .zip(planes[2].iter())
        .take(w * h)
    {
        e += wx * vx * vx + wy * vy * vy + wb * vb * vb;
    }
    e / wy
}

// ── SplinesData construction ────────────────────────────────────────────────

impl SplinesData {
    /// Build SplinesData from user-provided splines.
    ///
    /// Quantizes, dequantizes (for pixel-accurate rendering), generates
    /// segments, and builds the y-sorted lookup structure.
    pub(crate) fn from_splines(
        splines: Vec<Spline>,
        quantization_adjustment: i32,
        y_to_x: f32,
        y_to_b: f32,
        _image_width: usize,
        image_height: usize,
    ) -> Self {
        let mut quantized = Vec::with_capacity(splines.len());
        let mut all_segments: Vec<SplineSegment> = Vec::new();
        let mut segments_by_y: Vec<(usize, usize)> = Vec::new(); // (y, segment_index)

        for spline in &splines {
            let qs = QuantizedSpline::from_spline(spline, quantization_adjustment, y_to_x, y_to_b);

            // Dequantize for rendering (matches decoder reconstruction).
            let starting_point = spline.control_points[0];
            let dqs = qs.dequantize(starting_point, quantization_adjustment, y_to_x, y_to_b);

            // Generate segments from the dequantized spline.
            let segs = generate_segments(&dqs);
            let base_idx = all_segments.len();
            for (i, seg) in segs.iter().enumerate() {
                let seg_idx = base_idx + i;
                let y0 = finite_round_to_i64(
                    seg.center_y - seg.maximum_distance,
                    0,
                    image_height as i64,
                );
                let y1_raw = finite_round_to_i64(
                    seg.center_y + seg.maximum_distance,
                    0,
                    image_height as i64,
                );
                let y1 = y1_raw.saturating_add(1).min(image_height as i64);
                for y in y0..y1 {
                    segments_by_y.push((y as usize, seg_idx));
                }
            }
            all_segments.extend(segs);

            quantized.push(qs);
        }

        // Sort by y for efficient row-based rendering.
        segments_by_y.sort_by_key(|&(y, _)| y);

        let mut segment_indices = Vec::with_capacity(segments_by_y.len());
        let mut segment_y_start = vec![0usize; image_height + 1];

        for &(y, idx) in &segments_by_y {
            segment_indices.push(idx);
            if y < image_height {
                segment_y_start[y + 1] += 1;
            }
        }
        // Prefix-sum.
        for y in 0..image_height {
            segment_y_start[y + 1] += segment_y_start[y];
        }

        Self {
            quantization_adjustment,
            splines,
            quantized,
            segments: all_segments,
            segment_indices,
            segment_y_start,
        }
    }
}

// ── Bitstream encoding ──────────────────────────────────────────────────────

/// Encode splines section into LfGlobal.
///
/// Token stream layout (6 contexts):
/// - ctx 2: num_splines - 1
/// - ctx 1: starting positions (first absolute, rest delta-coded via pack_signed)
/// - ctx 0: quantization_adjustment (pack_signed)
/// - Per spline:
///   - ctx 3: num_control_points
///   - ctx 4: control point double-deltas (pack_signed)
///   - ctx 5: DCT coefficients (3×32 color + 32 sigma, pack_signed)
pub(crate) fn encode_splines_section(data: &SplinesData, writer: &mut BitWriter) -> Result<()> {
    let mut tokens = Vec::new();

    let num_splines = data.splines.len();
    // num_splines - 1
    tokens.push(Token::new(2, (num_splines - 1) as u32));

    // Starting positions: first is unsigned absolute, rest are signed deltas.
    let mut last_x = 0i64;
    let mut last_y = 0i64;
    for (i, spline) in data.splines.iter().enumerate() {
        let sp = spline.control_points[0];
        let x = sp.x.round() as i64;
        let y = sp.y.round() as i64;
        if i == 0 {
            tokens.push(Token::new(1, x as u32));
            tokens.push(Token::new(1, y as u32));
        } else {
            let dx = x - last_x;
            let dy = y - last_y;
            tokens.push(Token::new(1, pack_signed(dx as i32)));
            tokens.push(Token::new(1, pack_signed(dy as i32)));
        }
        last_x = x;
        last_y = y;
    }

    // Quantization adjustment.
    tokens.push(Token::new(0, pack_signed(data.quantization_adjustment)));

    // Per-spline data.
    for qs in &data.quantized {
        // num_control_points (double-deltas, not including starting point)
        tokens.push(Token::new(3, qs.control_points.len() as u32));

        // Control point double-deltas.
        for &(dd_x, dd_y) in &qs.control_points {
            tokens.push(Token::new(4, pack_signed(dd_x as i32)));
            tokens.push(Token::new(4, pack_signed(dd_y as i32)));
        }

        // Color DCT coefficients (3 channels × 32).
        for channel in &qs.color_dct {
            for &coeff in channel {
                tokens.push(Token::new(5, pack_signed(coeff)));
            }
        }

        // Sigma DCT coefficients (32).
        for &coeff in &qs.sigma_dct {
            tokens.push(Token::new(5, pack_signed(coeff)));
        }
    }

    // Write LZ77 disabled flag.
    writer.write(1, 0)?; // lz77_enabled = false

    // Build and write ANS entropy code, then tokens.
    let code =
        build_entropy_code_ans_with_options(&tokens, NUM_SPLINE_CONTEXTS, false, true, None, None);
    write_entropy_code_ans(&code, writer)?;
    write_tokens_ans(&tokens, &code, None, writer)?;

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_erf_accuracy() {
        // Golden data from Wikipedia error function table.
        let golden = [
            (0.0, 0.0),
            (0.1, 0.112_462_92),
            (0.2, 0.222_702_6),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (1.5, 0.966_105_16),
            (2.0, 0.995_322_3),
            (2.5, 0.999_593),
            (3.0, 0.999_977_9),
        ];
        for (x, expected) in golden {
            let got = fast_erf(x);
            assert!(
                (got - expected).abs() < 6e-4,
                "fast_erf({x}) = {got}, expected {expected}"
            );
            let got_neg = fast_erf(-x);
            assert!(
                (got_neg - (-expected)).abs() < 6e-4,
                "fast_erf(-{x}) = {got_neg}, expected {}",
                -expected
            );
        }
    }

    #[test]
    fn test_fast_cos_accuracy() {
        for i in 0..100 {
            let x = i as f32 / 100.0 * (5.0 * PI) - (2.5 * PI);
            let got = fast_cos(x);
            let expected = x.cos();
            assert!(
                (got - expected).abs() < 1e-4,
                "fast_cos({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_continuous_idct_values() {
        // Simple test: DC-only signal should be constant along the spline.
        let mut dct = [0.0f32; 32];
        dct[0] = 1.0;
        for t_idx in 0..32 {
            let t = t_idx as f32;
            let pc = PrecomputedCosines::new(t);
            let val = continuous_idct(&dct, &pc);
            // DC coefficient * SQRT_2 * cos(0) = 1.0 * SQRT_2 * 1.0 = SQRT_2
            // But dct[0]*cos(0*(t+0.5)*pi/32) = 1.0*1.0 = 1.0, times SQRT_2 = SQRT_2
            assert!(
                (val - SQRT_2).abs() < 0.01,
                "DC-only IDCT at t={t} = {val}, expected ~{SQRT_2}"
            );
        }
    }

    #[test]
    fn test_catmull_rom_basic() {
        // Two control points should produce a straight line with interpolation.
        let points = vec![SplinePoint::new(0.0, 0.0), SplinePoint::new(10.0, 0.0)];
        let interpolated = draw_centripetal_catmull_rom(&points);
        assert!(interpolated.len() > 2, "should produce intermediate points");
        // First and last should match input.
        assert!((interpolated[0].x - 0.0).abs() < 0.01);
        assert!((interpolated[0].y - 0.0).abs() < 0.01);
        let last = interpolated[interpolated.len() - 1];
        assert!((last.x - 10.0).abs() < 0.01);
        assert!((last.y - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_quantize_roundtrip() {
        // Create a simple spline with small DCT values, quantize, dequantize.
        let spline = Spline {
            control_points: vec![SplinePoint::new(10.0, 10.0), SplinePoint::new(50.0, 50.0)],
            color_dct: {
                let mut dct = [[0.0f32; 32]; 3];
                dct[1][0] = 0.5; // Y DC
                dct[0][0] = 0.1; // X DC
                dct[2][0] = 0.2; // B DC
                dct
            },
            sigma_dct: {
                let mut s = [0.0f32; 32];
                s[0] = 2.0;
                s
            },
        };

        let adj = 0;
        let y_to_x = 0.0;
        let y_to_b = 1.13;

        let qs = QuantizedSpline::from_spline(&spline, adj, y_to_x, y_to_b);
        let dqs = qs.dequantize(spline.control_points[0], adj, y_to_x, y_to_b);

        // Control points should roundtrip exactly (integer-rounded).
        assert_eq!(dqs.control_points.len(), 2);
        assert!((dqs.control_points[0].x - 10.0).abs() < 1.0);
        assert!((dqs.control_points[1].x - 50.0).abs() < 1.0);

        // Sigma should be close (within quantization error).
        assert!(
            (dqs.sigma_dct[0] - spline.sigma_dct[0]).abs() < 0.5,
            "sigma DC roundtrip: got {}, expected {}",
            dqs.sigma_dct[0],
            spline.sigma_dct[0]
        );
    }

    #[test]
    fn test_double_delta_encoding() {
        // Verify that delta-of-deltas encoding is correct.
        let spline = Spline {
            control_points: vec![
                SplinePoint::new(0.0, 0.0),
                SplinePoint::new(10.0, 0.0),
                SplinePoint::new(20.0, 5.0),
                SplinePoint::new(30.0, 15.0),
            ],
            color_dct: [[0.0; 32]; 3],
            sigma_dct: {
                let mut s = [0.0; 32];
                s[0] = 1.0;
                s
            },
        };

        let qs = QuantizedSpline::from_spline(&spline, 0, 0.0, 0.0);

        // Deltas: (10,0), (10,5), (10,10)
        // Double-deltas: (10,0), (0,5), (0,5)
        assert_eq!(qs.control_points.len(), 3); // 4 points - 1 starting point = 3
        assert_eq!(qs.control_points[0], (10, 0));
        assert_eq!(qs.control_points[1], (0, 5));
        assert_eq!(qs.control_points[2], (0, 5));
    }

    /// Chunk 2 invariant: a flat image has nothing for the detector
    /// to lock onto (gradient magnitude is zero everywhere).
    #[test]
    fn test_find_splines_returns_empty_for_constant_image() {
        let (w, h, stride) = (64usize, 64usize, 64usize);
        let plane = vec![0.0f32; stride * h];
        let out = find_splines(&plane, &plane, &plane, w, h, stride);
        assert!(
            out.is_empty(),
            "constant image must yield zero splines; got {} splines",
            out.len()
        );
    }

    /// Chunk 2 detector: a long bright horizontal "wire" produces a
    /// non-empty pre-gate candidate set (verified via the pipeline-internal
    /// polyline trace step). This test now bypasses the chunk-3
    /// trial-encode cost gate (recalibrated to 0.20 in chunk 4) by using
    /// the raw `find_splines` helper — single-ridge synthetics are
    /// expected to BE REJECTED by the production gate (they regress real
    /// encodes), but the detector itself must still produce candidates
    /// for the gate to evaluate. Multi-ridge synthetics that actually
    /// admit through the gate are exercised by
    /// `examples/auto_splines_corpus_bench.rs`.
    #[test]
    fn test_find_splines_finds_horizontal_ridge() {
        // Verify the detector produces the expected polylines for a single
        // bright horizontal ridge. We re-implement the early phases of
        // `find_splines_at_distance` to short-circuit before the cost gate,
        // since the cost gate (correctly, post-chunk-4) rejects this case
        // as a real-encode regression.
        let (w, h, stride) = (1024usize, 64usize, 1024usize);
        let mut y_plane = vec![0.0f32; stride * h];
        for x in 4..w - 4 {
            y_plane[32 * stride + x] = 0.8;
        }
        let (gx, gy, gmag) = sobel_xy(&y_plane, w, h, stride);
        let nms_mask = nms_along_gradient(&gx, &gy, &gmag, w, h);
        let ridge_mask = hessian_ridge_filter(&y_plane, &nms_mask, w, h, stride);
        let polylines = trace_polylines(&ridge_mask, &gmag, w, h);
        assert!(
            !polylines.is_empty(),
            "horizontal ridge should produce at least one polyline; got zero"
        );
        // Each polyline must trace at least the conservative-min ridge
        // length so the downstream subsampler can extract a curve.
        for p in &polylines {
            assert!(p.len() >= detect_params::MIN_POLYLINE_LEN);
        }
    }

    /// Photo-like content: a smoothly-varying gradient has no ridges
    /// — Hessian filter should reject all candidates, leaving no
    /// splines. This is the no-regression invariant for natural
    /// photos.
    #[test]
    fn test_find_splines_rejects_smooth_gradient() {
        let (w, h, stride) = (128usize, 128usize, 128usize);
        let mut y_plane = vec![0.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                // Slow ramp — gradient magnitude ~1/w per pixel,
                // well below MIN_GRAD_MAG.
                y_plane[y * stride + x] = x as f32 / w as f32;
            }
        }
        let zero = vec![0.0f32; stride * h];
        let out = find_splines(&y_plane, &y_plane, &zero, w, h, stride);
        assert!(
            out.is_empty(),
            "smooth gradient should yield zero splines; got {}",
            out.len()
        );
    }

    /// Sobel sanity: a vertical-edge step image has gx >> gy at the
    /// edge column.
    #[test]
    fn test_sobel_vertical_edge() {
        let (w, h, stride) = (16usize, 8usize, 16usize);
        let mut plane = vec![0.0f32; stride * h];
        for y in 0..h {
            for x in 8..w {
                plane[y * stride + x] = 1.0;
            }
        }
        let (gx, gy, _gmag) = sobel_xy(&plane, w, h, stride);
        // At interior column 7 or 8, gx should be strongly positive.
        let idx = 4 * w + 7;
        assert!(
            gx[idx].abs() > 0.5,
            "gx at edge should be large, got {}",
            gx[idx]
        );
        assert!(
            gy[idx].abs() < 0.1,
            "gy at vertical edge should be tiny, got {}",
            gy[idx]
        );
    }

    /// Hessian filter: corner (blob-like) should be rejected.
    #[test]
    fn test_hessian_rejects_corner() {
        let (w, h, stride) = (16usize, 16usize, 16usize);
        let mut plane = vec![0.0f32; stride * h];
        // Single bright pixel — isotropic, eigenvalues equal.
        plane[8 * stride + 8] = 1.0;
        let mut mask = vec![false; w * h];
        mask[8 * w + 8] = true;
        let ridge = hessian_ridge_filter(&plane, &mask, w, h, stride);
        assert!(
            !ridge[8 * w + 8],
            "isotropic bright pixel must be rejected by Hessian ratio filter"
        );
    }

    /// Hessian filter: thin horizontal line (ridge-like) should pass.
    #[test]
    fn test_hessian_accepts_horizontal_ridge() {
        let (w, h, stride) = (32usize, 16usize, 32usize);
        let mut plane = vec![0.0f32; stride * h];
        for x in 4..w - 4 {
            plane[8 * stride + x] = 1.0;
        }
        let mut mask = vec![false; w * h];
        mask[8 * w + 16] = true; // pixel on the ridge
        let ridge = hessian_ridge_filter(&plane, &mask, w, h, stride);
        assert!(
            ridge[8 * w + 16],
            "horizontal ridge pixel must pass Hessian ratio filter"
        );
    }

    /// Subsample sanity.
    #[test]
    fn test_subsample_polyline_endpoints() {
        let poly: Vec<(i32, i32)> = (0..100).map(|i| (i, 0)).collect();
        let cps = subsample_polyline(&poly, 5);
        assert_eq!(cps.len(), 5);
        assert!((cps[0].x - 0.0).abs() < 0.5);
        assert!((cps[4].x - 99.0).abs() < 1.0);
    }

    /// Chunk-3: bilinear sampling correctly interpolates between
    /// integer pixel centres. With a 2×1 plane `[0.0, 1.0]`, sampling
    /// at `x=0.5` must yield `0.5`, and edge clamping must hold beyond
    /// the right edge.
    #[test]
    fn test_bilinear_sample_interpolates_and_clamps() {
        let plane = vec![0.0f32, 1.0, 0.0, 1.0]; // 2x2: top row 0/1, bottom row 0/1
        // Mid-row at x=0.5 → 0.5
        let v_mid = bilinear_sample(&plane, 2, 2, 2, 0.5, 0.0);
        assert!((v_mid - 0.5).abs() < 1e-6, "mid-x got {v_mid}");
        // Edge clamp: x=10 → x clamped to (width-1)=1, sample == 1.0
        let v_right = bilinear_sample(&plane, 2, 2, 2, 10.0, 0.0);
        assert!((v_right - 1.0).abs() < 1e-6, "clamped-x got {v_right}");
        // Negative clamp: x=-3 → x clamped to 0
        let v_left = bilinear_sample(&plane, 2, 2, 2, -3.0, 1.0);
        assert!((v_left - 0.0).abs() < 1e-6, "neg-clamped-x got {v_left}");
    }

    /// Chunk-3: the Hessian's larger eigenvalue magnitude is large at
    /// a 1-pixel-wide ridge centre (high curvature across the ridge)
    /// and ~zero on a constant plane.
    #[test]
    fn test_hessian_lambda_large_on_ridge_vs_flat() {
        // Constant plane → all-zero Hessian → λ_large ≈ 0.
        let flat = vec![0.5f32; 9];
        let lam_flat = hessian_lambda_large(&flat, 3, 3, 3, 1.0, 1.0);
        assert!(lam_flat.abs() < 1e-6, "flat λ_large got {lam_flat}");

        // 1-px ridge at y=1 (centre row bright on dark background) →
        // strong negative curvature across the ridge → |λ_large| large.
        let ridge = vec![
            0.0f32, 0.0, 0.0, // y=0
            1.0, 1.0, 1.0, // y=1 (ridge)
            0.0, 0.0, 0.0, // y=2
        ];
        let lam_ridge = hessian_lambda_large(&ridge, 3, 3, 3, 1.0, 1.0);
        assert!(
            lam_ridge > 1.0,
            "ridge λ_large={lam_ridge} should be > 1 (strong across-ridge curvature)"
        );
    }

    /// Chunk-3: the near-coincident-candidate dedup drops second-of-
    /// pair splines whose start AND end control points are both within
    /// `DUP_RADIUS_PX` of an already-kept candidate. The high-contrast
    /// 1024×64 horizontal ridge from `test_find_splines_finds_…` would
    /// otherwise yield two essentially-identical splines (both sides
    /// of the ridge) — chunk 3 must keep at most one.
    #[test]
    fn test_dedup_keeps_single_horizontal_ridge() {
        const W: usize = 1024;
        const H: usize = 64;
        let mut xyb_y = vec![0.0f32; W * H];
        let y_mid = H / 2;
        for x in 4..W - 4 {
            xyb_y[y_mid * W + x] = 0.6;
        }
        let xyb_x = vec![0.0f32; W * H];
        let xyb_b = vec![0.0f32; W * H];

        // X channel: zero (grey ridge has no chroma in XYB).
        // Y channel: the ridge data.
        // B channel: zero.
        let splines = find_splines(&xyb_x[..], &xyb_y[..], &xyb_b[..], W, H, W);
        // Without dedup the chunk-2 detector returned 2 candidates; chunk 3
        // must keep at most one of the pair. We don't require the cost gate
        // to admit the spline on this very short canvas — chunk 3's trial-
        // encode gate is correctly stricter than chunk 2's analytical one,
        // and may reject a single short ridge if energy_drop is small. The
        // dedup contract is: AT MOST one survives, never two.
        assert!(
            splines.len() <= 1,
            "chunk-3 dedup must keep at most 1 of the two near-coincident \
             ridge tracings (got {})",
            splines.len()
        );
    }

    // ── Chunk-5 content discriminator tests ──────────────────────────────

    /// Chunk-5 invariant: a constant-color image is fully flat — every
    /// 8x8 block has near-max mask1x1 → median far above the threshold
    /// → classified as screenshot-like → auto-splines must be skipped.
    #[test]
    fn test_looks_like_screenshot_flat_image() {
        let (w, h) = (128usize, 128usize);
        let y_plane = alloc::vec![0.4f32; w * h];
        assert!(
            looks_like_screenshot(&y_plane, w, h, w, None).unwrap(),
            "constant-color image must be classified as screenshot-like"
        );
    }

    /// Chunk-5 invariant: a smoothly-varying photo-like gradient has
    /// non-trivial per-pixel Laplacian → low mask1x1 → median far below
    /// 95 → classified as non-screenshot → auto-splines proceeds.
    #[test]
    fn test_looks_like_screenshot_rejects_photo_gradient() {
        let (w, h) = (128usize, 128usize);
        let mut y_plane = alloc::vec![0.0f32; w * h];
        // High-variance gradient with cross-diagonal noise so the 1x1
        // Laplacian sees genuine spatial variation, not the smooth
        // single-axis ramp used by `test_find_splines_rejects_smooth_gradient`.
        for y in 0..h {
            for x in 0..w {
                let base = (x as f32 / w as f32) + 0.3 * (y as f32 / h as f32).sin();
                let noise = 0.1 * (((x * 7 + y * 13) % 17) as f32 / 17.0);
                y_plane[y * w + x] = base + noise;
            }
        }
        assert!(
            !looks_like_screenshot(&y_plane, w, h, w, None).unwrap(),
            "photo-like gradient with spatial noise must NOT be classified as screenshot"
        );
    }

    /// Chunk-5 invariant: handles strided plane buffers correctly. The
    /// encoder passes `xyb_y` with `stride == padded_width` (which may
    /// exceed `width` for boundary-padded XYB), so the discriminator
    /// must re-pack rows when stride > width.
    #[test]
    fn test_looks_like_screenshot_strided_input() {
        let (w, h, stride) = (64usize, 64usize, 96usize);
        let mut y_plane = alloc::vec![0.4f32; stride * h];
        // Fill the "in-bounds" region with a constant; junk in the
        // stride padding should be ignored.
        for y in 0..h {
            for x in 0..w {
                y_plane[y * stride + x] = 0.4;
            }
            for x in w..stride {
                // Junk values: would skew the median if NOT re-packed.
                y_plane[y * stride + x] = 1e3;
            }
        }
        assert!(
            looks_like_screenshot(&y_plane, w, h, stride, None).unwrap(),
            "strided flat image must still be classified as screenshot"
        );
    }

    /// Chunk-5 invariant: tiny images (below one 8x8 block in either
    /// dimension) get a safe `false` return — the caller will short-circuit
    /// in `find_splines_at_distance` at the `< 16` check anyway.
    #[test]
    fn test_looks_like_screenshot_tiny_image() {
        let y_plane = alloc::vec![0.4f32; 4 * 4];
        assert!(
            !looks_like_screenshot(&y_plane, 4, 4, 4, None).unwrap(),
            "tiny image must safely return false"
        );
    }

    #[test]
    fn test_splines_data_construction() {
        let spline = Spline {
            control_points: vec![SplinePoint::new(10.0, 10.0), SplinePoint::new(50.0, 50.0)],
            color_dct: {
                let mut dct = [[0.0f32; 32]; 3];
                dct[1][0] = 0.5;
                dct
            },
            sigma_dct: {
                let mut s = [0.0f32; 32];
                s[0] = 3.0;
                s
            },
        };

        let data = SplinesData::from_splines(vec![spline], 0, 0.0, 1.13, 64, 64);

        assert_eq!(data.splines.len(), 1);
        assert_eq!(data.quantized.len(), 1);
        assert!(!data.segments.is_empty(), "should have rendered segments");
        assert_eq!(
            data.segment_y_start.len(),
            65,
            "y_start should have height+1 entries"
        );
    }
}
