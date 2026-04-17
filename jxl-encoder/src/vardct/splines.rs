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
const CHANNEL_WEIGHT: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];

/// Number of entropy contexts for spline encoding.
const NUM_SPLINE_CONTEXTS: usize = 6;

/// Target rendering distance between sample points along the curve.
const DESIRED_RENDERING_DISTANCE: f32 = 1.0;

/// 1 / (2 * sqrt(2)), used in Gaussian splatting.
const ONE_OVER_2S2: f32 = 0.353_553_38;

/// Exponent for maximum_distance computation (fast mode, matches jxl-rs default).
const DISTANCE_EXP: f32 = 3.0;

/// Number of sub-points per Catmull-Rom segment.
const NUM_POINTS_PER_SEGMENT: usize = 16;

// ── Fast math ───────────────────────────────────────────────────────────────

/// Fast error function approximation (max error ~6e-4).
/// Ported from jxl-rs `fast_math.rs`.
#[allow(clippy::excessive_precision)]
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
#[allow(clippy::excessive_precision)]
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
            let x0 = (segment.center_x - segment.maximum_distance)
                .round()
                .max(0.0) as usize;
            let x1 = width.min((segment.center_x + segment.maximum_distance).round() as usize + 1);
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
                let y0 = 0i64.max((seg.center_y - seg.maximum_distance).round() as i64);
                let y1 = (image_height as i64)
                    .min((seg.center_y + seg.maximum_distance).round() as i64 + 1);
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
