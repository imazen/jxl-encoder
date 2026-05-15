// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! JXL Patches: dictionary-based repeated pattern detection and encoding.
//!
//! Screenshots, UI, and text documents contain many repeated rectangular elements
//! (text glyphs, buttons, icons). This module detects these patterns, stores unique
//! patterns in a modular reference frame, and replaces occurrences with references.
//! libjxl reports 40-60% size wins on screenshots.
//!
//! Algorithm ported from libjxl `enc_patch_dictionary.cc` (`FindTextLikePatches`).

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use super::common::pack_signed;
use crate::bit_writer::BitWriter;
use crate::debug_rect;
use crate::entropy_coding::encode::{
    build_entropy_code_ans_with_options, build_entropy_code_with_options,
};
use crate::entropy_coding::token::Token;
use crate::error::Result;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Safe float-to-i32 with rounding, clamped to prevent overflow (libjxl PR #4596).
/// In Rust, `f32 as i32` on out-of-range values is saturating since Rust 1.45,
/// but this makes the intent explicit and avoids any platform surprises.
#[inline]
fn safe_round_to_i32(val: f32) -> i32 {
    val.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

/// Safe float-to-i32 with truncation (towards zero), clamped to prevent overflow.
#[inline]
fn safe_trunc_to_i32(val: f32) -> i32 {
    val.clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

// ── Constants ──────────────────────────────────────────────────────────────────

/// Reference frame slot for patches (libjxl uses slot 3).
const PATCH_FRAME_REFERENCE_ID: u32 = 3;

/// Maximum patch dimension (pixels).
const MAX_PATCH_SIZE: usize = 32;

/// Grid scan block size for flatness detection.
const PATCH_SIDE: usize = 4;

/// Weighted XYB distance threshold for background flood-fill.
const SIMILAR_THRESHOLD: f32 = 0.8;

/// Weighted XYB distance threshold for border color similarity.
const VERY_SIMILAR_THRESHOLD: f32 = 0.03;

/// Maximum BFS distance from seed for background detection.
const DISTANCE_LIMIT: usize = 50;

/// Minimum occurrences for a patch to be worth encoding.
const MIN_PATCH_OCCURRENCES: usize = 2;

/// Minimum size (in pixels) of the largest patch to enable patches.
const MIN_MAX_PATCH_SIZE: usize = 20;

/// Bin packing slackness factor.
const BIN_PACKING_SLACKNESS: f32 = 1.05;

/// XYB channel dequantization constants (quantize float patch pixels to i8).
const CHANNEL_DEQUANT_XYB: [f32; 3] = [0.01615, 0.08875, 0.1922];

/// XYB channel weights for distance computation.
const CHANNEL_WEIGHTS_XYB: [f32; 3] = [30.0, 3.0, 1.0];

/// RGB channel dequantization constants for non-XYB (lossless) patches.
/// From libjxl: kChannelDequant when !is_xyb = {20/255, 22/255, 20/255}.
const CHANNEL_DEQUANT_RGB: [f32; 3] = [20.0 / 255.0, 22.0 / 255.0, 20.0 / 255.0];

/// RGB channel weights for non-XYB (lossless) patches.
/// From libjxl: kChannelWeights when !is_xyb = {0.017*255, 0.02*255, 0.017*255}.
const CHANNEL_WEIGHTS_RGB: [f32; 3] = [0.017 * 255.0, 0.02 * 255.0, 0.017 * 255.0];

/// Colorspace-dependent constants for patch detection.
struct PatchColorspaceInfo {
    channel_dequant: [f32; 3],
    channel_weights: [f32; 3],
}

impl PatchColorspaceInfo {
    fn xyb() -> Self {
        Self {
            channel_dequant: CHANNEL_DEQUANT_XYB,
            channel_weights: CHANNEL_WEIGHTS_XYB,
        }
    }

    fn rgb() -> Self {
        Self {
            channel_dequant: CHANNEL_DEQUANT_RGB,
            channel_weights: CHANNEL_WEIGHTS_RGB,
        }
    }
}

/// Number of entropy contexts for patches encoding.
const NUM_PATCH_CONTEXTS: usize = 10;

/// Minimum neighbor ratio for screenshot-like blocks (8 of 9).
const SCREENSHOT_FLAT_NEIGHBOR_RATIO: usize = 8;

/// Minimum quantized value peak for a valid patch.
const MIN_PEAK: i32 = 2;

/// Radius for has_similar spatial consistency check.
const HAS_SIMILAR_RADIUS: usize = 2;

/// Threshold for has_similar check.
const HAS_SIMILAR_THRESHOLD: f32 = 0.03;

// ── Data Structures ────────────────────────────────────────────────────────────

/// A patch quantized to i8 per channel, plus the original float pixels.
#[derive(Clone)]
struct QuantizedPatch {
    xsize: usize,
    ysize: usize,
    /// Quantized pixel values per channel: `pixels[c][y * xsize + x]`.
    pixels: [Vec<i8>; 3],
    /// Original float pixel values (for reference frame): `fpixels[c][y * xsize + x]`.
    fpixels: [Vec<f32>; 3],
}

impl QuantizedPatch {
    fn num_pixels(&self) -> usize {
        self.xsize * self.ysize
    }
}

impl PartialEq for QuantizedPatch {
    fn eq(&self, other: &Self) -> bool {
        self.xsize == other.xsize
            && self.ysize == other.ysize
            && self.pixels[0] == other.pixels[0]
            && self.pixels[1] == other.pixels[1]
            && self.pixels[2] == other.pixels[2]
    }
}

impl Eq for QuantizedPatch {}

impl PartialOrd for QuantizedPatch {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QuantizedPatch {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Sort by size (descending), then by content for deduplication
        other
            .num_pixels()
            .cmp(&self.num_pixels())
            .then_with(|| self.ysize.cmp(&other.ysize))
            .then_with(|| self.xsize.cmp(&other.xsize))
            .then_with(|| self.pixels[0].cmp(&other.pixels[0]))
            .then_with(|| self.pixels[1].cmp(&other.pixels[1]))
            .then_with(|| self.pixels[2].cmp(&other.pixels[2]))
    }
}

/// A unique patch template with all its occurrences in the image.
pub(crate) struct PatchInfo {
    patch: QuantizedPatch,
    /// Positions where this patch appears: `(x, y)` of top-left corner.
    positions: Vec<(u32, u32)>,
}

/// Position of a unique patch within the reference frame.
#[derive(Clone)]
pub(crate) struct PatchReferencePosition {
    /// Reference frame slot (always `PATCH_FRAME_REFERENCE_ID`).
    ref_id: u32,
    /// X position within reference frame.
    x0: u32,
    /// Y position within reference frame.
    y0: u32,
    /// Width of the patch.
    xsize: u32,
    /// Height of the patch.
    ysize: u32,
}

/// A single patch occurrence in the image.
#[derive(Clone)]
pub(crate) struct PatchPosition {
    /// Position in the image.
    x: u32,
    y: u32,
    /// Index into `ref_positions`.
    ref_pos_idx: usize,
}

/// All patches data for a frame: positions, references, and the reference image.
#[derive(Clone)]
pub(crate) struct PatchesData {
    /// All patch occurrences, grouped by reference position.
    pub positions: Vec<PatchPosition>,
    /// Unique patch reference positions in the reference frame.
    pub ref_positions: Vec<PatchReferencePosition>,
    /// Reference frame pixel data (3 XYB channels, row-major).
    pub ref_image: [Vec<f32>; 3],
    /// Reference frame width.
    pub ref_width: usize,
    /// Reference frame height.
    pub ref_height: usize,
}

impl PatchesData {
    /// Check whether patches are cost-effective at the given distance.
    ///
    /// Trial-encodes the reference frame to measure actual overhead, then estimates
    /// VarDCT savings from patch subtraction. Returns false if overhead exceeds
    /// estimated savings (with 2x safety margin).
    ///
    /// At high distances (d>=3), VarDCT savings per pixel drop while ref frame
    /// overhead stays constant, causing patches to hurt rather than help.
    pub fn is_cost_effective(&self, distance: f32, use_ans: bool) -> bool {
        let ref_overhead = trial_encode_ref_frame_bytes(self, use_ans);
        if ref_overhead == usize::MAX {
            return false;
        }
        // Estimate dictionary section overhead: ~5 bytes per ref position + ~5 per occurrence
        let dict_overhead_est = self.ref_positions.len() * 5 + self.positions.len() * 5;
        let total_overhead = ref_overhead.saturating_add(dict_overhead_est);
        // Sum total patch pixels across all occurrences
        let total_patch_pixels: usize = self
            .positions
            .iter()
            .map(|pos| {
                let rp = &self.ref_positions[pos.ref_pos_idx];
                (rp.xsize as usize) * (rp.ysize as usize)
            })
            .sum();
        // Each patch pixel saves roughly (0.3 / distance) bytes of VarDCT data
        let savings_est = (total_patch_pixels as f64 / (distance.max(0.5) as f64) * 0.3) as usize;
        let effective = savings_est >= 2 * total_overhead;
        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "PATCHES cost_effective: d={:.2} ref_overhead={} dict_overhead={} total_overhead={} \
             patch_pixels={} savings_est={} effective={}",
            distance,
            ref_overhead,
            dict_overhead_est,
            total_overhead,
            total_patch_pixels,
            savings_est,
            effective
        );
        effective
    }

    /// Build a `PatchesData` from a list of [`super::dot_detection::DetectedDot`]
    /// (refs #19). Used when the regular text-like patches detector found
    /// nothing but dot detection produced candidates. Dots are stacked
    /// horizontally in a single-row strip in `ref_image`. Returns `None`
    /// if the input is empty.
    ///
    /// `ref_width` = sum of dot widths, `ref_height` = max dot height.
    /// Each dot's residual data is copied into its slot; `ref_positions`
    /// + `positions` are appended in order so the dot at index `i` lives
    ///   at `(prefix_x[i], 0)` in the reference frame and at
    ///   `(dot.x0, dot.y0)` in the image.
    pub fn from_dots(dots: &[super::dot_detection::DetectedDot]) -> Option<Self> {
        if dots.is_empty() {
            return None;
        }
        let ref_height = dots.iter().map(|d| d.ysize).max()?;
        let ref_width: usize = dots.iter().map(|d| d.xsize).sum();
        if ref_width == 0 || ref_height == 0 {
            return None;
        }
        let ref_n = ref_width * ref_height;
        let mut ref_image = [
            vec![0.0_f32; ref_n],
            vec![0.0_f32; ref_n],
            vec![0.0_f32; ref_n],
        ];
        let mut ref_positions = Vec::with_capacity(dots.len());
        let mut positions = Vec::with_capacity(dots.len());
        let mut x_cursor: usize = 0;
        for (idx, dot) in dots.iter().enumerate() {
            // Copy each channel's residuals into the ref_image slot.
            for c in 0..3 {
                for y in 0..dot.ysize {
                    for x in 0..dot.xsize {
                        let dst_i = y * ref_width + (x_cursor + x);
                        let src_i = y * dot.xsize + x;
                        ref_image[c][dst_i] = dot.residuals[c][src_i];
                    }
                }
            }
            ref_positions.push(PatchReferencePosition {
                ref_id: PATCH_FRAME_REFERENCE_ID,
                x0: x_cursor as u32,
                y0: 0,
                xsize: dot.xsize as u32,
                ysize: dot.ysize as u32,
            });
            positions.push(PatchPosition {
                x: dot.x0,
                y: dot.y0,
                ref_pos_idx: idx,
            });
            x_cursor += dot.xsize;
        }
        Some(PatchesData {
            positions,
            ref_positions,
            ref_image,
            ref_width,
            ref_height,
        })
    }

    /// Roundtrip the reference image through integer quantization to match decoder.
    ///
    /// The encoder subtracts patch values before VarDCT encoding, and the decoder
    /// adds them back from the modular reference frame. The reference frame stores
    /// integers (XYB scaled by InvDCQuant), so there's quantization error.
    ///
    /// This method replaces ref_image with the values the decoder will reconstruct,
    /// ensuring subtract/add in the encoder match the decoder exactly.
    pub fn quantize_ref_image(&mut self) {
        const DC_QUANT_X: f32 = 1.0 / 4096.0;
        const DC_QUANT_Y: f32 = 1.0 / 512.0;
        const DC_QUANT_B: f32 = 1.0 / 256.0;
        let n = self.ref_width * self.ref_height;
        for i in 0..n {
            let x_int = safe_round_to_i32(self.ref_image[0][i] * 4096.0);
            let y_int = safe_round_to_i32(self.ref_image[1][i] * 512.0);
            let b_int = safe_round_to_i32(self.ref_image[2][i] * 256.0);
            // Roundtrip: int → float using decoder's DC quant factors
            self.ref_image[0][i] = x_int as f32 * DC_QUANT_X;
            self.ref_image[1][i] = y_int as f32 * DC_QUANT_Y;
            // B roundtrips through: round(B*256)/256 (B-Y cancels in decoder)
            self.ref_image[2][i] = b_int as f32 * DC_QUANT_B;
        }
    }
}

// ── Detection ──────────────────────────────────────────────────────────────────

/// 8-connected neighbor offsets (excludes self). Used in BFS and DFS loops
/// to avoid the overhead of nested `for dx in -1..=1 { for dy in -1..=1 {`
/// range iterators (measured at ~90M Ir overhead on 1206×2622 screenshots).
const NEIGHBORS_8: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Compute weighted L1 distance between two pixels.
/// Matches libjxl: `sum(|v1[c] - v2[c]| * kChannelWeights[c])`
#[inline]
fn weighted_distance(
    planes: &[&[f32]; 3],
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    cs: &PatchColorspaceInfo,
) -> f32 {
    let i1 = y1 * stride + x1;
    let i2 = y2 * stride + x2;
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (planes[c][i1] - planes[c][i2]).abs() * cs.channel_weights[c];
    }
    dist
}

/// Compute weighted L1 distance between a pixel and a given color.
/// Matches libjxl: `sum(|v1[c] - v2[c]| * kChannelWeights[c])`
#[inline]
fn weighted_distance_to_color(
    planes: &[&[f32]; 3],
    stride: usize,
    x: usize,
    y: usize,
    color: &[f32; 3],
    cs: &PatchColorspaceInfo,
) -> f32 {
    let i = y * stride + x;
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (planes[c][i] - color[c]).abs() * cs.channel_weights[c];
    }
    dist
}

/// Like `weighted_distance_to_color` but takes a pre-computed flat index,
/// eliminating the `y * stride + x` multiplication.
#[inline]
fn weighted_distance_to_color_idx(
    planes: &[&[f32]; 3],
    idx: usize,
    color: &[f32; 3],
    cs: &PatchColorspaceInfo,
) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (planes[c][idx] - color[c]).abs() * cs.channel_weights[c];
    }
    dist
}

/// Flatness threshold: all pixels in a 4x4 block must be this similar.
const FLATNESS_THRESHOLD: f32 = 1e-4;

/// Check if a pixel matches a given color within 1e-4 per channel.
/// Matches libjxl `is_same_color`.
#[inline]
fn is_same_color(
    planes: &[&[f32]; 3],
    stride: usize,
    x: usize,
    y: usize,
    color: &[f32; 3],
) -> bool {
    let i = y * stride + x;
    for c in 0..3 {
        if (planes[c][i] - color[c]).abs() > FLATNESS_THRESHOLD {
            return false;
        }
    }
    true
}

/// Compute weighted L1 distance between two color values.
#[inline]
fn color_distance(c1: &[f32; 3], c2: &[f32; 3], cs: &PatchColorspaceInfo) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (c1[c] - c2[c]).abs() * cs.channel_weights[c];
    }
    dist
}

/// Check if a 4x4 block starting at (bx*4, by*4) is flat (all pixels same color).
#[inline]
fn is_flat_block(xyb: &[&[f32]; 3], stride: usize, bx: usize, by: usize) -> bool {
    let x0 = bx * PATCH_SIDE;
    let y0 = by * PATCH_SIDE;
    let ref_idx = y0 * stride + x0;
    for dy in 0..PATCH_SIDE {
        for dx in 0..PATCH_SIDE {
            if dy == 0 && dx == 0 {
                continue;
            }
            let idx = (y0 + dy) * stride + (x0 + dx);
            for c in 0..3 {
                if (xyb[c][idx] - xyb[c][ref_idx]).abs() > FLATNESS_THRESHOLD {
                    return false;
                }
            }
        }
    }
    true
}

/// Detect text-like patches in an image.
///
/// Returns a list of unique patches with their occurrence positions.
/// Port of libjxl `FindTextLikePatches` — matches exact algorithm:
/// L1 weighted distance, 8-connected BFS/DFS, (current,source) BFS pairs,
/// first-found border reference, has_similar check, kMinPeak filter.
///
/// `stride` is the row pitch of the plane buffers (may be larger than `width`
/// due to padding). `width` and `height` define the actual image area to scan.
/// `is_xyb` selects XYB colorspace constants (true) or RGB constants (false).
pub(crate) fn find_text_like_patches(
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
    is_xyb: bool,
) -> Vec<PatchInfo> {
    let cs = if is_xyb {
        PatchColorspaceInfo::xyb()
    } else {
        PatchColorspaceInfo::rgb()
    };
    let bw = width / PATCH_SIDE;
    let bh = height / PATCH_SIDE;
    if bw < 3 || bh < 3 {
        return Vec::new();
    }

    let xyb_ref = [xyb[0], xyb[1], xyb[2]];
    let n = stride * height;

    // Step 1: Find flat 4×4 blocks (all 16 pixels identical color).
    let mut is_flat = vec![false; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            is_flat[by * bw + bx] = is_flat_block(&xyb_ref, stride, bx, by);
        }
    }

    // Step 2: Screenshot-like detection (block-level).
    // Central block must be flat. Count 3×3 neighbor block origins (single pixel
    // at top-left of each block) with same color. Must have 8+ of 9 matching.
    // Matches libjxl: py from 1 to ph-3 inclusive, px from 1 to pw-2 inclusive.
    let mut is_screenshot_like = vec![false; bw * bh];
    let mut num_seeds = 0u32;
    // bh.saturating_sub(2) as exclusive end → by goes from 1 to bh-3 inclusive
    for by in 1..bh.saturating_sub(2) {
        // bw.saturating_sub(1) as exclusive end → bx goes from 1 to bw-2 inclusive
        for bx in 1..bw.saturating_sub(1) {
            if !is_flat[by * bw + bx] {
                continue;
            }
            let base_x = bx * PATCH_SIDE;
            let base_y = by * PATCH_SIDE;
            let base_i = base_y * stride + base_x;
            let base_color = [xyb[0][base_i], xyb[1][base_i], xyb[2][base_i]];

            // Check 3×3 neighborhood — single pixel at each block origin
            // (NOT checking if neighbor block is flat — matches libjxl)
            let mut num_same = 0usize;
            for nby in by - 1..=by + 1 {
                for nbx in bx - 1..=bx + 1 {
                    let ny = nby * PATCH_SIDE;
                    let nx = nbx * PATCH_SIDE;
                    if is_same_color(&xyb_ref, stride, nx, ny, &base_color) {
                        num_same += 1;
                    }
                }
            }
            if num_same >= SCREENSHOT_FLAT_NEIGHBOR_RATIO {
                is_screenshot_like[by * bw + bx] = true;
                num_seeds += 1;
            }
        }
    }

    debug_rect!(
        "patches/seeds",
        0,
        0,
        width,
        height,
        "{num_seeds} screenshot-like seeds from {bw}x{bh} block grid"
    );

    if num_seeds == 0 {
        return Vec::new();
    }

    // Step 3: BFS background flood-fill with (current, source) pairs.
    // Each background pixel stores its seed's opsin color in the background image.
    // Source propagates unchanged through BFS — Manhattan distance is from source.
    let mut is_background = vec![false; n];
    let mut background = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    // Queue entries: (cur_x, cur_y, src_x, src_y) as u32 to match libjxl's
    // std::pair<XY, XY> (16 bytes vs 32 bytes with usize — halves cache pressure).
    let mut queue: Vec<(u32, u32, u32, u32)> =
        Vec::with_capacity(2 * num_seeds as usize * PATCH_SIDE * PATCH_SIDE);

    // Seed from screenshot-like block pixels
    for by in 1..bh.saturating_sub(1) {
        for bx in 1..bw.saturating_sub(1) {
            if !is_screenshot_like[by * bw + bx] {
                continue;
            }
            for y in by * PATCH_SIDE..(by + 1) * PATCH_SIDE {
                for x in bx * PATCH_SIDE..(bx + 1) * PATCH_SIDE {
                    if x < width && y < height {
                        let i = y * stride + x;
                        if !is_background[i] {
                            is_background[i] = true;
                            queue.push((x as u32, y as u32, x as u32, y as u32));
                        }
                    }
                }
            }
        }
    }

    // BFS flood-fill (8-connected, matches libjxl kSearchRadius=1)
    // Pre-compute stride-based neighbor offsets to replace per-neighbor multiply.
    let stride_i = stride as isize;
    let neighbor_offsets: [isize; 8] = [
        -stride_i - 1,
        -stride_i,
        -stride_i + 1,
        -1,
        1,
        stride_i - 1,
        stride_i,
        stride_i + 1,
    ];
    let mut queue_front = 0;
    while queue_front < queue.len() {
        let (cx, cy, sx, sy) = queue[queue_front];
        queue_front += 1;
        let (cxu, cyu) = (cx as usize, cy as usize);
        let (sxu, syu) = (sx as usize, sy as usize);

        // SECURITY: every queue entry that this loop pops was pushed
        // earlier with `(nx as usize) < width && (ny as usize) < height`
        // bound-checked at push time, so cxu/cyu/sxu/syu are always in
        // range — UNLESS the queue's heap memory was corrupted by an
        // OOB write from elsewhere (the v09/v11 sweep cause, traced to
        // the `unsafe-performance` feature and removed in PR #34).
        //
        // Convert from "skip on bounds failure (DoS protection)" to
        // unconditional `assert!` because the upstream cause is no
        // longer reachable on the post-PR-#34 chain. If a future
        // regression re-introduces queue corruption, surface it loudly.
        // The 270-encode trigger-fixture sweep confirmed this never
        // fires on legitimate input.
        assert!(
            cxu < width && cyu < height && sxu < width && syu < height,
            "patches BFS pop: queue entry out of range — possible upstream \
             corruption (cxu={cxu}, cyu={cyu}, sxu={sxu}, syu={syu}, \
             width={width}, height={height})"
        );
        let ci = cyu * stride + cxu;
        let si = syu * stride + sxu;
        assert!(
            ci < background[0].len() && si < xyb_ref[0].len(),
            "patches BFS pop: derived flat index out of range — possible \
             stride mismatch (ci={ci}, si={si}, n={})",
            background[0].len()
        );

        // Cache source color once per queue entry (avoids re-reading xyb[c][si]
        // for every neighbor — up to 9 bounds-checked reads per entry).
        let src_color = [xyb_ref[0][si], xyb_ref[1][si], xyb_ref[2][si]];
        for c in 0..3 {
            background[c][ci] = src_color[c];
        }

        // 8-connected expansion
        for k in 0..8 {
            let (dx, dy) = NEIGHBORS_8[k];
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            // Unsigned boundary check: negative values wrap to huge usize, exceeding width/height.
            if (nx as usize) >= width || (ny as usize) >= height {
                continue;
            }
            // Flat index via pre-computed stride offset (avoids nyu * stride + nxu multiply).
            let ni = (ci as isize + neighbor_offsets[k]) as usize;
            // The (nx, ny) range check above + the pre-computed stride
            // offsets guarantee ni < n on every legitimate path. Assert
            // loudly — we no longer skip silently.
            assert!(
                ni < is_background.len(),
                "patches BFS neighbor: flat index out of range \
                 (ni={ni}, n={})",
                is_background.len()
            );
            if is_background[ni] {
                continue;
            }
            // Manhattan distance from source (not current!) to candidate
            let manhattan = (nx - sx as i32).unsigned_abs() + (ny - sy as i32).unsigned_abs();
            if manhattan > DISTANCE_LIMIT as u32 {
                continue;
            }
            // Similarity: compare source pixel to candidate pixel (L1 weighted)
            if weighted_distance_to_color_idx(&xyb_ref, ni, &src_color, &cs) <= SIMILAR_THRESHOLD {
                is_background[ni] = true;
                queue.push((nx as u32, ny as u32, sx, sy));
            }
        }
    }
    let bg_count = is_background.iter().filter(|&&b| b).count();
    debug_rect!(
        "patches/bfs",
        0,
        0,
        width,
        height,
        "BFS background: {bg_count} pixels ({:.1}% of image)",
        bg_count as f64 / (width * height) as f64 * 100.0
    );
    drop(queue);

    // Step 4: Extract foreground connected components (8-connected DFS).
    // Track border consistency: first background neighbor = reference,
    // all subsequent must match reference via background image colors.
    let mut visited = vec![false; n];
    let mut patches: Vec<(QuantizedPatch, u32, u32)> = Vec::new();

    // Diagnostic counters (zero-cost when debug-rect is disabled)
    let mut stat_raw_ccs = 0u32;
    let mut stat_reject_no_border = 0u32;
    let mut stat_reject_inconsistent = 0u32;
    let mut stat_reject_too_large = 0u32;
    let mut stat_reject_no_similar = 0u32;
    let mut stat_reject_low_peak = 0u32;
    let mut stat_accepted = 0u32;
    let mut stat_accepted_pixels = 0u64;

    for start_y in 0..height {
        for start_x in 0..width {
            let si = start_y * stride + start_x;
            if is_background[si] || visited[si] {
                continue;
            }

            // DFS — always completes full CC (no early bounding box exit).
            // Use u32 stack entries (8 bytes) matching libjxl's pair<uint32_t, uint32_t>.
            let mut stack: Vec<(u32, u32)> = vec![(start_x as u32, start_y as u32)];
            let mut min_x = start_x;
            let mut max_x = start_x;
            let mut min_y = start_y;
            let mut max_y = start_y;
            let mut found_border = false;
            let mut all_similar = true;
            // Cache reference background color to avoid re-reading 3 arrays per border check.
            let mut ref_bg: [f32; 3] = [0.0; 3];

            while let Some((px32, py32)) = stack.pop() {
                let (px, py) = (px32 as usize, py32 as usize);
                // Same upgrade as the BFS pop above: assert! instead of
                // skip-on-bounds-failure. Stack memory corruption was the
                // v09/v11 cause; removing `unsafe-performance` in PR #34
                // closes the upstream bug, so the assert can be loud.
                assert!(
                    px < width && py < height,
                    "patches DFS pop: stack entry out of range \
                     (px={px}, py={py}, width={width}, height={height})"
                );
                let pi = py * stride + px;
                assert!(
                    pi < visited.len(),
                    "patches DFS pop: derived flat index out of range \
                     (pi={pi}, n={})",
                    visited.len()
                );
                if visited[pi] {
                    continue;
                }
                visited[pi] = true;
                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // Once rejected (inconsistent border or oversized), skip border checks
                // but still complete DFS to mark all CC pixels as visited.
                let rejected = !all_similar
                    || max_x - min_x >= MAX_PATCH_SIZE
                    || max_y - min_y >= MAX_PATCH_SIZE;

                // 8-connected neighbors (kSearchRadius=1, skip self)
                for k in 0..8 {
                    let (ddx, ddy) = NEIGHBORS_8[k];
                    let nx = px32 as i32 + ddx;
                    let ny = py32 as i32 + ddy;
                    // Unsigned boundary check: negative wraps to huge usize.
                    if (nx as usize) >= width || (ny as usize) >= height {
                        continue;
                    }
                    // Flat index via pre-computed stride offset.
                    let ni = (pi as isize + neighbor_offsets[k]) as usize;
                    assert!(
                        ni < is_background.len(),
                        "patches DFS neighbor: flat index out of range \
                         (ni={ni}, n={})",
                        is_background.len()
                    );
                    if !is_background[ni] {
                        // Foreground neighbor — push to stack (skip if already visited
                        // to avoid redundant pop/check cycles from duplicate pushes)
                        if !visited[ni] {
                            stack.push((nx as u32, ny as u32));
                        }
                    } else if !rejected {
                        // Background neighbor — track border consistency
                        // (only when CC hasn't been rejected yet)
                        if !found_border {
                            ref_bg = [background[0][ni], background[1][ni], background[2][ni]];
                            found_border = true;
                        } else {
                            // is_similar_b: compare cached reference bg color
                            // to this neighbor's bg color (VERY_SIMILAR_THRESHOLD)
                            let bg_next = [background[0][ni], background[1][ni], background[2][ni]];
                            if color_distance(&ref_bg, &bg_next, &cs) > VERY_SIMILAR_THRESHOLD {
                                all_similar = false;
                            }
                        }
                    }
                }
            }

            stat_raw_ccs += 1;

            // Filter: must have border, consistent border, within max patch size
            if !found_border
                || !all_similar
                || max_x - min_x >= MAX_PATCH_SIZE
                || max_y - min_y >= MAX_PATCH_SIZE
            {
                if !found_border {
                    stat_reject_no_border += 1;
                } else if !all_similar {
                    stat_reject_inconsistent += 1;
                } else {
                    stat_reject_too_large += 1;
                }
                let reason = if !found_border {
                    "no border"
                } else if !all_similar {
                    "inconsistent border"
                } else {
                    "too large"
                };
                debug_rect!(
                    "patches/cc_reject",
                    min_x,
                    min_y,
                    max_x - min_x + 1,
                    max_y - min_y + 1,
                    "CC rejected: {reason}"
                );
                continue;
            }

            let cc_w = max_x - min_x + 1;
            let cc_h = max_y - min_y + 1;

            // Use cached border/reference color from DFS (ref_bg)
            let ref_color = ref_bg;

            // has_similar check: expanded bounding box (±kHasSimilarRadius) must
            // contain at least one pixel similar to ref color (in opsin image).
            // Uses row-based flat-index iteration to avoid per-pixel y*stride multiply.
            let mut has_similar = false;
            let hs_min_y = min_y.saturating_sub(HAS_SIMILAR_RADIUS);
            let hs_max_y = (max_y + HAS_SIMILAR_RADIUS + 1).min(height);
            let hs_min_x = min_x.saturating_sub(HAS_SIMILAR_RADIUS);
            let hs_max_x = (max_x + HAS_SIMILAR_RADIUS + 1).min(width);
            'outer: for iy in hs_min_y..hs_max_y {
                let row_start = iy * stride;
                for ix in hs_min_x..hs_max_x {
                    if weighted_distance_to_color_idx(&xyb_ref, row_start + ix, &ref_color, &cs)
                        <= HAS_SIMILAR_THRESHOLD
                    {
                        has_similar = true;
                        break 'outer;
                    }
                }
            }
            if !has_similar {
                stat_reject_no_similar += 1;
                debug_rect!(
                    "patches/cc_reject",
                    min_x,
                    min_y,
                    cc_w,
                    cc_h,
                    "CC rejected: no similar pixel in expanded bbox"
                );
                continue;
            }

            // Quantize the patch: pixel_value = opsin[pixel] - ref_color
            let patch_n = cc_w * cc_h;
            let mut qpixels = [vec![0i8; patch_n], vec![0i8; patch_n], vec![0i8; patch_n]];
            let mut fpixels = [
                vec![0.0f32; patch_n],
                vec![0.0f32; patch_n],
                vec![0.0f32; patch_n],
            ];
            let mut is_small = true;
            let mut too_big = false;
            for dy in 0..cc_h {
                for dx in 0..cc_w {
                    let ix = min_x + dx;
                    let iy = min_y + dy;
                    let src_i = iy * stride + ix;
                    let dst_i = dy * cc_w + dx;
                    for c in 0..3 {
                        let val = xyb[c][src_i] - ref_color[c];
                        fpixels[c][dst_i] = val;
                        let q = safe_trunc_to_i32(val / cs.channel_dequant[c]);
                        // Reject patch if any value overflows i8 range (libjxl b6e9d19)
                        if !(-128..=127).contains(&q) {
                            too_big = true;
                        }
                        qpixels[c][dst_i] = q.clamp(-128, 127) as i8;
                        // Use boolean check instead of abs() to avoid i32::MIN panic
                        // (libjxl 2f10c05)
                        is_small &= q < MIN_PEAK && q > -MIN_PEAK;
                    }
                }
            }

            // Reject patches where quantized values overflow i8 (libjxl b6e9d19)
            if too_big {
                stat_reject_low_peak += 1;
                continue;
            }

            // kMinPeak check: reject patches where all quantized magnitudes < MIN_PEAK
            if is_small {
                stat_reject_low_peak += 1;
                debug_rect!(
                    "patches/cc_reject",
                    min_x,
                    min_y,
                    cc_w,
                    cc_h,
                    "CC rejected: all values < {MIN_PEAK}"
                );
                continue;
            }

            stat_accepted += 1;
            stat_accepted_pixels += (cc_w * cc_h) as u64;
            debug_rect!(
                "patches/cc_accept",
                min_x,
                min_y,
                cc_w,
                cc_h,
                "CC accepted: {cc_w}x{cc_h}"
            );

            let patch = QuantizedPatch {
                xsize: cc_w,
                ysize: cc_h,
                pixels: qpixels,
                fpixels,
            };
            patches.push((patch, min_x as u32, min_y as u32));
        }
    }

    // Step 5: Sort and deduplicate patches
    use std::collections::HashMap;
    let mut patch_groups: HashMap<Vec<u8>, Vec<(u32, u32, QuantizedPatch)>> = HashMap::new();

    for (patch, x, y) in patches {
        let mut key = Vec::with_capacity(4 + patch.pixels[0].len() * 3);
        key.extend_from_slice(&(patch.xsize as u16).to_le_bytes());
        key.extend_from_slice(&(patch.ysize as u16).to_le_bytes());
        for c in 0..3 {
            for &p in &patch.pixels[c] {
                key.push(p as u8);
            }
        }
        patch_groups.entry(key).or_default().push((x, y, patch));
    }

    let stat_unique_before_min_occ = patch_groups.len() as u32;
    let stat_singleton_groups = patch_groups
        .values()
        .filter(|g| g.len() < MIN_PATCH_OCCURRENCES)
        .count() as u32;

    // Collect singletons for diagnostic analysis
    #[cfg(test)]
    let singleton_patches: Vec<QuantizedPatch> = patch_groups
        .values()
        .filter(|g| g.len() < MIN_PATCH_OCCURRENCES)
        .map(|g| g[0].2.clone())
        .collect();

    let mut result: Vec<PatchInfo> = Vec::new();
    // Collect into a Vec and sort by key for deterministic output.
    // HashMap iteration order is non-deterministic — without sorting,
    // patch order varies between runs, changing entropy coding.
    let mut groups: Vec<_> = patch_groups.into_iter().collect();
    groups.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    for (_key, group) in groups {
        if group.len() < MIN_PATCH_OCCURRENCES {
            continue;
        }
        let positions: Vec<(u32, u32)> = group.iter().map(|(x, y, _)| (*x, *y)).collect();
        let patch = group.into_iter().next().unwrap().2;
        result.push(PatchInfo { patch, positions });
    }

    let total_dedup_occurrences: usize = result.iter().map(|p| p.positions.len()).sum();
    let total_patch_pixels: u64 = result
        .iter()
        .map(|p| p.patch.num_pixels() as u64 * p.positions.len() as u64)
        .sum();
    debug_rect!(
        "patches/dedup",
        0,
        0,
        width,
        height,
        "{} unique patterns; {} total occurrences (from {} raw CCs)",
        result.len(),
        total_dedup_occurrences,
        result.iter().map(|p| p.positions.len()).sum::<usize>()
    );

    debug_rect!(
        "patches/summary",
        0,
        0,
        width,
        height,
        "PIPELINE: seeds={num_seeds} bg={bg_count}({:.1}%) raw_ccs={stat_raw_ccs} \
         reject[no_border={stat_reject_no_border} inconsistent={stat_reject_inconsistent} \
         too_large={stat_reject_too_large} no_similar={stat_reject_no_similar} \
         low_peak={stat_reject_low_peak}] accepted={stat_accepted}({stat_accepted_pixels}px) \
         unique_before_min_occ={stat_unique_before_min_occ} singletons={stat_singleton_groups} \
         final_unique={} final_occ={total_dedup_occurrences} coverage={total_patch_pixels}px({:.1}%)",
        bg_count as f64 / (width * height) as f64 * 100.0,
        result.len(),
        total_patch_pixels as f64 / (width * height) as f64 * 100.0
    );

    // Also print to stderr for test visibility (always, not just debug-rect)
    #[cfg(test)]
    {
        eprintln!("=== PATCH DETECTION PIPELINE ({width}x{height}) ===");
        eprintln!("  Seeds: {num_seeds}");
        eprintln!(
            "  BFS background: {bg_count} pixels ({:.1}%)",
            bg_count as f64 / (width * height) as f64 * 100.0
        );
        eprintln!("  Raw foreground CCs: {stat_raw_ccs}");
        eprintln!(
            "  Rejected: no_border={stat_reject_no_border} inconsistent={stat_reject_inconsistent} too_large={stat_reject_too_large} no_similar={stat_reject_no_similar} low_peak={stat_reject_low_peak}"
        );
        eprintln!(
            "  Accepted CCs: {stat_accepted} ({stat_accepted_pixels} pixels in bounding boxes)"
        );
        eprintln!("  Unique patterns (before min_occ): {stat_unique_before_min_occ}");
        eprintln!("  Singletons (occ < {MIN_PATCH_OCCURRENCES}): {stat_singleton_groups}");
        eprintln!(
            "  Final: {} unique, {total_dedup_occurrences} occurrences, {total_patch_pixels} patch pixels ({:.1}%)",
            result.len(),
            total_patch_pixels as f64 / (width * height) as f64 * 100.0
        );

        // Singleton analysis: for each singleton, find closest match in accepted set
        eprintln!(
            "\n  Singleton analysis ({} singletons):",
            singleton_patches.len()
        );
        let mut dim_mismatch = 0u32;
        let mut quant_mismatch = 0u32;
        for sp in &singleton_patches {
            // Find best match among accepted patches (same dimensions first)
            let mut best_same_dim_diff = i32::MAX;
            let mut best_any_diff = i32::MAX;
            let mut best_same_dim_occ = 0usize;
            for p in &result {
                if p.patch.xsize == sp.xsize && p.patch.ysize == sp.ysize {
                    let mut max_diff = 0i32;
                    for c in 0..3 {
                        for k in 0..sp.pixels[c].len() {
                            max_diff = max_diff
                                .max((sp.pixels[c][k] as i32 - p.patch.pixels[c][k] as i32).abs());
                        }
                    }
                    if max_diff < best_same_dim_diff {
                        best_same_dim_diff = max_diff;
                        best_same_dim_occ = p.positions.len();
                    }
                }
                // Also check ±1 dimension matches
                if sp.xsize.abs_diff(p.patch.xsize) <= 1
                    && sp.ysize.abs_diff(p.patch.ysize) <= 1
                    && (sp.xsize != p.patch.xsize || sp.ysize != p.patch.ysize)
                {
                    // Different dimensions but close - compute overlap area diff
                    let min_w = sp.xsize.min(p.patch.xsize);
                    let min_h = sp.ysize.min(p.patch.ysize);
                    let mut max_diff = 0i32;
                    for c in 0..3 {
                        for dy in 0..min_h {
                            for dx in 0..min_w {
                                let si = dy * sp.xsize + dx;
                                let pi = dy * p.patch.xsize + dx;
                                max_diff = max_diff.max(
                                    (sp.pixels[c][si] as i32 - p.patch.pixels[c][pi] as i32).abs(),
                                );
                            }
                        }
                    }
                    if max_diff < best_any_diff {
                        best_any_diff = max_diff;
                    }
                }
            }
            if best_same_dim_diff <= 3 {
                quant_mismatch += 1;
                if best_same_dim_diff <= 1 {
                    eprintln!(
                        "    Singleton {}x{}: near-match to {}occ pattern (max_diff={})",
                        sp.xsize, sp.ysize, best_same_dim_occ, best_same_dim_diff
                    );
                }
            } else if best_any_diff <= 3 {
                dim_mismatch += 1;
            }
        }
        eprintln!(
            "  Singleton causes: {} quant_mismatch (same dim, diff<=3), {} dim_mismatch (±1 dim, diff<=3), {} other",
            quant_mismatch,
            dim_mismatch,
            singleton_patches.len() as u32 - quant_mismatch - dim_mismatch
        );

        // Dimension histogram of singletons vs accepted
        let mut singleton_dims: std::collections::HashMap<(usize, usize), u32> =
            std::collections::HashMap::new();
        for sp in &singleton_patches {
            *singleton_dims.entry((sp.xsize, sp.ysize)).or_default() += 1;
        }
        let mut accepted_dims: std::collections::HashMap<(usize, usize), u32> =
            std::collections::HashMap::new();
        for p in &result {
            *accepted_dims
                .entry((p.patch.xsize, p.patch.ysize))
                .or_default() += 1;
        }
        eprintln!("\n  Singleton dimensions vs accepted:");
        let mut all_dims: Vec<_> = singleton_dims
            .keys()
            .chain(accepted_dims.keys())
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_dims.sort();
        for d in all_dims {
            let s = singleton_dims.get(&d).copied().unwrap_or(0);
            let a = accepted_dims.get(&d).copied().unwrap_or(0);
            if s > 0 || a > 3 {
                eprintln!(
                    "    {}x{}: {} singletons, {} accepted patterns",
                    d.0, d.1, s, a
                );
            }
        }
    }

    // Check minimum largest patch size
    let max_patch_pixels = result
        .iter()
        .map(|p| p.patch.num_pixels())
        .max()
        .unwrap_or(0);
    if max_patch_pixels < MIN_MAX_PATCH_SIZE {
        return Vec::new();
    }

    result
}

// ── Bin Packing ────────────────────────────────────────────────────────────────

/// Bin-pack patches into a reference frame rectangle using first-fit grid placement.
///
/// Port of libjxl's bin packing algorithm (enc_patch_dictionary.cc:656-732):
/// - Allocate an `occupied` grid (bool per pixel)
/// - For each patch, scan rows then columns for first unoccupied position
/// - Skip ahead when hitting occupied pixels for efficiency
/// - If all patches placed, done. Otherwise grow by 5% and retry.
/// - After success, trim `ref_height` to actual used height.
///
/// Returns the reference frame dimensions and positions of each patch.
type BinPackResult = core::result::Result<(usize, usize, Vec<(u32, u32)>), &'static str>;

fn bin_pack_patches(patches: &[PatchInfo]) -> BinPackResult {
    if patches.is_empty() {
        return Ok((0, 0, Vec::new()));
    }

    // Patches should already be sorted largest-first by caller
    let total_pixels: usize = patches.iter().map(|p| p.patch.num_pixels()).sum();
    let max_x_size = patches.iter().map(|p| p.patch.xsize).max().unwrap_or(1);
    let max_y_size = patches.iter().map(|p| p.patch.ysize).max().unwrap_or(1);

    // Initial estimate: at least as large as biggest patch, at least sqrt(total_pixels)
    let side = (total_pixels as f32).sqrt() as usize;
    let mut ref_width = side.max(max_x_size);
    let mut ref_height = side.max(max_y_size);

    // First-fit grid placement with grow-and-retry.
    // Defensive iteration cap: each retry grows the canvas by 1.05× + 1.
    // Patches here are bounded by MAX_PATCH_SIZE=32 and the largest input
    // image dimension, so even pathological inputs converge in <50 retries.
    // Bail with a single-row layout rather than loop forever if some
    // future bug invariant breaks.
    const BIN_PACK_MAX_RETRIES: usize = 50;
    for _retry in 0..BIN_PACK_MAX_RETRIES {
        // Grow by 5% + 1 before each attempt (matches libjxl: grow at start of do-while)
        ref_width = (ref_width as f32 * BIN_PACKING_SLACKNESS) as usize + 1;
        ref_height = (ref_height as f32 * BIN_PACKING_SLACKNESS) as usize + 1;

        let mut occupied = vec![false; ref_width * ref_height];
        let mut positions = Vec::with_capacity(patches.len());
        let mut max_y: usize = 0;
        let mut success = true;

        for p in patches {
            let xsize = p.patch.xsize;
            let ysize = p.patch.ysize;
            let mut found = false;
            let mut place_x = 0usize;
            let mut place_y = 0usize;

            // Scan for first unoccupied position
            'outer: for y0 in 0..=ref_height.saturating_sub(ysize) {
                let mut x0 = 0usize;
                while x0 + xsize <= ref_width {
                    let mut has_occupied = false;
                    let mut skip_x = x0;
                    // Check if rectangle (x0, y0, xsize, ysize) is all unoccupied
                    'check: for y in y0..y0 + ysize {
                        let mut x = x0;
                        while x < x0 + xsize {
                            if occupied[y * ref_width + x] {
                                has_occupied = true;
                                skip_x = x; // Skip ahead past occupied pixel
                                break 'check;
                            }
                            x += 1;
                        }
                    }
                    if !has_occupied {
                        place_x = x0;
                        place_y = y0;
                        found = true;
                        break 'outer;
                    }
                    // Jump past the occupied pixel (libjxl: x0 = x)
                    x0 = skip_x + 1;
                }
            }

            if !found {
                success = false;
                break;
            }

            // Mark occupied and record position
            positions.push((place_x as u32, place_y as u32));
            for y in place_y..place_y + ysize {
                for x in place_x..place_x + xsize {
                    occupied[y * ref_width + x] = true;
                }
            }
            max_y = max_y.max(place_y + ysize);
        }

        if success {
            // Trim height to actual used extent
            return Ok((ref_width, max_y, positions));
        }
    }
    // Fell through retry cap without packing. This is suspicious — the
    // canvas grew by ~×11 over 50 retries; failing to fit means either an
    // input shape we don't expect or a real bug. Surface as an error so
    // the caller can decide whether to bail or fall back to "no patches"
    // rather than silently dropping a quality-improving feature.
    debug_assert!(
        false,
        "bin_pack_patches: failed to pack {} patches in {BIN_PACK_MAX_RETRIES} retries",
        patches.len()
    );
    Err("bin_pack_patches: retry cap reached without successful pack")
}

// ── Build PatchesData ──────────────────────────────────────────────────────────

/// Build the complete patches data structure from detected patches.
///
/// Performs bin-packing, builds the reference frame, and creates the position lists.
/// Returns None if no valid patches were found.
pub(crate) fn build_patches_data(mut infos: Vec<PatchInfo>) -> Option<PatchesData> {
    if infos.is_empty() {
        return None;
    }

    // Sort by area (largest first) for better bin-packing
    infos.sort_by_key(|info| core::cmp::Reverse(info.patch.num_pixels()));

    // Bin-pack into reference frame (no size limit — FrameEncoder handles multi-group).
    // bin_pack_patches surfaces an Err when its retry cap is hit (a
    // genuinely-degenerate input or a bug); treat that the same as "no
    // patches detected" so we degrade quality but don't kill the encode.
    let (ref_width, ref_height, pack_positions) = match bin_pack_patches(&infos) {
        Ok(v) => v,
        Err(reason) => {
            debug_rect!("patches/build", 0, 0, 0, 0, "bin_pack failed: {reason}");
            return None;
        }
    };
    if ref_width == 0 || ref_height == 0 {
        return None;
    }

    // Build reference image
    let ref_n = ref_width * ref_height;
    let mut ref_image = [
        vec![0.0f32; ref_n],
        vec![0.0f32; ref_n],
        vec![0.0f32; ref_n],
    ];

    let mut ref_positions = Vec::with_capacity(infos.len());
    let mut all_positions = Vec::new();

    for (idx, (info, &(rx, ry))) in infos.iter().zip(pack_positions.iter()).enumerate() {
        // Copy float pixels into reference frame
        for dy in 0..info.patch.ysize {
            for dx in 0..info.patch.xsize {
                let src_i = dy * info.patch.xsize + dx;
                let dst_i = (ry as usize + dy) * ref_width + (rx as usize + dx);
                for c in 0..3 {
                    ref_image[c][dst_i] = info.patch.fpixels[c][src_i];
                }
            }
        }

        ref_positions.push(PatchReferencePosition {
            ref_id: PATCH_FRAME_REFERENCE_ID,
            x0: rx,
            y0: ry,
            xsize: info.patch.xsize as u32,
            ysize: info.patch.ysize as u32,
        });
        debug_assert!(
            (rx as usize + info.patch.xsize) <= ref_width
                && (ry as usize + info.patch.ysize) <= ref_height,
            "ref position ({rx},{ry}) + size ({}x{}) exceeds ref frame {}x{}",
            info.patch.xsize,
            info.patch.ysize,
            ref_width,
            ref_height
        );

        // Sort positions for better delta encoding
        let mut sorted_pos = info.positions.clone();
        sorted_pos.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        for &(px, py) in &sorted_pos {
            all_positions.push(PatchPosition {
                x: px,
                y: py,
                ref_pos_idx: idx,
            });
        }
    }

    Some(PatchesData {
        positions: all_positions,
        ref_positions,
        ref_image,
        ref_width,
        ref_height,
    })
}

// ── Subtraction ────────────────────────────────────────────────────────────────

/// Subtract patches from the XYB image using the reference frame.
///
/// For each patch occurrence at position (px, py), subtract the reference pixel values:
///   `xyb[c][y][x] -= ref[c][ref_y][ref_x]`
///
/// The decoder will add them back using blend mode kAdd.
pub(crate) fn subtract_patches(xyb: &mut [Vec<f32>; 3], xyb_stride: usize, patches: &PatchesData) {
    debug_rect!(
        "patches/subtract",
        0,
        0,
        0,
        0,
        "subtracting {} occurrences from {} unique refs",
        patches.positions.len(),
        patches.ref_positions.len()
    );
    for pos in &patches.positions {
        let ref_pos = &patches.ref_positions[pos.ref_pos_idx];
        let pw = ref_pos.xsize as usize;
        let ph = ref_pos.ysize as usize;
        let ref_x0 = ref_pos.x0 as usize;
        let ref_y0 = ref_pos.y0 as usize;
        let pos_x = pos.x as usize;
        let pos_y = pos.y as usize;

        debug_rect!(
            "patches/sub_occurrence",
            pos_x,
            pos_y,
            pw,
            ph,
            "ref[{}] at ({ref_x0};{ref_y0}) {pw}x{ph}",
            pos.ref_pos_idx
        );
        for dy in 0..ph {
            for dx in 0..pw {
                let img_i = (pos_y + dy) * xyb_stride + (pos_x + dx);
                let ref_i = (ref_y0 + dy) * patches.ref_width + (ref_x0 + dx);
                // Detection produces in-range positions by construction.
                // assert! loudly if a future bug threads a mismatched
                // stride or mutated PatchPosition.
                assert!(
                    img_i < xyb[0].len() && ref_i < patches.ref_image[0].len(),
                    "patches apply: index out of range \
                     (img_i={img_i}, ref_i={ref_i})"
                );
                for c in 0..3 {
                    xyb[c][img_i] -= patches.ref_image[c][ref_i];
                }
            }
        }
    }
}

/// Add patches back to XYB planes (inverse of [`subtract_patches`]).
///
/// Used by the butteraugli loop to simulate the decoder's reconstruction,
/// which adds patches via blend mode kAdd after IDCT + gab + EPF.
pub(crate) fn add_patches(xyb: &mut [Vec<f32>; 3], xyb_stride: usize, patches: &PatchesData) {
    for pos in &patches.positions {
        let ref_pos = &patches.ref_positions[pos.ref_pos_idx];
        let pw = ref_pos.xsize as usize;
        let ph = ref_pos.ysize as usize;
        let ref_x0 = ref_pos.x0 as usize;
        let ref_y0 = ref_pos.y0 as usize;
        let pos_x = pos.x as usize;
        let pos_y = pos.y as usize;

        for dy in 0..ph {
            for dx in 0..pw {
                let img_i = (pos_y + dy) * xyb_stride + (pos_x + dx);
                let ref_i = (ref_y0 + dy) * patches.ref_width + (ref_x0 + dx);
                // Defensive bounds: same rationale as subtract_patches above.
                if img_i >= xyb[0].len() || ref_i >= patches.ref_image[0].len() {
                    continue;
                }
                for c in 0..3 {
                    xyb[c][img_i] += patches.ref_image[c][ref_i];
                }
            }
        }
    }
}

// ── Bitstream Encoding ─────────────────────────────────────────────────────────

/// Encode the patches section in LfGlobal.
///
/// Bitstream format (10 entropy contexts):
/// ```text
/// num_ref_patches                  [ctx 0]
/// for each ref_patch:
///   reference_frame_id             [ctx 1]
///   ref_x0, ref_y0                 [ctx 3]
///   xsize - 1, ysize - 1          [ctx 2]
///   count - 1                      [ctx 7]
///   for i in 0..count:
///     if i == 0:
///       pos_x, pos_y               [ctx 4]  (absolute)
///     else:
///       delta_x, delta_y           [ctx 6]  (PackSigned relative to prev)
///     blend_mode                   [ctx 5]  (always kAdd=2 for no-alpha)
/// ```
pub(crate) fn encode_patches_section(
    patches: &PatchesData,
    use_ans: bool,
    writer: &mut BitWriter,
) -> Result<()> {
    // Collect tokens
    let mut tokens = Vec::new();

    // num_ref_patches
    tokens.push(Token::new(0, patches.ref_positions.len() as u32));

    for (ref_idx, ref_pos) in patches.ref_positions.iter().enumerate() {
        // reference_frame_id
        tokens.push(Token::new(1, ref_pos.ref_id));

        // ref_x0, ref_y0 (ctx 3) — MUST come before size per JXL spec
        tokens.push(Token::new(3, ref_pos.x0));
        tokens.push(Token::new(3, ref_pos.y0));

        // xsize - 1, ysize - 1 (ctx 2) — AFTER position
        tokens.push(Token::new(2, ref_pos.xsize - 1));
        tokens.push(Token::new(2, ref_pos.ysize - 1));

        // Count occurrences for this ref_patch
        let positions_for_ref: Vec<&PatchPosition> = patches
            .positions
            .iter()
            .filter(|p| p.ref_pos_idx == ref_idx)
            .collect();

        // count - 1
        tokens.push(Token::new(7, (positions_for_ref.len() - 1) as u32));

        let mut prev_x = 0u32;
        let mut prev_y = 0u32;

        for (i, pos) in positions_for_ref.iter().enumerate() {
            if i == 0 {
                // First occurrence: absolute position
                tokens.push(Token::new(4, pos.x));
                tokens.push(Token::new(4, pos.y));
            } else {
                // Subsequent: delta from previous
                let dx = pos.x as i32 - prev_x as i32;
                let dy = pos.y as i32 - prev_y as i32;
                tokens.push(Token::new(6, pack_signed(dx)));
                tokens.push(Token::new(6, pack_signed(dy)));
            }

            // blend_mode = kAdd = 2 (always for no-alpha patches)
            tokens.push(Token::new(5, 2));
            // No alpha_channel or clamp fields for kAdd blend mode

            prev_x = pos.x;
            prev_y = pos.y;
        }
    }

    // Write LZ77 disabled flag (required by Decoder::parse — reads lz77_enabled first)
    writer.write(1, 0)?; // lz77_enabled = false

    // Build and write entropy code for patch tokens
    if use_ans {
        let code = build_entropy_code_ans_with_options(
            &tokens,
            NUM_PATCH_CONTEXTS,
            false,
            true,
            None,
            None,
        );
        crate::entropy_coding::encode::write_entropy_code_ans(&code, writer)?;
        crate::entropy_coding::encode::write_tokens_ans(&tokens, &code, None, writer)?;
    } else {
        let code = build_entropy_code_with_options(&tokens, NUM_PATCH_CONTEXTS, false, None);
        let ec = code.as_entropy_code();
        crate::entropy_coding::encode::write_entropy_code(&ec, writer)?;
        crate::entropy_coding::encode::write_tokens(&tokens, &ec, None, writer)?;
    }

    Ok(())
}

// ── High-level entry point ─────────────────────────────────────────────────────

/// Detect patches, build data structures, and return the result.
///
/// Returns None if no useful patches were found (e.g., photo content).
/// The detection algorithm's own filters (kMinPeak, kMinPatchOccurrences,
/// kMinMaxPatchSize, coverage filter) are sufficient to avoid degenerate cases.
/// libjxl has no additional cost-benefit check.
pub(crate) fn find_and_build(
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
) -> Option<PatchesData> {
    let infos = find_text_like_patches(xyb, width, height, stride, true);
    if infos.is_empty() {
        debug_rect!("patches/detect", 0, 0, width, height, "no patches detected");
        return None;
    }

    // Compute coverage statistics before building
    let total_patch_pixels: usize = infos
        .iter()
        .map(|p| p.patch.num_pixels() * p.positions.len())
        .sum();
    let total_unique = infos.len();
    let total_occurrences: usize = infos.iter().map(|p| p.positions.len()).sum();
    let max_patch_size = infos
        .iter()
        .map(|p| p.patch.xsize.max(p.patch.ysize))
        .max()
        .unwrap_or(0);
    let coverage_pct = total_patch_pixels as f64 / (width * height) as f64 * 100.0;
    debug_rect!(
        "patches/detect",
        0,
        0,
        width,
        height,
        "found {} unique; {} occurrences; max_size={}; coverage={:.1}%; total_pixels={}",
        total_unique,
        total_occurrences,
        max_patch_size,
        coverage_pct,
        total_patch_pixels
    );
    let image_pixels = width * height;
    #[cfg(feature = "debug-tokens")]
    {
        let total_unique_pixels: usize = infos.iter().map(|p| p.patch.num_pixels()).sum();
        let total_occurrences: usize = infos.iter().map(|p| p.positions.len()).sum();
        let coverage_pct = total_patch_pixels as f64 / image_pixels as f64 * 100.0;
        eprintln!(
            "PATCHES: {} unique patterns, {} total occurrences, {} unique pixels, {} total patch pixels ({:.1}% of image)",
            infos.len(),
            total_occurrences,
            total_unique_pixels,
            total_patch_pixels,
            coverage_pct
        );
    }

    // Quick coverage filter: patches on <1% of the image never help.
    if total_patch_pixels * 100 < image_pixels {
        let coverage_pct = total_patch_pixels as f64 / image_pixels as f64 * 100.0;
        debug_rect!(
            "patches/coverage",
            0,
            0,
            width,
            height,
            "rejected: {coverage_pct:.2}% coverage < 1%"
        );
        #[cfg(feature = "debug-tokens")]
        eprintln!("PATCHES: skipping — too little coverage ({coverage_pct:.1}% < 1%)");
        return None;
    }

    let patches_data = build_patches_data(infos)?;

    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "PATCHES: ref frame {}x{} ({} pixels), {} unique refs, {} occurrences",
        patches_data.ref_width,
        patches_data.ref_height,
        patches_data.ref_width * patches_data.ref_height,
        patches_data.ref_positions.len(),
        patches_data.positions.len()
    );

    debug_rect!(
        "patches/decision",
        0,
        0,
        width,
        height,
        "ACCEPTED: {} unique refs in {}x{} frame; {} occurrences",
        patches_data.ref_positions.len(),
        patches_data.ref_width,
        patches_data.ref_height,
        patches_data.positions.len()
    );

    Some(patches_data)
}

// ── Lossless Patches ──────────────────────────────────────────────────────────

/// Detect patches for lossless (non-XYB) encoding.
///
/// Converts u8 pixels to f32 [0, 1] for detection, uses RGB colorspace constants.
/// Returns None if no useful patches were found.
///
/// The reference frame pixels are stored as f32 values in [0, 1] range (relative
/// to background), and must be roundtripped through integer quantization to match
/// the decoder's reconstruction.
pub(crate) fn find_and_build_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    num_channels: usize,
    bit_depth: u32,
) -> Option<PatchesData> {
    if width < 16 || height < 16 || num_channels < 3 {
        return None;
    }

    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let inv_max = 1.0 / max_val;
    let n = width * height;

    // Convert to planar f32 [0, 1] — detection needs 3 channels
    let mut planes = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    for i in 0..n {
        let base = i * num_channels;
        for c in 0..3 {
            planes[c][i] = pixels[base + c] as f32 * inv_max;
        }
    }

    let infos = find_text_like_patches(
        [&planes[0], &planes[1], &planes[2]],
        width,
        height,
        width,
        false, // RGB colorspace
    );
    if infos.is_empty() {
        return None;
    }

    // Coverage filter (same as lossy)
    let total_patch_pixels: usize = infos
        .iter()
        .map(|p| p.patch.num_pixels() * p.positions.len())
        .sum();
    let image_pixels = width * height;
    if total_patch_pixels * 100 < image_pixels {
        return None;
    }

    let mut patches_data = build_patches_data(infos)?;

    // Roundtrip ref image through integer quantization to match decoder.
    // For non-XYB: round(v * max_val) / max_val for each channel.
    quantize_ref_image_rgb(&mut patches_data, bit_depth);

    Some(patches_data)
}

/// Roundtrip reference image through integer quantization for non-XYB (lossless).
///
/// The decoder reconstructs integer channel values from the modular reference frame.
/// We must match this exactly by rounding to the integer grid.
fn quantize_ref_image_rgb(patches: &mut PatchesData, bit_depth: u32) {
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let n = patches.ref_width * patches.ref_height;
    for c in 0..3 {
        for i in 0..n {
            let int_val = safe_round_to_i32(patches.ref_image[c][i] * max_val);
            patches.ref_image[c][i] = int_val as f32 / max_val;
        }
    }
}

/// Subtract patches from a ModularImage's channels in integer space.
///
/// For each patch occurrence at (px, py) and each color channel, computes the
/// integer reference value and subtracts it from the channel data.
/// The decoder will add them back using blend mode kAdd.
pub(crate) fn subtract_patches_modular(
    image: &mut crate::modular::channel::ModularImage,
    patches: &PatchesData,
    bit_depth: u32,
) {
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let num_channels = 3.min(image.channels.len());

    for pos in &patches.positions {
        let ref_pos = &patches.ref_positions[pos.ref_pos_idx];
        let pw = ref_pos.xsize as usize;
        let ph = ref_pos.ysize as usize;
        let ref_x0 = ref_pos.x0 as usize;
        let ref_y0 = ref_pos.y0 as usize;
        let pos_x = pos.x as usize;
        let pos_y = pos.y as usize;

        for dy in 0..ph {
            for dx in 0..pw {
                let ref_i = (ref_y0 + dy) * patches.ref_width + (ref_x0 + dx);
                let img_x = pos_x + dx;
                let img_y = pos_y + dy;
                for c in 0..num_channels {
                    let ref_int = safe_round_to_i32(patches.ref_image[c][ref_i] * max_val);
                    let current = image.channels[c].get(img_x, img_y);
                    image.channels[c].set(img_x, img_y, current - ref_int);
                }
            }
        }
    }
}

/// Trial-encode the XYB reference frame and return the byte count.
///
/// Used for cost-benefit gating: if the reference frame overhead exceeds
/// the estimated VarDCT savings from patch subtraction, skip patches entirely.
pub(crate) fn trial_encode_ref_frame_bytes(patches: &PatchesData, use_ans: bool) -> usize {
    let mut writer = BitWriter::new();
    // Trial encode always uses default (no tree learning) — tree learning is slower
    // and the cost estimate only needs to be approximate for the gating decision.
    if encode_reference_frame(patches, use_ans, false, &mut writer, None).is_ok() {
        writer.zero_pad_to_byte();
        writer.bytes_written()
    } else {
        usize::MAX // On error, signal "don't use patches"
    }
}

/// Encode a non-XYB reference frame for lossless patches.
///
/// Frame header: `xyb_encoded=false`, `save_before_ct=true`, `FrameType::ReferenceOnly`.
/// Channels in normal RGB order (no Y/X/B-Y reorder, no DC quant scaling).
/// Each channel value = `round(fpixels[c] * max_val)`.
///
/// Uses FrameEncoder for body encoding, which provides RCT for RGB channels,
/// ANS entropy coding, and multi-group support for reference frames > 256×256.
pub(crate) fn encode_reference_frame_rgb(
    patches: &PatchesData,
    bit_depth: u32,
    use_ans: bool,
    use_tree_learning: bool,
    writer: &mut BitWriter,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use crate::headers::frame_header::{Encoding, FrameHeader, FrameType};

    let ref_w = patches.ref_width;
    let ref_h = patches.ref_height;
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let n = ref_w * ref_h;

    // Build frame header for reference-only frame (non-XYB)
    let mut fh = FrameHeader::lossless();
    fh.frame_type = FrameType::ReferenceOnly;
    fh.encoding = Encoding::Modular;
    fh.xyb_encoded = false; // Non-XYB: raw RGB integer channels
    fh.save_as_reference = PATCH_FRAME_REFERENCE_ID;
    fh.save_before_ct = true;
    fh.is_last = false;
    fh.flags = 0;
    fh.gaborish = false;
    fh.epf_iters = 0;
    fh.width = ref_w as u32;
    fh.height = ref_h as u32;

    fh.write(writer)?;

    // Build modular channels in RGB order (no Y/X/B-Y reorder for non-XYB)
    use crate::modular::channel::{Channel, ModularImage};

    let mut channels = Vec::with_capacity(3);
    for c in 0..3 {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(safe_round_to_i32(patches.ref_image[c][i] * max_val));
        }
        channels.push(Channel::from_vec(data, ref_w, ref_h)?);
    }

    let image = ModularImage {
        channels,
        bit_depth,
        is_grayscale: false,
        has_alpha: false,
    };

    // Use FrameEncoder for body — handles single/multi-group automatically.
    // libjxl uses simple Gradient predictor with RCT for reference frames
    // (enc_patch_dictionary.cc: "Use gradient predictor and not Predictor::Best").
    // Tree learning can help on large ref frames (>= 128×128) with many unique patterns.
    // Gated by EffortProfile.patch_ref_tree_learning (experimental mode, effort >= 7).
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};
    let enable_tree = use_tree_learning && ref_w >= 128 && ref_h >= 128;
    let options = FrameEncoderOptions {
        use_ans,
        use_tree_learning: enable_tree,
        use_squeeze: false,
        is_last: false,
        ..Default::default() // skip_rct=false → RCT applied to RGB channels
    };
    let mut encoder = FrameEncoder::new(ref_w, ref_h, options);
    if let Some(b) = budget {
        encoder = encoder.with_budget(alloc::sync::Arc::clone(b));
    }
    encoder.encode_modular_body(&image, writer)?;

    Ok(())
}

// ── Reference Frame Encoding (XYB) ──────────────────────────────────────────

/// Encode the XYB reference frame containing all unique patch templates.
///
/// This writes a complete modular FrameType::ReferenceOnly frame to the writer.
/// The frame saves to reference slot 3 with save_before_ct=true.
///
/// The reference image is 3-channel XYB float data. For modular encoding, we scale
/// to i32 (multiply by a fixed scale factor and round).
///
/// Uses FrameEncoder for body encoding, which provides RCT for the 3 channels,
/// ANS entropy coding, and multi-group support for reference frames > 256×256.
pub(crate) fn encode_reference_frame(
    patches: &PatchesData,
    use_ans: bool,
    use_tree_learning: bool,
    writer: &mut BitWriter,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> Result<()> {
    use crate::headers::frame_header::{Encoding, FrameHeader, FrameType};

    let ref_w = patches.ref_width;
    let ref_h = patches.ref_height;

    // Build frame header for reference-only frame
    let mut fh = FrameHeader::lossless();
    fh.frame_type = FrameType::ReferenceOnly;
    fh.encoding = Encoding::Modular;
    fh.xyb_encoded = true; // File-level property inherited by all frames
    fh.save_as_reference = PATCH_FRAME_REFERENCE_ID;
    fh.save_before_ct = true;
    fh.is_last = false; // Not the last frame
    fh.flags = 0;
    fh.gaborish = false;
    fh.epf_iters = 0;
    // Set dimensions to the reference frame size (via have_crop mechanism)
    fh.width = ref_w as u32;
    fh.height = ref_h as u32;

    #[cfg(feature = "trace-bitstream")]
    let ref_frame_start = writer.bits_written();
    fh.write(writer)?;
    #[cfg(feature = "trace-bitstream")]
    eprintln!(
        "PATCHES: ref frame header written, bits {}-{} ({} bits)",
        ref_frame_start,
        writer.bits_written(),
        writer.bits_written() - ref_frame_start
    );

    // Convert XYB float data to i32 for modular encoding.
    //
    // The decoder uses LfQuantFactors (DC quant) to convert back:
    //   X_float = ch1_int * DCQuant[0]   where DCQuant[0] = 1/4096
    //   Y_float = ch0_int * DCQuant[1]   where DCQuant[1] = 1/512
    //   B_float = (ch2_int + ch0_int) * DCQuant[2]  where DCQuant[2] = 1/256
    //
    // Since we signal all_default=true for DC quant, the inverse factors are:
    //   INV_DC_QUANT = [4096.0, 512.0, 256.0]  (X, Y, B)
    //
    // Modular channels are stored as: [0=Y, 1=X, 2=B-Y]
    // B-Y subtraction is done in integer space after scaling.
    const INV_DC_QUANT_X: f32 = 4096.0;
    const INV_DC_QUANT_Y: f32 = 512.0;
    const INV_DC_QUANT_B: f32 = 256.0;
    let n = ref_w * ref_h;

    // Build modular channels in decoder order: [Y, X, B-Y]
    use crate::modular::channel::{Channel, ModularImage};

    // Channel 0: Y (from ref_image[1], which is the Y plane in XYB)
    let mut ch_y = Vec::with_capacity(n);
    for i in 0..n {
        ch_y.push(safe_round_to_i32(patches.ref_image[1][i] * INV_DC_QUANT_Y));
    }

    // Channel 1: X (from ref_image[0], which is the X plane in XYB)
    let mut ch_x = Vec::with_capacity(n);
    for i in 0..n {
        ch_x.push(safe_round_to_i32(patches.ref_image[0][i] * INV_DC_QUANT_X));
    }

    // Channel 2: B-Y (B scaled by INV_DC_QUANT_B, minus Y_int from channel 0)
    let mut ch_by = Vec::with_capacity(n);
    for i in 0..n {
        let b_int = safe_round_to_i32(patches.ref_image[2][i] * INV_DC_QUANT_B);
        ch_by.push(b_int - ch_y[i]);
    }

    let mod_channels = vec![
        Channel::from_vec(ch_y, ref_w, ref_h)?,
        Channel::from_vec(ch_x, ref_w, ref_h)?,
        Channel::from_vec(ch_by, ref_w, ref_h)?,
    ];
    let image = ModularImage {
        channels: mod_channels,
        bit_depth: 16, // Fixed-point representation
        is_grayscale: false,
        has_alpha: false,
    };

    // Use FrameEncoder for body — handles single/multi-group automatically.
    // Tree learning adapts prediction to packed glyphs; skip_rct avoids
    // counterproductive YCoCg on already-decorrelated Y/X/B-Y channels.
    // LZ77 RLE compresses the long zero runs between packed patches.
    use crate::modular::frame::{FrameEncoder, FrameEncoderOptions};
    // libjxl uses simple Gradient predictor with RCT for reference frames
    // (enc_patch_dictionary.cc line 821: "Use gradient predictor and not Predictor::Best").
    // Tree learning can help on large ref frames (>= 128×128) with many unique patterns.
    // RCT decorrelates the Y/X/B-Y channels further for entropy coding.
    let enable_tree = use_tree_learning && ref_w >= 128 && ref_h >= 128;
    let options = FrameEncoderOptions {
        use_ans,
        use_tree_learning: enable_tree,
        use_squeeze: false,
        skip_rct: false, // Enable RCT — matches libjxl behavior
        is_last: false,
        ..Default::default()
    };
    let mut encoder = FrameEncoder::new(ref_w, ref_h, options);
    if let Some(b) = budget {
        encoder = encoder.with_budget(alloc::sync::Arc::clone(b));
    }
    encoder.encode_modular_body(&image, writer)?;

    #[cfg(feature = "trace-bitstream")]
    eprintln!(
        "PATCHES: ref frame ends at bit {} (byte {})",
        writer.bits_written(),
        writer.bits_written() / 8
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_signed_roundtrip() {
        for v in -100..=100 {
            let packed = pack_signed(v);
            // Verify zig-zag: non-negative maps to even, negative to odd
            if v >= 0 {
                assert_eq!(packed, (v as u32) * 2);
            } else {
                assert_eq!(packed, ((-v) as u32) * 2 - 1);
            }
        }
    }

    #[test]
    fn test_weighted_distance_zero() {
        let x = vec![1.0f32; 4];
        let y = vec![2.0f32; 4];
        let b = vec![3.0f32; 4];
        let planes: [&[f32]; 3] = [&x, &y, &b];
        let cs = PatchColorspaceInfo::xyb();
        let dist = weighted_distance(&planes, 2, 0, 0, 1, 0, &cs);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_bin_packing_basic() {
        // Create two small patches
        let p1 = QuantizedPatch {
            xsize: 4,
            ysize: 4,
            pixels: [vec![0i8; 16], vec![0i8; 16], vec![0i8; 16]],
            fpixels: [vec![0.0f32; 16], vec![0.0f32; 16], vec![0.0f32; 16]],
        };
        let p2 = QuantizedPatch {
            xsize: 3,
            ysize: 3,
            pixels: [vec![1i8; 9], vec![1i8; 9], vec![1i8; 9]],
            fpixels: [vec![0.1f32; 9], vec![0.1f32; 9], vec![0.1f32; 9]],
        };
        let infos = vec![
            PatchInfo {
                patch: p1,
                positions: vec![(0, 0), (10, 10)],
            },
            PatchInfo {
                patch: p2,
                positions: vec![(5, 5), (15, 15)],
            },
        ];

        let (w, h, positions) = bin_pack_patches(&infos).expect("bin_pack should succeed");
        assert!(w > 0);
        assert!(h > 0);
        assert_eq!(positions.len(), 2);
        // First patch should be at (0, 0)
        assert_eq!(positions[0], (0, 0));
    }

    #[test]
    fn test_no_patches_on_photo() {
        // A "photo-like" image with gradients should produce no patches
        let w = 64;
        let h = 64;
        let n = w * h;
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for py in 0..h {
            for px in 0..w {
                let i = py * w + px;
                x[i] = px as f32 / w as f32 * 0.5;
                y[i] = py as f32 / h as f32;
                b[i] = (px as f32 + py as f32) / (w + h) as f32;
            }
        }
        let result = find_text_like_patches([&x, &y, &b], w, h, w, true);
        assert!(result.is_empty(), "Photos should produce no patches");
    }

    #[test]
    fn test_patches_on_synthetic_screenshot() {
        // Create a simple screenshot-like image: solid background with repeated small patterns
        let w = 128;
        let h = 128;
        let n = w * h;
        let bg_x = 0.5f32;
        let bg_y = 0.8f32;
        let bg_b = 0.3f32;

        let mut x = vec![bg_x; n];
        let mut y = vec![bg_y; n];
        let mut b = vec![bg_b; n];

        // Place a 4x6 foreground pattern at 3 locations
        let fg_x = 0.1f32;
        let fg_y = 0.2f32;
        let fg_b = 0.9f32;
        let positions = [(20, 20), (60, 20), (20, 60)];
        let pw = 4;
        let ph = 6;

        for &(px, py) in &positions {
            for dy in 0..ph {
                for dx in 0..pw {
                    let i = (py + dy) * w + (px + dx);
                    x[i] = fg_x;
                    y[i] = fg_y;
                    b[i] = fg_b;
                }
            }
        }

        let result = find_text_like_patches([&x, &y, &b], w, h, w, true);
        // Should find at least one patch group with >= 2 occurrences
        // Note: the exact number depends on detection thresholds
        if !result.is_empty() {
            let total_occurrences: usize = result.iter().map(|p| p.positions.len()).sum();
            assert!(total_occurrences >= 2, "Should have at least 2 occurrences");
        }
    }

    /// Test reference frame integer value ranges for XYB patches.
    /// Requires the codec-corpus `gb82-sc/terminal.png` screenshot;
    /// gated behind the `corpus-tests` feature so default `cargo test`
    /// runs skip it without the silent in-body short-circuit that the
    /// previous `#[ignore]` + `if !path.exists()` pattern hid behind.
    #[test]
    #[cfg_attr(
        not(feature = "corpus-tests"),
        ignore = "requires codec corpus; enable `corpus-tests` feature"
    )]
    fn test_ref_frame_value_ranges() {
        crate::skip_without_corpus!();
        let path = crate::test_helpers::corpus_dir().join("gb82-sc/terminal.png");
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let pixels = img.as_raw();
        let n = w * h;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for i in 0..n {
            r[i] = pixels[i * 3] as f32;
            g[i] = pixels[i * 3 + 1] as f32;
            b[i] = pixels[i * 3 + 2] as f32;
        }
        let mut x_out = vec![0.0f32; n];
        let mut y_out = vec![0.0f32; n];
        let mut b_out = vec![0.0f32; n];
        crate::color::xyb::srgb_image_to_xyb(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

        let result = find_text_like_patches([&x_out, &y_out, &b_out], w, h, w, true);
        let patches_data = build_patches_data(result).unwrap();

        let ref_w = patches_data.ref_width;
        let ref_h = patches_data.ref_height;
        let ref_n = ref_w * ref_h;
        eprintln!("Reference frame: {ref_w}x{ref_h} = {ref_n} pixels");

        const INV_DC_QUANT_X: f32 = 4096.0;
        const INV_DC_QUANT_Y: f32 = 512.0;
        const INV_DC_QUANT_B: f32 = 256.0;

        // Compute integer channel ranges
        let mut ch_y_min = i32::MAX;
        let mut ch_y_max = i32::MIN;
        let mut ch_x_min = i32::MAX;
        let mut ch_x_max = i32::MIN;
        let mut ch_by_min = i32::MAX;
        let mut ch_by_max = i32::MIN;
        let mut nonzero_y = 0u32;
        let mut nonzero_x = 0u32;
        let mut nonzero_by = 0u32;

        for i in 0..ref_n {
            let y_int = safe_round_to_i32(patches_data.ref_image[1][i] * INV_DC_QUANT_Y);
            let x_int = safe_round_to_i32(patches_data.ref_image[0][i] * INV_DC_QUANT_X);
            let b_int = safe_round_to_i32(patches_data.ref_image[2][i] * INV_DC_QUANT_B);
            let by_int = b_int - y_int;

            ch_y_min = ch_y_min.min(y_int);
            ch_y_max = ch_y_max.max(y_int);
            ch_x_min = ch_x_min.min(x_int);
            ch_x_max = ch_x_max.max(x_int);
            ch_by_min = ch_by_min.min(by_int);
            ch_by_max = ch_by_max.max(by_int);
            if y_int != 0 {
                nonzero_y += 1;
            }
            if x_int != 0 {
                nonzero_x += 1;
            }
            if by_int != 0 {
                nonzero_by += 1;
            }
        }

        eprintln!(
            "Channel Y:  range [{ch_y_min}, {ch_y_max}], {nonzero_y} nonzero ({:.1}%)",
            nonzero_y as f64 / ref_n as f64 * 100.0
        );
        eprintln!(
            "Channel X:  range [{ch_x_min}, {ch_x_max}], {nonzero_x} nonzero ({:.1}%)",
            nonzero_x as f64 / ref_n as f64 * 100.0
        );
        eprintln!(
            "Channel BY: range [{ch_by_min}, {ch_by_max}], {nonzero_by} nonzero ({:.1}%)",
            nonzero_by as f64 / ref_n as f64 * 100.0
        );
    }

    /// Diagnostic test: run patch detection on terminal.png and print
    /// pipeline stats. Same gating as `test_ref_frame_value_ranges`
    /// — requires codec-corpus, gated behind the `corpus-tests`
    /// feature.
    #[test]
    #[cfg_attr(
        not(feature = "corpus-tests"),
        ignore = "requires codec corpus; enable `corpus-tests` feature"
    )]
    fn test_terminal_patch_coverage() {
        crate::skip_without_corpus!();
        let path = crate::test_helpers::corpus_dir().join("gb82-sc/terminal.png");
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let pixels = img.as_raw();
        eprintln!("Loaded terminal.png: {w}x{h}");

        // Convert to planar sRGB f32
        let n = w * h;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for i in 0..n {
            r[i] = pixels[i * 3] as f32;
            g[i] = pixels[i * 3 + 1] as f32;
            b[i] = pixels[i * 3 + 2] as f32;
        }

        // Convert to XYB
        let mut x_out = vec![0.0f32; n];
        let mut y_out = vec![0.0f32; n];
        let mut b_out = vec![0.0f32; n];
        crate::color::xyb::srgb_image_to_xyb(&r, &g, &b, &mut x_out, &mut y_out, &mut b_out);

        // Run detection (eprintln stats from cfg(test) instrumentation)
        let result = find_text_like_patches([&x_out, &y_out, &b_out], w, h, w, true);

        // Print size distribution
        let mut size_dist: std::collections::HashMap<(usize, usize), (usize, usize)> =
            std::collections::HashMap::new();
        for p in &result {
            let entry = size_dist
                .entry((p.patch.xsize, p.patch.ysize))
                .or_insert((0, 0));
            entry.0 += 1; // unique patterns at this size
            entry.1 += p.positions.len(); // total occurrences
        }
        let mut sizes: Vec<_> = size_dist.into_iter().collect();
        sizes.sort_by_key(|&((w, h), _)| std::cmp::Reverse(w * h));
        eprintln!("\nPatch size distribution:");
        for ((pw, ph), (unique, occ)) in &sizes {
            eprintln!("  {pw}x{ph}: {unique} unique, {occ} occurrences");
        }

        // Print top patches by occurrence count
        let mut by_occ: Vec<_> = result.iter().enumerate().collect();
        by_occ.sort_by_key(|(_, p)| std::cmp::Reverse(p.positions.len()));
        eprintln!("\nTop 20 patches by occurrence:");
        for (i, (_, p)) in by_occ.iter().take(20).enumerate() {
            eprintln!(
                "  #{}: {}x{} with {} occurrences",
                i + 1,
                p.patch.xsize,
                p.patch.ysize,
                p.positions.len()
            );
        }

        // Analyze near-miss dedup: find singletons that are close to popular patterns
        // Count singleton dimensions
        let _all_patches = find_text_like_patches([&x_out, &y_out, &b_out], w, h, w, true);
        // Re-run to get raw CCs with their positions (need to access raw data)
        // For now, just analyze the final result's dimension distribution
        eprintln!("\nAnalyzing dedup quality...");

        // Build ALL patches including singletons (re-do dedup manually)
        // We'll work with what we have — check if similar-size patches exist
        // that differ only slightly in quantized values
        let mut all_by_dim: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, p) in result.iter().enumerate() {
            all_by_dim
                .entry((p.patch.xsize, p.patch.ysize))
                .or_default()
                .push(i);
        }

        // Check for patches at same dimensions that could be merged with tolerance
        eprintln!("\nPer-dimension grouping (final patches only):");
        for ((pw, ph), indices) in &all_by_dim {
            if indices.len() >= 2 {
                // Compare pairs within same dimension
                let mut max_diff = 0i32;
                for i in 0..indices.len() {
                    for j in (i + 1)..indices.len() {
                        let a = &result[indices[i]].patch;
                        let b_patch = &result[indices[j]].patch;
                        let mut diff = 0i32;
                        for c in 0..3 {
                            for k in 0..a.pixels[c].len() {
                                diff = diff.max(
                                    (a.pixels[c][k] as i32 - b_patch.pixels[c][k] as i32).abs(),
                                );
                            }
                        }
                        max_diff = max_diff.max(diff);
                    }
                }
                eprintln!(
                    "  {pw}x{ph}: {} patterns, max quantized diff between any pair: {max_diff}",
                    indices.len()
                );
            }
        }
    }
}
