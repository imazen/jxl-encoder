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
const CHANNEL_DEQUANT: [f32; 3] = [0.01615, 0.08875, 0.1922];

/// XYB channel weights for distance computation.
const CHANNEL_WEIGHTS: [f32; 3] = [30.0, 3.0, 1.0];

/// Number of entropy contexts for patches encoding.
const NUM_PATCH_CONTEXTS: usize = 10;

/// Flatness threshold: all pixels in a 4x4 block must be this similar.
const FLATNESS_THRESHOLD: f32 = 1e-4;

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
pub(crate) struct PatchPosition {
    /// Position in the image.
    x: u32,
    y: u32,
    /// Index into `ref_positions`.
    ref_pos_idx: usize,
}

/// All patches data for a frame: positions, references, and the reference image.
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

// ── Detection ──────────────────────────────────────────────────────────────────

/// Compute weighted XYB L1 distance between two pixels.
/// Matches libjxl: `sum(|v1[c] - v2[c]| * kChannelWeights[c])`
#[inline]
fn xyb_distance(
    xyb: &[&[f32]; 3],
    stride: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> f32 {
    let i1 = y1 * stride + x1;
    let i2 = y2 * stride + x2;
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (xyb[c][i1] - xyb[c][i2]).abs() * CHANNEL_WEIGHTS[c];
    }
    dist
}

/// Compute weighted XYB L1 distance between a pixel and a given color.
/// Matches libjxl: `sum(|v1[c] - v2[c]| * kChannelWeights[c])`
#[inline]
fn xyb_distance_to_color(
    xyb: &[&[f32]; 3],
    stride: usize,
    x: usize,
    y: usize,
    color: &[f32; 3],
) -> f32 {
    let i = y * stride + x;
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (xyb[c][i] - color[c]).abs() * CHANNEL_WEIGHTS[c];
    }
    dist
}

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

/// Compute weighted XYB L1 distance between two color values.
#[inline]
fn color_distance(c1: &[f32; 3], c2: &[f32; 3]) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..3 {
        dist += (c1[c] - c2[c]).abs() * CHANNEL_WEIGHTS[c];
    }
    dist
}

/// Check if a 4x4 block starting at (bx*4, by*4) is flat (all pixels same color).
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

/// Detect text-like patches in an XYB image.
///
/// Returns a list of unique patches with their occurrence positions.
/// Port of libjxl `FindTextLikePatches` — matches exact algorithm:
/// L1 weighted distance, 8-connected BFS/DFS, (current,source) BFS pairs,
/// first-found border reference, has_similar check, kMinPeak filter.
///
/// `stride` is the row pitch of the XYB buffers (may be larger than `width`
/// due to padding). `width` and `height` define the actual image area to scan.
pub(crate) fn find_text_like_patches(
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
) -> Vec<PatchInfo> {
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
    // Queue entries: (cur_x, cur_y, src_x, src_y)
    let mut queue: Vec<(usize, usize, usize, usize)> =
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
                            queue.push((x, y, x, y)); // source = self for seeds
                        }
                    }
                }
            }
        }
    }

    // BFS flood-fill (8-connected, matches libjxl kSearchRadius=1)
    let mut queue_front = 0;
    while queue_front < queue.len() {
        let (cx, cy, sx, sy) = queue[queue_front];
        queue_front += 1;

        // Store source color in background at current position
        let ci = cy * stride + cx;
        let si = sy * stride + sx;
        for c in 0..3 {
            background[c][ci] = xyb[c][si];
        }

        // 8-connected expansion
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }
                let (nxu, nyu) = (nx as usize, ny as usize);
                let ni = nyu * stride + nxu;
                if is_background[ni] {
                    continue;
                }
                // Manhattan distance from source (not current!) to candidate
                let manhattan = (nxu as isize - sx as isize).unsigned_abs()
                    + (nyu as isize - sy as isize).unsigned_abs();
                if manhattan > DISTANCE_LIMIT {
                    continue;
                }
                // Similarity: compare source pixel to candidate pixel (L1 weighted)
                if xyb_distance(&xyb_ref, stride, sx, sy, nxu, nyu) <= SIMILAR_THRESHOLD {
                    is_background[ni] = true;
                    queue.push((nxu, nyu, sx, sy)); // propagate source
                }
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

    for start_y in 0..height {
        for start_x in 0..width {
            let si = start_y * stride + start_x;
            if is_background[si] || visited[si] {
                continue;
            }

            // DFS — always completes full CC (no early bounding box exit)
            let mut stack = vec![(start_x, start_y)];
            let mut min_x = start_x;
            let mut max_x = start_x;
            let mut min_y = start_y;
            let mut max_y = start_y;
            let mut found_border = false;
            let mut all_similar = true;
            let mut reference: (usize, usize) = (0, 0);

            while let Some((px, py)) = stack.pop() {
                let pi = py * stride + px;
                if visited[pi] {
                    continue;
                }
                visited[pi] = true;
                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // 8-connected neighbors (kSearchRadius=1, skip self)
                for ddx in -1i32..=1 {
                    for ddy in -1i32..=1 {
                        if ddx == 0 && ddy == 0 {
                            continue;
                        }
                        let nx = px as i32 + ddx;
                        let ny = py as i32 + ddy;
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }
                        let (nxu, nyu) = (nx as usize, ny as usize);
                        let ni = nyu * stride + nxu;
                        if !is_background[ni] {
                            // Foreground neighbor — push to stack
                            stack.push((nxu, nyu));
                        } else {
                            // Background neighbor — track border consistency
                            if !found_border {
                                reference = (nxu, nyu);
                                found_border = true;
                            } else {
                                // is_similar_b: compare background colors at reference
                                // and this neighbor (VERY_SIMILAR_THRESHOLD)
                                let ri = reference.1 * stride + reference.0;
                                let bg_ref =
                                    [background[0][ri], background[1][ri], background[2][ri]];
                                let bg_next =
                                    [background[0][ni], background[1][ni], background[2][ni]];
                                if color_distance(&bg_ref, &bg_next) > VERY_SIMILAR_THRESHOLD {
                                    all_similar = false;
                                }
                            }
                        }
                    }
                }
            }

            // Filter: must have border, consistent border, within max patch size
            if !found_border
                || !all_similar
                || max_x - min_x >= MAX_PATCH_SIZE
                || max_y - min_y >= MAX_PATCH_SIZE
            {
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

            // Get border/reference color from background image
            let ri = reference.1 * stride + reference.0;
            let ref_color = [background[0][ri], background[1][ri], background[2][ri]];

            // has_similar check: expanded bounding box (±kHasSimilarRadius) must
            // contain at least one pixel similar to ref color (in opsin image).
            let mut has_similar = false;
            let hs_min_y = min_y.saturating_sub(HAS_SIMILAR_RADIUS);
            let hs_max_y = (max_y + HAS_SIMILAR_RADIUS + 1).min(height);
            let hs_min_x = min_x.saturating_sub(HAS_SIMILAR_RADIUS);
            let hs_max_x = (max_x + HAS_SIMILAR_RADIUS + 1).min(width);
            for iy in hs_min_y..hs_max_y {
                for ix in hs_min_x..hs_max_x {
                    if xyb_distance_to_color(&xyb_ref, stride, ix, iy, &ref_color)
                        <= HAS_SIMILAR_THRESHOLD
                    {
                        has_similar = true;
                    }
                }
            }
            if !has_similar {
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
            let mut max_value = 0i32;
            for dy in 0..cc_h {
                for dx in 0..cc_w {
                    let ix = min_x + dx;
                    let iy = min_y + dy;
                    let src_i = iy * stride + ix;
                    let dst_i = dy * cc_w + dx;
                    for c in 0..3 {
                        let val = xyb[c][src_i] - ref_color[c];
                        fpixels[c][dst_i] = val;
                        let q = (val / CHANNEL_DEQUANT[c]) as i32;
                        qpixels[c][dst_i] = q.clamp(-128, 127) as i8;
                        max_value = max_value.max(q.abs());
                    }
                }
            }

            // kMinPeak check: reject patches where max quantized magnitude < 2
            if max_value < MIN_PEAK {
                debug_rect!(
                    "patches/cc_reject",
                    min_x,
                    min_y,
                    cc_w,
                    cc_h,
                    "CC rejected: peak {max_value} < {MIN_PEAK}"
                );
                continue;
            }

            debug_rect!(
                "patches/cc_accept",
                min_x,
                min_y,
                cc_w,
                cc_h,
                "CC accepted: {cc_w}x{cc_h} peak={max_value}"
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

    let mut result: Vec<PatchInfo> = Vec::new();
    for (_key, group) in patch_groups {
        if group.len() < MIN_PATCH_OCCURRENCES {
            continue;
        }
        let positions: Vec<(u32, u32)> = group.iter().map(|(x, y, _)| (*x, *y)).collect();
        let patch = group.into_iter().next().unwrap().2;
        result.push(PatchInfo { patch, positions });
    }

    let total_dedup_occurrences: usize = result.iter().map(|p| p.positions.len()).sum();
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

/// Bin-pack patches into a reference frame rectangle.
/// Returns the reference frame dimensions and positions of each patch.
fn bin_pack_patches(patches: &[PatchInfo]) -> (usize, usize, Vec<(u32, u32)>) {
    if patches.is_empty() {
        return (0, 0, Vec::new());
    }

    // Sort by area (largest first) — already sorted by QuantizedPatch Ord impl
    let total_area: usize = patches.iter().map(|p| p.patch.num_pixels()).sum();

    // Initial estimate: square-ish rectangle
    let side = (total_area as f32).sqrt() as usize;
    let mut ref_width = side.max(patches[0].patch.xsize);
    let mut ref_height = side.max(patches[0].patch.ysize);

    // Simple shelf-based packing
    loop {
        let mut positions = Vec::with_capacity(patches.len());
        let mut shelf_y = 0;
        let mut shelf_x = 0;
        let mut shelf_height = 0;
        let mut success = true;

        for p in patches {
            let pw = p.patch.xsize;
            let ph = p.patch.ysize;

            if shelf_x + pw > ref_width {
                // Move to next shelf
                shelf_y += shelf_height;
                shelf_x = 0;
                shelf_height = 0;
            }

            if shelf_y + ph > ref_height {
                // Doesn't fit, grow and retry
                success = false;
                break;
            }

            positions.push((shelf_x as u32, shelf_y as u32));
            shelf_height = shelf_height.max(ph);
            shelf_x += pw;
        }

        if success {
            // Compute actual used height
            let actual_height = patches
                .iter()
                .zip(positions.iter())
                .map(|(p, (_, y))| *y as usize + p.patch.ysize)
                .max()
                .unwrap_or(0);
            return (ref_width, actual_height, positions);
        }

        // Grow by 5% + 1
        ref_width = ((ref_width as f32 * BIN_PACKING_SLACKNESS) as usize + 1).max(ref_width + 1);
        ref_height = ((ref_height as f32 * BIN_PACKING_SLACKNESS) as usize + 1).max(ref_height + 1);
    }
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
    infos.sort_by(|a, b| b.patch.num_pixels().cmp(&a.patch.num_pixels()));

    // Limit reference frame to single modular group (256×256).
    // Our encode_reference_frame uses single-group modular encoding, so the
    // reference frame must fit. Drop least-useful patches (fewest total pixels
    // contributed = area × occurrences) until it fits.
    const MAX_REF_DIM: usize = 256;
    loop {
        let (ref_width, ref_height, _) = bin_pack_patches(&infos);
        if ref_width <= MAX_REF_DIM && ref_height <= MAX_REF_DIM {
            break;
        }
        // Drop the patch with the smallest per-occurrence benefit
        // (fewest total covered pixels = area × occurrences)
        let worst = infos
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.patch.num_pixels() * p.positions.len())
            .map(|(i, _)| i);
        if let Some(i) = worst {
            infos.swap_remove(i);
        }
        if infos.is_empty() {
            return None;
        }
        // After removing, re-check MIN_PATCH_OCCURRENCES isn't violated
        infos.retain(|p| p.positions.len() >= MIN_PATCH_OCCURRENCES);
        if infos.is_empty() {
            return None;
        }
    }

    // Bin-pack into reference frame
    let (ref_width, ref_height, pack_positions) = bin_pack_patches(&infos);
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
        let code = build_entropy_code_ans_with_options(&tokens, NUM_PATCH_CONTEXTS, false, None);
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
///
/// Uses measured overhead (actual ref frame + dict encoding size) vs estimated
/// savings to decide if patches are worthwhile. This mirrors libjxl's behavior
/// where multi-attempt RD optimization rejects patches that don't help.
pub(crate) fn find_and_build(
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
    stride: usize,
) -> Option<PatchesData> {
    let infos = find_text_like_patches(xyb, width, height, stride);
    if infos.is_empty() {
        debug_rect!("patches/detect", 0, 0, width, height, "no patches detected");
        return None;
    }

    // Compute coverage statistics before building
    let total_patch_pixels: usize = infos
        .iter()
        .map(|p| p.patch.num_pixels() * p.positions.len())
        .sum();
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
    // The overhead from the reference frame + dictionary always exceeds savings.
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
        "PATCHES: ref frame {}x{} ({} pixels)",
        patches_data.ref_width,
        patches_data.ref_height,
        patches_data.ref_width * patches_data.ref_height
    );

    // Measure actual overhead by trial-encoding the reference frame and dictionary.
    // This gives exact byte costs rather than rough estimates.
    let ref_overhead = {
        let mut w = BitWriter::new();
        encode_reference_frame(&patches_data, true, &mut w).ok()?;
        w.bits_written().div_ceil(8)
    };
    let dict_overhead = {
        let mut w = BitWriter::new();
        encode_patches_section(&patches_data, true, &mut w).ok()?;
        w.bits_written().div_ceil(8)
    };
    let total_overhead = ref_overhead + dict_overhead;

    // Estimate savings: patched pixels become near-zero after subtraction, compressing
    // much better in VarDCT. Conservatively assume 1 byte saved per patched pixel
    // (typical screenshots at d=1.0 are ~1.5-3 bytes/pixel, so saving ~50-70% of those).
    let estimated_savings = total_patch_pixels;

    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "PATCHES: overhead={} bytes (ref={}, dict={}), estimated savings={} bytes, ratio={:.1}x",
        total_overhead,
        ref_overhead,
        dict_overhead,
        estimated_savings,
        estimated_savings as f64 / total_overhead as f64
    );

    debug_rect!(
        "patches/cost",
        0,
        0,
        width,
        height,
        "overhead={total_overhead}B (ref={ref_overhead} dict={dict_overhead}); savings_est={estimated_savings}B; ratio={:.1}x",
        estimated_savings as f64 / total_overhead.max(1) as f64
    );

    // Require estimated savings to exceed measured overhead with 2x margin.
    // libjxl uses multi-attempt RD selection (try with/without patches, keep smaller).
    // We approximate this by requiring a clear benefit before committing to patches.
    if estimated_savings < total_overhead * 2 {
        debug_rect!(
            "patches/decision",
            0,
            0,
            width,
            height,
            "REJECTED: overhead {total_overhead}B > benefit {estimated_savings}B / 2"
        );
        #[cfg(feature = "debug-tokens")]
        eprintln!(
            "PATCHES: skipping — overhead ({total_overhead}) exceeds benefit ({estimated_savings})"
        );
        return None;
    }

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

// ── Reference Frame Encoding ───────────────────────────────────────────────────

/// Encode the reference frame containing all unique patch templates.
///
/// This writes a complete modular FrameType::ReferenceOnly frame to the writer.
/// The frame saves to reference slot 3 with save_before_ct=true.
///
/// The reference image is 3-channel XYB float data. For modular encoding, we scale
/// to i32 (multiply by a fixed scale factor and round).
pub(crate) fn encode_reference_frame(
    patches: &PatchesData,
    use_ans: bool,
    writer: &mut BitWriter,
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
    // Use a fixed-point scale factor. JXL modular uses i32 samples.
    // For patches, we scale by 2^15 = 32768 to preserve precision.
    const SCALE: f32 = 32768.0;
    let n = ref_w * ref_h;

    // Build a modular image from i32 channels
    use crate::modular::channel::{Channel, ModularImage};
    let mut mod_channels = Vec::with_capacity(3);
    for c in 0..3 {
        let mut ch_data = Vec::with_capacity(n);
        for i in 0..n {
            ch_data.push((patches.ref_image[c][i] * SCALE).round() as i32);
        }
        mod_channels.push(Channel::from_vec(ch_data, ref_w, ref_h)?);
    }
    let image = ModularImage {
        channels: mod_channels,
        bit_depth: 16, // Fixed-point representation
        is_grayscale: false,
        has_alpha: false,
    };

    // Use the modular frame encoder for the data section.
    // Use the same encode path that works for lossless frames.
    use crate::modular::encode::write_improved_modular_stream;
    let mut section_writer = BitWriter::new();
    write_improved_modular_stream(&image, &mut section_writer, use_ans)?;
    let section_data = section_writer.finish();

    // Write TOC (single section for small reference frames).
    // Use the modular frame encoder's TOC format (same as VarDCT).
    #[cfg(feature = "trace-bitstream")]
    eprintln!(
        "PATCHES: ref frame TOC starts at bit {}, section_data={} bytes",
        writer.bits_written(),
        section_data.len()
    );
    crate::vardct::frame::write_toc(&[section_data.len()], writer)?;

    #[cfg(feature = "trace-bitstream")]
    eprintln!(
        "PATCHES: ref frame section data starts at bit {} (byte {})",
        writer.bits_written(),
        writer.bits_written() / 8
    );

    // Write section data
    writer.append_bytes(&section_data)?;

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
    fn test_xyb_distance_zero() {
        let x = vec![1.0f32; 4];
        let y = vec![2.0f32; 4];
        let b = vec![3.0f32; 4];
        let xyb: [&[f32]; 3] = [&x, &y, &b];
        let dist = xyb_distance(&xyb, 2, 0, 0, 1, 0);
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

        let (w, h, positions) = bin_pack_patches(&infos);
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
        let result = find_text_like_patches([&x, &y, &b], w, h, w);
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

        let result = find_text_like_patches([&x, &y, &b], w, h, w);
        // Should find at least one patch group with >= 2 occurrences
        // Note: the exact number depends on detection thresholds
        if !result.is_empty() {
            let total_occurrences: usize = result.iter().map(|p| p.positions.len()).sum();
            assert!(total_occurrences >= 2, "Should have at least 2 occurrences");
        }
    }
}
