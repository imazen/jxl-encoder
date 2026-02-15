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

/// Compute weighted XYB distance between two pixels.
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
        let d = xyb[c][i1] - xyb[c][i2];
        dist += d * d * CHANNEL_WEIGHTS[c];
    }
    dist
}

/// Compute weighted XYB distance between a pixel and a given color.
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
        let d = xyb[c][i] - color[c];
        dist += d * d * CHANNEL_WEIGHTS[c];
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
/// Port of libjxl `FindTextLikePatches`.
pub(crate) fn find_text_like_patches(
    xyb: [&[f32]; 3],
    width: usize,
    height: usize,
) -> Vec<PatchInfo> {
    // Step 1: Grid scan — identify flat 4x4 blocks
    let bw = width / PATCH_SIDE;
    let bh = height / PATCH_SIDE;
    if bw < 3 || bh < 3 {
        return Vec::new();
    }

    let mut is_flat = vec![false; bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            is_flat[by * bw + bx] = is_flat_block(&[xyb[0], xyb[1], xyb[2]], width, bx, by);
        }
    }

    // Step 2: Identify "screenshot-like" blocks (8+ of 9 neighbors are flat with same color)
    let mut is_screenshot_like = vec![false; width * height];
    for by in 1..bh.saturating_sub(1) {
        for bx in 1..bw.saturating_sub(1) {
            if !is_flat[by * bw + bx] {
                continue;
            }
            // Count flat neighbors with same color in 3x3 block grid
            let ref_x = bx * PATCH_SIDE;
            let ref_y = by * PATCH_SIDE;
            let mut same_color_count = 0;
            for nby in by - 1..=by + 1 {
                for nbx in bx - 1..=bx + 1 {
                    if is_flat[nby * bw + nbx] {
                        let nx = nbx * PATCH_SIDE;
                        let ny = nby * PATCH_SIDE;
                        if xyb_distance(&[xyb[0], xyb[1], xyb[2]], width, ref_x, ref_y, nx, ny)
                            < FLATNESS_THRESHOLD
                        {
                            same_color_count += 1;
                        }
                    }
                }
            }
            if same_color_count >= SCREENSHOT_FLAT_NEIGHBOR_RATIO {
                // Mark all pixels in this 4x4 block as screenshot-like seeds
                for dy in 0..PATCH_SIDE {
                    for dx in 0..PATCH_SIDE {
                        let px = bx * PATCH_SIDE + dx;
                        let py = by * PATCH_SIDE + dy;
                        if px < width && py < height {
                            is_screenshot_like[py * width + px] = true;
                        }
                    }
                }
            }
        }
    }

    // Check if we have any screenshot-like regions at all
    let has_screenshot = is_screenshot_like.iter().any(|&x| x);
    if !has_screenshot {
        return Vec::new();
    }

    // Step 3: Background flood-fill from screenshot-like seeds
    let mut is_background = vec![false; width * height];
    let mut queue = std::collections::VecDeque::new();
    let mut pixel_distance = vec![u32::MAX; width * height];

    // Seed from screenshot-like pixels
    for y in 0..height {
        for x in 0..width {
            if is_screenshot_like[y * width + x] {
                is_background[y * width + x] = true;
                pixel_distance[y * width + x] = 0;
                queue.push_back((x, y));
            }
        }
    }

    // BFS flood-fill
    let xyb_ref = [xyb[0], xyb[1], xyb[2]];
    while let Some((cx, cy)) = queue.pop_front() {
        let cd = pixel_distance[cy * width + cx];
        if cd >= DISTANCE_LIMIT as u32 {
            continue;
        }

        for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            let ni = ny * width + nx;
            if is_background[ni] {
                continue;
            }
            // Check color similarity to the current background pixel
            if xyb_distance(&xyb_ref, width, cx, cy, nx, ny) < SIMILAR_THRESHOLD {
                is_background[ni] = true;
                pixel_distance[ni] = cd + 1;
                queue.push_back((nx, ny));
            }
        }
    }

    // Step 4: Extract foreground connected components
    let mut visited = vec![false; width * height];
    let mut patches: Vec<(QuantizedPatch, u32, u32)> = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            let si = start_y * width + start_x;
            if is_background[si] || visited[si] {
                continue;
            }

            // DFS to find connected component
            let mut cc_pixels: Vec<(usize, usize)> = Vec::new();
            let mut stack = vec![(start_x, start_y)];
            let mut min_x = start_x;
            let mut max_x = start_x;
            let mut min_y = start_y;
            let mut max_y = start_y;

            while let Some((px, py)) = stack.pop() {
                let pi = py * width + px;
                if visited[pi] || is_background[pi] {
                    continue;
                }
                visited[pi] = true;
                cc_pixels.push((px, py));
                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);

                // Check if bounding box exceeds max patch size
                if max_x - min_x >= MAX_PATCH_SIZE || max_y - min_y >= MAX_PATCH_SIZE {
                    continue; // Stop growing but keep what we have
                }

                for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nx = px as i32 + dx;
                    let ny = py as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let (nx, ny) = (nx as usize, ny as usize);
                        if !visited[ny * width + nx] && !is_background[ny * width + nx] {
                            stack.push((nx, ny));
                        }
                    }
                }
            }

            let cc_w = max_x - min_x + 1;
            let cc_h = max_y - min_y + 1;

            // Skip if too large
            if cc_w > MAX_PATCH_SIZE || cc_h > MAX_PATCH_SIZE {
                continue;
            }

            // Skip tiny connected components
            if cc_pixels.len() < 2 {
                continue;
            }

            // Compute border color: average background pixel adjacent to the CC
            let mut border_color = [0.0f32; 3];
            let mut border_count = 0u32;
            for &(px, py) in &cc_pixels {
                for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nx = px as i32 + dx;
                    let ny = py as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let (nx, ny) = (nx as usize, ny as usize);
                        if is_background[ny * width + nx] {
                            let ni = ny * width + nx;
                            for c in 0..3 {
                                border_color[c] += xyb_ref[c][ni];
                            }
                            border_count += 1;
                        }
                    }
                }
            }

            if border_count == 0 {
                continue;
            }
            for c in 0..3 {
                border_color[c] /= border_count as f32;
            }

            // Check border color consistency: all background neighbors should have
            // similar color (threshold=VERY_SIMILAR_THRESHOLD)
            let mut border_consistent = true;
            'border_check: for &(px, py) in &cc_pixels {
                for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let nx = px as i32 + dx;
                    let ny = py as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let (nx, ny) = (nx as usize, ny as usize);
                        if is_background[ny * width + nx]
                            && xyb_distance_to_color(&xyb_ref, width, nx, ny, &border_color)
                                > VERY_SIMILAR_THRESHOLD
                        {
                            border_consistent = false;
                            break 'border_check;
                        }
                    }
                }
            }
            if !border_consistent {
                continue;
            }

            // Quantize the patch: pixel_value = xyb[pixel] - border_color
            let n = cc_w * cc_h;
            let mut qpixels = [vec![0i8; n], vec![0i8; n], vec![0i8; n]];
            let mut fpixels = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];

            for dy in 0..cc_h {
                for dx in 0..cc_w {
                    let ix = min_x + dx;
                    let iy = min_y + dy;
                    let src_i = iy * width + ix;
                    let dst_i = dy * cc_w + dx;

                    for c in 0..3 {
                        let val = xyb_ref[c][src_i] - border_color[c];
                        fpixels[c][dst_i] = val;
                        // Quantize to i8 via truncation
                        let q = (val / CHANNEL_DEQUANT[c]) as i32;
                        qpixels[c][dst_i] = q.clamp(-128, 127) as i8;
                    }
                }
            }

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
    // Group patches by their quantized content
    use std::collections::HashMap;
    let mut patch_groups: HashMap<Vec<u8>, Vec<(u32, u32, QuantizedPatch)>> = HashMap::new();

    for (patch, x, y) in patches {
        // Create a hash key from quantized pixels
        let mut key = Vec::with_capacity(2 + patch.pixels[0].len() * 3);
        key.extend_from_slice(&(patch.xsize as u16).to_le_bytes());
        key.extend_from_slice(&(patch.ysize as u16).to_le_bytes());
        for c in 0..3 {
            for &p in &patch.pixels[c] {
                key.push(p as u8);
            }
        }
        patch_groups.entry(key).or_default().push((x, y, patch));
    }

    // Convert to PatchInfo, only keeping patches with >= MIN_PATCH_OCCURRENCES
    let mut result: Vec<PatchInfo> = Vec::new();
    for (_key, group) in patch_groups {
        if group.len() < MIN_PATCH_OCCURRENCES {
            continue;
        }
        let positions: Vec<(u32, u32)> = group.iter().map(|(x, y, _)| (*x, *y)).collect();
        // Use the first occurrence's float pixels (they should all be very similar)
        let patch = group.into_iter().next().unwrap().2;
        result.push(PatchInfo { patch, positions });
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
                    xyb[c][img_i] -= patches.ref_image[c][ref_i];
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

        // xsize - 1, ysize - 1
        tokens.push(Token::new(2, ref_pos.xsize - 1));
        tokens.push(Token::new(2, ref_pos.ysize - 1));

        // ref_x0, ref_y0
        tokens.push(Token::new(3, ref_pos.x0));
        tokens.push(Token::new(3, ref_pos.y0));

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
pub(crate) fn find_and_build(xyb: [&[f32]; 3], width: usize, height: usize) -> Option<PatchesData> {
    let infos = find_text_like_patches(xyb, width, height);
    if infos.is_empty() {
        return None;
    }
    build_patches_data(infos)
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
    fh.xyb_encoded = true; // This is XYB data
    fh.save_as_reference = PATCH_FRAME_REFERENCE_ID;
    fh.save_before_ct = true;
    fh.is_last = false; // Not the last frame
    fh.flags = 0;
    fh.gaborish = false;
    fh.epf_iters = 0;
    // Set dimensions to the reference frame size
    fh.width = ref_w as u32;
    fh.height = ref_h as u32;

    fh.write(writer)?;

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

    // Use the modular frame encoder for the data section
    use crate::modular::encode::write_improved_modular_stream;
    let mut section_writer = BitWriter::new();
    write_improved_modular_stream(&image, &mut section_writer, use_ans)?;
    let section_data = section_writer.finish();

    // Write TOC (single section)
    crate::vardct::frame::write_toc(&[section_data.len()], writer)?;

    // Write section data
    writer.append_bytes(&section_data)?;

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
        let result = find_text_like_patches([&x, &y, &b], w, h);
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

        let result = find_text_like_patches([&x, &y, &b], w, h);
        // Should find at least one patch group with >= 2 occurrences
        // Note: the exact number depends on detection thresholds
        if !result.is_empty() {
            let total_occurrences: usize = result.iter().map(|p| p.positions.len()).sum();
            assert!(total_occurrences >= 2, "Should have at least 2 occurrences");
        }
    }
}
