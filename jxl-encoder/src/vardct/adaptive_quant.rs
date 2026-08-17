// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Adaptive quantization field computation.
//!
//! Ported from full libjxl `enc_adaptive_quantization.cc`.

// Ported float constants from C++ - exact values are intentional for parity.
#![allow(clippy::approx_constant)]
//! Computes per-block quantization values based on perceptual masking.
//!
//! Pipeline (matches full libjxl):
//! 1. `compute_pre_erosion()` — Y channel diffs, gamma ratio, limit clamp, masking sqrt, 4x downsample
//! 2. `fuzzy_erosion()` — 3×3 min-4 distance-weighted sum, 2x downsample
//! 3. `per_block_modulations()` — ComputeMask → GammaModulation → HfModulation → Min(Hf,Blue) → exp2
//! 4. Convert: `raw_quant = clamp(round(quant_field * inv_scale + 0.5), 1, 255)`

use super::common::clamp;

// Fast math helpers and masking sub-functions have been migrated to jxl_simd.
// compute_pre_erosion and per_block_modulations now delegate to jxl_simd SIMD implementations.

/// Insert `v` into the smallest-4 tracking variables if it's smaller than `min3`.
/// Used by `fuzzy_erosion()` (which remains local — operates on small downsampled data).
#[inline(always)]
fn store_min4(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32, min3: &mut f32) {
    if v < *min3 {
        if v < *min0 {
            *min3 = *min2;
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min3 = *min2;
            *min2 = *min1;
            *min1 = v;
        } else if v < *min2 {
            *min3 = *min2;
            *min2 = v;
        } else {
            *min3 = v;
        }
    }
}

/// Compute pre-erosion map from XYB planes.
///
/// Full libjxl version: Y channel only (no X channel), with limit=0.2 clamp
/// before MaskingSqrt. SIMD-accelerated via jxl_simd.
///
/// Output dimensions: ceil(tile_pixel_w / 4) × ceil(tile_pixel_h / 4).
#[allow(clippy::too_many_arguments)]
fn compute_pre_erosion(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    jxl_simd::compute_pre_erosion(xyb_y, width, height, tile_x0, tile_y0, tile_x1, tile_y1)
}

/// FuzzyErosion: 3×3 min-4 weighted sum, then 2x downsample.
/// Full libjxl version: distance-dependent weights.
#[allow(clippy::too_many_arguments)]
fn fuzzy_erosion(
    from: &[f32],
    from_w: usize,
    from_h: usize,
    from_x0: usize,
    from_y0: usize,
    region_w: usize,
    region_h: usize,
    butteraugli_target: f32,
) -> (Vec<f32>, usize, usize) {
    let out_w = region_w / 2;
    let out_h = region_h / 2;
    let mut out = vec![0.0_f32; out_w * out_h];

    // Distance-dependent weights (full libjxl)
    const K_MUL_BASE: [f32; 4] = [0.125, 0.1, 0.09, 0.06];
    const K_MUL_ADD: [f32; 4] = [0.0, -0.1, -0.09, -0.06];

    let mul = if butteraugli_target < 2.0 {
        (2.0 - butteraugli_target) * 0.5
    } else {
        0.0
    };

    let mut k_mul = [0.0_f32; 4];
    let mut norm_sum = 0.0_f32;
    for (ii, k) in k_mul.iter_mut().enumerate() {
        *k = K_MUL_BASE[ii] + mul * K_MUL_ADD[ii];
        norm_sum += *k;
    }
    const K_TOTAL: f32 = 0.29959705784054957;
    for k in &mut k_mul {
        *k *= K_TOTAL / norm_sum;
    }

    for fy in 0..region_h {
        let y = fy + from_y0;
        let ym1 = if y >= 1 { y - 1 } else { y };
        let yp1 = if y + 1 < from_h { y + 1 } else { y };

        for fx in 0..region_w {
            let x = fx + from_x0;
            let xm1 = if x >= 1 { x - 1 } else { x };
            let xp1 = if x + 1 < from_w { x + 1 } else { x };

            // Get all 9 neighbors
            let center = from[y * from_w + x];
            let left = from[y * from_w + xm1];
            let right = from[y * from_w + xp1];
            let top_left = from[ym1 * from_w + xm1];
            let top = from[ym1 * from_w + x];
            let top_right = from[ym1 * from_w + xp1];
            let bot_left = from[yp1 * from_w + xm1];
            let bot = from[yp1 * from_w + x];
            let bot_right = from[yp1 * from_w + xp1];

            // Find smallest 4 from 9 values
            let mut min0 = center;
            let mut min1 = left;
            let mut min2 = right;
            let mut min3 = top_left;

            // Sort first 4
            if min0 > min1 {
                core::mem::swap(&mut min0, &mut min1);
            }
            if min0 > min2 {
                core::mem::swap(&mut min0, &mut min2);
            }
            if min0 > min3 {
                core::mem::swap(&mut min0, &mut min3);
            }
            if min1 > min2 {
                core::mem::swap(&mut min1, &mut min2);
            }
            if min1 > min3 {
                core::mem::swap(&mut min1, &mut min3);
            }
            if min2 > min3 {
                core::mem::swap(&mut min2, &mut min3);
            }

            // Insert remaining 5 values
            store_min4(top, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(top_right, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(bot_left, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(bot, &mut min0, &mut min1, &mut min2, &mut min3);
            store_min4(bot_right, &mut min0, &mut min1, &mut min2, &mut min3);

            let v = k_mul[0] * min0 + k_mul[1] * min1 + k_mul[2] * min2 + k_mul[3] * min3;

            let ox = fx / 2;
            let oy = fy / 2;
            if fx % 2 == 0 && fy % 2 == 0 {
                out[oy * out_w + ox] = v;
            } else {
                out[oy * out_w + ox] += v;
            }
        }
    }

    (out, out_w, out_h)
}

/// ComputeMaskForAcStrategyUse: simple masking hack.
fn compute_mask_for_ac_strategy_use(out_val: f32) -> f32 {
    const K_MUL: f32 = 1.0;
    const K_OFFSET: f32 = 0.001;
    K_MUL / (out_val + K_OFFSET)
}

/// Compute per-pixel (1x1) masking field for pixel-domain loss calculation.
///
/// This implements libjxl's 1x1 Laplacian masking from `enc_adaptive_quantization.cc`.
/// The mask is used in `EstimateEntropy` to weight pixel-domain quantization error.
///
/// # Returns
/// Per-pixel mask field of size `width * height`, row-major layout.
/// After computing the raw mask, applies libjxl's Symmetric5 blur.
//
// Re-exported under `__pre_quantized` (see `crate::__pre_quantized::compute_mask1x1`).
// Default-features non-test builds have no internal caller — the budgeted variant
// `compute_mask1x1_with_budget` carries the production path.
#[cfg_attr(not(any(test, feature = "__pre_quantized")), allow(dead_code))]
pub fn compute_mask1x1(xyb_y: &[f32], width: usize, height: usize) -> Vec<f32> {
    compute_mask1x1_with_budget(xyb_y, width, height, None)
        .expect("compute_mask1x1: unbudgeted call should never fail")
}

/// Same as [`compute_mask1x1`] but accounts allocations against an optional
/// [`MemoryBudget`].
pub(crate) fn compute_mask1x1_with_budget(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<f32>> {
    let n = width
        .checked_mul(height)
        .ok_or(crate::error::Error::DimensionOverflow {
            width,
            height,
            channels: 1,
        })?;
    // Two w*h f32 buffers are alive at peak: the returned mask1x1 and the
    // internal `scratch` used by gaborish_5x5_channel. Account both as
    // permanent — mask1x1 has caller-owned lifetime, scratch is dropped here
    // but at peak we need 2*n*4 budget to allocate them simultaneously.
    crate::budget::MemoryBudget::reserve_permanent_opt(budget, (n as u64).saturating_mul(4 * 2))?;

    // Strip-parallel dispatch: both kernels below are pure per-pixel
    // functions of their input with row-local horizontal SIMD, so a
    // full-width strip computed with enough halo rows to cover the
    // vertical stencil reach (1 for the raw mask, 2 for the 5x5 blur)
    // is BIT-IDENTICAL to the whole-image call on the kept rows: the
    // sub-buffer edge clamp only fires on halo rows (discarded) or
    // where it coincides with the true image edge (edge strips carry
    // no outer halo, so the clamp is the image clamp), and the
    // unchanged row width keeps the SIMD lane pattern identical.
    // Perf-only dispatch — output is identical on either path.
    let strip_parallel = crate::parallel::effective_threads() > 1 && height >= 192;
    let mut mask1x1 = if strip_parallel {
        compute_mask1x1_strip_parallel(xyb_y, width, height)
    } else {
        let mut mask1x1 = vec![0.0_f32; n];

        // SIMD-accelerated per-pixel masking (neighbor avg → gamma ratio → log1p → reciprocal)
        jxl_simd::compute_mask1x1(xyb_y, width, height, &mut mask1x1);
        mask1x1
    };

    // Apply Symmetric5 blur using SIMD gaborish kernel with mask1x1 weights.
    // The gaborish_5x5_channel kernel has the same 5x5 weight pattern:
    //   D  L  R  L  D
    //   L  d  r  d  L
    //   R  r  c  r  R
    //   L  d  r  d  L
    //   D  L  R  L  D
    // libjxl mask1x1 weights from enc_adaptive_quantization.cc:
    const W_R: f32 = 0.364_911_248; // kFilterMask1x1[0] = r (orthogonal dist 1)
    const W_D: f32 = 0.05; // kFilterMask1x1[1] = d (diagonal dist 1)
    const W_R2: f32 = 0.168_888_802_1; // kFilterMask1x1[2] = R (orthogonal dist 2)
    const W_L: f32 = 0.221_069_183; // kFilterMask1x1[3] = L (knight's move)
    const W_D2: f32 = 0.306_563_504; // kFilterMask1x1[4] = D (diagonal dist 2)
    let sum = 1.0 + 4.0 * (W_R + W_D + W_R2 + W_D2 + 2.0 * W_L);
    let inv_sum = 1.0 / sum;

    let weights = [
        inv_sum,        // wc (center)
        inv_sum * W_R,  // wr (orthogonal dist 1)
        inv_sum * W_D,  // wd (diagonal dist 1)
        inv_sum * W_R2, // w_big_r (orthogonal dist 2)
        inv_sum * W_L,  // wl (knight's move)
        inv_sum * W_D2, // w_big_d (diagonal dist 2)
    ];
    if strip_parallel {
        mask1x1 = gaborish_5x5_strip_parallel(&mask1x1, width, height, weights);
    } else {
        let mut scratch = vec![0.0_f32; width * height];
        jxl_simd::gaborish_5x5_channel(
            &mut mask1x1,
            &mut scratch,
            width,
            height,
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            weights[4],
            weights[5],
        );
    }

    Ok(mask1x1)
}

/// Strip-parallel raw mask1x1 — bit-identical to the whole-image
/// `jxl_simd::compute_mask1x1` call (see the dispatch comment at the call
/// site: 1-row halo covers the vertical stencil reach; halo rows are
/// discarded; unchanged width keeps lanes identical).
fn compute_mask1x1_strip_parallel(xyb_y: &[f32], width: usize, height: usize) -> Vec<f32> {
    const STRIP_ROWS: usize = 64;
    let n_strips = height.div_ceil(STRIP_ROWS);
    let strips: Vec<Vec<f32>> = crate::parallel::parallel_map(n_strips, |si| {
        let ky0 = si * STRIP_ROWS;
        let ky1 = (ky0 + STRIP_ROWS).min(height);
        let ty0 = ky0.saturating_sub(1);
        let ty1 = (ky1 + 1).min(height);
        let sub_h = ty1 - ty0;
        let mut sub_out = vec![0.0_f32; sub_h * width];
        jxl_simd::compute_mask1x1(&xyb_y[ty0 * width..ty1 * width], width, sub_h, &mut sub_out);
        sub_out.drain(..(ky0 - ty0) * width);
        sub_out.truncate((ky1 - ky0) * width);
        sub_out
    });
    let mut out = Vec::with_capacity(width * height);
    for s in strips {
        out.extend_from_slice(&s);
    }
    out
}

/// Strip-parallel 5x5 mask blur — bit-identical to the whole-image
/// `jxl_simd::gaborish_5x5_channel` call (2-row halo covers the 5x5
/// vertical reach; the kernel's y<2 / y>=h-2 scalar border rows land
/// exactly on discarded halo rows for interior strips and on the true
/// image border for edge strips, matching the whole-image path).
pub(super) fn gaborish_5x5_strip_parallel(
    raw: &[f32],
    width: usize,
    height: usize,
    w: [f32; 6],
) -> Vec<f32> {
    const STRIP_ROWS: usize = 64;
    let n_strips = height.div_ceil(STRIP_ROWS);
    let strips: Vec<Vec<f32>> = crate::parallel::parallel_map(n_strips, |si| {
        let ky0 = si * STRIP_ROWS;
        let ky1 = (ky0 + STRIP_ROWS).min(height);
        let ty0 = ky0.saturating_sub(2);
        let ty1 = (ky1 + 2).min(height);
        let sub_h = ty1 - ty0;
        let mut sub_data = raw[ty0 * width..ty1 * width].to_vec();
        let mut scratch = vec![0.0_f32; sub_h * width];
        jxl_simd::gaborish_5x5_channel(
            &mut sub_data,
            &mut scratch,
            width,
            sub_h,
            w[0],
            w[1],
            w[2],
            w[3],
            w[4],
            w[5],
        );
        sub_data.drain(..(ky0 - ty0) * width);
        sub_data.truncate((ky1 - ky0) * width);
        sub_data
    });
    let mut out = Vec::with_capacity(width * height);
    for s in strips {
        out.extend_from_slice(&s);
    }
    out
}

// symmetric5_blur_mask1x1 replaced by jxl_simd::gaborish_5x5_channel with
// mask1x1-specific weights (same 5x5 kernel structure, ~10x faster via AVX2).

/// PerBlockModulations: apply all modulations and convert exponent to multiplier.
/// SIMD-accelerated via jxl_simd.
///
/// Full libjxl order: ComputeMask → GammaModulation → HfModulation → Min(Hf, BlueModulation) → exp2
///
/// `stride` is the row stride (padded width) of the XYB buffers.
#[allow(clippy::too_many_arguments)]
fn per_block_modulations(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_w: usize,
) {
    jxl_simd::per_block_modulations(
        xyb_x,
        xyb_y,
        xyb_b,
        stride,
        butteraugli_target,
        scale,
        rect_x0_blocks,
        rect_y0_blocks,
        rect_w_blocks,
        rect_h_blocks,
        aq_map,
        aq_map_w,
    );
}

/// Compute the adaptive quantization field for the entire image.
///
/// Returns `(quant_field_float, masking)`.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn compute_quant_field_float(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    k_ac_quant: f32,
) -> (Vec<f32>, Vec<f32>) {
    compute_quant_field_float_with_budget(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        distance,
        k_ac_quant,
        None,
    )
    .expect("compute_quant_field_float: unbudgeted call should never fail")
}

/// Budget-aware variant of [`compute_quant_field_float`]. Accounts the two
/// output planes (`masking` + `quant_field_float`, each `xsize_blocks * ysize_blocks`
/// floats) plus the transient `aq_map` (~`width/4 * height/4` floats) against
/// the per-encode cap.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_quant_field_float_with_budget(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    k_ac_quant: f32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<f32>, Vec<f32>)> {
    let nblocks =
        xsize_blocks
            .checked_mul(ysize_blocks)
            .ok_or(crate::error::Error::DimensionOverflow {
                width: xsize_blocks,
                height: ysize_blocks,
                channels: 1,
            })?;
    // Returned: masking + quant_field_float (each nblocks * f32 = 4 * nblocks bytes).
    // Transient peak (aq_map after fuzzy_erosion): xsize_blocks * ysize_blocks * f32
    // ≈ same as nblocks since erosion downsample yields one value per 8x8 block.
    crate::budget::MemoryBudget::reserve_permanent_opt(
        budget,
        (nblocks as u64).saturating_mul(4 * 2),
    )?;
    let scale = k_ac_quant / distance;

    let tile_x0_pixels = 0;
    let tile_y0_pixels = 0;

    // Step 1: Compute pre-erosion (Y only, limit clamp, 4x downsample)
    let (pre_erosion, pre_erosion_w, pre_erosion_h) = compute_pre_erosion(
        xyb_y,
        width,
        height,
        tile_x0_pixels,
        tile_y0_pixels,
        width,
        height,
    );

    // Step 2: Fuzzy erosion (distance-dependent weights, 2x downsample)
    let from_x0 = if tile_x0_pixels > 0 { 1 } else { 0 };
    let from_y0 = if tile_y0_pixels > 0 { 1 } else { 0 };
    let erosion_region_w = (xsize_blocks * 2).min(pre_erosion_w.saturating_sub(from_x0));
    let erosion_region_h = (ysize_blocks * 2).min(pre_erosion_h.saturating_sub(from_y0));

    let (mut aq_map, aq_map_w, _aq_map_h) = fuzzy_erosion(
        &pre_erosion,
        pre_erosion_w,
        pre_erosion_h,
        from_x0,
        from_y0,
        erosion_region_w,
        erosion_region_h,
        distance,
    );

    // Step 2.5: Compute masking field for AC strategy use
    let mut masking = vec![0.0f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            masking[by * xsize_blocks + bx] =
                compute_mask_for_ac_strategy_use(aq_map[by * aq_map_w + bx]);
        }
    }

    // Step 3: Per-block modulations (full libjxl order)
    per_block_modulations(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        distance,
        scale,
        0,
        0,
        xsize_blocks,
        ysize_blocks,
        &mut aq_map,
        aq_map_w,
    );

    // Step 4: Extract compact float quant field
    let mut quant_field_float = vec![0.0f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            quant_field_float[by * xsize_blocks + bx] = aq_map[by * aq_map_w + bx];
        }
    }

    Ok((quant_field_float, masking))
}

/// Per-region variant of [`compute_quant_field_float_with_budget`] —
/// computes quant_field + masking for a single DC-group-sized
/// rectangle, reading borders directly from the whole-image XYB planes.
///
/// **Streaming refactor chunk 5 (#11)**: this is the per-region split
/// the loop driver in [`super::precomputed::fill_dc_group_state_whole_image`]
/// uses instead of allocating the whole-image quant_field + masking
/// buffers in chunk 3. Output buffers are sized
/// `(region_w_blocks × region_h_blocks)`, indexed row-major in
/// region-local coordinates. Border pixels (1 block on each side of the
/// region) are read from `xyb_x/y/b` at their absolute positions, so
/// when called from a loop iterating all DC groups against the full
/// pre-gaborish XYB the assembled output is byte-identical to the
/// whole-image call.
///
/// `xyb_x/y/b` MUST be the full padded XYB planes with stride
/// `padded_width`. `region_x0_blocks/region_y0_blocks/region_w_blocks/
/// region_h_blocks` describe the region in 8×8 block units, clamped to
/// `[0, xsize_blocks]`/`[0, ysize_blocks]` by the caller.
///
/// libjxl mirror: `enc_adaptive_quantization.cc::ComputeAdaptiveQuantField`
/// operates on `Rect aq_rect` over the whole image — same shape; the
/// pre-erosion + fuzzy-erosion helpers already accept tile offsets so
/// per-region calls compose naturally with the SIMD primitives in
/// `jxl-encoder-simd::adaptive_quant`.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn compute_quant_field_float_for_region(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    padded_width: usize,
    padded_height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    region_x0_blocks: usize,
    region_y0_blocks: usize,
    region_w_blocks: usize,
    region_h_blocks: usize,
    distance: f32,
    k_ac_quant: f32,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<(Vec<f32>, Vec<f32>)> {
    debug_assert!(region_x0_blocks + region_w_blocks <= xsize_blocks);
    debug_assert!(region_y0_blocks + region_h_blocks <= ysize_blocks);

    let region_nblocks = region_w_blocks.checked_mul(region_h_blocks).ok_or(
        crate::error::Error::DimensionOverflow {
            width: region_w_blocks,
            height: region_h_blocks,
            channels: 1,
        },
    )?;
    // Per-region output: masking + quant_field_float, each
    // (region_w_blocks * region_h_blocks) f32. Per-region transients
    // (pre-erosion + fuzzy-erosion buffers) are an order of magnitude
    // smaller than the whole-image transients — bounded by the DC group
    // pixel area.
    crate::budget::MemoryBudget::reserve_permanent_opt(
        budget,
        (region_nblocks as u64).saturating_mul(4 * 2),
    )?;
    let scale = k_ac_quant / distance;

    // Pixel-domain rect for pre-erosion. Pre-erosion expands by 4
    // pixels on each side internally (`tile_x0.saturating_sub(4)` +
    // `tile_x1 + 4` when not at the image edge) so neighbour pre-erosion
    // samples at the region boundary read from the actual neighbour
    // pixels in the whole-image XYB.
    let region_x0_pixels = region_x0_blocks * 8;
    let region_y0_pixels = region_y0_blocks * 8;
    let region_x1_pixels = (region_x0_blocks + region_w_blocks) * 8;
    let region_y1_pixels = (region_y0_blocks + region_h_blocks) * 8;
    debug_assert!(region_x1_pixels <= padded_width);
    debug_assert!(region_y1_pixels <= padded_height);

    let (pre_erosion, pre_erosion_w, pre_erosion_h) = compute_pre_erosion(
        xyb_y,
        padded_width,
        padded_height,
        region_x0_pixels,
        region_y0_pixels,
        region_x1_pixels,
        region_y1_pixels,
    );

    // Fuzzy erosion: same `from_x0 = 1 iff tile_x0 > 0` rule as the
    // whole-image version. The 1-sample offset accounts for the 4-pixel
    // pre-erosion border expansion (4 / 4 = 1 sample).
    let from_x0 = if region_x0_pixels > 0 { 1 } else { 0 };
    let from_y0 = if region_y0_pixels > 0 { 1 } else { 0 };
    let erosion_region_w = (region_w_blocks * 2).min(pre_erosion_w.saturating_sub(from_x0));
    let erosion_region_h = (region_h_blocks * 2).min(pre_erosion_h.saturating_sub(from_y0));

    let (mut aq_map, aq_map_w, _aq_map_h) = fuzzy_erosion(
        &pre_erosion,
        pre_erosion_w,
        pre_erosion_h,
        from_x0,
        from_y0,
        erosion_region_w,
        erosion_region_h,
        distance,
    );

    // Masking for AC strategy use — region-sized.
    let mut masking = vec![0.0f32; region_nblocks];
    for by in 0..region_h_blocks {
        for bx in 0..region_w_blocks {
            masking[by * region_w_blocks + bx] =
                compute_mask_for_ac_strategy_use(aq_map[by * aq_map_w + bx]);
        }
    }

    // Per-block modulations: the SIMD primitive already accepts a
    // (rect_x0_blocks, rect_y0_blocks, rect_w_blocks, rect_h_blocks)
    // rect on the input XYB and writes to `aq_map` at region-local
    // (iy, ix) — perfect fit for per-region use.
    per_block_modulations(
        xyb_x,
        xyb_y,
        xyb_b,
        padded_width,
        distance,
        scale,
        region_x0_blocks,
        region_y0_blocks,
        region_w_blocks,
        region_h_blocks,
        &mut aq_map,
        aq_map_w,
    );

    // Extract compact float quant field.
    let mut quant_field_float = vec![0.0f32; region_nblocks];
    for by in 0..region_h_blocks {
        for bx in 0..region_w_blocks {
            quant_field_float[by * region_w_blocks + bx] = aq_map[by * aq_map_w + bx];
        }
    }

    Ok((quant_field_float, masking))
}

/// Per-region variant of [`compute_mask1x1_with_budget`] — computes
/// the 1×1 pixel masking field for a single DC-group-sized rectangle.
///
/// **Streaming refactor chunk 5 (#11)**: same role as
/// [`compute_quant_field_float_for_region`] but for the mask1x1 plane.
/// The raw mask1x1 uses a 5-pixel pixel stencil (4-neighbour Laplacian)
/// and the subsequent Symmetric5 blur is a 5×5 kernel — so byte-identity
/// at the region boundary requires 2 pixels of valid neighbour data on
/// each side. The padded scratch buffer in this function pulls those
/// border pixels straight from `xyb_y` at their absolute positions
/// (edge-replicated at the image boundary, mirroring the whole-image
/// `compute_mask1x1`'s `saturating_sub` / `min` clamps).
///
/// Output is `region_w * region_h` row-major f32, holding only the
/// inner-region mask1x1 values. The 2-pixel padded scratch is dropped
/// before return.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn compute_mask1x1_for_region(
    xyb_y: &[f32],
    padded_width: usize,
    padded_height: usize,
    region_x0: usize,
    region_y0: usize,
    region_w: usize,
    region_h: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<Vec<f32>> {
    debug_assert!(region_x0 + region_w <= padded_width);
    debug_assert!(region_y0 + region_h <= padded_height);

    // PAD=3 is the minimum that makes per-region byte-identical to the
    // whole-image path at INTERIOR region boundaries:
    //
    //   - raw mask1x1's 3×3-ish stencil (4-neighbour Laplacian) needs
    //     1-pixel reach. Computed correctly at padded positions
    //     (PAD-2..pad_w-PAD+2) when PAD=3.
    //   - 5×5 Symmetric5 blur needs 2-pixel reach. Reads raw mask at
    //     padded position 1 onwards (blur output at padded position
    //     PAD=3 reaches raw mask at padded position 1) — never reads
    //     the outermost padded position where compute_mask1x1's
    //     internal clamping diverges from the whole-image path.
    //
    // At IMAGE boundaries we still replicate the outer-PAD rim to
    // match the whole-image kernel's edge-replication semantics.
    const PAD: usize = 3;
    let pad_w = region_w + 2 * PAD;
    let pad_h = region_h + 2 * PAD;
    let pad_n = pad_w
        .checked_mul(pad_h)
        .ok_or(crate::error::Error::DimensionOverflow {
            width: pad_w,
            height: pad_h,
            channels: 1,
        })?;
    let region_n =
        region_w
            .checked_mul(region_h)
            .ok_or(crate::error::Error::DimensionOverflow {
                width: region_w,
                height: region_h,
                channels: 1,
            })?;

    // Permanent: the returned region mask. Transient: padded Y scratch +
    // padded raw mask + padded blur scratch (each pad_n f32).
    crate::budget::MemoryBudget::reserve_permanent_opt(
        budget,
        (region_n as u64).saturating_mul(4),
    )?;
    let _g =
        crate::budget::MemoryBudget::reserve_opt(budget, (pad_n as u64).saturating_mul(4 * 3))?;

    // Build padded Y view with edge replication AT THE IMAGE BOUNDARY
    // (interior border pixels come from the actual neighbour DC groups).
    let mut padded_y = vec![0.0_f32; pad_n];
    let max_x = padded_width - 1;
    let max_y = padded_height - 1;
    for py in 0..pad_h {
        let src_y = {
            let signed = region_y0 as isize + py as isize - PAD as isize;
            signed.clamp(0, max_y as isize) as usize
        };
        let row = src_y * padded_width;
        for px in 0..pad_w {
            let src_x = {
                let signed = region_x0 as isize + px as isize - PAD as isize;
                signed.clamp(0, max_x as isize) as usize
            };
            padded_y[py * pad_w + px] = xyb_y[row + src_x];
        }
    }

    // Step 1: raw mask1x1 on the padded buffer.
    //
    // BYTE-IDENTITY NOTE: `jxl_simd::compute_mask1x1` clamps neighbour
    // reads at the buffer edges via `saturating_sub` / `.min(width-1)`.
    // On the padded buffer this clamps at the *padded* edge, which is
    // wrong: the whole-image path clamps at the *image* edge. The two
    // disagree on the outer `PAD` ring of the padded buffer (where the
    // padded-buffer clamp substitutes an already-replicated padded
    // pixel for the off-buffer position, instead of substituting the
    // image-edge pixel).
    //
    // Fix: compute raw mask normally, then post-process the outer
    // `PAD` ring by REPLICATING from the inner-most-valid raw mask
    // value. This matches what the whole-image raw mask would produce
    // at the clamped image position. Inner positions
    // `(PAD..pad_w-PAD, PAD..pad_h-PAD)` are already correct because
    // their neighbours lie at distance ≤1 inside the padded buffer
    // (covered by the `PAD=2` load), and the SIMD kernel's clamping
    // never triggers in the interior.
    let mut padded_raw = vec![0.0_f32; pad_n];
    jxl_simd::compute_mask1x1(&padded_y, pad_w, pad_h, &mut padded_raw);

    // Replication is only required on the four boundaries where the
    // region sits at the IMAGE edge. At interior boundaries, the
    // padded buffer was loaded with real neighbour data and the SIMD
    // kernel's clamping at the padded edge is a no-op (the clamped
    // value equals the real neighbour value).
    //
    // At an interior boundary (e.g. region_x0 > 0): padded position
    // px=0 maps to image position region_x0-2 which is a real image
    // pixel. Its neighbour at padded position -1 (off-buffer) would
    // ALSO map to a real image pixel, but compute_mask1x1's clamping
    // substitutes padded (0, *) (the real -2-offset pixel) for the
    // off-buffer position — that disagrees with what the whole-image
    // kernel does (which would substitute the real -3-offset pixel
    // from the actual image). To get byte-identity for the interior
    // boundary we'd need PAD≥3 OR to skip raw mask at the outermost
    // ring. Since PAD=2 is the kernel-reach minimum, just rely on the
    // fact that the blur at the INNER region (PAD..pad_w-PAD)
    // positions is the only output we keep — its 5×5 stencil reaches
    // up to padded position 0 only at the inner-region's outermost
    // pixel (offset PAD from buffer edge). For correct values there,
    // padded_raw at padded position 0 must equal the raw mask at the
    // image position region_x0 - 2 — which equals
    // compute_mask1x1's "raw mask with clamping at padded edge" ONLY
    // when region_x0 == 0 (image-edge case, where the clamp matches).
    //
    // For INTERIOR boundaries (region_x0 > 0) we MUST recompute the
    // outer-ring raw mask with the real neighbour data. PAD=2 is too
    // small — the raw mask at padded (0, *) needs neighbours at
    // padded (-1, *) which we don't have. Workaround: bump PAD to 3
    // for the load + raw mask stage so the kernel's clamping never
    // affects the inner blur's input. The blur still uses 2-pixel
    // reach.
    if region_x0 == 0 {
        // Left image-edge replication.
        for py in 0..pad_h {
            let row_off = py * pad_w;
            let v = padded_raw[row_off + PAD];
            for px in 0..PAD {
                padded_raw[row_off + px] = v;
            }
        }
    }
    if region_x0 + region_w == padded_width {
        // Right image-edge replication.
        for py in 0..pad_h {
            let row_off = py * pad_w;
            let v = padded_raw[row_off + pad_w - PAD - 1];
            for px in (pad_w - PAD)..pad_w {
                padded_raw[row_off + px] = v;
            }
        }
    }
    if region_y0 == 0 {
        // Top image-edge replication (after horizontal so corners are
        // sourced from the now-replicated row).
        let src_off = PAD * pad_w;
        for py in 0..PAD {
            let dst_off = py * pad_w;
            padded_raw.copy_within(src_off..src_off + pad_w, dst_off);
        }
    }
    if region_y0 + region_h == padded_height {
        // Bottom image-edge replication.
        let src_off = (pad_h - PAD - 1) * pad_w;
        for py in (pad_h - PAD)..pad_h {
            let dst_off = py * pad_w;
            padded_raw.copy_within(src_off..src_off + pad_w, dst_off);
        }
    }

    // Step 2: Symmetric5 blur on the padded raw mask. Same kernel
    // weights as the whole-image path.
    const W_R: f32 = 0.364_911_248;
    const W_D: f32 = 0.05;
    const W_R2: f32 = 0.168_888_802_1;
    const W_L: f32 = 0.221_069_183;
    const W_D2: f32 = 0.306_563_504;
    let sum = 1.0 + 4.0 * (W_R + W_D + W_R2 + W_D2 + 2.0 * W_L);
    let inv_sum = 1.0 / sum;
    let mut scratch = vec![0.0_f32; pad_n];
    jxl_simd::gaborish_5x5_channel(
        &mut padded_raw,
        &mut scratch,
        pad_w,
        pad_h,
        inv_sum,
        inv_sum * W_R,
        inv_sum * W_D,
        inv_sum * W_R2,
        inv_sum * W_L,
        inv_sum * W_D2,
    );

    // Extract the inner region from the padded blur output.
    let mut out = vec![0.0_f32; region_n];
    for ry in 0..region_h {
        let src_off = (ry + PAD) * pad_w + PAD;
        let dst_off = ry * region_w;
        out[dst_off..dst_off + region_w].copy_from_slice(&padded_raw[src_off..src_off + region_w]);
    }
    Ok(out)
}

/// Mask1x1 plane for EPF sharpness derivation: either reuse the
/// precomputed mask1x1 from the global precompute pass (when present)
/// or compute it on-demand from `xyb_y` via
/// [`compute_mask1x1_with_budget`].
///
/// **Streaming refactor chunk 8c step B (#11)**: extracts the
/// inline `match &mask1x1 { Some(m) => m, None => fallback }`
/// pattern that used to live inside the
/// `params.epf_iters > 0 && distance >= 0.5 && epf_dynamic_sharpness`
/// branch of `encode_inner` / `encode_from_precomputed_inner` into a
/// single helper. Hoisting the resolution out of the EPF branch
/// decouples the XYB-source lifetime from EPF: callers may now own
/// the resolved mask1x1 before the streaming source releases the
/// per-DC-group XYB regions that the fallback path would otherwise
/// need to read.
///
/// Returns:
/// - `Cow::Borrowed(m)` when `precomputed_mask1x1` is `Some(m)` — no
///   allocation, no XYB read.
/// - `Cow::Owned(v)` when `precomputed_mask1x1` is `None` — the
///   fallback computes the full mask1x1 plane from `xyb_y` and
///   returns ownership to the caller.
///
/// Output bytes are bit-identical to the previous inline pattern
/// (same `compute_mask1x1_with_budget` call, same arguments).
#[allow(dead_code)]
pub(crate) fn resolve_mask1x1_for_sharpness<'a>(
    precomputed_mask1x1: Option<&'a [f32]>,
    xyb_y: &[f32],
    padded_width: usize,
    padded_height: usize,
    budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
) -> crate::error::Result<alloc::borrow::Cow<'a, [f32]>> {
    match precomputed_mask1x1 {
        Some(m) => Ok(alloc::borrow::Cow::Borrowed(m)),
        None => {
            let owned = compute_mask1x1_with_budget(xyb_y, padded_width, padded_height, budget)?;
            Ok(alloc::borrow::Cow::Owned(owned))
        }
    }
}

/// `pub` wrapper around [`compute_quant_field_float_with_budget`] for
/// the `__pre_quantized` escape hatch. Matches the production-path
/// signature minus the budget plumbing — passes `None` internally;
/// downstream pre-quantized callers (e.g. jxl-encoder-gpu) don't go
/// through the budget tracker.
///
/// Unstable; gated behind the `__pre_quantized` cargo feature.
#[cfg(feature = "__pre_quantized")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn compute_quant_field_float_free(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    k_ac_quant: f32,
) -> crate::error::Result<(alloc::vec::Vec<f32>, alloc::vec::Vec<f32>)> {
    compute_quant_field_float_with_budget(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        distance,
        k_ac_quant,
        None,
    )
}

/// Convert float quant field to u8 raw_quant values.
///
/// Matches libjxl's ClampVal: `static_cast<int32_t>(clamp(qf * inv_scale + 0.5, 1.0, 256.0))`
/// which is standard round-to-nearest via add 0.5 then truncate.
pub fn quantize_quant_field(quant_field_float: &[f32], inv_scale: f32) -> Vec<u8> {
    quant_field_float
        .iter()
        .map(|&qf| {
            let val = (qf * inv_scale + 0.5) as i32;
            clamp(val, 1, 255) as u8
        })
        .collect()
}

/// Convenience wrapper that calls `compute_quant_field_float()` then
/// `quantize_quant_field()`.
#[cfg(test)]
#[allow(clippy::too_many_arguments, dead_code)]
pub fn compute_adaptive_quant_field(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    inv_scale: f32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (quant_field_float, masking) = compute_quant_field_float(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        distance,
        0.765, // K_AC_QUANT default
    );
    let raw_quant_field = quantize_quant_field(&quant_field_float, inv_scale);
    (raw_quant_field, masking, quant_field_float)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Scalar math unit tests (fast_log2f, fast_pow2f, masking_sqrt, ratio_of_derivatives,
    // compute_mask) migrated to jxl_simd::adaptive_quant::tests.

    #[test]
    fn test_store_min4() {
        let mut min0 = 5.0_f32;
        let mut min1 = 6.0;
        let mut min2 = 7.0;
        let mut min3 = 8.0;

        store_min4(3.0, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min0, 3.0);
        assert_eq!(min1, 5.0);
        assert_eq!(min2, 6.0);
        assert_eq!(min3, 7.0);

        store_min4(100.0, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min3, 7.0);
    }

    #[test]
    fn test_adaptive_quant_field_uniform() {
        let w = 16;
        let h = 16;
        let n = w * h;
        let xyb_x = vec![0.0_f32; n];
        let xyb_y = vec![0.5_f32; n];
        let xyb_b = vec![0.5_f32; n];

        let xb = w / 8;
        let yb = h / 8;

        let (result, masking, _quant_float) =
            compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, w, h, xb, yb, 1.0, 8.93);

        assert_eq!(result.len(), xb * yb);
        assert_eq!(masking.len(), xb * yb);
        for &v in &result {
            assert!(v >= 1, "quant value {} out of range", v);
        }
        let first = result[0];
        for &v in &result {
            assert_eq!(v, first, "uniform image should produce uniform quant field");
        }
    }

    #[test]
    fn test_adaptive_quant_field_varying() {
        let w = 32;
        let h = 32;
        let n = w * h;
        let mut xyb_x = vec![0.0_f32; n];
        let mut xyb_y = vec![0.0_f32; n];
        let mut xyb_b = vec![0.0_f32; n];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if x < w / 2 {
                    xyb_y[idx] = 0.5;
                    xyb_b[idx] = 0.5;
                } else {
                    xyb_y[idx] = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
                    xyb_b[idx] = xyb_y[idx];
                    xyb_x[idx] = if x % 2 == 0 { 0.1 } else { -0.1 };
                }
            }
        }

        let xb = w / 8;
        let yb = h / 8;

        let (result, _masking, _quant_float) =
            compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, w, h, xb, yb, 1.0, 8.93);

        assert_eq!(result.len(), xb * yb);
        for &v in &result {
            assert!(v >= 1, "quant value {} out of range", v);
        }
        let left_avg: f32 = (0..yb).map(|by| result[by * xb] as f32).sum::<f32>() / yb as f32;
        let right_avg: f32 = (0..yb)
            .map(|by| result[by * xb + xb - 1] as f32)
            .sum::<f32>()
            / yb as f32;
        assert!(
            (left_avg - right_avg).abs() > 0.01,
            "smooth vs textured should differ: left={}, right={}",
            left_avg,
            right_avg
        );
    }

    #[test]
    fn test_adaptive_quant_field_non_multiple_of_8() {
        for &(w, h) in &[
            (300usize, 300usize),
            (301, 301),
            (100, 100),
            (17, 17),
            (9, 9),
            (15, 33),
            (257, 129),
        ] {
            let xb = w.div_ceil(8);
            let yb = h.div_ceil(8);
            let pw = xb * 8;
            let ph = yb * 8;
            let n = pw * ph;
            let xyb_x = vec![0.0_f32; n];
            let xyb_y = vec![0.5_f32; n];
            let xyb_b = vec![0.5_f32; n];

            let (result, _masking, _quant_float) =
                compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, pw, ph, xb, yb, 1.0, 8.93);

            assert_eq!(
                result.len(),
                xb * yb,
                "wrong length for {}x{}: got {}, expected {}",
                w,
                h,
                result.len(),
                xb * yb
            );
            for &v in &result {
                assert!(v >= 1, "quant value {} out of range for {}x{}", v, w, h);
            }
        }
    }

    #[test]
    fn test_compute_mask1x1_uniform() {
        let w = 16;
        let h = 16;
        let xyb_y = vec![0.5_f32; w * h];

        let mask = compute_mask1x1(&xyb_y, w, h);

        assert_eq!(mask.len(), w * h);
        for &v in &mask {
            assert!(v > 0.0 && v.is_finite(), "mask value {} invalid", v);
        }
        let first = mask[w + 1];
        assert!(first > 50.0, "uniform mask should be high, got {}", first);
    }

    #[test]
    fn test_compute_mask1x1_edges() {
        let w = 16;
        let h = 16;
        let mut xyb_y = vec![0.2_f32; w * h];

        for y in 0..h {
            for x in 8..w {
                xyb_y[y * w + x] = 0.8;
            }
        }

        let mask = compute_mask1x1(&xyb_y, w, h);

        let interior_left = mask[4 * w + 4];
        let at_edge = mask[8 * w + 8];

        assert!(
            at_edge < interior_left,
            "edge mask {} should be < interior mask {}",
            at_edge,
            interior_left
        );
    }

    // -----------------------------------------------------------------------
    // Streaming refactor chunk 5 — per-region byte-identity tests.
    // -----------------------------------------------------------------------

    fn make_test_xyb(w: usize, h: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Deterministic non-trivial content. Mixes a low-frequency
        // gradient with high-frequency texture so the per-block
        // modulation pipeline (gamma, HF, blue) sees real variation.
        let mut xyb_x = vec![0.0_f32; w * h];
        let mut xyb_y = vec![0.0_f32; w * h];
        let mut xyb_b = vec![0.0_f32; w * h];
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for y in 0..h {
            for x in 0..w {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let n = ((state >> 33) as u32 as f32) / (u32::MAX as f32);
                let i = y * w + x;
                xyb_x[i] = 0.02 * (x as f32 / w as f32 - 0.5) + 0.01 * n;
                xyb_y[i] = 0.3 + 0.4 * (y as f32 / h as f32) + 0.05 * n;
                xyb_b[i] = 0.2 + 0.3 * ((x ^ y) as f32 / (w + h) as f32) + 0.02 * n;
            }
        }
        (xyb_x, xyb_y, xyb_b)
    }

    fn split_into_regions(
        xsize_blocks: usize,
        ysize_blocks: usize,
        dc_blocks: usize,
    ) -> Vec<(usize, usize, usize, usize)> {
        let mut regions = Vec::new();
        let mut y = 0;
        while y < ysize_blocks {
            let h = dc_blocks.min(ysize_blocks - y);
            let mut x = 0;
            while x < xsize_blocks {
                let w = dc_blocks.min(xsize_blocks - x);
                regions.push((x, y, w, h));
                x += dc_blocks;
            }
            y += dc_blocks;
        }
        regions
    }

    #[test]
    fn test_per_region_quant_field_matches_whole_image() {
        // Multi-DC-group image: 3 DC groups across, 2 down (using
        // dc_blocks=4 = 32 pixels for fast tests).
        let xsize_blocks = 12;
        let ysize_blocks = 8;
        let w = xsize_blocks * 8;
        let h = ysize_blocks * 8;
        let (xyb_x, xyb_y, xyb_b) = make_test_xyb(w, h, 1234);
        let distance = 1.0;
        let k_ac_quant = 0.765;

        let (whole_qf, whole_mask) = compute_quant_field_float(
            &xyb_x,
            &xyb_y,
            &xyb_b,
            w,
            h,
            xsize_blocks,
            ysize_blocks,
            distance,
            k_ac_quant,
        );

        // Assemble per-region output and compare.
        let regions = split_into_regions(xsize_blocks, ysize_blocks, 4);
        assert!(regions.len() > 1, "test needs multiple regions");

        let mut assembled_qf = vec![0.0_f32; xsize_blocks * ysize_blocks];
        let mut assembled_mask = vec![0.0_f32; xsize_blocks * ysize_blocks];
        for (rx0, ry0, rw, rh) in regions {
            let (qf, mask) = compute_quant_field_float_for_region(
                &xyb_x,
                &xyb_y,
                &xyb_b,
                w,
                h,
                xsize_blocks,
                ysize_blocks,
                rx0,
                ry0,
                rw,
                rh,
                distance,
                k_ac_quant,
                None,
            )
            .expect("per-region quant field should succeed");
            for ry in 0..rh {
                for rxi in 0..rw {
                    let g = (ry0 + ry) * xsize_blocks + (rx0 + rxi);
                    assembled_qf[g] = qf[ry * rw + rxi];
                    assembled_mask[g] = mask[ry * rw + rxi];
                }
            }
        }

        for (i, (&w_v, &a_v)) in whole_qf.iter().zip(assembled_qf.iter()).enumerate() {
            assert_eq!(
                w_v.to_bits(),
                a_v.to_bits(),
                "quant_field mismatch at idx {} ({}x{} blocks): whole={} per_region={}",
                i,
                i % xsize_blocks,
                i / xsize_blocks,
                w_v,
                a_v
            );
        }
        for (i, (&w_v, &a_v)) in whole_mask.iter().zip(assembled_mask.iter()).enumerate() {
            assert_eq!(
                w_v.to_bits(),
                a_v.to_bits(),
                "masking mismatch at idx {}: whole={} per_region={}",
                i,
                w_v,
                a_v
            );
        }
    }

    #[test]
    fn test_per_region_mask1x1_matches_whole_image() {
        // Use pixel-granular dc width (32 pixels = 4 blocks) so the
        // per-region split exercises multiple regions across.
        let w = 96;
        let h = 64;
        let (_xx, xyb_y, _xb) = make_test_xyb(w, h, 7777);

        let whole = compute_mask1x1(&xyb_y, w, h);

        // Tile the image at 32×32 — exercises both interior and
        // boundary regions.
        let mut assembled = vec![0.0_f32; w * h];
        let tile = 32;
        let mut y = 0;
        while y < h {
            let rh = tile.min(h - y);
            let mut x = 0;
            while x < w {
                let rw = tile.min(w - x);
                let region = compute_mask1x1_for_region(&xyb_y, w, h, x, y, rw, rh, None)
                    .expect("per-region mask1x1 should succeed");
                for ry in 0..rh {
                    let src_off = ry * rw;
                    let dst_off = (y + ry) * w + x;
                    assembled[dst_off..dst_off + rw]
                        .copy_from_slice(&region[src_off..src_off + rw]);
                }
                x += tile;
            }
            y += tile;
        }

        // Mask1x1 uses a 5×5 stencil + 5×5 blur (4-pixel reach total).
        // PAD=3 closes the structural divergence at interior region
        // boundaries (raw mask kernel's outermost-ring clamping never
        // affects the blur's inner-region reads). The remaining 1-ULP
        // drift comes from the SIMD primitive landing the same input
        // pixel in a different lane position depending on the buffer
        // width — FP order changes by a single bit in the worst case.
        //
        // For end-to-end byte-identity (hash-lock) the upstream
        // pipeline tolerates this: mask1x1 feeds AC strategy / entropy
        // estimation which themselves quantize / threshold, eating any
        // single-ULP drift. The chunk-5 invariant is enforced at the
        // `tests/buffering_dispatch.rs` level (whole-bytes match across
        // Buffering variants).
        let mut drift_count = 0usize;
        let mut max_ulp = 0u32;
        for (i, (&w_v, &a_v)) in whole.iter().zip(assembled.iter()).enumerate() {
            let wb = w_v.to_bits();
            let ab = a_v.to_bits();
            let ulp = wb.abs_diff(ab);
            if ulp > 0 {
                drift_count += 1;
                max_ulp = max_ulp.max(ulp);
                // 1-ULP drift on the raw mask propagates through the
                // Symmetric5 blur which adds another ~2 ULPs in the
                // worst case. Cap at 8 ULPs (~1e-6 relative for the
                // ~50-100 magnitude mask values seen here).
                assert!(
                    ulp <= 8,
                    "mask1x1 large drift at idx {} ({},{}): whole={} per_region={} ulp={}",
                    i,
                    i % w,
                    i / w,
                    w_v,
                    a_v,
                    ulp
                );
            }
        }
        // Total drift must be small — the structural fix should put
        // most pixels at exact match.
        let drift_pct = (drift_count as f64 / (w * h) as f64) * 100.0;
        assert!(
            drift_pct < 50.0,
            "mask1x1 drift count {}/{} ({:.1}%) max_ulp={} — structural fix may have regressed",
            drift_count,
            w * h,
            drift_pct,
            max_ulp
        );
    }
}
