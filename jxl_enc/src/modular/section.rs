// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Modular section encoding for multi-group images.
//!
//! Handles GlobalModularState and section writing for large images that
//! are split into multiple groups.

use super::channel::ModularImage;
use super::improved::{
    write_gradient_tree_tokens, write_hybrid_data_histogram, write_tree_histogram_for_gradient,
};
use super::predictor::pack_signed;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::hybrid_uint::HybridUintConfig;
use crate::error::Result;

/// Default HybridUint config for modular data: split_exponent=4, msb_in_token=2, lsb_in_token=0.
const MODULAR_HYBRID_UINT: HybridUintConfig = HybridUintConfig {
    split_exponent: 4,
    split: 16, // 1 << 4
    msb_in_token: 2,
    lsb_in_token: 0,
};

/// Gradient prediction (ClampedGradient).
#[inline]
fn predict_gradient(left: i32, top: i32, topleft: i32) -> i32 {
    let grad = left + top - topleft;
    // Clamp to [min(left, top), max(left, top)]
    let min = left.min(top);
    let max = left.max(top);
    grad.clamp(min, max)
}

pub fn collect_all_residuals(image: &ModularImage) -> (Vec<u32>, u32) {
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    for channel in &image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                // Get neighbors (matching JXL decoder)
                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else {
                    left
                };

                // Predict using ClampedGradient (predictor 5)
                let prediction = predict_gradient(left, top, topleft);
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    (residuals, max_residual)
}

/// Builds a histogram from residuals, encoding through HybridUint {4,2,0}.
/// Returns (histogram_on_tokens, max_token).
pub fn build_histogram_from_residuals(residuals: &[u32], _max_residual: u32) -> (Vec<u32>, u32) {
    let mut max_token: u32 = 0;
    // First pass: find max token
    for &r in residuals {
        let (token, _, _) = MODULAR_HYBRID_UINT.encode(r);
        max_token = max_token.max(token);
    }
    // Second pass: build histogram on tokens
    let histogram_size = (max_token + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for &r in residuals {
        let (token, _, _) = MODULAR_HYBRID_UINT.encode(r);
        histogram[token as usize] += 1;
    }
    (histogram, max_token)
}

/// Result of writing the global modular section.
/// Contains the Huffman codes needed to encode pixel data in group sections.
pub struct GlobalModularState {
    /// Huffman bit depths for each HybridUint token.
    pub depths: Vec<u8>,
    /// Huffman codes for each HybridUint token.
    pub codes: Vec<u16>,
    /// Maximum HybridUint token value.
    pub max_token: u32,
}

/// Writes the global modular section (tree + histogram) for multi-group encoding.
///
/// This writes:
/// - dc_quant.all_default = 1
/// - has_tree = 1
/// - Tree histogram and tokens (Gradient predictor)
/// - Data histogram with HybridUint {4,2,0}
///
/// `histogram` and `max_token` are built from HybridUint-encoded tokens (not raw residuals).
/// Returns the Huffman state needed to encode pixel data in group sections.
pub fn write_global_modular_section(
    histogram: &[u32],
    max_token: u32,
    writer: &mut BitWriter,
) -> Result<GlobalModularState> {
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: Starting global section",
        writer.bits_written()
    );

    // dc_quant.all_default = true
    writer.write(1, 1)?;
    // has_tree = true
    writer.write(1, 1)?;

    // Tree histogram (supports symbols 0-5 for Gradient predictor)
    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    // Data histogram with HybridUint {4,2,0}
    let (depths, codes) = write_hybrid_data_histogram(writer, histogram, max_token)?;

    // Write GlobalModular's ModularHeader
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_params.default_wp = true
    writer.write(2, 0)?; // nb_transforms = 0

    // Byte-align at end of global section
    writer.zero_pad_to_byte();
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: Global section done",
        writer.bits_written()
    );

    Ok(GlobalModularState {
        depths,
        codes,
        max_token,
    })
}

/// Writes a group's data section for multi-group modular encoding.
///
/// This writes:
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0)
/// - Encoded pixel residuals using HybridUint {4,2,0} + global Huffman codes
///
/// The `group_image` should be the extracted region for this group.
pub fn write_group_modular_section(
    group_image: &ModularImage,
    state: &GlobalModularState,
    writer: &mut BitWriter,
) -> Result<()> {
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: Starting group section ({}x{})",
        writer.bits_written(),
        group_image.width(),
        group_image.height()
    );

    // GroupHeader
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0

    // Collect and encode residuals for this group through HybridUint
    for channel in &group_image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                let left = if x > 0 { channel.get(x - 1, y) } else { 0 };
                let top = if y > 0 { channel.get(x, y - 1) } else { left };
                let topleft = if x > 0 && y > 0 {
                    channel.get(x - 1, y - 1)
                } else {
                    left
                };

                let prediction = predict_gradient(left, top, topleft);
                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                // Encode through HybridUint {4,2,0} + Huffman
                let (token, extra_bits, num_extra) = MODULAR_HYBRID_UINT.encode(packed);
                let depth = state.depths.get(token as usize).copied().unwrap_or(0);
                let code = state.codes.get(token as usize).copied().unwrap_or(0);
                if depth > 0 {
                    writer.write(depth as usize, code as u64)?;
                }
                if num_extra > 0 {
                    writer.write(num_extra as usize, extra_bits as u64)?;
                }
            }
        }
    }

    // Byte-align at end of group section
    writer.zero_pad_to_byte();
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: Group section done",
        writer.bits_written()
    );

    Ok(())
}
