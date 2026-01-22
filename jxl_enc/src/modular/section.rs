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
    write_gradient_tree_tokens, write_integer_config, write_tree_histogram_for_gradient,
    write_varlen_u16,
};
use super::predictor::pack_signed;
use crate::bit_writer::BitWriter;
use crate::entropy_coding::huffman_tree::build_and_store_huffman_tree;
use crate::error::Result;
#[allow(unused_imports)]
use std::collections::HashMap;

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

/// Builds a histogram from residuals.
pub fn build_histogram_from_residuals(residuals: &[u32], max_residual: u32) -> Vec<u32> {
    let histogram_size = (max_residual + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for &r in residuals {
        if (r as usize) < histogram_size {
            histogram[r as usize] += 1;
        }
    }
    histogram
}

/// Result of writing the global modular section.
/// Contains the Huffman codes needed to encode pixel data in group sections.
pub struct GlobalModularState {
    /// Huffman bit depths for each symbol.
    pub depths: Vec<u8>,
    /// Huffman codes for each symbol.
    pub codes: Vec<u16>,
    /// Maximum residual value (determines alphabet size).
    pub max_residual: u32,
}

/// Writes the global modular section (tree + histogram) for multi-group encoding.
///
/// This writes:
/// - dc_quant.all_default = 1
/// - has_tree = 1
/// - Tree histogram and tokens (Gradient predictor)
/// - Data histogram (no LZ77 for simplicity in multi-group)
///
/// Returns the Huffman state needed to encode pixel data in group sections.
pub fn write_global_modular_section(
    histogram: &[u32],
    max_residual: u32,
    writer: &mut BitWriter,
) -> Result<GlobalModularState> {
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: Starting global section",
        writer.bits_written()
    );

    // dc_quant.all_default = true
    writer.write(1, 1)?;
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: dc_quant.all_default = 1",
        writer.bits_written()
    );

    // has_tree = true
    writer.write(1, 1)?;
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: has_tree = 1",
        writer.bits_written()
    );

    // Tree histogram (supports symbols 0-5 for Gradient predictor)
    let (tree_depths, tree_codes) = write_tree_histogram_for_gradient(writer)?;
    // Tree tokens for single leaf with Gradient predictor
    write_gradient_tree_tokens(writer, &tree_depths, &tree_codes)?;

    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: Starting data histogram",
        writer.bits_written()
    );

    // Data histogram (no LZ77 for multi-group simplicity)
    writer.write(1, 0)?; // lz77.enabled = 0
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: lz77.enabled = 0",
        writer.bits_written()
    );
    writer.write(1, 1)?; // use_prefix_code = 1
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );

    // IntegerConfig: raw symbols with split_exponent = 15
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: IntegerConfig (split_exp=15, raw symbols)",
        writer.bits_written()
    );

    // alphabet_size-1 using VarLenUint16 encoding
    write_varlen_u16(writer, max_residual as u16)?;
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: alphabet_size-1 = {}",
        writer.bits_written(),
        max_residual
    );

    // Build and store Huffman table
    let (depths, codes) = if histogram.len() > 1 {
        let table = build_and_store_huffman_tree(histogram, writer)?;
        crate::trace::debug_eprintln!(
            "GLOBAL_MODULAR [bit {}]: After Huffman table",
            writer.bits_written()
        );
        (table.depths, table.codes)
    } else {
        // Single symbol - no bits needed
        (vec![0u8; histogram.len()], vec![0u16; histogram.len()])
    };

    // Write GlobalModular's ModularHeader
    // Even for multi-group where all channels are > group_dim,
    // the decoder still parses this header before filtering channels.
    writer.write(1, 1)?; // use_global_tree = true (use the tree we just wrote)
    writer.write(1, 1)?; // wp_params.default_wp = true
    writer.write(2, 0)?; // nb_transforms = 0
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: After GlobalModular ModularHeader",
        writer.bits_written()
    );

    // Byte-align at end of global section
    writer.zero_pad_to_byte();
    crate::trace::debug_eprintln!(
        "GLOBAL_MODULAR [bit {}]: Global section done",
        writer.bits_written()
    );

    Ok(GlobalModularState {
        depths,
        codes,
        max_residual,
    })
}

/// Writes a group's data section for multi-group modular encoding.
///
/// This writes:
/// - GroupHeader (use_global_tree=1, wp_header.all_default=1, num_transforms=0)
/// - Encoded pixel residuals using the global histogram
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
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: After GroupHeader",
        writer.bits_written()
    );

    // Build code map for encoding
    let code_map: HashMap<u32, (u16, u8)> = state
        .depths
        .iter()
        .zip(state.codes.iter())
        .enumerate()
        .filter(|(_, (d, _))| **d > 0)
        .map(|(i, (d, c))| (i as u32, (*c, *d)))
        .collect();

    crate::trace::debug_eprintln!(
        "GROUP_MODULAR: code_map has {} entries, max_residual={}",
        code_map.len(),
        state.max_residual
    );
    for (&symbol, &(code, depth)) in &code_map {
        crate::trace::debug_eprintln!(
            "GROUP_MODULAR:   symbol {} -> code {:b} (depth {})",
            symbol, code, depth
        );
    }

    // Collect and encode residuals for this group
    let mut encoded_count = 0;
    for channel in &group_image.channels {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);

                // Get neighbors (same as global collection)
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

                // Encode using the global Huffman table
                if let Some(&(code, depth)) = code_map.get(&packed) {
                    if depth > 0 {
                        writer.write(depth as usize, code as u64)?;
                        encoded_count += 1;
                    }
                } else {
                    // Symbol not in histogram - this shouldn't happen if histogram was built correctly
                    crate::trace::debug_eprintln!(
                        "WARNING: residual {} not in code_map (max={})",
                        packed, state.max_residual
                    );
                }
            }
        }
    }

    // Byte-align at end of group section
    writer.zero_pad_to_byte();
    crate::trace::debug_eprintln!(
        "GROUP_MODULAR [bit {}]: Group section done ({} values encoded)",
        writer.bits_written(),
        encoded_count
    );

    Ok(())
}
