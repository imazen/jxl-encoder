// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Modular encoder for lossless encoding.
//!
//! This produces valid modular streams using:
//! - Single leaf tree with Zero predictor
//! - Prefix (Huffman) codes with full code length table support
//! - No transforms
//!
//! The encoder properly encodes all pixel values, achieving lossless compression.

use crate::bit_writer::BitWriter;
use crate::entropy_coding::huffman_tree::{
    build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
};
use crate::error::Result;
use crate::modular::channel::ModularImage;
use std::collections::HashMap;

/// Pack a signed integer into an unsigned one (zigzag encoding).
/// 0 -> 0, 1 -> 2, -1 -> 1, 2 -> 4, -2 -> 3, etc.
#[inline]
pub fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    }
}

/// Writes the entropy coding header for a histogram.
///
/// Format:
/// - lz77.enabled = 0 (1 bit)
/// - context_map: if num_contexts > 1, is_simple=1 (1 bit), bits_per_entry=0 (2 bits)
/// - use_prefix_code = 1 (1 bit)
/// - HybridUint config (split_exponent = 15)
/// - alphabet_size via varint16
/// - Huffman table (via build_and_store_huffman_tree)
fn write_histogram_with_full_huffman(
    writer: &mut BitWriter,
    num_contexts: usize,
    histogram: &[u32],
) -> Result<()> {
    let alphabet_size = histogram.len();
    let max_symbol = alphabet_size.saturating_sub(1) as u32;

    crate::trace::debug_eprintln!(
        "DEBUG write_histogram: num_contexts={}, alphabet_size={}, max_symbol={}",
        num_contexts, alphabet_size, max_symbol
    );

    // lz77.enabled = 0
    crate::trace::debug_eprintln!("  TRACE [bit {}]: lz77.enabled = 0", writer.bits_written());
    writer.write(1, 0)?;

    // Context map (only if num_contexts > 1)
    if num_contexts > 1 {
        crate::trace::debug_eprintln!(
            "  TRACE [bit {}]: context_map is_simple=1, bits_per_entry=0",
            writer.bits_written()
        );
        writer.write(1, 1)?; // is_simple = 1
        writer.write(2, 0)?; // bits_per_entry = 0
    }

    // use_prefix_code = 1
    crate::trace::debug_eprintln!(
        "  TRACE [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?;

    // HybridUint config: split_exponent = 15
    // When split_exponent == log_alpha_size (both 15 for prefix codes),
    // msb_in_token and lsb_in_token are implicit and not written
    let split_exp = 15;
    crate::trace::debug_eprintln!(
        "  TRACE [bit {}]: split_exponent = {} (4 bits)",
        writer.bits_written(),
        split_exp
    );
    writer.write(4, split_exp as u64)?;

    // alphabet_size via varint16 (write max_symbol since decoder adds 1)
    crate::trace::debug_eprintln!(
        "  TRACE [bit {}]: alphabet_size = {} via varint16",
        writer.bits_written(),
        max_symbol
    );
    write_varint16(writer, max_symbol as u16)?;

    // Huffman table using full encoder
    crate::trace::debug_eprintln!(
        "  TRACE [bit {}]: Writing Huffman table (alphabet_size={})",
        writer.bits_written(),
        alphabet_size
    );

    if alphabet_size <= 1 {
        // Special case: alphabet_size <= 1, no Huffman table needed
        crate::trace::debug_eprintln!("  TRACE: alphabet_size<=1, no Huffman table needed");
    } else {
        // Use the full Huffman encoder
        build_and_store_huffman_tree(histogram, writer)?;
    }

    crate::trace::debug_eprintln!(
        "  TRACE [bit {}]: After Huffman table",
        writer.bits_written()
    );

    Ok(())
}

/// Writes a single-symbol histogram (for tree encoding).
fn write_single_symbol_histogram(
    writer: &mut BitWriter,
    num_contexts: usize,
    symbol: u16,
) -> Result<()> {
    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map (only if num_contexts > 1)
    if num_contexts > 1 {
        writer.write(1, 1)?;
        writer.write(2, 0)?;
    }

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // split_exponent = 15
    writer.write(4, 15)?;

    // alphabet_size via varint16
    write_varint16(writer, symbol)?;

    // Huffman table
    let al_size = (symbol + 1) as usize;
    if al_size > 1 {
        writer.write(2, 1)?; // simple_code_or_skip = 1
        writer.write(2, 0)?; // num_symbols - 1 = 0
        let max_bits = ceil_log2(al_size);
        writer.write(max_bits, symbol as u64)?;
    }

    Ok(())
}

/// Encodes a varint16 value to the bitstream.
fn write_varint16(writer: &mut BitWriter, value: u16) -> Result<()> {
    if value == 0 {
        writer.write(1, 0)?;
    } else if value == 1 {
        writer.write(1, 1)?;
        writer.write(4, 0)?; // nbits = 0 means value 1
    } else {
        writer.write(1, 1)?;
        let nbits = 15 - value.leading_zeros() as usize;
        let mantissa = value - (1 << nbits);
        writer.write(4, nbits as u64)?;
        writer.write(nbits, mantissa as u64)?;
    }
    Ok(())
}

fn ceil_log2(x: usize) -> usize {
    if x <= 1 {
        0
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}

/// Collect all residuals, build histogram, and compute statistics.
fn collect_residuals_and_histogram(image: &ModularImage) -> (Vec<u32>, Vec<u32>, u32) {
    let mut residuals = Vec::new();
    let mut max_residual: u32 = 0;

    // With Zero predictor (guess=0, multiplier=1), residual = pixel value
    for channel in &image.channels {
        for y in 0..channel.height() {
            for x in 0..channel.width() {
                let pixel = channel.get(x, y);
                let packed = pack_signed(pixel);
                residuals.push(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    // Build histogram (size = max_residual + 1)
    let histogram_size = (max_residual + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for &r in &residuals {
        histogram[r as usize] += 1;
    }

    (residuals, histogram, max_residual)
}

/// Build symbol-to-code mapping from depths.
fn build_codes_from_depths(depths: &[u8], codes: &[u16]) -> HashMap<u32, (u16, u8)> {
    let mut map = HashMap::new();
    for (symbol, (&depth, &code)) in depths.iter().zip(codes.iter()).enumerate() {
        if depth > 0 {
            map.insert(symbol as u32, (code, depth));
        } else if depths.iter().filter(|&&d| d > 0).count() == 0 {
            // Single symbol case - depth 0 means implicit (0 bits)
            map.insert(symbol as u32, (0, 0));
        }
    }
    map
}

/// Writes a modular stream for an image.
///
/// Supports arbitrary images with any number of unique pixel values.
pub fn write_minimal_modular_stream(image: &ModularImage, writer: &mut BitWriter) -> Result<()> {
    // Collect all residuals and build histogram
    let (residuals, histogram, max_residual) = collect_residuals_and_histogram(image);

    let num_symbols = histogram.iter().filter(|&&c| c > 0).count();
    crate::trace::debug_eprintln!(
        "DEBUG: {} residuals, {} unique symbols, max={}, histogram_size={}",
        residuals.len(),
        num_symbols,
        max_residual,
        histogram.len()
    );

    let start_bits = writer.bits_written();
    crate::trace::debug_eprintln!("TRACE [bit {}]: Starting modular stream", start_bits);

    // === Global section (LfGlobal) ===

    // dc_quant.all_default = true
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: Writing dc_quant.all_default = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?;

    // has_tree = true
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: Writing has_tree = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?;

    // === Tree histograms ===
    // NUM_TREE_CONTEXTS = 6, all tokens are 0 for single-leaf tree
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: Writing tree histograms (6 contexts, symbol 0)",
        writer.bits_written()
    );
    write_single_symbol_histogram(writer, 6, 0)?;
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: After tree histograms",
        writer.bits_written()
    );

    // === Tree tokens ===
    // With single-symbol histogram, implicit zeros (0 bits each)

    // === Leaf histograms ===
    // 1 context for single-leaf tree
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: Writing data histograms (1 context, {} symbols)",
        writer.bits_written(),
        num_symbols
    );
    write_histogram_with_full_huffman(writer, 1, &histogram)?;
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: After data histograms",
        writer.bits_written()
    );

    // === GroupHeader ===
    crate::trace::debug_eprintln!("TRACE [bit {}]: Writing GroupHeader", writer.bits_written());
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0
    crate::trace::debug_eprintln!(
        "TRACE [bit {}]: After GroupHeader (4 bits)",
        writer.bits_written()
    );

    // === Pixel tokens ===
    // Build Huffman codes from histogram
    let depths = create_huffman_tree(&histogram, 15);
    let codes = convert_bit_depths_to_symbols(&depths);
    let code_map = build_codes_from_depths(&depths, &codes);

    crate::trace::debug_eprintln!("DEBUG: Built {} Huffman codes", code_map.len());
    for (&symbol, &(code, depth)) in &code_map {
        crate::trace::debug_eprintln!(
            "  Huffman: symbol {} -> code={:b}, depth={}",
            symbol, code, depth
        );
    }
    crate::trace::debug_eprintln!(
        "DEBUG: First 10 residuals: {:?}",
        &residuals[..residuals.len().min(10)]
    );

    // Encode each residual
    for &r in &residuals {
        if let Some(&(code, depth)) = code_map.get(&r) {
            if depth > 0 {
                writer.write(depth as usize, code as u64)?;
            }
            // depth == 0 means single symbol, no bits needed
        } else {
            // This shouldn't happen if histogram was built correctly
            crate::trace::debug_eprintln!("WARNING: No code for residual {}", r);
        }
    }

    // Byte-align
    writer.zero_pad_to_byte();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_stream() {
        let data: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let image = ModularImage::from_rgb8(&data, 2, 2).unwrap();

        let mut writer = BitWriter::new();
        write_minimal_modular_stream(&image, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        println!("Minimal stream: {} bytes", bytes.len());
        for b in &bytes {
            print!("{:02x} ", b);
        }
        println!();
    }

    #[test]
    fn test_many_symbols() {
        // 4x4 gradient with 16 unique values - previously failed
        let data: Vec<u8> = (0u8..16).collect();

        let image = ModularImage::from_gray8(&data, 4, 4).unwrap();

        let mut writer = BitWriter::new();
        write_minimal_modular_stream(&image, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("Encoded 16-symbol image: {} bytes", bytes.len());
    }

    #[test]
    fn test_256_symbols() {
        // 16x16 gradient with 256 unique values
        let data: Vec<u8> = (0u8..=255).collect();

        let image = ModularImage::from_gray8(&data, 16, 16).unwrap();

        let mut writer = BitWriter::new();
        write_minimal_modular_stream(&image, &mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        crate::trace::debug_eprintln!("Encoded 256-symbol image: {} bytes", bytes.len());
    }
}
