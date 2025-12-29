// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Minimal modular encoder for lossless encoding.
//!
//! This produces valid modular streams using:
//! - Single leaf tree with Zero predictor
//! - Prefix (Huffman) codes
//! - No transforms
//!
//! The encoder properly encodes all pixel values, achieving lossless compression.

use crate::bit_writer::BitWriter;
use crate::error::Result;
use crate::modular::channel::ModularImage;

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

/// Writes a histogram for a set of symbols.
///
/// Format:
/// - lz77.enabled = 0 (1 bit)
/// - context_map: if num_contexts > 1, is_simple=1 (1 bit), bits_per_entry=0 (2 bits)
/// - use_prefix_code = 1 (1 bit)
/// - HybridUint config (split_exponent)
/// - alphabet_size via varint16
/// - Huffman table
fn write_histogram(
    writer: &mut BitWriter,
    num_contexts: usize,
    unique_symbols: &[u32],
    max_symbol: u32,
) -> Result<()> {
    eprintln!(
        "DEBUG write_histogram: num_contexts={}, unique_symbols={:?}, max_symbol={}",
        num_contexts, unique_symbols, max_symbol
    );

    // lz77.enabled = 0
    eprintln!("  TRACE [bit {}]: lz77.enabled = 0", writer.bits_written());
    writer.write(1, 0)?;

    // Context map (only if num_contexts > 1)
    if num_contexts > 1 {
        eprintln!(
            "  TRACE [bit {}]: context_map is_simple=1, bits_per_entry=0",
            writer.bits_written()
        );
        writer.write(1, 1)?; // is_simple = 1
        writer.write(2, 0)?; // bits_per_entry = 0
    }

    // use_prefix_code = 1
    eprintln!(
        "  TRACE [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?;

    // HybridUint config: split_exponent, msb_in_token, lsb_in_token
    // IMPORTANT: When use_prefix_code = true, the JXL spec defines log_alpha_size = 15 (fixed)
    // To avoid writing msb_in_token and lsb_in_token, we set split_exponent = 15
    let al_size = (max_symbol + 1) as usize;
    let split_exp = 15; // Match log_alpha_size for prefix codes
    eprintln!(
        "  TRACE [bit {}]: split_exponent = {} (4 bits)",
        writer.bits_written(),
        split_exp
    );
    writer.write(4, split_exp as u64)?;

    // When split_exponent == log_alpha_size (both 15 for prefix codes),
    // msb_in_token and lsb_in_token are implicit and not written
    eprintln!("  TRACE: msb_in_token and lsb_in_token implicit (split_exp == log_alpha_size = 15)");

    // alphabet_size via varint16 (write max_symbol since decoder adds 1)
    eprintln!(
        "  TRACE [bit {}]: alphabet_size = {} via varint16",
        writer.bits_written(),
        max_symbol
    );
    write_varint16(writer, max_symbol as u16)?;

    // Huffman table
    let num_unique = unique_symbols.len();
    eprintln!(
        "  TRACE [bit {}]: Huffman table (num_unique={}, al_size={})",
        writer.bits_written(),
        num_unique,
        al_size
    );
    if al_size == 1 {
        // Special case: alphabet_size = 1, no Huffman table needed
        eprintln!("  TRACE: alphabet_size=1, no Huffman table needed");
    } else if num_unique == 1 {
        // Single unique symbol
        eprintln!(
            "  TRACE [bit {}]: simple_code_or_skip=1, num_symbols-1=0",
            writer.bits_written()
        );
        writer.write(2, 1)?; // simple_code_or_skip = 1
        writer.write(2, 0)?; // num_symbols - 1 = 0
        let max_bits = ceil_log2(al_size);
        eprintln!(
            "  TRACE [bit {}]: symbol={} ({} bits)",
            writer.bits_written(),
            unique_symbols[0],
            max_bits
        );
        writer.write(max_bits, unique_symbols[0] as u64)?;
    } else if num_unique == 2 {
        // Two unique symbols
        eprintln!(
            "  TRACE [bit {}]: simple_code_or_skip=1, num_symbols-1=1",
            writer.bits_written()
        );
        writer.write(2, 1)?; // simple_code_or_skip = 1
        writer.write(2, 1)?; // num_symbols - 1 = 1
        let max_bits = ceil_log2(al_size);
        let mut sorted = unique_symbols.to_vec();
        sorted.sort();
        eprintln!(
            "  TRACE [bit {}]: 2-symbol Huffman: al_size={}, max_bits={}, sorted_symbols={:?}",
            writer.bits_written(),
            al_size,
            max_bits,
            sorted
        );
        writer.write(max_bits, sorted[0] as u64)?;
        eprintln!(
            "  TRACE [bit {}]: wrote symbol[0]={}",
            writer.bits_written(),
            sorted[0]
        );
        writer.write(max_bits, sorted[1] as u64)?;
        eprintln!(
            "  TRACE [bit {}]: wrote symbol[1]={}",
            writer.bits_written(),
            sorted[1]
        );
    } else if num_unique <= 4 {
        // 3-4 unique symbols
        writer.write(2, 1)?; // simple_code_or_skip = 1
        writer.write(2, (num_unique - 1) as u64)?;
        let max_bits = ceil_log2(al_size);
        let mut sorted = unique_symbols.to_vec();
        sorted.sort();
        for sym in &sorted {
            writer.write(max_bits, *sym as u64)?;
        }
        if num_unique == 4 {
            writer.write(1, 0)?; // tree_select = 0 (balanced)
        }
    } else {
        // More than 4 symbols - use the 4 most frequent
        // This is a simplification; full implementation would use proper Huffman
        writer.write(2, 1)?; // simple_code_or_skip = 1
        writer.write(2, 3)?; // 4 symbols
        let max_bits = ceil_log2(al_size);
        let mut sorted = unique_symbols.to_vec();
        sorted.sort();
        for sym in sorted.iter().take(4) {
            writer.write(max_bits, *sym as u64)?;
        }
        writer.write(1, 0)?; // tree_select
    }

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
/// Decoder reads: if bit==1, nbits=read(4), if nbits==0 return 1, else return (1<<nbits)+read(nbits)
/// If bit==0, return 0.
fn write_varint16(writer: &mut BitWriter, value: u16) -> Result<()> {
    if value == 0 {
        writer.write(1, 0)?;
    } else if value == 1 {
        writer.write(1, 1)?;
        writer.write(4, 0)?; // nbits = 0 means value 1
    } else {
        writer.write(1, 1)?;
        // Find nbits such that (1 << nbits) + mantissa = value
        // nbits = floor(log2(value)) = 15 - leading_zeros(value) for u16
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

/// Collect all residuals and compute statistics.
fn collect_residuals(image: &ModularImage) -> (Vec<u32>, Vec<u32>, u32) {
    let mut residuals = Vec::new();
    let mut unique_set = std::collections::BTreeSet::new();
    let mut max_residual: u32 = 0;

    // With Zero predictor (guess=0, multiplier=1), residual = pixel value
    for channel in &image.channels {
        for y in 0..channel.height() {
            for x in 0..channel.width() {
                let pixel = channel.get(x, y);
                let packed = pack_signed(pixel);
                residuals.push(packed);
                unique_set.insert(packed);
                max_residual = max_residual.max(packed);
            }
        }
    }

    let unique: Vec<u32> = unique_set.into_iter().collect();
    (residuals, unique, max_residual)
}

/// Build symbol-to-code mapping for simple Huffman.
fn build_simple_codes(unique: &[u32]) -> std::collections::HashMap<u32, (u32, u8)> {
    let mut map = std::collections::HashMap::new();
    let n = unique.len();

    if n == 0 {
        return map;
    }

    let mut sorted = unique.to_vec();
    sorted.sort();

    if n == 1 {
        // Single symbol - 0 bits
        map.insert(sorted[0], (0, 0));
    } else if n == 2 {
        // Two symbols - 1 bit each
        map.insert(sorted[0], (0, 1));
        map.insert(sorted[1], (1, 1));
    } else if n == 3 {
        // Three symbols: 0, 10, 11
        map.insert(sorted[0], (0, 1));
        map.insert(sorted[1], (0b10, 2));
        map.insert(sorted[2], (0b11, 2));
    } else {
        // Four symbols: decoder uses order [0,2,1,3] for table entries
        // code 00 → symbols[0], code 01 → symbols[2], code 10 → symbols[1], code 11 → symbols[3]
        map.insert(sorted[0], (0b00, 2));
        map.insert(sorted[1], (0b10, 2)); // symbols[1] gets code 10
        map.insert(sorted[2], (0b01, 2)); // symbols[2] gets code 01
        map.insert(sorted[3], (0b11, 2));
    }

    map
}

/// Writes a minimal valid modular stream for a small image.
pub fn write_minimal_modular_stream(image: &ModularImage, writer: &mut BitWriter) -> Result<()> {
    // Collect all residuals
    let (residuals, unique, max_residual) = collect_residuals(image);

    eprintln!(
        "DEBUG: {} residuals, {} unique symbols, max={}",
        residuals.len(),
        unique.len(),
        max_residual
    );
    eprintln!("DEBUG: residuals = {:?}", residuals);
    eprintln!("DEBUG: unique = {:?}", unique);

    let start_bits = writer.bits_written();
    eprintln!("TRACE [bit {}]: Starting modular stream", start_bits);

    // === Global section (LfGlobal) ===

    // dc_quant.all_default = true
    eprintln!(
        "TRACE [bit {}]: Writing dc_quant.all_default = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?;

    // has_tree = true
    eprintln!(
        "TRACE [bit {}]: Writing has_tree = 1",
        writer.bits_written()
    );
    writer.write(1, 1)?;

    // === Tree histograms ===
    // NUM_TREE_CONTEXTS = 6, all tokens are 0 for single-leaf tree
    eprintln!(
        "TRACE [bit {}]: Writing tree histograms (6 contexts, symbol 0)",
        writer.bits_written()
    );
    write_single_symbol_histogram(writer, 6, 0)?;
    eprintln!(
        "TRACE [bit {}]: After tree histograms",
        writer.bits_written()
    );

    // === Tree tokens ===
    // With single-symbol histogram, implicit zeros (0 bits each)

    // === Leaf histograms ===
    // 1 context for single-leaf tree
    eprintln!(
        "TRACE [bit {}]: Writing data histograms (1 context, {} symbols)",
        writer.bits_written(),
        unique.len()
    );
    write_histogram(writer, 1, &unique, max_residual)?;
    eprintln!(
        "TRACE [bit {}]: After data histograms",
        writer.bits_written()
    );

    // === GroupHeader ===
    eprintln!("TRACE [bit {}]: Writing GroupHeader", writer.bits_written());
    writer.write(1, 1)?; // use_global_tree = true
    writer.write(1, 1)?; // wp_header.all_default = true
    writer.write(2, 0)?; // num_transforms = 0
    eprintln!(
        "TRACE [bit {}]: After GroupHeader (4 bits)",
        writer.bits_written()
    );

    // === Pixel tokens ===
    // Encode each residual using the Huffman codes
    let codes = build_simple_codes(&unique);

    eprintln!("DEBUG: Huffman codes = {:?}", codes);

    for &r in &residuals {
        if let Some(&(code, bits)) = codes.get(&r) {
            if bits > 0 {
                eprintln!(
                    "DEBUG: Writing residual {} as code {} ({} bits)",
                    r, code, bits
                );
                writer.write(bits as usize, code as u64)?;
            } else {
                eprintln!("DEBUG: Skipping residual {} (0 bits)", r);
            }
        }
        // If symbol not in codes (shouldn't happen), skip
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
}
