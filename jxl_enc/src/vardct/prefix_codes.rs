// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Prefix code (Huffman) encoding utilities for VarDCT.
//!
//! Contains helper functions for:
//! - Building canonical Huffman codes
//! - Writing prefix codes to bitstream
//! - Alphabet size encoding
//! - Variable-length integer encoding

use crate::bit_writer::BitWriter;
use crate::error::Result;
#[allow(unused_imports)]
use crate::trace_write;

/// Write a variable-length uint8.
#[allow(dead_code)] // May be used in future for custom quant matrices
pub(crate) fn write_var_len_uint8(writer: &mut BitWriter, n: u8) -> Result<()> {
    if n == 0 {
        writer.write(1, 0)?;
    } else {
        writer.write(1, 1)?;
        let nbits = 8 - n.leading_zeros();
        writer.write(3, (nbits - 1) as u64)?;
        writer.write((nbits - 1) as usize, (n as u64) - (1u64 << (nbits - 1)))?;
    }
    Ok(())
}

/// Reverse the bits of a value within a given bit length.
pub(crate) fn bit_reverse(value: u32, len: u8) -> u32 {
    if len == 0 {
        return 0;
    }
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..len {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Build canonical Huffman codes for near-flat distribution.
///
/// Returns (codes, code_lengths) where:
/// - codes[i] is the bit-reversed canonical code for symbol i (for LSB-first bitstream)
/// - code_lengths[i] is the bit length for symbol i
///
/// For n symbols where d = ceil(log2(n)):
/// - First (2^d - n) symbols get depth d-1
/// - Remaining symbols get depth d
pub(crate) fn build_canonical_huffman_codes(alphabet_size: usize) -> (Vec<u32>, Vec<u8>) {
    if alphabet_size <= 1 {
        return (vec![0], vec![0]);
    }

    let d = (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize;
    let pow2_d = 1usize << d;

    // Number of symbols at each depth
    let symbols_short = pow2_d.saturating_sub(alphabet_size); // depth d-1
    let _symbols_long = alphabet_size.saturating_sub(symbols_short); // depth d

    let (depth_short, depth_long) = if d <= 1 || symbols_short == 0 {
        (d, d)
    } else {
        (d - 1, d)
    };

    // Build code lengths array
    let mut code_lengths = vec![0u8; alphabet_size];
    for cl in code_lengths.iter_mut().take(symbols_short) {
        *cl = depth_short as u8;
    }
    for cl in code_lengths.iter_mut().skip(symbols_short) {
        *cl = depth_long as u8;
    }

    // Build canonical codes
    // Sort symbols by (code_length, symbol) - already sorted since shorter symbols come first
    let mut codes = vec![0u32; alphabet_size];
    let mut code = 0u32;
    let mut prev_len = 0u8;

    for sym in 0..alphabet_size {
        let len = code_lengths[sym];
        if len == 0 {
            continue;
        }
        if len > prev_len {
            code <<= len - prev_len;
        }
        // Store bit-reversed code for LSB-first bitstream
        codes[sym] = bit_reverse(code, len);
        code += 1;
        prev_len = len;
    }

    (codes, code_lengths)
}

/// Write alphabet size for Huffman/prefix codes.
///
/// The format for prefix codes is:
/// - If count == 1: write 0 (1 bit)
/// - If count > 1: write 1 (1 bit), then encode (count - 1) as:
///   - n = ceil(log2(count - 1)) bits needed
///   - write n (4 bits)
///   - write (count - 1 - (1 << n)) using n bits
pub(crate) fn write_alphabet_size(writer: &mut BitWriter, size: usize) -> Result<()> {
    if size == 1 {
        writer.write(1, 0)?; // count = 1
    } else {
        writer.write(1, 1)?; // count > 1
        let count_minus_1 = size - 1;
        // Find n such that 1 + (1 << n) + extra = count, where extra < (1 << n)
        // count - 1 = (1 << n) + extra
        // n = floor(log2(count - 1))
        let n = (usize::BITS - 1 - count_minus_1.leading_zeros()) as usize;
        let extra = count_minus_1 - (1 << n);
        writer.write(4, n as u64)?;
        writer.write(n, extra as u64)?;
    }
    Ok(())
}

/// Write a prefix code (Huffman) for a given alphabet size.
///
/// The format is based on Brotli prefix codes:
/// - hskip (2 bits): 1 = simple code, 0/2/3 = complex code with skip count
///
/// For simple codes (alphabet_size <= 4):
/// - nsym - 1 (2 bits)
/// - symbols (alphabet_bits each)
/// - tree_selector (1 bit) for 4 symbols
///
/// For complex codes (alphabet_size > 4):
/// - Uses Brotli-style code length encoding
pub(crate) fn write_prefix_code(writer: &mut BitWriter, alphabet_size: usize) -> Result<()> {
    if alphabet_size == 0 {
        return Ok(());
    }

    // Calculate bits needed to represent symbols
    let max_bits = if alphabet_size <= 1 {
        0
    } else {
        (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize
    };

    if alphabet_size == 1 {
        // Single symbol: simple code with nsym=1
        // hskip=1, nsym-1=0, symbol=0
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 0)?; // nsym - 1 = 0
        writer.write(max_bits.max(1), 0)?; // symbol 0 (at least 1 bit)
    } else if alphabet_size == 2 {
        // Two symbols: simple code with nsym=2
        // hskip=1, nsym-1=1, sym0, sym1
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 1)?; // nsym - 1 = 1
        writer.write(max_bits, 0)?; // symbol 0
        writer.write(max_bits, 1)?; // symbol 1
    } else if alphabet_size == 3 {
        // Three symbols: simple code with nsym=3
        // Depths: 1, 2, 2 → codes: 0, 10, 11
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 2)?; // nsym - 1 = 2
        writer.write(max_bits, 0)?; // symbol 0 (depth 1)
        writer.write(max_bits, 1)?; // symbol 1 (depth 2)
        writer.write(max_bits, 2)?; // symbol 2 (depth 2)
    } else if alphabet_size == 4 {
        // Four symbols: simple code with nsym=4
        // Depths: 2, 2, 2, 2 (flat) → tree_selector=0
        writer.write(2, 1)?; // hskip = 1 (simple)
        writer.write(2, 3)?; // nsym - 1 = 3
        writer.write(max_bits, 0)?; // symbol 0
        writer.write(max_bits, 1)?; // symbol 1
        writer.write(max_bits, 2)?; // symbol 2
        writer.write(max_bits, 3)?; // symbol 3
        writer.write(1, 0)?; // tree_selector = 0 (flat: depths 2,2,2,2)
    } else {
        // Complex code: use Brotli-style encoding
        // For simplicity, encode a flat distribution where all symbols have
        // the same code length.
        write_complex_prefix_code(writer, alphabet_size)?;
    }

    Ok(())
}

/// Write a flat ANS histogram for large alphabets.
///
/// For flat ANS distribution:
/// - use_prefix_code = 0 (use ANS)
/// - IntegerConfig with appropriate log_alphabet_size
/// - is_flat = 1 (flat distribution)
/// - log_alphabet_size bits for the alphabet size
#[allow(dead_code)]
pub(crate) fn write_flat_ans_histogram(writer: &mut BitWriter, alphabet_size: usize) -> Result<()> {
    // use_prefix_code = 0 (use ANS)
    writer.write(1, 0)?;

    // Calculate log_alphabet_size (ceil(log2(alphabet_size)))
    let log_alpha = if alphabet_size <= 1 {
        0
    } else {
        (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize
    };

    // IntegerConfig for ANS
    // split_exponent_bits = add_log2_ceil(log_alphabet_size) + 3
    // When log_alphabet_size is known, split_exponent is read with the calculated bits
    // For log_alphabet_size, use add_log2_ceil which returns ceil(log2(max + 1)) for range [0, max]
    // log_alphabet_size fits in [0, 15] so we need 4 bits for it
    let split_bits = if log_alpha == 0 { 0 } else { log_alpha };
    writer.write(4, split_bits as u64)?; // split_exponent = log_alpha (raw symbols)

    // When split_exponent == log_alphabet_size (which it does for raw symbols),
    // msb_in_token and lsb_in_token are implicitly 0 (not read).

    // Write alphabet_size using the standard encoding
    write_alphabet_size(writer, alphabet_size)?;

    // Now write the ANS distribution
    // is_flat = 1 (flat distribution - all symbols equiprobable)
    writer.write(1, 1)?;

    // For flat distribution, write log_alphabet_size bits for (alphabet_size - 1)
    // This tells the decoder how many symbols are in the flat distribution
    if log_alpha > 0 {
        writer.write(log_alpha, (alphabet_size - 1) as u64)?;
    }

    Ok(())
}

/// Write a complex prefix code using Brotli-style encoding.
///
/// For alphabet_size > 4, we need to encode using the Brotli format:
/// 1. Write hskip (2 bits): 0, 2, or 3 (number of code length symbols to skip)
/// 2. Write code length code lengths for each symbol in storage order
/// 3. Write actual code lengths using the code length Huffman tree
pub(crate) fn write_complex_prefix_code(
    writer: &mut BitWriter,
    alphabet_size: usize,
) -> Result<()> {
    eprintln!(
        "COMPLEX_PREFIX [bit {}]: alphabet_size={}",
        writer.bits_written(),
        alphabet_size
    );

    // Compute code lengths for a near-flat distribution
    // For n symbols with ceil(log2(n)) = d:
    // - Some symbols get depth d-1 (shorter codes)
    // - Remaining symbols get depth d (longer codes)
    let d = (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize;
    let pow2_d = 1usize << d;

    // Number of symbols at depth d-1 vs depth d
    // x + y = n, 2x + y = 2^d → x = 2^d - n
    let symbols_short = pow2_d.saturating_sub(alphabet_size); // depth d-1
    let symbols_long = alphabet_size.saturating_sub(symbols_short); // depth d

    // If d == 1, all symbols have depth 1 (special case)
    let (depth_short, depth_long) = if d <= 1 { (1, 1) } else { (d - 1, d) };

    eprintln!(
        "COMPLEX_PREFIX: d={}, symbols_short={} (depth {}), symbols_long={} (depth {})",
        d, symbols_short, depth_short, symbols_long, depth_long
    );

    // Build code length array
    let mut code_lengths = vec![0u8; alphabet_size];
    for cl in code_lengths.iter_mut().take(symbols_short) {
        *cl = depth_short as u8;
    }
    for cl in code_lengths.iter_mut().skip(symbols_short) {
        *cl = depth_long as u8;
    }

    // Now encode using Brotli format
    // Storage order for code length symbols
    const STORAGE_ORDER: [u8; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    // Code length code lengths encoding:
    // Decoder uses U32(0, 4, 3, 8) then additional bits for base=8.
    // In LSB-first bitstream order:
    // 0 → 00 (2 bits) - selector 0 → 0
    // 1 → 0111 (4 bits) - selector 3 (11) → 8, then 1, then 0 → len 1
    // 2 → 011 (3 bits) - selector 3 (11) → 8, then 0 → len 2
    // 3 → 10 (2 bits) - selector 2 → 3
    // 4 → 01 (2 bits) - selector 1 → 4
    // 5 → 1111 (4 bits) - selector 3 (11) → 8, then 1, then 1 → len 5
    const CL_CODE_BITS: [u8; 6] = [0b00, 0b0111, 0b011, 0b10, 0b01, 0b1111];
    const CL_CODE_LENS: [u8; 6] = [2, 4, 3, 2, 2, 4];

    // Determine which code length symbols are used
    let cl_used_short = depth_short; // code length symbol for short codes
    let cl_used_long = depth_long; // code length symbol for long codes

    // Build code length tree for the used symbols
    // We only use 1-2 distinct code lengths
    let single_depth = symbols_short == 0 || symbols_long == 0 || depth_short == depth_long;

    if single_depth {
        // All symbols have the same code length
        let cl_sym = if symbols_short > 0 {
            cl_used_short
        } else {
            cl_used_long
        };

        // Find position in storage order
        let pos = STORAGE_ORDER
            .iter()
            .position(|&x| x == cl_sym as u8)
            .unwrap_or(0);

        // hskip: number of zeros at start of code length code lengths
        // We can skip up to 3 of the first symbols in storage order if they're 0
        let skip = pos.min(3);

        // Write hskip (2 bits)
        // 0 = no skip, 2 = skip first 2, 3 = skip first 3
        let hskip = if skip >= 2 { skip } else { 0 };
        writer.write(2, hskip as u64)?;

        // Write code length code lengths
        // For single symbol in code length tree, we write code length 0 for it
        // (special case: nonzero_count == 1)
        for &cl_sym_at_i in &STORAGE_ORDER[hskip..=pos] {
            let cl_cl = if cl_sym_at_i as usize == cl_sym {
                1u8
            } else {
                0u8
            };
            writer.write(
                CL_CODE_LENS[cl_cl as usize] as usize,
                CL_CODE_BITS[cl_cl as usize] as u64,
            )?;
        }

        // Now emit code lengths for each symbol
        // With single code length symbol having code 0 (since it's the only one),
        // we don't emit any bits per symbol
        // But we need to emit `alphabet_size` instances...
        // Actually, the decoder reads using the code length Huffman tree.
        // With only one symbol (cl_sym), each read returns that symbol with 0 bits.

        // Nothing more to write - decoder will read 0 bits per code length
    } else {
        // Two distinct code lengths
        // Build a code length tree with symbols depth_short and depth_long

        // Find positions in storage order
        let pos_short = STORAGE_ORDER
            .iter()
            .position(|&x| x == cl_used_short as u8)
            .unwrap_or(0);
        let pos_long = STORAGE_ORDER
            .iter()
            .position(|&x| x == cl_used_long as u8)
            .unwrap_or(0);
        let min_pos = pos_short.min(pos_long);
        let max_pos = pos_short.max(pos_long);

        // Skip if possible (both positions > skip)
        let skip = if min_pos >= 3 {
            3
        } else if min_pos >= 2 {
            2
        } else {
            0
        };

        eprintln!(
            "COMPLEX_PREFIX: pos_short={}, pos_long={}, min={}, max={}, skip={}",
            pos_short, pos_long, min_pos, max_pos, skip
        );

        // hskip
        writer.write(2, skip as u64)?;
        eprintln!(
            "COMPLEX_PREFIX [bit {}]: wrote hskip={}",
            writer.bits_written(),
            skip
        );

        // Write code length code lengths up to and including max_pos
        // Both symbols get code length 1 (since we have exactly 2 symbols,
        // they each get 1 bit: 0 for one, 1 for the other)
        let mut space = 32i32;
        let mut num_codes = 0;
        for (idx, &storage_val) in STORAGE_ORDER[skip..=max_pos].iter().enumerate() {
            let cl_sym_at_i = storage_val as usize;
            let cl_cl = if cl_sym_at_i == cl_used_short || cl_sym_at_i == cl_used_long {
                1u8 // code length 1 for this symbol
            } else {
                0u8 // not used
            };
            writer.write(
                CL_CODE_LENS[cl_cl as usize] as usize,
                CL_CODE_BITS[cl_cl as usize] as u64,
            )?;
            eprintln!(
                "COMPLEX_PREFIX [bit {}]: storage[{}]={} -> cl_cl={}, bits={:#b} ({} bits)",
                writer.bits_written(),
                skip + idx,
                cl_sym_at_i,
                cl_cl,
                CL_CODE_BITS[cl_cl as usize],
                CL_CODE_LENS[cl_cl as usize]
            );
            if cl_cl != 0 {
                space -= 32 >> cl_cl;
                num_codes += 1;
            }
        }
        eprintln!(
            "COMPLEX_PREFIX: After cl-cl: space={}, num_codes={}",
            space, num_codes
        );

        // Assign codes to the two code length symbols
        // The one that appears first in storage order gets code 0
        // The one that appears second gets code 1
        let (first_cl, _second_cl) = if pos_short < pos_long {
            (cl_used_short, cl_used_long)
        } else {
            (cl_used_long, cl_used_short)
        };

        // Now emit code lengths for each alphabet symbol
        // symbols_short symbols use first_cl (if pos_short < pos_long) or second_cl
        // symbols_long symbols use the other
        let start_bit = writer.bits_written();
        for &sym_cl in code_lengths.iter().take(alphabet_size) {
            let bit = if sym_cl as usize == first_cl {
                0u64
            } else {
                1u64
            };
            writer.write(1, bit)?;
        }
        eprintln!(
            "COMPLEX_PREFIX [bit {}]: wrote {} code lengths ({} bits)",
            writer.bits_written(),
            alphabet_size,
            writer.bits_written() - start_bit
        );
    }

    Ok(())
}

/// Write a signed 32-bit integer using JXL S32 encoding.
///
/// S32 uses a variable-length encoding:
/// - Small values (±63) use fewer bits
/// - Larger values use more bits
#[allow(dead_code)]
pub(crate) fn write_signed_varint_traced(
    writer: &mut BitWriter,
    value: i32,
    _field: &str,
) -> Result<()> {
    // Convert signed to unsigned using zigzag encoding
    let unsigned = if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    };

    // Use variable-length encoding similar to U32
    if unsigned == 0 {
        trace_write!(
            writer,
            2,
            0,
            field,
            &format!("selector=0 → {} (zigzag=0)", value)
        )?;
    } else if unsigned <= 16 {
        trace_write!(writer, 2, 1, &format!("{}.selector", field), "1")?;
        trace_write!(
            writer,
            4,
            (unsigned - 1) as u64,
            &format!("{}.value", field),
            &format!("{} (zigzag={})", value, unsigned)
        )?;
    } else if unsigned <= 272 {
        trace_write!(writer, 2, 2, &format!("{}.selector", field), "2")?;
        trace_write!(
            writer,
            8,
            (unsigned - 17) as u64,
            &format!("{}.value", field),
            &format!("{} (zigzag={})", value, unsigned)
        )?;
    } else {
        trace_write!(writer, 2, 3, &format!("{}.selector", field), "3")?;
        trace_write!(
            writer,
            12,
            (unsigned.saturating_sub(273)) as u64,
            &format!("{}.value", field),
            &format!("{} (zigzag={})", value, unsigned)
        )?;
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn write_signed_varint(writer: &mut BitWriter, value: i32) -> Result<()> {
    // Convert signed to unsigned using zigzag encoding
    let unsigned = if value >= 0 {
        (value as u32) << 1
    } else {
        ((-value as u32) << 1) - 1
    };

    // Use variable-length encoding similar to U32
    // U32Enc: Val(0), BitsOffset(4, 1), BitsOffset(8, 17), BitsOffset(12, 273)
    if unsigned == 0 {
        writer.write(2, 0)?; // selector 0 = 0
    } else if unsigned <= 16 {
        writer.write(2, 1)?; // selector 1
        writer.write(4, (unsigned - 1) as u64)?;
    } else if unsigned <= 272 {
        writer.write(2, 2)?; // selector 2
        writer.write(8, (unsigned - 17) as u64)?;
    } else {
        writer.write(2, 3)?; // selector 3
        writer.write(12, (unsigned.saturating_sub(273)) as u64)?;
    }

    Ok(())
}
