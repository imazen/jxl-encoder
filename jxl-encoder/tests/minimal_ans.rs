//! Minimal ANS encoding test.
//!
//! Tests the ANS encoding with a single distribution and verifies
//! the bitstream is correct.

use jxl_encoder::bit_writer::BitWriter;
use jxl_encoder::entropy_coding::ans::{AnsDistribution, AnsEncoder};

/// Test encoding and decoding a single symbol with a flat distribution.
#[test]
fn test_single_symbol_roundtrip() {
    // Create a flat distribution with 2 symbols (each freq = 2048)
    let dist = AnsDistribution::flat(2).unwrap();

    println!("Distribution created:");
    println!("  symbol 0: freq={}", dist.symbols[0].freq);
    println!("  symbol 1: freq={}", dist.symbols[1].freq);

    // Encode symbol 0
    let mut encoder = AnsEncoder::new();
    let initial_state = encoder.state();
    println!("\nEncoding symbol 0:");
    println!("  initial state: 0x{:08x}", initial_state);

    encoder.put_symbol(&dist.symbols[0]);
    let final_state = encoder.state();
    println!("  final state: 0x{:08x}", final_state);

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoder.finalize(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("\nBitstream ({} bytes): {:02x?}", bytes.len(), bytes);

    // Manually decode to verify
    // Read 32-bit state (little-endian)
    let state = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    println!("\nDecoding:");
    println!("  read state: 0x{:08x}", state);

    // Decode symbol
    let idx = state & 0xFFF;
    let symbol = if idx < 2048 { 0 } else { 1 };
    let offset = idx % 2048;
    let freq = 2048u32;

    println!("  idx: {}", idx);
    println!("  decoded symbol: {}", symbol);
    println!("  offset: {}", offset);

    // Update state
    let next_state = (state >> 12) * freq + offset;
    println!("  next_state: 0x{:08x}", next_state);

    // Verify
    assert_eq!(symbol, 0, "Decoded wrong symbol");
    assert_eq!(next_state, 0x130000, "Final state should be 0x130000");

    println!("\n✓ Single symbol roundtrip passed!");
}

/// Test encoding and decoding multiple symbols using the jxl-rs compatible decoder.
#[test]
fn test_multiple_symbols_roundtrip() {
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};

    // Create a flat distribution with 4 symbols
    let counts = [1024i32, 1024, 1024, 1024]; // Each symbol gets 1024/4096 probability
    let dist = AnsDistribution::from_normalized_counts(&counts).unwrap();

    // Symbols to encode (in forward order, we'll reverse for encoding)
    let symbols: Vec<usize> = vec![0, 1, 2, 3, 0, 1];

    println!("Encoding {} symbols: {:?}", symbols.len(), symbols);

    // Encode in reverse order
    let mut encoder = AnsEncoder::new();
    for &sym in symbols.iter().rev() {
        encoder.put_symbol(&dist.symbols[sym]);
    }

    let final_state = encoder.state();
    println!("Encoder final state: 0x{:08x}", final_state);

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoder.finalize(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("Bitstream ({} bytes): {:02x?}", bytes.len(), bytes);

    // Build decoder histogram by writing and reading back
    // (In practice, the histogram is serialized separately)
    use jxl_encoder::entropy_coding::ans::{ANSEncodingHistogram, ANSHistogramStrategy};
    use jxl_encoder::entropy_coding::histogram::Histogram;
    let histo = Histogram::from_counts(&[1024, 1024, 1024, 1024]);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let mut hist_writer = BitWriter::new();
    ans_histo.write(&mut hist_writer).unwrap();
    let hist_bytes = hist_writer.finish_with_padding();

    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).unwrap();

    // Decode using jxl-rs compatible decoder
    let mut br = BitReader::new(&bytes);
    let mut ans_reader = AnsReader::init(&mut br).unwrap();

    println!("\nDecoding:");
    let mut decoded = Vec::new();

    for i in 0..symbols.len() {
        let symbol = decoded_hist.read(&mut br, &mut ans_reader.0) as usize;
        println!(
            "  step {}: symbol={}, state=0x{:08x}",
            i, symbol, ans_reader.0
        );
        decoded.push(symbol);
    }

    println!("\nFinal state: 0x{:08x}", ans_reader.0);
    println!("Decoded: {:?}", decoded);
    println!("Expected: {:?}", symbols);

    assert_eq!(decoded, symbols);
    assert!(
        ans_reader.check_final_state().is_ok(),
        "Final state should be 0x130000, got 0x{:08x}",
        ans_reader.0
    );

    println!("\n✓ Multiple symbols roundtrip passed!");
}

/// Test encoding and decoding with a non-flat distribution using jxl-rs compatible decoder.
#[test]
fn test_nonflat_distribution_roundtrip() {
    use jxl_encoder::entropy_coding::ans::AnsDistribution;
    use jxl_encoder::entropy_coding::ans::{ANSEncodingHistogram, ANSHistogramStrategy};
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Create a skewed distribution
    let histo = Histogram::from_counts(&[100, 50, 25, 10]);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    println!("Non-flat distribution:");
    for (i, sym) in dist.symbols.iter().enumerate() {
        println!("  symbol {}: freq={}", i, sym.freq);
    }

    // Symbols to encode (matching the distribution)
    let symbols: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 3];

    println!("\nEncoding {} symbols: {:?}", symbols.len(), symbols);

    // Encode in reverse order
    let mut encoder = AnsEncoder::new();
    for (i, &sym) in symbols.iter().rev().enumerate() {
        let state_before = encoder.state();
        encoder.put_symbol(&dist.symbols[sym]);
        println!(
            "  encoded sym {} (idx {}): state 0x{:08x} -> 0x{:08x}",
            sym,
            symbols.len() - 1 - i,
            state_before,
            encoder.state()
        );
    }

    let final_state = encoder.state();
    println!("Encoder final state: 0x{:08x}", final_state);

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoder.finalize(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("Bitstream ({} bytes): {:02x?}", bytes.len(), bytes);

    // Build decoder histogram by writing and reading back
    let mut hist_writer = BitWriter::new();
    ans_histo.write(&mut hist_writer).unwrap();
    let hist_bytes = hist_writer.finish_with_padding();

    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).unwrap();

    println!(
        "\nDecoded histogram frequencies: {:?}",
        &decoded_hist.frequencies[..4]
    );

    // Decode using jxl-rs compatible decoder
    let mut br = BitReader::new(&bytes);
    let mut ans_reader = AnsReader::init(&mut br).unwrap();

    println!("\nDecoding:");
    let mut decoded = Vec::new();

    for i in 0..symbols.len() {
        let symbol = decoded_hist.read(&mut br, &mut ans_reader.0) as usize;
        println!(
            "  step {}: symbol={}, state=0x{:08x}",
            i, symbol, ans_reader.0
        );
        decoded.push(symbol);
    }

    println!("\nFinal state: 0x{:08x}", ans_reader.0);
    println!("Decoded: {:?}", decoded);
    println!("Expected: {:?}", symbols);

    assert_eq!(decoded, symbols);
    assert!(
        ans_reader.check_final_state().is_ok(),
        "Final state should be 0x130000, got 0x{:08x}",
        ans_reader.0
    );

    println!("\n✓ Non-flat distribution roundtrip passed!");
}

/// Test that the histogram we write can be decoded correctly.
#[test]
fn test_histogram_serialization() {
    use jxl_encoder::entropy_coding::ans::{ANSEncodingHistogram, ANSHistogramStrategy};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Create and normalize a histogram
    let histo = Histogram::from_counts(&[100, 50, 25, 10]);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

    println!("Original histogram:");
    println!("  counts: {:?}", ans_histo.counts);
    println!(
        "  method: {}, alphabet_size: {}, omit_pos: {}",
        ans_histo.method, ans_histo.alphabet_size, ans_histo.omit_pos
    );

    // Write to bitstream
    let mut writer = BitWriter::new();
    ans_histo.write(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("\nSerialized ({} bytes): {:02x?}", bytes.len(), bytes);

    // Verify the bytes match what we expect
    // For method=1 (small code), num_symbols=2:
    // - 1 bit: small_tree = 1
    // - 1 bit: nsym - 1 = 1 (for 2 symbols)
    // - VarLenUint8 for each symbol index
    // - 12 bits for first symbol's count

    // For method > 1 (general code):
    // - 1 bit: small_tree = 0
    // - 1 bit: flat = 0
    // - shift encoding
    // - VarLenUint8 for alphabet_size - 3
    // - logcount for each symbol
    // - precision bits for non-omit symbols

    // This is a general histogram (4 symbols), so method should be > 1
    assert!(ans_histo.method > 1, "Expected general histogram");

    // Basic sanity checks
    let sum: i32 = ans_histo.counts.iter().sum();
    assert_eq!(sum, 4096, "Sum should be 4096");

    println!("\n✓ Histogram serialization test passed!");
}

/// Test that the histogram bytes we write can be decoded by jxl-rs decoder.
/// The actual roundtrip verification is done by test_decode_general_histogram in ans_decode.rs
#[test]
fn test_histogram_byte_decode() {
    use jxl_encoder::entropy_coding::ans::{ANSEncodingHistogram, ANSHistogramStrategy};
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Create a simple 4-symbol histogram
    let histo = Histogram::from_counts(&[100, 50, 25, 10]);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

    println!("Original histogram:");
    println!("  counts: {:?}", ans_histo.counts);
    println!(
        "  method: {} (shift={})",
        ans_histo.method,
        ans_histo.method as i32 - 1
    );
    println!(
        "  alphabet_size: {}, omit_pos: {}",
        ans_histo.alphabet_size, ans_histo.omit_pos
    );

    // Write to bitstream
    let mut writer = BitWriter::new();
    ans_histo.write(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("\nSerialized ({} bytes): {:02x?}", bytes.len(), bytes);

    // Decode using the actual jxl-rs decoder
    let mut br = BitReader::new(&bytes);
    let decoded = AnsHistogram::decode(&mut br, 6).unwrap();

    println!(
        "Decoded frequencies: {:?}",
        &decoded.frequencies[..ans_histo.alphabet_size]
    );

    // Verify frequencies match what we wrote
    for i in 0..ans_histo.alphabet_size {
        assert_eq!(
            decoded.frequencies[i], ans_histo.counts[i] as u16,
            "Frequency mismatch at symbol {}",
            i
        );
    }

    // Verify sum is 4096
    let sum: u16 = decoded.frequencies.iter().sum();
    assert_eq!(sum, 4096, "Sum should be 4096");

    println!("\n✓ Histogram byte decode passed!");
}

/// Encode a uint value using HybridUint config (split=4, msb=2, lsb=0)
fn hybrid_uint_encode(value: u32) -> (u32, u32, u32) {
    if value < 16 {
        (value, 0, 0) // (token, nbits, bits)
    } else {
        let n = 31 - value.leading_zeros(); // floor_log2
        let m = value - (1 << n);
        let token = (n << 2) + (m >> (n - 2));
        let nbits = n - 2;
        let bits = value & ((1u32 << nbits) - 1);
        (token, nbits, bits)
    }
}

/// Decode a HybridUint token + extra bits
fn hybrid_uint_decode(token: u32, extra_bits: u32) -> u32 {
    if token < 16 {
        token
    } else {
        let n = token >> 2; // Number of total bits in value
        let msb = token & 3; // 2 MSB bits encoded in token
        let base = (1 << n) + (msb << (n - 2));
        base + extra_bits
    }
}

/// Test ANS with HybridUint encoding (tokens with extra bits)
/// This tests the combination that full image encoding uses.
#[test]
fn test_ans_with_hybrid_uint() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{
        ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
    };
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Test values that will produce varying HybridUint tokens
    let values: Vec<u32> = vec![0, 1, 2, 5, 10, 20, 50, 100, 200];

    // Build histogram of token symbols
    let mut symbol_counts = vec![0i32; 64];
    for &val in &values {
        let (token, _, _) = hybrid_uint_encode(val);
        symbol_counts[token as usize] += 1;
    }

    let histo = Histogram::from_counts(&symbol_counts);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    println!("HybridUint test:");
    println!("  Values: {:?}", values);
    for &val in &values {
        let (token, nbits, bits) = hybrid_uint_encode(val);
        println!(
            "  val={} -> token={}, nbits={}, bits={}",
            val, token, nbits, bits
        );
    }

    // Encode in reverse order (as full encoder does)
    let mut encoder = AnsEncoder::new();
    for &val in values.iter().rev() {
        let (token, nbits, bits) = hybrid_uint_encode(val);
        encoder.push_bits(bits, nbits as u8);
        let info = dist
            .get(token as usize)
            .expect("Symbol not in distribution");
        encoder.put_symbol(info);
    }

    println!("  Encoder final state: 0x{:08x}", encoder.state());

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoder.finalize(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("  Encoded {} bytes: {:02x?}", bytes.len(), bytes);

    // Build decoder histogram
    let mut hist_writer = BitWriter::new();
    ans_histo.write(&mut hist_writer).unwrap();
    let hist_bytes = hist_writer.finish_with_padding();

    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).unwrap();

    // Decode using jxl-rs compatible decoder
    let mut br = BitReader::new(&bytes);
    let mut ans_reader = AnsReader::init(&mut br).unwrap();

    println!("Decoding:");
    let mut decoded_values = Vec::new();
    for i in 0..values.len() {
        let token = decoded_hist.read(&mut br, &mut ans_reader.0);
        // Decode HybridUint extra bits
        let nbits = if token < 16 { 0 } else { (token >> 2) - 2 };
        let extra_bits = if nbits > 0 {
            br.read(nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };
        let value = hybrid_uint_decode(token, extra_bits);
        println!(
            "  step {}: token={}, nbits={}, extra_bits={}, value={}, state=0x{:08x}",
            i, token, nbits, extra_bits, value, ans_reader.0
        );
        decoded_values.push(value);
    }

    println!("  Final state: 0x{:08x}", ans_reader.0);
    println!("  Original: {:?}", values);
    println!("  Decoded:  {:?}", decoded_values);

    assert_eq!(decoded_values, values, "Decoded values should match");
    assert!(
        ans_reader.check_final_state().is_ok(),
        "Final state should be 0x130000, got 0x{:08x}",
        ans_reader.0
    );
}

/// Test ANS encoding of a simple token stream similar to what the full encoder produces.
/// This mimics the DC token stream with multiple contexts.
#[test]
fn test_ans_multi_context() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{
        ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
    };
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Simulate tokens from different contexts (like DC encoding)
    // Context 0: DC values
    // Context 1: Other metadata
    let tokens: Vec<(u32, u32)> = vec![
        (0, 10), // context 0, value 10
        (0, 12), // context 0, value 12
        (1, 5),  // context 1, value 5
        (0, 8),  // context 0, value 8
        (1, 3),  // context 1, value 3
    ];

    // Build per-context histograms (simplified - assume all map to single distribution)
    let mut symbol_counts = vec![0i32; 64];
    for &(_, val) in &tokens {
        let (token, _, _) = hybrid_uint_encode(val);
        symbol_counts[token as usize] += 1;
    }

    let histo = Histogram::from_counts(&symbol_counts);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    println!("Multi-context test:");
    println!("  Tokens: {:?}", tokens);
    let display_len = ans_histo.counts.len().min(16);
    println!(
        "  Distribution counts[0..{}]: {:?}",
        display_len,
        &ans_histo.counts[..display_len]
    );

    // Encode in reverse order
    let mut encoder = AnsEncoder::new();
    for &(ctx, val) in tokens.iter().rev() {
        let (token, nbits, bits) = hybrid_uint_encode(val);
        encoder.push_bits(bits, nbits as u8);
        let info = dist
            .get(token as usize)
            .expect("Symbol not in distribution");
        println!(
            "  Encoding ctx={}, val={}, token={}, state before=0x{:08x}",
            ctx,
            val,
            token,
            encoder.state()
        );
        encoder.put_symbol(info);
    }

    println!("  Final encoder state: 0x{:08x}", encoder.state());

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoder.finalize(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();
    println!("  Encoded {} bytes", bytes.len());

    // Build decoder histogram
    let mut hist_writer = BitWriter::new();
    ans_histo.write(&mut hist_writer).unwrap();
    let hist_bytes = hist_writer.finish_with_padding();

    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).unwrap();

    // Decode
    let mut br = BitReader::new(&bytes);
    let mut ans_reader = AnsReader::init(&mut br).unwrap();

    println!("Decoding:");
    let mut decoded_tokens: Vec<(u32, u32)> = Vec::new();
    for (i, &(ctx, _)) in tokens.iter().enumerate() {
        let token = decoded_hist.read(&mut br, &mut ans_reader.0);
        let nbits = if token < 16 { 0 } else { (token >> 2) - 2 };
        let extra_bits = if nbits > 0 {
            br.read(nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };
        let value = hybrid_uint_decode(token, extra_bits);
        println!(
            "  step {}: ctx={}, token={}, value={}, state=0x{:08x}",
            i, ctx, token, value, ans_reader.0
        );
        decoded_tokens.push((ctx, value));
    }

    println!("  Final state: 0x{:08x}", ans_reader.0);

    // Verify
    let decoded_values: Vec<u32> = decoded_tokens.iter().map(|(_, v)| *v).collect();
    let expected_values: Vec<u32> = tokens.iter().map(|(_, v)| *v).collect();
    assert_eq!(
        decoded_values, expected_values,
        "Decoded values should match"
    );
    assert!(
        ans_reader.check_final_state().is_ok(),
        "Final state should be 0x130000, got 0x{:08x}",
        ans_reader.0
    );
}

/// Test the complete ANS entropy code format (header + tokens).
/// This mimics exactly what write_entropy_code_ans + write_tokens_ans produce.
#[test]
fn test_ans_full_entropy_code_format() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{
        ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
    };
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    // Create tokens like the encoder does
    let tokens: Vec<(u8, u32)> = vec![
        (0, 5),  // context 0, value 5
        (1, 10), // context 1, value 10
        (0, 3),  // context 0, value 3
        (2, 20), // context 2, value 20
        (1, 8),  // context 1, value 8
    ];

    let _num_contexts = 3;

    // Build histogram from tokens (all contexts map to one histogram for simplicity)
    let mut symbol_counts = vec![0i32; 64];
    for &(_, val) in &tokens {
        let (token, _, _) = hybrid_uint_encode(val);
        symbol_counts[token as usize] += 1;
    }

    let histo = Histogram::from_counts(&symbol_counts);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    println!("=== ANS Full Entropy Code Format Test ===");
    println!("Tokens: {:?}", tokens);
    println!(
        "Histogram method={}, alphabet_size={}",
        ans_histo.method, ans_histo.alphabet_size
    );

    // === WRITE ENTROPY CODE HEADER ===
    let mut header_writer = BitWriter::new();

    // 1. Context map (simple format: all contexts map to histogram 0)
    header_writer.write(1, 1).unwrap(); // simple_context_map = true
    header_writer.write(2, 0).unwrap(); // nbits = 0 (all zeros)

    // 2. use_prefix_code = 0 (ANS)
    header_writer.write(1, 0).unwrap();

    // 3. log_alpha_size - 5 (we use 6)
    header_writer.write(2, 1).unwrap(); // 6 - 5 = 1

    // 4. HybridUint config (for log_alpha_size=6)
    // split_exponent: 3 bits, value 4
    // msb_in_token: 3 bits, value 2
    // lsb_in_token: 2 bits, value 0
    header_writer.write(3, 4).unwrap();
    header_writer.write(3, 2).unwrap();
    header_writer.write(2, 0).unwrap();

    // 5. Distribution
    ans_histo.write(&mut header_writer).unwrap();

    let header_bytes = header_writer.finish_with_padding();
    println!(
        "Header bytes ({} bytes): {:02x?}",
        header_bytes.len(),
        &header_bytes[..header_bytes.len().min(32)]
    );

    // === WRITE TOKEN STREAM ===
    let mut encoder = AnsEncoder::new();
    for &(_ctx, val) in tokens.iter().rev() {
        let (token, nbits, bits) = hybrid_uint_encode(val);
        encoder.push_bits(bits, nbits as u8);
        let info = dist
            .get(token as usize)
            .expect("Symbol not in distribution");
        encoder.put_symbol(info);
    }

    let mut token_writer = BitWriter::new();
    encoder.finalize(&mut token_writer).unwrap();
    let token_bytes = token_writer.finish_with_padding();
    println!(
        "Token bytes ({} bytes): {:02x?}",
        token_bytes.len(),
        token_bytes
    );

    // === DECODE HEADER ===
    let mut header_br = BitReader::new(&header_bytes);

    // 1. Read context map
    let is_simple = header_br.read(1).unwrap() != 0;
    assert!(is_simple, "Expected simple context map");
    let nbits = header_br.read(2).unwrap() as usize;
    assert_eq!(nbits, 0, "Expected nbits=0");
    println!("Decoded context map: simple, nbits={}", nbits);

    // 2. Read use_prefix_code
    let use_prefix = header_br.read(1).unwrap() != 0;
    assert!(!use_prefix, "Expected ANS (use_prefix_code=0)");
    println!("Decoded use_prefix_code={}", use_prefix);

    // 3. Read log_alpha_size
    let log_alpha = header_br.read(2).unwrap() as usize + 5;
    assert_eq!(log_alpha, 6, "Expected log_alpha_size=6");
    println!("Decoded log_alpha_size={}", log_alpha);

    // 4. Read HybridUint config
    let split_exp = header_br.read(3).unwrap();
    let msb_in_tok = header_br.read(3).unwrap();
    let lsb_in_tok = header_br.read(2).unwrap();
    println!(
        "Decoded HybridUint: split={}, msb={}, lsb={}",
        split_exp, msb_in_tok, lsb_in_tok
    );
    assert_eq!(split_exp, 4);
    assert_eq!(msb_in_tok, 2);
    assert_eq!(lsb_in_tok, 0);

    // 5. Read distribution
    let decoded_hist = AnsHistogram::decode(&mut header_br, log_alpha).unwrap();
    println!(
        "Decoded histogram frequencies: {:?}",
        &decoded_hist.frequencies[..decoded_hist.frequencies.len().min(16)]
    );

    // Verify frequencies match
    for i in 0..ans_histo.alphabet_size {
        assert_eq!(
            decoded_hist.frequencies[i], ans_histo.counts[i] as u16,
            "Frequency mismatch at symbol {}: wrote {}, got {}",
            i, ans_histo.counts[i], decoded_hist.frequencies[i]
        );
    }

    // === DECODE TOKENS ===
    let mut token_br = BitReader::new(&token_bytes);
    let mut ans_reader = AnsReader::init(&mut token_br).unwrap();
    println!("Initial ANS state: 0x{:08x}", ans_reader.0);

    let mut decoded_tokens = Vec::new();
    for (i, &(ctx, _expected_val)) in tokens.iter().enumerate() {
        // Read ANS symbol
        let token = decoded_hist.read(&mut token_br, &mut ans_reader.0);

        // Read HybridUint extra bits
        let nbits = if token < 16 { 0 } else { (token >> 2) - 2 };
        let extra_bits = if nbits > 0 {
            token_br.read(nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };
        let value = hybrid_uint_decode(token, extra_bits);

        println!(
            "  step {}: ctx={}, token={}, nbits={}, extra={}, value={}, state=0x{:08x}",
            i, ctx, token, nbits, extra_bits, value, ans_reader.0
        );
        decoded_tokens.push((ctx, value));
    }

    println!("Final ANS state: 0x{:08x}", ans_reader.0);

    // Verify
    let decoded_values: Vec<u32> = decoded_tokens.iter().map(|(_, v)| *v).collect();
    let expected_values: Vec<u32> = tokens.iter().map(|(_, v)| *v).collect();

    assert_eq!(
        decoded_values, expected_values,
        "Decoded values should match"
    );
    assert!(
        ans_reader.check_final_state().is_ok(),
        "ANS checksum failed: final state 0x{:08x}, expected 0x130000",
        ans_reader.0
    );

    println!("=== Test passed! ===");
}

/// Test with LZ77 flag included (matching real encoder format).
#[test]
fn test_ans_with_lz77_flag() {
    use jxl_encoder::bit_writer::BitWriter;
    use jxl_encoder::entropy_coding::ans::{
        ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution, AnsEncoder,
    };
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};
    use jxl_encoder::entropy_coding::histogram::Histogram;

    let tokens: Vec<u32> = vec![5, 10, 3, 20, 8];

    // Build histogram
    let mut symbol_counts = vec![0i32; 64];
    for &val in &tokens {
        let (token, _, _) = hybrid_uint_encode(val);
        symbol_counts[token as usize] += 1;
    }

    let histo = Histogram::from_counts(&symbol_counts);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    println!("=== Test with LZ77 Flag ===");

    // === WRITE COMPLETE HEADER (like encoder does) ===
    let mut header_writer = BitWriter::new();

    // 0. LZ77 enabled = false (1 bit)
    header_writer.write(1, 0).unwrap();

    // 1. Context map (simple, single histogram)
    header_writer.write(1, 1).unwrap(); // simple
    header_writer.write(2, 0).unwrap(); // nbits = 0

    // 2. use_prefix_code = 0 (ANS)
    header_writer.write(1, 0).unwrap();

    // 3. log_alpha_size - 5 = 1
    header_writer.write(2, 1).unwrap();

    // 4. HybridUint config
    header_writer.write(3, 4).unwrap();
    header_writer.write(3, 2).unwrap();
    header_writer.write(2, 0).unwrap();

    // 5. Distribution
    ans_histo.write(&mut header_writer).unwrap();

    let header_bytes = header_writer.finish_with_padding();
    println!(
        "Header with LZ77 ({} bytes): {:02x?}",
        header_bytes.len(),
        &header_bytes[..header_bytes.len().min(32)]
    );

    // === WRITE TOKEN STREAM ===
    let mut encoder = AnsEncoder::new();
    for &val in tokens.iter().rev() {
        let (token, nbits, bits) = hybrid_uint_encode(val);
        encoder.push_bits(bits, nbits as u8);
        let info = dist.get(token as usize).expect("Symbol not found");
        encoder.put_symbol(info);
    }

    let mut token_writer = BitWriter::new();
    encoder.finalize(&mut token_writer).unwrap();
    let token_bytes = token_writer.finish_with_padding();
    println!(
        "Token bytes ({} bytes): {:02x?}",
        token_bytes.len(),
        token_bytes
    );

    // === DECODE HEADER ===
    let mut br = BitReader::new(&header_bytes);

    // 0. LZ77 enabled
    let lz77_enabled = br.read(1).unwrap() != 0;
    assert!(!lz77_enabled, "Expected LZ77 disabled");
    println!("LZ77 enabled: {}", lz77_enabled);

    // 1. Context map
    let is_simple = br.read(1).unwrap() != 0;
    let nbits = br.read(2).unwrap() as usize;
    println!("Context map: simple={}, nbits={}", is_simple, nbits);

    // 2. use_prefix_code
    let use_prefix = br.read(1).unwrap() != 0;
    assert!(!use_prefix, "Expected ANS");
    println!("use_prefix_code: {}", use_prefix);

    // 3. log_alpha_size
    let log_alpha = br.read(2).unwrap() as usize + 5;
    println!("log_alpha_size: {}", log_alpha);

    // 4. HybridUint config
    let split_exp = br.read(3).unwrap();
    let msb = br.read(3).unwrap();
    let lsb = br.read(2).unwrap();
    println!("HybridUint: split={}, msb={}, lsb={}", split_exp, msb, lsb);

    // 5. Distribution
    let decoded_hist = AnsHistogram::decode(&mut br, log_alpha).unwrap();
    println!("Decoded frequencies: {:?}", &decoded_hist.frequencies[..16]);

    // === DECODE TOKENS ===
    let mut token_br = BitReader::new(&token_bytes);
    let mut ans_reader = AnsReader::init(&mut token_br).unwrap();

    let mut decoded_values = Vec::new();
    for i in 0..tokens.len() {
        let token = decoded_hist.read(&mut token_br, &mut ans_reader.0);
        let nbits = if token < 16 { 0 } else { (token >> 2) - 2 };
        let extra = if nbits > 0 {
            token_br.read(nbits as usize).unwrap_or(0) as u32
        } else {
            0
        };
        let value = hybrid_uint_decode(token, extra);
        println!(
            "  step {}: token={}, value={}, state=0x{:08x}",
            i, token, value, ans_reader.0
        );
        decoded_values.push(value);
    }

    println!("Final state: 0x{:08x}", ans_reader.0);
    println!("Decoded: {:?}", decoded_values);
    println!("Expected: {:?}", tokens);

    assert_eq!(decoded_values, tokens);
    assert!(
        ans_reader.check_final_state().is_ok(),
        "ANS checksum failed: 0x{:08x}",
        ans_reader.0
    );

    println!("=== Test passed! ===");
}
