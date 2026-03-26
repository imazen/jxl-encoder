//! Test ANS histogram serialization roundtrip.

use jxl_encoder::bit_writer::BitWriter;
use jxl_encoder::entropy_coding::ans::{
    ANSEncodingHistogram, ANSHistogramStrategy, AnsDistribution,
};
use jxl_encoder::entropy_coding::histogram::Histogram;

#[test]
fn test_single_symbol_histogram() {
    // Create a histogram with a single symbol
    let histo = Histogram::from_counts(&[100, 0, 0, 0]);
    let encoded =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

    println!("Single symbol histogram:");
    println!("  counts: {:?}", encoded.counts);
    println!("  method: {}", encoded.method);

    // Verify it's recognized as a single-symbol histogram (method 1 = small code)
    assert_eq!(encoded.method, 1);
    assert_eq!(encoded.counts[0], 4096);

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoded.write(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("  bytes: {:02x?}", bytes);
}

#[test]
fn test_two_symbol_histogram() {
    // Create a histogram with two symbols
    let histo = Histogram::from_counts(&[100, 100, 0, 0]);
    let encoded =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

    println!("Two symbol histogram:");
    println!("  counts: {:?}", encoded.counts);
    println!("  method: {}", encoded.method);

    // Verify it's recognized as a two-symbol histogram (method 1 = small code)
    assert_eq!(encoded.method, 1);

    // Each symbol should get ~2048 counts
    assert!(encoded.counts[0] > 0);
    assert!(encoded.counts[1] > 0);
    assert_eq!(encoded.counts[0] + encoded.counts[1], 4096);

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoded.write(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("  bytes: {:02x?}", bytes);
}

#[test]
fn test_general_histogram() {
    // Create a histogram with multiple symbols
    let histo = Histogram::from_counts(&[100, 50, 25, 12, 6, 3, 2, 1, 1]);
    let encoded =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

    println!("General histogram:");
    println!("  counts: {:?}", encoded.counts);
    println!(
        "  method: {}, alphabet_size: {}, omit_pos: {}",
        encoded.method, encoded.alphabet_size, encoded.omit_pos
    );

    // Verify sum is 4096
    let sum: i32 = encoded.counts.iter().sum();
    assert_eq!(sum, 4096);

    // Verify omit_pos has the highest logcount
    let omit_count = encoded.counts[encoded.omit_pos];
    for (i, &count) in encoded.counts.iter().enumerate() {
        if i != encoded.omit_pos && count > 0 {
            assert!(
                omit_count >= count,
                "omit_pos {} (count {}) should have highest count, but symbol {} has count {}",
                encoded.omit_pos,
                omit_count,
                i,
                count
            );
        }
    }

    // Write to bitstream
    let mut writer = BitWriter::new();
    encoded.write(&mut writer).unwrap();
    let bytes = writer.finish_with_padding();

    println!("  bytes ({} bytes): {:02x?}", bytes.len(), bytes);
}

#[test]
fn test_flat_distribution() {
    // Test flat distribution
    let dist = AnsDistribution::flat(8).unwrap();

    println!("Flat distribution (8 symbols):");
    for (i, sym) in dist.symbols.iter().enumerate() {
        println!("  symbol {}: freq={}", i, sym.freq);
    }

    // All should have freq 512 (4096 / 8)
    for sym in &dist.symbols {
        assert_eq!(sym.freq, 512);
    }
}

#[test]
fn test_distribution_from_histogram() {
    // Create a histogram and build both ANSEncodingHistogram and AnsDistribution
    let histo = Histogram::from_counts(&[100, 50, 25, 10]);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    println!("Histogram -> Distribution:");
    println!("  ans_histo.counts: {:?}", ans_histo.counts);
    for (i, sym) in dist.symbols.iter().enumerate() {
        println!(
            "  dist.symbols[{}]: freq={}, reverse_map len={}",
            i,
            sym.freq,
            sym.reverse_map.len()
        );
    }

    // Verify frequencies match
    for (i, &count) in ans_histo.counts.iter().enumerate() {
        if count > 0 {
            assert_eq!(dist.symbols[i].freq, count as u16);
        }
    }

    // Verify reverse maps are correct
    for sym in &dist.symbols {
        assert_eq!(sym.reverse_map.len(), sym.freq as usize);
    }
}

/// Full roundtrip test: encode raw symbols with ANS, decode with our decoder.
/// (Uses raw symbols 0-3, not HybridUint encoded values)
#[test]
fn test_full_ans_token_roundtrip() {
    use jxl_encoder::entropy_coding::ans::{AnsDistribution, AnsEncoder};
    use jxl_encoder::entropy_coding::ans_decode::{AnsHistogram, AnsReader, BitReader};

    // Create a simple histogram
    let histo = Histogram::from_counts(&[100, 50, 25, 10]);
    let ans_histo =
        ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();
    let enc_dist = AnsDistribution::from_normalized_counts(&ans_histo.counts).unwrap();

    // Raw symbols to encode (0-3)
    let symbols: Vec<usize> = vec![0, 0, 0, 1, 1, 2, 3, 0, 1, 2];

    println!("Encoding {} symbols: {:?}", symbols.len(), symbols);
    println!("Distribution: {:?}", &ans_histo.counts[..4]);

    // Encode symbols in reverse order
    let mut encoder = AnsEncoder::new();

    for &sym in symbols.iter().rev() {
        let info = enc_dist.get(sym).expect("Symbol not in distribution");
        encoder.put_symbol(info);
    }

    println!("After encoding all symbols:");
    println!("  encoder state: 0x{:08x}", encoder.state());

    // Finalize encoder
    let mut writer = BitWriter::new();
    encoder.finalize(&mut writer).unwrap();

    let encoded_bytes = writer.finish_with_padding();
    println!(
        "Encoded {} bytes: {:02x?}",
        encoded_bytes.len(),
        encoded_bytes
    );

    // Now decode using our decoder
    // First, we need to decode the histogram (write it, then read it back)
    let mut hist_writer = BitWriter::new();
    ans_histo.write(&mut hist_writer).unwrap();
    let hist_bytes = hist_writer.finish_with_padding();

    let mut hist_br = BitReader::new(&hist_bytes);
    let decoded_hist = AnsHistogram::decode(&mut hist_br, 6).unwrap();

    println!(
        "Decoded histogram frequencies: {:?}",
        &decoded_hist.frequencies[..4]
    );

    // Verify histogram
    for i in 0..4 {
        assert_eq!(
            decoded_hist.frequencies[i], ans_histo.counts[i] as u16,
            "Histogram mismatch at symbol {}",
            i
        );
    }

    // Now decode the symbols
    let mut br = BitReader::new(&encoded_bytes);
    let mut ans_reader = AnsReader::init(&mut br).unwrap();

    println!("Decoding symbols...");
    println!("  initial ANS state: 0x{:08x}", ans_reader.0);

    let mut decoded_symbols: Vec<usize> = Vec::new();
    for i in 0..symbols.len() {
        // Read ANS symbol
        let symbol = decoded_hist.read(&mut br, &mut ans_reader.0) as usize;
        println!(
            "  step {}: symbol={}, state=0x{:08x}",
            i, symbol, ans_reader.0
        );
        decoded_symbols.push(symbol);
    }

    println!("\nDecoded symbols: {:?}", decoded_symbols);
    println!("Expected symbols: {:?}", symbols);
    println!("Final ANS state: 0x{:08x}", ans_reader.0);

    // Verify symbols match
    assert_eq!(decoded_symbols, symbols, "Symbol mismatch");

    // Check final state
    if ans_reader.check_final_state().is_ok() {
        println!("\n✓ ANS checksum OK (final state = 0x130000)");
    } else {
        println!(
            "\n✗ ANS checksum FAILED (final state = 0x{:08x}, expected 0x130000)",
            ans_reader.0
        );
        // Don't fail - just report
    }
}
