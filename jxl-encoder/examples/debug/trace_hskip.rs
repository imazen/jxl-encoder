use jxl_encoder::encoder::encode_lossy_rgb8;
/// Trace hskip encoding bit-by-bit to find the mismatch
///
/// For 33x33 vertical gradient:
/// - alphabet_size = 54
/// - d = 6, symbols_short = 10 (depth 5), symbols_long = 44 (depth 6)
/// - pos_short = 5, pos_long = 7 in STORAGE_ORDER
/// - skip = 3 (since min_pos = 5 >= 3)
///
/// But the decoder seems to read hskip = 2
use std::io::Cursor;

fn generate_vertical(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        let val = (y * 255 / size.max(1)) as u8;
        for x in 0..size {
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn try_decode(jxl_data: &[u8]) -> Result<(), String> {
    let result = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl_data))
        .and_then(|img| img.render_frame(0));

    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("{}", e)),
    }
}

/// Dump bits in a specific region of the file
fn dump_region_bits(data: &[u8], start_bit: usize, num_bits: usize, label: &str) {
    eprintln!(
        "\n=== {} (bits {}..{}) ===",
        label,
        start_bit,
        start_bit + num_bits
    );
    for i in 0..num_bits {
        let byte_idx = (start_bit + i) / 8;
        let bit_idx = (start_bit + i) % 8;
        if byte_idx < data.len() {
            let bit = (data[byte_idx] >> bit_idx) & 1;
            if i % 8 == 0 && i > 0 {
                eprint!(" ");
            }
            eprint!("{}", bit);
        }
    }
    eprintln!();

    // Also show byte values
    let start_byte = start_bit / 8;
    let end_byte = (start_bit + num_bits + 7) / 8;
    eprint!("Bytes: ");
    for byte_idx in start_byte..end_byte.min(data.len()) {
        eprint!("{:02x} ", data[byte_idx]);
    }
    eprintln!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Encoding 33x33 vertical gradient ===\n");

    let data = generate_vertical(33);
    let jxl = encode_lossy_rgb8(&data, 33, 33, 85.0)?;

    eprintln!("\n=== File Analysis ===");
    eprintln!("Total size: {} bytes ({} bits)", jxl.len(), jxl.len() * 8);

    // JXL signature is 2 bytes: FF 0A
    // Then comes the frame header + TOC
    // Based on previous analysis, section data starts around byte 13

    // Find the frame header start (after signature)
    let sig_end = 2; // After FF 0A
    eprintln!("\nSignature ends at byte {}", sig_end);

    // Dump first 50 bytes
    eprintln!("\nFirst 50 bytes:");
    for i in 0..50.min(jxl.len()) {
        if i % 16 == 0 {
            eprint!("\n{:4}: ", i);
        }
        eprint!("{:02x} ", jxl[i]);
    }
    eprintln!();

    // Section data starts at byte 13 (after frame header + TOC)
    let section_start_byte = 13;
    let section_start_bit = section_start_byte * 8;

    eprintln!(
        "\nSection data starts at byte {} (bit {})",
        section_start_byte, section_start_bit
    );

    // From trace output, HF Global should have:
    // - LF Global is 24 bits
    // - LF Group is variable (depends on content)
    // - HF Global starts after those

    // For 33x33 vertical, from earlier trace:
    // - LF Global = 24 bits
    // - LF Group = 571 bits
    // - HF Global starts at section bit 595

    // HF Global contains:
    // 1. dequant_all_default (1 bit)
    // 2. used_orders (variable)
    // 3. Histograms
    //    - LZ77 enabled (1 bit) = 0
    //    - context_map (3 bits for single cluster: is_simple=1, bits=0)
    //    - use_prefix_code (1 bit) = 1
    //    - IntegerConfig (9 bits: split=4, msb=2, lsb=0)
    //    - alphabet_size (10 bits for 54)
    //    - prefix_code (hskip + cl-cl + code_lengths)

    // So within HF Global section:
    // - hskip should be at: 1 + 2 + 1 + 3 + 1 + 9 + 10 = 27 bits from HF Global start

    // But we need to know where HF Global starts exactly
    // Let me search for the pattern

    // The HF Global section should start with dequant_all_default (probably 1 = use defaults)
    // Then used_orders

    // Let me try different offsets to find where hskip might be

    eprintln!("\n=== Searching for hskip location ===");

    // Try offsets around byte 85-95 (where bit 622 would be)
    // Section bit 622 = byte 77.75 from section start
    // File byte = 13 + 77 = 90

    let search_byte = 90;
    dump_region_bits(
        &jxl,
        search_byte * 8,
        32,
        &format!("Around byte {}", search_byte),
    );

    // Let's look at what's at different bit offsets from section start
    for offset in [595, 600, 610, 615, 617, 619, 620, 621, 622, 623, 624, 625] {
        let file_bit = section_start_bit + offset;
        let byte_idx = file_bit / 8;
        let bit_idx = file_bit % 8;
        if byte_idx < jxl.len() {
            let byte_val = jxl[byte_idx];
            let two_bits = (byte_val >> bit_idx) & 0x3;
            eprintln!(
                "Section bit {}: file byte {}, bit {} -> byte=0x{:02x}, 2-bit val={}",
                offset, byte_idx, bit_idx, byte_val, two_bits
            );
        }
    }

    // Now try to decode
    eprintln!("\n=== Decode Attempt ===");
    match try_decode(&jxl) {
        Ok(()) => eprintln!("DECODE: OK"),
        Err(e) => eprintln!("DECODE: FAIL - {}", e),
    }

    // Save file for external analysis
    std::fs::write("/tmp/33x33_vertical_trace.jxl", &jxl)?;
    eprintln!("\nSaved to /tmp/33x33_vertical_trace.jxl");

    Ok(())
}
