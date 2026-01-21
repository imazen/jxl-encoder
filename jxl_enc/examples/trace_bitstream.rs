use jxl_enc::encoder::encode_lossy_rgb8;
/// Trace bitstream encoding bit-by-bit for debugging
///
/// Compares vertical (failing) vs horizontal (working) at 33x33
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

fn generate_horizontal(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let val = (x * 255 / size.max(1)) as u8;
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

fn dump_bytes(data: &[u8], start: usize, count: usize) {
    eprintln!(
        "Bytes {}-{}: {:02x?}",
        start,
        start + count,
        &data[start..start + count.min(data.len() - start)]
    );
}

fn dump_bits_at(data: &[u8], bit_offset: usize, num_bits: usize) {
    let mut bits = Vec::new();
    for i in 0..num_bits {
        let byte_idx = (bit_offset + i) / 8;
        let bit_idx = (bit_offset + i) % 8;
        if byte_idx < data.len() {
            let bit = (data[byte_idx] >> bit_idx) & 1;
            bits.push(if bit == 1 { '1' } else { '0' });
        }
    }
    eprintln!(
        "Bits {}-{}: {}",
        bit_offset,
        bit_offset + num_bits,
        bits.iter().collect::<String>()
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Encoding 33x33 vertical (expected to FAIL) ===\n");
    let v_data = generate_vertical(33);
    let v_jxl = encode_lossy_rgb8(&v_data, 33, 33, 85.0)?;

    eprintln!("\n=== Raw bitstream analysis for vertical ===");
    eprintln!("Total size: {} bytes", v_jxl.len());

    // Dump the header area
    dump_bytes(&v_jxl, 0, 20);

    // Find where sections start (after TOC)
    // TOC should be around byte 10-15 for a small frame
    eprintln!("\nBits around TOC:");
    dump_bits_at(&v_jxl, 80, 32);

    // Section data starts after TOC byte-alignment
    // For single-group, sections are: LF Global, LF Group, HF Global, Pass Group
    // Based on trace: LF Global=24 bits, LF Group=571 bits, HF Global=97 bits
    let toc_end_byte = 13; // Approximate based on trace
    eprintln!("\nSection data (starting around byte {}):", toc_end_byte);
    dump_bytes(&v_jxl, toc_end_byte, 40);

    // HF Global should start after LF Global (24 bits) + LF Group (571 bits) = 595 bits = 74.375 bytes
    let hf_global_bit = 24 + 571; // = 595
    let hf_global_byte = toc_end_byte + hf_global_bit / 8;
    eprintln!(
        "\nHF Global area (around byte {}, bit offset {}):",
        hf_global_byte, hf_global_bit
    );
    dump_bits_at(&v_jxl, toc_end_byte * 8 + hf_global_bit, 100);

    // Pass Group should start after HF Global (97 bits) = 692 bits from section start
    let pass_group_bit = 24 + 571 + 97; // = 692
    eprintln!("\nPass Group area (bit offset {}):", pass_group_bit);
    dump_bits_at(&v_jxl, toc_end_byte * 8 + pass_group_bit, 50);

    eprintln!("\n=== Decoding vertical ===");
    match try_decode(&v_jxl) {
        Ok(()) => eprintln!("DECODE: OK"),
        Err(e) => eprintln!("DECODE: FAIL - {}", e),
    }

    eprintln!("\n\n=== Encoding 33x33 horizontal (expected to work) ===\n");
    let h_data = generate_horizontal(33);
    let h_jxl = encode_lossy_rgb8(&h_data, 33, 33, 85.0)?;

    eprintln!("\n=== Raw bitstream analysis for horizontal ===");
    eprintln!("Total size: {} bytes", h_jxl.len());
    dump_bytes(&h_jxl, 0, 20);

    // For horizontal: LF Global=24 bits, LF Group=472 bits, HF Global=89 bits
    let h_hf_global_bit = 24 + 472; // = 496
    let h_pass_group_bit = 24 + 472 + 89; // = 585
    eprintln!("\nHF Global area (bit offset {}):", h_hf_global_bit);
    dump_bits_at(&h_jxl, toc_end_byte * 8 + h_hf_global_bit, 100);

    eprintln!("\nPass Group area (bit offset {}):", h_pass_group_bit);
    dump_bits_at(&h_jxl, toc_end_byte * 8 + h_pass_group_bit, 50);

    eprintln!("\n=== Decoding horizontal ===");
    match try_decode(&h_jxl) {
        Ok(()) => eprintln!("DECODE: OK"),
        Err(e) => eprintln!("DECODE: FAIL - {}", e),
    }

    Ok(())
}
