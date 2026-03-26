//! Parse JXL file headers to compare encoding differences.

fn main() {
    parse_file("/tmp/test_gray128_for_jxlrs.jxl", "OURS");
    eprintln!("\n{}", "=".repeat(60));
    parse_file("/tmp/gray128_libjxl.jxl", "LIBJXL");
}

fn parse_file(path: &str, name: &str) {
    let data = std::fs::read(path).expect("read file");
    eprintln!("\n=== {} ({} bytes) ===", name, data.len());

    // JXL bitstream is LSB first within each byte
    let mut bit_pos = 0usize;

    // Read bits helper (LSB first)
    let read_bits = |data: &[u8], pos: &mut usize, n: usize| -> u64 {
        let mut result = 0u64;
        for i in 0..n {
            let byte_idx = (*pos + i) / 8;
            let bit_idx = (*pos + i) % 8;
            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                result |= (bit as u64) << i;
            }
        }
        *pos += n;
        result
    };

    // Signature (16 bits)
    let sig1 = read_bits(&data, &mut bit_pos, 8);
    let sig2 = read_bits(&data, &mut bit_pos, 8);
    eprintln!(
        "Signature: 0x{:02x} 0x{:02x} (expected: 0xff 0x0a)",
        sig1, sig2
    );

    // Size header
    let small = read_bits(&data, &mut bit_pos, 1);
    eprintln!("small = {}", small);

    if small == 1 {
        // For small images: ysize_div8_minus_1 using u2S encoding
        let h_div8 = read_bits(&data, &mut bit_pos, 1) == 1;
        eprintln!("h_div8 = {}", h_div8);

        let ysize_val = if h_div8 {
            // ysize_div8_minus_1 = value, ysize = 8 * (value + 1)
            let ysize_div8_minus_1 = read_bits(&data, &mut bit_pos, 5) as usize;
            8 * (ysize_div8_minus_1 + 1)
        } else {
            read_bits(&data, &mut bit_pos, 9) as usize + 1
        };
        eprintln!("ysize = {}", ysize_val);

        let ratio = read_bits(&data, &mut bit_pos, 3);
        eprintln!("ratio = {}", ratio);

        let xsize_val = if ratio == 0 {
            let w_div8 = read_bits(&data, &mut bit_pos, 1) == 1;
            eprintln!("w_div8 = {}", w_div8);
            if w_div8 {
                let xsize_div8_minus_1 = read_bits(&data, &mut bit_pos, 5) as usize;
                8 * (xsize_div8_minus_1 + 1)
            } else {
                read_bits(&data, &mut bit_pos, 9) as usize + 1
            }
        } else {
            // Computed from ratio
            match ratio {
                1 => ysize_val,           // 1:1
                2 => ysize_val * 12 / 10, // 6:5
                3 => ysize_val * 4 / 3,   // 4:3
                4 => ysize_val * 3 / 2,   // 3:2
                5 => ysize_val * 16 / 9,  // 16:9
                6 => ysize_val * 5 / 4,   // 5:4
                _ => ysize_val * 2,       // 2:1
            }
        };
        eprintln!("xsize = {}", xsize_val);
    } else {
        eprintln!("(large image header - not parsing)");
    }

    eprintln!("After size header: bit_pos = {}", bit_pos);

    // ImageMetadata
    let all_default = read_bits(&data, &mut bit_pos, 1);
    eprintln!("metadata.all_default = {}", all_default);

    if all_default == 0 {
        let extra_fields = read_bits(&data, &mut bit_pos, 1);
        eprintln!("metadata.extra_fields = {}", extra_fields);

        if extra_fields == 0 {
            // bit_depth
            let bd_all_default = read_bits(&data, &mut bit_pos, 1);
            if bd_all_default == 1 {
                eprintln!("bit_depth: all_default");
            } else {
                let float_sample = read_bits(&data, &mut bit_pos, 1);
                eprintln!("bit_depth.float_sample = {}", float_sample);
                if float_sample == 0 {
                    let bits_per_sample_sel = read_bits(&data, &mut bit_pos, 2);
                    let bits_per_sample = match bits_per_sample_sel {
                        0 => 8,
                        1 => 10,
                        2 => 12,
                        _ => read_bits(&data, &mut bit_pos, 6) as usize + 1,
                    };
                    eprintln!(
                        "bit_depth.bits_per_sample = {} (sel={})",
                        bits_per_sample, bits_per_sample_sel
                    );
                    // TODO: exp_bits if floating point
                }
            }
        }

        // modular_16_bit_buffer_sufficient
        let mod16 = read_bits(&data, &mut bit_pos, 1);
        eprintln!("modular_16_bit_buffer_sufficient = {}", mod16);

        // num_extra_channels (u2S)
        let nec_sel = read_bits(&data, &mut bit_pos, 2);
        let num_extra = match nec_sel {
            0 => 0,
            1 => read_bits(&data, &mut bit_pos, 4) as usize + 1,
            2 => read_bits(&data, &mut bit_pos, 8) as usize + 17,
            _ => read_bits(&data, &mut bit_pos, 12) as usize + 273,
        };
        eprintln!("num_extra_channels = {} (sel={})", num_extra, nec_sel);

        // xyb_encoded
        let xyb = read_bits(&data, &mut bit_pos, 1);
        eprintln!("xyb_encoded = {}", xyb);

        // color_encoding
        let ce_all_default = read_bits(&data, &mut bit_pos, 1);
        eprintln!("color_encoding.all_default = {}", ce_all_default);
    }

    eprintln!("After metadata: bit_pos = {}", bit_pos);

    // Preview (if extra_fields and has preview)
    // For now, skip

    // Animation (if extra_fields and has animation)
    // For now, skip

    // transform_data
    let td_all_default = read_bits(&data, &mut bit_pos, 1);
    eprintln!("transform_data.all_default = {}", td_all_default);

    eprintln!("After transform_data: bit_pos = {}", bit_pos);

    // Now we're at the frame header (if single frame)
    // Pad to byte boundary for TOC
    eprintln!("\nCurrent bit position: {}", bit_pos);
    eprintln!("Next bytes at bit {}: ", bit_pos);
    let byte_start = bit_pos / 8;
    for i in 0..10.min(data.len() - byte_start) {
        eprintln!(
            "  [{}] 0x{:02x} = {:08b}",
            byte_start + i,
            data[byte_start + i],
            data[byte_start + i]
        );
    }
}
