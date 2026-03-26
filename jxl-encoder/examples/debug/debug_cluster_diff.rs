//! Debug the difference between 5-cluster and 6-cluster encoding.
//!
//! Vertical 64x64 gradient: 5 clusters, alphabet_size=50 - FAILS
//! Horizontal 64x64 gradient: 6 clusters, alphabet_size=42 - WORKS
//!
//! This test investigates why.

use std::io::Cursor;

fn generate_vertical_gray(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        let val = (y * 255 / height.max(1)) as u8;
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn generate_horizontal_gray(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let val = (x * 255 / width.max(1)) as u8;
            let idx = (y * width + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use jxl_encoder::encoder::encode_lossy_rgb8;

    eprintln!("=== Comparing 5-cluster vs 6-cluster encoding ===\n");

    // Generate both images
    let v_data = generate_vertical_gray(64, 64);
    let h_data = generate_horizontal_gray(64, 64);

    eprintln!("--- Encoding Vertical 64x64 (should produce 5 clusters) ---");
    let v_jxl = encode_lossy_rgb8(&v_data, 64, 64, 85.0)?;
    eprintln!("\nVertical: {} bytes", v_jxl.len());

    eprintln!("\n--- Encoding Horizontal 64x64 (should produce 6 clusters) ---");
    let h_jxl = encode_lossy_rgb8(&h_data, 64, 64, 85.0)?;
    eprintln!("\nHorizontal: {} bytes", h_jxl.len());

    // Try to decode both
    eprintln!("\n=== Decoding ===");

    let v_decode = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&v_jxl))
        .and_then(|img| img.render_frame(0));
    match &v_decode {
        Ok(_) => eprintln!("Vertical: DECODE OK"),
        Err(e) => eprintln!("Vertical: DECODE FAIL - {}", e),
    }

    let h_decode = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&h_jxl))
        .and_then(|img| img.render_frame(0));
    match &h_decode {
        Ok(_) => eprintln!("Horizontal: DECODE OK"),
        Err(e) => eprintln!("Horizontal: DECODE FAIL - {}", e),
    }

    // Compare bytes at specific offsets
    eprintln!("\n=== Byte comparison ===");

    // Find where they differ
    let min_len = v_jxl.len().min(h_jxl.len());
    let mut first_diff = None;
    for i in 0..min_len {
        if v_jxl[i] != h_jxl[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(diff_pos) = first_diff {
        eprintln!("First difference at byte {}", diff_pos);
        eprintln!(
            "  Vertical bytes: {:02x?}",
            &v_jxl[diff_pos.saturating_sub(4)..=(diff_pos + 8).min(v_jxl.len() - 1)]
        );
        eprintln!(
            "  Horizontal bytes: {:02x?}",
            &h_jxl[diff_pos.saturating_sub(4)..=(diff_pos + 8).min(h_jxl.len() - 1)]
        );
    } else {
        eprintln!("Files are identical up to length {}", min_len);
    }

    // Dump hex of both files around the HF Global section
    // JXL header is about 40-80 bytes, LF Global around 20-50 bits, LF Group variable
    eprintln!("\n=== Hex dump (first 150 bytes) ===");
    eprintln!("Vertical:");
    for chunk in v_jxl.iter().take(150).collect::<Vec<_>>().chunks(16) {
        eprintln!("  {:02x?}", chunk);
    }
    eprintln!("\nHorizontal:");
    for chunk in h_jxl.iter().take(150).collect::<Vec<_>>().chunks(16) {
        eprintln!("  {:02x?}", chunk);
    }

    Ok(())
}
