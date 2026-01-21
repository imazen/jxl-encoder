//! Hex comparison between working (32x35) and failing (32x36) VarDCT outputs.
//!
//! Generates both files and shows byte-by-byte comparison.

use jxl_enc::encoder::encode_lossy_rgb8;
use std::fs;

fn create_gradient_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            pixels[idx] = (x * 255 / width.max(1)) as u8; // R
            pixels[idx + 1] = (y * 255 / height.max(1)) as u8; // G
            pixels[idx + 2] = 128; // B
        }
    }
    pixels
}

fn print_hex_dump(data: &[u8], start: usize, count: usize) {
    for (i, chunk) in data[start..].chunks(16).take(count / 16 + 1).enumerate() {
        print!("{:04x}: ", start + i * 16);
        for (j, &byte) in chunk.iter().enumerate() {
            if j == 8 {
                print!(" ");
            }
            print!("{:02x} ", byte);
        }
        for _ in chunk.len()..16 {
            print!("   ");
        }
        print!(" |");
        for &byte in chunk {
            if byte.is_ascii_graphic() || byte == b' ' {
                print!("{}", byte as char);
            } else {
                print!(".");
            }
        }
        println!("|");
    }
}

fn find_divergence(a: &[u8], b: &[u8]) -> Option<usize> {
    for (i, (&a_byte, &b_byte)) in a.iter().zip(b.iter()).enumerate() {
        if a_byte != b_byte {
            return Some(i);
        }
    }
    if a.len() != b.len() {
        Some(a.len().min(b.len()))
    } else {
        None
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VarDCT Size Testing ===\n");

    // Test a range of sizes to find failure threshold
    let test_sizes = [
        (32, 35),
        (32, 36),
        (32, 40),
        (40, 40),
        (48, 48),
        (64, 64),
        (128, 128),
        (256, 256),
        (257, 257), // Multi-group
        (300, 300),
    ];

    for (width, height) in test_sizes {
        let pixels = create_gradient_image(width, height);
        print!("Testing {}x{}... ", width, height);

        match encode_lossy_rgb8(&pixels, width, height, 85.0) {
            Ok(jxl_data) => {
                print!("encoded {} bytes, ", jxl_data.len());

                // Try to decode
                match jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&jxl_data))
                    .and_then(|img| img.render_frame(0))
                {
                    Ok(_) => println!("decoded OK"),
                    Err(e) => {
                        println!("DECODE FAILED: {}", e);
                        // Save the failing file for analysis
                        let path = format!("/tmp/fail_{}x{}.jxl", width, height);
                        fs::write(&path, &jxl_data)?;
                        println!("  Saved to {}", path);
                    }
                }
            }
            Err(e) => println!("ENCODE FAILED: {}", e),
        }
    }

    // Skip the detailed hex comparison - just focus on finding failures
    let working_pixels = create_gradient_image(32, 35);
    let failing_pixels = create_gradient_image(32, 36);
    let working_jxl = encode_lossy_rgb8(&working_pixels, 32, 35, 85.0)?;
    let failing_jxl = encode_lossy_rgb8(&failing_pixels, 32, 36, 85.0)?;
    fs::write("/tmp/working_32x35.jxl", &working_jxl)?;
    fs::write("/tmp/failing_32x36.jxl", &failing_jxl)?;

    println!("\n=== Working (32x35) - First 256 bytes ===");
    print_hex_dump(&working_jxl, 0, 256.min(working_jxl.len()));

    println!("\n=== Failing (32x36) - First 256 bytes ===");
    print_hex_dump(&failing_jxl, 0, 256.min(failing_jxl.len()));

    // Find first difference
    if let Some(pos) = find_divergence(&working_jxl, &failing_jxl) {
        println!("\n=== First divergence at byte {} (0x{:04x}) ===", pos, pos);
        let start = pos.saturating_sub(32);
        println!("\nWorking around divergence:");
        print_hex_dump(&working_jxl, start, 64);
        println!("\nFailing around divergence:");
        print_hex_dump(&failing_jxl, start, 64);
    } else {
        println!("\nNo divergence found (files identical)");
    }

    // Try to decode with jxl-oxide
    println!("\n=== Decode Test ===");

    print!("Decoding working (32x35) with jxl-oxide... ");
    match jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&working_jxl))
        .and_then(|img| img.render_frame(0))
    {
        Ok(_) => println!("OK"),
        Err(e) => println!("ERROR: {}", e),
    }

    print!("Decoding failing (32x36) with jxl-oxide... ");
    match jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&failing_jxl))
        .and_then(|img| img.render_frame(0))
    {
        Ok(_) => println!("OK"),
        Err(e) => println!("ERROR: {}", e),
    }

    // Also try djxl
    println!("\n=== djxl Test ===");

    let djxl_working = std::process::Command::new("djxl")
        .args(["/tmp/working_32x35.jxl", "/tmp/working_32x35.ppm"])
        .output()?;
    println!(
        "djxl working: {}",
        if djxl_working.status.success() {
            "OK"
        } else {
            "FAILED"
        }
    );
    if !djxl_working.status.success() {
        println!(
            "  stderr: {}",
            String::from_utf8_lossy(&djxl_working.stderr)
        );
    }

    let djxl_failing = std::process::Command::new("djxl")
        .args(["/tmp/failing_32x36.jxl", "/tmp/failing_32x36.ppm"])
        .output()?;
    println!(
        "djxl failing: {}",
        if djxl_failing.status.success() {
            "OK"
        } else {
            "FAILED"
        }
    );
    if !djxl_failing.status.success() {
        println!(
            "  stderr: {}",
            String::from_utf8_lossy(&djxl_failing.stderr)
        );
    }

    Ok(())
}
