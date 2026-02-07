//! Analyze what makes vertical gradients fail.
//!
//! In a vertical gradient, each row has the same value. This means:
//! - In each 8x8 block, each row has a constant value
//! - DCT will have large coefficients in the first column (DC and vertical frequencies)
//! - AC coefficients in other columns will be ~0

use jxl_encoder::encoder::encode_lossy_rgb8;
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

// Horizontal gradient - varies along x, constant along y within each column
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

// Mixed gradient - varies in both x and y (like hex_compare example)
fn generate_mixed_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = (x * 255 / width.max(1)) as u8;
            data[idx + 1] = (y * 255 / height.max(1)) as u8;
            data[idx + 2] = 128;
        }
    }
    data
}

fn test_decode(jxl_data: &[u8]) -> Result<(), String> {
    match jxl_oxide::JxlImage::builder()
        .read(Cursor::new(jxl_data))
        .and_then(|img| img.render_frame(0))
    {
        Ok(_) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (64, 64);

    println!("=== Testing 64x64 with different patterns ===\n");

    // Vertical gradient (FAILS)
    print!("Vertical gray gradient: ");
    let data = generate_vertical_gray(w, h);
    let jxl = encode_lossy_rgb8(&data, w, h, 85.0)?;
    match test_decode(&jxl) {
        Ok(_) => println!("{} bytes, OK", jxl.len()),
        Err(e) => println!("{} bytes, FAIL: {}", jxl.len(), e),
    }

    // Horizontal gradient (WORKS)
    print!("Horizontal gray gradient: ");
    let data = generate_horizontal_gray(w, h);
    let jxl = encode_lossy_rgb8(&data, w, h, 85.0)?;
    match test_decode(&jxl) {
        Ok(_) => println!("{} bytes, OK", jxl.len()),
        Err(e) => println!("{} bytes, FAIL: {}", jxl.len(), e),
    }

    // Mixed gradient (WORKS per earlier test)
    print!("Mixed color gradient (R=x, G=y, B=128): ");
    let data = generate_mixed_gradient(w, h);
    let jxl = encode_lossy_rgb8(&data, w, h, 85.0)?;
    match test_decode(&jxl) {
        Ok(_) => println!("{} bytes, OK", jxl.len()),
        Err(e) => println!("{} bytes, FAIL: {}", jxl.len(), e),
    }

    // Try a vertical gradient with slight noise to see if that helps
    print!("Vertical gradient + tiny noise: ");
    let mut data = generate_vertical_gray(w, h);
    // Add tiny variation to break perfect row uniformity
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let noise = ((x + y) % 3) as i8 - 1; // -1, 0, +1
            data[idx] = (data[idx] as i16 + noise as i16).clamp(0, 255) as u8;
            data[idx + 1] = (data[idx + 1] as i16 + noise as i16).clamp(0, 255) as u8;
            data[idx + 2] = (data[idx + 2] as i16 + noise as i16).clamp(0, 255) as u8;
        }
    }
    let jxl = encode_lossy_rgb8(&data, w, h, 85.0)?;
    match test_decode(&jxl) {
        Ok(_) => println!("{} bytes, OK", jxl.len()),
        Err(e) => println!("{} bytes, FAIL: {}", jxl.len(), e),
    }

    // Try uniform color (no gradient at all)
    print!("Uniform gray (no gradient): ");
    let data = vec![128u8; w * h * 3];
    let jxl = encode_lossy_rgb8(&data, w, h, 85.0)?;
    match test_decode(&jxl) {
        Ok(_) => println!("{} bytes, OK", jxl.len()),
        Err(e) => println!("{} bytes, FAIL: {}", jxl.len(), e),
    }

    // Try lower quality for vertical gradient
    print!("Vertical gradient @ quality=50: ");
    let data = generate_vertical_gray(w, h);
    let jxl = encode_lossy_rgb8(&data, w, h, 50.0)?;
    match test_decode(&jxl) {
        Ok(_) => println!("{} bytes, OK", jxl.len()),
        Err(e) => println!("{} bytes, FAIL: {}", jxl.len(), e),
    }

    Ok(())
}
