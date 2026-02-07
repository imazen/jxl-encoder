//! Test vertical gradient encoding to isolate the failure.

use jxl_encoder::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_horizontal_gradient(width: usize, height: usize) -> Vec<u8> {
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

fn generate_vertical_gradient(width: usize, height: usize) -> Vec<u8> {
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
    let test_sizes = [(32, 32), (64, 64), (100, 100), (33, 47)];

    println!("=== Horizontal Gradient ===");
    for (w, h) in test_sizes {
        let data = generate_horizontal_gradient(w, h);
        print!("  {}x{}: ", w, h);
        match encode_lossy_rgb8(&data, w, h, 85.0) {
            Ok(jxl) => {
                print!("{} bytes, ", jxl.len());
                match test_decode(&jxl) {
                    Ok(_) => println!("OK"),
                    Err(e) => println!("DECODE FAIL: {}", e),
                }
            }
            Err(e) => println!("ENCODE FAIL: {}", e),
        }
    }

    println!("\n=== Vertical Gradient ===");
    for (w, h) in test_sizes {
        let data = generate_vertical_gradient(w, h);
        print!("  {}x{}: ", w, h);
        match encode_lossy_rgb8(&data, w, h, 85.0) {
            Ok(jxl) => {
                print!("{} bytes, ", jxl.len());
                // Save the failing one
                if w == 64 && h == 64 {
                    std::fs::write("/tmp/v_grad_64x64.jxl", &jxl)?;
                    println!("(saved to /tmp/v_grad_64x64.jxl)");
                }
                match test_decode(&jxl) {
                    Ok(_) => println!("OK"),
                    Err(e) => {
                        println!("DECODE FAIL: {}", e);
                        // Save the failing file for analysis
                        let path = format!("/tmp/v_grad_fail_{}x{}.jxl", w, h);
                        std::fs::write(&path, &jxl)?;
                        println!("    Saved to {}", path);
                    }
                }
            }
            Err(e) => println!("ENCODE FAIL: {}", e),
        }
    }

    Ok(())
}
