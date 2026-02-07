use jxl_encoder::encoder::encode_lossy_rgb8;
/// Debug single-cluster vs multi-cluster encoding
///
/// This test compares the bitstream structure between images that result in
/// 1 cluster (fails) vs 2 clusters (works) to find the encoding bug.
use std::io::Cursor;

fn generate_gray_image(width: usize, height: usize, pattern: &str) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let val = match pattern {
                "vertical" => (y * 255 / height.max(1)) as u8,
                "horizontal" => (x * 255 / width.max(1)) as u8,
                "uniform" => 128u8,
                "checkerboard" => {
                    if (x / 4 + y / 4) % 2 == 0 {
                        64
                    } else {
                        192
                    }
                }
                _ => 128u8,
            };
            let idx = (y * width + x) * 3;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Testing different patterns to understand cluster count ===\n");

    // Test different patterns at 33x33 (the size that fails for vertical gradients)
    let patterns = ["vertical", "horizontal", "uniform", "checkerboard"];

    for pattern in patterns {
        eprintln!("\n============================================================");
        eprintln!("=== Pattern: {} at 33x33 ===", pattern);
        eprintln!("============================================================\n");

        let data = generate_gray_image(33, 33, pattern);

        // Encode with tracing enabled
        let jxl = encode_lossy_rgb8(&data, 33, 33, 85.0)?;

        eprintln!("\nEncoded size: {} bytes", jxl.len());

        // Try to decode
        match try_decode(&jxl) {
            Ok(()) => eprintln!("DECODE: OK"),
            Err(e) => eprintln!("DECODE: FAIL - {}", e),
        }
    }

    // Also test the boundary sizes to confirm the 1-cluster hypothesis
    eprintln!("\n\n============================================================");
    eprintln!("=== Testing boundary sizes with vertical gradient ===");
    eprintln!("============================================================\n");

    for size in [32, 33, 34, 35] {
        eprintln!("\n--- Size {}x{} ---", size, size);
        let data = generate_gray_image(size, size, "vertical");
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;

        match try_decode(&jxl) {
            Ok(()) => eprintln!("Result: OK ({} bytes)", jxl.len()),
            Err(e) => eprintln!("Result: FAIL - {}", e),
        }
    }

    Ok(())
}
