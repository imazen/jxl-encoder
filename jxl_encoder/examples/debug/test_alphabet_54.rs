/// Test alphabet size 54 specifically
use jxl_encoder::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_pattern(size: usize, scale: u8) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            // Create a pattern that should produce different coefficients
            let val = (((x + y) * scale as usize / size) % 256) as u8;
            let idx = (y * size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn try_decode(jxl_data: &[u8]) -> String {
    match jxl_oxide::JxlImage::builder().read(Cursor::new(jxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => "OK".to_string(),
            Err(e) => format!("FAIL: {:?}", e),
        },
        Err(e) => format!("Parse FAIL: {:?}", e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try to find what produces alphabet=54
    eprintln!("=== Finding patterns that produce alphabet=54 ===\n");

    for scale in [64, 128, 192, 255] {
        for size in [16, 24, 32, 33, 34, 40] {
            let data = generate_pattern(size, scale);
            let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
            let result = try_decode(&jxl);
            if result.contains("FAIL") {
                eprintln!("size={}, scale={}: FAIL", size, scale);
            }
        }
    }

    Ok(())
}
