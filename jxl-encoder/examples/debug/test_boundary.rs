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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test boundary between 32x32 (works) and 40x40 (fails)
    for size in [32, 33, 34, 35, 36, 37, 38, 39, 40] {
        eprintln!("\n=== Testing {}x{} ===", size, size);
        let data = generate_vertical_gray(size, size);
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;

        let decode_result = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(&jxl))
            .and_then(|img| img.render_frame(0));

        let status = match &decode_result {
            Ok(_) => "OK".to_string(),
            Err(e) => format!("FAIL: {}", e),
        };

        eprintln!("Result: {}", status);
    }
    Ok(())
}
