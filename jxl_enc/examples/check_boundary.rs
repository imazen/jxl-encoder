/// Check alphabet sizes around the failure boundary
use jxl_enc::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_gradient(size: usize, steepness: f32) -> Vec<u8> {
    let mut data = vec![0u8; size * size * 3];
    for y in 0..size {
        let val = ((y as f32 * steepness) as u32).min(255) as u8;
        for x in 0..size {
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
    // Test 16x16 with different gradient steepness to hit different alphabet sizes
    let size = 16;

    for steepness in [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0] {
        let data = generate_gradient(size, steepness);
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
        let result = try_decode(&jxl);
        eprintln!(
            "{}x{} steepness={}: {} bytes, decode: {}",
            size,
            size,
            steepness,
            jxl.len(),
            result
        );
    }

    Ok(())
}
