/// Minimal debug test for multi-block VarDCT failure
use jxl_enc::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_gradient(size: usize) -> Vec<u8> {
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

fn try_decode(jxl_data: &[u8]) -> Result<(), String> {
    match jxl_oxide::JxlImage::builder().read(Cursor::new(jxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Render error: {:?}", e)),
        },
        Err(e) => Err(format!("Parse error: {:?}", e)),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Detail on failing case ===");

    // Test the specific failing case
    let base_size = 24;
    let steepness = 3.5f32;
    let mut data = vec![0u8; base_size * base_size * 3];
    for y in 0..base_size {
        let val = ((y as f32 * steepness) as u32).min(255) as u8;
        for x in 0..base_size {
            let idx = (y * base_size + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    let jxl = encode_lossy_rgb8(&data, base_size, base_size, 85.0)?;
    let result = try_decode(&jxl);
    eprintln!("Result: {:?}", result);

    Ok(())
}
