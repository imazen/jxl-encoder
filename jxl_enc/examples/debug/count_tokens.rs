/// Count tokens for working vs failing cases
use jxl_enc::encoder::encode_lossy_rgb8;
use std::io::Cursor;

fn generate_vertical(size: usize) -> Vec<u8> {
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

fn try_decode(jxl_data: &[u8]) -> String {
    match jxl_oxide::JxlImage::builder().read(Cursor::new(jxl_data)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => "OK".to_string(),
            Err(_) => "FAIL".to_string(),
        },
        Err(_) => "Parse FAIL".to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compare a working and failing case with same block count
    for &size in &[31, 27] {
        eprintln!("\n=== Testing {}x{} ===", size, size);
        let data = generate_vertical(size);
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
        let result = try_decode(&jxl);
        eprintln!("Result: {} ({} bytes)", result, jxl.len());
    }

    Ok(())
}
