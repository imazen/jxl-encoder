/// Compare 33x33 vs 34x34 encoding details
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
            Err(e) => format!("FAIL: {:?}", e),
        },
        Err(e) => format!("Parse FAIL: {:?}", e),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compare 33x33 and 34x34
    eprintln!("=== Testing specific sizes around 33 ===\n");

    for size in 31..=36 {
        let data = generate_vertical(size);
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
        let blocks = (size + 7) / 8;
        eprintln!(
            "{}x{}: {} blocks, {} bytes, decode: {}",
            size,
            size,
            blocks * blocks,
            jxl.len(),
            try_decode(&jxl)
        );
    }

    Ok(())
}
