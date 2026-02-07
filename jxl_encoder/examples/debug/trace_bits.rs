/// Trace bit positions during 8x8 vs 33x33 encoding to find misalignment
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
    // Test sizes near the failure boundary
    for size in [8, 9, 16, 24, 27, 30, 31, 32, 33, 34, 36, 40] {
        let data = generate_vertical(size);
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;
        let blocks = ((size + 7) / 8) * ((size + 7) / 8);
        let result = try_decode(&jxl);
        let status = if result == "OK" { "OK" } else { "FAIL" };
        eprintln!(
            "{}x{}: {} blocks, {} bytes, decode: {}",
            size,
            size,
            blocks,
            jxl.len(),
            status
        );
    }

    Ok(())
}
