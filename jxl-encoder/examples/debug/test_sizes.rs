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
    for size in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80] {
        let data = generate_vertical_gray(size, size);
        let jxl = encode_lossy_rgb8(&data, size, size, 85.0)?;

        let decode_result = jxl_oxide::JxlImage::builder()
            .read(Cursor::new(&jxl))
            .and_then(|img| img.render_frame(0));

        let status = match &decode_result {
            Ok(_) => "OK".to_string(),
            Err(e) => {
                let err = format!("{}", e);
                if err.contains("non_zeros too large") {
                    "FAIL: non_zeros".to_string()
                } else {
                    format!("FAIL: {}", &err[..err.len().min(40)])
                }
            }
        };

        println!("{}x{} ({} bytes): {}", size, size, jxl.len(), status);
    }
    Ok(())
}
