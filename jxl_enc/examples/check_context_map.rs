//! Check context map values for the failing case

use std::io::Cursor;

use jxl_enc::encoder::encode_lossy_rgb8;

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
    let (w, h) = (64, 64);

    eprintln!("=== Encoding 64x64 Vertical Gradient ===");
    let v_data = generate_vertical_gray(w, h);
    let v_jxl = encode_lossy_rgb8(&v_data, w, h, 85.0)?;

    eprintln!("\n=== File size: {} bytes ===", v_jxl.len());

    // Try to decode and see the error
    let decode = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(&v_jxl))
        .and_then(|img| img.render_frame(0));

    match &decode {
        Ok(_) => eprintln!("Decode: OK"),
        Err(e) => eprintln!("Decode: FAIL - {}", e),
    }

    Ok(())
}
