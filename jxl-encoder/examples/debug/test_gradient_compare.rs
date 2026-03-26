fn main() {
    let (w, h) = (64, 64);

    // Horizontal gradient (works)
    eprintln!("\n=== Testing {}x{} HORIZONTAL gradient ===", w, h);
    let mut h_data = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let val = (x * 255 / w.max(1)) as u8;
            let idx = (y * w + x) * 3;
            h_data[idx] = val;
            h_data[idx + 1] = val;
            h_data[idx + 2] = val;
        }
    }

    let h_encoded = jxl_encoder::encoder::encode_lossy_rgb8(&h_data, w, h, 1.0).unwrap();
    eprintln!("  Encoded: {} bytes", h_encoded.len());

    let h_result = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&h_encoded))
        .and_then(|img| img.render_frame(0));
    match h_result {
        Ok(_) => eprintln!("  DECODE: OK"),
        Err(e) => eprintln!("  DECODE ERROR: {:?}", e),
    }

    // Vertical gradient (fails)
    eprintln!("\n=== Testing {}x{} VERTICAL gradient ===", w, h);
    let mut v_data = vec![0u8; w * h * 3];
    for y in 0..h {
        let val = (y * 255 / h.max(1)) as u8;
        for x in 0..w {
            let idx = (y * w + x) * 3;
            v_data[idx] = val;
            v_data[idx + 1] = val;
            v_data[idx + 2] = val;
        }
    }

    let v_encoded = jxl_encoder::encoder::encode_lossy_rgb8(&v_data, w, h, 1.0).unwrap();
    eprintln!("  Encoded: {} bytes", v_encoded.len());

    let v_result = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(&v_encoded))
        .and_then(|img| img.render_frame(0));
    match v_result {
        Ok(_) => eprintln!("  DECODE: OK"),
        Err(e) => eprintln!("  DECODE ERROR: {:?}", e),
    }

    // Save both for comparison
    std::fs::write("/tmp/h_grad_64.jxl", &h_encoded).unwrap();
    std::fs::write("/tmp/v_grad_64.jxl", &v_encoded).unwrap();
    eprintln!("\nSaved to /tmp/h_grad_64.jxl and /tmp/v_grad_64.jxl");
}
