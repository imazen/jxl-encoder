fn main() {
    // Compare histogram encoding for working and failing cases

    // Working: uniform 40x40
    let data_uniform = vec![128u8; 40 * 40 * 3];
    let result1 = jxl_encoder::encoder::encode_lossy_rgb8(&data_uniform, 40, 40, 1.0);
    match &result1 {
        Ok(enc) => {
            eprintln!("UNIFORM 40x40: {} bytes", enc.len());
            std::fs::write("/tmp/uniform_40.jxl", enc).ok();
        }
        Err(e) => eprintln!("UNIFORM 40x40: ENCODE FAIL - {:?}", e),
    }

    // Failing: v_gradient 40x40
    let mut data_gradient = vec![0u8; 40 * 40 * 3];
    for y in 0..40 {
        let val = (y * 255 / 40) as u8;
        for x in 0..40 {
            let idx = (y * 40 + x) * 3;
            data_gradient[idx] = val;
            data_gradient[idx + 1] = val;
            data_gradient[idx + 2] = val;
        }
    }
    let result2 = jxl_encoder::encoder::encode_lossy_rgb8(&data_gradient, 40, 40, 1.0);
    match &result2 {
        Ok(enc) => {
            eprintln!("GRADIENT 40x40: {} bytes", enc.len());
            std::fs::write("/tmp/gradient_40.jxl", enc).ok();
        }
        Err(e) => eprintln!("GRADIENT 40x40: ENCODE FAIL - {:?}", e),
    }

    // Try to decode both
    if let Ok(enc) = &result1 {
        match jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(enc))
            .and_then(|img| img.render_frame(0))
        {
            Ok(_) => eprintln!("UNIFORM decode: OK"),
            Err(e) => eprintln!("UNIFORM decode: {:?}", e),
        }
    }

    if let Ok(enc) = &result2 {
        match jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(enc))
            .and_then(|img| img.render_frame(0))
        {
            Ok(_) => eprintln!("GRADIENT decode: OK"),
            Err(e) => eprintln!("GRADIENT decode: {:?}", e),
        }
    }
}
