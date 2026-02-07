fn main() {
    // Test uniform 40x40 (5x5 blocks) - same size as failing gradient
    let (w, h) = (40, 40);

    // Uniform gray
    let data_uniform: Vec<u8> = vec![128; w * h * 3];
    let result = jxl_encoder::encoder::encode_lossy_rgb8(&data_uniform, w, h, 1.0);
    match result {
        Ok(encoded) => {
            let decode = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .and_then(|img| img.render_frame(0));
            match decode {
                Ok(_) => eprintln!("UNIFORM 40x40: OK ({} bytes)", encoded.len()),
                Err(e) => eprintln!("UNIFORM 40x40: DECODE FAIL - {:?}", e),
            }
        }
        Err(e) => eprintln!("UNIFORM 40x40: ENCODE FAIL - {:?}", e),
    }

    // Gradient
    let mut data_grad = vec![0u8; w * h * 3];
    for y in 0..h {
        let val = (y * 255 / h.max(1)) as u8;
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data_grad[idx] = val;
            data_grad[idx + 1] = val;
            data_grad[idx + 2] = val;
        }
    }
    let result = jxl_encoder::encoder::encode_lossy_rgb8(&data_grad, w, h, 1.0);
    match result {
        Ok(encoded) => {
            let decode = jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .and_then(|img| img.render_frame(0));
            match decode {
                Ok(_) => eprintln!("GRADIENT 40x40: OK ({} bytes)", encoded.len()),
                Err(e) => eprintln!("GRADIENT 40x40: DECODE FAIL - {:?}", e),
            }
        }
        Err(e) => eprintln!("GRADIENT 40x40: ENCODE FAIL - {:?}", e),
    }
}
