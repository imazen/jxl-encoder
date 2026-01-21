fn main() {
    for size in [16, 32, 33, 48, 64, 80, 96, 128] {
        let width = size;
        let height = size;

        // v_gradient: each row same value (vertical gradient)
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

        let result = jxl_enc::encoder::encode_lossy_rgb8(&data, width, height, 1.0);
        match result {
            Ok(encoded) => {
                let decode_result = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&encoded))
                    .and_then(|img| img.render_frame(0));
                match decode_result {
                    Ok(_) => {
                        eprintln!("v_gradient {}x{}: OK ({} bytes)", size, size, encoded.len())
                    }
                    Err(e) => eprintln!("v_gradient {}x{}: DECODE FAIL - {:?}", size, size, e),
                }
            }
            Err(e) => eprintln!("v_gradient {}x{}: ENCODE FAIL - {:?}", size, size, e),
        }
    }
}
