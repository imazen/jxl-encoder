fn main() {
    // Test vertical gradient at various sizes
    let sizes = [
        (32, 32),
        (33, 33),
        (34, 34),
        (35, 35),
        (36, 36),
        (37, 37),
        (38, 38),
        (39, 39),
        (40, 40),
    ];

    for (w, h) in sizes {
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
        let v_result = jxl_oxide::JxlImage::builder()
            .read(std::io::Cursor::new(&v_encoded))
            .and_then(|img| img.render_frame(0));

        match v_result {
            Ok(_) => eprintln!("{:>3}x{:<3} vertical: OK ({} bytes)", w, h, v_encoded.len()),
            Err(e) => eprintln!("{:>3}x{:<3} vertical: FAIL - {:?}", w, h, e),
        }
    }
}
