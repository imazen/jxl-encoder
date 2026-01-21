fn main() {
    // Test exact row boundaries
    let test_cases = [
        (32, 32, "4x4 exactly"),
        (32, 33, "4x5 (33 tall)"),
        (32, 40, "4x5 (40 tall)"),
        (32, 41, "4x6 (41 tall)"),
    ];

    for (width, height, desc) in test_cases {
        let blocks_x = (width + 7) / 8;
        let blocks_y = (height + 7) / 8;

        // v_gradient
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
                    Ok(_) => eprintln!(
                        "{}x{} ({}x{} blocks) {}: OK",
                        width, height, blocks_x, blocks_y, desc
                    ),
                    Err(e) => eprintln!(
                        "{}x{} ({}x{} blocks) {}: FAIL - {:?}",
                        width, height, blocks_x, blocks_y, desc, e
                    ),
                }
            }
            Err(e) => eprintln!("{}x{} {}: ENCODE FAIL - {:?}", width, height, desc, e),
        }
    }
}
