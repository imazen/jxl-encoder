fn main() {
    // Test different block configurations to find the failure boundary
    let test_cases = [
        (32, 32, "4x4 blocks"),
        (40, 32, "5x4 blocks - 5 cols, 4 rows"),
        (32, 40, "4x5 blocks - 4 cols, 5 rows"),
        (40, 40, "5x5 blocks"),
        (33, 33, "5x5 blocks (33px)"),
        (24, 24, "3x3 blocks"),
        (25, 24, "4x3 blocks edge"),
    ];

    for (width, height, desc) in test_cases {
        // v_gradient: each row same value
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
                        "{}x{} {}: OK ({} bytes)",
                        width,
                        height,
                        desc,
                        encoded.len()
                    ),
                    Err(e) => eprintln!("{}x{} {}: DECODE FAIL - {:?}", width, height, desc, e),
                }
            }
            Err(e) => eprintln!("{}x{} {}: ENCODE FAIL - {:?}", width, height, desc, e),
        }
    }
}
