fn main() {
    // Test uniform gray images (minimal AC coefficients)
    let test_cases = [
        (32, 32, "4x4"),
        (32, 40, "4x5"),
        (40, 40, "5x5"),
        (64, 64, "8x8"),
    ];

    for (width, height, desc) in test_cases {
        // Uniform gray
        let data = vec![128u8; width * height * 3];

        let result = jxl_enc::encoder::encode_lossy_rgb8(&data, width, height, 1.0);
        match result {
            Ok(encoded) => {
                let decode_result = jxl_oxide::JxlImage::builder()
                    .read(std::io::Cursor::new(&encoded))
                    .and_then(|img| img.render_frame(0));
                match decode_result {
                    Ok(_) => eprintln!(
                        "UNIFORM {}x{} ({}): OK ({} bytes)",
                        width,
                        height,
                        desc,
                        encoded.len()
                    ),
                    Err(e) => eprintln!("UNIFORM {}x{} ({}): FAIL - {:?}", width, height, desc, e),
                }
            }
            Err(e) => eprintln!(
                "UNIFORM {}x{} ({}): ENCODE FAIL - {:?}",
                width, height, desc, e
            ),
        }
    }
}
