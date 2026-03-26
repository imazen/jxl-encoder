fn main() {
    use jxl_encoder::{LossyConfig, PixelLayout};

    // Generate horizontal gradient exactly like the test
    let width = 300usize;
    let height = 300usize;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let val = (x * 255 / width) as u8;
            let idx = (y * width + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }

    println!("Original first row (u8): {:?}", &data[0..15]);

    // Encode
    let encoded = match LossyConfig::new(1.0)
        .encode_request(width as u32, height as u32, PixelLayout::Rgb8)
        .encode(&data)
    {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Encode failed: {:?}", e);
            return;
        }
    };
    println!("Encoded {} bytes", encoded.len());
    std::fs::write("/tmp/test_300_debug.jxl", &encoded).ok();
    println!("Saved to /tmp/test_300_debug.jxl");
}
