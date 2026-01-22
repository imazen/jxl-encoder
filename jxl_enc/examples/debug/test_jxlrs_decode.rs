use jxl_enc::encoder::encode_lossy_rgb8;

fn main() {
    let width = 8;
    let height = 8;
    let data = vec![128u8; width * height * 3];

    eprintln!("Input: uniform gray 128");

    let result = encode_lossy_rgb8(&data, width, height, 1.0);
    match result {
        Ok(encoded) => {
            eprintln!("Encoded {} bytes", encoded.len());

            // Save for debugging
            std::fs::write("/tmp/test_gray128_for_jxlrs.jxl", &encoded).unwrap();

            // Try to decode with jxl-rs
            // Note: jxl-rs is the decoder crate, jxl_dec would need to be added as a dependency
            eprintln!("\nDecode with jxl-oxide:");
            match jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .and_then(|img| img.render_frame(0))
            {
                Ok(frame) => {
                    let fb = frame.image_all_channels();
                    let samples: Vec<f32> = fb.buf().to_vec();

                    // Print raw float values
                    eprintln!("First pixel raw values (0-1 range):");
                    eprintln!("  R = {:.6}", samples[0]);
                    eprintln!("  G = {:.6}", samples[1]);
                    eprintln!("  B = {:.6}", samples[2]);

                    // What these float values represent
                    eprintln!("\nAs 8-bit sRGB:");
                    eprintln!("  R = {:.1}", samples[0] * 255.0);
                    eprintln!("  G = {:.1}", samples[1] * 255.0);
                    eprintln!("  B = {:.1}", samples[2] * 255.0);

                    // Average
                    let avg = (samples[0] + samples[1] + samples[2]) / 3.0;
                    eprintln!("\nAverage: {:.6} = {:.1} sRGB", avg, avg * 255.0);
                }
                Err(e) => eprintln!("Decode error: {:?}", e),
            }
        }
        Err(e) => eprintln!("Encode error: {:?}", e),
    }
}
