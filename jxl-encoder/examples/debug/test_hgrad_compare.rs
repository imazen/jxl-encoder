use jxl_encoder::encoder::Encoder;

fn main() {
    let mut pixels = vec![0u8; 8 * 8 * 3];
    for row in 0..8 {
        for col in 0..8 {
            let val = (col * 32) as u8;
            let idx = (row * 8 + col) * 3;
            pixels[idx] = val;
            pixels[idx + 1] = val;
            pixels[idx + 2] = val;
        }
    }
    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");
    std::fs::write("/tmp/jxl_compare/hgrad_ours.jxl", &encoded).expect("write failed");
    println!("Encoded {} bytes to /tmp/jxl_compare/hgrad_ours.jxl", encoded.len());
}
