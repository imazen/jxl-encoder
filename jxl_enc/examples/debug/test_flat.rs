#![allow(unused)]
use jxl_enc::encoder::Encoder;
use std::io::Cursor;

fn main() {
    // Create 8x8 flat gray image (all pixels = 128)
    let pixels = vec![128u8; 8 * 8 * 3];

    println!("Input: flat gray (128)");

    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");
    
    // Decode with jxl-oxide
    let img = jxl_oxide::JxlImage::builder().read(Cursor::new(&encoded)).unwrap();
    let frame = img.render_frame(0).unwrap();
    let fb = frame.image_all_channels();
    let samples: Vec<f32> = fb.buf().to_vec();
    
    println!("\nDecoded (R channel, first 4x4):");
    for row in 0..4 {
        let mut row_vals = Vec::new();
        for col in 0..4 {
            let idx = row * 8 * 3 + col * 3;
            row_vals.push((samples[idx] * 255.0).round() as i32);
        }
        println!("  row {}: {:?}", row, row_vals);
    }
    
    // Check if it's close to 128
    let first_pixel = (samples[0] * 255.0).round() as i32;
    println!("\nFirst pixel R value: {} (expected ~128)", first_pixel);
}
