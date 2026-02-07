#![allow(unused)]
use jxl_encoder::encoder::Encoder;
use std::io::Cursor;

fn main() {
    // Create 8x8 image with only row 0 bright (255)
    let mut pixels = vec![0u8; 8 * 8 * 3];
    for col in 0..8 {
        let idx = 0 * 8 * 3 + col * 3;  // row 0
        pixels[idx] = 255;     // R
        pixels[idx + 1] = 255; // G
        pixels[idx + 2] = 255; // B
    }

    println!("Input: row 0 is bright (255), all others are 0");

    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");
    
    // Decode with jxl-oxide
    let img = jxl_oxide::JxlImage::builder().read(Cursor::new(&encoded)).unwrap();
    let frame = img.render_frame(0).unwrap();
    let fb = frame.image_all_channels();
    let samples: Vec<f32> = fb.buf().to_vec();
    
    println!("\nDecoded (R channel):");
    for row in 0..8 {
        let mut row_vals = Vec::new();
        for col in 0..8 {
            let idx = row * 8 * 3 + col * 3;
            row_vals.push((samples[idx] * 255.0).round() as i32);
        }
        println!("  row {}: {:?}", row, row_vals);
    }
}
