#![allow(unused)]
use jxl_encoder::encoder::Encoder;
use std::io::Cursor;

fn main() {
    // Create simple 8x8 test pattern
    // Row 0 = 255, Row 1-7 = 0
    // This should create strong AC at vertical frequency
    let mut pixels = vec![0u8; 8 * 8 * 3];
    for col in 0..8 {
        let idx = col * 3;
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
    }

    println!("Input: Row 0 = 255, Rows 1-7 = 0");

    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");
    
    // Save for external testing
    std::fs::write("/tmp/jxl_test/row0_bright.jxl", &encoded).expect("write failed");
    println!("Encoded {} bytes, saved to /tmp/jxl_test/row0_bright.jxl", encoded.len());
    
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
    
    println!("\nExpected: Row 0 should be ~255, Rows 1-7 should be ~0");
}
