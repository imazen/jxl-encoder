//! Debug gradient encoding to see AC coefficient handling

use jxl_encoder::encoder::Encoder;
use std::io::Cursor;

fn main() {
    // Create 8x8 vertical gradient
    let mut pixels = Vec::with_capacity(8 * 8 * 3);
    for row in 0..8 {
        let val = (row * 32) as u8;
        for _col in 0..8 {
            pixels.push(val);
            pixels.push(val);
            pixels.push(val);
        }
    }

    println!("Input gradient (first col): {:?}", (0..8).map(|r| pixels[r * 8 * 3]).collect::<Vec<_>>());

    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");

    // Save to file for comparison
    std::fs::write("/tmp/jxl_test/grad8_ours.jxl", &encoded).expect("write failed");

    println!("Encoded {} bytes", encoded.len());
    
    // Decode with jxl-oxide
    match jxl_oxide::JxlImage::builder().read(Cursor::new(&encoded)) {
        Ok(img) => {
            match img.render_frame(0) {
                Ok(frame) => {
                    let fb = frame.image_all_channels();
                    let samples: Vec<f32> = fb.buf().to_vec();
                    
                    // Get first column (should show gradient)
                    println!("\nDecoded (first col, R channel):");
                    for row in 0..8 {
                        let idx = row * 8 * 3;  // RGB
                        let r = (samples[idx] * 255.0) as u8;
                        println!("  row {}: {}", row, r);
                    }
                }
                Err(e) => println!("Render error: {:?}", e),
            }
        }
        Err(e) => println!("Decode error: {:?}", e),
    }
}
