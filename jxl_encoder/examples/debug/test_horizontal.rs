#![allow(unused)]
use jxl_encoder::encoder::Encoder;
use std::io::Cursor;

fn main() {
    // Create 8x8 horizontal gradient (varies with column, constant per row)
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

    println!("Input: horizontal gradient (col 0=0, col 7=224)");
    println!("  Row 0: {:?}", (0..8).map(|c| pixels[c*3]).collect::<Vec<_>>());

    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");
    
    // Decode with jxl-oxide
    let img = jxl_oxide::JxlImage::builder().read(Cursor::new(&encoded)).unwrap();
    let frame = img.render_frame(0).unwrap();
    let fb = frame.image_all_channels();
    let samples: Vec<f32> = fb.buf().to_vec();
    
    println!("\nDecoded (R channel):");
    for row in 0..4 {
        let mut row_vals = Vec::new();
        for col in 0..8 {
            let idx = row * 8 * 3 + col * 3;
            row_vals.push((samples[idx] * 255.0).round() as i32);
        }
        println!("  row {}: {:?}", row, row_vals);
    }
}
