#![allow(unused)]
use jxl_enc::encoder::Encoder;
use jxl_enc::color::xyb::srgb_to_xyb;
use jxl_enc::vardct::quantizer::QuantizerParams;
use jxl_enc::vardct::transform::transform_xyb_image;
use std::io::Cursor;

fn main() {
    // Create 8x8 image with specific pattern:
    // First row slightly different from rest
    let mut pixels = vec![100u8; 8 * 8 * 3]; // baseline 100
    
    // Make row 0 have value 200 to create vertical frequency
    for col in 0..8 {
        let idx = 0 * 8 * 3 + col * 3;
        pixels[idx] = 200;
        pixels[idx + 1] = 200;
        pixels[idx + 2] = 200;
    }

    println!("Input: row 0 = 200, rows 1-7 = 100");
    println!("This should create vertical frequency energy at v=1, u=0");
    
    // Check what our transform produces
    let mut xyb_data = vec![0.0f32; 8 * 8 * 3];
    for i in 0..64 {
        let (x, y, b) = srgb_to_xyb(pixels[i*3] as f32, pixels[i*3+1] as f32, pixels[i*3+2] as f32);
        xyb_data[i * 3] = x;
        xyb_data[i * 3 + 1] = y;
        xyb_data[i * 3 + 2] = b;
    }
    
    let quantizer = QuantizerParams::from_distance(1.0);
    let transformed = transform_xyb_image(&xyb_data, 8, 8, &quantizer);
    
    println!("\nQuantized Y channel (first 10 non-DC positions):");
    let y_ac = &transformed.ac_coeffs[63..126]; // Y channel for block 0
    for i in 0..10 {
        let orig_idx = i + 1;
        let v = orig_idx / 8;
        let u = orig_idx % 8;
        if y_ac[i] != 0 {
            println!("  ac[{}] = {} (pos {}: v={}, u={})", i, y_ac[i], orig_idx, v, u);
        }
    }

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
        for col in 0..4 {
            let idx = row * 8 * 3 + col * 3;
            row_vals.push((samples[idx] * 255.0).round() as i32);
        }
        println!("  row {}: {:?}", row, row_vals);
    }
    
    println!("\nExpected: row 0 should be brighter (~200), rows 1-7 ~100");
}
