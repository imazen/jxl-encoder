//! Debug AC coefficient encoding
use jxl_encoder::encoder::Encoder;
use jxl_encoder::color::xyb::srgb_to_xyb;
use jxl_encoder::vardct::quantizer::QuantizerParams;
use jxl_encoder::vardct::transform::transform_xyb_image;
use jxl_oxide::JxlImage;
use std::io::Cursor;

fn main() {
    // Create 8x8 vertical gradient (same as debug_gradient)
    let mut pixels = Vec::with_capacity(8 * 8 * 3);
    for row in 0..8 {
        let val = (row * 32) as u8;
        for _col in 0..8 {
            pixels.push(val);
            pixels.push(val);
            pixels.push(val);
        }
    }

    // Convert to XYB
    let mut xyb_data = vec![0.0f32; 8 * 8 * 3];
    for i in 0..64 {
        let r = pixels[i * 3] as f32;
        let g = pixels[i * 3 + 1] as f32;
        let b = pixels[i * 3 + 2] as f32;
        let (x, y, bb) = srgb_to_xyb(r, g, b);
        xyb_data[i * 3] = x;
        xyb_data[i * 3 + 1] = y;
        xyb_data[i * 3 + 2] = bb;
    }
    
    // Get quantizer params
    let quantizer = QuantizerParams::from_distance(1.0);
    
    // Transform and quantize
    let transformed = transform_xyb_image(&xyb_data, 8, 8, &quantizer);
    
    println!("DC coefficients: X={}, Y={}, B={}", 
        transformed.dc_coeffs[0], transformed.dc_coeffs[1], transformed.dc_coeffs[2]);
    
    // Show non-zero AC coefficients
    let mut nonzero_count = 0;
    let mut max_ac = 0i32;
    for (i, &coeff) in transformed.ac_coeffs.iter().enumerate() {
        if coeff != 0 {
            nonzero_count += 1;
            if coeff.abs() > max_ac {
                max_ac = coeff.abs();
            }
            if nonzero_count <= 20 {
                let block = i / 189;
                let in_block = i % 189;
                let channel = in_block / 63;
                let pos = in_block % 63;
                let ch_name = ["X", "Y", "B"][channel];
                println!("  AC[block={}, ch={}, pos={}] = {}", block, ch_name, pos, coeff);
            }
        }
    }
    println!("Total non-zero AC: {}, max |AC|: {}", nonzero_count, max_ac);
    
    // Now encode and decode
    let encoder = Encoder::new();
    let encoded = encoder.encode_lossy_rgb8(&pixels, 8, 8, 1.0).expect("encode failed");
    
    println!("\nEncoded {} bytes", encoded.len());
    
    match JxlImage::builder().read(Cursor::new(&encoded)) {
        Ok(img) => {
            match img.render_frame(0) {
                Ok(render) => {
                    let fb = render.image_all_channels();
                    let buf = fb.buf();
                    println!("\nDecoded first column:");
                    for row in 0..8 {
                        let idx = row * 8 * 3;
                        let r = (buf[idx] * 255.0).round() as i32;
                        println!("  row {}: {} (expected {})", row, r, row * 32);
                    }
                }
                Err(e) => println!("Render error: {:?}", e),
            }
        }
        Err(e) => println!("Parse error: {:?}", e),
    }
}
