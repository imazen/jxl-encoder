#![allow(unused)]
use jxl_encoder::encoder::Encoder;
use jxl_encoder::color::xyb::srgb_to_xyb;
use jxl_encoder::vardct::quantizer::QuantizerParams;
use jxl_encoder::vardct::transform::transform_xyb_image;
use jxl_encoder::vardct::tokenize::ZIGZAG_ORDER_8X8;
use jxl_encoder::vardct::enc_coeff::pack_signed;
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
    
    let quantizer = QuantizerParams::from_distance(1.0);
    let transformed = transform_xyb_image(&xyb_data, 8, 8, &quantizer);
    
    // Get Y channel AC coefficients
    let y_ac_start = 0 * 63 + 1 * 63;
    let y_ac = &transformed.ac_coeffs[y_ac_start..y_ac_start + 63];
    
    // Count non-zeros for Y channel
    let nzeros: i32 = y_ac.iter().filter(|&&c| c != 0).count() as i32;
    println!("Y channel non-zeros: {}", nzeros);
    
    println!("\nTokens that would be emitted for Y channel:");
    println!("  Non-zero count token: value = {}", nzeros);
    
    let mut nzeros_left = nzeros;
    let mut prev = if nzeros <= 4 { 1 } else { 0 };
    
    for k in 0..30 {
        if nzeros_left == 0 {
            break;
        }
        
        let idx = ZIGZAG_ORDER_8X8[k + 1];
        let coeff = y_ac[idx - 1]; // -1 because AC starts at 0
        let u_coeff = pack_signed(coeff);
        
        let v = idx / 8;
        let u = idx % 8;
        
        println!("  k={:2}: zigzag_idx={:2} (v={}, u={}) -> coeff={:4} -> packed={:4}", 
                 k, idx, v, u, coeff, u_coeff);
        
        if coeff != 0 {
            prev = 1;
            nzeros_left -= 1;
        } else {
            prev = 0;
        }
    }
    
    println!("\nExpected decoder behavior:");
    println!("  Decoder reads nzeros = {} at first", nzeros);
    println!("  Then reads {} coefficient tokens", nzeros);
    println!("  At k=1 (zigzag position 2 after DC), decoder places coeff at natural_order[2] = (0,1) = index 8");
    println!("  That's the vertical frequency position");
}
