#![allow(unused)]
use jxl_enc::encoder::Encoder;
use jxl_enc::color::xyb::srgb_to_xyb;
use jxl_enc::vardct::quantizer::QuantizerParams;
use jxl_enc::vardct::transform::transform_xyb_image;
use jxl_enc::vardct::tokenize::ZIGZAG_ORDER_8X8;

fn main() {
    // Create 8x8 HORIZONTAL gradient (varies with column)
    let mut pixels = Vec::with_capacity(8 * 8 * 3);
    for row in 0..8 {
        for col in 0..8 {
            let val = (col * 32) as u8;  // HORIZONTAL: varies with col
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
    
    println!("DC Y = {}", transformed.dc_coeffs[1]);
    
    // Y channel AC coefficients
    let y_ac_start = 0 * 63 + 1 * 63; // block 0, channel Y (index 1)
    let y_ac = &transformed.ac_coeffs[y_ac_start..y_ac_start + 63];
    
    println!("\nY channel AC coefficients (raw storage order):");
    for i in 0..63 {
        if y_ac[i] != 0 {
            let orig_idx = i + 1; // position in full 64-element block
            let row = orig_idx / 8;
            let col = orig_idx % 8;
            println!("  ac[{}] = {} (orig_idx={}, row={}, col={})", i, y_ac[i], orig_idx, row, col);
        }
    }
    
    println!("\nCoefficients in ZIGZAG scan order (what encoder emits):");
    for k in 0..20 { // first 20 scan positions
        let zigzag_idx = ZIGZAG_ORDER_8X8[k + 1]; // +1 to skip DC
        let ac_idx = zigzag_idx - 1; // -1 because AC array skips DC
        let coeff = if ac_idx < 63 { y_ac[ac_idx] } else { 0 };
        let row = zigzag_idx / 8;
        let col = zigzag_idx % 8;
        println!("  scan_pos={}: zigzag_idx={} (row={}, col={}) -> coeff={}", 
                 k, zigzag_idx, row, col, coeff);
    }
    
    println!("\nExpected JXL natural order positions:");
    // JXL natural order for 8x8: (x, y) coordinates
    let natural_order = [
        (0,0), (1,0), (0,1), (0,2), (1,1), (2,0), (3,0), (2,1), (1,2), (0,3),
        (4,0), (3,1), (2,2), (1,3), (0,4), (0,5), (1,4), (2,3), (3,2), (4,1)
    ];
    for (k, (dx, dy)) in natural_order.iter().enumerate().skip(1).take(10) {
        let expected_idx = dy * 8 + dx; // (x,y) -> row-major index
        println!("  scan_pos={}: natural (dx={}, dy={}) -> index {} (row={}, col={})", 
                 k-1, dx, dy, expected_idx, dy, dx);
    }
}
