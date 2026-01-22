#![allow(unused)]
use jxl_enc::encoder::Encoder;
use jxl_enc::color::xyb::srgb_to_xyb;
use jxl_enc::vardct::quantizer::QuantizerParams;
use jxl_enc::vardct::transform::transform_xyb_image;
use jxl_enc::vardct::tokenize::ZIGZAG_ORDER_8X8;
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

    println!("Analyzing coefficient mapping:");
    
    // ZIGZAG_ORDER_8X8 maps scan position -> coefficient index
    // Index i means row i/8, col i%8 in our row-major storage
    
    println!("\nZIGZAG scan order interpretation:");
    for k in 0..10 {
        let idx = ZIGZAG_ORDER_8X8[k];
        let row = idx / 8;
        let col = idx % 8;
        println!("  scan[{}] -> idx {} (row={}, col={}) = (v={}, u={})", k, idx, row, col, row, col);
    }
    
    println!("\nJXL natural order interpretation (from jxl-oxide):");
    println!("  scan[0] -> (x=0, y=0) placed at grid[0*8+0]=0");
    println!("  scan[1] -> (x=1, y=0) placed at grid[0*8+1]=1");
    println!("  scan[2] -> (x=0, y=1) placed at grid[1*8+0]=8");
    println!("  ...");
    
    println!("\nKey insight: JXL uses (x, y) = (col, row) in grid.get_mut(x, y)");
    println!("  grid.get_mut(x, y) accesses index y*width + x");
    println!("  So (x=1, y=0) -> index 0*8+1 = 1");
    println!("  And (x=0, y=1) -> index 1*8+0 = 8");
    
    println!("\nBut natural_order produces coordinates for (freq_x, freq_y)");
    println!("  freq_x = horizontal frequency = u");
    println!("  freq_y = vertical frequency = v");
    println!("  So scan[1] = (1, 0) means (u=1, v=0) = horizontal freq");
    println!("  And scan[2] = (0, 1) means (u=0, v=1) = vertical freq");
    
    println!("\nIn our DCT output (row-major, index = v*8+u):");
    println!("  index 1 = (v=0, u=1) = horizontal freq");
    println!("  index 8 = (v=1, u=0) = vertical freq");
    
    println!("\nThe question: when decoder reads scan[2] and places at grid (0,1):");
    println!("  grid[1*8+0] = 8, which is (v=1, u=0) in frequency domain");
    println!("  We're emitting our index 8 at scan[2]");
    println!("  Our index 8 = (v=1, u=0) in frequency domain");
    println!("  This MATCHES! Both are vertical frequency.");
    
    println!("\nSo WHY is there a transpose?");
    println!("  Maybe the issue is in how the INVERSE DCT interprets grid positions?");
    println!("  Or maybe we have a bug elsewhere in the encoding...");
}
