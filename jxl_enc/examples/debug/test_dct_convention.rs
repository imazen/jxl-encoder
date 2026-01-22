#![allow(unused)]
use jxl_enc_transforms::dct8;

fn main() {
    // Vertical gradient: input[y][x] = y (varies with y, constant in x)
    let mut input = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            input[y * 8 + x] = y as f32;
        }
    }
    
    let mut output = [0.0f32; 64];
    dct8(&input, &mut output);
    
    println!("Input (vertical gradient): input[y*8+x] = y");
    println!("  First row (y=0): {:?}", &input[0..8]);
    println!("  Second row (y=1): {:?}", &input[8..16]);
    
    println!("\nDCT output (non-zero values):");
    for i in 0..64 {
        if output[i].abs() > 0.01 {
            let row = i / 8;
            let col = i % 8;
            println!("  output[{}] (row={}, col={}) = {:.4}", i, row, col, output[i]);
        }
    }
    
    println!("\nExpected for vertical gradient:");
    println!("  Energy at (v>0, u=0) = column 0 of output matrix");
    println!("  i.e., indices 0, 8, 16, 24, ... (row-major with v*8+u)");
}
