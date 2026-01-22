//! Test DCT coefficient ordering
use jxl_enc_transforms::dct8;
use std::f32::consts::PI;

/// Reference DCT using textbook formula
fn dct8_ref(input: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut output = [[0.0f32; 8]; 8];
    for v in 0..8 {  // vertical frequency
        for u in 0..8 {  // horizontal frequency
            let cu = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
            let mut sum = 0.0;
            for y in 0..8 {  // spatial y position
                for x in 0..8 {  // spatial x position
                    let cos_u = ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
                    let cos_v = ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
                    sum += input[y][x] * cos_u * cos_v;
                }
            }
            output[v][u] = 0.25 * cu * cv * sum;
        }
    }
    output
}

fn main() {
    // Create vertical gradient: constant across x, varies with y
    let mut input_2d = [[0.0f32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            input_2d[y][x] = y as f32; // varies with y (row), constant in x (col)
        }
    }
    
    println!("Input (vertical gradient):");
    for y in 0..8 {
        println!("  row {}: {:?}", y, input_2d[y]);
    }
    
    // Reference DCT
    let ref_dct = dct8_ref(&input_2d);
    println!("\nReference DCT (output[v][u]):");
    println!("  DC (v=0,u=0) = {:.4}", ref_dct[0][0]);
    println!("  Vertical freq (v=1,u=0) = {:.4}", ref_dct[1][0]);
    println!("  Horizontal freq (v=0,u=1) = {:.4}", ref_dct[0][1]);
    
    // Our library DCT
    let mut input_1d = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            input_1d[y * 8 + x] = input_2d[y][x];
        }
    }
    let mut output_1d = [0.0f32; 64];
    dct8(&input_1d, &mut output_1d);
    
    println!("\nLibrary DCT (output[index]):");
    println!("  DC (index 0) = {:.4}", output_1d[0]);
    println!("  Index 1 = {:.4}", output_1d[1]);
    println!("  Index 8 = {:.4}", output_1d[8]);
    
    // Check which position has the vertical frequency
    println!("\nNon-zero coefficients:");
    for i in 0..64 {
        if output_1d[i].abs() > 0.01 {
            let row = i / 8;
            let col = i % 8;
            println!("  output[{}] = output[row={}, col={}] = {:.4}", i, row, col, output_1d[i]);
        }
    }
}
