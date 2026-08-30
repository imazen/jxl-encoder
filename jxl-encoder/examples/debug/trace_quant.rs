use jxl_encoder::__test_exports::xyb::srgb_to_xyb;
use std::f32::consts::PI;

// Simple DCT-II implementation for testing
fn dct8x8(input: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut output = [[0.0f32; 8]; 8];
    
    for v in 0..8 {
        for u in 0..8 {
            let cu = if u == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
            let cv = if v == 0 { 1.0 / 2.0_f32.sqrt() } else { 1.0 };
            
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
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
    // Create Y channel values for gradient
    let mut y_block = [[0.0f32; 8]; 8];
    for row in 0..8 {
        let srgb = (row * 32) as f32;
        let (_, y, _) = srgb_to_xyb(srgb, srgb, srgb);
        for col in 0..8 {
            y_block[row][col] = y;
        }
    }
    
    println!("Y channel input values:");
    for row in 0..8 {
        println!("  row {}: {:.4}", row, y_block[row][0]);
    }
    
    // Compute DCT
    let dct = dct8x8(&y_block);
    
    println!("\nDCT coefficients (first column - vertical frequencies):");
    for v in 0..8 {
        println!("  DCT[{},0] = {:.6}", v, dct[v][0]);
    }
    
    println!("\nDC coefficient: {:.6}", dct[0][0]);
    println!("Largest AC: {:.6}", dct[1][0]);
    
    // What would quantization do?
    let global_scale = 8813_u32;
    let quant = 1_i32;
    let qac = (global_scale as f32 / 65536.0) * quant as f32;
    let y_weight_dc = 560.0;
    let y_weight_ac1 = 560.0;  // Approx same for nearby positions
    
    println!("\nQuantization with global_scale={}, quant={}:", global_scale, quant);
    println!("  qac = {:.6}", qac);
    println!("  DC: {:.4} * {:.4} * {:.4} = {:.4}", dct[0][0], y_weight_dc, qac, dct[0][0] * y_weight_dc * qac);
    println!("  AC[1,0]: {:.4} * {:.4} * {:.4} = {:.4}", dct[1][0], y_weight_ac1, qac, dct[1][0] * y_weight_ac1 * qac);
}
