//! Debug AC coefficient values

use jxl_encoder::__test_exports::xyb::srgb_to_xyb;
use jxl_encoder::vardct::quant_weights::get_dct8_inv_dequant_per_channel;
use std::f32::consts::PI;

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
    
    let dct = dct8x8(&y_block);
    let inv_dequant = get_dct8_inv_dequant_per_channel();
    
    // Quantization parameters
    let global_scale = 8813_u32;
    let raw_quant = 1_i32;
    let qac = (global_scale as f32 / 65536.0) * raw_quant as f32;
    let threshold = 0.58_f32;
    
    println!("qac = {:.6}", qac);
    println!("threshold = {:.2}", threshold);
    println!("\nY channel DCT coefficients and quantization:");
    println!("Pos  DCT_coeff   Weight    val        quantized");
    println!("---- ----------  --------  ---------  ---------");
    
    for v in 0..4 {
        for u in 0..4 {
            let pos = v * 8 + u;
            let coeff = dct[v][u];
            let weight = inv_dequant[1][pos];  // Y channel
            let val = weight * qac * coeff;
            let quantized = if val.abs() >= threshold { val.round() as i32 } else { 0 };
            if pos == 0 || quantized != 0 || pos < 4 {
                println!("({},{}) {:10.4}  {:8.2}  {:9.4}  {:9}", u, v, coeff, weight, val, quantized);
            }
        }
    }
}
