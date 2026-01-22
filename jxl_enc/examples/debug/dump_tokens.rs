//! Dump token patterns for vertical vs horizontal gradients.

use jxl_enc::color::srgb_to_xyb;
use jxl_enc::vardct::context::{BlockContextMap, NON_ZERO_BUCKETS};
use jxl_enc::vardct::tokenize::tokenize_block;
use jxl_enc_transforms::dct::dct8;

fn generate_vertical_gray(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        let val = (y * 255 / height.max(1)) as u8;
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn generate_horizontal_gray(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let val = (x * 255 / width.max(1)) as u8;
            let idx = (y * width + x) * 3;
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }
    data
}

fn analyze_block(pixels_rgb: &[u8], width: usize, bx: usize, by: usize) {
    println!("\n  Block at ({}, {})", bx, by);

    // Extract 8x8 block and convert to XYB
    let mut block_y = [[0.0f32; 8]; 8];
    for row in 0..8 {
        for col in 0..8 {
            let px = bx * 8 + col;
            let py = by * 8 + row;
            let idx = (py * width + px) * 3;
            let r = pixels_rgb[idx] as f32 / 255.0;
            let g = pixels_rgb[idx + 1] as f32 / 255.0;
            let b = pixels_rgb[idx + 2] as f32 / 255.0;

            // Simplified XYB - just use Y channel
            let (_, y, _) = srgb_to_xyb(r, g, b);
            block_y[row][col] = y;
        }
    }

    // Print Y channel values
    println!("    Y channel (first 4 rows):");
    for row in 0..4 {
        print!("      ");
        for col in 0..8 {
            print!("{:5.2} ", block_y[row][col]);
        }
        println!();
    }

    // Do DCT
    let mut input = [0.0f32; 64];
    for row in 0..8 {
        for col in 0..8 {
            input[row * 8 + col] = block_y[row][col];
        }
    }
    let mut dct_coeffs = [0.0f32; 64];
    dct8(&input, &mut dct_coeffs);

    // Count non-zeros after quantization (simulated)
    let quant_scale = 10.0; // Rough quantization
    let mut quantized = [0i32; 64];
    for i in 0..64 {
        quantized[i] = (dct_coeffs[i] * quant_scale).round() as i32;
    }

    // Print DCT coefficients (first 4 rows)
    println!("    DCT coeffs (first 4 rows, quantized):");
    for row in 0..4 {
        print!("      ");
        for col in 0..8 {
            print!("{:5} ", quantized[row * 8 + col]);
        }
        println!();
    }

    // Count non-zeros (excluding DC at index 0)
    let nonzeros: usize = quantized[1..].iter().filter(|&&c| c != 0).count();
    println!("    Non-zeros (excluding DC): {}", nonzeros);

    // Tokenize
    let bcm = BlockContextMap::new_default();
    let order: Vec<usize> = (0..64).collect();
    let mut tokens = Vec::new();
    tokenize_block(&quantized, &order, 0, &bcm, 0, &mut tokens);

    println!("    Token count: {}", tokens.len());
    if !tokens.is_empty() {
        let nz_token = &tokens[0];
        println!(
            "    First token (non_zeros): ctx={}, value={}",
            nz_token.context, nz_token.value
        );

        // Check if context is in the non_zeros range
        let num_contexts = bcm.num_contexts;
        let nz_ctx_start = 0;
        let nz_ctx_end = num_contexts * NON_ZERO_BUCKETS;
        println!(
            "    Non-zeros context range: 0..{} (num_contexts={})",
            nz_ctx_end, num_contexts
        );
        println!(
            "    This token's context {} is in non_zeros range: {}",
            nz_token.context,
            nz_token.context < nz_ctx_end as u32
        );
    }
}

fn main() {
    let (w, h) = (64, 64);

    println!("=== Analyzing 64x64 Vertical Gradient ===");
    let v_data = generate_vertical_gray(w, h);
    // Analyze first two blocks
    analyze_block(&v_data, w, 0, 0);
    analyze_block(&v_data, w, 1, 0);
    analyze_block(&v_data, w, 0, 1);

    println!("\n=== Analyzing 64x64 Horizontal Gradient ===");
    let h_data = generate_horizontal_gray(w, h);
    analyze_block(&h_data, w, 0, 0);
    analyze_block(&h_data, w, 1, 0);
    analyze_block(&h_data, w, 0, 1);
}
