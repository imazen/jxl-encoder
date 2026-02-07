//! Test DCT behavior to understand scaling.

use jxl_enc_transforms::dct8;

fn main() {
    // Test 1: What does our DCT produce for constant input?
    let constant_value = 0.6034f32;
    let input = [constant_value; 64];
    let mut dct_output = [0.0f32; 64];
    dct8(&input, &mut dct_output);

    eprintln!("=== Our DCT (forward) ===");
    eprintln!("Input: constant {}", constant_value);
    eprintln!("DCT DC output[0] = {}", dct_output[0]);
    eprintln!(
        "DC / 8 = {} (should equal input for orthonormal)",
        dct_output[0] / 8.0
    );
    eprintln!("DC / input = {}", dct_output[0] / constant_value);

    // Verify: For orthonormal DCT, DC = N * average = 8 * average
    eprintln!(
        "\nExpected: DC = 8 * {} = {}",
        constant_value,
        8.0 * constant_value
    );
    eprintln!("Actual: DC = {}", dct_output[0]);
    eprintln!(
        "Match: {}",
        (dct_output[0] - 8.0 * constant_value).abs() < 0.001
    );

    // The key insight: We divide by 8 to get the average for the LF image
    eprintln!("\n=== Quantization path ===");
    let dc_avg = dct_output[0] / 8.0;
    eprintln!("dc_avg = DCT[0] / 8 = {}", dc_avg);

    // Our quantization: qdc = INV_LF_QUANT[Y] * global_scale_float * quant_dc
    let global_scale_float = 8813.0 / 65536.0;
    let quant_dc = 10.0;
    let inv_lf_quant_y = 512.0;
    let qdc = inv_lf_quant_y * global_scale_float * quant_dc;
    eprintln!(
        "qdc = {} * {} * {} = {}",
        inv_lf_quant_y, global_scale_float, quant_dc, qdc
    );

    let quantized = (qdc * dc_avg).round() as i32;
    eprintln!("quantized DC = round({} * {}) = {}", qdc, dc_avg, quantized);

    // Decoder dequantization
    eprintln!("\n=== Decoder dequantization ===");
    let lf_quant_y = 1.0 / 512.0;
    let inv_quant_lf = 65536.0 / (8813.0 * 10.0);
    let fac_y = lf_quant_y * inv_quant_lf;
    eprintln!("fac_y = {} * {} = {}", lf_quant_y, inv_quant_lf, fac_y);

    let decoded_y = quantized as f32 * fac_y;
    eprintln!("decoded_y = {} * {} = {}", quantized, fac_y, decoded_y);
    eprintln!("Original Y = {}", constant_value);
    eprintln!("Match: {}", (decoded_y - constant_value).abs() < 0.01);

    // So the math is correct! The issue must be in how the modular stream is written/read.
    eprintln!("\n=== Summary ===");
    eprintln!("The quantization/dequantization math is correct.");
    eprintln!("DC 415 should decode to Y = {:.6}", decoded_y);
    eprintln!(
        "But actual output is Y = 0.758665 (ratio = {:.4})",
        0.758665 / decoded_y
    );
    eprintln!("");
    eprintln!("The 1.2587x factor must come from somewhere in the bitstream encoding/decoding.");
}
