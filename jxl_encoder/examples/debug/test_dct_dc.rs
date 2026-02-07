use jxl_enc_transforms::dct8;

fn main() {
    // Test 1: Constant block of 1.0
    let input = [1.0f32; 64];
    let mut output = [0.0f32; 64];
    dct8(&input, &mut output);

    eprintln!("Constant input = 1.0 for all 64 pixels:");
    eprintln!("  DCT DC output[0] = {}", output[0]);
    eprintln!("  Expected DC for orthonormal DCT = 8 * avg = 8.0");
    eprintln!("  Ratio output/8 = {}", output[0] / 8.0);

    // Test 2: Constant block of 0.6034 (XYB Y for sRGB 128)
    let y_value = 0.6034f32;
    let input2 = [y_value; 64];
    let mut output2 = [0.0f32; 64];
    dct8(&input2, &mut output2);

    eprintln!("\nConstant input = {} (XYB Y for gray 128):", y_value);
    eprintln!("  DCT DC output[0] = {}", output2[0]);
    eprintln!("  DC / 8 = {}", output2[0] / 8.0);
    eprintln!("  Expected: DC / 8 should equal input = {}", y_value);

    // Check if AC coefficients are zero (as expected for constant input)
    let max_ac = output2[1..64]
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    eprintln!("\n  Max |AC| for constant input: {} (should be ~0)", max_ac);
}
