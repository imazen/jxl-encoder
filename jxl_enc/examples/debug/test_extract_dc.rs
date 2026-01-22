//! Try to understand what DC value is being decoded.
//!
//! We'll use a simpler approach: calculate what DC value would produce
//! the observed output, and compare with what we encoded.

fn main() {
    // Known values from our encoder:
    let y_dc_encoded = 415; // From our trace: y_dc=[415]
    let global_scale = 8813u32;
    let quant_dc = 10u32;

    // Dequantization formula (from decoder):
    // LF_QUANT[Y] = 1/512
    // inv_quant_lf = 65536 / (global_scale * quant_lf)
    // fac_y = LF_QUANT[Y] * inv_quant_lf
    // decoded_y = quantized_dc * fac_y

    let lf_quant_y = 1.0f32 / 512.0;
    let inv_quant_lf = 65536.0 / (global_scale as f32 * quant_dc as f32);
    let fac_y = lf_quant_y * inv_quant_lf;
    let decoded_y_expected = y_dc_encoded as f32 * fac_y;

    eprintln!("=== Expected decoding (based on our DC=415) ===");
    eprintln!("inv_quant_lf = {:.6}", inv_quant_lf);
    eprintln!("fac_y = {:.9}", fac_y);
    eprintln!(
        "decoded_y = {} * {:.9} = {:.6}",
        y_dc_encoded, fac_y, decoded_y_expected
    );

    // What XYB Y value would produce sRGB 175.8?
    // sRGB 175.8/255 = 0.6894
    // linear = (0.6894 + 0.055)^2.4 / 1.055^2.4 = ???
    // Actually, let me compute: what linear value gives sRGB 0.6894?
    let srgb_observed = 0.6894f32; // 175.8 / 255

    // sRGB to linear
    let linear_observed = if srgb_observed <= 0.04045 {
        srgb_observed / 12.92
    } else {
        ((srgb_observed + 0.055) / 1.055).powf(2.4)
    };

    eprintln!("\n=== Observed output ===");
    eprintln!(
        "sRGB normalized: {:.4} (= {:.1} / 255)",
        srgb_observed,
        srgb_observed * 255.0
    );
    eprintln!("linear: {:.6}", linear_observed);

    // For gray: linear = L^3 - bias where L = M = Y
    // So: L^3 = linear + bias
    // L = cbrt(linear + bias)
    let bias = 0.0037930732552754493f32;
    let l_cubed = linear_observed + bias;
    let l_observed = l_cubed.cbrt();

    eprintln!(
        "L^3 = linear + bias = {:.6} + {:.10} = {:.6}",
        linear_observed, bias, l_cubed
    );
    eprintln!("L = Y = {:.6}", l_observed);

    // What DC value would produce this Y?
    let dc_that_produces_observed = l_observed / fac_y;

    eprintln!("\n=== Comparison ===");
    eprintln!(
        "Y expected from DC={}: {:.6}",
        y_dc_encoded, decoded_y_expected
    );
    eprintln!("Y observed: {:.6}", l_observed);
    eprintln!("Ratio: {:.4}", l_observed / decoded_y_expected);
    eprintln!(
        "\nDC that would produce observed Y: {:.1}",
        dc_that_produces_observed
    );
    eprintln!("DC we encoded: {}", y_dc_encoded);
    eprintln!(
        "Ratio: {:.4}",
        dc_that_produces_observed / y_dc_encoded as f32
    );

    // Is there a simple factor?
    let ratio = l_observed / decoded_y_expected;
    eprintln!("\n=== Possible factors ===");
    eprintln!("sqrt(2) = {:.4}", 2.0f32.sqrt());
    eprintln!("sqrt(1.5) = {:.4}", 1.5f32.sqrt());
    eprintln!("8/sqrt(64) = 1.0");
    eprintln!("ratio^2 = {:.4}", ratio * ratio);
    eprintln!("1/ratio = {:.4}", 1.0 / ratio);
    eprintln!("1/ratio^2 = {:.4}", 1.0 / (ratio * ratio));
}
