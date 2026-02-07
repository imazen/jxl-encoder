use jxl_enc::color::xyb::srgb_to_xyb;
use jxl_enc::vardct::quant_weights::INV_LF_QUANT;
use jxl_enc::vardct::quantizer::QuantizerParams;
use jxl_enc_transforms::dct8;

fn main() {
    // Gray 128 pixel values
    let (x_xyb, y_xyb, b_xyb) = srgb_to_xyb(128.0, 128.0, 128.0);
    eprintln!(
        "sRGB 128 -> XYB: X={:.6}, Y={:.6}, B={:.6}",
        x_xyb, y_xyb, b_xyb
    );

    // Create constant block
    let y_block = [y_xyb; 64];
    let mut dct_y = [0.0f32; 64];
    dct8(&y_block, &mut dct_y);

    eprintln!("\nDCT of constant Y={:.6} block:", y_xyb);
    eprintln!("  DCT[0] (DC) = {:.6}", dct_y[0]);

    // Our quantization
    let qp = QuantizerParams::from_distance(1.0);
    eprintln!("\nQuantizerParams:");
    eprintln!("  global_scale = {}", qp.global_scale);
    eprintln!("  quant_dc = {}", qp.quant_dc);

    let global_scale_float = qp.global_scale as f32 / 65536.0;
    let inv_lf_quant_y = INV_LF_QUANT[1]; // Y channel

    eprintln!("\nQuantization calculation:");
    eprintln!("  global_scale_float = {:.6}", global_scale_float);
    eprintln!("  INV_LF_QUANT[Y] = {:.1}", inv_lf_quant_y);

    // Our formula: dc_avg = dct[0] / 8, then quantize
    let dc_avg = dct_y[0] / 8.0;
    eprintln!("\n  dc_avg = DCT[0] / 8 = {:.6}", dc_avg);

    let qdc = inv_lf_quant_y * global_scale_float * qp.quant_dc as f32;
    eprintln!("  qdc = inv_lf_quant * global_scale_float * quant_dc");
    eprintln!(
        "      = {:.1} * {:.6} * {} = {:.4}",
        inv_lf_quant_y, global_scale_float, qp.quant_dc, qdc
    );

    let dc_val = qdc * dc_avg;
    eprintln!(
        "  dc_val = qdc * dc_avg = {:.4} * {:.6} = {:.4}",
        qdc, dc_avg, dc_val
    );

    let quantized_dc = dc_val.round() as i32;
    eprintln!("  quantized_dc = round({:.4}) = {}", dc_val, quantized_dc);

    // What the decoder will do
    eprintln!("\n=== Decoder dequantization ===");
    let lf_quant_y = 1.0 / inv_lf_quant_y; // = 1/512
    let inv_quant_lf = 65536.0 / (qp.global_scale as f32 * qp.quant_dc as f32);
    eprintln!("  LF_QUANT[Y] = 1/512 = {:.9}", lf_quant_y);
    eprintln!("  inv_quant_lf = 65536 / (global_scale * quant_dc)");
    eprintln!(
        "               = 65536 / ({} * {}) = {:.6}",
        qp.global_scale, qp.quant_dc, inv_quant_lf
    );

    let fac_y = lf_quant_y * inv_quant_lf;
    eprintln!("  fac_y = LF_QUANT[Y] * inv_quant_lf = {:.9}", fac_y);

    let decoded_y = quantized_dc as f32 * fac_y;
    eprintln!(
        "  decoded_y = {} * {:.9} = {:.6}",
        quantized_dc, fac_y, decoded_y
    );

    // Now what IDCT does
    eprintln!("\n=== IDCT ===");
    eprintln!(
        "  Input: DC={:.6} (LF value), all other coefficients = 0",
        decoded_y
    );
    eprintln!("  Output: Each pixel = DC = {:.6}", decoded_y);

    // XYB to sRGB
    eprintln!("\n=== XYB -> sRGB ===");
    eprintln!("  XYB: X=0, Y={:.6}, B={:.6}", decoded_y, decoded_y);
    // Y = 0.5 * (L + M), X = 0.5 * (L - M) = 0 => L = M
    // So L = M = Y
    let l = decoded_y;
    let m = decoded_y;
    eprintln!("  L = M = Y = {:.6}", l);

    // L^3 - bias = mixed
    let bias = 0.0037930732552754493f32;
    let mixed = l * l * l - bias;
    eprintln!(
        "  mixed = L^3 - bias = {:.6}^3 - {:.10} = {:.6}",
        l, bias, mixed
    );

    // For gray, linear_rgb = mixed (since opsin matrix rows sum to 1)
    let linear_rgb = mixed;
    eprintln!("  linear_rgb = {:.6} (for gray)", linear_rgb);

    // Linear to sRGB
    let srgb_normalized = if linear_rgb <= 0.0031308 {
        linear_rgb * 12.92
    } else {
        1.055 * linear_rgb.powf(1.0 / 2.4) - 0.055
    };
    eprintln!("  sRGB normalized = {:.6}", srgb_normalized);

    let srgb_8bit = (srgb_normalized * 255.0).round();
    eprintln!("  sRGB 8-bit = {:.1}", srgb_8bit);

    eprintln!("\n=== Expected vs Actual ===");
    eprintln!("  Expected: sRGB 128");
    eprintln!("  Calculated: sRGB {:.1}", srgb_8bit);
}
