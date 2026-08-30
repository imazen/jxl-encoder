use jxl_encoder::__test_exports::xyb::srgb_to_xyb;
use jxl_encoder::encoder::encode_lossy_rgb8;
use jxl_encoder::vardct::quantizer::QuantizerParams;

fn main() {
    // Test uniform gray 128
    let width = 8;
    let height = 8;
    let data = vec![128u8; width * height * 3];

    eprintln!("Input: uniform gray 128");
    eprintln!("Expected output: gray ~128");

    // Show XYB conversion
    let (x, y, b) = srgb_to_xyb(128.0, 128.0, 128.0);
    eprintln!("\nsRGB 128 -> XYB: X={:.4}, Y={:.4}, B={:.4}", x, y, b);

    // Show quantizer params
    let qp = QuantizerParams::from_distance(1.0);
    eprintln!("\nQuantizerParams for distance=1.0:");
    eprintln!("  global_scale = {}", qp.global_scale);
    eprintln!("  quant_dc = {}", qp.quant_dc);
    eprintln!(
        "  global_scale_float = {}",
        qp.global_scale as f32 / 65536.0
    );

    // Calculate expected DC
    let inv_lf_quant_y = 512.0f32;
    let global_scale_float = qp.global_scale as f32 / 65536.0;
    let qdc_factor = inv_lf_quant_y * global_scale_float * qp.quant_dc as f32;
    eprintln!("\nDC quantization factor (Y): {:.4}", qdc_factor);
    eprintln!(
        "Expected DC (if DCT DC = 8 * Y): {:.4} * {:.4} = {:.1}",
        y,
        qdc_factor,
        y * qdc_factor
    );

    // Show decoder's dequant factor
    let inv_quant_lf = 65536.0 / (qp.global_scale as f32 * qp.quant_dc as f32);
    let fac_y = (1.0 / 512.0) * inv_quant_lf;
    eprintln!("\nDecoder's Y dequant factor: {:.6}", fac_y);
    eprintln!(
        "If DC=415: decoded Y = 415 * {:.6} = {:.4}",
        fac_y,
        415.0 * fac_y
    );

    // What XYB Y corresponds to sRGB 176?
    let (x176, y176, b176) = srgb_to_xyb(176.0, 176.0, 176.0);
    eprintln!(
        "\nsRGB 176 -> XYB: X={:.4}, Y={:.4}, B={:.4}",
        x176, y176, b176
    );
    eprintln!(
        "Ratio of decoded Y: {:.4} / {:.4} = {:.4}",
        y176,
        y,
        y176 / y
    );

    let result = encode_lossy_rgb8(&data, width, height, 1.0);
    match result {
        Ok(encoded) => {
            eprintln!("Encoded {} bytes", encoded.len());

            // Save to file for external analysis
            std::fs::write("/tmp/test_gray128.jxl", &encoded).unwrap();
            eprintln!("Saved to /tmp/test_gray128.jxl");

            // Decode with jxl-oxide
            match jxl_oxide::JxlImage::builder()
                .read(std::io::Cursor::new(&encoded))
                .and_then(|img| img.render_frame(0))
            {
                Ok(frame) => {
                    let fb = frame.image_all_channels();
                    let width = fb.width();
                    let height = fb.height();
                    let samples: Vec<f32> = fb.buf().to_vec();

                    // Print first few pixels
                    eprintln!("\nDecoded pixels (first 8):");
                    for y in 0..1 {
                        for x in 0..8 {
                            let idx = (y * width + x) * 3;
                            let r = samples[idx as usize];
                            let g = samples[idx as usize + 1];
                            let b = samples[idx as usize + 2];
                            // Convert from 0-1 to 0-255
                            eprintln!(
                                "  ({},{}) = R:{:.1} G:{:.1} B:{:.1}",
                                x,
                                y,
                                r * 255.0,
                                g * 255.0,
                                b * 255.0
                            );
                        }
                    }

                    // Calculate average
                    let total: f32 = samples.chunks(3).map(|c| (c[0] + c[1] + c[2]) / 3.0).sum();
                    let avg = total / (width * height) as f32 * 255.0;
                    eprintln!("\nAverage decoded gray: {:.1}", avg);
                    eprintln!(
                        "Error from 128: {:.1} ({:.1}%)",
                        (avg - 128.0).abs(),
                        (avg - 128.0).abs() / 128.0 * 100.0
                    );
                }
                Err(e) => eprintln!("Decode error: {:?}", e),
            }
        }
        Err(e) => eprintln!("Encode error: {:?}", e),
    }
}
