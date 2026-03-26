//! Debug what DC value the decoder is receiving.

use jxl_encoder::encoder::encode_lossy_rgb8;

fn main() {
    let width = 8;
    let height = 8;

    // All gray 128
    let data = vec![128u8; width * height * 3];

    eprintln!("Input: uniform gray 128");

    let result = encode_lossy_rgb8(&data, width, height, 1.0);
    match result {
        Ok(encoded) => {
            eprintln!("Encoded {} bytes", encoded.len());

            // Dump first 8 bytes for debugging
            eprintln!("\nActual bytes:");
            for (i, &b) in encoded.iter().take(8).enumerate() {
                eprintln!("  byte {}: 0x{:02x} = {:08b}", i, b, b);
            }

            // Save file for comparison
            std::fs::write("/tmp/test_gray128_for_jxlrs.jxl", &encoded).ok();

            // Decode and get raw LF values
            match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(&encoded)) {
                Ok(img) => {
                    eprintln!("Image parsed successfully");

                    // Try to get raw frame data
                    match img.render_frame(0) {
                        Ok(frame) => {
                            let fb = frame.image_all_channels();
                            let samples: Vec<f32> = fb.buf().to_vec();

                            // The output is sRGB [0,1] normalized
                            let r = samples[0];
                            let g = samples[1];
                            let b = samples[2];

                            eprintln!("\nDecoded sRGB (0-1): R={:.6}, G={:.6}, B={:.6}", r, g, b);
                            eprintln!(
                                "Decoded sRGB (0-255): R={:.1}, G={:.1}, B={:.1}",
                                r * 255.0,
                                g * 255.0,
                                b * 255.0
                            );

                            // Work backwards to find what Y value the decoder computed
                            // sRGB -> linear
                            let linear = if r <= 0.04045 {
                                r / 12.92
                            } else {
                                ((r + 0.055) / 1.055).powf(2.4)
                            };

                            // For gray: linear = L^3 - bias where L = M = Y (in XYB)
                            let bias = 0.0037930732552754493f32;
                            let l_cubed = linear + bias;
                            let y_decoded = l_cubed.cbrt();

                            eprintln!("\nReverse-computed values:");
                            eprintln!("  linear = {:.6}", linear);
                            eprintln!("  Y (XYB) = {:.6}", y_decoded);

                            // What DC value produces this Y?
                            // From quantization: DC * fac_y = Y
                            // where fac_y = LF_QUANT[1] * 65536 / (global_scale * quant_lf)
                            //             = (1/512) * 65536 / (8813 * 10)
                            //             = 0.001452
                            let fac_y = (1.0 / 512.0) * 65536.0 / (8813.0 * 10.0);
                            let dc_implied = y_decoded / fac_y;

                            eprintln!("  fac_y = {:.9}", fac_y);
                            eprintln!("  DC (implied from Y) = {:.1}", dc_implied);
                            eprintln!("  DC we encoded = 415");
                            eprintln!("  Ratio = {:.4}", dc_implied / 415.0);

                            // What Y value should DC=415 produce?
                            let y_expected = 415.0 * fac_y;
                            eprintln!("\n  Y expected from DC=415 = {:.6}", y_expected);
                            eprintln!("  Y actual = {:.6}", y_decoded);
                            eprintln!("  Error = {:.4}x", y_decoded / y_expected);
                        }
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                }
                Err(e) => eprintln!("Parse error: {:?}", e),
            }
        }
        Err(e) => eprintln!("Encode error: {:?}", e),
    }
}
