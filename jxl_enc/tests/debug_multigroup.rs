use jxl_enc::heuristics::HeuristicLevel;
use jxl_enc::vardct::VarDctOptions;
use jxl_enc::vardct::encoder::VarDctEncoder;
use jxl_enc::vardct::transform::transform_xyb_image;

#[test]
#[ignore]
fn compare_single_vs_multi_tokens() {
    // Create identical gradient data for both sizes
    // Just use raw floats [0,1] as "XYB" - structure doesn't matter for comparison
    let create_gradient = |width: usize, height: usize| -> Vec<f32> {
        let mut xyb = vec![0.0f32; width * height * 3];
        // X plane (channel 0)
        for y in 0..height {
            for x in 0..width {
                let val = x as f32 / width as f32 * 0.5;
                xyb[y * width + x] = val;
            }
        }
        // Y plane (channel 1 - main luma)
        for y in 0..height {
            for x in 0..width {
                let val = x as f32 / width as f32;
                xyb[width * height + y * width + x] = val;
            }
        }
        // B plane (channel 2)
        for y in 0..height {
            for x in 0..width {
                let val = x as f32 / width as f32 * 0.3;
                xyb[width * height * 2 + y * width + x] = val;
            }
        }
        xyb
    };

    // 256x256 - single group
    let xyb_256 = create_gradient(256, 256);
    let options = VarDctOptions {
        distance: 1.0,
        ac_strategy_heuristics: HeuristicLevel::Dct8Only,
        ..Default::default()
    };
    let encoder_256 = VarDctEncoder::new(256, 256, options.clone());
    let quant_field_256 = encoder_256.quant_field();
    let transformed_256 =
        transform_xyb_image(&xyb_256, 256, 256, encoder_256.quantizer(), quant_field_256);

    // Tokenize for 256x256
    let (tokens_256, _) = encoder_256
        .tokenize_ac_coefficients(&transformed_256.ac_coeffs)
        .expect("tokenize failed");

    eprintln!("256x256: {} total tokens", tokens_256.len());
    eprintln!("256x256 first 20 tokens:");
    for (i, t) in tokens_256.iter().take(20).enumerate() {
        eprintln!("  [{:2}] ctx={:4} val={:4}", i, t.context, t.value);
    }

    // 257x257 - multi group
    let xyb_257 = create_gradient(257, 257);
    let encoder_257 = VarDctEncoder::new(257, 257, options);
    let quant_field_257 = encoder_257.quant_field();
    let transformed_257 =
        transform_xyb_image(&xyb_257, 257, 257, encoder_257.quantizer(), quant_field_257);

    // Tokenize first group
    let group0_tokens =
        encoder_257.tokenize_ac_coefficients_for_group(&transformed_257.ac_coeffs, 0);

    eprintln!("\n257x257 group 0: {} tokens", group0_tokens.len());
    eprintln!("257x257 group 0 first 20 tokens:");
    for (i, t) in group0_tokens.iter().take(20).enumerate() {
        eprintln!("  [{:2}] ctx={:4} val={:4}", i, t.context, t.value);
    }

    // Compare first 20 tokens
    eprintln!("\nComparison (first 20 tokens):");
    let mut mismatches = 0;
    for i in 0..20.min(tokens_256.len()).min(group0_tokens.len()) {
        let t256 = &tokens_256[i];
        let t257 = &group0_tokens[i];
        let ctx_match = t256.context == t257.context;
        let val_match = t256.value == t257.value;
        let match_str = if ctx_match && val_match {
            "MATCH"
        } else if !ctx_match && !val_match {
            "DIFF BOTH"
        } else if !ctx_match {
            "DIFF CTX"
        } else {
            "DIFF VAL"
        };
        if !ctx_match || !val_match {
            mismatches += 1;
        }
        eprintln!(
            "  [{:2}] 256: ctx={:4} val={:4} | 257: ctx={:4} val={:4} | {}",
            i, t256.context, t256.value, t257.context, t257.value, match_str
        );
    }

    eprintln!("\n{} mismatches in first 20 tokens", mismatches);
}
