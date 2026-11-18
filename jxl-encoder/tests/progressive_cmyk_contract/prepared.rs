use super::*;
use jxl_encoder::{
    __pre_quantized::{DistanceParams, EncoderPrecomputed, VarDctEncoder},
    api::ExtraChannel,
};

#[test]
fn prepared_black_engines_preserve_progressive_color_and_black() {
    let out = directory();
    let profile: Vec<u8> = include_str!("../fixtures/ps_cmyk.hex")
        .split_whitespace()
        .flat_map(|line| {
            (0..line.len())
                .step_by(2)
                .map(move |i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
        })
        .collect();
    let physical: Vec<[f32; 3]> = include_str!("../fixtures/cmyk_lcms_swatches.hex")
        .lines()
        .map(|line| {
            line.split_whitespace()
                .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect();
    let oracle = std::env::var_os("CMYK_RAW_FRAME_ORACLE").expect("pinned native observer");
    std::fs::write(out.join("profile.icc"), &profile).unwrap();
    for (policy, strategy, mode) in encoder_policies::policies() {
        for (w, h) in [(32_usize, 17_usize), (259, 17)] {
            let linear: Vec<f32> = (0..w * h)
                .flat_map(|i| physical[swatch(i % w, i / w, 0, false)])
                .collect();
            std::fs::write(
                out.join(format!("{policy}-{w}x{h}.f32le")),
                linear
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            for wide in [false, true] {
                let black: Vec<u8> = (0..w * h)
                    .map(|i| SWATCHES[swatch(i % w, i / w, 0, false)][3])
                    .collect();
                let black16: Vec<u16> = black.iter().map(|v| u16::from(*v) * 257).collect();
                let extras = [if wide {
                    ExtraChannel::black_u16(&black16)
                } else {
                    ExtraChannel::black(&black)
                }];
                let mut enc = VarDctEncoder::new(0.01);
                enc.profile = LossyConfig::new(0.01)
                    .with_effort(3)
                    .with_strategy(strategy.clone())
                    .with_mode(mode)
                    .resolved_profile();
                enc.encoder_mode = mode;
                enc.effort = 3;
                enc.force_strategy = Some(3);
                enc.enable_patches = false;
                enc.enable_gaborish = false;
                enc.epf_level_override = Some(0);
                enc.icc_profile = Some(profile.clone());
                let pre = EncoderPrecomputed::compute(
                    w,
                    h,
                    &linear,
                    enc.distance,
                    enc.cfl_enabled,
                    enc.ac_strategy_enabled,
                    enc.pixel_domain_loss,
                    enc.enable_noise,
                    enc.enable_denoise,
                    enc.enable_gaborish,
                    enc.force_strategy,
                    &enc.profile,
                    None,
                )
                .unwrap();
                let mut qf = vec![1; pre.xsize_blocks * pre.ysize_blocks];
                let params = DistanceParams::compute_for_profile(enc.distance, &enc.profile);
                let transform = enc
                    .transform_and_quantize_for_test(&pre, &mut qf, &params)
                    .unwrap();
                for entry in ["precomputed", "prequantized"] {
                    let mut baseline = None;
                    for progressive in [
                        ProgressiveMode::Single,
                        ProgressiveMode::QuantizedAcFullAc,
                        ProgressiveMode::DcVlfLfAc,
                    ] {
                        enc.progressive = progressive;
                        let raw = if entry == "precomputed" {
                            enc.encode_from_precomputed_with_extras(&pre, &qf, &extras)
                        } else {
                            enc.encode_from_pre_quantized_ac_with_extras(
                                &pre,
                                &qf,
                                &transform.quant_dc,
                                &transform.quant_ac,
                                &transform.nzeros,
                                &transform.raw_nzeros,
                                &extras,
                            )
                        }
                        .unwrap();
                        let mut bytes = jxl_encoder::wrap_in_container(&raw, None, None);
                        // Raw engine methods leave the Level 10 wrapper to their caller.
                        bytes.splice(32..32, b"\0\0\0\x09jxll\x0a".iter().copied());
                        schedule(&bytes, 1, progressive);
                        let actual = render(
                            &out.join(format!(
                                "{policy}-{w}x{h}-wide{wide}-{entry}-{progressive:?}"
                            )),
                            &bytes,
                            (w, h),
                            1,
                            (wide, false),
                            &oracle,
                            &physical,
                        );
                        if let Some(expected) = &baseline {
                            same_final(&actual, expected, "prepared progressive final bits");
                        } else {
                            baseline = Some(actual);
                        }
                    }
                }
            }
        }
    }
}
