//! `JXL_SECTIONED_TREE_PREDICTORS` override coverage (#99 item 1, the
//! content-adaptive per-group predictor selector) plus the NON-VACUITY
//! guard for the `it` binary's adaptive hash-lock cell.
//!
//! Env-mutating — lives in the env-overrides binary; every test takes
//! `env_serial()`.

use crate::env_serial;
use jxl_encoder::api::SectionedTrees;
use jxl_encoder::{LosslessConfig, PixelLayout};

/// MUST stay byte-identical to `hash_lock_features::photoish_rgb_1024x1024`
/// — this test's whole job is to prove that cell exercises the adaptive
/// path, so a drifted copy would prove nothing.
fn photoish_rgb_1024x1024() -> Vec<u8> {
    let (w, h) = (1024usize, 1024usize);
    let hash = |x: i64, y: i64, c: i64| -> f32 {
        let mut v = (x.wrapping_mul(0x27d4_eb2d)
            ^ y.wrapping_mul(0x1656_67b1)
            ^ c.wrapping_mul(0x9e37_79b9)) as u64;
        v ^= v >> 33;
        v = v.wrapping_mul(0xff51_afd7_ed55_8ccd);
        v ^= v >> 29;
        ((v >> 40) as u32 as f32) / 16_777_216.0
    };
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                let mut acc = 0.0f32;
                let mut amp = 0.5f32;
                for oct in 0..6u32 {
                    let step = 1usize << (6 - oct);
                    let (gx, gy) = (x / step, y / step);
                    let (fx, fy) = (
                        (x % step) as f32 / step as f32,
                        (y % step) as f32 / step as f32,
                    );
                    let c0 = c as i64 * 7 + oct as i64 * 131;
                    let v00 = hash(gx as i64, gy as i64, c0);
                    let v10 = hash(gx as i64 + 1, gy as i64, c0);
                    let v01 = hash(gx as i64, gy as i64 + 1, c0);
                    let v11 = hash(gx as i64 + 1, gy as i64 + 1, c0);
                    let top = v00 + (v10 - v00) * fx;
                    let bot = v01 + (v11 - v01) * fx;
                    acc += amp * (top + (bot - top) * fy);
                    amp *= 0.55;
                }
                if x > 700 && y < 300 {
                    acc = acc * 0.35 + 0.6;
                }
                if (x as i64 - y as i64).abs() < 6 {
                    acc = 1.0 - acc;
                }
                // Per-pixel grain: without it the field is too smooth and the
                // probe tree stays shallow (measured: 191 leaves, under the
                // trust floor), so the cell would not cover the adaptive path.
                acc += (hash(x as i64, y as i64, c as i64 * 977 + 31) - 0.5) * 0.08;
                out[(y * w + x) * 3 + c] = (acc.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    out
}

fn encode(px: &[u8], predictors: Option<&str>) -> Vec<u8> {
    // SAFETY: `env_serial()` is held by the caller and this binary is the
    // only env mutator; the encoder reads the hook per per-group learn.
    unsafe {
        match predictors {
            Some(v) => std::env::set_var("JXL_SECTIONED_TREE_PREDICTORS", v),
            None => std::env::remove_var("JXL_SECTIONED_TREE_PREDICTORS"),
        }
    }
    let out = LosslessConfig::new()
        .with_effort(7)
        .with_threads(1)
        .with_sectioned_trees(SectionedTrees::On)
        .encode_request(1024, 1024, PixelLayout::Rgb8)
        .encode(px)
        .expect("sectioned lossless encode");
    // SAFETY: as above.
    unsafe { std::env::remove_var("JXL_SECTIONED_TREE_PREDICTORS") };
    out
}

/// jxl-rs (the PRIMARY decoder) -> tightly packed RGB8.
fn decode_jxl_rs_rgb8(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut decoder_init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_init = fallback,
            Err(e) => panic!("jxl-rs header decode error: {e:?}"),
        }
    };
    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;
    assert_eq!((width, height), (w, h), "jxl-rs dims");
    let num_extras = basic_info.extra_channels.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![None; num_extras],
    });
    let mut decoder_frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info error: {e:?}"),
        }
    };
    let channels = 3usize;
    let mut output_image = Image::<u8>::new((width * channels, height)).expect("alloc");
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        output_image
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    loop {
        match decoder_frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder_frame = fallback,
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }
    let mut out = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        out.extend_from_slice(&output_image.row(y)[..w * channels]);
    }
    out
}

/// Non-vacuity guard for `it`'s
/// `lossless_mg_rgb_1024x1024_photoish_e7_sectioned` lock cell: the
/// fixture must actually TRIP the probe selector, otherwise the lock is
/// silently covering the old fixed-K path instead of the new one.
#[test]
fn photoish_1024_actually_trips_the_probe_selector() {
    let _g = env_serial();
    let px = photoish_rgb_1024x1024();
    let adaptive = encode(&px, None);
    let fixed_k = encode(&px, Some("off"));
    assert_ne!(
        adaptive, fixed_k,
        "the 1024x1024 photoish fixture no longer trips the sectioned probe \
         selector (adaptive output equals the fixed-K root-cost output), so \
         the hash-lock cell that depends on it has stopped covering the \
         adaptive path. Pick a fixture whose probe tree exceeds \
         SECTIONED_PROBE_MIN_LEAVES."
    );
}

/// The override is honoured in BOTH directions and every arm decodes.
#[test]
fn sectioned_tree_predictors_override_round_trips() {
    let _g = env_serial();
    let px = photoish_rgb_1024x1024();
    for arm in [None, Some("off"), Some("auto"), Some("4")] {
        let data = encode(&px, arm);
        let decoded = zenjxl_decoder::decode(&data).expect("zenjxl-decoder decode");
        assert_eq!(
            (decoded.width, decoded.height),
            (1024, 1024),
            "{arm:?} dims"
        );
        assert_eq!(decoded.channels, 4, "{arm:?} channels");
        let rgb: Vec<u8> = decoded
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        assert_eq!(
            rgb, px,
            "{arm:?}: zenjxl-decoder round trip must be pixel-exact"
        );
        // jxl-rs is the PRIMARY decoder for roundtrip validation.
        assert_eq!(
            decode_jxl_rs_rgb8(&data, 1024, 1024),
            px,
            "{arm:?}: jxl-rs round trip must be pixel-exact"
        );
    }
}
