//! Public-knob coverage for the sectioned local-tree lossless mode
//! (imazen/jxl-encoder#96, `LosslessConfig::with_sectioned_trees`):
//!
//! * `On` engages on a multi-group image (bitstream differs from default)
//!   and round-trips pixel-exact via zenjxl-decoder.
//! * The default (`Auto` with the built-in cap) is byte-identical to `Off`
//!   at this size — ordinary encodes are untouched.
//! * `Auto` + a memory limit BELOW the whole-image estimate engages the
//!   sectioned path (same bytes as `On`) instead of failing allocation.
//!
//! No env mutation — this lives in the normal `it` binary.

use jxl_encoder::api::SectionedTrees;
use jxl_encoder::{Limits, LosslessConfig, PixelLayout};

fn prng_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    let mut state: u32 = 0x9e37_79b9;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = (state >> 24) as u8;
            out.push(((x * 255 / w) as u8).wrapping_add(n & 0x1f));
            out.push(((y * 255 / h) as u8) ^ (n & 0x0f));
            out.push((((x + y) * 128 / (w + h)) as u8).wrapping_add(n >> 3));
        }
    }
    out
}

fn encode(pixels: &[u8], w: u32, h: u32, cfg: LosslessConfig) -> Vec<u8> {
    cfg.with_effort(7)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .expect("lossless encode")
}

fn decode_rgb(bytes: &[u8], w: usize, h: usize) -> Vec<u8> {
    let d = zenjxl_decoder::decode(bytes).expect("decode");
    assert_eq!((d.width, d.height, d.channels), (w, h, 4));
    d.data
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect()
}

#[test]
fn sectioned_knob_engages_and_roundtrips_and_auto_is_budget_driven() {
    let (w, h) = (512usize, 512usize);
    let pixels = prng_rgb(w, h);

    let default = encode(&pixels, w as u32, h as u32, LosslessConfig::new());
    let off = encode(
        &pixels,
        w as u32,
        h as u32,
        LosslessConfig::new().with_sectioned_trees(SectionedTrees::Off),
    );
    let on = encode(
        &pixels,
        w as u32,
        h as u32,
        LosslessConfig::new().with_sectioned_trees(SectionedTrees::On),
    );

    // Auto policy (owner-approved 2026-08-19): sectioned at effort <= 7
    // when the encode runs multi-threaded, global otherwise. This binary
    // builds WITHOUT the `parallel` feature in CI (threads signal = 1),
    // so Auto must match Off here; under an ad-hoc `--features parallel`
    // run on a multi-core host it must match On instead.
    if cfg!(feature = "parallel") {
        assert_eq!(
            default, on,
            "Auto at e<=7 with parallel threads must match On (2026-08-19 policy)"
        );
    } else {
        assert_eq!(
            default, off,
            "Auto without parallel threads must match Off at this size"
        );
    }
    assert_ne!(on, off, "On must actually change the bitstream");
    assert_eq!(decode_rgb(&off, w, h), pixels, "global-tree roundtrip");
    assert_eq!(decode_rgb(&on, w, h), pixels, "sectioned roundtrip");

    // Hybrid: per-group best-of-both. Must round-trip pixel-exact and be
    // no larger than either pure mode (ties keep the global stream, so
    // equality with `off` is legal when no group wins locally).
    let hybrid = encode(
        &pixels,
        w as u32,
        h as u32,
        LosslessConfig::new().with_sectioned_trees(SectionedTrees::Hybrid),
    );
    assert!(
        hybrid.len() <= off.len() && hybrid.len() <= on.len(),
        "hybrid ({}) must be <= global ({}) and <= sectioned ({})",
        hybrid.len(),
        off.len(),
        on.len()
    );
    assert_eq!(decode_rgb(&hybrid, w, h), pixels, "hybrid roundtrip");

    // Auto + a cap below the whole-image estimate (512x512 e7 lossless
    // estimates well above 32 MiB) engages the sectioned path instead of
    // failing the encode.
    let limits = Limits::new().with_max_memory_bytes(32 << 20);
    let tight = LosslessConfig::new()
        .with_effort(7)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_limits(&limits)
        .encode(&pixels)
        .expect("budget-capped lossless encode");
    assert_eq!(
        tight, on,
        "Auto under a tight budget must produce the sectioned bitstream"
    );
}

/// The streaming `LosslessEncoder` honours the same knob as the one-shot
/// request. Before 2026-08-27 its `FrameEncoderOptions` left
/// `sectioned_trees` at the `Auto` default regardless of the config, so
/// `with_sectioned_trees(On)` / `(Off)` were silently ignored on that
/// path (and its pre-flight admitted on the whole-image band only).
/// Pinned by byte-equality with the one-shot encode under each explicit
/// mode — the two modes differ from each other (asserted), so whichever
/// one `Auto` would have resolved to, a dropped knob shows up.
#[test]
fn streaming_encoder_honours_sectioned_knob() {
    let (w, h) = (512usize, 512usize);
    let pixels = prng_rgb(w, h);
    let mut by_mode = Vec::new();
    for mode in [SectionedTrees::Off, SectionedTrees::On] {
        let oneshot = encode(
            &pixels,
            w as u32,
            h as u32,
            LosslessConfig::new().with_sectioned_trees(mode),
        );
        let mut enc = LosslessConfig::new()
            .with_effort(7)
            .with_sectioned_trees(mode)
            .encoder(w as u32, h as u32, PixelLayout::Rgb8)
            .expect("streaming encoder");
        enc.push_rows(&pixels, h as u32).expect("push rows");
        let streamed = enc.finish().expect("streaming finish");
        assert_eq!(
            streamed, oneshot,
            "streaming LosslessEncoder must produce the one-shot bitstream under {mode:?}"
        );
        assert_eq!(decode_rgb(&streamed, w, h), pixels, "{mode:?} roundtrip");
        by_mode.push(streamed);
    }
    assert_ne!(
        by_mode[0], by_mode[1],
        "Off and On must differ (else the check is vacuous)"
    );
}
