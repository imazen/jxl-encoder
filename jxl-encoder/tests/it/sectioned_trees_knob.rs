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

// ---------------------------------------------------------------------------
// #96 residual scope (2026-08-28): meta-channel + patches content must ENGAGE
// the sectioned mode instead of silently falling back to the whole-image
// global tree. Each fixture pins engagement by `On != Off` (before the fix
// the two were byte-identical on this content — the fallback) and pixel-
// exact roundtrips via zenjxl-decoder AND jxl-rs (the primary decoder).
// ---------------------------------------------------------------------------

/// 512x512 RGB, 17 colour tuples in 8x8 blocks — fires the FULL-image
/// palette transform (palette meta channel in stream 0, index channel
/// split per group). Same generator as the hash-lock fixture.
fn blocky17_rgb_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut seed = 7777u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u8
    };
    let palette: Vec<[u8; 3]> = (0..17).map(|_| [lcg(), lcg(), lcg()]).collect();
    let mut out = vec![0u8; w * h * 3];
    for by in 0..h / 8 {
        for bx in 0..w / 8 {
            let c = palette[(bx * 31 + by * 17 + (bx * by) % 7) % 17];
            for y in (by * 8)..(by * 8 + 8) {
                for x in (bx * 8)..(bx * 8 + 8) {
                    let i = (y * w + x) * 3;
                    out[i..i + 3].copy_from_slice(&c);
                }
            }
        }
    }
    out
}

/// 512x512 RGB where every channel draws from 16 sparse values (density
/// 16/241 well under the 50 % ChannelCompact filter) but the per-pixel
/// tuples are PRNG-mixed, so the tuple count (~4096) exceeds
/// `MAX_PALETTE_COLORS` and the full palette does NOT fire — this is the
/// ChannelCompact (per-channel palette meta channels) route.
fn sparse_channels_rgb_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut out = Vec::with_capacity(w * h * 3);
    let mut state: u32 = 0x2545_f491;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Smooth-ish structure so trees are non-trivial: quantized
            // gradient plus a little PRNG jitter, all on the 16-value lattice.
            let r = ((x / 32) as u32 + (state >> 30)) & 15;
            let g = ((y / 32) as u32 + (state >> 28)) & 15;
            let b = (((x + y) / 64) as u32 + (state >> 26)) & 15;
            out.push((r * 16) as u8);
            out.push((g * 16) as u8);
            out.push((b * 16) as u8);
        }
    }
    out
}

/// 512x512 RGB "glyph page": a FLAT background with eight 8x8 textured
/// glyph templates stamped ~1000 times on a grid, so the lossless patches
/// detector (flat-background flood fill + repeated connected components)
/// finds a dictionary that clears its cost gate. Same shape as the
/// animation-path fixture `synthetic_screenshot_256`, scaled to 4 groups.
fn glyph_page_rgb_512x512() -> Vec<u8> {
    let (w, h) = (512usize, 512usize);
    let mut out = vec![200u8; w * h * 3];
    // Eight deterministic 12x12 templates: each pixel's colour is a hash
    // of (template, x, y), so the interiors are textured (a plain block
    // would compress fine without patches) yet identical per occurrence.
    let template = |t: usize, x: usize, y: usize| -> [u8; 3] {
        let mut v = (t as u32 + 1)
            .wrapping_mul(0x9e37_79b9)
            .wrapping_add((x as u32).wrapping_mul(0x85eb_ca6b))
            .wrapping_add((y as u32).wrapping_mul(0xc2b2_ae35));
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b_3c6d);
        v ^= v >> 12;
        [
            (v & 0x7f) as u8,
            ((v >> 8) & 0x7f) as u8,
            ((v >> 16) & 0x7f) as u8,
        ]
    };
    for row in 0..(h / 16) {
        for col in 0..(w / 16) {
            let t = (row * 5 + col * 3) % 8;
            let ox = col * 16 + 4;
            let oy = row * 16 + 4;
            for gy in 0..8 {
                for gx in 0..8 {
                    let i = ((oy + gy) * w + ox + gx) * 3;
                    out[i..i + 3].copy_from_slice(&template(t, gx, gy));
                }
            }
        }
    }
    out
}

/// Decode via jxl-rs and return tightly-packed RGB8.
fn decode_jxl_rs_rgb8(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let mut decoder_init = JxlDecoder::<states::Initialized>::new(options);
    let mut decoder = loop {
        match decoder_init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_init = fallback;
            }
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder = fallback;
            }
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
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                decoder_frame = fallback;
            }
            Err(e) => panic!("jxl-rs frame decode error: {e:?}"),
        }
    }
    let mut out = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        out.extend_from_slice(&output_image.row(y)[..w * channels]);
    }
    out
}

/// Encodes `pixels` under `Off` and `On`, asserts the sectioned mode
/// ENGAGED (bitstreams differ) and that both round-trip pixel-exact through
/// zenjxl-decoder and jxl-rs. Returns `(off, on)`.
fn assert_sectioned_engages_and_roundtrips(
    pixels: &[u8],
    w: usize,
    h: usize,
    cfg: impl Fn() -> LosslessConfig,
    what: &str,
) -> (Vec<u8>, Vec<u8>) {
    let off = encode(
        pixels,
        w as u32,
        h as u32,
        cfg().with_sectioned_trees(SectionedTrees::Off),
    );
    let on = encode(
        pixels,
        w as u32,
        h as u32,
        cfg().with_sectioned_trees(SectionedTrees::On),
    );
    assert_ne!(
        on, off,
        "{what}: SectionedTrees::On must engage (differ from Off) — a byte-identical \
         pair means the content fell back to the whole-image global tree"
    );
    for (name, bytes) in [("global", &off), ("sectioned", &on)] {
        // Optional dump for out-of-process decoder checks (djxl):
        // `JXL_SECTIONED_DUMP_DIR=dir` writes `<what>_<name>.jxl` + the
        // raw `<what>.rgb` source. Read-only env access; no mutation.
        if let Ok(dir) = std::env::var("JXL_SECTIONED_DUMP_DIR") {
            std::fs::write(format!("{dir}/{what}_{name}.jxl"), bytes).expect("dump jxl");
            std::fs::write(format!("{dir}/{what}.rgb"), pixels).expect("dump rgb");
        }
        assert_eq!(
            decode_rgb(bytes, w, h),
            pixels,
            "{what}: {name} zenjxl-decoder roundtrip"
        );
        assert_eq!(
            decode_jxl_rs_rgb8(bytes, w, h),
            pixels,
            "{what}: {name} jxl-rs roundtrip"
        );
    }
    (off, on)
}

#[test]
fn sectioned_engages_on_full_palette_content() {
    let (w, h) = (512usize, 512usize);
    let pixels = blocky17_rgb_512x512();
    let (off, on) =
        assert_sectioned_engages_and_roundtrips(&pixels, w, h, LosslessConfig::new, "palette");
    // The palette meta channel is coded once in stream 0 either way; the
    // per-group index streams are tiny on 17-colour blocks. Sectioned must
    // stay in the same order of magnitude (no runaway per-group headers).
    assert!(
        on.len() <= off.len() * 2,
        "palette sectioned ({}) vs global ({}) byte blow-up",
        on.len(),
        off.len()
    );
}

#[test]
fn sectioned_engages_on_channel_compact_content() {
    let (w, h) = (512usize, 512usize);
    let pixels = sparse_channels_rgb_512x512();
    assert_sectioned_engages_and_roundtrips(&pixels, w, h, LosslessConfig::new, "compact");
}

#[test]
fn sectioned_engages_with_lossless_patches() {
    let (w, h) = (512usize, 512usize);
    let pixels = glyph_page_rgb_512x512();
    let (_off, on) = assert_sectioned_engages_and_roundtrips(
        &pixels,
        w,
        h,
        || LosslessConfig::new().with_patches(true),
        "patches",
    );
    // The patches dictionary must actually be in play on this fixture
    // (else the sectioned+patches combination is untested): with patches
    // disabled the sectioned bitstream must differ.
    let on_no_patches = encode(
        &pixels,
        w as u32,
        h as u32,
        LosslessConfig::new()
            .with_patches(false)
            .with_sectioned_trees(SectionedTrees::On),
    );
    assert_ne!(
        on, on_no_patches,
        "patches fixture must fire the lossless patches detector (bitstreams equal => vacuous)"
    );
    assert_eq!(
        decode_rgb(&on_no_patches, w, h),
        pixels,
        "no-patches roundtrip"
    );
}
