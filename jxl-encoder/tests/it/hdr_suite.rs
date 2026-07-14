// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Comprehensive HDR suite — modeled on libjxl's testing shape:
//!
//! - **Signaling-preservation matrix** (libjxl `AllEncodings()` +
//!   `PreserveOriginalProfileTest`): TF × primaries grid, lossless AND
//!   lossy, decoded header must reproduce the declared encoding.
//! - **Intensity-target rules** (libjxl `luminance.cc:SetIntensityTarget`
//!   and decode_test's `IsPQ ? 10000 : 255` convention): PQ → 10,000,
//!   HLG → 1,000, SDR → 255, explicit values win (issue #73).
//! - **Quality-tracks-distance gates** for PQ and HLG (the #73
//!   acceptance shape — destruction shows up as distance-flat scores).
//! - **Layout equivalence**: PQ-coded f32 layouts vs integer + PQ CE.
//! - **Tag-only lossless**: modular pixels are invariant under the CE.
//!
//! All fixtures are procedural (zero committed bytes) and 512×512 where
//! the path under test is size-dependent (multi-group rule, CLAUDE.md).

use jxl_encoder::headers::ColorEncoding;
use jxl_encoder::headers::color_encoding::{Primaries, TransferFunction};
use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

// ── fixtures ────────────────────────────────────────────────────────────────

/// HDR-photo-shaped content: vertical code-value sweep + sparkle
/// speculars + low-byte noise, as PQ/HLG-coded u16.
fn hdr_ramp_rgb16(w: usize, h: usize, code_max: f32) -> Vec<u8> {
    let mut seed = 4242u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u16
    };
    let top = (code_max * 65535.0) as usize;
    let mut out = Vec::with_capacity(w * h * 6);
    for y in 0..h {
        let base = (top * y / h + 3277) as u16;
        for x in 0..w {
            let sparkle = if (x * 31 + y * 17) % 97 == 0 {
                12000
            } else {
                0
            };
            let r = base.saturating_add(sparkle).saturating_add(lcg() & 0x3f);
            let g = base.saturating_add(lcg() & 0x3f);
            let b = base.saturating_sub(2000).saturating_add(lcg() & 0x3f);
            for v in [r, g, b] {
                out.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    out
}

fn ce(tf: TransferFunction, primaries: Primaries) -> ColorEncoding {
    let mut c = ColorEncoding::srgb();
    c.transfer_function = tf;
    c.primaries = primaries;
    c
}

fn decode_header(bytes: &[u8]) -> jxl_oxide::JxlImage {
    jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(bytes))
        .expect("jxl-oxide parse")
}

// ── 1. signaling-preservation matrix ────────────────────────────────────────

/// libjxl `PreserveOriginalProfileTest` shape: every declared
/// (transfer function × primaries) must come back from the decoded
/// header, for lossless AND lossy. 64×64 keeps the 30-encode grid fast;
/// multi-group signaling is covered by the 512² gates below.
#[test]
fn signaling_preserved_across_tf_primaries_matrix() {
    let (w, h) = (64usize, 64usize);
    let pixels = hdr_ramp_rgb16(w, h, 0.75);
    let tfs = [
        (TransferFunction::Srgb, "Srgb"),
        (TransferFunction::Linear, "Linear"),
        (TransferFunction::Pq, "Pq"),
        (TransferFunction::Hlg, "Hlg"),
        (TransferFunction::Bt709, "Bt709"),
    ];
    let prims = [
        (Primaries::Srgb, "Srgb"),
        (Primaries::P3, "P3"),
        (Primaries::Bt2100, "Bt2100"),
    ];
    for (tf, tf_name) in tfs {
        for (pr, pr_name) in prims {
            let enc = ce(tf, pr);
            let lossless = LosslessConfig::new()
                .with_effort(3)
                .encode_request(w as u32, h as u32, PixelLayout::Rgb16)
                .with_color_encoding(enc.clone())
                .encode(&pixels)
                .unwrap_or_else(|e| panic!("lossless {tf_name}/{pr_name}: {e}"));
            let lossy = LossyConfig::new(1.0)
                .with_effort(3)
                .encode_request(w as u32, h as u32, PixelLayout::Rgb16)
                .with_color_encoding(enc)
                .encode(&pixels)
                .unwrap_or_else(|e| panic!("lossy {tf_name}/{pr_name}: {e}"));
            for (mode, bytes) in [("lossless", &lossless), ("lossy", &lossy)] {
                let img = decode_header(bytes);
                let got = format!("{:?}", img.image_header().metadata.colour_encoding);
                assert!(
                    got.contains(tf_name),
                    "{mode} {tf_name}/{pr_name}: TF lost — header says {got}"
                );
                // jxl-oxide's primaries debug names: Srgb / P3 / Bt2100.
                assert!(
                    got.contains(pr_name),
                    "{mode} {tf_name}/{pr_name}: primaries lost — header says {got}"
                );
            }
        }
    }
}

// ── 2. intensity-target rules (issue #73) ───────────────────────────────────

fn lossy_intensity(
    pixels: &[u8],
    w: u32,
    h: u32,
    enc: Option<ColorEncoding>,
    explicit: Option<f32>,
) -> f32 {
    let cfg = LossyConfig::new(1.0).with_effort(3);
    let mut req = cfg.encode_request(w, h, PixelLayout::Rgb16);
    if let Some(c) = enc {
        req = req.with_color_encoding(c);
    }
    if let Some(it) = explicit {
        req = req.with_intensity_target(it);
    }
    let bytes = req.encode(pixels).expect("encode");
    let img = decode_header(&bytes);
    img.image_header().metadata.tone_mapping.intensity_target
}

/// libjxl `SetIntensityTarget` parity: PQ defaults to 10,000 nits, HLG
/// to 1,000, SDR stays 255 — and an explicit request value beats the
/// dispatch in every case.
#[test]
fn intensity_target_dispatch_rules() {
    let (w, h) = (64u32, 64u32);
    let pixels = hdr_ramp_rgb16(w as usize, h as usize, 0.6);

    let pq = ce(TransferFunction::Pq, Primaries::Srgb);
    let hlg = ce(TransferFunction::Hlg, Primaries::Srgb);

    assert_eq!(
        lossy_intensity(&pixels, w, h, Some(pq.clone()), None),
        10_000.0
    );
    assert_eq!(
        lossy_intensity(&pixels, w, h, Some(hlg.clone()), None),
        1_000.0
    );
    assert_eq!(lossy_intensity(&pixels, w, h, None, None), 255.0);
    assert_eq!(
        lossy_intensity(&pixels, w, h, Some(pq), Some(4_000.0)),
        4_000.0
    );
    assert_eq!(
        lossy_intensity(&pixels, w, h, Some(hlg), Some(600.0)),
        600.0
    );
    assert_eq!(lossy_intensity(&pixels, w, h, None, Some(80.0)), 80.0);
}

// ── 3. HLG quality tracks distance (PQ twin lives in
//      hdr_pq_lossy_roundtrip.rs) ──────────────────────────────────────────

fn hlg_inv_oetf(e: f32) -> f32 {
    // BT.2100 HLG inverse OETF with the libjxl convention: IDENTITY
    // OOTF (`cms/transfer_functions.h TF_HLG_Base::OOTF` — system gamma
    // 1.0 at 334 nits), display == scene, [0,1] with 1.0 = 12x diffuse
    // white. Matches the encoder's `hlg_to_linear_f` exactly.
    const A: f32 = 0.17883277;
    const B: f32 = 0.28466892;
    const C: f32 = 0.55991073;
    let e = e.clamp(0.0, 1.0);
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - C) / A).exp() + B) / 12.0
    }
}

/// HLG twin of the #73 acceptance gate: destruction reads as
/// distance-flat scores; correctness reads as monotone tracking.
#[test]
fn hlg_lossy_quality_tracks_distance_multigroup() {
    use butteraugli::{ButteraugliParams, butteraugli_linear};
    use imgref::Img;
    use rgb::RGB;

    let (w, h) = (512usize, 512usize);
    let pixels = hdr_ramp_rgb16(w, h, 0.9);
    let enc = ce(TransferFunction::Hlg, Primaries::Bt2100);

    let score_at = |d: f32| -> (f64, usize) {
        let bytes = LossyConfig::new(d)
            .with_effort(5)
            .encode_request(w as u32, h as u32, PixelLayout::Rgb16)
            .with_color_encoding(enc.clone())
            .encode(&pixels)
            .expect("encode");
        // No request_color_encoding: jxl-oxide's default output IS the
        // file's own encoding (HLG here), so the floats below are
        // HLG-coded like the source — no CMS re-encode in the loop.
        let image = decode_header(&bytes);
        let render = image.render_frame(0).expect("render");
        let fb = render.image_all_channels();
        let to_img = |vals: &[f32]| -> Img<Vec<RGB<f32>>> {
            let px: Vec<RGB<f32>> = vals
                .chunks_exact(3)
                .map(|c| RGB::new(hlg_inv_oetf(c[0]), hlg_inv_oetf(c[1]), hlg_inv_oetf(c[2])))
                .collect();
            Img::new(px, w, h)
        };
        let dec: Vec<f32> = fb.buf().to_vec();
        let src: Vec<f32> = pixels
            .chunks_exact(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]) as f32 / 65535.0)
            .collect();
        let params = ButteraugliParams::default().with_intensity_target(1_000.0);
        let res = butteraugli_linear(to_img(&src).as_ref(), to_img(&dec).as_ref(), &params)
            .expect("butteraugli");
        (res.score, bytes.len())
    };

    let (s1, b1) = score_at(1.0);
    let (s4, b4) = score_at(4.0);
    eprintln!("HLG diag: d=1 -> {s1:.3} ({b1} B), d=4 -> {s4:.3} ({b4} B)");
    assert!(s1 < 8.0, "HLG d=1 score {s1:.2} — destruction regime");
    assert!(
        s1 < s4,
        "HLG quality must track distance: {s1:.2} !< {s4:.2}"
    );
    assert!(b4 < b1, "HLG bytes must shrink with distance");
}

// ── 4. layout equivalence: PQ f32 vs integer + CE ───────────────────────────

/// `RgbPqF32` (PQ-coded floats) and `Rgb16` + PQ CE carry the same
/// signal; both must land in the same intensity regime (header 10,000)
/// and produce comparable encodes — a scale bug on either path shows up
/// as a wild byte ratio.
#[test]
fn pq_f32_layout_matches_integer_plus_ce() {
    let (w, h) = (256usize, 256usize);
    let u16_px = hdr_ramp_rgb16(w, h, 0.7);
    let f32_px: Vec<u8> = u16_px
        .chunks_exact(2)
        .flat_map(|c| (u16::from_ne_bytes([c[0], c[1]]) as f32 / 65535.0).to_le_bytes())
        .collect();

    let int_bytes = LossyConfig::new(1.0)
        .with_effort(3)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb16)
        .with_color_encoding(ce(TransferFunction::Pq, Primaries::Srgb))
        .encode(&u16_px)
        .expect("int encode");
    let f32_bytes = LossyConfig::new(1.0)
        .with_effort(3)
        .encode_request(w as u32, h as u32, PixelLayout::RgbPqF32)
        .encode(&f32_px)
        .expect("f32 encode");

    for (name, b) in [("int+CE", &int_bytes), ("PqF32", &f32_bytes)] {
        let img = decode_header(b);
        assert_eq!(
            img.image_header().metadata.tone_mapping.intensity_target,
            10_000.0,
            "{name}: PQ intensity default missing"
        );
    }
    let ratio = int_bytes.len() as f64 / f32_bytes.len() as f64;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "PQ via Rgb16+CE ({}) vs RgbPqF32 ({}) diverged: ratio {ratio:.2} — \
         one path is on the wrong intensity scale",
        int_bytes.len(),
        f32_bytes.len(),
    );
}

// ── 5. lossless is tag-only ─────────────────────────────────────────────────

/// Modular (lossless) stores original samples; the CE must change ONLY
/// signaling. Decoded pixels are bit-identical with and without the PQ
/// tag, and exact vs the source.
#[test]
fn lossless_pixels_invariant_under_pq_tag() {
    let (w, h) = (512usize, 512usize); // multi-group
    let pixels = hdr_ramp_rgb16(w, h, 0.8);

    let tagged = LosslessConfig::new()
        .with_effort(5)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb16)
        .with_color_encoding(ce(TransferFunction::Pq, Primaries::Srgb))
        .encode(&pixels)
        .expect("tagged encode");
    let plain = LosslessConfig::new()
        .with_effort(5)
        .encode(&pixels, w as u32, h as u32, PixelLayout::Rgb16)
        .expect("plain encode");

    // jxl-oxide renders through f32 (and a CMS leg for the tagged file),
    // which is not u16-exact — exactness lives in the djxl/jxl-rs CLI
    // verifications. The IN-PROCESS invariance claim: the two files'
    // decoded pixels are IDENTICAL (the CE changed signaling only), and
    // both land within 1 LSB of the source through oxide's f32 path.
    let decode_u16 = |bytes: &[u8]| -> Vec<u16> {
        // No request_color_encoding: jxl-oxide's default output IS the
        // file's own encoding, so lossless modular samples come back
        // unconverted for both arms. (Requesting the file's own PQ
        // encoding explicitly is NOT an identity — it routes through a
        // CMS roundtrip and drifts the samples.)
        let image = decode_header(bytes);
        let render = image.render_frame(0).expect("render");
        render
            .image_all_channels()
            .buf()
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
            .collect()
    };

    let src: Vec<u16> = pixels
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect();
    let dt = decode_u16(&tagged);
    let dp = decode_u16(&plain);
    assert_eq!(
        dt, dp,
        "PQ tag changed decoded pixels — must be signaling-only"
    );
    let max_err = dp
        .iter()
        .zip(&src)
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .max()
        .unwrap();
    assert!(
        max_err <= 1,
        "lossless decode drifted {max_err} LSB from source through oxide"
    );
}

// ── 6. HDR with alpha + grayscale tags ──────────────────────────────────────

#[test]
fn rgba16_pq_lossy_roundtrips_sane() {
    let (w, h) = (256usize, 256usize);
    let rgb = hdr_ramp_rgb16(w, h, 0.6);
    let mut rgba = Vec::with_capacity(w * h * 8);
    for (i, px) in rgb.chunks_exact(6).enumerate() {
        rgba.extend_from_slice(px);
        let a: u16 = if i % 7 == 0 { 0xC000 } else { 0xFFFF };
        rgba.extend_from_slice(&a.to_ne_bytes());
    }
    let bytes = LossyConfig::new(1.0)
        .with_effort(3)
        .encode_request(w as u32, h as u32, PixelLayout::Rgba16)
        .with_color_encoding(ce(TransferFunction::Pq, Primaries::Srgb))
        .encode(&rgba)
        .expect("rgba pq encode");
    let img = decode_header(&bytes);
    assert_eq!(
        img.image_header().metadata.tone_mapping.intensity_target,
        10_000.0
    );
    let got = format!("{:?}", img.image_header().metadata.colour_encoding);
    assert!(got.contains("Pq"), "alpha path lost PQ: {got}");
    let image = img;
    image.render_frame(0).expect("render");
}

#[test]
fn gray16_pq_lossless_tag_preserved() {
    let (w, h) = (512usize, 512usize); // multi-group
    let mut seed = 99u64;
    let mut lcg = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 56) as u16
    };
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = ((i / w * 50) as u16).saturating_add(lcg() & 0xff);
            v.to_ne_bytes()
        })
        .collect();
    let bytes = LosslessConfig::new()
        .with_effort(3)
        .encode_request(w as u32, h as u32, PixelLayout::Gray16)
        .with_color_encoding(ce(TransferFunction::Pq, Primaries::Srgb))
        .encode(&pixels)
        .expect("gray pq encode");
    let img = decode_header(&bytes);
    let got = format!("{:?}", img.image_header().metadata.colour_encoding);
    assert!(got.contains("Pq"), "gray lossless lost PQ tag: {got}");
}
