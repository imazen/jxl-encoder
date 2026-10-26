// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Auxiliary frames: the LF frame (`with_lf_frame`) and the patch reference
//! frame.
//!
//! Small images: with an LF frame the main VarDCT frame's LfGlobal tree only
//! covers AC metadata, and below ~320 AC-metadata samples that tree is a
//! single leaf, so its entropy code has one context. A one-context code has no
//! context map in the bitstream, but the writer emitted a 3-bit simple map, so
//! every LF-frame encode up to 128x64 was undecodable.
//!
//! Images with alpha:
//! `with_lf_frame(true)` writes the DC as a separate LF frame, and patches
//! write a reference-only frame holding the patch dictionary. Every frame must
//! carry as many extra channels as the codestream metadata declares, and each
//! patch position must carry a blend mode per extra channel. Both frames used
//! to omit the image's alpha channel, so every such file was rejected by
//! jxl-rs and djxl v0.12. They now carry zero-filled placeholder extra
//! channels, as libjxl does.
//!
//! Each cell decodes fully through jxl-rs and djxl v0.12; for alpha cells
//! djxl's RGBA output must reproduce every alpha sample (alpha is coded
//! losslessly here).

use jxl_encoder::{LosslessConfig, LossyConfig, PixelLayout};

/// Real graphics content at any size: the frymire fixture, mirror-tiled,
/// with a deterministic alpha ramp.
fn frymire_rgba(width: u32, height: u32) -> Vec<u8> {
    let source = image::load_from_memory(include_bytes!("../images/frymire-srgb.png"))
        .expect("decode real screenshot fixture")
        .to_rgb8();
    let (sw, sh) = source.dimensions();
    let fold = |v: u32, n: u32| {
        let period = 2 * n;
        let m = v % period;
        if m < n { m } else { period - 1 - m }
    };
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let p = source.get_pixel(fold(x, sw), fold(y, sh)).0;
            out.extend_from_slice(&[p[0], p[1], p[2], ((x * 3 + y * 5) % 256) as u8]);
        }
    }
    out
}

/// The gb82 terminal screenshot (reliably triggers patches), cropped, with
/// a blocky alpha pattern.
fn terminal_rgba() -> (Vec<u8>, u32, u32) {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
    let dir = corpus.get("gb82-sc").expect("gb82-sc corpus");
    let img = image::open(dir.join("terminal.png"))
        .expect("terminal.png")
        .to_rgb8();
    let (w, h) = (img.width().min(600), img.height().min(400));
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            let a = if (x / 16 + y / 16) % 3 == 0 { 128 } else { 255 };
            out.extend_from_slice(&[p[0], p[1], p[2], a]);
        }
    }
    (out, w, h)
}

fn decode_jxl_rs(data: &[u8], label: &str) -> (usize, usize) {
    use jxl::api::{
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let mut init = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match init.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => init = fallback,
            Err(e) => panic!("jxl-rs header {label}: {e:?}"),
        }
    };
    let (width, height) = decoder.basic_info().size;
    let format = decoder.current_pixel_format().clone();
    let channels = format.color_type.samples_per_pixel();
    let extras = format.extra_channel_format.len();
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![Some(JxlDataFormat::f32()); extras],
    });
    let mut frame = loop {
        match decoder.process(&mut input) {
            Ok(ProcessingResult::Complete { result }) => break result,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => decoder = fallback,
            Err(e) => panic!("jxl-rs frame info {label}: {e:?}"),
        }
    };
    let mut color = Image::<f32>::new((width * channels, height)).expect("alloc color");
    let mut extra: Vec<Image<f32>> = (0..extras)
        .map(|_| Image::<f32>::new((width, height)).expect("alloc extra"))
        .collect();
    let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
        color
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    for e in &mut extra {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            e.get_rect_mut(Rect {
                origin: (0, 0),
                size: (width, height),
            })
            .into_raw(),
        ));
    }
    loop {
        match frame.process(&mut input, &mut buffers) {
            Ok(ProcessingResult::Complete { .. }) => break,
            Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => frame = fallback,
            Err(e) => panic!("jxl-rs frame decode {label}: {e:?}"),
        }
    }
    (width, height)
}

/// Decode through djxl v0.12 and return its RGBA8 output.
fn decode_djxl_rgba(data: &[u8], label: &str) -> image::RgbaImage {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/aux-frames");
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join(format!("{label}.jxl"));
    let output = dir.join(format!("{label}.png"));
    std::fs::write(&input, data).unwrap();
    let result = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(&output)
        .arg("--num_threads=1")
        .output()
        .expect("run djxl v0.12");
    assert!(
        result.status.success(),
        "djxl {label}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    image::open(&output).expect("read djxl PNG").to_rgba8()
}

fn assert_decodes_with_alpha(bytes: &[u8], source: &[u8], w: u32, h: u32, label: &str) {
    assert_eq!(
        decode_jxl_rs(bytes, label),
        (w as usize, h as usize),
        "jxl-rs {label}"
    );
    let decoded = decode_djxl_rgba(bytes, label);
    assert_eq!(decoded.dimensions(), (w, h), "djxl {label}");
    for (i, (d, s)) in decoded.as_raw().chunks(4).zip(source.chunks(4)).enumerate() {
        assert_eq!(d[3], s[3], "djxl {label}: alpha sample {i}");
    }
}

#[test]
fn lf_frame_with_alpha_decodes() {
    for (w, h) in [(259u32, 133u32), (300, 260), (600, 400)] {
        let source = frymire_rgba(w, h);
        for effort in [5u8, 7] {
            let label = format!("lf-rgba-{w}x{h}-e{effort}");
            let bytes = LossyConfig::new(2.0)
                .with_effort(effort)
                .with_lf_frame(true)
                .encode(&source, w, h, PixelLayout::Rgba8)
                .expect("encode with LF frame");
            assert_decodes_with_alpha(&bytes, &source, w, h, &label);
        }
    }
}

#[test]
fn patches_with_alpha_decode() {
    let (source, w, h) = terminal_rgba();
    for effort in [5u8, 7] {
        let label = format!("patches-lossless-e{effort}");
        let bytes = LosslessConfig::new()
            .with_effort(effort)
            .with_patches(true)
            .encode(&source, w, h, PixelLayout::Rgba8)
            .expect("lossless encode with patches");
        assert_decodes_with_alpha(&bytes, &source, w, h, &label);
    }
    for effort in [7u8, 9] {
        let label = format!("patches-lossy-e{effort}");
        let bytes = LossyConfig::new(1.0)
            .with_effort(effort)
            .with_patches(true)
            .encode(&source, w, h, PixelLayout::Rgba8)
            .expect("lossy encode with patches");
        assert_decodes_with_alpha(&bytes, &source, w, h, &label);
    }
}

#[test]
fn lf_frame_small_images_decode() {
    for (w, h) in [(16u32, 16u32), (64, 32), (96, 48), (128, 64), (128, 128)] {
        let rgba = frymire_rgba(w, h);
        let rgb: Vec<u8> = rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        for effort in [3u8, 5, 8] {
            let label = format!("lf-rgb-{w}x{h}-e{effort}");
            let bytes = LossyConfig::new(2.0)
                .with_effort(effort)
                .with_lf_frame(true)
                .encode(&rgb, w, h, PixelLayout::Rgb8)
                .expect("encode with LF frame");
            assert_eq!(decode_jxl_rs(&bytes, &label), (w as usize, h as usize));
            assert_eq!(decode_djxl_rgba(&bytes, &label).dimensions(), (w, h));
            let label = format!("lf-rgba-{w}x{h}-e{effort}");
            let bytes = LossyConfig::new(2.0)
                .with_effort(effort)
                .with_lf_frame(true)
                .encode(&rgba, w, h, PixelLayout::Rgba8)
                .expect("encode RGBA with LF frame");
            assert_decodes_with_alpha(&bytes, &rgba, w, h, &label);
        }
    }
}

/// Lossy effort 1 and 2 skip code optimisation and default to the
/// single-pass entropy path, which cannot write extra channels. It used to
/// take that path anyway and drop alpha from the file; frames it cannot
/// represent now take the two-pass path.
#[test]
fn lossy_low_effort_keeps_alpha() {
    for (w, h) in [(64u32, 32u32), (300, 260)] {
        let source = frymire_rgba(w, h);
        for effort in [1u8, 2] {
            let label = format!("lossy-rgba-{w}x{h}-e{effort}");
            let bytes = LossyConfig::new(1.0)
                .with_effort(effort)
                .encode(&source, w, h, PixelLayout::Rgba8)
                .expect("low-effort lossy RGBA encode");
            assert_decodes_with_alpha(&bytes, &source, w, h, &label);
        }
    }
}

/// An LF frame cannot represent a resampled main frame's DC grid (libjxl
/// refuses the combination). LF frame and progressive DC requests are
/// ignored under resampling; such files used to be undecodable.
#[test]
fn lf_frame_with_resampling_decodes() {
    for (w, h) in [(64u32, 32u32), (300, 260)] {
        let rgba = frymire_rgba(w, h);
        let rgb: Vec<u8> = rgba.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        for factor in [2u32, 4, 8] {
            for pdc in [0u8, 1, 2] {
                for (layout, px) in [(PixelLayout::Rgb8, &rgb), (PixelLayout::Rgba8, &rgba)] {
                    let label = format!("lf-rs{factor}-pdc{pdc}-{layout:?}-{w}x{h}");
                    let mut cfg = LossyConfig::new(2.0)
                        .with_effort(7)
                        .with_resampling(factor)
                        .with_lf_frame(true);
                    if pdc > 0 {
                        cfg = cfg.with_progressive_dc(pdc);
                    }
                    let bytes = cfg
                        .encode(px, w, h, layout)
                        .expect("LF frame + resampling encode");
                    assert_eq!(
                        decode_jxl_rs(&bytes, &label),
                        (w as usize, h as usize),
                        "jxl-rs {label}"
                    );
                    assert_eq!(
                        decode_djxl_rgba(&bytes, &label).dimensions(),
                        (w, h),
                        "djxl {label}"
                    );
                }
            }
        }
    }
}

/// Mean absolute error in 8-bit levels between djxl's output and the source.
fn djxl_rgb_error(data: &[u8], source_rgb: &[u8], label: &str) -> f64 {
    let decoded = decode_djxl_rgba(data, label);
    let mut sum = 0f64;
    for (d, s) in decoded.as_raw().chunks(4).zip(source_rgb.chunks(3)) {
        for c in 0..3 {
            sum += (d[c] as f64 - s[c] as f64).abs();
        }
    }
    sum / source_rgb.len() as f64
}

/// cjxl v0.12 with `--progressive_dc=1` (its LF frame) on the same source.
fn cjxl_lf_frame(source_png: &std::path::Path, distance: f32, effort: u8, out: &std::path::Path) {
    let status = std::process::Command::new(jxl_encoder::test_helpers::cjxl_path())
        .arg(source_png)
        .arg(out)
        .arg(format!("--distance={distance}"))
        .arg(format!("--effort={effort}"))
        .arg("--progressive_dc=1")
        .arg("--resampling=1")
        .arg("--num_threads=1")
        .output()
        .expect("run cjxl v0.12");
    assert!(
        status.status.success(),
        "cjxl: {}",
        String::from_utf8_lossy(&status.stderr)
    );
}

/// The LF frame used to quantize DC at the main distance through squeeze
/// tables shifted by six levels, and skipped libjxl's 4:2:0 chroma squeeze
/// on DC images of 8x8 blocks or less. libjxl codes it at 0.02x the distance
/// (enc_cache.cc:137-139) with `DefaultSqueezeParameters` (squeeze.cc:387).
/// On small photos that cost up to 37x cjxl's error (a 1x1 crop at d4.5:
/// 3.67 levels against 0.10). Our LF frame must stay near cjxl's on real
/// crops from 1x1 to 64x48, through djxl and decodable by jxl-rs.
#[test]
fn lf_frame_error_tracks_cjxl_on_small_photos() {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus");
    let dir = corpus
        .get("CID22/CID22-512/validation")
        .expect("CID22 validation");
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/aux-frames");
    std::fs::create_dir_all(&out).unwrap();
    let mut worse = Vec::new();
    let mut cells = 0;
    for name in ["3156482.png", "225228.png", "2253934.png", "162520.png"] {
        let img = image::open(dir.join(name)).expect("CID22 image").to_rgb8();
        for (w, h) in [(1u32, 1u32), (3, 2), (7, 5), (17, 9), (64, 48)] {
            for (ox, oy) in [(37u32, 211u32), (400, 400)] {
                let crop = image::imageops::crop_imm(&img, ox, oy, w, h).to_image();
                let rgb = crop.as_raw().clone();
                let label = format!("lf-cjxl-{name}-{w}x{h}-{ox}");
                let png = out.join(format!("{label}.png"));
                crop.save(&png).unwrap();
                for distance in [1.0f32, 4.5, 7.0] {
                    let ours = LossyConfig::new(distance)
                        .with_effort(7)
                        .with_lf_frame(true)
                        .encode(&rgb, w, h, PixelLayout::Rgb8)
                        .expect("encode");
                    assert_eq!(decode_jxl_rs(&ours, &label), (w as usize, h as usize));
                    let reference = out.join(format!("{label}-d{distance}-cjxl.jxl"));
                    cjxl_lf_frame(&png, distance, 7, &reference);
                    let e_ours = djxl_rgb_error(&ours, &rgb, &format!("{label}-d{distance}"));
                    let e_ref = djxl_rgb_error(
                        &std::fs::read(&reference).unwrap(),
                        &rgb,
                        &format!("{label}-d{distance}-ref"),
                    );
                    cells += 1;
                    // djxl's 8-bit output is dithered: allow half a level on top.
                    if e_ours > 1.25 * e_ref + 0.5 {
                        worse.push(format!(
                            "{label} d{distance}: {e_ours:.2} vs cjxl {e_ref:.2}"
                        ));
                    }
                }
            }
        }
    }
    // Measured 2026-10-26: 2 of 120 (3x2 and 7x5 crops of 3156482 at d7);
    // the code before the fix put 88 of 120 over.
    eprintln!("lf_frame vs cjxl: {} of {cells} cells over the bound", worse.len());
    for w in &worse {
        eprintln!("  {w}");
    }
    assert!(
        worse.len() <= 3,
        "{} of {cells} cells over 1.25x cjxl's LF-frame error + 0.5 levels:\n{}",
        worse.len(),
        worse.join("\n")
    );
}
