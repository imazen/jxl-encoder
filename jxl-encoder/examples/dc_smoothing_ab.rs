//! T4 — is libjxl's adaptive DC smoothing worth having?
//!
//! We set `kSkipAdaptiveDCSmoothing` (`FrameHeader.flags` 0x80) on **every**
//! lossy VarDCT frame. libjxl sets it only for JPEG transcode
//! (`enc_frame.cc:513`); an ordinary lossy encode ships `flags == 0` and lets
//! the decoder run `AdaptiveDCSmoothing` over the reconstructed DC.
//!
//! The thing that makes this cheap to answer: **the smoothing is a pure
//! decoder-side post-filter.** libjxl's encoder-side call
//! (`enc_cache.cc:242`) runs *after* `AddVarDCTDC` has already tokenized the
//! DC and only touches `shared.dc_storage`, its own reconstruction copy —
//! which is what the "only useful in tests and if inspection is enabled" TODO
//! immediately above it means. `ComputeCoefficients` (`enc_group.cc`) *writes*
//! the DC plane and never reads a smoothed one, so no AC decision depends on
//! it. There is therefore no encoder-side algorithm to port: the difference
//! between the two arms is one header bit.
//!
//! The filter itself (`compressed_dc.cc::AdaptiveDCSmoothing`): per DC pixel,
//! a 3x3 weighted blur `sm`, a `gap = max_c |mc - sm| / dc_quant_c` floored at
//! 0.5, and `out = mc + (sm - mc) * max(0, 3 - 4*gap)`. So it fully replaces
//! the DC with its blur where the local deviation is within half a
//! quantization step (deviation indistinguishable from quantization noise),
//! tapers off, and does nothing once the deviation exceeds 0.75 steps. It is a
//! banding/blocking suppressor for smooth regions, and it should therefore do
//! most of its work at LOW bitrate on SMOOTH content.
//!
//! ## Arms
//!
//! Both arms are `EncoderStrategy::Custom` seeded from the Zenjxl defaults so
//! `dc_adaptive_smoothing` is the ONLY difference. Everything else — quant
//! field, AC strategy, entropy coding — is bit-identical by construction.
//!
//! ## Self-check built into the output
//!
//! The `pixels_differ` column exists to catch the failure mode that would make
//! this whole measurement meaningless: a decoder that ignores the flag. If it
//! reads `false` everywhere, the two arms decoded identically and the numbers
//! below say nothing about DC smoothing — they say the decoder does not
//! implement it. Treat any all-false run as a broken measurement, not as
//! "smoothing has no effect".
//!
//! ```bash
//! cargo run -p jxl-encoder --release --example dc_smoothing_ab -- <png-dir> <out.tsv>
//! ```

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderImprovementsCustom, EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

fn srgb_to_linear_f32(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let srgb = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn decode_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let mut img = jxl_oxide::JxlImage::builder()
        .read(Cursor::new(bytes))
        .ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

struct Scored {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    decoded: Vec<f32>,
}

/// The reference image in both the linear (butteraugli) and sRGB-u8 (ssim2)
/// forms the two metrics want, computed once per fixture.
struct Reference<'a> {
    linear: &'a Img<Vec<RGB<f32>>>,
    srgb: &'a Img<Vec<[u8; 3]>>,
}

fn encode_and_score(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    smoothing: bool,
    reference: &Reference<'_>,
) -> Option<Scored> {
    let custom = EncoderImprovementsCustom {
        dc_adaptive_smoothing: smoothing,
        ..EncoderImprovementsCustom::default() // == Zenjxl
    };
    let bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Custom(Box::new(custom)))
        .encode(rgb, w, h, PixelLayout::Rgb8)
        .ok()?;
    let (dw, dh, dec) = decode_linear(&bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let dec_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let butteraugli = butteraugli_linear(
        reference.linear.as_ref(),
        dec_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let dec_srgb: Vec<[u8; 3]> = dec
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let ssim2 = fast_ssim2::compute_ssimulacra2(
        reference.srgb.as_ref(),
        Img::new(dec_srgb, dw, dh).as_ref(),
    )
    .ok()?;
    Some(Scored {
        bytes: bytes.len(),
        butteraugli,
        ssim2,
        decoded: dec,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| ".".into()));
    let out = PathBuf::from(
        args.get(2)
            .cloned()
            .unwrap_or_else(|| "dc_smoothing_ab.tsv".into()),
    );

    // Distances span the aggressive-compression band the project cares about
    // (a smoothing filter gated on "deviation within a quantization step"
    // should do more work as the step grows), plus a near-lossless anchor.
    let distances = [0.5_f32, 1.0, 2.0, 4.0, 7.0, 10.0];
    let efforts = [5_u8, 7];

    let mut pngs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {dir:?}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("png"))
        .collect();
    pngs.sort();
    assert!(!pngs.is_empty(), "no PNGs in {dir:?}");

    let mut f = std::fs::File::create(&out).expect("create tsv");
    writeln!(
        f,
        "image\twidth\theight\teffort\tdistance\tbytes_skip\tbytes_smooth\tbytes_delta\t\
         bfly_skip\tbfly_smooth\tbfly_delta_pct\tssim2_skip\tssim2_smooth\tssim2_delta\t\
         pixels_differ\tmax_abs_pixel_delta"
    )
    .unwrap();

    for png in &pngs {
        let Some((rgb, w, h)) = load_png(png) else {
            eprintln!("skip (not RGB8-loadable): {png:?}");
            continue;
        };
        let orig_linear: Img<Vec<RGB<f32>>> = Img::new(
            rgb.chunks(3)
                .map(|c| {
                    RGB::new(
                        srgb_to_linear_f32(c[0]),
                        srgb_to_linear_f32(c[1]),
                        srgb_to_linear_f32(c[2]),
                    )
                })
                .collect(),
            w as usize,
            h as usize,
        );
        let orig_srgb: Img<Vec<[u8; 3]>> = Img::new(
            rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            w as usize,
            h as usize,
        );
        let name = png.file_stem().unwrap().to_string_lossy().to_string();
        for &e in &efforts {
            for &d in &distances {
                let reference = Reference {
                    linear: &orig_linear,
                    srgb: &orig_srgb,
                };
                let (Some(a), Some(b)) = (
                    encode_and_score(&rgb, w, h, d, e, false, &reference),
                    encode_and_score(&rgb, w, h, d, e, true, &reference),
                ) else {
                    eprintln!("{name} e{e} d{d}: encode/decode failed");
                    continue;
                };
                let max_delta = a
                    .decoded
                    .iter()
                    .zip(b.decoded.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                let bfly_pct = if a.butteraugli > 0.0 {
                    (b.butteraugli - a.butteraugli) / a.butteraugli * 100.0
                } else {
                    0.0
                };
                writeln!(
                    f,
                    "{name}\t{w}\t{h}\t{e}\t{d}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:+.3}\t{:.4}\t{:.4}\t\
                     {:+.4}\t{}\t{:.6}",
                    a.bytes,
                    b.bytes,
                    b.bytes as i64 - a.bytes as i64,
                    a.butteraugli,
                    b.butteraugli,
                    bfly_pct,
                    a.ssim2,
                    b.ssim2,
                    b.ssim2 - a.ssim2,
                    max_delta > 0.0,
                    max_delta,
                )
                .unwrap();
                println!(
                    "{name:22} e{e} d{d:<4} bytes {:>7} -> {:>7} ({:+})  bfly {:.4} -> {:.4} \
                     ({:+.2}%)  ssim2 {:.3} -> {:.3} ({:+.3})  differ={}",
                    a.bytes,
                    b.bytes,
                    b.bytes as i64 - a.bytes as i64,
                    a.butteraugli,
                    b.butteraugli,
                    bfly_pct,
                    a.ssim2,
                    b.ssim2,
                    b.ssim2 - a.ssim2,
                    max_delta > 0.0
                );
            }
        }
    }
    eprintln!("wrote {out:?}");
}
