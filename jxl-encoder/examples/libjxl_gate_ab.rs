//! T4 — A/B one boolean libjxl-parity gate, with cjxl as the reference arm.
//!
//! Both arms are `EncoderStrategy::Custom` seeded from the Zenjxl defaults with
//! exactly ONE gate flipped, so everything else — quant field, AC strategy,
//! entropy coding — is bit-identical by construction and the delta is
//! attributable to the gate alone. A third arm shells out to `cjxl` at the same
//! effort/distance when it is on PATH, because "our two arms differ by X" is a
//! much weaker statement than "arm B lands on cjxl's RD point and arm A does
//! not".
//!
//! ```bash
//! cargo run -p jxl-encoder --release --example libjxl_gate_ab -- \
//!     <png-dir> <out.tsv> [gate-name]
//! ```
//!
//! Gates: `dc_adaptive_smoothing` (default), `x_qm_scale_from_original_distance`,
//! `header_all_default_fast_paths`.
//!
//! ## `dc_adaptive_smoothing` — what it measures
//!
//! We set `kSkipAdaptiveDCSmoothing` (`FrameHeader.flags` 0x80) on every lossy
//! VarDCT frame; libjxl sets it only for JPEG transcode (`enc_frame.cc:513`).
//! The thing that makes this cheap to answer: **the smoothing is a pure
//! decoder-side post-filter.** libjxl's encoder-side call (`enc_cache.cc:242`)
//! runs *after* `AddVarDCTDC` has already tokenized the DC and only touches
//! `shared.dc_storage`, its own reconstruction copy — which is what the "only
//! useful in tests and if inspection is enabled" TODO immediately above it
//! means. `ComputeCoefficients` (`enc_group.cc`) *writes* the DC plane and never
//! reads a smoothed one, so no AC decision depends on it. There is no
//! encoder-side algorithm to port: the two arms differ by one header bit.
//!
//! The filter (`compressed_dc.cc::AdaptiveDCSmoothing`): per DC pixel a 3x3
//! weighted blur `sm`, a `gap = max_c abs(mc - sm) / dc_quant_c` floored at 0.5,
//! and `out = mc + (sm - mc) * max(0, 3 - 4*gap)`. It fully replaces the DC with
//! its blur where the local deviation is within half a quantization step, and
//! does nothing once it exceeds 0.75 — a banding suppressor for smooth regions,
//! so it should do most of its work at LOW bitrate on SMOOTH content.
//!
//! ## `x_qm_scale_from_original_distance` — what it measures
//!
//! libjxl captures `original_butteraugli_distance` before the d >= 10
//! auto-resample rewrites `butteraugli_distance` to `d*0.25 + 0.25`, and scores
//! `x_qm_scale`'s {2.5, 5.5, 9.5} ladder against the ORIGINAL. We ported the
//! rewrite but not the capture. Only cells at d >= 10 can differ.
//!
//! ## Self-check built into the output
//!
//! The `pixels_differ` column catches the failure mode that would make the
//! whole measurement meaningless: a decoder (or a gate) that changes nothing.
//! If it reads `false` everywhere, the two arms decoded identically and the
//! quality columns say nothing about the gate. Treat an all-false run as a
//! broken measurement, not as "the gate has no effect".

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

/// Set one named boolean gate on an otherwise-Zenjxl bundle. Panics on an
/// unknown name rather than silently measuring nothing.
fn improvements_with_gate(gate: &str, on: bool) -> EncoderImprovementsCustom {
    let mut c = EncoderImprovementsCustom::default(); // == Zenjxl
    match gate {
        "dc_adaptive_smoothing" => c.dc_adaptive_smoothing = on,
        "x_qm_scale_from_original_distance" => c.x_qm_scale_from_original_distance = on,
        "header_all_default_fast_paths" => c.header_all_default_fast_paths = on,
        other => panic!(
            "unknown gate {other:?}; known: dc_adaptive_smoothing, \
             x_qm_scale_from_original_distance, header_all_default_fast_paths"
        ),
    }
    c
}

fn encode_and_score(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    gate: (&str, bool),
    reference: &Reference<'_>,
) -> Option<Scored> {
    let custom = improvements_with_gate(gate.0, gate.1);
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

/// Score an already-encoded bitstream. Split out of `encode_and_score` so the
/// cjxl reference arm goes through the identical decode + metric path — a
/// reference measured a different way is not a reference.
fn score_bytes(bytes: &[u8], w: u32, h: u32, reference: &Reference<'_>) -> Option<Scored> {
    let (dw, dh, dec) = decode_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec.chunks(3).map(|c| RGB::new(c[0], c[1], c[2])).collect();
    let butteraugli = butteraugli_linear(
        reference.linear.as_ref(),
        Img::new(dec_pixels, dw, dh).as_ref(),
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

/// cjxl reference arm. Returns `None` when cjxl is absent or fails, so the
/// harness stays usable on a machine without it — the columns simply read
/// empty rather than the run aborting.
fn cjxl_reference(
    png: &Path,
    distance: f32,
    effort: u8,
    w: u32,
    h: u32,
    reference: &Reference<'_>,
    scratch_dir: &Path,
) -> Option<Scored> {
    // Scratch lands next to the caller's output file, never in the system temp
    // dir (project rule: `/tmp` is wiped at unpredictable times).
    let out = scratch_dir.join(format!(
        "libjxl_gate_ab_ref_{}_{}_{}.jxl",
        std::process::id(),
        effort,
        (distance * 10.0) as u32
    ));
    let status = std::process::Command::new("cjxl")
        .args(["-d", &distance.to_string(), "-e", &effort.to_string()])
        .arg("--num_threads=0")
        .arg(png)
        .arg(&out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out).ok()?;
    let _ = std::fs::remove_file(&out);
    score_bytes(&bytes, w, h, reference)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| ".".into()));
    let out = PathBuf::from(
        args.get(2)
            .cloned()
            .unwrap_or_else(|| "libjxl_gate_ab.tsv".into()),
    );
    let gate: String = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "dc_adaptive_smoothing".into());
    // Fail fast on a typo rather than measuring the default gate under the
    // wrong name in the output file.
    let _ = improvements_with_gate(&gate, true);
    let scratch_dir = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    eprintln!("gate under test: {gate}");

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
        "gate\timage\twidth\theight\teffort\tdistance\tbytes_off\tbytes_on\tbytes_delta\t\
         bfly_off\tbfly_on\tbfly_delta_pct\tssim2_off\tssim2_on\tssim2_delta\t\
         pixels_differ\tmax_abs_pixel_delta\tcjxl_bytes\tcjxl_bfly\tcjxl_ssim2"
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
                let cj = cjxl_reference(png, d, e, w, h, &reference, &scratch_dir);
                let (Some(a), Some(b)) = (
                    encode_and_score(&rgb, w, h, d, e, (&gate, false), &reference),
                    encode_and_score(&rgb, w, h, d, e, (&gate, true), &reference),
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
                let (cjb, cjf, cjs) = match &cj {
                    Some(c) => (
                        c.bytes.to_string(),
                        format!("{:.5}", c.butteraugli),
                        format!("{:.4}", c.ssim2),
                    ),
                    None => (String::new(), String::new(), String::new()),
                };
                writeln!(
                    f,
                    "{gate}\t{name}\t{w}\t{h}\t{e}\t{d}\t{}\t{}\t{}\t{:.5}\t{:.5}\t{:+.3}\t\
                     {:.4}\t{:.4}\t{:+.4}\t{}\t{:.6}\t{cjb}\t{cjf}\t{cjs}",
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
                let cjs_txt = match &cj {
                    Some(c) => format!(
                        "  | cjxl {:>7} bfly {:.4} ssim2 {:.3}",
                        c.bytes, c.butteraugli, c.ssim2
                    ),
                    None => String::new(),
                };
                println!(
                    "{name:20} e{e} d{d:<4} bytes {:>7} -> {:>7} ({:+})  bfly {:.4} -> {:.4} \
                     ({:+.2}%)  ssim2 {:.3} -> {:.3} ({:+.3})  differ={}{cjs_txt}",
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
