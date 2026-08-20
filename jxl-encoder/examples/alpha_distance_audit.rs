//! `alpha_distance` parity audit vs `cjxl --alpha_distance`.
//!
//! Sweeps three RGBA test images at `alpha_distance ∈ {0.5, 1.0, 2.0, 5.0}`
//! with both jxl-encoder (`LossyConfig::with_alpha_distance(...)`) and cjxl
//! (libjxl `--alpha_distance D`). Decodes both via jxl-rs, extracts the
//! alpha plane, and reports byte size + alpha MAE per (image, distance).
//!
//! Target: parity within 5% bytes and parity within MAE (the underlying
//! quantizer formula is byte-for-byte ported from libjxl
//! `enc_modular.cc:973-1027` + `QuantizeChannel`, so byte-identical alpha
//! reconstruction is the expected outcome; size delta floats with VarDCT
//! RGB encoder differences).
//!
//! Outputs a TSV row per (image, alpha_distance, encoder). Run with:
//!
//! ```bash
//! CJXL=/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl \
//!     cargo run --release -p jxl-encoder --example alpha_distance_audit -- \
//!     --output /mnt/v/output/jxl-encoder/alpha-distance-audit-2026-05-17/sweep.tsv
//! ```

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use jxl_encoder::{LossyConfig, PixelLayout};

const COLOR_DISTANCE: f32 = 1.0;
const ALPHA_DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 5.0];

struct Image {
    label: &'static str,
    path: &'static str,
    alpha_character: &'static str,
}

const IMAGES: &[Image] = &[
    Image {
        label: "red_night_opaque",
        path: "/home/lilith/work/codec-corpus/imageflow/test_inputs/red-night.png",
        alpha_character: "100% opaque (trivial alpha)",
    },
    Image {
        label: "gradients_semitrans_ui",
        path: "/home/lilith/work/codec-corpus/imageflow/test_inputs/gradients.png",
        alpha_character: "semi-transparent UI gradient (98% mid-alpha)",
    },
    Image {
        label: "alpha_nonpremul_photo_mask",
        path: "/home/lilith/work/codec-corpus/jxl/reference/conformance/alpha_nonpremultiplied.png",
        alpha_character: "photographic alpha mask (99% mid-alpha)",
    },
];

fn main() {
    let mut output: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" | "-o" => {
                output = Some(PathBuf::from(args.next().expect("--output PATH")));
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    let output = output.expect("--output PATH required");
    let cjxl = env::var("CJXL")
        .unwrap_or_else(|_| "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl".to_string());
    assert!(
        Path::new(&cjxl).exists(),
        "cjxl not found at {cjxl} — set CJXL=/path/to/cjxl"
    );

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("mkdir output parent");
    }
    let mut tsv = fs::File::create(&output).expect("create output tsv");
    writeln!(
        tsv,
        "image\talpha_character\twidth\theight\tencoder\talpha_distance\tcolor_distance\tbytes\talpha_mae\talpha_max_err\trgb_mae\tnotes"
    )
    .unwrap();

    let tmpdir = tempdir_or("/tmp/alpha_audit");
    fs::create_dir_all(&tmpdir).expect("mkdir tmp");

    let header_repo = "alpha_distance parity audit (jxl-encoder vs cjxl v0.12.0)";
    eprintln!("# {header_repo}");
    eprintln!("# Output: {}", output.display());
    eprintln!();

    for img in IMAGES {
        eprintln!("=== {} :: {} ===", img.label, img.alpha_character);
        let (w, h, rgba_in) = read_png_rgba8(img.path);
        eprintln!("    size: {w}x{h}, {} bytes RGBA", rgba_in.len());

        for &ad in ALPHA_DISTANCES {
            // jxl-encoder encode
            let enc_jxl = encode_jxl_encoder(&rgba_in, w, h, COLOR_DISTANCE, ad);
            let dec_jxl = decode_jxl_rs_rgba8(&enc_jxl);
            let (mae_a_jxl, max_a_jxl) = alpha_err(&rgba_in, &dec_jxl);
            let mae_rgb_jxl = rgb_mae(&rgba_in, &dec_jxl);
            write_row(
                &mut tsv,
                img,
                w,
                h,
                "jxl_encoder",
                ad,
                enc_jxl.len(),
                mae_a_jxl,
                max_a_jxl,
                mae_rgb_jxl,
                "",
            );

            // cjxl encode (default --responsive=1, libjxl default for lossy
            // output, applies Squeeze transform before quantizing).
            let cjxl_jxl = encode_cjxl(&cjxl, img.path, COLOR_DISTANCE, ad, &tmpdir, true);
            let dec_cjxl = decode_jxl_rs_rgba8(&cjxl_jxl);
            let (mae_a_cjxl, max_a_cjxl) = alpha_err(&rgba_in, &dec_cjxl);
            let mae_rgb_cjxl = rgb_mae(&rgba_in, &dec_cjxl);
            write_row(
                &mut tsv,
                img,
                w,
                h,
                "cjxl_v0.12.0",
                ad,
                cjxl_jxl.len(),
                mae_a_cjxl,
                max_a_cjxl,
                mae_rgb_cjxl,
                "default --responsive=1",
            );

            // cjxl --responsive=0 (no-squeeze, same algorithm as jxl-encoder).
            // Apples-to-apples comparison: matches our encoder's pipeline
            // (raw-pixel quantize → gradient predictor) so MAE/bytes deltas
            // here isolate quantizer-formula and entropy-coder differences.
            let cjxl_r0 = encode_cjxl(&cjxl, img.path, COLOR_DISTANCE, ad, &tmpdir, false);
            let dec_cjxl_r0 = decode_jxl_rs_rgba8(&cjxl_r0);
            let (mae_a_r0, max_a_r0) = alpha_err(&rgba_in, &dec_cjxl_r0);
            let mae_rgb_r0 = rgb_mae(&rgba_in, &dec_cjxl_r0);
            write_row(
                &mut tsv,
                img,
                w,
                h,
                "cjxl_v0.12.0_r0",
                ad,
                cjxl_r0.len(),
                mae_a_r0,
                max_a_r0,
                mae_rgb_r0,
                "--responsive=0 (apples-to-apples)",
            );

            let dbytes_r1 = enc_jxl.len() as f64 / cjxl_jxl.len() as f64 - 1.0;
            let dbytes_r0 = enc_jxl.len() as f64 / cjxl_r0.len() as f64 - 1.0;
            eprintln!(
                "  d_alpha={ad:>4} | jxl_enc {:>7}B mae={:>6.3} max={:>3} | cjxl(r1) {:>7}B mae={:>6.3} max={:>3} {:+5.1}% | cjxl(r0) {:>7}B mae={:>6.3} max={:>3} {:+5.1}%",
                enc_jxl.len(),
                mae_a_jxl,
                max_a_jxl,
                cjxl_jxl.len(),
                mae_a_cjxl,
                max_a_cjxl,
                dbytes_r1 * 100.0,
                cjxl_r0.len(),
                mae_a_r0,
                max_a_r0,
                dbytes_r0 * 100.0,
            );
        }
        eprintln!();
    }

    eprintln!("Wrote {}", output.display());
}

#[allow(clippy::too_many_arguments)]
fn write_row(
    tsv: &mut fs::File,
    img: &Image,
    w: u32,
    h: u32,
    encoder: &str,
    alpha_distance: f32,
    bytes: usize,
    mae_a: f64,
    max_a: u32,
    mae_rgb: f64,
    notes: &str,
) {
    writeln!(
        tsv,
        "{label}\t{character}\t{w}\t{h}\t{encoder}\t{alpha_distance}\t{cd}\t{bytes}\t{mae_a:.5}\t{max_a}\t{mae_rgb:.5}\t{notes}",
        label = img.label,
        character = img.alpha_character,
        cd = COLOR_DISTANCE,
    )
    .unwrap();
}

fn tempdir_or(default: &str) -> PathBuf {
    env::var("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

fn read_png_rgba8(path: &str) -> (u32, u32, Vec<u8>) {
    let img = image::open(path).expect("decode png");
    let rgba = img.into_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let buf = rgba.into_raw();
    (w, h, buf)
}

fn encode_jxl_encoder(
    rgba: &[u8],
    w: u32,
    h: u32,
    color_distance: f32,
    alpha_distance: f32,
) -> Vec<u8> {
    LossyConfig::new(color_distance)
        .with_alpha_distance(Some(alpha_distance))
        .encode(rgba, w, h, PixelLayout::Rgba8)
        .expect("jxl-encoder lossy alpha encode")
        .to_vec()
}

fn encode_cjxl(
    cjxl: &str,
    input_png: &str,
    color_distance: f32,
    alpha_distance: f32,
    tmpdir: &Path,
    responsive: bool,
) -> Vec<u8> {
    let stem = Path::new(input_png)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let tag = if responsive { "r1" } else { "r0" };
    let out = tmpdir.join(format!(
        "{stem}_d{color_distance}_ad{alpha_distance}_{tag}.jxl"
    ));
    let _ = fs::remove_file(&out);
    let r_flag = if responsive { "1" } else { "0" };
    let status = Command::new(cjxl)
        .arg(input_png)
        .arg(&out)
        .arg("-d")
        .arg(format!("{color_distance}"))
        .arg("--alpha_distance")
        .arg(format!("{alpha_distance}"))
        .arg("-R")
        .arg(r_flag)
        .arg("-e")
        .arg("7")
        .arg("--quiet")
        .status()
        .expect("spawn cjxl");
    assert!(status.success(), "cjxl failed for {input_png}");
    fs::read(&out).expect("read cjxl output")
}

fn decode_jxl_rs_rgba8(data: &[u8]) -> Vec<u8> {
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl::image::{Image, Rect};

    let mut input = data;
    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);

    let mut decoder_init = decoder;
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
    let num_extras = basic_info.extra_channels.len();

    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Rgba,
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

    let channels = 4usize;
    let mut output_image =
        Image::<u8>::new((width * channels, height)).expect("alloc decode image");
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

    let mut pixels = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        pixels.extend_from_slice(output_image.row(y));
    }
    pixels
}

/// Returns (MAE, max abs err) for the alpha plane only.
fn alpha_err(input_rgba: &[u8], decoded_rgba: &[u8]) -> (f64, u32) {
    assert_eq!(input_rgba.len(), decoded_rgba.len());
    let mut sum: u64 = 0;
    let mut maxe: u32 = 0;
    let mut n: u64 = 0;
    for (a, b) in input_rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(decoded_rgba.as_chunks::<4>().0)
    {
        let e = (a[3] as i32 - b[3] as i32).unsigned_abs();
        sum += e as u64;
        if e > maxe {
            maxe = e;
        }
        n += 1;
    }
    (sum as f64 / n as f64, maxe)
}

fn rgb_mae(input_rgba: &[u8], decoded_rgba: &[u8]) -> f64 {
    assert_eq!(input_rgba.len(), decoded_rgba.len());
    let mut sum: u64 = 0;
    let mut n: u64 = 0;
    for (a, b) in input_rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(decoded_rgba.as_chunks::<4>().0)
    {
        for c in 0..3 {
            let e = (a[c] as i32 - b[c] as i32).unsigned_abs() as u64;
            sum += e;
            n += 1;
        }
    }
    sum as f64 / n as f64
}
