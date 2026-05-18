//! W41-1 paired A/B for distance-aware `min_peak` patches gate (issue #52).
//!
//! Cells: 3 screenshots (`imac_g3`, `codec_wiki`, `terminal`) ×
//!        3 photos (`cid22/1025469`, `cid22/1189261`, `cid22/1418519`) ×
//!        6 distances {0.5, 1.0, 2.0, 3.0, 4.0, 5.0} × effort 7
//!        = 36 cells per run.
//!
//! Run once with binary built on `main@origin` (baseline) and once with
//! the W41-1 fix applied; both writes append rows with a `--label <s>`
//! discriminator into the same TSV so a post-hoc grouper can pair them.
//!
//! Metrics: Rust butteraugli (`butteraugli_linear`) + Rust fast-ssim2
//! decoded via jxl-oxide (`srgb_linear`) — avoids the `butteraugli_main`
//! PNG-metadata bug (CLAUDE.md). 3 repeats per cell for wall-clock
//! median/best, take the best.
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!       --features 'std parallel butteraugli-loop' \
//!       --example patches_min_peak_distance_ab \
//!       -- --label fix --output benchmarks/patches_min_peak_distance_2026-05-19.tsv

use butteraugli::{ButteraugliParams, butteraugli_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
const EFFORT: u8 = 7;
const REPEATS: usize = 3;

struct Source {
    label: &'static str,
    class: &'static str,
    path: PathBuf,
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
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

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn encode(rgb_u8: &[u8], w: u32, h: u32, d: f32) -> Result<Vec<u8>, String> {
    let cfg = LossyConfig::new(d).with_effort(EFFORT);
    cfg.encode(rgb_u8, w, h, PixelLayout::Rgb8)
        .map_err(|e| format!("encode failed: {e:?}"))
}

#[derive(Debug, Clone, Copy)]
struct Measure {
    bytes: usize,
    butteraugli: f64,
    ssim2: f64,
    encode_ms_median: f64,
    encode_ms_best: f64,
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

fn measure(
    rgb_u8: &[u8],
    w: u32,
    h: u32,
    d: f32,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Result<Measure, String> {
    let mut times: Vec<f64> = Vec::with_capacity(REPEATS);
    let mut last_bytes: Option<Vec<u8>> = None;
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let bytes = encode(rgb_u8, w, h, d)?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        last_bytes = Some(bytes);
    }
    let bytes = last_bytes.unwrap();

    let (dw, dh, decoded_linear) =
        decode_jxl_linear(&bytes).ok_or_else(|| "decode failed".to_string())?;
    if dw != w as usize || dh != h as usize {
        return Err(format!("decoded {}x{} != {}x{}", dw, dh, w, h));
    }

    let dec_pixels: Vec<RGB<f32>> = decoded_linear
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);

    let dec_srgb: Vec<[u8; 3]> = decoded_linear
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);

    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let median = median_f64(&mut times);

    Ok(Measure {
        bytes: bytes.len(),
        butteraugli: bfly,
        ssim2,
        encode_ms_median: median,
        encode_ms_best: best,
    })
}

struct Args {
    label: String,
    output: PathBuf,
}

fn parse_args() -> Args {
    let mut label: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--label" => label = Some(it.next().expect("--label VALUE")),
            "--output" => output = Some(PathBuf::from(it.next().expect("--output VALUE"))),
            other => panic!("unknown arg: {other}"),
        }
    }
    Args {
        label: label.expect("--label required"),
        output: output.unwrap_or_else(|| {
            PathBuf::from("benchmarks/patches_min_peak_distance_2026-05-19.tsv")
        }),
    }
}

fn main() {
    let args = parse_args();

    let corpus = PathBuf::from(
        std::env::var("CODEC_CORPUS_DIR")
            .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus")),
    );

    let sources = vec![
        Source {
            label: "screen/imac_g3",
            class: "screenshot",
            path: corpus.join("gb82-sc/imac_g3.png"),
        },
        Source {
            label: "screen/codec_wiki",
            class: "screenshot",
            path: corpus.join("gb82-sc/codec_wiki.png"),
        },
        Source {
            label: "screen/terminal",
            class: "screenshot",
            path: corpus.join("gb82-sc/terminal.png"),
        },
        Source {
            label: "screen/windows95",
            class: "screenshot",
            path: corpus.join("gb82-sc/windows95.png"),
        },
        Source {
            label: "cid22/1025469",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1025469.png"),
        },
        Source {
            label: "cid22/1189261",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1189261.png"),
        },
        Source {
            label: "cid22/1418519",
            class: "photo",
            path: corpus.join("CID22/CID22-512/validation/1418519.png"),
        },
    ];

    // Materialize header in tmp before any cells (idempotent: if output
    // exists with header already, just append rows).
    let header = "label\timage\tclass\twidth\theight\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms_median\tencode_ms_best\n";
    let need_header = !args.output.exists()
        || std::fs::read_to_string(&args.output)
            .map(|s| s.is_empty())
            .unwrap_or(true);
    if let Some(parent) = args.output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if need_header {
        let _ = std::fs::write(&args.output, header);
    }

    let params = ButteraugliParams::default();
    let mut all_rows: Vec<String> = Vec::new();

    for src in &sources {
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("skip {}: {}", src.path.display(), e);
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let rgb_u8 = img.to_rgb8().into_raw();
        let orig_linear: Vec<RGB<f32>> = rgb_u8
            .chunks(3)
            .map(|c| {
                RGB::new(
                    srgb_to_linear_f32(c[0]),
                    srgb_to_linear_f32(c[1]),
                    srgb_to_linear_f32(c[2]),
                )
            })
            .collect();
        let orig_linear_img = Img::new(orig_linear, w as usize, h as usize);
        let orig_srgb: Vec<[u8; 3]> = rgb_u8.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let orig_srgb_img = Img::new(orig_srgb, w as usize, h as usize);

        for &d in DISTANCES {
            match measure(&rgb_u8, w, h, d, &orig_linear_img, &orig_srgb_img, &params) {
                Ok(m) => {
                    let row = format!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\t{:.2}\n",
                        args.label,
                        src.label,
                        src.class,
                        w,
                        h,
                        EFFORT,
                        d,
                        m.bytes,
                        m.butteraugli,
                        m.ssim2,
                        m.encode_ms_median,
                        m.encode_ms_best,
                    );
                    eprintln!(
                        "[{}] {} d={:.2}: {} B, bfly={:.4}, ssim2={:.2}, ms_best={:.1}",
                        args.label, src.label, d, m.bytes, m.butteraugli, m.ssim2, m.encode_ms_best
                    );
                    all_rows.push(row);
                }
                Err(e) => eprintln!("[{}] {} d={:.2}: ERR {}", args.label, src.label, d, e),
            }
        }
    }

    // Append all rows atomically.
    let mut combined = std::fs::read_to_string(&args.output).unwrap_or_default();
    for r in &all_rows {
        combined.push_str(r);
    }
    let tmp = args.output.with_extension("tsv.tmp");
    let _ = std::fs::write(&tmp, &combined);
    std::fs::rename(&tmp, &args.output).expect("atomic mv");
    eprintln!("wrote {} rows to {}", all_rows.len(), args.output.display());

    // Sidecar .meta — preserve invocation if first run, else append.
    let meta_path = args.output.with_extension("meta");
    let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "?".into());
    let datestamp = chrono_now();
    let meta_line = format!(
        "label={}\thost={}\tutc={}\tdistances={:?}\teffort={}\trepeats={}\tcorpus={}\n",
        args.label,
        host,
        datestamp,
        DISTANCES,
        EFFORT,
        REPEATS,
        corpus.display()
    );
    let existing = std::fs::read_to_string(&meta_path).unwrap_or_default();
    let _ = std::fs::write(&meta_path, format!("{existing}{meta_line}"));
}

fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Simple YYYY-MM-DD HH:MM:SS UTC (no chrono dep)
    let days = (secs / 86400) as i64;
    let secs_today = secs % 86400;
    let h = secs_today / 3600;
    let m = (secs_today % 3600) / 60;
    let s = secs_today % 60;
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, day, h, m, s)
}

#[allow(dead_code)]
fn _path_str(p: &Path) -> String {
    p.display().to_string()
}
