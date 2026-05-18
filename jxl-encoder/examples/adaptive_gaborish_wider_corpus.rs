//! W20-1 follow-on: wider-corpus A/B for adaptive Gaborish kernel strength.
//!
//! Decides whether `LossyConfig::with_adaptive_gaborish(true)` should flip
//! to default-on. Initial 5-photo bench at d=1.0/e7 showed -1.74% bytes
//! (commit ecd1ec3c). This sweep widens the corpus, distances, and efforts
//! AND adds quality metrics (butteraugli + ssim2).
//!
//! Adaptive gaborish pre-bakes per-tile sharpening into post-Gab samples
//! handed to the DCT. The decoder always applies the same fixed 3x3
//! inverse blur, so the decoded pixels DIFFER between fixed (mul=1.0) and
//! adaptive (mul in [0.8, 1.0]). Bytes-only A/B is NOT enough — quality
//! must also be measured to confirm the savings aren't paid for in
//! perceptual degradation.
//!
//! Grid:
//!   - 25 stratified CID22-512 validation photos
//!   - 5 stratified gb82-sc screenshots
//!   - distances: {0.5, 1.0, 2.0, 4.0}
//!   - efforts:   {5, 7}
//!   - modes:     {fixed=default, adapt=with_adaptive_gaborish(true)}
//!   - = 30 * 4 * 2 * 2 = 480 cells
//!
//! Decode with jxl-oxide in linear sRGB (per CLAUDE.md), measure with
//! Rust butteraugli + fast-ssim2 (metadata-immune).
//!
//! TSV header:
//!   image\tclass\twidth\theight\tdistance\teffort\tmode\tbytes\tbutteraugli\tssim2
//!
//! Append-safe: each (image, distance, effort) writes two rows (fixed, adapt)
//! followed by a flush. Resumable: if the output already has a final row for
//! a given (image, distance, effort, mode) tuple, that cell is skipped.
//!
//! Usage:
//!   cargo run --release -p jxl-encoder --example adaptive_gaborish_wider_corpus \
//!       -- --out benchmarks/adaptive_gaborish_wider_corpus_2026-05-18.tsv
//!
//! Optional env:
//!   ADAPT_GAB_LIMIT_IMAGES=N   (cap images for a smoke run)
//!   ADAPT_GAB_LIMIT_CELLS=N    (cap total cells; stops cleanly mid-image)

use butteraugli::{ButteraugliParams, butteraugli_linear, srgb_to_linear};
use image::GenericImageView;
use imgref::Img;
use jxl_encoder::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DISTANCES: &[f32] = &[0.5, 1.0, 2.0, 4.0];
const EFFORTS: &[u8] = &[5, 7];

struct Source {
    label: String,
    class: &'static str,
    path: PathBuf,
}

fn cid22_picks() -> Vec<&'static str> {
    vec![
        "1025469.png",
        "1044329.png",
        "1279330.png",
        "1418519.png",
        "1475938.png",
        "1544947.png",
        "159550.png",
        "162520.png",
        "2079234.png",
        "2190188.png",
        "2253934.png",
        "2670327.png",
        "2736139.png",
        "2887497.png",
        "2936831.png",
        "3156482.png",
        "3637739.png",
        "3653963.png",
        "3762075.png",
        "4215100.png",
        "5055743.png",
        "6078297.png",
        "70497.png",
        "7062219.png",
        "792079.png",
    ]
}

fn screenshot_picks() -> Vec<&'static str> {
    vec![
        "codec_wiki.png",
        "graph.png",
        "imac_dark.png",
        "imac_g3_strip.png",
        "windows95.png",
    ]
}

fn build_sources(corpus: &Path) -> Vec<Source> {
    let mut out = Vec::new();
    let cid22_dir = corpus.join("CID22/CID22-512/validation");
    for name in cid22_picks() {
        let p = cid22_dir.join(name);
        if !p.exists() {
            eprintln!("WARN: missing CID22 image: {}", p.display());
            continue;
        }
        out.push(Source {
            label: format!("cid22/{}", name.trim_end_matches(".png")),
            class: "photo",
            path: p,
        });
    }
    let ss_dir = corpus.join("gb82-sc");
    for name in screenshot_picks() {
        let p = ss_dir.join(name);
        if !p.exists() {
            eprintln!("WARN: missing screenshot: {}", p.display());
            continue;
        }
        out.push(Source {
            label: format!("gb82-sc/{}", name.trim_end_matches(".png")),
            class: "screenshot",
            path: p,
        });
    }
    out
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

fn measure_bytes_and_quality(
    bytes: &[u8],
    w: usize,
    h: usize,
    orig_linear_img: &Img<Vec<RGB<f32>>>,
    orig_srgb_img: &Img<Vec<[u8; 3]>>,
    params: &ButteraugliParams,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w || dh != h {
        eprintln!("  dimension mismatch: got {}x{} expected {}x{}", dw, dh, w, h);
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, w, h);
    let bfly = butteraugli_linear(orig_linear_img.as_ref(), dec_linear_img.as_ref(), params)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = dec_lin
        .chunks(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, w, h);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb_img.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);
    Some((bfly, ssim2))
}

fn load_existing(path: &Path) -> HashSet<String> {
    let mut seen = HashSet::new();
    let Ok(f) = std::fs::File::open(path) else {
        return seen;
    };
    let r = BufReader::new(f);
    for line in r.lines().map_while(Result::ok) {
        if line.starts_with('#') || line.starts_with("image\t") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        // image class width height distance effort mode bytes butteraugli ssim2
        if cols.len() >= 7 {
            let key = format!("{}|{}|{}|{}", cols[0], cols[4], cols[5], cols[6]);
            seen.insert(key);
        }
    }
    seen
}

fn parse_args() -> (PathBuf, Option<usize>, Option<usize>) {
    let mut args = std::env::args().skip(1);
    let mut out = PathBuf::from("benchmarks/adaptive_gaborish_wider_corpus_2026-05-18.tsv");
    while let Some(a) = args.next() {
        if a == "--out" {
            if let Some(v) = args.next() {
                out = PathBuf::from(v);
            }
        }
    }
    let lim_img = std::env::var("ADAPT_GAB_LIMIT_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok());
    let lim_cells = std::env::var("ADAPT_GAB_LIMIT_CELLS")
        .ok()
        .and_then(|s| s.parse().ok());
    (out, lim_img, lim_cells)
}

fn write_header(path: &Path) -> std::io::Result<()> {
    if path.exists() && std::fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(
        f,
        "image\tclass\twidth\theight\tdistance\teffort\tmode\tbytes\tbutteraugli\tssim2"
    )?;
    Ok(())
}

fn main() {
    let (out_path, lim_img, lim_cells) = parse_args();
    eprintln!("# output: {}", out_path.display());
    eprintln!("# distances: {:?}", DISTANCES);
    eprintln!("# efforts: {:?}", EFFORTS);

    let corpus = std::env::var("CODEC_CORPUS_DIR")
        .unwrap_or_else(|_| String::from("/home/lilith/work/codec-corpus"));
    let mut sources = build_sources(&PathBuf::from(corpus));
    if let Some(n) = lim_img {
        sources.truncate(n);
    }
    eprintln!("# {} sources", sources.len());

    if let Err(e) = write_header(&out_path) {
        eprintln!("ERROR writing header to {}: {}", out_path.display(), e);
        std::process::exit(1);
    }
    let seen = load_existing(&out_path);
    eprintln!("# resume: {} existing rows", seen.len());

    let params = ButteraugliParams::default();

    let mut cells_written = 0usize;
    let mut cells_skipped = 0usize;

    'outer: for (si, src) in sources.iter().enumerate() {
        eprintln!("\n=== [{}/{}] {} ===", si + 1, sources.len(), src.label);
        let img = match image::open(&src.path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  open fail: {e}");
                continue;
            }
        };
        let (w_u32, h_u32) = img.dimensions();
        let (w, h) = (w_u32 as usize, h_u32 as usize);
        let rgb = img.to_rgb8();
        let linear_rgb: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();
        let orig_pixels: Vec<RGB<f32>> = linear_rgb
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let orig_linear_img = Img::new(orig_pixels, w, h);
        let orig_srgb: Vec<[u8; 3]> = rgb.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let orig_srgb_img = Img::new(orig_srgb, w, h);

        let mut file = match OpenOptions::new().append(true).open(&out_path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("ERROR reopening output: {e}");
                std::process::exit(1);
            }
        };

        for &d in DISTANCES {
            for &eff in EFFORTS {
                for &mode in &["fixed", "adapt"] {
                    if let Some(n) = lim_cells {
                        if cells_written >= n {
                            eprintln!("# cells limit hit: {}", cells_written);
                            break 'outer;
                        }
                    }
                    let key = format!("{}|{:.2}|{}|{}", src.label, d, eff, mode);
                    if seen.contains(&key) {
                        cells_skipped += 1;
                        continue;
                    }
                    let t0 = Instant::now();
                    let cfg = LossyConfig::new(d)
                        .with_effort(eff)
                        .with_adaptive_gaborish(mode == "adapt");
                    let enc_bytes = match cfg.encode(rgb.as_raw(), w_u32, h_u32, PixelLayout::Rgb8)
                    {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!("  d={} e={} mode={}: encode failed: {e:?}", d, eff, mode);
                            continue;
                        }
                    };
                    let enc_ms = t0.elapsed().as_millis();
                    let (bfly, ssim2) = match measure_bytes_and_quality(
                        &enc_bytes,
                        w,
                        h,
                        &orig_linear_img,
                        &orig_srgb_img,
                        &params,
                    ) {
                        Some(t) => t,
                        None => {
                            eprintln!("  d={} e={} mode={}: decode/metric fail", d, eff, mode);
                            continue;
                        }
                    };
                    let _ = writeln!(
                        file,
                        "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{:.6}\t{:.4}",
                        src.label,
                        src.class,
                        w,
                        h,
                        d,
                        eff,
                        mode,
                        enc_bytes.len(),
                        bfly,
                        ssim2
                    );
                    let _ = file.flush();
                    cells_written += 1;
                    eprintln!(
                        "  d={} e={} mode={}: {} B  bfly={:.4}  ssim2={:.2}  ({}ms)",
                        d, eff, mode, enc_bytes.len(), bfly, ssim2, enc_ms
                    );
                }
            }
        }
    }
    eprintln!(
        "\n# done. wrote {} cells, skipped {} existing.",
        cells_written, cells_skipped
    );
}
