//! Focused A/B bench for the W42-2 patches `group_size_shift` fix.
//!
//! Runs ours (post-fix) vs cjxl 0.12.0 on a screenshot subset where the
//! patches-ref-frame group_size_shift change is expected to bite, plus a
//! photo subset to verify zero regression on PatchesDispatch::Auto's
//! short-circuit path. Single-thread, default `LossyConfig::new(d)` plus
//! `with_effort(7)` (the effort gate for the W42-1 wedge).
//!
//! Output schema mirrors `rd_curve_audit_vs_cjxl`:
//!   image | class | width | height | encoder | effort | distance |
//!   bytes | butteraugli | ssim2 | encode_ms
//!
//! Reproducer:
//! ```bash
//! cargo run -p jxl-encoder --release --example patches_group_size_shift_ab \
//!   -- --output benchmarks/patches_group_size_shift_2026-05-18.tsv
//! ```
//!
//! Decode via jxl-oxide linear-sRGB (per CLAUDE.md PNG-metadata immunity).

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

// Screenshot wedge cells (W42-1) — small patches ref frames.
const SCREENSHOTS: &[&str] = &[
    "imac_g3.png",
    "terminal.png",
    "codec_wiki.png",
    "windows95.png",
];

// Photo guard cells — PatchesDispatch::Auto must short-circuit here.
const PHOTOS: &[&str] = &["1025469.png", "1418519.png", "1531677.png"];

// Distances that exercise the wedge (cjxl streaming-off path triggers at d>=3).
// We also sample d=2.0 to verify our advantage over cjxl is preserved.
const DISTANCES: &[f32] = &[2.0, 3.0, 4.0, 5.0];

const EFFORT: u8 = 7;

fn corpus_dir() -> PathBuf {
    std::env::var("CODEC_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/lilith/work/codec-corpus"))
}

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
    PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn srgb_to_linear_f32(s: u8) -> f32 {
    let c = s as f32 / 255.0;
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

fn rgb_to_linear_img(rgb: &[u8], w: u32, h: u32) -> Img<Vec<RGB<f32>>> {
    let pixels: Vec<RGB<f32>> = rgb
        .chunks(3)
        .map(|c| {
            RGB::new(
                srgb_to_linear_f32(c[0]),
                srgb_to_linear_f32(c[1]),
                srgb_to_linear_f32(c[2]),
            )
        })
        .collect();
    Img::new(pixels, w as usize, h as usize)
}

fn rgb_to_srgb_arr3(rgb: &[u8], w: u32, h: u32) -> Img<Vec<[u8; 3]>> {
    let pixels: Vec<[u8; 3]> = rgb.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    Img::new(pixels, w as usize, h as usize)
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
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

fn score_jxl(
    bytes: &[u8],
    orig_linear: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<(f64, f64)> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
    let bfly = butteraugli_linear(
        orig_linear.as_ref(),
        dec_lin_img.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
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
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_ours(rgb: &[u8], w: u32, h: u32, distance: f32, effort: u8) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

fn encode_cjxl(
    src_png: &Path,
    distance: f32,
    effort: u8,
    work_dir: &Path,
    tag: &str,
) -> Option<(Vec<u8>, f64)> {
    let out = work_dir.join(format!("{tag}_cjxl_e{effort}_d{:.2}.jxl", distance));
    let _ = std::fs::remove_file(&out);
    let start = Instant::now();
    let status = Command::new(cjxl_bin())
        .arg(src_png)
        .arg(&out)
        .args(["-e", &effort.to_string()])
        .args(["-d", &format!("{distance}")])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .output()
        .ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if !status.status.success() {
        eprintln!(
            "cjxl failed for {} e{} d{}: {}",
            src_png.display(),
            effort,
            distance,
            String::from_utf8_lossy(&status.stderr)
        );
        return None;
    }
    let bytes = std::fs::read(&out).ok()?;
    let _ = std::fs::remove_file(&out);
    Some((bytes, ms))
}

struct Args {
    output: PathBuf,
    output_meta: PathBuf,
}

fn parse_args() -> Args {
    let mut output: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--output" => output = Some(PathBuf::from(it.next().expect("--output VALUE"))),
            other => panic!("unknown arg: {other}"),
        }
    }
    let output = output
        .unwrap_or_else(|| PathBuf::from("benchmarks/patches_group_size_shift_2026-05-18.tsv"));
    let output_meta = output.with_extension("meta");
    Args {
        output,
        output_meta,
    }
}

fn row_header() -> String {
    "image\tclass\twidth\theight\tencoder\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms\n"
        .to_string()
}

fn row_tsv(
    image: &str,
    class: &str,
    w: u32,
    h: u32,
    encoder: &str,
    effort: u8,
    distance: f32,
    bytes: usize,
    bfly: f64,
    ssim2: f64,
    encode_ms: f64,
) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\n",
        image, class, w, h, encoder, effort, distance, bytes, bfly, ssim2, encode_ms,
    )
}

fn write_atomic(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)
            .expect("open tmp");
        f.write_all(contents.as_bytes()).expect("write");
        f.sync_all().ok();
    }
    std::fs::rename(&tmp, path).expect("atomic mv");
}

fn append_atomic(path: &Path, contents: &str) {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut combined = existing;
    combined.push_str(contents);
    write_atomic(path, &combined);
}

fn main() {
    let args = parse_args();
    let corpus = corpus_dir();
    let sc_dir = corpus.join("gb82-sc");
    let cid_dir = corpus.join("CID22/CID22-512/validation");

    // Atomic mv pattern: write to /tmp, then rename into benchmarks/.
    let tmp_out = std::env::temp_dir().join(format!(
        "patches_group_size_shift_ab_{}.tsv",
        std::process::id()
    ));
    write_atomic(&tmp_out, &row_header());

    let work_dir = std::env::temp_dir().join(format!(
        "patches_group_size_shift_ab_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work_dir).expect("workdir");

    // Discover images.
    let mut images: Vec<(String, &str, PathBuf)> = Vec::new();
    for name in SCREENSHOTS {
        let p = sc_dir.join(name);
        if p.exists() {
            images.push((name.to_string(), "screenshot", p));
        } else {
            eprintln!("missing screenshot: {}", p.display());
        }
    }
    for name in PHOTOS {
        let p = cid_dir.join(name);
        if p.exists() {
            images.push((name.to_string(), "photo", p));
        } else {
            eprintln!("missing photo: {}", p.display());
        }
    }

    let total_cells = images.len() * DISTANCES.len() * 2;
    eprintln!(
        "patches_group_size_shift A/B: {} images × {} distances × 2 encoders = {} cells",
        images.len(),
        DISTANCES.len(),
        total_cells,
    );

    let mut completed = 0usize;
    for (name, class, path) in &images {
        let (rgb, w, h) = match load_png(path) {
            Some(t) => t,
            None => {
                eprintln!("skip {} (load failed)", path.display());
                continue;
            }
        };
        let orig_lin = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

        for &d in DISTANCES {
            // ours
            if let Some((bytes, ms)) = encode_ours(&rgb, w, h, d, EFFORT) {
                if let Some((bfly, ss2)) = score_jxl(&bytes, &orig_lin, &orig_srgb, w, h) {
                    append_atomic(
                        &tmp_out,
                        &row_tsv(
                            name,
                            class,
                            w,
                            h,
                            "ours",
                            EFFORT,
                            d,
                            bytes.len(),
                            bfly,
                            ss2,
                            ms,
                        ),
                    );
                }
            } else {
                eprintln!("ours encode failed: {} d={}", name, d);
            }
            // cjxl
            if let Some((bytes, ms)) = encode_cjxl(path, d, EFFORT, &work_dir, name) {
                if let Some((bfly, ss2)) = score_jxl(&bytes, &orig_lin, &orig_srgb, w, h) {
                    append_atomic(
                        &tmp_out,
                        &row_tsv(
                            name,
                            class,
                            w,
                            h,
                            "cjxl",
                            EFFORT,
                            d,
                            bytes.len(),
                            bfly,
                            ss2,
                            ms,
                        ),
                    );
                }
            } else {
                eprintln!("cjxl encode failed: {} d={}", name, d);
            }
            completed += 2;
            eprintln!("  [{}/{}] {} d={}", completed, total_cells, name, d);
        }
    }

    // Atomic mv into benchmarks/.
    if let Some(parent) = args.output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::rename(&tmp_out, &args.output).expect("atomic mv to benchmarks/");

    // Meta sidecar.
    let git_rev = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let host = Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let cjxl_ver = Command::new(cjxl_bin())
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into());
    let meta = format!(
        "# patches group_size_shift A/B (W42-2)\n\
         git_rev: {git_rev}\n\
         host: {host}\n\
         cjxl_version: {cjxl_ver}\n\
         distances: {:?}\n\
         effort: {}\n\
         screenshots: {:?}\n\
         photos: {:?}\n\
         corpus: {}\n\
         schema: image\\tclass\\twidth\\theight\\tencoder\\teffort\\tdistance\\tbytes\\tbutteraugli\\tssim2\\tencode_ms\n\
         decoder: jxl-oxide (request_color_encoding=srgb_linear)\n\
         metrics: butteraugli_linear (default params), fast_ssim2 (sRGB u8 from linear)\n\
         reproducer: cargo run -p jxl-encoder --release --example patches_group_size_shift_ab -- --output {}\n",
        DISTANCES,
        EFFORT,
        SCREENSHOTS,
        PHOTOS,
        corpus.display(),
        args.output.display(),
    );
    let _ = std::fs::write(&args.output_meta, meta);
    eprintln!("wrote {}", args.output.display());
    eprintln!("wrote {}", args.output_meta.display());
}
