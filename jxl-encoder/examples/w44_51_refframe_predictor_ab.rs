//! W44-51: Patches ref-frame predictor A/B (tree-learn vs libjxl-parity Gradient).
//!
//! Hypothesis H3 from W44-48: our `patch_ref_tree_learning=true` (default at e>=7)
//! encodes the patches ref frame with tree-learned `Predictor::Variable`, while
//! libjxl unconditionally uses `Predictor::Gradient`
//! (`enc_patch_dictionary.cc:828`). The denser per-pixel ref-frame bytes
//! from Gradient may subtract MORE energy from the main residual, matching
//! the cjxl "+1590 patches / -4600 main" pattern from W44-46.
//!
//! Test: A/B `LossyInternalParams::patch_ref_tree_learning` Some(true) vs Some(false)
//! on codec_wiki e7 d=3-6 (4 wedge cells), plus the other 3 gb82-sc screenshots
//! to verify no regression.
//!
//! Output schema mirrors `patches_group_size_shift_ab`:
//!   image | width | height | encoder | effort | distance |
//!   bytes | butteraugli | ssim2 | encode_ms
//!
//! Where encoder is one of {"ours_tree", "ours_grad", "cjxl"}.
//!
//! Reproducer (requires __expert feature):
//! ```bash
//! cargo run -p jxl-encoder --release --features __expert \
//!     --example w44_51_refframe_predictor_ab \
//!     -- --output benchmarks/w44_51_refframe_predictor_ab_2026-05-19.tsv
//! ```

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SCREENSHOTS: &[&str] = &[
    "codec_wiki.png", // wedge target
    "imac_g3.png",
    "terminal.png",
    "windows95.png",
];

// Codec_wiki wedge per W44-37/W44-46/W44-48 lives at e7 d=3..6.
// Bench the same range across all 4 screenshots so we can spot regressions.
const DISTANCES: &[f32] = &[3.0, 4.0, 5.0, 6.0];

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
        .map(|c| RGB::new(srgb_to_linear_f32(c[0]), srgb_to_linear_f32(c[1]), srgb_to_linear_f32(c[2])))
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
        .map(|c| [linear_to_srgb_u8(c[0]), linear_to_srgb_u8(c[1]), linear_to_srgb_u8(c[2])])
        .collect();
    let dec_srgb_img: Img<Vec<[u8; 3]>> = Img::new(dec_srgb, dw, dh);
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref()).ok()?;
    Some((bfly, ssim2))
}

fn encode_ours(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    tree_learning: bool,
) -> Option<(Vec<u8>, f64)> {
    let mut params = LossyInternalParams::default();
    params.patch_ref_tree_learning = Some(tree_learning);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_internal_params(params);
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
        .unwrap_or_else(|| PathBuf::from("benchmarks/w44_51_refframe_predictor_ab_2026-05-19.tsv"));
    let output_meta = output.with_extension("meta");
    Args { output, output_meta }
}

fn row_header() -> String {
    "image\twidth\theight\tencoder\teffort\tdistance\tbytes\tbutteraugli\tssim2\tencode_ms\n"
        .to_string()
}

fn row_tsv(
    image: &str,
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
        "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.4}\t{:.2}\n",
        image, w, h, encoder, effort, distance, bytes, bfly, ssim2, encode_ms,
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

    let tmp_out = std::env::temp_dir().join(format!(
        "w44_51_refframe_predictor_ab_{}.tsv",
        std::process::id()
    ));
    write_atomic(&tmp_out, &row_header());

    let work_dir = std::env::temp_dir().join(format!(
        "w44_51_refframe_predictor_ab_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work_dir).expect("workdir");

    let mut done = 0usize;

    let mut cells: Vec<(String, PathBuf)> = Vec::new();
    for name in SCREENSHOTS {
        let p = sc_dir.join(name);
        if p.exists() {
            cells.push((name.to_string(), p));
        } else {
            eprintln!("missing: {}", p.display());
        }
    }

    let total = cells.len() * DISTANCES.len() * 3; // 3 encoders per cell

    for (img_name, src_png) in &cells {
        let (rgb, w, h) = match load_png(src_png) {
            Some(v) => v,
            None => {
                eprintln!("failed to load: {}", src_png.display());
                continue;
            }
        };
        let orig_linear = rgb_to_linear_img(&rgb, w, h);
        let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

        for &d in DISTANCES {
            // ours_tree (current production default at e7+: patch_ref_tree_learning=true)
            if let Some((bytes, ms)) = encode_ours(&rgb, w, h, d, EFFORT, true) {
                if let Some((bfly, ssim2)) = score_jxl(&bytes, &orig_linear, &orig_srgb, w, h) {
                    append_atomic(
                        &tmp_out,
                        &row_tsv(img_name, w, h, "ours_tree", EFFORT, d, bytes.len(), bfly, ssim2, ms),
                    );
                }
            }
            done += 1;
            eprintln!("[{done}/{total}] {img_name} d={d:.2} ours_tree done");

            // ours_grad (libjxl parity: patch_ref_tree_learning=false → Gradient)
            if let Some((bytes, ms)) = encode_ours(&rgb, w, h, d, EFFORT, false) {
                if let Some((bfly, ssim2)) = score_jxl(&bytes, &orig_linear, &orig_srgb, w, h) {
                    append_atomic(
                        &tmp_out,
                        &row_tsv(img_name, w, h, "ours_grad", EFFORT, d, bytes.len(), bfly, ssim2, ms),
                    );
                }
            }
            done += 1;
            eprintln!("[{done}/{total}] {img_name} d={d:.2} ours_grad done");

            // cjxl reference
            if let Some((bytes, ms)) = encode_cjxl(src_png, d, EFFORT, &work_dir, img_name) {
                if let Some((bfly, ssim2)) = score_jxl(&bytes, &orig_linear, &orig_srgb, w, h) {
                    append_atomic(
                        &tmp_out,
                        &row_tsv(img_name, w, h, "cjxl", EFFORT, d, bytes.len(), bfly, ssim2, ms),
                    );
                }
            }
            done += 1;
            eprintln!("[{done}/{total}] {img_name} d={d:.2} cjxl done");
        }
    }

    // Atomic mv from /tmp into benchmarks/
    if let Some(parent) = args.output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::rename(&tmp_out, &args.output).expect("final atomic mv");

    // Write companion .meta with reproducer + W44-48 context
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let meta = format!(
        "W44-51 — patches ref-frame predictor A/B (tree-learn vs libjxl-parity Gradient)\n\
         Date: 2026-05-19\n\
         Commit: {git_hash}\n\
         Host: {}\n\
         \n\
         Hypothesis (H3 from W44-48):\n\
         - libjxl unconditionally uses Predictor::Gradient for patches ref frames\n\
           (`~/work/jxl-efforts/libjxl/lib/jxl/enc_patch_dictionary.cc:828`).\n\
         - Our default at e7+ enables tree-learned Predictor::Variable when\n\
           ref_w>=128 AND ref_h>=128 (`vardct/patches.rs:2921`).\n\
         - codec_wiki e7 d=3..6 ref frame is 133×130 → triggers tree-learn.\n\
         - cjxl ships +1590 patches bytes / -4600 main bytes per W44-46:\n\
           denser per-pixel Gradient residuals subtract MORE main-AC energy.\n\
         - A/B: toggle `patch_ref_tree_learning: Some(true|false)` via __expert.\n\
         \n\
         Sweep design:\n\
         - 4 gb82-sc screenshots × 4 distances {{3,4,5,6}} × 3 encoders\n\
           = 48 encodes (16 ours_tree, 16 ours_grad, 16 cjxl).\n\
         - effort=7, threads=1, single-pass (no buttloop, no retries).\n\
         - Decode + score via jxl-oxide linear-sRGB (PNG-metadata immune\n\
           per CLAUDE.md).\n\
         \n\
         Reproducer:\n\
         ```\n\
         cargo run -p jxl-encoder --release --features __expert \\\n\
             --example w44_51_refframe_predictor_ab \\\n\
             -- --output benchmarks/w44_51_refframe_predictor_ab_2026-05-19.tsv\n\
         ```\n\
         \n\
         Cross-references:\n\
         - w44_48_threshold_relaxation_null_2026-05-19 (ruled out chunks B+C)\n\
         - w44_46_codec_wiki_patches_density_2026-05-19 (per-section ledger)\n\
         - w44_37_codec_wiki_wedge_2026-05-18 (origin of wedge)\n\
         - patches_group_size_shift_ab_2026-05-18 (W42-2 fix already shipped)\n\
        ",
        std::process::id(),
    );
    std::fs::write(&args.output_meta, meta).expect("write meta");

    eprintln!(
        "\nDone. Wrote {} ({} rows) and {}",
        args.output.display(),
        done,
        args.output_meta.display()
    );
}
