//! W44-AUDIT-6 Phase 2C — extend M3 discriminator to W44-105 buttloop seed at e8/e9.
//!
//! Phase 1 shipped the M3>=80 exclude on the W44-109 pre-buttloop path
//! (e ∈ {5, 6, 7}). Phase 2C is the symmetric extension to the W44-105
//! buttloop seed scale (e ∈ {8, 9}, where `butteraugli_iters > 0`).
//!
//! AUDIT-4 measured codec_wiki at the W44-105 site:
//!   - e8 d=4: Zenjxl +20.72% bytes vs cjxl (lift over-allocates)
//!   - e9 d=4: Zenjxl  +4.88% bytes vs cjxl
//! Symmetric to e7 d=4 +44.03% wedge that Phase 1 closed.
//!
//! Modes (per cell):
//!   A_baseline_no_audit6  — JXL_W44_AUDIT_6_DISABLE=1 (pre-Phase-2C)
//!   B_shipped             — production default (Phase 2C exclude ACTIVE)
//!   C_no_w105             — JXL_BUTTLOOP_INITIAL_QF_SCALE=1.0 (disables W44-105 seed lift)
//!   D_cjxl                — cjxl v0.12.0
//!
//! Cells:
//!   codec_wiki e8 d=4   (primary target — W44-105 lift fires)
//!   codec_wiki e9 d=4   (primary target — W44-105 lift fires)
//!   codec_wiki e8 d=3   (below MIN_DISTANCE=3.5 → gate doesn't fire → A==B==C expected)
//!   imac_g3 e8 d=4      (W44-105 win cluster — A==B byte-identical expected, A != C)
//!   imac_g3 e9 d=4      (W44-105 win cluster — A==B byte-identical expected, A != C)
//!   terminal e8 d=4     (W44-176 terminal-class exclude already on → A==B==C expected)
//!
//! Pass criteria (per Phase 2C acceptance gate c):
//!   - codec_wiki e8 d=4 bytes go from +20.72% → near +0.22% (mirroring Phase 1)
//!   - codec_wiki e9 d=4 bytes go from +4.88% → near baseline (W44-105 disabled)
//!   - imac_g3 e8/e9 d=4: A == B byte-identical (gate doesn't fire on M3=14)
//!   - terminal e8 d=4: A == B == C byte-identical (W44-176 wins)
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_6_phase2c_w105_bisect

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const CELLS: &[(&str, &str, u8, f32, bool, &str)] = &[
    // (image_id, path, effort, distance, gate_expected_to_fire_in_B, class)
    (
        "codec_wiki",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        8,
        4.0,
        true, // M3=146 > 80, gate fires, A != B expected
        "TARGET_WEDGE",
    ),
    (
        "codec_wiki",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        9,
        4.0,
        true,
        "TARGET_WEDGE",
    ),
    (
        "codec_wiki",
        "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png",
        8,
        3.0,
        false, // d<3.5 → gate cannot fire → A==B==C expected
        "TARGET_BELOW_GATE",
    ),
    (
        "imac_g3",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        8,
        4.0,
        false, // M3=14 < 80, gate doesn't fire → A==B byte-identical, A != C
        "WIN_CLUSTER",
    ),
    (
        "imac_g3",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
        9,
        4.0,
        false,
        "WIN_CLUSTER",
    ),
    (
        "terminal",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        8,
        4.0,
        false, // W44-176 terminal-class exclude already wins → A==B==C
        "W44_176_OWNED",
    ),
];

fn srgb_u8_to_linear_f32(x: u8) -> f32 {
    let x = x as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(x: f32) -> u8 {
    let x = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Some((rgb.into_raw(), w, h))
}

fn make_imgs(pixels: &[u8], w: u32, h: u32) -> (Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>) {
    let lin: Vec<RGB<f32>> = pixels
        .chunks_exact(3)
        .map(|c| {
            RGB::new(
                srgb_u8_to_linear_f32(c[0]),
                srgb_u8_to_linear_f32(c[1]),
                srgb_u8_to_linear_f32(c[2]),
            )
        })
        .collect();
    let srgb: Vec<[u8; 3]> = pixels.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    (
        Img::new(lin, w as usize, h as usize),
        Img::new(srgb, w as usize, h as usize),
    )
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
        .chunks_exact(3)
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
        .chunks_exact(3)
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

fn encode_ours(
    pixels: &[u8],
    w: u32,
    h: u32,
    effort: u8,
    distance: f32,
    mode: &str,
) -> Option<(Vec<u8>, u128)> {
    unsafe {
        std::env::remove_var("JXL_W44_AUDIT_6_DISABLE");
        std::env::remove_var("JXL_BUTTLOOP_INITIAL_QF_SCALE");
        match mode {
            "A_baseline_no_audit6" => {
                std::env::set_var("JXL_W44_AUDIT_6_DISABLE", "1");
            }
            "B_shipped" => {}
            "C_no_w105" => {
                // Force the buttloop initial qf scale to 1.0 (disables W44-105
                // lift entirely). Per W44-132 Chunk F, this env var promotes
                // `AutoScale4` → `AutoScale(1.0)` BEFORE the gate predicate,
                // so the gate evaluates against scale=1.0 → no lift applied
                // regardless of is_screenshot/distance/m3.
                std::env::set_var("JXL_BUTTLOOP_INITIAL_QF_SCALE", "1.0");
            }
            _ => unreachable!("unknown mode: {mode}"),
        }
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_strategy(EncoderStrategy::Zenjxl);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok()?;
    let ms = t0.elapsed().as_millis();
    unsafe {
        std::env::remove_var("JXL_W44_AUDIT_6_DISABLE");
        std::env::remove_var("JXL_BUTTLOOP_INITIAL_QF_SCALE");
    }
    Some((buf, ms))
}

fn encode_cjxl(src_path: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit6_p2c_{}_{}_{}_{}.jxl",
        src_path.file_stem()?.to_string_lossy(),
        effort,
        (distance * 1000.0) as u32,
        std::process::id()
    ));
    let t0 = Instant::now();
    let status = Command::new(cjxl)
        .arg(src_path)
        .arg(&tmp)
        .arg("-e")
        .arg(format!("{}", effort))
        .arg("-d")
        .arg(format!("{}", distance))
        .arg("--quiet")
        .status()
        .ok()?;
    let ms = t0.elapsed().as_millis();
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    out.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    let out_path: PathBuf =
        PathBuf::from("benchmarks/w44_audit_6_phase2c_w105_bisect_2026-05-24.tsv");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(&out_path).expect("create TSV");
    writeln!(
        f,
        "image_id\teffort\tdistance\tclass\tmode\tbytes\tsha8\tbfly\tssim2\tms\t\
         delta_bytes_pct_vs_cjxl\tdelta_ssim2_vs_cjxl"
    )
    .unwrap();

    let bench_start = Instant::now();
    for &(image_id, image_path, effort, dist, gate_expected, class) in CELLS {
        let (pixels, w, h) = match load_png(Path::new(image_path)) {
            Some(t) => t,
            None => {
                eprintln!("[bench] FAIL: load {}", image_path);
                continue;
            }
        };
        let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

        eprintln!(
            "\n[cell] {} e{} d={} class={} gate_expected_in_B={}",
            image_id, effort, dist, class, gate_expected
        );

        // cjxl reference first
        let (cjxl_b, cjxl_ms) = encode_cjxl(Path::new(image_path), effort, dist).unwrap();
        let cjxl_bytes = cjxl_b.len();
        let (cjxl_bfly, cjxl_ssim2) =
            score_jxl(&cjxl_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
        eprintln!(
            "  D=cjxl                 {} B  bfly={:.3} ssim2={:.2} ({} ms)",
            cjxl_bytes, cjxl_bfly, cjxl_ssim2, cjxl_ms
        );
        writeln!(
            f,
            "{}\t{}\t{}\t{}\tD_cjxl\t{}\t{}\t{:.4}\t{:.4}\t{}\t0\t0",
            image_id,
            effort,
            dist,
            class,
            cjxl_bytes,
            &sha256_hex(&cjxl_b)[..8],
            cjxl_bfly,
            cjxl_ssim2,
            cjxl_ms
        )
        .unwrap();

        for mode in ["A_baseline_no_audit6", "B_shipped", "C_no_w105"] {
            let (buf, ms) = encode_ours(&pixels, w, h, effort, dist, mode).unwrap();
            let (bfly, ssim2) = score_jxl(&buf, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
            let dbytes = (buf.len() as f64 - cjxl_bytes as f64) / cjxl_bytes as f64 * 100.0;
            let dssim2 = ssim2 - cjxl_ssim2;
            let sha = sha256_hex(&buf);
            eprintln!(
                "  {:<22} {} B  sha={} bfly={:.3} ssim2={:.2} ({} ms)  Δb={:+.2}% Δss2={:+.2}",
                mode,
                buf.len(),
                &sha[..8],
                bfly,
                ssim2,
                ms,
                dbytes,
                dssim2
            );
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}",
                image_id,
                effort,
                dist,
                class,
                mode,
                buf.len(),
                &sha[..8],
                bfly,
                ssim2,
                ms,
                dbytes,
                dssim2
            )
            .unwrap();
        }
    }

    eprintln!(
        "\n[bench W44-AUDIT-6 P2C] done in {:.1}s; TSV: {}",
        bench_start.elapsed().as_secs_f64(),
        out_path.display()
    );
}
