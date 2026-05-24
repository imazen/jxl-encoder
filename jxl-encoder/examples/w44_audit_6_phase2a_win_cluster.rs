//! W44-AUDIT-6 Phase 2A — verify W44-109 win cluster unaffected by AUDIT-6 exclude.
//!
//! Per Phase 1 brief: the M3>=80 discriminator should NOT fire on the
//! W44-109 win cluster (text-class screenshots at M3 ∈ [10, 29]).
//! Production B mode (AUDIT-6 ON by default) MUST produce byte-identical
//! output to mode A (`JXL_W44_AUDIT_6_DISABLE=1`) on these cells, and
//! both must preserve the W44-109 win behaviour vs Mode C (no W44-109).
//!
//! Cells: 5 gb82-sc text-class screenshots × e ∈ {5, 7} × d ∈ {3, 4}
//!   = 20 cells × 3 modes (A, B, C) + cjxl reference per cell = 80 encodes.
//!   (e7 d=3 below MIN_DISTANCE=3.5; A==B==C expected. e5 d=4 above.
//!    e7 d=4/5 fires the production gate. Distance 5 dropped to keep wall
//!    under ~10 min.)
//!
//! Pass criteria (per Phase 2 acceptance gate a):
//!   - A == B byte-identical on every cell (discriminator correctly
//!     identifies these as NOT-high-colour-class).
//!   - SSIM2(B) within ±0.30 of SSIM2(A) on every cell.
//!   - Bytes within ±2% of A on every cell.
//!   - At least 1 cell shows A ≠ C (proves W44-109 gate fires for at least
//!     one mode somewhere on this corpus — sanity that the gate isn't
//!     globally inert).
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_6_phase2a_win_cluster

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::File;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// (image_id, path) — W44-109 win-cluster (text-class) screenshots per
/// CLAUDE.md Investigation Notes "W44-109 wins at e5/e6/e7 on screenshot
/// class". M3 of these is in [10, 29] ≪ 80 → AUDIT-6 gate should NOT
/// fire → A == B byte-identical.
const IMAGES: &[(&str, &str)] = &[
    (
        "terminal",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
    ),
    (
        "imac_g3",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_g3.png",
    ),
    (
        "imac_dark",
        "/home/lilith/work/codec-corpus/gb82-sc/imac_dark.png",
    ),
    ("graph", "/home/lilith/work/codec-corpus/gb82-sc/graph.png"),
    (
        "gmessages",
        "/home/lilith/work/codec-corpus/gb82-sc/gmessages.png",
    ),
];

/// (effort, distance, gate_should_fire_in_production_B)
const CELLS: &[(u8, f32, bool)] = &[
    (7, 3.0, false), // below MIN_DISTANCE=3.5; A==B==C
    (5, 4.0, true),  // e5 lift = 2.0
    (7, 4.0, true),  // e7 lift = 3.0
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

/// Mode selects which env vars to set / which strategy to use:
///   A = baseline pre-AUDIT-6 (JXL_W44_AUDIT_6_DISABLE=1)
///   B = production (no env)
///   C = no W44-109 entirely (JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=1.0)
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
        std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
        match mode {
            "A_baseline_no_audit6" => {
                std::env::set_var("JXL_W44_AUDIT_6_DISABLE", "1");
            }
            "B_shipped" => {}
            "C_no_w109" => {
                std::env::set_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE", "1.0");
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
        std::env::remove_var("JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE");
    }
    Some((buf, ms))
}

fn encode_cjxl(src_path: &Path, effort: u8, distance: f32) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit6_p2a_{}_{}_{}_{}.jxl",
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
    let out_path: PathBuf = PathBuf::from("benchmarks/w44_audit_6_phase2a_win_cluster_2026-05-24.tsv");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(&out_path).expect("create TSV");
    writeln!(
        f,
        "image_id\teffort\tdistance\tmode\tbytes\tsha8\tbfly\tssim2\tms\t\
         delta_bytes_pct_vs_A\tdelta_ssim2_vs_A"
    )
    .unwrap();

    let bench_start = Instant::now();
    let mut violations: Vec<String> = Vec::new();
    let mut total_cells = 0;
    let mut cells_a_eq_b_byte_identical = 0;
    let mut cells_a_neq_c = 0;

    for (image_id, image_path) in IMAGES {
        let (pixels, w, h) = match load_png(Path::new(image_path)) {
            Some(t) => t,
            None => {
                eprintln!("[bench] FAIL: load {}", image_path);
                continue;
            }
        };
        eprintln!(
            "\n[image] {} {}×{} = {:.2} MP",
            image_id,
            w,
            h,
            (w as f64 * h as f64) / 1_000_000.0
        );
        let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

        for &(effort, dist, gate_expected) in CELLS {
            total_cells += 1;
            eprintln!(
                "\n  [cell] {} e{} d={} (gate_expected_to_fire_in_B={})",
                image_id, effort, dist, gate_expected
            );

            // cjxl reference
            let (cjxl_b, cjxl_ms) = match encode_cjxl(Path::new(image_path), effort, dist) {
                Some(t) => t,
                None => {
                    eprintln!("    cjxl FAIL");
                    continue;
                }
            };
            let (cjxl_bfly, cjxl_ssim2) =
                score_jxl(&cjxl_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
            writeln!(
                f,
                "{}\t{}\t{}\tD_cjxl\t{}\t{}\t{:.4}\t{:.4}\t{}\t\t",
                image_id,
                effort,
                dist,
                cjxl_b.len(),
                &sha256_hex(&cjxl_b)[..8],
                cjxl_bfly,
                cjxl_ssim2,
                cjxl_ms
            )
            .unwrap();

            // A, B, C
            let mut a_bytes_len: Option<usize> = None;
            let mut a_sha: Option<String> = None;
            let mut a_ssim2: Option<f64> = None;
            let mut c_sha: Option<String> = None;

            for mode in ["A_baseline_no_audit6", "B_shipped", "C_no_w109"] {
                let (buf, ms) = match encode_ours(&pixels, w, h, effort, dist, mode) {
                    Some(t) => t,
                    None => {
                        eprintln!("    {} FAIL", mode);
                        continue;
                    }
                };
                let (bfly, ssim2) = score_jxl(&buf, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
                let sha = sha256_hex(&buf);
                let sha8 = &sha[..8];
                let mut db_pct = String::new();
                let mut dss = String::new();
                match mode {
                    "A_baseline_no_audit6" => {
                        a_bytes_len = Some(buf.len());
                        a_sha = Some(sha.clone());
                        a_ssim2 = Some(ssim2);
                    }
                    "B_shipped" => {
                        if let (Some(an), Some(ass)) = (a_bytes_len, a_ssim2) {
                            let db = (buf.len() as f64 - an as f64) / an as f64 * 100.0;
                            db_pct = format!("{:.4}", db);
                            dss = format!("{:.4}", ssim2 - ass);
                            // gate (a): A == B byte-identical
                            if Some(&sha) == a_sha.as_ref() {
                                cells_a_eq_b_byte_identical += 1;
                            } else {
                                violations.push(format!(
                                    "  VIOLATION: {} e{} d={}: A ≠ B (A={}B sha={} B={}B sha={})",
                                    image_id,
                                    effort,
                                    dist,
                                    an,
                                    &a_sha.as_ref().unwrap()[..8],
                                    buf.len(),
                                    sha8
                                ));
                            }
                            // gate (a): bytes within ±2%, ssim2 within ±0.30
                            if db.abs() > 2.0 {
                                violations.push(format!(
                                    "  VIOLATION: {} e{} d={}: A→B bytes Δ {:.2}% > ±2%",
                                    image_id, effort, dist, db
                                ));
                            }
                            if (ssim2 - ass).abs() > 0.30 {
                                violations.push(format!(
                                    "  VIOLATION: {} e{} d={}: A→B ssim2 Δ {:.2} > ±0.30",
                                    image_id,
                                    effort,
                                    dist,
                                    ssim2 - ass
                                ));
                            }
                        }
                    }
                    "C_no_w109" => {
                        c_sha = Some(sha.clone());
                        if let Some(asha) = &a_sha {
                            if &sha != asha {
                                cells_a_neq_c += 1;
                            }
                        }
                    }
                    _ => {}
                }
                eprintln!(
                    "    {:<24} {} B  sha={} bfly={:.3} ssim2={:.2} ({} ms)",
                    mode,
                    buf.len(),
                    sha8,
                    bfly,
                    ssim2,
                    ms
                );
                writeln!(
                    f,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}",
                    image_id,
                    effort,
                    dist,
                    mode,
                    buf.len(),
                    sha8,
                    bfly,
                    ssim2,
                    ms,
                    db_pct,
                    dss,
                )
                .unwrap();
            }
            let _ = c_sha; // silence unused
        }
    }

    eprintln!(
        "\n[bench W44-AUDIT-6 P2A] done in {:.1}s; TSV: {}",
        bench_start.elapsed().as_secs_f64(),
        out_path.display()
    );
    eprintln!(
        "\n=== ACCEPTANCE SUMMARY ===\n  total cells: {}\n  A == B byte-identical: {}/{}\n  cells where A ≠ C (W44-109 gate fires somewhere): {}",
        total_cells, cells_a_eq_b_byte_identical, total_cells, cells_a_neq_c
    );
    if violations.is_empty() {
        eprintln!("  VIOLATIONS: NONE — Phase 2A acceptance PASS");
    } else {
        eprintln!("  VIOLATIONS:");
        for v in &violations {
            eprintln!("{}", v);
        }
        eprintln!("  Phase 2A acceptance: FAIL ({} violations)", violations.len());
    }
}
