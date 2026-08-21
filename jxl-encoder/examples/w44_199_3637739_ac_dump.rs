//! W44-199 Phase 1: 3637739 vs 1418519 per-block AC strategy + qac dump @ e7 d=4.
//!
//! W44-198 honest-stopped on the 3637739 cid22 high-d photo cluster: the
//! W44-91/96/166 discriminator + W44-29 entropy_mul lift pattern is
//! saturated — best-case -1.67% bytes vs +5-6% gap. This chunk tests
//! Candidate (b) from W44-198: AC strategy selection mechanism.
//!
//! Cells:
//!   3637739 e7 d=4 (worst-loser middle cell, mask_med=76 NOGATE class)
//!   1418519 e7 d=4 (WINNER baseline, mask_med=92)
//!
//! Approach (mirrors W44-103 terminal d=4 and W44-121 codec_wiki d=3):
//! 1. Encode both images with cjxl-rs (ours, Zenjxl default) AND cjxl
//!    (libjxl reference v0.12.0 with W44-76 dump patch at the debug
//!    worktree `libjxl--w44-76-per-block-debug-dump`).
//! 2. Decode each, compute global + 3x3-region SSIM2 / butteraugli.
//! 3. Per-block strategy dump on both sides via env var
//!    `JXL_W44_76_PER_BLOCK_DUMP=<dir>`.
//!
//! Diagnostic question:
//!   Does our strategy distribution on 3637739 diverge from cjxl in a
//!   way that DOESN'T happen on 1418519 (the WINNER baseline)?
//!
//! Output: benchmarks/w44_199_3637739_ac_dump_2026-05-22.tsv
//!         $HOME/tmp/w44_199_dumps_<label>/<image>_e7_d4_{ours,cjxl}/per_block_*.tsv
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_199_3637739_ac_dump
use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const IMG_3637739: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/3637739.png";
const IMG_1418519: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png";
const DISTANCE: f32 = 4.0;
const EFFORT: u8 = 7;

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    // Prefer the W44-76 debug worktree (has the per-block dump patch).
    let dbg = PathBuf::from(
        "/home/lilith/work/jxl-efforts/libjxl--w44-76-per-block-debug-dump/build/tools/cjxl",
    );
    if dbg.exists() {
        return dbg;
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}

// sRGB ↔ linear ─────────────────────────────────────────────────────────────

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

fn extract_region_lin(
    full: &Img<Vec<RGB<f32>>>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Img<Vec<RGB<f32>>> {
    let stride = full.width();
    let mut out = Vec::with_capacity(w * h);
    let buf = full.buf();
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            out.push(buf[y * stride + x]);
        }
    }
    Img::new(out, w, h)
}

fn extract_region_srgb(
    full: &Img<Vec<[u8; 3]>>,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
) -> Img<Vec<[u8; 3]>> {
    let stride = full.width();
    let mut out = Vec::with_capacity(w * h);
    let buf = full.buf();
    for y in y0..(y0 + h) {
        for x in x0..(x0 + w) {
            out.push(buf[y * stride + x]);
        }
    }
    Img::new(out, w, h)
}

fn score_region(
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    dec_lin: &Img<Vec<RGB<f32>>>,
    dec_srgb: &Img<Vec<[u8; 3]>>,
) -> Option<(f64, f64)> {
    let bfly = butteraugli_linear(
        orig_lin.as_ref(),
        dec_lin.as_ref(),
        &ButteraugliParams::default(),
    )
    .ok()?
    .score as f64;
    let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb.as_ref()).ok()?;
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
    dump_dir: Option<&Path>,
) -> Option<(Vec<u8>, f64)> {
    let tmp = scratch_root().join(format!(
        "w44_199_cjxl_{}_e{}_d{}.jxl",
        src_png.file_stem().unwrap().to_string_lossy(),
        effort,
        distance.to_string().replace('.', "_")
    ));
    let bin = cjxl_bin();
    let mut cmd = Command::new(&bin);
    cmd.arg(src_png)
        .arg(&tmp)
        .args(["-e", &effort.to_string()])
        .args(["-d", &distance.to_string()])
        .args(["--num_threads", "1"])
        .arg("--quiet");
    if let Some(dir) = dump_dir {
        cmd.env("JXL_W44_76_PER_BLOCK_DUMP", dir);
    }
    let start = Instant::now();
    let status = cmd.status().ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

struct CellResult {
    bytes: usize,
    encode_ms: f64,
    global_bfly: f64,
    global_ssim2: f64,
    region_ssim2: [[f64; 3]; 3],
    region_bfly: [[f64; 3]; 3],
}

fn measure(
    bytes: &[u8],
    encode_ms: f64,
    orig_lin: &Img<Vec<RGB<f32>>>,
    orig_srgb: &Img<Vec<[u8; 3]>>,
    w: u32,
    h: u32,
) -> Option<CellResult> {
    let (dw, dh, dec_lin) = decode_jxl_linear(bytes)?;
    if dw != w as usize || dh != h as usize {
        return None;
    }
    let dec_pixels: Vec<RGB<f32>> = dec_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
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

    let (gb, gs) = score_region(orig_lin, orig_srgb, &dec_lin_img, &dec_srgb_img)?;

    // 3x3 regions
    // Round down to multiple of 32 to stay aligned with DCT32 blocks.
    let region_w = (dw / 3) & !31;
    let region_h = (dh / 3) & !31;
    let mut region_ssim2 = [[0.0; 3]; 3];
    let mut region_bfly = [[0.0; 3]; 3];
    for ry in 0..3usize {
        for rx in 0..3usize {
            let x0 = rx * region_w;
            let y0 = ry * region_h;
            let rw = if rx == 2 { dw - x0 } else { region_w };
            let rh = if ry == 2 { dh - y0 } else { region_h };
            if rw < 64 || rh < 64 {
                continue;
            }
            let o_lin = extract_region_lin(orig_lin, x0, y0, rw, rh);
            let o_srgb = extract_region_srgb(orig_srgb, x0, y0, rw, rh);
            let d_lin = extract_region_lin(&dec_lin_img, x0, y0, rw, rh);
            let d_srgb = extract_region_srgb(&dec_srgb_img, x0, y0, rw, rh);
            if let Some((b, s)) = score_region(&o_lin, &o_srgb, &d_lin, &d_srgb) {
                region_bfly[ry][rx] = b;
                region_ssim2[ry][rx] = s;
            }
        }
    }

    Some(CellResult {
        bytes: bytes.len(),
        encode_ms,
        global_bfly: gb,
        global_ssim2: gs,
        region_ssim2,
        region_bfly,
    })
}

fn run_one(
    src_path: &Path,
    label: &str,
    tsv: &mut std::fs::File,
    dump_root: &Path,
    distance: f32,
    effort: u8,
) {
    let (rgb, w, h) = load_png(src_path).unwrap_or_else(|| panic!("load {}", src_path.display()));
    eprintln!("=== {} ({}x{}) ===", label, w, h);
    let orig_lin = rgb_to_linear_img(&rgb, w, h);
    let orig_srgb = rgb_to_srgb_arr3(&rgb, w, h);

    // OURS with W44-76 dump
    let ours_dump = dump_root.join(format!("{}_e{}_d{}_ours", label, effort, distance as i32));
    std::fs::create_dir_all(&ours_dump).unwrap();
    // SAFETY: single-threaded harness, child encode runs sequentially.
    // W44-76 value diff (2026-08-21): the W44-201 coeff dump in wildcard
    // mode (255/255) emits per_position_coeffs.tsv with the same 6-col
    // format as the instrumented cjxl's per_block_libjxl_coeffs.tsv.
    unsafe {
        std::env::set_var("JXL_W44_76_PER_BLOCK_DUMP", &ours_dump);
        std::env::set_var("JXL_W44_201_COEFFS_DUMP", &ours_dump);
        std::env::set_var("JXL_W44_201_COEFFS_STRATEGY", "255");
        std::env::set_var("JXL_W44_201_COEFFS_CHANNEL", "255");
    }
    let Some((ours_bytes, ours_ms)) = encode_ours(&rgb, w, h, distance, effort) else {
        eprintln!("  ours encode FAILED");
        return;
    };
    unsafe {
        std::env::remove_var("JXL_W44_76_PER_BLOCK_DUMP");
        std::env::remove_var("JXL_W44_201_COEFFS_DUMP");
        std::env::remove_var("JXL_W44_201_COEFFS_STRATEGY");
        std::env::remove_var("JXL_W44_201_COEFFS_CHANNEL");
    }
    let Some(ours_r) = measure(&ours_bytes, ours_ms, &orig_lin, &orig_srgb, w, h) else {
        eprintln!("  ours measure FAILED");
        return;
    };
    eprintln!(
        "  ours: bytes={} bfly={:.4} ssim2={:.4} ({:.0}ms)",
        ours_r.bytes, ours_r.global_bfly, ours_r.global_ssim2, ours_r.encode_ms
    );

    // CJXL with W44-76 dump
    let cjxl_dump = dump_root.join(format!("{}_e{}_d{}_cjxl", label, effort, distance as i32));
    std::fs::create_dir_all(&cjxl_dump).unwrap();
    let Some((cjxl_bytes, cjxl_ms)) = encode_cjxl(src_path, distance, effort, Some(&cjxl_dump))
    else {
        eprintln!("  cjxl encode FAILED");
        return;
    };
    let Some(cjxl_r) = measure(&cjxl_bytes, cjxl_ms, &orig_lin, &orig_srgb, w, h) else {
        eprintln!("  cjxl measure FAILED");
        return;
    };
    eprintln!(
        "  cjxl: bytes={} bfly={:.4} ssim2={:.4} ({:.0}ms)",
        cjxl_r.bytes, cjxl_r.global_bfly, cjxl_r.global_ssim2, cjxl_r.encode_ms
    );
    eprintln!(
        "  delta: bytes={:+.2}%  ssim2={:+.4}  bfly={:+.2}%",
        100.0 * (ours_r.bytes as f64 - cjxl_r.bytes as f64) / cjxl_r.bytes as f64,
        ours_r.global_ssim2 - cjxl_r.global_ssim2,
        100.0 * (ours_r.global_bfly - cjxl_r.global_bfly) / cjxl_r.global_bfly,
    );

    for (enc_label, r) in [("ours", &ours_r), ("cjxl", &cjxl_r)] {
        write!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.6}\t{:.4}",
            label, enc_label, effort, distance, r.bytes, r.encode_ms, r.global_bfly, r.global_ssim2
        )
        .unwrap();
        for ry in 0..3 {
            for rx in 0..3 {
                write!(tsv, "\t{:.4}", r.region_ssim2[ry][rx]).unwrap();
            }
        }
        for ry in 0..3 {
            for rx in 0..3 {
                write!(tsv, "\t{:.6}", r.region_bfly[ry][rx]).unwrap();
            }
        }
        writeln!(tsv).unwrap();
    }

    eprintln!("  per-region SSIM2 (ours - cjxl):");
    for ry in 0..3 {
        let row_top = if ry == 0 {
            "top"
        } else if ry == 1 {
            "mid"
        } else {
            "bot"
        };
        eprintln!(
            "    {}: {:+.3} {:+.3} {:+.3}",
            row_top,
            ours_r.region_ssim2[ry][0] - cjxl_r.region_ssim2[ry][0],
            ours_r.region_ssim2[ry][1] - cjxl_r.region_ssim2[ry][1],
            ours_r.region_ssim2[ry][2] - cjxl_r.region_ssim2[ry][2],
        );
    }
}

/// `$HOME/tmp` — the only sanctioned scratch root (`/tmp` is wiped
/// mid-session; CLAUDE.md "/tmp IS BANNED"). Falls back to the system
/// temp dir only when `$HOME` is unset.
fn scratch_root() -> PathBuf {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join("tmp"))
        .unwrap_or_else(std::env::temp_dir)
}

fn main() {
    // CLI override: `<image.png> <distance> <label> [effort]` dumps an
    // arbitrary cell to label-derived outputs (2026-08-21 mechanism-2
    // extension); no-arg default = the historical W44-199 pair.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = PathBuf::from("/home/lilith/work/zen/jxl-encoder/benchmarks");
    let (tsv_path, meta_path, dump_root);
    if args.len() >= 3 {
        let label = args[2].to_ascii_lowercase();
        tsv_path = out_dir.join(format!("w44_199_ac_dump_{label}.tsv"));
        meta_path = out_dir.join(format!("w44_199_ac_dump_{label}.meta"));
        dump_root = scratch_root().join(format!("w44_199_dumps_{label}"));
    } else {
        tsv_path = out_dir.join("w44_199_3637739_ac_dump_2026-05-22.tsv");
        meta_path = out_dir.join("w44_199_3637739_ac_dump_2026-05-22.meta");
        dump_root = scratch_root().join("w44_199_dumps");
    }
    std::fs::create_dir_all(&dump_root).expect("mkdir dumps");

    let mut tsv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tsv_path)
        .expect("open tsv");
    writeln!(
        tsv,
        "image\tencoder\teffort\tdistance\tbytes\tencode_ms\tglobal_bfly\tglobal_ssim2\t\
         r00_ssim2\tr01_ssim2\tr02_ssim2\tr10_ssim2\tr11_ssim2\tr12_ssim2\tr20_ssim2\tr21_ssim2\tr22_ssim2\t\
         r00_bfly\tr01_bfly\tr02_bfly\tr10_bfly\tr11_bfly\tr12_bfly\tr20_bfly\tr21_bfly\tr22_bfly"
    )
    .unwrap();

    let cases: Vec<(String, String, f32, u8)> = if args.len() >= 3 {
        let d: f32 = args[1].parse().expect("distance f32");
        let e: u8 = args
            .get(3)
            .map_or(EFFORT, |v| v.parse().expect("effort u8"));
        vec![(args[2].clone(), args[0].clone(), d, e)]
    } else {
        vec![
            ("3637739_LOSER".into(), IMG_3637739.into(), DISTANCE, EFFORT),
            (
                "1418519_WINNER".into(),
                IMG_1418519.into(),
                DISTANCE,
                EFFORT,
            ),
        ]
    };
    for (label, path, d, e) in &cases {
        run_one(Path::new(path), label, &mut tsv, &dump_root, *d, *e);
    }

    eprintln!();
    eprintln!("Wrote {}", tsv_path.display());

    let meta = format!(
        "# W44-199 Phase 1: 3637739 vs 1418519 per-block AC strategy + qac dump
# Bench: encode each image at d={DISTANCE} effort {EFFORT} with cjxl-rs (Zenjxl default)
# vs cjxl (libjxl reference v0.12.0, W44-76 dump patch).
# Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
#
# Predecessors:
#   W44-198 honest-stop: Phase-1 probe of 28 features across 41 CID22 ruled out
#     discriminator-based fix. W44-29 entropy_mul lift saturated at -1.67%% best.
#   W44-103/121: per-region SSIM2 attribution dumps for terminal d=4, codec_wiki d=3.
#   W44-76: per-block strategy dump infrastructure (both ours-side module
#     w44_76_dump.rs + libjxl-side patch w44_76_libjxl_per_block_dump_2026-05-19.patch
#     applied to /home/lilith/work/jxl-efforts/libjxl--w44-76-per-block-debug-dump/).
#
# Diagnostic question (Candidate b from W44-198):
#   Does our AC strategy distribution on 3637739 diverge from cjxl in a way
#   that DOESN'T happen on 1418519 (the WINNER baseline)?
#
# Hypotheses:
#   H1: Strategy mismatch (e.g., we over-pick DCT16X16 / DCT8 where cjxl picks
#       DCT32X32). Fix path: cost-model tweak via per-block content axis.
#   H2: Strategy match, per-block pixel work differs (CfL / buttloop / recon).
#       Honest-stop and propose Candidate (c) buttloop next.
#
# Per-block strategy dumps (libjxl-wire space, both sides):
#   $HOME/tmp/w44_199_dumps/3637739_LOSER_e{EFFORT}_d{distance_i}_ours/per_block_ours.tsv
#   $HOME/tmp/w44_199_dumps/3637739_LOSER_e{EFFORT}_d{distance_i}_cjxl/per_block_libjxl.tsv
#   $HOME/tmp/w44_199_dumps/1418519_WINNER_e{EFFORT}_d{distance_i}_ours/per_block_ours.tsv
#   $HOME/tmp/w44_199_dumps/1418519_WINNER_e{EFFORT}_d{distance_i}_cjxl/per_block_libjxl.tsv
#
# 3x3 spatial region grid (region indices (ry, rx) where (0,0)=top-left):
#   top: r00 r01 r02
#   mid: r10 r11 r12
#   bot: r20 r21 r22
# Each region is `(w/3) & !31` x `(h/3) & !31` aligned to DCT32 blocks.
#
# Reproducer:
#   cargo run -p jxl-encoder --release --features 'parallel butteraugli-loop ssim2-loop' \\
#     --example w44_199_3637739_ac_dump
",
        DISTANCE = DISTANCE,
        EFFORT = EFFORT,
        distance_i = DISTANCE as i32,
    );
    std::fs::write(&meta_path, meta).unwrap();
    eprintln!("Wrote {}", meta_path.display());
}
