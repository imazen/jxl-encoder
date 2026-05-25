//! W44-AUDIT-5 Phase 1 — diagnostic per-block AC strategy/quant dump for
//! codec_wiki e7 d=4 under EncoderStrategy::Libjxl, plus comparison-only
//! dumps for Zenjxl and (re-bench) cjxl SSIM2/bytes.
//!
//! Purpose: AUDIT-4 (`0e65668f`) showed that on `codec_wiki.png e7 d=4`:
//!   - cjxl                : 62,710 B  SSIM2 84.86
//!   - Libjxl-strategy ours: 70,217 B  SSIM2 79.35   (+12% bytes, -5.51 SSIM2)
//!   - Zenjxl-strategy ours: 90,323 B  SSIM2 84.89   (+44% bytes, +0.03 SSIM2)
//!
//! The +12pp gap of Libjxl-strategy vs cjxl is the "structural" wedge we
//! are diagnosing in W44-AUDIT-5. AUDIT-4 pre-narrowed 3 candidates:
//!   (1) per-block butteraugli measurement at e7
//!   (2) kFavor2X2AtHighQuality weight curve at d=4
//!   (3) strategy-search step_size for 32x32+ at d>=4
//!
//! This harness:
//!   (a) re-benches the 3 cells (cjxl / Libjxl / Zenjxl) to confirm the
//!       +5.51 SSIM2 deficit is reproducible.
//!   (b) dumps per-block (bx, by, strategy_wire, channel, nzeros, qac) for
//!       BOTH Libjxl and Zenjxl using the existing W44-76 dump infra (env
//!       var `JXL_W44_76_PER_BLOCK_DUMP=<dir>`).
//!   (c) computes per-strategy block-count histograms from each dump and
//!       emits a single TSV row per (strategy, channel) showing
//!       (libjxl_blocks, zenjxl_blocks, delta).
//!
//! No production code is modified. Output:
//!   benchmarks/w44_audit_5_phase1_per_block_dump_2026-05-24.tsv (histogram)
//!   /tmp/w44_audit_5_phase1_libjxl/per_block_ours.tsv         (raw dump)
//!   /tmp/w44_audit_5_phase1_zenjxl/per_block_ours.tsv         (raw dump)
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features "parallel butteraugli-loop ssim2-loop __expert" \
//!     --example w44_audit_5_phase1_per_block_dump

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use rgb::RGB;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const IMAGE_PATH: &str = "/home/lilith/work/codec-corpus/gb82-sc/codec_wiki.png";
const EFFORT: u8 = 7;
const DISTANCE: f32 = 4.0;

const LIBJXL_DUMP_DIR: &str = "/tmp/w44_audit_5_phase1_libjxl";
const ZENJXL_DUMP_DIR: &str = "/tmp/w44_audit_5_phase1_zenjxl";

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

fn encode_with_dump(
    pixels: &[u8],
    w: u32,
    h: u32,
    strategy: EncoderStrategy,
    dump_dir: &str,
) -> Option<(Vec<u8>, u128)> {
    // Clean dump dir.
    let _ = std::fs::remove_dir_all(dump_dir);
    let _ = std::fs::create_dir_all(dump_dir);
    // SAFETY: harness is sequential and ensures cleanup after the call.
    unsafe {
        std::env::set_var("JXL_W44_76_PER_BLOCK_DUMP", dump_dir);
    }
    let cfg = LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_strategy(strategy);
    let t0 = Instant::now();
    let buf = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(pixels)
        .ok();
    let ms = t0.elapsed().as_millis();
    unsafe {
        std::env::remove_var("JXL_W44_76_PER_BLOCK_DUMP");
    }
    buf.map(|b| (b, ms))
}

fn encode_cjxl(src_path: &Path) -> Option<(Vec<u8>, u128)> {
    let cjxl = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
    let tmp = std::env::temp_dir().join(format!(
        "cjxl_audit5p1_{}_{}_{}_{}.jxl",
        src_path.file_stem()?.to_string_lossy(),
        EFFORT,
        (DISTANCE * 1000.0) as u32,
        std::process::id()
    ));
    let t0 = Instant::now();
    let status = Command::new(cjxl)
        .arg(src_path)
        .arg(&tmp)
        .arg("-e")
        .arg(format!("{}", EFFORT))
        .arg("-d")
        .arg(format!("{}", DISTANCE))
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

/// Wire AC strategy codes used by libjxl bitstream (same as our
/// `STRATEGY_CODE_LUT` output). Names per `enc_ac_strategy.cc::AcStrategyType`.
fn strategy_wire_name(code: u8) -> &'static str {
    match code {
        0 => "DCT8",
        1 => "Hornuss",
        2 => "DCT2X2",
        3 => "DCT4X4",
        4 => "DCT16X16",
        5 => "DCT32X32",
        6 => "DCT16X8",
        7 => "DCT8X16",
        8 => "DCT32X8",
        9 => "DCT8X32",
        10 => "DCT32X16",
        11 => "DCT16X32",
        12 => "DCT4X8",
        13 => "DCT8X4",
        14 => "AFV0",
        15 => "AFV1",
        16 => "AFV2",
        17 => "AFV3",
        18 => "DCT64X64",
        19 => "DCT64X32",
        20 => "DCT32X64",
        21 => "DCT128X128",
        22 => "DCT128X64",
        23 => "DCT64X128",
        24 => "DCT256X256",
        25 => "DCT256X128",
        26 => "DCT128X256",
        27 => "IDENTITY",
        _ => "UNKNOWN",
    }
}

/// Read per-block dump file and return strategy-count histogram per channel.
/// Returns map: (channel, strategy_code) → (block_count, nzeros_sum, qac_sum)
fn read_dump(path: &Path) -> std::io::Result<BTreeMap<(usize, u8), (u64, u64, u64)>> {
    let mut histo: BTreeMap<(usize, u8), (u64, u64, u64)> = BTreeMap::new();
    let f = File::open(path)?;
    let r = BufReader::new(f);
    for line in r.lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') || line.starts_with("bx\t") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 6 {
            continue;
        }
        let strategy: u8 = match cols[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let channel: usize = match cols[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let nzeros: u64 = match cols[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let qac: u64 = match cols[5].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let e = histo.entry((channel, strategy)).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += nzeros;
        e.2 += qac;
    }
    Ok(histo)
}

fn main() {
    eprintln!(
        "[bench W44-AUDIT-5 Phase 1] codec_wiki e{} d={} per-block diagnostic",
        EFFORT, DISTANCE
    );

    let (pixels, w, h) = load_png(Path::new(IMAGE_PATH)).expect("load codec_wiki");
    eprintln!(
        "  loaded {}×{} = {:.2} MP",
        w,
        h,
        (w as f64 * h as f64) / 1_000_000.0
    );
    let (lin_img, srgb_img) = make_imgs(&pixels, w, h);

    // ── Re-bench (Step 1 — reproducibility) ──────────────────────────────
    let bench_start = Instant::now();
    let (cjxl_b, cjxl_ms) = encode_cjxl(Path::new(IMAGE_PATH)).expect("cjxl");
    let (cjxl_bfly, cjxl_ssim2) =
        score_jxl(&cjxl_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
    eprintln!(
        "  cjxl    = {:7} B  bfly={:.3} ssim2={:.2}  ({} ms)",
        cjxl_b.len(),
        cjxl_bfly,
        cjxl_ssim2,
        cjxl_ms
    );

    // Libjxl strategy WITH dump.
    let (lib_b, lib_ms) = encode_with_dump(&pixels, w, h, EncoderStrategy::Libjxl, LIBJXL_DUMP_DIR)
        .expect("encode libjxl");
    let (lib_bfly, lib_ssim2) = score_jxl(&lib_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
    let lib_dpct = (lib_b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0;
    eprintln!(
        "  libjxl  = {:7} B  bfly={:.3} ssim2={:.2}  ({} ms)  Δ={:+.2}% vs cjxl  ΔSSIM2={:+.2}",
        lib_b.len(),
        lib_bfly,
        lib_ssim2,
        lib_ms,
        lib_dpct,
        lib_ssim2 - cjxl_ssim2,
    );

    // Zenjxl strategy WITH dump.
    let (zen_b, zen_ms) = encode_with_dump(&pixels, w, h, EncoderStrategy::Zenjxl, ZENJXL_DUMP_DIR)
        .expect("encode zenjxl");
    let (zen_bfly, zen_ssim2) = score_jxl(&zen_b, &lin_img, &srgb_img, w, h).unwrap_or((0.0, 0.0));
    let zen_dpct = (zen_b.len() as f64 - cjxl_b.len() as f64) / cjxl_b.len() as f64 * 100.0;
    eprintln!(
        "  zenjxl  = {:7} B  bfly={:.3} ssim2={:.2}  ({} ms)  Δ={:+.2}% vs cjxl  ΔSSIM2={:+.2}",
        zen_b.len(),
        zen_bfly,
        zen_ssim2,
        zen_ms,
        zen_dpct,
        zen_ssim2 - cjxl_ssim2,
    );

    // ── Read per-block dumps and compute strategy histograms ─────────────
    let lib_dump = read_dump(Path::new(&format!(
        "{}/per_block_ours.tsv",
        LIBJXL_DUMP_DIR
    )))
    .expect("read libjxl dump");
    let zen_dump = read_dump(Path::new(&format!(
        "{}/per_block_ours.tsv",
        ZENJXL_DUMP_DIR
    )))
    .expect("read zenjxl dump");

    // Aggregate over channels for the overall strategy histogram, then also
    // emit per-channel rows so we can see Y/X/B independently.
    let all_strategies: std::collections::BTreeSet<(usize, u8)> =
        lib_dump.keys().chain(zen_dump.keys()).cloned().collect();

    let out_path: PathBuf =
        PathBuf::from("benchmarks/w44_audit_5_phase1_per_block_dump_2026-05-24.tsv");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(&out_path).expect("create TSV");
    writeln!(
        f,
        "# W44-AUDIT-5 Phase 1 — per-strategy block-count histogram diff"
    )
    .unwrap();
    writeln!(
        f,
        "# codec_wiki e{} d={} encoded with EncoderStrategy::Libjxl vs Zenjxl",
        EFFORT, DISTANCE
    )
    .unwrap();
    writeln!(
        f,
        "# cjxl   bytes={}  ssim2={:.2}",
        cjxl_b.len(),
        cjxl_ssim2
    )
    .unwrap();
    writeln!(
        f,
        "# libjxl bytes={}  ssim2={:.2}  ({:+.2}% vs cjxl, ΔSSIM2={:+.2})",
        lib_b.len(),
        lib_ssim2,
        lib_dpct,
        lib_ssim2 - cjxl_ssim2,
    )
    .unwrap();
    writeln!(
        f,
        "# zenjxl bytes={}  ssim2={:.2}  ({:+.2}% vs cjxl, ΔSSIM2={:+.2})",
        zen_b.len(),
        zen_ssim2,
        zen_dpct,
        zen_ssim2 - cjxl_ssim2,
    )
    .unwrap();
    writeln!(
        f,
        "channel\tstrategy_wire\tstrategy_name\tlibjxl_blocks\tzenjxl_blocks\tdelta_blocks\tlibjxl_mean_nzeros\tzenjxl_mean_nzeros\tlibjxl_mean_qac\tzenjxl_mean_qac"
    )
    .unwrap();

    let chan_name = |c: usize| match c {
        0 => "Y",
        1 => "X",
        2 => "B",
        _ => "?",
    };

    for (channel, strategy) in &all_strategies {
        let lib = lib_dump
            .get(&(*channel, *strategy))
            .cloned()
            .unwrap_or((0, 0, 0));
        let zen = zen_dump
            .get(&(*channel, *strategy))
            .cloned()
            .unwrap_or((0, 0, 0));
        let lib_mean_nz = if lib.0 > 0 {
            lib.1 as f64 / lib.0 as f64
        } else {
            0.0
        };
        let zen_mean_nz = if zen.0 > 0 {
            zen.1 as f64 / zen.0 as f64
        } else {
            0.0
        };
        let lib_mean_qac = if lib.0 > 0 {
            lib.2 as f64 / lib.0 as f64
        } else {
            0.0
        };
        let zen_mean_qac = if zen.0 > 0 {
            zen.2 as f64 / zen.0 as f64
        } else {
            0.0
        };
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{:+}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
            chan_name(*channel),
            strategy,
            strategy_wire_name(*strategy),
            lib.0,
            zen.0,
            (zen.0 as i64) - (lib.0 as i64),
            lib_mean_nz,
            zen_mean_nz,
            lib_mean_qac,
            zen_mean_qac,
        )
        .unwrap();
    }
    drop(f);

    // Also dump a "Y-channel only" condensed histogram on stderr.
    eprintln!("");
    eprintln!("Strategy histogram (Y channel only, sorted by libjxl block count desc):");
    let mut y_rows: Vec<(u8, u64, u64)> = all_strategies
        .iter()
        .filter(|(c, _)| *c == 0)
        .map(|(_, s)| {
            let lib_n = lib_dump.get(&(0, *s)).map(|v| v.0).unwrap_or(0);
            let zen_n = zen_dump.get(&(0, *s)).map(|v| v.0).unwrap_or(0);
            (*s, lib_n, zen_n)
        })
        .collect();
    y_rows.sort_by_key(|(_, l, _)| std::cmp::Reverse(*l));
    eprintln!(
        "  {:<12} {:>10} {:>10} {:>10}",
        "strategy", "libjxl", "zenjxl", "delta"
    );
    for (s, lib_n, zen_n) in &y_rows {
        eprintln!(
            "  {:<12} {:>10} {:>10} {:>+10}",
            strategy_wire_name(*s),
            lib_n,
            zen_n,
            (*zen_n as i64) - (*lib_n as i64),
        );
    }
    eprintln!(
        "[bench W44-AUDIT-5 Phase 1] total {:.1}s, output {}",
        bench_start.elapsed().as_secs_f64(),
        out_path.display()
    );
}
