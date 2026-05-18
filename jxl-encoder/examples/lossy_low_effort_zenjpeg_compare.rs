//! W38: per-phase wall-clock baseline at e2..=e5 + zenjpeg cross-codec comparator.
//!
//! Extends W36-1's `lossy_phase_baseline` example downward in effort space and
//! adds a zenjpeg `HybridProgressive` comparator for the same 8 images at
//! qualities that approximate the JXL distance grid {0.5, 1.0, 2.0, 4.0}.
//!
//! Produces two TSVs:
//!
//! 1. `benchmarks/lossy_phase_baseline_low_effort_2026-05-19.tsv` — jxl-encoder
//!    per-phase breakdown at e2/e3/e4/e5, same format as W36-1 (extends downward).
//! 2. `benchmarks/lossy_phase_low_effort_with_zenjpeg_2026-05-19.tsv` — cross-codec
//!    table: codec, image, effort_or_quality, distance/quality, bytes, wall_ms,
//!    butteraugli, ssim2.
//!
//! Phase timings: reuses the `__JXL_ENC_PHASE_TIMING` env-var path from W36-1
//! (`encode_inner` + `encode_two_pass_to_writer` markers).
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel butteraugli-loop' \
//!     --example lossy_low_effort_zenjpeg_compare -- \
//!     --out-jxl benchmarks/lossy_phase_baseline_low_effort_2026-05-19.tsv \
//!     --out-cross benchmarks/lossy_phase_low_effort_with_zenjpeg_2026-05-19.tsv

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use jxl_encoder::{LossyConfig, PixelLayout};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout as ZjPixelLayout, Quality,
};

const PHASES: &[&str] = &[
    "total",
    "xyb",
    "patches",
    "splines",
    "quant_field",
    "gaborish",
    "cfl1",
    "acstrat",
    "cfl2",
    "buttloop",
    "xform",
    "sharp",
    "entropy",
    "tok_dc",
    "co",
    "bcm",
    "ac_tok",
    "lz77",
    "build_codes",
    "pass2_write",
];

// JXL distance -> zenjpeg jpegli-approx-quality. Aligned with the rough mapping
// the task spec gives (d=0.5≈q95, d=1.0≈q85, d=2.0≈q75, d=4.0≈q60). zenjpeg's
// `ApproxJpegli` scale is jpegli native quality (calibrated against SSIMULACRA2);
// we ALSO measure butteraugli on the resulting JPEG so the cross-codec table is
// honest even if the rough mapping drifts.
const DIST_TO_JPEG_Q: &[(f32, f32)] = &[(0.5, 95.0), (1.0, 85.0), (2.0, 75.0), (4.0, 60.0)];

fn main() {
    // Worker subprocess for per-encode phase timing (mirrors W36-1's pattern).
    if std::env::var("__JXL_PHASE_WORKER").is_ok() {
        let img_path = std::env::var("__W_IMG").unwrap();
        let effort: u8 = std::env::var("__W_EFFORT").unwrap().parse().unwrap();
        let distance: f32 = std::env::var("__W_DIST").unwrap().parse().unwrap();
        worker_jxl_encode(&img_path, effort, distance);
        return;
    }

    // Parse args.
    let args: Vec<String> = std::env::args().collect();
    let mut out_jxl = PathBuf::from("benchmarks/lossy_phase_baseline_low_effort_2026-05-19.tsv");
    let mut out_cross =
        PathBuf::from("benchmarks/lossy_phase_low_effort_with_zenjpeg_2026-05-19.tsv");
    let mut runs_per_cell = 3usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out-jxl" if i + 1 < args.len() => {
                out_jxl = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out-cross" if i + 1 < args.len() => {
                out_cross = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--runs" if i + 1 < args.len() => {
                runs_per_cell = args[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let smoke = std::env::var("__W38_SMOKE").is_ok();
    let efforts: &[u8] = if smoke { &[5] } else { &[2, 3, 4, 5] };
    let distances: &[f32] = if smoke { &[1.0] } else { &[0.5, 1.0, 2.0, 4.0] };

    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    let cid_dir = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let scrn_dir = format!("{}/work/codec-corpus/gb82-sc", base);

    // Same 8 images as W36-1 — preserves apples-to-apples comparison.
    let (cid_names, scrn_names): (&[&str], &[&str]) = if smoke {
        (&["1025469.png"], &[])
    } else {
        (
            &[
                "1025469.png",
                "1044329.png",
                "1189261.png",
                "1279330.png",
                "1418519.png",
            ],
            &["codec_wiki.png", "imac_g3.png", "terminal.png"],
        )
    };

    let mut images: Vec<(String, String)> = Vec::new();
    for n in cid_names {
        images.push((format!("cid22/{}", n), format!("{}/{}", cid_dir, n)));
    }
    for n in scrn_names {
        images.push((format!("scrn/{}", n), format!("{}/{}", scrn_dir, n)));
    }

    eprintln!(
        "Corpus: {} images, JXL: {} efforts × {} dist × {} runs = {} encodes, zenjpeg: {} qualities × {} runs = {} encodes",
        images.len(),
        efforts.len(),
        distances.len(),
        runs_per_cell,
        images.len() * efforts.len() * distances.len() * runs_per_cell,
        DIST_TO_JPEG_Q.len(),
        runs_per_cell,
        images.len() * DIST_TO_JPEG_Q.len() * runs_per_cell,
    );

    // -- Phase 1: JXL per-phase sweep, e2..=e5 ----------------------------
    let mut jxl_rows: Vec<String> = Vec::new();
    let mut header_jxl = String::from("image\teffort\tdistance\twidth\theight\tbytes");
    for p in PHASES {
        header_jxl.push('\t');
        header_jxl.push_str(&format!("ms_{}", p));
    }
    jxl_rows.push(header_jxl);

    let bin_path = std::env::current_exe().unwrap();
    eprintln!("JXL subprocess binary: {}", bin_path.display());

    let total_jxl_cells = images.len() * efforts.len() * distances.len();
    let mut jxl_cell_idx = 0usize;
    let t_jxl_start = Instant::now();

    // Cache per-image dimensions + reference images for metric computation.
    type RefBundle = (
        u32,
        u32,
        Vec<u8>,
        imgref::ImgVec<rgb::RGB<f32>>,
        imgref::ImgVec<[u8; 3]>,
    );
    let mut img_meta: std::collections::HashMap<String, RefBundle> =
        std::collections::HashMap::new();

    for (img_label, img_path) in &images {
        // Pre-load to get dims + reference images (linear-f32 RGB for butteraugli,
        // sRGB u8 RGB for ssim2). Mirrors the proven pattern in
        // examples/adaptive_gaborish_wider_corpus.rs.
        let img = image::open(img_path).expect("open");
        let rgb8 = img.to_rgb8();
        let (w, h) = (rgb8.width(), rgb8.height());
        let raw = rgb8.as_raw().clone();
        let (ref_lin, ref_srgb) = build_reference(&raw, w as usize, h as usize);
        img_meta.insert(img_label.clone(), (w, h, raw, ref_lin, ref_srgb));

        for &effort in efforts {
            for &distance in distances {
                jxl_cell_idx += 1;
                eprintln!(
                    "[jxl {}/{}] {} e{} d={} ({}x{})",
                    jxl_cell_idx, total_jxl_cells, img_label, effort, distance, w, h
                );
                let mut runs: Vec<PhaseResult> = Vec::with_capacity(runs_per_cell);
                for run in 0..runs_per_cell {
                    match subprocess_jxl(&bin_path, img_path, effort, distance) {
                        Ok(pr) => runs.push(pr),
                        Err(e) => eprintln!("  run {} ERR: {}", run, e),
                    }
                }
                if runs.is_empty() {
                    eprintln!("  all runs failed");
                    continue;
                }
                let med = median_phases(&runs);
                let bytes_med = median_bytes(&runs);
                let mut row = format!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    img_label, effort, distance, w, h, bytes_med
                );
                for p in PHASES {
                    let v = med.get(*p).copied().unwrap_or(0.0);
                    row.push('\t');
                    row.push_str(&format!("{:.2}", v));
                }
                jxl_rows.push(row);
            }
        }

        // Periodic flush (every 4 images = 64 cells).
        write_atomic(&out_jxl, &jxl_rows);
    }
    let jxl_dur = t_jxl_start.elapsed();
    eprintln!("JXL phase sweep done in {:.1}s", jxl_dur.as_secs_f64());
    write_atomic(&out_jxl, &jxl_rows);

    // -- Phase 2: cross-codec table (jxl e2..e7 + zenjpeg per quality) ------
    //
    // For each (image, distance, codec_config), produce one row with:
    //   image, codec, effort_or_quality, distance, bytes, wall_ms, butteraugli, ssim2
    //
    // jxl-encoder cells: e2..e7 (we re-bench e6/e7 in-process for clean wall_ms
    // — the W36-1 subprocess wall_ms includes the 5-10 ms subprocess fork tax,
    // skewing low-effort comparison).
    // zenjpeg cells: HybridProgressive at q60/75/85/95 mapped to d∈{4,2,1,0.5}.

    let cross_efforts: &[u8] = &[2, 3, 4, 5, 6, 7];
    let mut cross_rows: Vec<String> = Vec::new();
    cross_rows.push(
        "image\tcodec\teffort_or_q\tdistance\twidth\theight\tbytes\twall_ms\tbutteraugli\tssim2"
            .to_string(),
    );

    let t_cross_start = Instant::now();
    let total_cross_cells =
        images.len() * cross_efforts.len() * distances.len() + images.len() * DIST_TO_JPEG_Q.len();
    let mut cross_idx = 0usize;

    for (img_label, _img_path) in &images {
        let (w, h, rgb8, ref_lin, ref_srgb) = img_meta.get(img_label).expect("img cached").clone();
        let w = w;
        let h = h;

        // jxl cells, in-process
        for &effort in cross_efforts {
            for &distance in distances {
                cross_idx += 1;
                eprintln!(
                    "[cross {}/{}] {} jxl e{} d={}",
                    cross_idx, total_cross_cells, img_label, effort, distance
                );
                let mut wall_runs = Vec::with_capacity(runs_per_cell);
                let mut bytes_opt: Option<Vec<u8>> = None;
                for _ in 0..runs_per_cell {
                    let cfg = LossyConfig::new(distance).with_effort(effort);
                    let t0 = Instant::now();
                    let bytes = cfg.encode(&rgb8, w, h, PixelLayout::Rgb8).expect("jxl");
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    wall_runs.push(ms);
                    bytes_opt = Some(bytes);
                }
                wall_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let wall_med = wall_runs[wall_runs.len() / 2];
                let bytes = bytes_opt.unwrap();
                let (bfly, ss2) = decode_jxl_and_measure(&bytes, w, h, &ref_lin, &ref_srgb);
                cross_rows.push(format!(
                    "{}\tjxl\te{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.5}\t{:.4}",
                    img_label,
                    effort,
                    distance,
                    w,
                    h,
                    bytes.len(),
                    wall_med,
                    bfly,
                    ss2
                ));
            }
        }

        // zenjpeg cells, in-process
        for &(distance, q) in DIST_TO_JPEG_Q {
            cross_idx += 1;
            eprintln!(
                "[cross {}/{}] {} zenjpeg q={} (d≈{})",
                cross_idx, total_cross_cells, img_label, q, distance
            );
            let mut wall_runs = Vec::with_capacity(runs_per_cell);
            let mut bytes_opt: Option<Vec<u8>> = None;
            for _ in 0..runs_per_cell {
                let cfg =
                    EncoderConfig::ycbcr(Quality::ApproxJpegli(q), ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::HybridProgressive);
                let t0 = Instant::now();
                let bytes = cfg
                    .encode_bytes(&rgb8, w, h, ZjPixelLayout::Rgb8Srgb)
                    .expect("zenjpeg");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                wall_runs.push(ms);
                bytes_opt = Some(bytes);
            }
            wall_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let wall_med = wall_runs[wall_runs.len() / 2];
            let bytes = bytes_opt.unwrap();
            let (bfly, ss2) = decode_zenjpeg_and_measure(&bytes, w, h, &ref_lin, &ref_srgb);
            cross_rows.push(format!(
                "{}\tzenjpeg_hybrid\tq{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.5}\t{:.4}",
                img_label,
                q as u32,
                distance,
                w,
                h,
                bytes.len(),
                wall_med,
                bfly,
                ss2
            ));
        }

        // Periodic flush per image.
        write_atomic(&out_cross, &cross_rows);
    }
    let cross_dur = t_cross_start.elapsed();
    eprintln!("Cross-codec sweep done in {:.1}s", cross_dur.as_secs_f64());
    write_atomic(&out_cross, &cross_rows);

    eprintln!(
        "Sweep total wall: jxl={:.1}s cross={:.1}s",
        jxl_dur.as_secs_f64(),
        cross_dur.as_secs_f64()
    );
    eprintln!("Wrote {}", out_jxl.display());
    eprintln!("Wrote {}", out_cross.display());
}

// --- Worker subprocess: single jxl encode with phase timing ---------------

fn worker_jxl_encode(img_path: &str, effort: u8, distance: f32) {
    let img = match image::open(img_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("WORKER_ERR: {}", e);
            std::process::exit(1);
        }
    };
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    // SAFETY: worker subprocess, single-threaded at this point.
    unsafe {
        std::env::set_var("__JXL_ENC_PHASE_TIMING", "1");
    }
    let cfg = LossyConfig::new(distance).with_effort(effort);
    let t0 = Instant::now();
    let bytes = match cfg.encode(rgb.as_raw(), w, h, PixelLayout::Rgb8) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WORKER_ENCODE_ERR: {}", e);
            std::process::exit(2);
        }
    };
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("BYTES {}", bytes.len());
    println!("TOTAL_MS {:.2}", total_ms);
}

// --- Subprocess driver: capture stdout + stderr phase lines ----------------

#[derive(Default, Clone)]
struct PhaseResult {
    bytes: usize,
    phases: std::collections::HashMap<String, f32>,
}

fn subprocess_jxl(
    bin_path: &std::path::Path,
    img_path: &str,
    effort: u8,
    distance: f32,
) -> Result<PhaseResult, String> {
    let out = Command::new(bin_path)
        .env("__JXL_PHASE_WORKER", "1")
        .env("__W_IMG", img_path)
        .env("__W_EFFORT", effort.to_string())
        .env("__W_DIST", distance.to_string())
        .env("__JXL_ENC_PHASE_TIMING", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawn: {}", e))?;
    if !out.status.success() {
        return Err(format!(
            "exit {}: stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut pr = PhaseResult::default();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("BYTES ") {
            pr.bytes = rest.trim().parse().unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("TOTAL_MS ") {
            pr.phases
                .insert("total".into(), rest.trim().parse().unwrap_or(0.0));
        }
    }
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("encode_inner:") {
            parse_kv(&mut pr, rest);
        } else if let Some(rest) = line.strip_prefix("encode_two_pass:") {
            parse_kv(&mut pr, rest);
        }
    }
    Ok(pr)
}

fn parse_kv(pr: &mut PhaseResult, line: &str) {
    for tok in line.split_whitespace() {
        let tok = tok.trim_end_matches(',');
        if let Some((k, v)) = tok.split_once('=') {
            let v = v.trim_end_matches("ms");
            if let Ok(parsed) = v.parse::<f32>() {
                if k == "total" && pr.phases.contains_key("total") {
                    continue;
                }
                pr.phases.insert(k.to_string(), parsed);
            }
        }
    }
}

fn median_bytes(runs: &[PhaseResult]) -> usize {
    let mut v: Vec<usize> = runs.iter().map(|r| r.bytes).collect();
    v.sort();
    v[v.len() / 2]
}

fn median_phases(runs: &[PhaseResult]) -> std::collections::HashMap<String, f32> {
    let mut out = std::collections::HashMap::new();
    for p in PHASES {
        let mut vals: Vec<f32> = runs
            .iter()
            .filter_map(|r| r.phases.get(*p).copied())
            .collect();
        if vals.is_empty() {
            continue;
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = vals[vals.len() / 2];
        out.insert((*p).to_string(), m);
    }
    out
}

// --- Decode + metric measurement (both codecs) -----------------------------
//
// Pattern mirrors `examples/adaptive_gaborish_wider_corpus.rs`: butteraugli
// runs on linear sRGB f32 RGB pixels via `butteraugli_linear`; SSIMULACRA2
// runs on sRGB u8 RGB via `fast_ssim2::compute_ssimulacra2`. The reference
// image is constructed once per source (linear f32 + sRGB u8) and reused
// across all codec/effort/distance cells for that image.

#[allow(clippy::manual_range_contains)]
fn srgb_u8_to_linear_f32(byte: u8) -> f32 {
    let x = byte as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
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

/// Build a reference (linear f32 ImgVec, sRGB u8 ImgVec) pair from the source
/// PNG's u8 sRGB pixels. Stored once per source image.
fn build_reference(
    rgb_u8: &[u8],
    w: usize,
    h: usize,
) -> (imgref::ImgVec<rgb::RGB<f32>>, imgref::ImgVec<[u8; 3]>) {
    use rgb::RGB;
    let lin_pixels: Vec<RGB<f32>> = rgb_u8
        .chunks_exact(3)
        .map(|c| {
            RGB::new(
                srgb_u8_to_linear_f32(c[0]),
                srgb_u8_to_linear_f32(c[1]),
                srgb_u8_to_linear_f32(c[2]),
            )
        })
        .collect();
    let srgb_pixels: Vec<[u8; 3]> = rgb_u8.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    (
        imgref::ImgVec::new(lin_pixels, w, h),
        imgref::ImgVec::new(srgb_pixels, w, h),
    )
}

fn decode_jxl_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    let reader = std::io::Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).ok()?;
    let fb = render.image_all_channels();
    Some((fb.width(), fb.height(), fb.buf().to_vec()))
}

/// Decode a JPEG bytestream via zenjpeg → linear-srgb f32 RGB triples.
fn decode_zenjpeg_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    use zenjpeg::decoder::{Decoder, OutputTarget};
    let dec = Decoder::new().output_target(OutputTarget::SrgbF32);
    let result = dec.decode(bytes, enough::Unstoppable).ok()?;
    let (dw, dh) = result.dimensions();
    let w = dw as usize;
    let h = dh as usize;
    let _ = (w, h); // dimensions returned separately; we use them via the tuple below.
    let srgb_f32 = result.into_pixels_f32()?;
    // zenjpeg returns nonlinear sRGB f32 ∈ [0,1] in `SrgbF32` mode. Apply
    // sRGB OETF^-1 to align with the reference image's linear-f32 domain.
    let lin: Vec<f32> = srgb_f32
        .iter()
        .map(|&x| {
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    Some((w, h, lin))
}

fn measure_quality(
    decoded_linear_rgb_f32: &[f32],
    dw: usize,
    dh: usize,
    ref_linear: &imgref::ImgVec<rgb::RGB<f32>>,
    ref_srgb: &imgref::ImgVec<[u8; 3]>,
) -> (f64, f64) {
    use butteraugli::{ButteraugliParams, butteraugli_linear};
    use imgref::Img;
    use rgb::RGB;
    let dec_pixels: Vec<RGB<f32>> = decoded_linear_rgb_f32
        .chunks_exact(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dec_linear_img = Img::new(dec_pixels, dw, dh);
    let bparams = ButteraugliParams::default();
    let bfly = butteraugli_linear(ref_linear.as_ref(), dec_linear_img.as_ref(), &bparams)
        .map(|r| r.score)
        .unwrap_or(f64::NAN);
    let dec_srgb: Vec<[u8; 3]> = decoded_linear_rgb_f32
        .chunks_exact(3)
        .map(|c| {
            [
                linear_to_srgb_u8(c[0]),
                linear_to_srgb_u8(c[1]),
                linear_to_srgb_u8(c[2]),
            ]
        })
        .collect();
    let dec_srgb_img = Img::new(dec_srgb, dw, dh);
    let ss2 = fast_ssim2::compute_ssimulacra2(ref_srgb.as_ref(), dec_srgb_img.as_ref())
        .unwrap_or(f64::NAN);
    (bfly, ss2)
}

fn decode_jxl_and_measure(
    bytes: &[u8],
    w: u32,
    h: u32,
    ref_linear: &imgref::ImgVec<rgb::RGB<f32>>,
    ref_srgb: &imgref::ImgVec<[u8; 3]>,
) -> (f64, f64) {
    let Some((dw, dh, dec)) = decode_jxl_linear(bytes) else {
        return (f64::NAN, f64::NAN);
    };
    if dw as u32 != w || dh as u32 != h {
        return (f64::NAN, f64::NAN);
    }
    measure_quality(&dec, dw, dh, ref_linear, ref_srgb)
}

fn decode_zenjpeg_and_measure(
    bytes: &[u8],
    w: u32,
    h: u32,
    ref_linear: &imgref::ImgVec<rgb::RGB<f32>>,
    ref_srgb: &imgref::ImgVec<[u8; 3]>,
) -> (f64, f64) {
    let Some((dw, dh, dec)) = decode_zenjpeg_linear(bytes) else {
        return (f64::NAN, f64::NAN);
    };
    if dw as u32 != w || dh as u32 != h {
        return (f64::NAN, f64::NAN);
    }
    measure_quality(&dec, dw, dh, ref_linear, ref_srgb)
}

// --- Atomic file write -----------------------------------------------------

fn write_atomic(out_path: &std::path::Path, rows: &[String]) {
    let tmp = format!(
        "/tmp/lossy_low_effort_{}_{}.tsv.partial",
        std::process::id(),
        out_path.file_name().unwrap().to_string_lossy()
    );
    {
        let mut f = fs::File::create(&tmp).expect("create tmp");
        for r in rows {
            writeln!(f, "{}", r).unwrap();
        }
    }
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::rename(&tmp, out_path).expect("rename");
}
