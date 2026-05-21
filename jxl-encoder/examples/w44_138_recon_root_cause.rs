// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-138 — Phase 1 root-cause investigation: buttloop recon vs decoder
//! render divergence on representative cells, post W44-117/118/120 chain.
//!
//! Builds on W44-116's per-step XYB capture infrastructure (already shipped).
//! Adds:
//!
//!   1. **Per-block AC-strategy spatial attribution**: for each 8x8 block,
//!      compute the max-abs diff between the encoder's `after_recon_xyb`
//!      and jxl-rs's decoded XYB (back-transformed from sRGB). Aggregate
//!      by raw_strategy to identify WHICH strategy "owns" the divergence.
//!
//!   2. **W44-117 firing detection**: per-cell, compare with and without
//!      `JXL_W44_117_DISABLE=1` to determine whether the W44-117 EPF seed
//!      actually fires (W44-118 gated it on `is_screenshot`, so most
//!      photos never trigger it).
//!
//!   3. **Step monotonicity check**: per-stage `step_div(N)` should
//!      monotonically decrease (each step is closer to decoder output).
//!      Steps that INCREASE `step_div` are the bug — report them.
//!
//!   4. **Stage-divergent flag**: identify the FIRST stage where
//!      step_div fails to monotonically decrease. That stage is the
//!      Phase 2 fix target.
//!
//! Background (must read in order):
//!   - W44-111: butteraugli metric divergence ruled out
//!   - W44-112: SetQuantField/parallel AdjustQuantBlockAC ruled out
//!   - W44-113: read-only audit of reconstruct_xyb vs decoder
//!   - W44-114: AFV IDCT bit-parity verified
//!   - W44-115: all per-strategy IDCT bit-parity verified
//!   - W44-116: per-step XYB capture shipped, EPF identified as
//!     residual mismatch source on photos
//!   - W44-117: EPF sharpness seed at iter-0 (one-shot fix)
//!   - W44-118: W44-117 gated to `is_screenshot` only (photo regression)
//!   - W44-119: chain CANNOT be retired even with W44-117 active
//!     (W44-117 + chain are orthogonal additive corrections)
//!   - W44-120: W44-117 only fires at d>=1.0 (regression at d<1)
//!
//! ## Run
//!
//! ```bash
//! CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release \
//!     --example w44_138_recon_root_cause \
//!     --features '__internal_recon_hook butteraugli-loop' \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > /tmp/w44_138_recon_root_cause.tsv
//! ```

#[cfg(not(all(feature = "__internal_recon_hook", feature = "butteraugli-loop")))]
fn main() {
    eprintln!(
        "w44_138_recon_root_cause requires --features '__internal_recon_hook butteraugli-loop'."
    );
    std::process::exit(2);
}

#[cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]
fn main() {
    inner::run();
}

#[cfg(all(feature = "__internal_recon_hook", feature = "butteraugli-loop"))]
mod inner {
    use jxl_encoder::vardct::__recon_hook;
    use jxl_encoder::{LossyConfig, PixelLayout};
    use std::path::PathBuf;

    const STRATEGY_NAMES: [&str; 16] = [
        "DCT8",      // 0
        "DCT16X8",   // 1
        "DCT8X16",   // 2
        "DCT16X16",  // 3
        "DCT32X32",  // 4
        "DCT4X8",    // 5
        "DCT8X4",    // 6
        "DCT4X4",    // 7
        "IDENTITY",  // 8
        "DCT2X2",    // 9
        "DCT32X16",  // 10
        "DCT16X32",  // 11
        "AFV0",      // 12
        "AFV1",      // 13
        "AFV2",      // 14
        "AFV3",      // 15
    ];

    /// sRGB normalized [0,1] f32 → linear light.
    fn srgb_to_linear_val(c: f32) -> f32 {
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn decode_jxl_rs(data: &[u8]) -> (usize, usize, Vec<f32>) {
        use jxl::api::{
            JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
            JxlPixelFormat, ProcessingResult, states,
        };
        use jxl::image::{Image, Rect};

        let mut input = data;
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("jxl-rs: EOF during header");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs header decode error: {:?}", e),
            }
        };

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;
        let channels = 3;

        let format = JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };
        decoder.set_pixel_format(format);

        let mut decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("jxl-rs: EOF before frame");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs frame info decode error: {:?}", e),
            }
        };

        let mut output_image = Image::<f32>::new((width * channels, height))
            .expect("jxl-rs: failed to create output buffer");

        let mut buffers = vec![JxlOutputBuffer::from_image_rect_mut(
            output_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * channels, height),
                })
                .into_raw(),
        )];

        loop {
            match decoder.process(&mut input, &mut buffers) {
                Ok(ProcessingResult::Complete { .. }) => break,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        panic!("jxl-rs: EOF during frame");
                    }
                    decoder = fallback;
                }
                Err(e) => panic!("jxl-rs frame decode error: {:?}", e),
            }
        }

        let mut pixels = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            pixels.extend_from_slice(output_image.row(y));
        }

        (width, height, pixels)
    }

    /// Convert padded XYB → cropped linear-RGB.
    fn xyb_to_cropped_linear_rgb(
        xyb: &__recon_hook::Xyb,
        padded_width: usize,
        width: usize,
        height: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let padded_height = xyb.x.len() / padded_width;
        let padded_pixels = padded_width * padded_height;

        let mut padded_r = vec![0.0f32; padded_pixels];
        let mut padded_g = vec![0.0f32; padded_pixels];
        let mut padded_b = vec![0.0f32; padded_pixels];

        jxl_simd::xyb_to_linear_rgb_planar(
            &xyb.x,
            &xyb.y,
            &xyb.b,
            &mut padded_r,
            &mut padded_g,
            &mut padded_b,
            padded_pixels,
        );

        let n = width * height;
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        for y in 0..height {
            let dst = y * width;
            let src = y * padded_width;
            r[dst..dst + width].copy_from_slice(&padded_r[src..src + width]);
            g[dst..dst + width].copy_from_slice(&padded_g[src..src + width]);
            b[dst..dst + width].copy_from_slice(&padded_b[src..src + width]);
        }
        (r, g, b)
    }

    fn per_pixel_max_abs(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs() as f64)
            .fold(0.0_f64, f64::max)
    }

    fn per_pixel_mean_abs(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len()) as f64;
        if n == 0.0 {
            return 0.0;
        }
        let sum: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs() as f64)
            .sum();
        sum / n
    }

    /// Returns `(arg_x, arg_y, value)` for the location of the max-abs diff
    /// over all 3 channels. Useful for "where in the image is the bug?".
    fn arg_max_abs_xy(
        ra: &[f32],
        ga: &[f32],
        ba: &[f32],
        rb: &[f32],
        gb: &[f32],
        bb: &[f32],
        width: usize,
        height: usize,
    ) -> (usize, usize, f64, char) {
        let mut best = (0usize, 0usize, 0.0f64, 'R');
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let dr = (ra[i] - rb[i]).abs() as f64;
                let dg = (ga[i] - gb[i]).abs() as f64;
                let db = (ba[i] - bb[i]).abs() as f64;
                if dr > best.2 {
                    best = (x, y, dr, 'R');
                }
                if dg > best.2 {
                    best = (x, y, dg, 'G');
                }
                if db > best.2 {
                    best = (x, y, db, 'B');
                }
            }
        }
        best
    }

    /// For each 8x8 block, compute the max-abs diff over its 64 pixels.
    /// Then aggregate by AC strategy.
    /// Returns: per-strategy (count, max_diff_overall, mean_of_per_block_maxes).
    fn aggregate_per_strategy(
        ra: &[f32],
        ga: &[f32],
        ba: &[f32],
        rb: &[f32],
        gb: &[f32],
        bb: &[f32],
        width: usize,
        height: usize,
        raw_strategy: &[u8],
        xsize_blocks: usize,
        ysize_blocks: usize,
    ) -> Vec<(u8, usize, f64, f64)> {
        // Compute per-8x8-block max-abs.
        let mut per_block_max = vec![0.0_f64; xsize_blocks * ysize_blocks];
        for by in 0..ysize_blocks {
            for bx in 0..xsize_blocks {
                let y0 = by * 8;
                let x0 = bx * 8;
                let mut bmax = 0.0_f64;
                for py in 0..8 {
                    let y = y0 + py;
                    if y >= height {
                        break;
                    }
                    for px in 0..8 {
                        let x = x0 + px;
                        if x >= width {
                            break;
                        }
                        let i = y * width + x;
                        let dr = (ra[i] - rb[i]).abs() as f64;
                        let dg = (ga[i] - gb[i]).abs() as f64;
                        let db = (ba[i] - bb[i]).abs() as f64;
                        let m = dr.max(dg).max(db);
                        if m > bmax {
                            bmax = m;
                        }
                    }
                }
                per_block_max[by * xsize_blocks + bx] = bmax;
            }
        }

        // Aggregate by raw_strategy.
        let mut bucket: Vec<(u8, usize, f64, f64, f64)> = Vec::new(); // (rs, count, max, sum, sum_sq)
        for s in 0..16u8 {
            bucket.push((s, 0, 0.0, 0.0, 0.0));
        }
        for i in 0..raw_strategy.len() {
            let rs = raw_strategy[i] as usize;
            if rs >= 16 {
                continue;
            }
            let m = per_block_max[i];
            let b = &mut bucket[rs];
            b.1 += 1;
            if m > b.2 {
                b.2 = m;
            }
            b.3 += m;
            b.4 += m * m;
        }

        bucket
            .into_iter()
            .map(|(rs, count, max, sum, _)| {
                let mean = if count == 0 { 0.0 } else { sum / count as f64 };
                (rs, count, max, mean)
            })
            .filter(|(_, count, _, _)| *count > 0)
            .collect()
    }

    struct Cell {
        image: String,
        path: PathBuf,
        distance: f32,
        effort: u8,
    }

    fn parse_cells() -> Vec<Cell> {
        let home = std::env::var("HOME").expect("HOME");
        let corpus = format!("{home}/work/codec-corpus");

        if let Ok(spec) = std::env::var("W44_138_CELLS") {
            spec.split(',')
                .filter_map(|s| {
                    let parts: Vec<&str> = s.split(':').collect();
                    if parts.len() != 3 {
                        eprintln!("bad cell spec '{}'", s);
                        return None;
                    }
                    let img = parts[0].to_string();
                    let d: f32 = parts[1].parse().ok()?;
                    let eff: u8 = parts[2].parse().ok()?;
                    let path = if img.contains('/') {
                        PathBuf::from(&img)
                    } else if img.starts_with("terminal")
                        || img.starts_with("codec_wiki")
                        || img.starts_with("imac")
                        || img.starts_with("graph")
                        || img.starts_with("windows")
                    {
                        PathBuf::from(format!("{corpus}/gb82-sc/{img}.png"))
                    } else {
                        PathBuf::from(format!("{corpus}/CID22/CID22-512/validation/{img}.png"))
                    };
                    Some(Cell {
                        image: img,
                        path,
                        distance: d,
                        effort: eff,
                    })
                })
                .collect()
        } else {
            // Default Phase 1: 3 representative cells, both with and without W44-117.
            // - terminal e8 d=4 (screenshot, W44-117 ON)
            // - codec_wiki e8 d=4 (screenshot, W44-117 ON)
            // - 1418519 e8 d=5 (photo, W44-117 OFF per W44-118 gate)
            // Sanity check: 1025469 d=2 e8 (photo at lower d, W44-117 OFF)
            vec![
                Cell {
                    image: "terminal".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/terminal.png")),
                    distance: 4.0,
                    effort: 8,
                },
                Cell {
                    image: "codec_wiki".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/codec_wiki.png")),
                    distance: 4.0,
                    effort: 8,
                },
                Cell {
                    image: "1418519".into(),
                    path: PathBuf::from(format!(
                        "{corpus}/CID22/CID22-512/validation/1418519.png"
                    )),
                    distance: 5.0,
                    effort: 8,
                },
                Cell {
                    image: "1025469".into(),
                    path: PathBuf::from(format!(
                        "{corpus}/CID22/CID22-512/validation/1025469.png"
                    )),
                    distance: 2.0,
                    effort: 8,
                },
            ]
        }
    }

    /// One-shot encode + per-stage capture for one cell.
    fn run_one(
        cell: &Cell,
        force_w44_117_off: bool,
    ) -> Option<RunResult> {
        if !cell.path.exists() {
            eprintln!("MISSING {}: {}", cell.image, cell.path.display());
            return None;
        }

        let img = image::open(&cell.path).expect("open image");
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb_u8: Vec<u8> = img.to_rgb8().into_raw();

        let _ = __recon_hook::take_last();
        let _ = __recon_hook::take_last_steps();
        __recon_hook::set_capture_enabled(true);
        __recon_hook::set_steps_capture_enabled(true);

        // Set env for this run. We rely on the encoder reading the env-var
        // BEFORE consuming the EpfSharpnessSeed policy (W44-132).
        let prev_env = std::env::var("JXL_W44_117_DISABLE").ok();
        if force_w44_117_off {
            // SAFETY: single-threaded test scope; we restore below.
            unsafe { std::env::set_var("JXL_W44_117_DISABLE", "1"); }
        } else {
            // SAFETY: same.
            unsafe { std::env::remove_var("JXL_W44_117_DISABLE"); }
        }

        let cfg = LossyConfig::new(cell.distance).with_effort(cell.effort);
        let bitstream = cfg
            .encode(&rgb_u8, w as u32, h as u32, PixelLayout::Rgb8)
            .expect("encode");

        // Restore env.
        // SAFETY: single-threaded scope.
        unsafe {
            match prev_env {
                Some(v) => std::env::set_var("JXL_W44_117_DISABLE", v),
                None => std::env::remove_var("JXL_W44_117_DISABLE"),
            }
        }

        __recon_hook::set_capture_enabled(false);
        __recon_hook::set_steps_capture_enabled(false);

        let recon = __recon_hook::take_last().expect("recon hook");
        let steps = __recon_hook::take_last_steps().expect("steps hook");

        // Decode shipped bitstream with jxl-rs, linearize sRGB f32 → linear-light.
        let (dw, dh, jxl_rs_srgb) = decode_jxl_rs(&bitstream);
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        let n = w * h;

        let mut jxlrs_r = vec![0.0f32; n];
        let mut jxlrs_g = vec![0.0f32; n];
        let mut jxlrs_b = vec![0.0f32; n];
        for i in 0..n {
            jxlrs_r[i] = srgb_to_linear_val(jxl_rs_srgb[i * 3].clamp(0.0, 1.0));
            jxlrs_g[i] = srgb_to_linear_val(jxl_rs_srgb[i * 3 + 1].clamp(0.0, 1.0));
            jxlrs_b[i] = srgb_to_linear_val(jxl_rs_srgb[i * 3 + 2].clamp(0.0, 1.0));
        }

        let padded_width = steps.padded_width;

        let step_optionals: [(&str, Option<&__recon_hook::Xyb>); 5] = [
            ("after_recon_xyb", Some(&steps.after_recon_xyb)),
            ("after_gab", steps.after_gab.as_ref()),
            ("after_epf", steps.after_epf.as_ref()),
            ("after_patches", steps.after_patches.as_ref()),
            ("after_splines", steps.after_splines.as_ref()),
        ];

        let clamp_planes = |r: Vec<f32>, g: Vec<f32>, b: Vec<f32>| {
            let r: Vec<f32> = r.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let g: Vec<f32> = g.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let b: Vec<f32> = b.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            (r, g, b)
        };

        let mut stage_results: Vec<StageDiag> = Vec::new();
        let mut after_recon_lin: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
        let mut prev_max = f64::NAN;
        for (step_name, xyb_opt) in step_optionals.iter() {
            if let Some(xyb) = xyb_opt {
                let (r, g, b) = xyb_to_cropped_linear_rgb(xyb, padded_width, w, h);
                let (r, g, b) = clamp_planes(r, g, b);
                let max_r = per_pixel_max_abs(&r, &jxlrs_r);
                let max_g = per_pixel_max_abs(&g, &jxlrs_g);
                let max_b = per_pixel_max_abs(&b, &jxlrs_b);
                let max_overall = max_r.max(max_g).max(max_b);
                let mean_r = per_pixel_mean_abs(&r, &jxlrs_r);
                let mean_g = per_pixel_mean_abs(&g, &jxlrs_g);
                let mean_b = per_pixel_mean_abs(&b, &jxlrs_b);
                let delta_vs_prev = if prev_max.is_nan() {
                    f64::NAN
                } else {
                    max_overall - prev_max
                };
                let (argx, argy, argmag, argch) =
                    arg_max_abs_xy(&r, &g, &b, &jxlrs_r, &jxlrs_g, &jxlrs_b, w, h);
                stage_results.push(StageDiag {
                    step: step_name.to_string(),
                    ran: true,
                    max_r,
                    max_g,
                    max_b,
                    max_overall,
                    mean_r,
                    mean_g,
                    mean_b,
                    delta_vs_prev,
                    argmax_x: argx,
                    argmax_y: argy,
                    argmax_ch: argch,
                    argmax_mag: argmag,
                });
                if *step_name == "after_recon_xyb" {
                    after_recon_lin = Some((r, g, b));
                }
                prev_max = max_overall;
            } else {
                stage_results.push(StageDiag {
                    step: step_name.to_string(),
                    ran: false,
                    max_r: f64::NAN,
                    max_g: f64::NAN,
                    max_b: f64::NAN,
                    max_overall: f64::NAN,
                    mean_r: f64::NAN,
                    mean_g: f64::NAN,
                    mean_b: f64::NAN,
                    delta_vs_prev: f64::NAN,
                    argmax_x: 0,
                    argmax_y: 0,
                    argmax_ch: '_',
                    argmax_mag: f64::NAN,
                });
            }
        }

        // Per-strategy aggregation of after_recon_xyb diff.
        let per_strategy = if let Some((r, g, b)) = after_recon_lin.as_ref() {
            aggregate_per_strategy(
                r,
                g,
                b,
                &jxlrs_r,
                &jxlrs_g,
                &jxlrs_b,
                w,
                h,
                &recon.raw_strategy,
                recon.xsize_blocks,
                recon.ysize_blocks,
            )
        } else {
            Vec::new()
        };

        // Find first non-monotonic stage (delta_vs_prev > 0).
        let first_nonmonotonic = stage_results
            .iter()
            .find(|s| s.ran && !s.delta_vs_prev.is_nan() && s.delta_vs_prev > 0.0)
            .map(|s| s.step.clone());

        Some(RunResult {
            width: w,
            height: h,
            bitstream_bytes: bitstream.len(),
            stage_results,
            per_strategy,
            first_nonmonotonic,
        })
    }

    struct StageDiag {
        step: String,
        ran: bool,
        max_r: f64,
        max_g: f64,
        max_b: f64,
        max_overall: f64,
        mean_r: f64,
        mean_g: f64,
        mean_b: f64,
        delta_vs_prev: f64,
        argmax_x: usize,
        argmax_y: usize,
        argmax_ch: char,
        argmax_mag: f64,
    }

    struct RunResult {
        width: usize,
        height: usize,
        bitstream_bytes: usize,
        stage_results: Vec<StageDiag>,
        per_strategy: Vec<(u8, usize, f64, f64)>,
        first_nonmonotonic: Option<String>,
    }

    pub fn run() {
        let cells = parse_cells();

        // TSV header.
        println!(
            "image\tdistance\teffort\twidth\theight\t\
             w44_117_mode\tstep\tstep_ran\t\
             max_overall\tmax_r\tmax_g\tmax_b\t\
             mean_r\tmean_g\tmean_b\t\
             delta_vs_prev_max_overall\t\
             argmax_x\targmax_y\targmax_ch\targmax_mag\t\
             first_nonmonotonic\tbytes"
        );

        for cell in &cells {
            for &(mode_label, force_off) in
                &[("default", false), ("disable_117", true)]
            {
                eprintln!(
                    "cell={} d={} e={} mode={}",
                    cell.image, cell.distance, cell.effort, mode_label
                );
                let Some(result) = run_one(cell, force_off) else {
                    continue;
                };

                for stage in &result.stage_results {
                    let first_nm = result.first_nonmonotonic.as_deref().unwrap_or("");
                    println!(
                        "{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t\
                         {:.6}\t{:.6}\t{:.6}\t{:.6}\t\
                         {:.6}\t{:.6}\t{:.6}\t\
                         {:.6}\t\
                         {}\t{}\t{}\t{:.6}\t\
                         {}\t{}",
                        cell.image,
                        cell.distance,
                        cell.effort,
                        result.width,
                        result.height,
                        mode_label,
                        stage.step,
                        if stage.ran { 1 } else { 0 },
                        stage.max_overall,
                        stage.max_r,
                        stage.max_g,
                        stage.max_b,
                        stage.mean_r,
                        stage.mean_g,
                        stage.mean_b,
                        stage.delta_vs_prev,
                        stage.argmax_x,
                        stage.argmax_y,
                        stage.argmax_ch,
                        stage.argmax_mag,
                        first_nm,
                        result.bitstream_bytes,
                    );
                }

                // Append per-strategy rows: step=per_strategy/<NAME>.
                for (rs, count, max_per_strategy, mean_per_strategy) in &result.per_strategy {
                    let name = STRATEGY_NAMES.get(*rs as usize).copied().unwrap_or("?");
                    let first_nm = result.first_nonmonotonic.as_deref().unwrap_or("");
                    // Per-strategy row uses 22 columns matching the per-stage header.
                    // Overload semantics:
                    //   max_overall = max(per-8x8-block max) inside this strategy's blocks.
                    //   mean_r      = mean(per-8x8-block max) inside this strategy's blocks.
                    //   delta_vs_prev = count of blocks of this strategy.
                    //   argmax_*    = unused.
                    println!(
                        "{}\t{:.2}\t{}\t{}\t{}\t{}\tper_strategy/{}\t1\t\
                         {:.6}\t{:.6}\t{:.6}\t{:.6}\t\
                         {:.6}\t{:.6}\t{:.6}\t\
                         {:.6}\t\
                         {}\t{}\t{}\t{:.6}\t\
                         {}\t{}",
                        cell.image,
                        cell.distance,
                        cell.effort,
                        result.width,
                        result.height,
                        mode_label,
                        name,
                        max_per_strategy,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        mean_per_strategy,
                        f64::NAN,
                        f64::NAN,
                        *count as f64,
                        0,
                        0,
                        '_',
                        f64::NAN,
                        first_nm,
                        result.bitstream_bytes,
                    );
                }
            }
        }
    }
}
