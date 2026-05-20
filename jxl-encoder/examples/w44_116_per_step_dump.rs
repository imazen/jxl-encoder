// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-116 — Per-step XYB capture to identify the first divergent step in the
//! buttloop's recon pipeline vs jxl-rs decoded output.
//!
//! Background (read first):
//!   - W44-111: ruled out butteraugli metric divergence
//!   - W44-112: ruled out SetQuantField/AdjustQuantBlockAC parallelism
//!   - W44-113: source-diff audit ranked 4 candidates; #1 AFV, #2 dequant
//!     bias, #3 DC CfL algebra, #4 per-strategy IDCT precision
//!   - W44-114: AFV IDCT bit-parity tests PASS (rules out #1)
//!   - W44-115: per-strategy IDCT precision tests PASS (rules out #4)
//!
//! Remaining R/G linear-RGB residual must come from one of:
//!   (a) dequant + CfL + LFFromDC ORDERING in `reconstruct.rs:799-967`
//!       (individual steps at parity but pipeline order may diverge)
//!   (b) `gab_smooth` boundary handling (re-audit)
//!   (c) per-block `apply_epf` sharpness map per-pixel diff
//!   (d) `add_patches` / `add_splines` execution-order interactions
//!
//! Strategy: capture XYB AFTER each step of the buttloop's pipeline, then
//! convert each snapshot to linear-RGB via the parity-guaranteed
//! `xyb_to_linear_rgb_planar`. Compare each to jxl-rs decoded output. The
//! decoder always runs the FULL pipeline, so the encoder's most-complete
//! snapshot (after_splines or whichever last step ran) is what should match
//! jxl-rs. The earlier snapshots are missing the subsequent steps, so they
//! diverge from jxl-rs by the magnitude of those missing steps.
//!
//! Identifying the bug:
//!   - If each successive step DECREASES max-abs vs jxl-rs (monotonic
//!     convergence): every step is at parity; divergence is INSIDE
//!     `reconstruct.rs` itself (the FIRST snapshot already diverges from
//!     what the decoder produces after its own first step). Drill into the
//!     dequant/CfL/LFFromDC inside `reconstruct_xyb`.
//!   - If some step INCREASES max-abs vs jxl-rs (non-monotonic): that step
//!     is the bug. The encoder's step is producing pixels the decoder
//!     doesn't.
//!
//! Run:
//!
//! ```bash
//! CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release \
//!     --example w44_116_per_step_dump \
//!     --features '__internal_recon_hook butteraugli-loop' \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > /tmp/w44_116_per_step_dump.tsv
//! ```
//!
//! Env vars:
//!   W44_116_CELLS=img1:d:eff,img2:d:eff,...
//!     (default: 1025469 d=2 e8 + 1025469 d=4 e8 + terminal d=2/d=4 e8)

#[cfg(not(all(feature = "__internal_recon_hook", feature = "butteraugli-loop")))]
fn main() {
    eprintln!(
        "w44_116_per_step_dump requires --features '__internal_recon_hook butteraugli-loop'."
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

    /// Convert padded XYB → cropped linear-RGB. Mirrors what the buttloop
    /// does at the end of its pipeline.
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

    struct Cell {
        image: String,
        path: PathBuf,
        distance: f32,
        effort: u8,
    }

    fn parse_cells() -> Vec<Cell> {
        let home = std::env::var("HOME").expect("HOME");
        let corpus = format!("{home}/work/codec-corpus");

        if let Ok(spec) = std::env::var("W44_116_CELLS") {
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
            // Default: the W44-111 reference cell + 3 spot-checks
            vec![
                Cell {
                    image: "1025469".into(),
                    path: PathBuf::from(format!("{corpus}/CID22/CID22-512/validation/1025469.png")),
                    distance: 2.0,
                    effort: 8,
                },
                Cell {
                    image: "1025469".into(),
                    path: PathBuf::from(format!("{corpus}/CID22/CID22-512/validation/1025469.png")),
                    distance: 4.0,
                    effort: 8,
                },
                Cell {
                    image: "terminal".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/terminal.png")),
                    distance: 2.0,
                    effort: 8,
                },
                Cell {
                    image: "terminal".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/terminal.png")),
                    distance: 4.0,
                    effort: 8,
                },
            ]
        }
    }

    pub fn run() {
        let cells = parse_cells();
        let no_gab = std::env::var("W44_116_NO_GAB").is_ok();

        if no_gab {
            eprintln!(
                "W44_116_NO_GAB set: disabling gaborish on encoder side. \
                 jxl-rs will also skip its inverse gaborish step because the \
                 bitstream signals it. `after_recon_xyb` should now match \
                 jxl-rs at parity if reconstruct_xyb is correct."
            );
        }

        // TSV header: per-step max-abs and mean-abs of each XYB-step's linear-RGB
        // vs jxl-rs decoded linear-RGB.
        println!(
            "image\tdistance\teffort\twidth\theight\tgab\t\
             step\tstep_ran\t\
             maxabs_r\tmaxabs_g\tmaxabs_b\t\
             meanabs_r\tmeanabs_g\tmeanabs_b\t\
             delta_vs_prev_maxabs_r\tdelta_vs_prev_maxabs_g\tdelta_vs_prev_maxabs_b\t\
             bytes"
        );

        for cell in &cells {
            if !cell.path.exists() {
                eprintln!("MISSING {}: {}", cell.image, cell.path.display());
                continue;
            }

            let img = image::open(&cell.path).expect("open image");
            let (w, h) = (img.width() as usize, img.height() as usize);
            let rgb_u8: Vec<u8> = img.to_rgb8().into_raw();

            // Enable BOTH hooks: per-step XYB and linear-RGB cropped recon.
            let _ = __recon_hook::take_last();
            let _ = __recon_hook::take_last_steps();
            __recon_hook::set_capture_enabled(true);
            __recon_hook::set_steps_capture_enabled(true);

            let mut cfg = LossyConfig::new(cell.distance).with_effort(cell.effort);
            if no_gab {
                cfg = cfg.with_gaborish(false);
            }
            if std::env::var("W44_116_NO_EPF").is_ok() {
                cfg = cfg.with_epf_level(0);
            }
            let bitstream = cfg
                .encode(&rgb_u8, w as u32, h as u32, PixelLayout::Rgb8)
                .expect("encode");

            __recon_hook::set_capture_enabled(false);
            __recon_hook::set_steps_capture_enabled(false);

            let recon = __recon_hook::take_last().expect("recon hook");
            let steps = __recon_hook::take_last_steps().expect("steps hook");
            assert_eq!(recon.width, w);
            assert_eq!(recon.height, h);
            assert_eq!(steps.width, w);
            assert_eq!(steps.height, h);
            // Note: `recon.r` is CROPPED to (width, height) per
            // butteraugli_loop's cropping pass before storing into the
            // existing InternalRecon. `steps.{after_*}.{x,y,b}` are PADDED
            // to `padded_width * padded_height`. Don't confuse the two.
            assert_eq!(recon.r.len(), w * h);
            assert_eq!(
                steps.after_recon_xyb.x.len(),
                steps.padded_width * steps.padded_height,
                "step xyb size mismatch"
            );

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

            // For each step that ran, convert XYB → cropped linear-RGB and compare to jxl-rs.
            // Clamp to [0,1] like W44-111 does (out-of-gamut floats hurt the delta number
            // without changing the bug story).
            let clamp_planes = |r: Vec<f32>, g: Vec<f32>, b: Vec<f32>| {
                let r: Vec<f32> = r.iter().map(|x| x.clamp(0.0, 1.0)).collect();
                let g: Vec<f32> = g.iter().map(|x| x.clamp(0.0, 1.0)).collect();
                let b: Vec<f32> = b.iter().map(|x| x.clamp(0.0, 1.0)).collect();
                (r, g, b)
            };

            // (step_name, ran, optional XYB)
            let step_optionals: [(&str, Option<&__recon_hook::Xyb>); 5] = [
                ("after_recon_xyb", Some(&steps.after_recon_xyb)),
                ("after_gab", steps.after_gab.as_ref()),
                ("after_epf", steps.after_epf.as_ref()),
                ("after_patches", steps.after_patches.as_ref()),
                ("after_splines", steps.after_splines.as_ref()),
            ];

            let mut prev_max: Option<(f64, f64, f64)> = None;

            for (step_name, xyb_opt) in step_optionals.iter() {
                let (ran, max_r, max_g, max_b, mean_r, mean_g, mean_b) = if let Some(xyb) = xyb_opt
                {
                    let (r, g, b) = xyb_to_cropped_linear_rgb(xyb, padded_width, w, h);
                    let (r, g, b) = clamp_planes(r, g, b);
                    let max_r = per_pixel_max_abs(&r, &jxlrs_r);
                    let max_g = per_pixel_max_abs(&g, &jxlrs_g);
                    let max_b = per_pixel_max_abs(&b, &jxlrs_b);
                    let mean_r = per_pixel_mean_abs(&r, &jxlrs_r);
                    let mean_g = per_pixel_mean_abs(&g, &jxlrs_g);
                    let mean_b = per_pixel_mean_abs(&b, &jxlrs_b);
                    (1u8, max_r, max_g, max_b, mean_r, mean_g, mean_b)
                } else {
                    (
                        0u8,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                    )
                };

                let (delta_r, delta_g, delta_b) = if ran == 1 {
                    if let Some((pr, pg, pb)) = prev_max {
                        (max_r - pr, max_g - pg, max_b - pb)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    }
                } else {
                    (f64::NAN, f64::NAN, f64::NAN)
                };

                if ran == 1 {
                    prev_max = Some((max_r, max_g, max_b));
                }

                let gab_label = if no_gab { "off" } else { "on" };
                println!(
                    "{}\t{:.2}\t{}\t{}\t{}\t{}\t\
                     {}\t{}\t\
                     {:.6}\t{:.6}\t{:.6}\t\
                     {:.6}\t{:.6}\t{:.6}\t\
                     {:.6}\t{:.6}\t{:.6}\t\
                     {}",
                    cell.image,
                    cell.distance,
                    cell.effort,
                    w,
                    h,
                    gab_label,
                    step_name,
                    ran,
                    max_r,
                    max_g,
                    max_b,
                    mean_r,
                    mean_g,
                    mean_b,
                    delta_r,
                    delta_g,
                    delta_b,
                    bitstream.len()
                );
            }

            // Sanity row: existing InternalRecon (the buttloop's final linear-RGB)
            // compared against jxl-rs. This MUST match the last "ran=1" step
            // above (within FP noise from the duplicate xyb_to_linear_rgb_planar
            // call). Anchor for regression on this example itself.
            let recon_r_c: Vec<f32> = recon.r.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let recon_g_c: Vec<f32> = recon.g.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let recon_b_c: Vec<f32> = recon.b.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let max_r = per_pixel_max_abs(&recon_r_c, &jxlrs_r);
            let max_g = per_pixel_max_abs(&recon_g_c, &jxlrs_g);
            let max_b = per_pixel_max_abs(&recon_b_c, &jxlrs_b);
            let mean_r = per_pixel_mean_abs(&recon_r_c, &jxlrs_r);
            let mean_g = per_pixel_mean_abs(&recon_g_c, &jxlrs_g);
            let mean_b = per_pixel_mean_abs(&recon_b_c, &jxlrs_b);
            let gab_label = if no_gab { "off" } else { "on" };
            println!(
                "{}\t{:.2}\t{}\t{}\t{}\t{}\t\
                 {}\t{}\t\
                 {:.6}\t{:.6}\t{:.6}\t\
                 {:.6}\t{:.6}\t{:.6}\t\
                 {:.6}\t{:.6}\t{:.6}\t\
                 {}",
                cell.image,
                cell.distance,
                cell.effort,
                w,
                h,
                gab_label,
                "anchor_internal_recon",
                1,
                max_r,
                max_g,
                max_b,
                mean_r,
                mean_g,
                mean_b,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                bitstream.len()
            );
        }
    }
}
