// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-111 — Butteraugli metric divergence root-cause investigation.
//!
//! Setup: W44-105 ship memo documented `2.07 (ours) vs 47.7 (libjxl)` butteraugli
//! scores on **bit-identical initial quant fields** for terminal e8 d=4. The
//! qac-scale palliative (W44-105/107/108/109) trades bytes for SSIM2 on
//! screenshot-class to work around the gap. This example bisects whether the
//! 23× metric gap comes from:
//!
//!   1. **Different reconstructions** at same qac (because our buttloop's internal
//!      recon diverges from what the decoder produces — the existing Layer-1 test
//!      already proved this with 5.4× CfL pass-2 fix and ~0.05 max-abs residual)
//!   2. **Different metric implementations** (our `butteraugli` crate vs libjxl's
//!      internal butteraugli) when fed the SAME pixels
//!
//! Methodology:
//!
//!  - Encode at d=4.0 e8 on a corpus of (image × distance) cells.
//!  - Capture the buttloop's INTERNAL recon at iter=2 via the `__recon_hook`.
//!  - Decode the SHIPPED bitstream with jxl-rs (sRGB f32 → linearize).
//!  - Run our `butteraugli` crate three times:
//!     a. butteraugli(original linear, internal_recon linear) — what the buttloop sees
//!     b. butteraugli(original linear, jxl_rs_decoded linear) — what the user gets
//!     c. butteraugli(internal_recon linear, jxl_rs_decoded linear) — recon-vs-decode gap
//!
//!  - Output: per-cell tabulated scores + linear-RGB max-abs diffs + pnorm_3.
//!
//! Interpretation:
//!
//!  - If (a) << (b): buttloop converges on an over-optimistic recon → the
//!    Chunk-3 partial fix isn't enough; need to make our internal recon match
//!    decoded output bit-for-bit (or use real encode→decode roundtrip).
//!  - If (a) ≈ (b) but both << libjxl's iter-0 score: divergence is in the
//!    *libjxl* recon itself (not ours), suggesting libjxl's iter-0 recon has
//!    a separate bug that we're paradoxically beating.
//!  - If (a) and (b) both far from (c): metric implementation does something
//!    asymmetric (e.g. our crate's max-norm vs libjxl's).
//!
//! Run:
//!
//! ```bash
//! CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!   cargo run --release \
//!     --example w44_111_metric_divergence \
//!     --features '__internal_recon_hook butteraugli-loop' \
//!     --manifest-path jxl-encoder/Cargo.toml \
//!     > /tmp/w44_111_metric_divergence.tsv
//! ```
//!
//! Env vars:
//!   W44_111_CELLS=img1:d:eff,img2:d:eff,...  (default: terminal d=4 e8 + 1025469 d=2 e8)

#[cfg(not(all(feature = "__internal_recon_hook", feature = "butteraugli-loop")))]
fn main() {
    eprintln!(
        "w44_111_metric_divergence requires --features '__internal_recon_hook butteraugli-loop'."
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

    /// sRGB normalized [0,1] f32 → linear light. Matches `butteraugli::srgb_to_linear`
    /// and the tests/buttloop_recon_parity.rs helper.
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

    /// Run butteraugli on planar linear RGB inputs.
    fn buttloop_score(
        r1: &[f32],
        g1: &[f32],
        b1: &[f32],
        r2: &[f32],
        g2: &[f32],
        b2: &[f32],
        width: usize,
        height: usize,
    ) -> Option<(f64, f64)> {
        let params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(false);
        let r = butteraugli::ButteraugliReference::new_linear_planar(
            r1, g1, b1, width, height, width, params,
        )
        .ok()?;
        let result = r.compare_linear_planar(r2, g2, b2, width).ok()?;
        Some((result.score, result.pnorm_3))
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

        if let Ok(spec) = std::env::var("W44_111_CELLS") {
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
            // Default: the W44-105 wedge + the chunk-3 detailed photo
            vec![
                Cell {
                    image: "terminal".into(),
                    path: PathBuf::from(format!("{corpus}/gb82-sc/terminal.png")),
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
            ]
        }
    }

    pub fn run() {
        let cells = parse_cells();

        // TSV header
        println!(
            "image\tdistance\teffort\twidth\theight\t\
             score_orig_vs_intrecon\tpnorm3_orig_vs_intrecon\t\
             score_orig_vs_jxlrs\tpnorm3_orig_vs_jxlrs\t\
             score_intrecon_vs_jxlrs\tpnorm3_intrecon_vs_jxlrs\t\
             maxabs_intrecon_vs_jxlrs_r\tmaxabs_intrecon_vs_jxlrs_g\tmaxabs_intrecon_vs_jxlrs_b\t\
             meanabs_intrecon_vs_jxlrs_r\tmeanabs_intrecon_vs_jxlrs_g\tmeanabs_intrecon_vs_jxlrs_b\t\
             maxabs_orig_vs_jxlrs_r\tmaxabs_orig_vs_jxlrs_g\tmaxabs_orig_vs_jxlrs_b\t\
             meanabs_orig_vs_jxlrs_r\tmeanabs_orig_vs_jxlrs_g\tmeanabs_orig_vs_jxlrs_b\t\
             maxabs_orig_vs_intrecon_r\tmaxabs_orig_vs_intrecon_g\tmaxabs_orig_vs_intrecon_b\t\
             meanabs_orig_vs_intrecon_r\tmeanabs_orig_vs_intrecon_g\tmeanabs_orig_vs_intrecon_b\t\
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

            // Convert sRGB u8 to linear-light planar for the metric reference side.
            let n = w * h;
            let mut orig_r = vec![0.0f32; n];
            let mut orig_g = vec![0.0f32; n];
            let mut orig_b = vec![0.0f32; n];
            for i in 0..n {
                orig_r[i] = butteraugli::srgb_to_linear(rgb_u8[i * 3]);
                orig_g[i] = butteraugli::srgb_to_linear(rgb_u8[i * 3 + 1]);
                orig_b[i] = butteraugli::srgb_to_linear(rgb_u8[i * 3 + 2]);
            }

            // Encode + capture internal recon at final buttloop iter.
            let cfg = LossyConfig::new(cell.distance).with_effort(cell.effort);
            let _ = __recon_hook::take_last();
            __recon_hook::set_capture_enabled(true);
            let bitstream = cfg
                .encode(&rgb_u8, w as u32, h as u32, PixelLayout::Rgb8)
                .expect("encode");
            __recon_hook::set_capture_enabled(false);
            let recon = __recon_hook::take_last().expect("recon hook");
            assert_eq!(recon.width, w);
            assert_eq!(recon.height, h);

            // The internal recon is linear-RGB f32 (NOT clamped). For metric input
            // clamp to [0,1] like the Layer-1 test does — out-of-gamut floats hurt
            // butteraugli without informing the user-facing quality story.
            let recon_r: Vec<f32> = recon.r.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let recon_g: Vec<f32> = recon.g.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            let recon_b: Vec<f32> = recon.b.iter().map(|x| x.clamp(0.0, 1.0)).collect();

            // Decode shipped bitstream with jxl-rs, linearize sRGB f32 → linear-light.
            let (dw, dh, jxl_rs_srgb) = decode_jxl_rs(&bitstream);
            assert_eq!(dw, w);
            assert_eq!(dh, h);

            let mut jxlrs_r = vec![0.0f32; n];
            let mut jxlrs_g = vec![0.0f32; n];
            let mut jxlrs_b = vec![0.0f32; n];
            for i in 0..n {
                jxlrs_r[i] = srgb_to_linear_val(jxl_rs_srgb[i * 3].clamp(0.0, 1.0));
                jxlrs_g[i] = srgb_to_linear_val(jxl_rs_srgb[i * 3 + 1].clamp(0.0, 1.0));
                jxlrs_b[i] = srgb_to_linear_val(jxl_rs_srgb[i * 3 + 2].clamp(0.0, 1.0));
            }

            // Three butteraugli scores.
            let (s_oi, p_oi) = buttloop_score(
                &orig_r, &orig_g, &orig_b, &recon_r, &recon_g, &recon_b, w, h,
            )
            .unwrap_or((-1.0, -1.0));
            let (s_oj, p_oj) = buttloop_score(
                &orig_r, &orig_g, &orig_b, &jxlrs_r, &jxlrs_g, &jxlrs_b, w, h,
            )
            .unwrap_or((-1.0, -1.0));
            let (s_ij, p_ij) = buttloop_score(
                &recon_r, &recon_g, &recon_b, &jxlrs_r, &jxlrs_g, &jxlrs_b, w, h,
            )
            .unwrap_or((-1.0, -1.0));

            // Per-channel max-abs / mean-abs deltas (linear RGB).
            let max_ij_r = per_pixel_max_abs(&recon_r, &jxlrs_r);
            let max_ij_g = per_pixel_max_abs(&recon_g, &jxlrs_g);
            let max_ij_b = per_pixel_max_abs(&recon_b, &jxlrs_b);
            let mean_ij_r = per_pixel_mean_abs(&recon_r, &jxlrs_r);
            let mean_ij_g = per_pixel_mean_abs(&recon_g, &jxlrs_g);
            let mean_ij_b = per_pixel_mean_abs(&recon_b, &jxlrs_b);

            let max_oj_r = per_pixel_max_abs(&orig_r, &jxlrs_r);
            let max_oj_g = per_pixel_max_abs(&orig_g, &jxlrs_g);
            let max_oj_b = per_pixel_max_abs(&orig_b, &jxlrs_b);
            let mean_oj_r = per_pixel_mean_abs(&orig_r, &jxlrs_r);
            let mean_oj_g = per_pixel_mean_abs(&orig_g, &jxlrs_g);
            let mean_oj_b = per_pixel_mean_abs(&orig_b, &jxlrs_b);

            let max_oi_r = per_pixel_max_abs(&orig_r, &recon_r);
            let max_oi_g = per_pixel_max_abs(&orig_g, &recon_g);
            let max_oi_b = per_pixel_max_abs(&orig_b, &recon_b);
            let mean_oi_r = per_pixel_mean_abs(&orig_r, &recon_r);
            let mean_oi_g = per_pixel_mean_abs(&orig_g, &recon_g);
            let mean_oi_b = per_pixel_mean_abs(&orig_b, &recon_b);

            println!(
                "{}\t{:.2}\t{}\t{}\t{}\t\
                 {:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t\
                 {:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t\
                 {:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t\
                 {:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t\
                 {}",
                cell.image,
                cell.distance,
                cell.effort,
                w,
                h,
                s_oi,
                p_oi,
                s_oj,
                p_oj,
                s_ij,
                p_ij,
                max_ij_r,
                max_ij_g,
                max_ij_b,
                mean_ij_r,
                mean_ij_g,
                mean_ij_b,
                max_oj_r,
                max_oj_g,
                max_oj_b,
                mean_oj_r,
                mean_oj_g,
                mean_oj_b,
                max_oi_r,
                max_oi_g,
                max_oi_b,
                mean_oi_r,
                mean_oi_g,
                mean_oi_b,
                bitstream.len()
            );
        }
    }
}
