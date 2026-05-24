// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! W44-RECON-DEEP/A8 — HONEST-STOP test for the entire W44-RECON-DEEP arc.
//!
//! Question: does the buttloop's internal recon score match the score of
//! the SHIPPED bitstream (decoded by the real decoder)? If yes within
//! noise, the W44-138 P-1 / W44-117 / W44-118 chain has no quality EV;
//! the arc can pivot to perf/architecture. If no, P-1 is critical.
//!
//! Method: per-cell, measure four scores for both butteraugli and SSIM2:
//!
//!   - `bfly_internal` / `ssim2_internal`: source vs buttloop's last-iter
//!     internal recon (linear-RGB captured via `__internal_recon_hook`).
//!     This is the target the buttloop converged toward.
//!   - `bfly_decoded` / `ssim2_decoded`: source vs jxl-rs decode of the
//!     SHIPPED bitstream. This is what users see.
//!   - `bfly_oxide` / `ssim2_oxide`: source vs jxl-oxide decode of the
//!     SHIPPED bitstream. Cross-decoder sanity check (must match jxl-rs
//!     within tiny FP noise).
//!   - `bfly_cjxl` / `ssim2_cjxl`: source vs jxl-rs decode of cjxl's
//!     bitstream at the same effort + distance. Reference target.
//!
//! Derived columns:
//!   - `delta_internal_vs_decoded` = `bfly_internal - bfly_decoded`
//!     (positive = buttloop optimistic; if abs is small, the buttloop's
//!     target IS what the user gets, so P-1 has zero quality EV)
//!   - `delta_decoded_vs_cjxl` = `bfly_decoded - bfly_cjxl`
//!     (positive = our output worse than cjxl; helpful framing for "are
//!     we under-converging vs cjxl?")
//!
//! Verdict:
//!   - All cells with `|delta_internal_vs_decoded| < 0.30` butteraugli
//!     → NEGLIGIBLE-GAP, pivot to perf chunks.
//!   - One or more cells with `|delta_internal_vs_decoded| >= 0.50`
//!     butteraugli → CRITICAL-GAP, ship P-1.
//!   - Else MIXED — per-cluster narrative.
//!
//! Build:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo build --release \
//!       --example w44_recon_deep_a8_recon_vs_decoded \
//!       --features '__internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!       --manifest-path jxl-encoder/Cargo.toml
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!       --example w44_recon_deep_a8_recon_vs_decoded \
//!       --features '__internal_recon_hook butteraugli-loop ssim2-loop parallel' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       > benchmarks/w44_recon_deep_a8_recon_vs_decoded_2026-05-23.tsv
//!
//! Inputs that informed this design:
//!   - memory/w44_138_buttloop_recon_root_cause_2026-05-20.md (Phase 2 P-3
//!     candidate: "audit the actual SSIM2 cost of the +0.034 EPF delta")
//!   - memory/w44_116_per_step_dump_identifies_epf_sharpness_2026-05-20.md
//!     (confirmed EPF sharpness divergence ~0.05-0.17 R linear-RGB max-abs)
//!   - examples/w44_138_recon_root_cause.rs (recon-hook capture pattern)
//!   - examples/w44_170_cjxl_step025_sweep.rs (canonical cjxl + scoring)
//!   - examples/w44_139_ssim2_cost_gate.rs (single-knob SSIM2 cost-gate
//!     pattern, here generalized to internal vs decoded scoring).

#[cfg(not(all(
    feature = "__internal_recon_hook",
    feature = "butteraugli-loop",
    feature = "ssim2-loop"
)))]
fn main() {
    eprintln!(
        "w44_recon_deep_a8_recon_vs_decoded requires --features '__internal_recon_hook butteraugli-loop ssim2-loop'."
    );
    std::process::exit(2);
}

#[cfg(all(
    feature = "__internal_recon_hook",
    feature = "butteraugli-loop",
    feature = "ssim2-loop"
))]
fn main() {
    inner::run();
}

#[cfg(all(
    feature = "__internal_recon_hook",
    feature = "butteraugli-loop",
    feature = "ssim2-loop"
))]
mod inner {
    use butteraugli::{ButteraugliParams, butteraugli_linear};
    use imgref::Img;
    use jxl_encoder::vardct::__recon_hook;
    use jxl_encoder::{Limits, LossyConfig, PixelLayout};
    use rgb::RGB;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Instant;

    // ── Color helpers (must match the encoder's sRGB transfer for parity) ──

    fn srgb_to_linear_val(c: f32) -> f32 {
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn srgb_u8_to_linear_val(b: u8) -> f32 {
        srgb_to_linear_val(b as f32 / 255.0)
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

    // ── Decoder wrappers ────────────────────────────────────────────────────

    /// Decode JXL via jxl-rs (the PRIMARY decoder, per CLAUDE.md).
    /// Returns (width, height, linear-RGB f32 interleaved RGBRGB...).
    fn decode_jxl_rs_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
        use jxl::api::{
            JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
            JxlPixelFormat, ProcessingResult, states,
        };
        use jxl::image::{Image, Rect};

        let mut input = bytes;
        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut decoder = loop {
            match decoder.process(&mut input) {
                Ok(ProcessingResult::Complete { result }) => break result,
                Ok(ProcessingResult::NeedsMoreInput { fallback, .. }) => {
                    if input.is_empty() {
                        return None;
                    }
                    decoder = fallback;
                }
                Err(_) => return None,
            }
        };

        let basic_info = decoder.basic_info().clone();
        let (width, height) = basic_info.size;
        let channels = 3;

        // Request sRGB f32, then linearize ourselves (the same path as the
        // W44-138 + W44-170 harnesses use).
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
                        return None;
                    }
                    decoder = fallback;
                }
                Err(_) => return None,
            }
        };

        let mut output_image = Image::<f32>::new((width * channels, height)).ok()?;
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
                        return None;
                    }
                    decoder = fallback;
                }
                Err(_) => return None,
            }
        }

        let mut srgb = Vec::with_capacity(width * height * channels);
        for y in 0..height {
            srgb.extend_from_slice(output_image.row(y));
        }
        // Linearize (sRGB f32 [0,1] → linear-RGB f32).
        let linear: Vec<f32> = srgb
            .into_iter()
            .map(|s| srgb_to_linear_val(s.clamp(0.0, 1.0)))
            .collect();
        Some((width, height, linear))
    }

    /// Decode JXL via jxl-oxide (linear-sRGB direct output).
    fn decode_jxl_oxide_linear(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
        let reader = Cursor::new(bytes);
        let mut img = jxl_oxide::JxlImage::builder().read(reader).ok()?;
        img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = img.render_frame(0).ok()?;
        let fb = render.image_all_channels();
        Some((fb.width(), fb.height(), fb.buf().to_vec()))
    }

    // ── Scoring ─────────────────────────────────────────────────────────────

    fn score_linear(
        orig_linear: &Img<Vec<RGB<f32>>>,
        orig_srgb: &Img<Vec<[u8; 3]>>,
        dec_lin_interleaved: &[f32],
        dw: usize,
        dh: usize,
    ) -> (f64, f64) {
        let dec_pixels: Vec<RGB<f32>> = dec_lin_interleaved
            .chunks(3)
            .map(|c| RGB::new(c[0], c[1], c[2]))
            .collect();
        let dec_lin_img: Img<Vec<RGB<f32>>> = Img::new(dec_pixels, dw, dh);
        let bfly = butteraugli_linear(
            orig_linear.as_ref(),
            dec_lin_img.as_ref(),
            &ButteraugliParams::default(),
        )
        .map(|r| r.score as f64)
        .unwrap_or(f64::NAN);

        let dec_srgb: Vec<[u8; 3]> = dec_lin_interleaved
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
        let ssim2 = fast_ssim2::compute_ssimulacra2(orig_srgb.as_ref(), dec_srgb_img.as_ref())
            .unwrap_or(f64::NAN);
        (bfly, ssim2)
    }

    /// Score the planar linear-RGB recon (as captured by InternalRecon) by
    /// interleaving on the fly. Avoids one extra allocation pass.
    fn score_planar(
        orig_linear: &Img<Vec<RGB<f32>>>,
        orig_srgb: &Img<Vec<[u8; 3]>>,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        w: usize,
        h: usize,
    ) -> (f64, f64) {
        let n = w * h;
        let mut interleaved = Vec::with_capacity(n * 3);
        for i in 0..n {
            interleaved.push(r[i]);
            interleaved.push(g[i]);
            interleaved.push(b[i]);
        }
        score_linear(orig_linear, orig_srgb, &interleaved, w, h)
    }

    // ── Encoders ────────────────────────────────────────────────────────────

    fn cjxl_bin() -> PathBuf {
        if let Ok(p) = std::env::var("CJXL") {
            return PathBuf::from(p);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        PathBuf::from(home).join("work/jxl-efforts/libjxl/build/tools/cjxl")
    }

    fn encode_cjxl(src_png: &Path, effort: u8, distance: f32) -> Option<Vec<u8>> {
        let tmp = std::env::temp_dir().join(format!(
            "w44_recon_deep_a8_cjxl_e{effort}_d{:.2}_{}.jxl",
            distance,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let status = Command::new(cjxl_bin())
            .arg(src_png)
            .arg(&tmp)
            .args(["-e", &effort.to_string()])
            .args(["-d", &format!("{distance}")])
            .args(["--num_threads", "1"])
            .arg("--quiet")
            .output()
            .ok()?;
        if !status.status.success() {
            eprintln!(
                "  cjxl failed: {}",
                String::from_utf8_lossy(&status.stderr)
            );
            return None;
        }
        let bytes = std::fs::read(&tmp).ok()?;
        let _ = std::fs::remove_file(&tmp);
        Some(bytes)
    }

    fn encode_ours_with_recon_capture(
        rgb: &[u8],
        w: u32,
        h: u32,
        distance: f32,
        effort: u8,
    ) -> Option<(Vec<u8>, __recon_hook::InternalRecon)> {
        // Drain prior captures.
        let _ = __recon_hook::take_last();
        __recon_hook::set_capture_enabled(true);

        // Lift the 2 GiB memory cap for the larger screenshots — matching
        // W44-139's pattern.
        let limits = Limits::new().with_max_memory_bytes(8 * 1024 * 1024 * 1024);

        let cfg = LossyConfig::new(distance)
            .with_effort(effort)
            .with_threads(8);
        let bitstream = cfg
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_limits(&limits)
            .encode(rgb)
            .ok()?;

        __recon_hook::set_capture_enabled(false);
        let recon = __recon_hook::take_last()?;
        Some((bitstream, recon))
    }

    // ── Cells ───────────────────────────────────────────────────────────────

    struct Cell {
        image: String,
        path: PathBuf,
        distance: f32,
        effort: u8,
        notes: &'static str,
    }

    fn cells() -> Vec<Cell> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into());
        let corpus = PathBuf::from(format!("{home}/work/codec-corpus"));
        let cid22 = corpus.join("CID22/CID22-512/validation");
        let gb82 = corpus.join("gb82-sc");
        let clic = corpus.join("clic2025-1024");

        vec![
            // W44-138 P-1 photo target — the wedge cell where the buttloop
            // internal recon over-smooths by +0.034 R linear-RGB at after_epf.
            Cell {
                image: "1418519".into(),
                path: cid22.join("1418519.png"),
                distance: 5.0,
                effort: 8,
                notes: "W44-138-P1-target",
            },
            // W44-138 spec cells — additional photos + screenshots at the
            // same e8 d=4-5 band.
            Cell {
                image: "terminal".into(),
                path: gb82.join("terminal.png"),
                distance: 4.0,
                effort: 8,
                notes: "W44-138-screen",
            },
            Cell {
                image: "codec_wiki".into(),
                path: gb82.join("codec_wiki.png"),
                distance: 4.0,
                effort: 8,
                notes: "W44-138-screen",
            },
            Cell {
                image: "1025469".into(),
                path: cid22.join("1025469.png"),
                distance: 2.0,
                effort: 8,
                notes: "W44-138-spec-monotonic-photo",
            },
            Cell {
                image: "1025469".into(),
                path: cid22.join("1025469.png"),
                distance: 4.0,
                effort: 8,
                notes: "W44-138-spec-photo",
            },
            // W44-117 distance gate test — terminal at d=0.8 (W44-117 doesn't
            // fire below d=1.0 per W44-120, but the same screen-class still
            // applies; below the gate).
            Cell {
                image: "terminal".into(),
                path: gb82.join("terminal.png"),
                distance: 0.8,
                effort: 8,
                notes: "W44-117-low-d-screen",
            },
            // Variant Z photos at d=5 (W44-99 LC / W44-100 HC family).
            Cell {
                image: "1420710".into(),
                path: cid22.join("1420710.png"),
                distance: 5.0,
                effort: 8,
                notes: "W44-99-HC-photo",
            },
            Cell {
                image: "1531677".into(),
                path: cid22.join("1531677.png"),
                distance: 5.0,
                effort: 8,
                notes: "W44-99-LC-photo",
            },
            // W44-198 cluster — the highest-EV unresolved photo surface.
            // Lifted from spec'd e7 to e8 because the buttloop is
            // gated at `effort >= 8` (speed_tier <= kKitten); at e7
            // there is no internal-recon target to compare against —
            // the encoder ships whatever adaptive_quant + the
            // post-buttloop transform_and_quantize produce, so the
            // hook returns None and the A8 measurement is vacuous.
            // The W44-198 cluster cell at e8 d=4 IS a buttloop cell.
            Cell {
                image: "3637739".into(),
                path: cid22.join("3637739.png"),
                distance: 4.0,
                effort: 8,
                notes: "W44-198-cluster-photo-e8-lifted",
            },
            // W44-178/183 CfL cluster — lifted to e8 for the same
            // reason (buttloop gate).
            Cell {
                image: "clic_097cb426".into(),
                path: clic.join("097cb426910ba8ce2525dd8bb7fb1777.png"),
                distance: 2.0,
                effort: 8,
                notes: "W44-178-CfL-photo-e8-lifted",
            },
            // imac_dark — large dark screenshot.
            Cell {
                image: "imac_dark".into(),
                path: gb82.join("imac_dark.png"),
                distance: 4.0,
                effort: 8,
                notes: "screen-large-dark",
            },
            // graph — diagram-class screenshot at low d. Lifted from
            // e7 to e8 (buttloop gate).
            Cell {
                image: "graph".into(),
                path: gb82.join("graph.png"),
                distance: 0.5,
                effort: 8,
                notes: "screen-line-art-e8-lifted",
            },
        ]
    }

    // ── Source helpers ──────────────────────────────────────────────────────

    /// Load an sRGB-u8 PNG and return (raw, linear-Img, srgb-Img, w, h).
    fn load_source(
        path: &Path,
    ) -> Option<(Vec<u8>, Img<Vec<RGB<f32>>>, Img<Vec<[u8; 3]>>, u32, u32)> {
        let img = image::open(path).ok()?;
        let (w, h) = (img.width(), img.height());
        let raw = img.to_rgb8().into_raw();
        let lin: Vec<RGB<f32>> = raw
            .chunks(3)
            .map(|c| {
                RGB::new(
                    srgb_u8_to_linear_val(c[0]),
                    srgb_u8_to_linear_val(c[1]),
                    srgb_u8_to_linear_val(c[2]),
                )
            })
            .collect();
        let lin_img = Img::new(lin, w as usize, h as usize);
        let srgb_pixels: Vec<[u8; 3]> = raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let srgb_img = Img::new(srgb_pixels, w as usize, h as usize);
        Some((raw, lin_img, srgb_img, w, h))
    }

    // ── Run ─────────────────────────────────────────────────────────────────

    pub fn run() {
        let cells = cells();
        eprintln!("W44-RECON-DEEP/A8: internal-recon vs decoded vs cjxl scoring");
        eprintln!("Cells: {}", cells.len());
        eprintln!();

        // TSV header (single row per cell).
        println!(
            "image\tdistance\teffort\twidth\theight\tnotes\t\
             bfly_internal\tbfly_decoded\tbfly_oxide\tbfly_cjxl\t\
             ssim2_internal\tssim2_decoded\tssim2_oxide\tssim2_cjxl\t\
             delta_bfly_internal_vs_decoded\tdelta_bfly_decoded_vs_cjxl\t\
             delta_ssim2_internal_vs_decoded\tdelta_ssim2_decoded_vs_cjxl\t\
             ours_bytes\tcjxl_bytes\t\
             ours_encode_ms\tcjxl_encode_ms\tstatus"
        );

        for (i, cell) in cells.iter().enumerate() {
            eprintln!(
                "[{}/{}] {} d={} e={} ({})",
                i + 1,
                cells.len(),
                cell.image,
                cell.distance,
                cell.effort,
                cell.notes
            );
            if !cell.path.exists() {
                eprintln!("  MISSING: {}", cell.path.display());
                println!(
                    "{}\t{}\t{}\t0\t0\t{}\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     0\t0\t0\t0\tMISSING_SRC",
                    cell.image, cell.distance, cell.effort, cell.notes
                );
                continue;
            }

            let Some((raw, orig_lin, orig_srgb, w, h)) = load_source(&cell.path) else {
                eprintln!("  LOAD FAIL");
                println!(
                    "{}\t{}\t{}\t0\t0\t{}\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     0\t0\t0\t0\tLOAD_FAIL",
                    cell.image, cell.distance, cell.effort, cell.notes
                );
                continue;
            };

            // 1. Encode ours + capture buttloop internal recon.
            let t0 = Instant::now();
            let Some((bitstream, recon)) =
                encode_ours_with_recon_capture(&raw, w, h, cell.distance, cell.effort)
            else {
                eprintln!("  OURS ENCODE FAIL");
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     NaN\tNaN\tNaN\tNaN\t\
                     0\t0\t0\t0\tOURS_ENCODE_FAIL",
                    cell.image, cell.distance, cell.effort, w, h, cell.notes
                );
                continue;
            };
            let ours_encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let ours_bytes = bitstream.len();
            eprintln!(
                "  ours encoded ({} B in {:.0} ms, recon {}x{}, iter {}/{})",
                ours_bytes, ours_encode_ms, recon.width, recon.height, recon.iter, recon.iters
            );
            if recon.width != w as usize || recon.height != h as usize {
                eprintln!(
                    "  RECON DIM MISMATCH: {}x{} vs source {}x{}",
                    recon.width, recon.height, w, h
                );
            }

            // 2. Score internal recon (linear-RGB planar) vs source.
            let (bfly_internal, ssim2_internal) = score_planar(
                &orig_lin,
                &orig_srgb,
                &recon.r,
                &recon.g,
                &recon.b,
                recon.width,
                recon.height,
            );
            eprintln!(
                "  internal:  bfly={:.4}  ssim2={:.4}",
                bfly_internal, ssim2_internal
            );

            // 3. Decode shipped bitstream via jxl-rs + score.
            let (bfly_decoded, ssim2_decoded) =
                if let Some((dw, dh, dec_lin)) = decode_jxl_rs_linear(&bitstream) {
                    if dw == w as usize && dh == h as usize {
                        score_linear(&orig_lin, &orig_srgb, &dec_lin, dw, dh)
                    } else {
                        eprintln!("  jxl-rs decode dim mismatch: {}x{}", dw, dh);
                        (f64::NAN, f64::NAN)
                    }
                } else {
                    eprintln!("  jxl-rs decode failed");
                    (f64::NAN, f64::NAN)
                };
            eprintln!(
                "  decoded:   bfly={:.4}  ssim2={:.4}",
                bfly_decoded, ssim2_decoded
            );

            // 4. Decode via jxl-oxide + score.
            let (bfly_oxide, ssim2_oxide) =
                if let Some((dw, dh, dec_lin)) = decode_jxl_oxide_linear(&bitstream) {
                    if dw == w as usize && dh == h as usize {
                        score_linear(&orig_lin, &orig_srgb, &dec_lin, dw, dh)
                    } else {
                        eprintln!("  jxl-oxide decode dim mismatch: {}x{}", dw, dh);
                        (f64::NAN, f64::NAN)
                    }
                } else {
                    eprintln!("  jxl-oxide decode failed");
                    (f64::NAN, f64::NAN)
                };
            eprintln!(
                "  oxide:     bfly={:.4}  ssim2={:.4}",
                bfly_oxide, ssim2_oxide
            );

            // 5. Encode cjxl, decode via jxl-rs, score.
            let t_cjxl = Instant::now();
            let (cjxl_bytes, bfly_cjxl, ssim2_cjxl, cjxl_encode_ms) =
                if let Some(cjxl_bits) = encode_cjxl(&cell.path, cell.effort, cell.distance) {
                    let cms = t_cjxl.elapsed().as_secs_f64() * 1000.0;
                    let cbytes = cjxl_bits.len();
                    let (b, s) = if let Some((dw, dh, dec_lin)) = decode_jxl_rs_linear(&cjxl_bits) {
                        if dw == w as usize && dh == h as usize {
                            score_linear(&orig_lin, &orig_srgb, &dec_lin, dw, dh)
                        } else {
                            eprintln!("  cjxl jxl-rs decode dim mismatch: {}x{}", dw, dh);
                            (f64::NAN, f64::NAN)
                        }
                    } else {
                        eprintln!("  cjxl jxl-rs decode failed");
                        (f64::NAN, f64::NAN)
                    };
                    (cbytes, b, s, cms)
                } else {
                    eprintln!("  cjxl encode failed");
                    (0, f64::NAN, f64::NAN, 0.0)
                };
            eprintln!(
                "  cjxl:      bfly={:.4}  ssim2={:.4}",
                bfly_cjxl, ssim2_cjxl
            );

            let delta_bfly_internal_vs_decoded = bfly_internal - bfly_decoded;
            let delta_bfly_decoded_vs_cjxl = bfly_decoded - bfly_cjxl;
            let delta_ssim2_internal_vs_decoded = ssim2_internal - ssim2_decoded;
            let delta_ssim2_decoded_vs_cjxl = ssim2_decoded - ssim2_cjxl;

            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t\
                 {:.4}\t{:.4}\t{:.4}\t{:.4}\t\
                 {:.4}\t{:.4}\t{:.4}\t{:.4}\t\
                 {:+.4}\t{:+.4}\t\
                 {:+.4}\t{:+.4}\t\
                 {}\t{}\t{:.1}\t{:.1}\tOK",
                cell.image,
                cell.distance,
                cell.effort,
                w,
                h,
                cell.notes,
                bfly_internal,
                bfly_decoded,
                bfly_oxide,
                bfly_cjxl,
                ssim2_internal,
                ssim2_decoded,
                ssim2_oxide,
                ssim2_cjxl,
                delta_bfly_internal_vs_decoded,
                delta_bfly_decoded_vs_cjxl,
                delta_ssim2_internal_vs_decoded,
                delta_ssim2_decoded_vs_cjxl,
                ours_bytes,
                cjxl_bytes,
                ours_encode_ms,
                cjxl_encode_ms,
            );
        }

        eprintln!();
        eprintln!("Done. See TSV stdout.");
    }
}
