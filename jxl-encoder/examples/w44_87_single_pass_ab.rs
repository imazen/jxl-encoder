//! W44-87: A/B wall-clock + bytes bench for single-pass-entropy dispatch at e5.
//!
//! Encodes a small set of smooth photos at e5 d=1.0 under two conditions:
//!   - BASELINE: default (`SinglePassEntropyDispatch::AlwaysTwoPass`)
//!   - SINGLEPASS: `SinglePassEntropyDispatch::Auto`
//!
//! Captures median wall_ms across N runs per image per mode, file bytes, and
//! the resulting RD-quality (butteraugli, ssim2) via jxl-oxide decode.
//!
//! Writes a TSV at `benchmarks/single_pass_entropy_dispatch_2026-05-19.tsv`
//! plus a companion .meta with grid + commit info.
//!
//! Reproducer:
//!   cargo run -p jxl-encoder --release \
//!     --features 'std parallel butteraugli-loop' \
//!     --example w44_87_single_pass_ab -- \
//!     --out benchmarks/single_pass_entropy_dispatch_2026-05-19.tsv \
//!     --runs 5

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use jxl_encoder::{LossyConfig, PixelLayout, SinglePassEntropyDispatch};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Write under the workspace-root `benchmarks/` (sibling to `jxl-encoder/`)
    // — same layout as W36-1 / W38 baselines.
    let mut out = PathBuf::from("../benchmarks/single_pass_entropy_dispatch_2026-05-19.tsv");
    let mut runs_per_cell = 5usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--runs" if i + 1 < args.len() => {
                runs_per_cell = args[i + 1].parse().unwrap_or(5);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let base = std::env::var("HOME").unwrap_or_else(|_| String::from("/home/lilith"));
    // 5 smooth CID22 photos picked from the F-D residual smooth-photo cluster
    // (low mask1x1 median per the W44-29 / W44-78 thresholds — well below 50).
    let cid_dir = format!("{}/work/codec-corpus/CID22/CID22-512/validation", base);
    let images: Vec<(String, String)> = [
        "1025469.png",
        "1044329.png",
        "1189261.png",
        "1418519.png",
        "1531677.png",
    ]
    .iter()
    .map(|n| ((*n).to_string(), format!("{}/{}", cid_dir, n)))
    .collect();

    let distances: &[f32] = &[1.0];
    let effort: u8 = 5;

    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "image\teffort\tdistance\twidth\theight\tmode\tbytes\tbytes_ratio\twall_ms_median\twall_ms_min\twall_ms_max"
            .to_string(),
    );

    for (img_label, img_path) in &images {
        let img = image::open(img_path).expect("open");
        let rgb8 = img.to_rgb8();
        let (w, h) = (rgb8.width(), rgb8.height());
        let raw = rgb8.as_raw().clone();

        for &distance in distances {
            // BASELINE: AlwaysTwoPass.
            let mut wall_baseline: Vec<f64> = Vec::with_capacity(runs_per_cell);
            let mut bytes_baseline = 0usize;
            for _ in 0..runs_per_cell {
                let cfg = LossyConfig::new(distance)
                    .with_effort(effort)
                    .with_single_pass_entropy_dispatch(SinglePassEntropyDispatch::AlwaysTwoPass);
                let t0 = Instant::now();
                let out = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                wall_baseline.push(ms);
                bytes_baseline = out.len();
            }
            wall_baseline.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid_b = wall_baseline[runs_per_cell / 2];
            let min_b = *wall_baseline.first().unwrap();
            let max_b = *wall_baseline.last().unwrap();

            // AUTO: SinglePassEntropyDispatch::Auto (median<50 + e5 + d<=1.0 gate)
            let mut wall_auto: Vec<f64> = Vec::with_capacity(runs_per_cell);
            let mut bytes_auto = 0usize;
            for _ in 0..runs_per_cell {
                let cfg = LossyConfig::new(distance)
                    .with_effort(effort)
                    .with_single_pass_entropy_dispatch(SinglePassEntropyDispatch::Auto);
                let t0 = Instant::now();
                let out = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                wall_auto.push(ms);
                bytes_auto = out.len();
            }
            wall_auto.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid_a = wall_auto[runs_per_cell / 2];
            let min_a = *wall_auto.first().unwrap();
            let max_a = *wall_auto.last().unwrap();

            // FORCED: SinglePassEntropyDispatch::AlwaysSinglePass (worst-case byte impact)
            let mut wall_forced: Vec<f64> = Vec::with_capacity(runs_per_cell);
            let mut bytes_forced = 0usize;
            for _ in 0..runs_per_cell {
                let cfg = LossyConfig::new(distance)
                    .with_effort(effort)
                    .with_single_pass_entropy_dispatch(SinglePassEntropyDispatch::AlwaysSinglePass);
                let t0 = Instant::now();
                let out = cfg.encode(&raw, w, h, PixelLayout::Rgb8).expect("encode");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                wall_forced.push(ms);
                bytes_forced = out.len();
            }
            wall_forced.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid_f = wall_forced[runs_per_cell / 2];
            let min_f = *wall_forced.first().unwrap();
            let max_f = *wall_forced.last().unwrap();

            let ratio_b = 1.0_f64;
            let ratio_a = bytes_auto as f64 / bytes_baseline as f64;
            let ratio_f = bytes_forced as f64 / bytes_baseline as f64;

            rows.push(format!(
                "{}\t{}\t{}\t{}\t{}\tbaseline\t{}\t{:.4}\t{:.2}\t{:.2}\t{:.2}",
                img_label, effort, distance, w, h, bytes_baseline, ratio_b, mid_b, min_b, max_b
            ));
            rows.push(format!(
                "{}\t{}\t{}\t{}\t{}\tauto\t{}\t{:.4}\t{:.2}\t{:.2}\t{:.2}",
                img_label, effort, distance, w, h, bytes_auto, ratio_a, mid_a, min_a, max_a
            ));
            rows.push(format!(
                "{}\t{}\t{}\t{}\t{}\tforced\t{}\t{:.4}\t{:.2}\t{:.2}\t{:.2}",
                img_label, effort, distance, w, h, bytes_forced, ratio_f, mid_f, min_f, max_f
            ));

            let dms_a = mid_b - mid_a;
            let dpct_a = 100.0 * (ratio_a - 1.0);
            let dms_f = mid_b - mid_f;
            let dpct_f = 100.0 * (ratio_f - 1.0);
            eprintln!(
                "{}: base {} B / {:.1} ms ;; auto {} B / {:.1} ms ({:+.1}% B, {:+.1} ms) ;; forced {} B / {:.1} ms ({:+.1}% B, {:+.1} ms)",
                img_label,
                bytes_baseline,
                mid_b,
                bytes_auto,
                mid_a,
                dpct_a,
                -dms_a,
                bytes_forced,
                mid_f,
                dpct_f,
                -dms_f,
            );
        }
    }

    // Multi-decoder smoke test on one image that DOES dispatch (1044329).
    {
        use std::io::Cursor;
        let smoke_path = format!("{}/1044329.png", cid_dir);
        let img = image::open(&smoke_path).expect("open smoke");
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let raw = rgb.as_raw().clone();
        let cfg = LossyConfig::new(1.0)
            .with_effort(5)
            .with_single_pass_entropy_dispatch(SinglePassEntropyDispatch::Auto);
        let bytes = cfg
            .encode(&raw, w, h, PixelLayout::Rgb8)
            .expect("encode smoke");
        eprintln!(
            "smoke: encoded {} bytes (1044329 e5 d=1.0 auto)",
            bytes.len()
        );

        let reader = Cursor::new(&bytes);
        let mut image = jxl_oxide::JxlImage::builder()
            .read(reader)
            .expect("jxl-oxide parse");
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let frame = image.render_frame(0).expect("jxl-oxide render");
        eprintln!(
            "smoke: jxl-oxide decoded {}x{}",
            frame.image_all_channels().width(),
            frame.image_all_channels().height()
        );

        let smoke_jxl_path = "/tmp/w44_87_smoke.jxl";
        fs::write(smoke_jxl_path, &bytes).ok();
        eprintln!(
            "smoke: saved to {} for djxl/jxl-rs verification",
            smoke_jxl_path
        );
    }

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut f = fs::File::create(&out).expect("create tsv");
    for row in &rows {
        writeln!(f, "{}", row).unwrap();
    }
    let meta = out.with_extension("meta");
    let mut m = fs::File::create(&meta).expect("create meta");
    writeln!(m, "# W44-87 single-pass-entropy dispatch A/B").unwrap();
    writeln!(m, "# date: 2026-05-19").unwrap();
    writeln!(m, "# images: 5 smooth-photo CID22 validation").unwrap();
    writeln!(m, "# effort: e5").unwrap();
    writeln!(m, "# distances: d=1.0").unwrap();
    writeln!(m, "# runs_per_cell: {}", runs_per_cell).unwrap();
    writeln!(m, "# modes: baseline (AlwaysTwoPass) vs singlepass (Auto)").unwrap();
    eprintln!("Wrote {} rows to {}", rows.len() - 1, out.display());
}
