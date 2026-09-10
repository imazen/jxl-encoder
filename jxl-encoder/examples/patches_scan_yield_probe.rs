//! Does the patches scan earn its wall?
//!
//! `benchmarks/e7_cost_attribution_2026-09-10.*` measured patch DETECTION at
//! 13-72 % of lossy e7 wall (and 20-84 % of e5 on graphics, where issue #43
//! chunk 2a lifts the gate down), returning BYTE-IDENTICAL output on 3 of 4
//! content classes. The W36-3 `PatchesDispatch::Auto` photo-skip gate
//! (`median(per-8x8-block mean of mask1x1) > 60`) was calibrated on 16 CLIC
//! photos plus one synthetic Win95 mockup, so it has never been scored against
//! real graphics.
//!
//! This probe scores it. For every cell it encodes twice — patches ON (the
//! shipped default) and `with_patches(false)` — and records
//! [`jxl_encoder::__internals::LastPatchesDetectStats`], every per-stage
//! candidate count the scan already computes. `used` is ground truth: the
//! bytes moved. A cell with `scanned=1 used=0` is pure wasted wall.
//!
//! The question the columns exist to answer: is there a stat available EARLY in
//! the scan that predicts `used`? Phase timing puts steps 1-2 (which produce
//! `num_seeds`) at 0.3 ms of an 11.6 ms scan and the BFS at ~90 %, so a
//! `num_seeds` cut is nearly free while a `bg_count` cut saves almost nothing.
//!
//! Run:
//! ```text
//! cargo run --release --features "__expert __internals" \
//!   --example patches_scan_yield_probe -- <corpus-root> <out.tsv> [per_stratum]
//! ```
use jxl_encoder::__internals::take_last_patches_detect_stats;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::io::Write;
use std::time::Instant;

/// Tiny / small / medium / large, per the sweep-discipline size rule. A size is
/// emitted only when the source spans it in both axes, so nothing is upscaled.
const SIZES: [u32; 4] = [64, 256, 1024, 2048];

/// The full ladder runs at <= 1024. 2048 is expensive, so it gets the sparse
/// one — it exists to show the size trend, not to be the calibration corpus.
const DISTANCES: [f32; 10] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0];
const DISTANCES_LARGE: [f32; 3] = [1.0, 4.0, 10.0];

const EFFORTS: [u8; 2] = [5, 7];

fn centre_crop(rgb: &image::RgbImage, n: u32) -> Vec<u8> {
    let (x0, y0) = ((rgb.width() - n) / 2, (rgb.height() - n) / 2);
    image::imageops::crop_imm(rgb, x0, y0, n, n)
        .to_image()
        .as_raw()
        .clone()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let root = args
        .next()
        .expect("usage: probe <corpus-root> <out.tsv> [n]");
    let out_path = args
        .next()
        .expect("usage: probe <corpus-root> <out.tsv> [n]");
    let per_stratum: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    // Deterministic pick: strata sorted, then the first `per_stratum` SDR PNGs
    // of each in sorted order. Random sampling over-represents the modal class.
    let mut strata: Vec<_> = std::fs::read_dir(&root)
        .expect("corpus root")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    strata.sort();

    let mut picks: Vec<(String, std::path::PathBuf)> = Vec::new();
    for s in &strata {
        let stratum = s.file_name().unwrap().to_string_lossy().to_string();
        let mut found: Vec<std::path::PathBuf> = Vec::new();
        let mut stack = vec![s.clone()];
        while let Some(dir) = stack.pop() {
            let mut entries: Vec<_> = match std::fs::read_dir(&dir) {
                Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path()).collect(),
                Err(_) => continue,
            };
            entries.sort();
            for p in entries {
                if p.is_dir() {
                    stack.push(p);
                } else if p.to_string_lossy().ends_with(".sdr.png") {
                    found.push(p);
                }
            }
        }
        found.sort();
        for p in found.into_iter().take(per_stratum) {
            picks.push((stratum.clone(), p));
        }
    }
    eprintln!("{} images across {} strata", picks.len(), strata.len());

    let mut out = std::fs::File::create(&out_path).expect("create tsv");
    writeln!(
        out,
        "stratum\timage\tsize\tdistance\teffort\tscanned\tused\tbytes_on\tbytes_off\t\
ms_on\tms_off\tnum_seeds\tbg_count\traw_ccs\treject_no_border\treject_inconsistent\t\
reject_too_large\treject_no_similar\treject_low_peak\taccepted_ccs\taccepted_pixels\t\
unique_before_min_occ\tsingletons_dropped\tfinal_unique\tfinal_occurrences\t\
final_total_patch_pixels"
    )
    .unwrap();

    for (i, (stratum, path)) in picks.iter().enumerate() {
        let rgb = match image::open(path) {
            Ok(im) => im.to_rgb8(),
            Err(e) => {
                eprintln!("skip {}: {e}", path.display());
                continue;
            }
        };
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        eprintln!(
            "[{}/{}] {stratum}/{name} {}x{}",
            i + 1,
            picks.len(),
            rgb.width(),
            rgb.height()
        );
        for &n in &SIZES {
            if rgb.width() < n || rgb.height() < n {
                continue;
            }
            let src = centre_crop(&rgb, n);
            let ds: &[f32] = if n >= 2048 {
                &DISTANCES_LARGE
            } else {
                &DISTANCES
            };
            for &d in ds {
                for &e in &EFFORTS {
                    // min of 2, arms interleaved, so neither arm eats every
                    // cold start (see e7_cost_attribution.rs).
                    let (mut ms_on, mut ms_off) = (f64::INFINITY, f64::INFINITY);
                    let (mut bytes_on, mut bytes_off) = (0usize, 0usize);
                    let mut stats = None;
                    for rep in 0..2 {
                        for k in 0..2 {
                            let on = (k + rep) % 2 == 0;
                            let cfg = LossyConfig::new(d).with_effort(e);
                            let cfg = if on { cfg } else { cfg.with_patches(false) };
                            let _ = take_last_patches_detect_stats();
                            let t = Instant::now();
                            let enc = cfg
                                .encode_request(n, n, PixelLayout::Rgb8)
                                .encode(&src)
                                .expect("encode");
                            let ms = t.elapsed().as_secs_f64() * 1000.0;
                            let s = take_last_patches_detect_stats();
                            if on {
                                ms_on = ms_on.min(ms);
                                bytes_on = enc.len();
                                if s.is_some() {
                                    stats = s;
                                }
                            } else {
                                ms_off = ms_off.min(ms);
                                bytes_off = enc.len();
                            }
                        }
                    }
                    let scanned = u8::from(stats.is_some());
                    let used = u8::from(bytes_on != bytes_off);
                    let s = stats.unwrap_or_default();
                    writeln!(
                        out,
                        "{stratum}\t{name}\t{n}\t{d}\t{e}\t{scanned}\t{used}\t{bytes_on}\t\
{bytes_off}\t{ms_on:.2}\t{ms_off:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        s.num_seeds,
                        s.bg_count,
                        s.raw_ccs,
                        s.reject_no_border,
                        s.reject_inconsistent,
                        s.reject_too_large,
                        s.reject_no_similar,
                        s.reject_low_peak,
                        s.accepted_ccs,
                        s.accepted_pixels,
                        s.unique_before_min_occ,
                        s.singletons_dropped,
                        s.final_unique,
                        s.final_occurrences,
                        s.final_total_patch_pixels
                    )
                    .unwrap();
                }
            }
            out.flush().unwrap();
        }
    }
    eprintln!("wrote {out_path}");
}
