//! Paired A/B bench for the issue #43 chunk 2a VarDCT
//! `adapt_to_image_lossy` per-image dispatch — pixel + distance gate
//! that drops `try_dct32=false` on tiny + very-low-d cells at
//! effort >= 7.
//!
//! Per-cell A/B (interleaved sample-major):
//!  A = dispatch ON (default; `LossyConfig` does
//!      `effective_profile_for_image -> adapt_to_image_lossy`, which
//!      drops `try_dct32=true→false` when
//!      `pixels < LOSSY_TINY_IMAGE_PIXEL_THRESHOLD (100_000)` AND
//!      `distance < LOSSY_VERY_LOW_DISTANCE_THRESHOLD (0.5)` AND
//!      `effort >= 7`). Note that chunk 1 ALSO fires on the same
//!      cell (drops `try_dct64=false`); chunk 2a is an additional
//!      drop on top.
//!  B = dispatch OFF (force `try_dct32=true` AND `try_dct64=true`
//!      via `__expert` `LossyInternalParams` — the override-respect
//!      logic in `effective_profile_for_image` skips the adapter
//!      so the override survives).
//!
//! Cells (5 total):
//! - Gate-firing (chunk 2a fires + chunk 1 fires):
//!   tiny_256x256 (65,536 px) at d=0.25  (e7)
//!   tiny_100x100 (10,000 px) at d=0.25  (e7)
//!   tiny_200x200 (40,000 px) at d=0.4   (e7)
//! - Control (chunk 2a does NOT fire; chunk 1 may or may not):
//!   ctrl_256x256 at d=0.5  (distance at threshold — gate NOT firing)
//!   ctrl_400x400 (160,000 px) at d=0.25  (pixel above 100k — gate NOT firing)
//!
//! Usage:
//!   cargo run --release -p jxl-encoder \
//!       --features 'std parallel __expert' \
//!       --example issue_43_chunk_2a_dct32_paired_ab \
//!       > benchmarks/issue_43_chunk_2a_2026-05-25.tsv

use jxl_encoder::api::{LossyConfig, PixelLayout};
use jxl_encoder::effort::LossyInternalParams;
use sha2::Digest;
use std::time::Instant;

/// Cells: (label, width, height, distance, gate_fires, use_real_photo)
/// Real photo cells use 7256805.png cropped to fit.
const CELLS: &[(&str, u32, u32, f32, bool, bool)] = &[
    // Synthetic gate-firing (chunk 2a fires).
    ("synth_tiny_256x256", 256, 256, 0.25, true, false), //  65,536 px,  d<0.5
    ("synth_tiny_100x100", 100, 100, 0.25, true, false), //  10,000 px,  d<0.5
    ("synth_tiny_200x200", 200, 200, 0.40, true, false), //  40,000 px,  d<0.5
    // Real photo gate-firing (chunk 2a fires).
    ("photo_tiny_256x256", 256, 256, 0.25, true, true), //  65,536 px,  d<0.5
    ("photo_tiny_256x256_d04", 256, 256, 0.40, true, true), //  d=0.4
    // Control (chunk 2a does NOT fire).
    ("ctrl_256x256_d05", 256, 256, 0.5, false, false), //  d=0.5 (== threshold, NOT <)
    ("ctrl_400x400", 400, 400, 0.25, false, false),    // 160,000 px, above pixel gate
];

const REAL_PHOTO_PATH: &str = "/home/lilith/work/codec-corpus/CID22/CID22-512/training/7256805.png";

fn parse_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Load a real photo (PNG) center-cropped to (w, h).
fn load_real_cropped(path: &str, w: u32, h: u32) -> Vec<u8> {
    let img = image::open(path).expect("open photo").to_rgb8();
    let (iw, ih) = img.dimensions();
    let cw = w.min(iw);
    let ch = h.min(ih);
    let x0 = (iw - cw) / 2;
    let y0 = (ih - ch) / 2;
    let mut out = Vec::with_capacity((cw * ch * 3) as usize);
    for y in y0..(y0 + ch) {
        let row_start = ((y * iw + x0) * 3) as usize;
        let row_end = ((y * iw + x0 + cw) * 3) as usize;
        out.extend_from_slice(&img.as_raw()[row_start..row_end]);
    }
    out
}

/// Generate a synthetic RGB8 image with mixed structure (smooth gradient,
/// edges, speckle, and a 32×32 flat patch) so AC strategy search has
/// real work — DCT32 candidates can actually win on the smooth patches
/// when not gated out.
fn synthetic_rgb8(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    let mut state: u32 = 0x1357_9BDF;
    for y in 0..h {
        for x in 0..w {
            // Quadrant-based mix: smooth diagonal / bars+speckle / flat / glyph.
            let qy = (y * 4) / h.max(1);
            let (r, g, b) = if qy == 0 {
                let v = ((x + y) * 255 / (w + h).max(1)) as u8;
                (v, v.wrapping_add(20), v.wrapping_sub(20))
            } else if qy == 1 {
                let bars_g = if (x / 4) % 2 == 0 { 30u8 } else { 220u8 };
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let speckle = ((state >> 24) as u8) & 0x3F;
                let bx = (((x ^ y) as u8).wrapping_add(speckle)) | 0x10;
                ((x as u8) ^ 0x55, bars_g, bx)
            } else if qy == 2 {
                let qx = (x * 4) / w.max(1);
                let base = (qx as u8) * 50 + 40;
                (base, base.wrapping_add(20), base.wrapping_sub(10))
            } else {
                let lx = x % 8;
                let ly = y % 8;
                let on = lx == 3 || lx == 4 || ly == 3 || ly == 4;
                if on { (240, 230, 220) } else { (10, 18, 30) }
            };
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

/// Variant A: dispatch ON (the new default for chunk 2a). Uses
/// `LossyConfig` straight — `effective_profile_for_image` does the
/// per-image flips when tiny + very-low-d + effort >= 7.
fn encode_dispatch_on(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    threads: usize,
) -> (usize, f64, u64) {
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut hsh = sha2::Sha256::new();
    hsh.update(&bytes);
    let digest = hsh.finalize();
    let prefix = u64::from_be_bytes(digest[..8].try_into().unwrap());
    (bytes.len(), ms, prefix)
}

/// Variant B: dispatch OFF (pre-chunk-2a baseline). Pins
/// `try_dct32=Some(true)` AND `try_dct64=Some(true)` via the `__expert`
/// internal-params override, which `effective_profile_for_image`
/// honours by skipping the per-image adapter. At control cells where
/// neither chunk fires, A and B produce byte-identical output.
fn encode_dispatch_off(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    threads: usize,
) -> (usize, f64, u64) {
    let mut params = LossyInternalParams::default();
    params.try_dct32 = Some(true);
    params.try_dct64 = Some(true);
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads)
        .with_internal_params(params);
    let start = Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).expect("encode");
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut hsh = sha2::Sha256::new();
    hsh.update(&bytes);
    let digest = hsh.finalize();
    let prefix = u64::from_be_bytes(digest[..8].try_into().unwrap());
    (bytes.len(), ms, prefix)
}

fn refresh_marker(activity: &str) {
    let repo_root = std::env::var("REPO_ROOT")
        .unwrap_or_else(|_| "/home/lilith/work/zen/jxl-encoder--issue-43-chunk-2a".to_string());
    let path = format!("{}/.workongoing", repo_root);
    let _ = std::fs::write(path, format!("issue-43-chunk-2a bench {}\n", activity));
}

fn main() {
    let samples = parse_usize("SAMPLES", 5);
    let threads = parse_usize("THREADS", 1);
    let effort: u8 = std::env::var("EFFORT")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(7);

    eprintln!(
        "# issue_43_chunk_2a paired A/B\n# samples={} threads={} effort={}",
        samples, threads, effort
    );
    println!("# issue_43_chunk_2a paired A/B");
    println!("# A = dispatch ON  (chunk 2a fires: try_dct32=false on tiny+very-low-d at e>=7)");
    println!("# B = dispatch OFF (force try_dct32=true AND try_dct64=true via __expert override)");
    println!(
        "# Note: chunk 1 (try_dct64) also fires on the chunk 2a cell; pinning both for fair A/B."
    );
    println!(
        "# threads={} effort={} samples={}",
        threads, effort, samples
    );
    println!(
        "label\twidth\theight\tpixels\tdistance\teffort\tgate_fires\tsample\tvariant\tbytes\tencode_ms\tsha256_prefix"
    );

    // Materialize per-cell images once.
    let cells: Vec<(String, u32, u32, f32, bool, Vec<u8>)> = CELLS
        .iter()
        .map(|(label, w, h, d, gate, use_real)| {
            let rgb = if *use_real {
                load_real_cropped(REAL_PHOTO_PATH, *w, *h)
            } else {
                synthetic_rgb8(*w, *h)
            };
            (label.to_string(), *w, *h, *d, *gate, rgb)
        })
        .collect();

    // Warm-up at the smallest A/B per cell so first-run effects
    // amortize.
    for (label, w, h, d, _, rgb) in &cells {
        refresh_marker(&format!("warmup {} d{:.2}", label, d));
        let _ = encode_dispatch_on(rgb, *w, *h, *d, effort, threads);
        let _ = encode_dispatch_off(rgb, *w, *h, *d, effort, threads);
    }

    // Sample-major interleave: each sample s walks every cell,
    // alternating A then B so paired Δ stays thermally close.
    for s in 1..=samples {
        for (label, w, h, d, gate, rgb) in &cells {
            let px = (*w as u64) * (*h as u64);
            refresh_marker(&format!("sample {}/{}: {} d{:.2} A", s, samples, label, d));
            let (bytes_a, ms_a, sha_a) = encode_dispatch_on(rgb, *w, *h, *d, effort, threads);
            println!(
                "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\tA\t{}\t{:.2}\t{:016x}",
                label, *w, *h, px, d, effort, gate, s, bytes_a, ms_a, sha_a
            );

            refresh_marker(&format!("sample {}/{}: {} d{:.2} B", s, samples, label, d));
            let (bytes_b, ms_b, sha_b) = encode_dispatch_off(rgb, *w, *h, *d, effort, threads);
            println!(
                "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\tB\t{}\t{:.2}\t{:016x}",
                label, *w, *h, px, d, effort, gate, s, bytes_b, ms_b, sha_b
            );
        }
    }

    refresh_marker("paired A/B complete");
}
