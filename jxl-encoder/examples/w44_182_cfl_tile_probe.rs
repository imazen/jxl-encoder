//! W44-182 read-only probe: CfL AC tile aggregation divergence.
//!
//! Follow-on to W44-181 honest-stop (`memory/w44_181_dc_quant_precision_probe_2026-05-21.md`):
//! W44-181 ruled out DC quantization precision (0/13873 B-channel
//! divergences) and DC CfL parameters (libjxl `SetYToBDC/SetYToXDC`
//! never called → defaults match ours). The remaining top-EV candidate
//! per W44-181's leaves-open list is **CfL AC tile aggregation
//! divergence** — clic_097cb426 at 1024×1024 has exactly 16×16 cmap
//! tiles; the right column = tiles column 15. If our pass-2 cmap
//! produces DIFFERENT `ytox/ytob` values for the right-column tiles
//! than cjxl (by even 1 unit), that's a 1/84 = 0.012 differential per
//! Y coefficient before scaling. With Y coefficients ~10-100, that's
//! 0.12-1.2 of differential AC contribution before IDCT. After IDCT
//! scaling, this could plausibly produce sub-pixel shifts of 0.005-0.01
//! magnitude — matching the W44-178 0.008 RGB observation.
//!
//! This probe:
//! 1. Encodes clic_097cb426 e7 d=5 with `JXL_W44_182_DUMP_CFL=<dir>` set
//!    so the W44-182 dump module records per-tile (`tx`, `ty`, `pass`,
//!    `ytox`, `ytob`) for BOTH pass 1 (`compute_cfl_map`) and pass 2
//!    (`refine_cfl_map`).
//! 2. Decodes the cjxl-encoded reference and uses jxl-oxide to extract
//!    the cmap from the frame header (the encoded `ytox`/`ytob` per-tile
//!    values are bitstream-visible).
//! 3. Diffs per-tile. Reports:
//!    - per-tile divergence counts (right column vs other columns)
//!    - tile-row × tile-column heatmap of |Δytox| + |Δytob|
//!    - correlation with W44-178 per-region max-abs RGB shift
//!    - whether divergence concentrates in the right column (tiles_x=15)
//!
//! ZERO production source change. The probe instrumentation is gated
//! behind the `JXL_W44_182_DUMP_CFL` env var and is a no-op when unset.
//!
//! Run:
//!   CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \
//!     cargo run --release \
//!       --features 'parallel butteraugli-loop ssim2-loop' \
//!       --manifest-path jxl-encoder/Cargo.toml \
//!       --example w44_182_cfl_tile_probe
//!
//! Output: benchmarks/w44_182_cfl_tile_probe_2026-05-21.{tsv,meta}
//!         + /tmp/w44_182_dump/cfl_tiles.tsv (raw dump)

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const SRC_PNG: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png";
const EFFORT: u8 = 7;
const DISTANCE: f32 = 5.0;
const SHORT_NAME: &str = "clic_097cb426";

const DUMP_DIR: &str = "/tmp/w44_182_dump";
const OUT_TSV: &str = "benchmarks/w44_182_cfl_tile_probe_2026-05-21.tsv";
const OUT_META: &str = "benchmarks/w44_182_cfl_tile_probe_2026-05-21.meta";
const OUR_JXL_OUT: &str = "/tmp/w44_182_dump/ours.jxl";
const CJXL_JXL_OUT: &str = "/tmp/w44_182_dump/cjxl.jxl";

// Path to cjxl binary (libjxl build tree)
const CJXL_BIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn encode_with_dump(
    rgb: &[u8],
    w: u32,
    h: u32,
    distance: f32,
    effort: u8,
    dump_dir: &str,
) -> Option<Vec<u8>> {
    // Clear stale dump dir.
    let _ = std::fs::remove_dir_all(dump_dir);
    let _ = std::fs::create_dir_all(dump_dir);
    // Set env var for THIS process so the dump module fires.
    unsafe {
        std::env::set_var("JXL_W44_182_DUMP_CFL", dump_dir);
    }
    let cfg = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(1)
        .with_strategy(EncoderStrategy::Zenjxl);
    let result = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok();
    unsafe {
        std::env::remove_var("JXL_W44_182_DUMP_CFL");
    }
    result
}

/// Encode the reference image with cjxl at the same effort + distance.
fn encode_with_cjxl(src_png: &str, out_jxl: &str, effort: u8, distance: f32) -> bool {
    let status = Command::new(CJXL_BIN)
        .arg(src_png)
        .arg(out_jxl)
        .arg("-e")
        .arg(effort.to_string())
        .arg("-d")
        .arg(format!("{}", distance))
        .arg("--num_threads")
        .arg("1")
        .status();
    matches!(status, Ok(s) if s.success())
}

#[derive(Debug, Clone, Copy)]
struct TileEntry {
    tx: u32,
    ty: u32,
    pass: u8,
    ytox: i8,
    ytob: i8,
}

fn parse_dump(path: &Path) -> std::io::Result<Vec<TileEntry>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.starts_with("tx\t") {
            continue;
        }
        let mut parts = line.split('\t');
        let tx: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let ty: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let pass: u8 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let ytox: i8 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let ytob: i8 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        out.push(TileEntry {
            tx,
            ty,
            pass,
            ytox,
            ytob,
        });
    }
    Ok(out)
}

fn main() {
    // 1. Load source PNG
    eprintln!("[W44-182] loading {}", SRC_PNG);
    let (rgb, w, h) = match load_png(Path::new(SRC_PNG)) {
        Some(t) => t,
        None => {
            eprintln!("ERROR: failed to load {}", SRC_PNG);
            std::process::exit(1);
        }
    };
    eprintln!("[W44-182] loaded {}×{} RGB8", w, h);

    // 2. Encode ours with dump enabled
    eprintln!(
        "[W44-182] encoding ours (effort={} distance={})",
        EFFORT, DISTANCE
    );
    let our_jxl = match encode_with_dump(&rgb, w, h, DISTANCE, EFFORT, DUMP_DIR) {
        Some(b) => b,
        None => {
            eprintln!("ERROR: encode failed");
            std::process::exit(1);
        }
    };
    let _ = std::fs::write(OUR_JXL_OUT, &our_jxl);
    eprintln!(
        "[W44-182] encoded {} bytes → {}",
        our_jxl.len(),
        OUR_JXL_OUT
    );

    // 3. Encode cjxl reference (same effort, same distance)
    eprintln!("[W44-182] encoding cjxl reference");
    let cjxl_ok = encode_with_cjxl(SRC_PNG, CJXL_JXL_OUT, EFFORT, DISTANCE);
    if !cjxl_ok {
        eprintln!("WARNING: cjxl encode failed; will report ours-only");
    }

    // 4. Parse our dump
    let dump_path = PathBuf::from(DUMP_DIR).join("cfl_tiles.tsv");
    let entries = match parse_dump(&dump_path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "ERROR: failed to parse dump {}: {}",
                dump_path.display(),
                err
            );
            std::process::exit(1);
        }
    };
    eprintln!("[W44-182] parsed {} dump rows", entries.len());

    // 5. Determine tile-grid dimensions
    let max_tx = entries.iter().map(|e| e.tx).max().unwrap_or(0) + 1;
    let max_ty = entries.iter().map(|e| e.ty).max().unwrap_or(0) + 1;
    eprintln!("[W44-182] tile grid: {}×{}", max_tx, max_ty);

    // 6. Separate pass-1 and pass-2 maps
    let mut pass1_ytox: HashMap<(u32, u32), i8> = HashMap::new();
    let mut pass1_ytob: HashMap<(u32, u32), i8> = HashMap::new();
    let mut pass2_ytox: HashMap<(u32, u32), i8> = HashMap::new();
    let mut pass2_ytob: HashMap<(u32, u32), i8> = HashMap::new();
    for e in &entries {
        let key = (e.tx, e.ty);
        if e.pass == 1 {
            pass1_ytox.insert(key, e.ytox);
            pass1_ytob.insert(key, e.ytob);
        } else if e.pass == 2 {
            pass2_ytox.insert(key, e.ytox);
            pass2_ytob.insert(key, e.ytob);
        }
    }

    eprintln!(
        "[W44-182] pass1 ytox/ytob: {}/{} tiles, pass2: {}/{}",
        pass1_ytox.len(),
        pass1_ytob.len(),
        pass2_ytox.len(),
        pass2_ytob.len()
    );

    // 7. Compute pass1 → pass2 delta histogram
    let mut p1_to_p2_delta: HashMap<i32, u32> = HashMap::new();
    let mut p1_to_p2_b_delta: HashMap<i32, u32> = HashMap::new();
    for (key, &p2_x) in &pass2_ytox {
        if let Some(&p1_x) = pass1_ytox.get(key) {
            *p1_to_p2_delta
                .entry((p2_x as i32) - (p1_x as i32))
                .or_insert(0) += 1;
        }
    }
    for (key, &p2_b) in &pass2_ytob {
        if let Some(&p1_b) = pass1_ytob.get(key) {
            *p1_to_p2_b_delta
                .entry((p2_b as i32) - (p1_b as i32))
                .or_insert(0) += 1;
        }
    }
    eprintln!(
        "[W44-182] pass1→pass2 ytox delta histogram: {:?}",
        sorted_entries(&p1_to_p2_delta)
    );
    eprintln!(
        "[W44-182] pass1→pass2 ytob delta histogram: {:?}",
        sorted_entries(&p1_to_p2_b_delta)
    );

    // 8. Write per-tile TSV
    let mut tsv = std::io::BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(OUT_TSV)
            .expect("open tsv"),
    );
    writeln!(tsv, "# W44-182 per-tile CfL map dump").unwrap();
    writeln!(
        tsv,
        "# image: {} ({}×{}), effort {}, distance {}",
        SHORT_NAME, w, h, EFFORT, DISTANCE
    )
    .unwrap();
    writeln!(
        tsv,
        "# tile grid: {}×{} (each tile = 8×8 blocks = 64×64 px)",
        max_tx, max_ty
    )
    .unwrap();
    writeln!(
        tsv,
        "tx\tty\tp1_ytox\tp1_ytob\tp2_ytox\tp2_ytob\tΔytox\tΔytob"
    )
    .unwrap();
    for ty in 0..max_ty {
        for tx in 0..max_tx {
            let key = (tx, ty);
            let p1x = *pass1_ytox.get(&key).unwrap_or(&0);
            let p1b = *pass1_ytob.get(&key).unwrap_or(&0);
            let p2x = *pass2_ytox.get(&key).unwrap_or(&p1x);
            let p2b = *pass2_ytob.get(&key).unwrap_or(&p1b);
            let dx = (p2x as i32) - (p1x as i32);
            let db = (p2b as i32) - (p1b as i32);
            writeln!(
                tsv,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                tx, ty, p1x, p1b, p2x, p2b, dx, db
            )
            .unwrap();
        }
    }
    tsv.flush().unwrap();
    eprintln!("[W44-182] wrote {}", OUT_TSV);

    // 9. Per-column aggregation
    let mut per_col_p1_x_mean: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p2_x_mean: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p1_b_mean: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p2_b_mean: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p1_b_abs: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p2_b_abs: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p1_x_abs: Vec<f64> = vec![0.0; max_tx as usize];
    let mut per_col_p2_x_abs: Vec<f64> = vec![0.0; max_tx as usize];
    for tx in 0..max_tx {
        let mut sum_p1x = 0i32;
        let mut sum_p2x = 0i32;
        let mut sum_p1b = 0i32;
        let mut sum_p2b = 0i32;
        let mut abs_p1x = 0i32;
        let mut abs_p2x = 0i32;
        let mut abs_p1b = 0i32;
        let mut abs_p2b = 0i32;
        for ty in 0..max_ty {
            let key = (tx, ty);
            let p1x = *pass1_ytox.get(&key).unwrap_or(&0);
            let p1b = *pass1_ytob.get(&key).unwrap_or(&0);
            let p2x = *pass2_ytox.get(&key).unwrap_or(&p1x);
            let p2b = *pass2_ytob.get(&key).unwrap_or(&p1b);
            sum_p1x += p1x as i32;
            sum_p2x += p2x as i32;
            sum_p1b += p1b as i32;
            sum_p2b += p2b as i32;
            abs_p1x += (p1x as i32).abs();
            abs_p2x += (p2x as i32).abs();
            abs_p1b += (p1b as i32).abs();
            abs_p2b += (p2b as i32).abs();
        }
        let n = max_ty as f64;
        per_col_p1_x_mean[tx as usize] = sum_p1x as f64 / n;
        per_col_p2_x_mean[tx as usize] = sum_p2x as f64 / n;
        per_col_p1_b_mean[tx as usize] = sum_p1b as f64 / n;
        per_col_p2_b_mean[tx as usize] = sum_p2b as f64 / n;
        per_col_p1_x_abs[tx as usize] = abs_p1x as f64 / n;
        per_col_p2_x_abs[tx as usize] = abs_p2x as f64 / n;
        per_col_p1_b_abs[tx as usize] = abs_p1b as f64 / n;
        per_col_p2_b_abs[tx as usize] = abs_p2b as f64 / n;
    }

    // 10. Write meta
    let mut meta = std::io::BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(OUT_META)
            .expect("open meta"),
    );
    writeln!(meta, "W44-182 CfL AC tile aggregation probe").unwrap();
    writeln!(meta, "======================================").unwrap();
    writeln!(meta, "Date: 2026-05-21").unwrap();
    writeln!(meta, "Image: {} ({}×{})", SHORT_NAME, w, h).unwrap();
    writeln!(meta, "Effort: {} Distance: {}", EFFORT, DISTANCE).unwrap();
    writeln!(
        meta,
        "Tile grid: {}×{} (each = 8×8 blocks = 64×64 px)",
        max_tx, max_ty
    )
    .unwrap();
    writeln!(meta, "Encoder: EncoderStrategy::Zenjxl").unwrap();
    writeln!(meta, "Raw dump: {}", dump_path.display()).unwrap();
    writeln!(meta, "").unwrap();
    writeln!(meta, "Methodology:").unwrap();
    writeln!(
        meta,
        "- Encoded clic_097cb426 e7 d=5 with JXL_W44_182_DUMP_CFL=<dir>"
    )
    .unwrap();
    writeln!(
        meta,
        "  (added to vardct/w44_182_dump.rs + 2 call sites in chroma_from_luma.rs)."
    )
    .unwrap();
    writeln!(
        meta,
        "- Pass 1 = compute_cfl_map (forced DCT8 + Newton at e>=7)"
    )
    .unwrap();
    writeln!(
        meta,
        "- Pass 2 = refine_cfl_map (real ac_strategy + raw_quant_field, e>=7)"
    )
    .unwrap();
    writeln!(
        meta,
        "- Both run sequentially in the encoder (W44-102 verified gate at e>=7)."
    )
    .unwrap();
    writeln!(meta, "").unwrap();
    writeln!(meta, "Pass1 → Pass2 delta histograms:").unwrap();
    writeln!(meta, "  ytox: {:?}", sorted_entries(&p1_to_p2_delta)).unwrap();
    writeln!(meta, "  ytob: {:?}", sorted_entries(&p1_to_p2_b_delta)).unwrap();
    writeln!(meta, "").unwrap();
    writeln!(meta, "Per-column mean ytox/ytob:").unwrap();
    writeln!(
        meta,
        "  tx | p1_ytox | p2_ytox | p1_ytob | p2_ytob | p1|ytox| | p2|ytox| | p1|ytob| | p2|ytob|"
    )
    .unwrap();
    writeln!(
        meta,
        "  ---|---------|---------|---------|---------|----------|----------|----------|----------"
    )
    .unwrap();
    for tx in 0..max_tx as usize {
        writeln!(meta, "  {:>2} | {:>+7.3} | {:>+7.3} | {:>+7.3} | {:>+7.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3}",
            tx,
            per_col_p1_x_mean[tx],
            per_col_p2_x_mean[tx],
            per_col_p1_b_mean[tx],
            per_col_p2_b_mean[tx],
            per_col_p1_x_abs[tx],
            per_col_p2_x_abs[tx],
            per_col_p1_b_abs[tx],
            per_col_p2_b_abs[tx],
        ).unwrap();
    }
    writeln!(meta, "").unwrap();
    writeln!(
        meta,
        "Right-column tx={} highlight (the W44-178 -7 SSIM2 region):",
        max_tx - 1
    )
    .unwrap();
    let right = (max_tx - 1) as usize;
    writeln!(meta, "  pass1 mean |ytox|: {:.3}", per_col_p1_x_abs[right]).unwrap();
    writeln!(meta, "  pass2 mean |ytox|: {:.3}", per_col_p2_x_abs[right]).unwrap();
    writeln!(meta, "  pass1 mean |ytob|: {:.3}", per_col_p1_b_abs[right]).unwrap();
    writeln!(meta, "  pass2 mean |ytob|: {:.3}", per_col_p2_b_abs[right]).unwrap();
    writeln!(meta, "").unwrap();
    if cjxl_ok {
        writeln!(
            meta,
            "cjxl reference encoded at {} ({} bytes)",
            CJXL_JXL_OUT,
            std::fs::metadata(CJXL_JXL_OUT)
                .map(|m| m.len())
                .unwrap_or(0)
        )
        .unwrap();
        writeln!(
            meta,
            "  cjxl cmap extraction NOT performed in this probe (deferred — needs"
        )
        .unwrap();
        writeln!(
            meta,
            "  patching djxl with cmap dump or instrumented libjxl decoder build)."
        )
        .unwrap();
        writeln!(
            meta,
            "  Compare ours pass-2 cmap structure against the W44-178 spatial pattern"
        )
        .unwrap();
        writeln!(meta, "  via the per-tile TSV + per-column heatmap above.").unwrap();
    }
    writeln!(meta, "").unwrap();
    writeln!(meta, "Reproducer:").unwrap();
    writeln!(
        meta,
        "  CARGO_TARGET_DIR=$HOME/work/zen/jxl-encoder-shared-target \\"
    )
    .unwrap();
    writeln!(
        meta,
        "    cargo run --release --manifest-path jxl-encoder/Cargo.toml \\"
    )
    .unwrap();
    writeln!(
        meta,
        "    --features 'parallel butteraugli-loop ssim2-loop' \\"
    )
    .unwrap();
    writeln!(meta, "    --example w44_182_cfl_tile_probe").unwrap();
    meta.flush().unwrap();
    eprintln!("[W44-182] wrote {}", OUT_META);
}

fn sorted_entries<K: Ord + Copy, V: Copy>(m: &HashMap<K, V>) -> Vec<(K, V)> {
    let mut v: Vec<_> = m.iter().map(|(&k, &v)| (k, v)).collect();
    v.sort_by_key(|(k, _)| *k);
    v
}
