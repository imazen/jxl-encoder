// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Corpus builder + per-K byte/wall sweep for the sectioned per-group
//! predictor-prune gate (imazen/jxl-encoder#99 item 1).
//!
//! The sectioned local-tree writer prunes each group's candidate predictor
//! set to a FIXED `SECTIONED_PRUNE_PREDICTORS_K = 8` by root cost. Root-cost
//! ranking is a known-bad selector on flat/palette content (see
//! `probe_prune_candidates`' rustdoc and
//! `benchmarks/jxl_probe_prune_2026-08-15.md`), which is why K=4 costs
//! palette screenshots +1.35 % bytes while photos pay +0.02 %
//! (`benchmarks/jxl_sectioned_prune_k_2026-08-28.tsv`). This harness produces
//! the corpus and the measurements a content-adaptive K has to be fitted and
//! validated on.
//!
//! Four modes, run in order:
//!
//! ```text
//! sectioned_k_corpus features <out.tsv> <root>...
//!     Walk every PNG under each root, decode to RGB8, emit one row of
//!     content features per image (plus sha256 + dims). Feature set is
//!     deliberately the axes predictor choice depends on: flat-run mass,
//!     activity-context distribution, per-predictor residual entropies and
//!     the number of DISTINCT context winners.
//!
//! sectioned_k_corpus render <picks.tsv> <outdir> <out_manifest.tsv>
//!     Materialize renditions of the picked origins. `picks.tsv` carries
//!     one row per (origin, kind, target) triple; `kind` is `resize`
//!     (CatmullRom / Mitchell-Netravali B=0 C=0.5 — balanced blur/ringing)
//!     or `crop` (native detail statistics preserved). Upscaling is
//!     REFUSED, not silently performed.
//!
//! sectioned_k_corpus sweep <manifest.tsv> <out.tsv> <effort> <threads> <arm>...
//!     Encode every rendition once per arm and record bytes + wall.
//!     An arm is `k<N>` (JXL_TREE_PRUNE_PREDICTORS=N) or `adapt`
//!     (JXL_SECTIONED_TREE_PREDICTORS=auto, the probe-tree selector).
//!     One process, many encodes: bytes are exact, wall is comparable
//!     within a row (same process, same page cache).
//!
//! ```
//!
//! All modes write TSV with a `#` provenance header.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------- helpers

fn sha256_file(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read for sha");
    let d = Sha256::digest(&bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn provenance(cmd: &str) -> String {
    format!(
        "# {cmd}  commit={}  host={}  date={}\n",
        option_env!("JXL_BENCH_COMMIT").unwrap_or("see .meta"),
        std::env::var("HOSTNAME").unwrap_or_else(|_| "mac".into()),
        std::env::var("JXL_BENCH_DATE").unwrap_or_else(|_| "unset".into()),
    )
}

/// libjxl's `Gradient` (a.k.a. ClampedGradient): clamp(W + N - NW, min, max).
#[inline]
fn gradient(w: i32, n: i32, nw: i32) -> i32 {
    let g = w + n - nw;
    let (lo, hi) = if w < n { (w, n) } else { (n, w) };
    g.clamp(lo, hi)
}

/// Shannon cost in bits of a residual histogram indexed by a coarse
/// magnitude token (matching what the tree learner's cost model sees:
/// bit-length bucket + extra bits).
fn hist_bits(hist: &[u64; 40], ebits: u64) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut bits = 0.0;
    for &c in hist.iter() {
        if c > 0 {
            let p = c as f64 / t;
            bits -= c as f64 * p.log2();
        }
    }
    bits + ebits as f64
}

/// Coarse token for a signed residual: 0 for zero, else 1 + 2*bitlen +
/// sign. Extra bits = bitlen.saturating_sub(1) — the same shape the
/// gather's `gather_token_ebits` uses, at a granularity that is enough to
/// rank predictors.
#[inline]
fn token_of(r: i32) -> (usize, u32) {
    if r == 0 {
        return (0, 0);
    }
    let m = r.unsigned_abs();
    let bl = 32 - m.leading_zeros();
    let t = (2 * bl as usize + usize::from(r < 0)).min(39);
    (t, bl.saturating_sub(1))
}

// The five cheap static predictors the feature pass ranks. Weighted is
// deliberately excluded (its value is context-dependent and it is always
// retained by every prune path, so it carries no discriminating signal).
const NPRED: usize = 5;
#[allow(dead_code)]
const PRED_NAMES: [&str; NPRED] = ["Zero", "West", "North", "NorthWest", "Gradient"];
/// Activity contexts: bit-length of |W - N|, capped at 7 (the same axis
/// `probe_prune_candidates` bins on).
const NCTX: usize = 8;

struct Features {
    w: u32,
    h: u32,
    /// fraction of pixels whose RGB equals the left neighbour
    frac_left_equal: f64,
    /// fraction of pixels whose green Gradient residual is exactly 0
    frac_grad_zero: f64,
    /// distinct RGB triples per sampled pixel (palette-ness), sampled 1-in-4
    uniq_per_px: f64,
    /// mass of the |W-N| == 0 activity context (flat mass)
    frac_ctx_flat: f64,
    /// Shannon entropy of the 8-bin activity-context histogram, in bits
    ctx_entropy: f64,
    /// bits/px of the best static predictor over the whole image
    best_bpp: f64,
    /// median static predictor bits/px divided by the best (root spread —
    /// high means one predictor dominates, low means they tie)
    root_spread: f64,
    /// number of DISTINCT per-context winners over contexts holding >= 1 %
    /// of the mass (the context-conditional analogue of `root_spread`)
    ctx_winners: usize,
    /// per-context winner mass NOT covered by the single global winner
    frac_ctx_mismatch: f64,
}

/// Green-channel feature pass. Green is the modular residual channel that
/// dominates the tree's cost, and the whole feature set is a rank, so the
/// single-channel restriction costs nothing and keeps the pass cheap.
fn features(rgb: &[u8], w: u32, h: u32) -> Features {
    let (wu, hu) = (w as usize, h as usize);
    let at = |x: usize, y: usize| -> i32 { rgb[(y * wu + x) * 3 + 1] as i32 };
    let mut hists = [[0u64; 40]; NPRED];
    let mut ebits = [0u64; NPRED];
    let mut ctx_hists = [[[0u64; 40]; NPRED]; NCTX];
    let mut ctx_ebits = [[0u64; NPRED]; NCTX];
    let mut ctx_mass = [0u64; NCTX];
    let mut left_equal = 0u64;
    let mut grad_zero = 0u64;
    let mut n_px = 0u64;
    let mut uniq = std::collections::HashSet::<u32>::new();
    for y in 0..hu {
        for x in 0..wu {
            let c = at(x, y);
            let wv = if x > 0 { at(x - 1, y) } else { 0 };
            let nv = if y > 0 { at(x, y - 1) } else { wv };
            let nwv = if x > 0 && y > 0 { at(x - 1, y - 1) } else { wv };
            let preds = [0, wv, nv, nwv, gradient(wv, nv, nwv)];
            let act = {
                let d = (wv - nv).unsigned_abs();
                ((32 - d.leading_zeros()) as usize).min(NCTX - 1)
            };
            ctx_mass[act] += 1;
            for p in 0..NPRED {
                let (t, e) = token_of(c - preds[p]);
                hists[p][t] += 1;
                ebits[p] += e as u64;
                ctx_hists[act][p][t] += 1;
                ctx_ebits[act][p] += e as u64;
            }
            if preds[4] == c {
                grad_zero += 1;
            }
            if x > 0 {
                let i = (y * wu + x) * 3;
                if rgb[i..i + 3] == rgb[i - 3..i] {
                    left_equal += 1;
                }
            }
            if (y * wu + x) % 4 == 0 && uniq.len() < (1 << 20) {
                let i = (y * wu + x) * 3;
                uniq.insert((rgb[i] as u32) << 16 | (rgb[i + 1] as u32) << 8 | (rgb[i + 2] as u32));
            }
            n_px += 1;
        }
    }
    let npx = n_px.max(1) as f64;
    let mut costs: Vec<f64> = (0..NPRED)
        .map(|p| hist_bits(&hists[p], ebits[p]) / npx)
        .collect();
    let global_winner = (0..NPRED)
        .min_by(|&a, &b| costs[a].total_cmp(&costs[b]))
        .unwrap();
    let best = costs[global_winner];
    let mut sorted = costs.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[NPRED / 2];
    costs.truncate(NPRED);

    let total_mass: u64 = ctx_mass.iter().sum();
    let tm = total_mass.max(1) as f64;
    let mut ctx_entropy = 0.0;
    for &m in ctx_mass.iter() {
        if m > 0 {
            let p = m as f64 / tm;
            ctx_entropy -= p * p.log2();
        }
    }
    let mut winners = std::collections::BTreeSet::<usize>::new();
    let mut mismatch = 0u64;
    for c in 0..NCTX {
        if (ctx_mass[c] as f64) < 0.01 * tm {
            continue;
        }
        let wnr = (0..NPRED)
            .min_by(|&a, &b| {
                hist_bits(&ctx_hists[c][a], ctx_ebits[c][a])
                    .total_cmp(&hist_bits(&ctx_hists[c][b], ctx_ebits[c][b]))
            })
            .unwrap();
        winners.insert(wnr);
        if wnr != global_winner {
            mismatch += ctx_mass[c];
        }
    }
    Features {
        w,
        h,
        frac_left_equal: left_equal as f64 / npx,
        frac_grad_zero: grad_zero as f64 / npx,
        uniq_per_px: uniq.len() as f64 / (npx / 4.0).max(1.0),
        frac_ctx_flat: ctx_mass[0] as f64 / tm,
        ctx_entropy,
        best_bpp: best,
        root_spread: if best > 0.0 { median / best } else { 1.0 },
        ctx_winners: winners.len(),
        frac_ctx_mismatch: mismatch as f64 / tm,
    }
}

fn walk_pngs(root: &Path, out: &mut Vec<PathBuf>) {
    for e in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
    {
        let p = e.path();
        if p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")) {
            out.push(p.to_path_buf());
        }
    }
    out.sort();
}

// ---------------------------------------------------------------- modes

fn mode_features(args: &[String]) {
    let out_path = &args[0];
    let mut files = Vec::new();
    for root in &args[1..] {
        walk_pngs(Path::new(root), &mut files);
    }
    let mut f = std::fs::File::create(out_path).expect("create out");
    f.write_all(provenance("sectioned_k_corpus features").as_bytes())
        .unwrap();
    writeln!(
        f,
        "path\tsha256\tw\th\tfrac_left_equal\tfrac_grad_zero\tuniq_per_px\tfrac_ctx_flat\t\
         ctx_entropy\tbest_bpp\troot_spread\tctx_winners\tfrac_ctx_mismatch"
    )
    .unwrap();
    let n = files.len();
    for (i, p) in files.iter().enumerate() {
        let img = match image::open(p) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("skip {}: {e}", p.display());
                continue;
            }
        };
        let (w, h) = (img.width(), img.height());
        // Feature pass is O(pixels); cap the analysed region so a 12 MP
        // source costs the same as a 2 MP one. The cap is a top-left crop,
        // never a resample (resampling would change the very statistics
        // being measured).
        const CAP: u32 = 1536;
        let img = if w > CAP || h > CAP {
            img.crop_imm(0, 0, w.min(CAP), h.min(CAP))
        } else {
            img
        };
        let (cw, ch) = (img.width(), img.height());
        let rgb = img.to_rgb8().into_raw();
        let ft = features(&rgb, cw, ch);
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.6}",
            p.display(),
            sha256_file(p),
            w,
            h,
            ft.frac_left_equal,
            ft.frac_grad_zero,
            ft.uniq_per_px,
            ft.frac_ctx_flat,
            ft.ctx_entropy,
            ft.best_bpp,
            ft.root_spread,
            ft.ctx_winners,
            ft.frac_ctx_mismatch
        )
        .unwrap();
        let _ = (ft.w, ft.h);
        if i % 50 == 0 {
            eprintln!("[features] {i}/{n}");
            f.flush().unwrap();
        }
    }
    eprintln!("[features] done {n}");
}

fn mode_render(args: &[String]) {
    let picks = &args[0];
    let outdir = Path::new(&args[1]);
    let manifest_path = &args[2];
    std::fs::create_dir_all(outdir).expect("mkdir outdir");
    let text = std::fs::read_to_string(picks).expect("read picks");
    let mut mf = std::fs::File::create(manifest_path).expect("create manifest");
    mf.write_all(provenance("sectioned_k_corpus render").as_bytes())
        .unwrap();
    writeln!(
        mf,
        "rendition\torigin\torigin_sha256\tcluster\tsplit\tkind\tw\th\tsha256\tpath"
    )
    .unwrap();
    let mut cache: Option<(String, image::DynamicImage)> = None;
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c[0] == "origin" {
            continue; // header
        }
        // origin \t cluster \t split \t kind \t target
        let (origin, cluster, split, kind, target) = (c[0], c[1], c[2], c[3], c[4]);
        let target: u32 = target.parse().expect("target dim");
        let src = match &cache {
            Some((p, im)) if p == origin => im.clone(),
            _ => {
                let im = image::open(origin).expect("open origin");
                cache = Some((origin.to_string(), im.clone()));
                im
            }
        };
        let (sw, sh) = (src.width(), src.height());
        let long = sw.max(sh);
        assert!(
            target <= long,
            "UPSCALE REFUSED: target {target} > source long edge {long} for {origin}"
        );
        let out = match kind {
            "resize" => {
                let scale = target as f64 / long as f64;
                let nw = ((sw as f64 * scale).round() as u32).max(1);
                let nh = ((sh as f64 * scale).round() as u32).max(1);
                image::DynamicImage::ImageRgb8(image::imageops::resize(
                    &src.to_rgb8(),
                    nw,
                    nh,
                    image::imageops::FilterType::CatmullRom,
                ))
            }
            "crop" => {
                let cw = target.min(sw);
                let chh = target.min(sh);
                // Centre crop: the middle of a frame carries the subject on
                // photos and the content body on screenshots, where a
                // top-left crop lands on chrome/margins.
                let x = (sw - cw) / 2;
                let y = (sh - chh) / 2;
                image::DynamicImage::ImageRgb8(src.crop_imm(x, y, cw, chh).to_rgb8())
            }
            other => panic!("unknown kind {other:?} (resize|crop)"),
        };
        let stem = Path::new(origin)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let name = format!("c{cluster}_{stem}_{kind}{target}.png");
        let path = outdir.join(&name);
        out.save(&path).expect("save rendition");
        writeln!(
            mf,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            name,
            origin,
            sha256_file(Path::new(origin)),
            cluster,
            split,
            kind,
            out.width(),
            out.height(),
            sha256_file(&path),
            path.display()
        )
        .unwrap();
        eprintln!("[render] {name} {}x{}", out.width(), out.height());
    }
    eprintln!("[render] done");
}

struct Rendition {
    name: String,
    origin: String,
    cluster: String,
    split: String,
    kind: String,
    path: String,
}

fn read_manifest(path: &str) -> Vec<Rendition> {
    let text = std::fs::read_to_string(path).expect("read manifest");
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() || line.starts_with("rendition\t") {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        out.push(Rendition {
            name: c[0].into(),
            origin: c[1].into(),
            cluster: c[3].into(),
            split: c[4].into(),
            kind: c[5].into(),
            path: c[9].into(),
        });
    }
    out
}

fn encode_once(rgb: &[u8], w: u32, h: u32, effort: u8, threads: usize) -> (usize, f64) {
    use jxl_encoder::api::SectionedTrees;
    use jxl_encoder::{LosslessConfig, PixelLayout};
    let t0 = std::time::Instant::now();
    let enc = LosslessConfig::new()
        .with_effort(effort)
        .with_threads(threads)
        .with_sectioned_trees(SectionedTrees::On)
        .encode_request(w, h, PixelLayout::Rgb8)
        .encode(rgb)
        .expect("encode");
    (enc.len(), t0.elapsed().as_secs_f64() * 1000.0)
}

fn apply_arm(arm: &str) {
    // SAFETY: single-threaded driver loop, called between encodes. No
    // encoder thread exists at this point (the rayon pool is built inside
    // `encode`), so nothing else reads the environment concurrently.
    // (set_var is unsafe in the 2024 edition; the harness controls when.)
    unsafe {
        std::env::remove_var("JXL_TREE_PRUNE_PREDICTORS");
        std::env::remove_var("JXL_SECTIONED_TREE_PREDICTORS");
        std::env::remove_var("JXL_SECTIONED_PROBE_COVERAGE");
        if let Some(n) = arm.strip_prefix("adapt") {
            // `adapt<N>`: probe selector with the cumulative static
            // leaf-mass coverage forced to N percent (the fit axis).
            // Bare `adapt` uses the built-in default.
            std::env::set_var("JXL_SECTIONED_TREE_PREDICTORS", "auto");
            if !n.is_empty() {
                std::env::set_var("JXL_SECTIONED_PROBE_COVERAGE", n);
            }
            return;
        }
        if let Some(k) = arm.strip_prefix('k') {
            // A `k<N>` arm is the FIXED-K root-cost prune, so it must also
            // switch the probe selector OFF — otherwise the probe list wins
            // and every k-arm silently measures `adapt`.
            std::env::set_var("JXL_SECTIONED_TREE_PREDICTORS", "off");
            std::env::set_var("JXL_TREE_PRUNE_PREDICTORS", k);
        } else if arm != "default" {
            panic!("unknown arm {arm:?} (k<N>|adapt[<N>]|default)");
        }
    }
}

fn mode_sweep(args: &[String]) {
    let manifest = &args[0];
    let out_path = &args[1];
    let effort: u8 = args[2].parse().expect("effort");
    let threads: usize = args[3].parse().expect("threads");
    let arms: Vec<String> = args[4..].to_vec();
    let rows = read_manifest(manifest);
    let mut f = std::fs::File::create(out_path).expect("create out");
    f.write_all(provenance("sectioned_k_corpus sweep").as_bytes())
        .unwrap();
    writeln!(
        f,
        "rendition\torigin\tcluster\tsplit\tkind\tw\th\teffort\tthreads\tarm\tbytes\twall_ms"
    )
    .unwrap();
    for (i, r) in rows.iter().enumerate() {
        let img = image::open(&r.path).expect("open rendition");
        let (w, h) = (img.width(), img.height());
        let rgb = img.to_rgb8().into_raw();
        drop(img);
        for arm in &arms {
            apply_arm(arm);
            let (bytes, wall) = encode_once(&rgb, w, h, effort, threads);
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}",
                r.name,
                r.origin,
                r.cluster,
                r.split,
                r.kind,
                w,
                h,
                effort,
                threads,
                arm,
                bytes,
                wall
            )
            .unwrap();
        }
        f.flush().unwrap();
        eprintln!("[sweep] {}/{} {}", i + 1, rows.len(), r.name);
    }
    eprintln!("[sweep] done");
}

/// `phases <img.png> <effort> <threads> <arm>...` — per-phase wall split for
/// one image under each arm. Requires the `profile-phases` feature; without
/// it the snapshot is empty and the mode says so instead of printing zeros.
fn mode_phases(args: &[String]) {
    let path = &args[0];
    let effort: u8 = args[1].parse().expect("effort");
    let threads: usize = args[2].parse().expect("threads");
    let img = image::open(path).expect("open");
    let (w, h) = (img.width(), img.height());
    let rgb = img.to_rgb8().into_raw();
    drop(img);
    for arm in &args[3..] {
        apply_arm(arm);
        // Warm the arm once so the reported split is not the first-touch run.
        let _ = encode_once(&rgb, w, h, effort, threads);
        jxl_encoder::__test_exports::profile_phases::reset();
        let (bytes, wall) = encode_once(&rgb, w, h, effort, threads);
        let snap = jxl_encoder::__test_exports::profile_phases::take_snapshot();
        println!("== {arm}: {bytes} bytes, {wall:.1} ms total");
        if snap.is_empty() {
            println!("   (no phase data — build with --features profile-phases)");
        }
        let mut rows: Vec<_> = snap.into_iter().collect();
        rows.sort_by_key(|&(_, ns)| core::cmp::Reverse(ns));
        for (name, ns) in rows {
            let ms = ns as f64 / 1.0e6;
            println!("   {:36} {ms:9.1} ms  {:5.1} %", name, 100.0 * ms / wall);
        }
    }
}

/// `croptl <in.png> <out.png> <W> <H>` — top-left crop, matching what
/// `mem_probe`'s `MEM_PROBE_CROP` does internally, so the same region can be
/// handed to an external encoder (cjxl) for a like-for-like wall comparison.
fn mode_croptl(args: &[String]) {
    let (src, dst) = (&args[0], &args[1]);
    let (w, h) = (
        args[2].parse::<u32>().expect("W"),
        args[3].parse::<u32>().expect("H"),
    );
    let img = image::open(src).expect("open src");
    assert!(
        w <= img.width() && h <= img.height(),
        "crop {w}x{h} exceeds source {}x{}",
        img.width(),
        img.height()
    );
    img.crop_imm(0, 0, w, h)
        .to_rgb8()
        .save(dst)
        .expect("save crop");
    eprintln!(
        "[croptl] {dst} {w}x{h} sha256={}",
        sha256_file(Path::new(dst))
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!(
            "usage:\n  \
             sectioned_k_corpus features <out.tsv> <root>...\n  \
             sectioned_k_corpus render <picks.tsv> <outdir> <out_manifest.tsv>\n  \
             sectioned_k_corpus sweep <manifest.tsv> <out.tsv> <effort> <threads> <arm>...\n  \
             sectioned_k_corpus croptl <in.png> <out.png> <W> <H>"
        );
        std::process::exit(2);
    }
    match a[1].as_str() {
        "features" => mode_features(&a[2..]),
        "render" => mode_render(&a[2..]),
        "sweep" => mode_sweep(&a[2..]),
        "croptl" => mode_croptl(&a[2..]),
        "phases" => mode_phases(&a[2..]),
        other => {
            eprintln!("unknown mode {other:?}");
            std::process::exit(2);
        }
    }
}
