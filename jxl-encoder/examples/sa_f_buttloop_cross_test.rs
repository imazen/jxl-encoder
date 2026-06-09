//! SA-F: cross-test our `butteraugli` crate vs libjxl's `butteraugli_main`
//! on the SAME (ref, recon) inputs to discriminate SA-B's metric-divergence claim.
//!
//! SA-B (commit `33729573`) claimed: ours butteraugli=4.31 vs cjxl
//! butteraugli=10.24 (2.4× ratio) at clic_22ea12 e9 d=4 iter 0. That comparison
//! was iter-0 inside the *encoder buttloop* — both encoders running their own
//! reconstructions through their own metric. So the 2.4× could be (a) the metric
//! itself diverging on the SAME inputs, or (b) the encoders producing different
//! reconstructions that genuinely score differently in BOTH metrics.
//!
//! This test resolves (a) vs (b) by:
//!   1. Encoding the source with both cjxl-rs (ours) and cjxl (libjxl) at e9 d=4.
//!   2. Decoding each .jxl to linear-sRGB f32 via jxl-oxide.
//!   3. Scoring all 4 (ref, recon) combinations using OUR `butteraugli_linear`
//!      AND libjxl's `butteraugli_main` (via PFM round-trip with
//!      `--colorspace RGB_D65_SRG_Rel_Lin` hint).
//!
//! Output: 4-pair score matrix TSV + per-pair PFMs (kept in /tmp for inspection).

use butteraugli::{ButteraugliParams, butteraugli_linear};
use imgref::Img;
use rgb::RGB;
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const SRC: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png";
const CJXL_RS: &str = "/home/lilith/work/zen/jxl-encoder--sa-f-cross-test/target/release/cjxl-rs";
const CJXL: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl";
const BUTTERAUGLI_MAIN: &str = "/home/lilith/work/jxl-efforts/libjxl/build/tools/butteraugli_main";
const OUT_DIR: &str = "/tmp/sa_f";
const BENCH_TSV: &str = "/home/lilith/work/zen/jxl-encoder--sa-f-cross-test/jxl-encoder/benchmarks/sa_f_buttloop_cross_test_2026-05-25.tsv";
const BENCH_META: &str = "/home/lilith/work/zen/jxl-encoder--sa-f-cross-test/jxl-encoder/benchmarks/sa_f_buttloop_cross_test_2026-05-25.meta";

fn srgb_to_linear(v: u8) -> f32 {
    let f = v as f32 / 255.0;
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

fn load_png_linear(path: &Path) -> (Vec<f32>, u32, u32) {
    let img = image::open(path).expect("failed to open PNG");
    let rgb = img.to_rgb8();
    let w = rgb.width();
    let h = rgb.height();
    let lin: Vec<f32> = rgb.as_raw().iter().map(|&v| srgb_to_linear(v)).collect();
    (lin, w, h)
}

fn decode_jxl_linear(jxl_path: &Path) -> (Vec<f32>, u32, u32) {
    let bytes = std::fs::read(jxl_path).expect("read .jxl");
    let reader = Cursor::new(bytes);
    let mut img = jxl_oxide::JxlImage::builder()
        .read(reader)
        .expect("jxl-oxide parse");
    img.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = img.render_frame(0).expect("jxl-oxide render");
    let fb = render.image_all_channels();
    (fb.buf().to_vec(), fb.width() as u32, fb.height() as u32)
}

/// Write a linear-sRGB f32 RGB buffer as binary little-endian PFM.
/// butteraugli_main with `--colorspace RGB_D65_SRG_Rel_Lin` interprets PFM data
/// as linear sRGB. PFM header: "PF\n<w> <h>\n-1.0\n" then raw f32 little-endian
/// rows in bottom-up order.
fn write_pfm(path: &Path, data: &[f32], w: u32, h: u32) {
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("create pfm {}: {}", path.display(), e));
    // -1.0 = little-endian
    write!(f, "PF\n{} {}\n-1.0\n", w, h).unwrap();
    // PFM is bottom-up; write rows in reverse order.
    for y in (0..h as usize).rev() {
        let row_start = y * w as usize * 3;
        let row_end = row_start + w as usize * 3;
        let row_bytes: &[u8] = bytemuck::cast_slice(&data[row_start..row_end]);
        std::io::Write::write_all(&mut f, row_bytes).unwrap();
    }
}

fn run_libjxl_butteraugli(ref_pfm: &Path, dist_pfm: &Path) -> Option<(f64, f64)> {
    // butteraugli_main outputs:
    //   "<distance>\n"
    //   "<p>-norm: <pnorm>\n"
    let out = Command::new(BUTTERAUGLI_MAIN)
        .args([
            ref_pfm.to_str().unwrap(),
            dist_pfm.to_str().unwrap(),
            "--colorspace",
            "RGB_D65_SRG_Rel_Lin",
            "--pnorm",
            "3.0",
        ])
        .output()
        .expect("run butteraugli_main");
    if !out.status.success() {
        eprintln!(
            "butteraugli_main failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut max_score: Option<f64> = None;
    let mut pnorm: Option<f64> = None;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.contains("-norm:") {
            // e.g. "3-norm: 1.234"
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() == 2 {
                pnorm = parts[1].parse::<f64>().ok();
            }
        } else if let Ok(v) = trimmed.parse::<f64>() {
            // First numeric-only line is the max distance.
            if max_score.is_none() {
                max_score = Some(v);
            }
        }
    }
    Some((max_score?, pnorm?))
}

fn run_ours_butteraugli(ref_lin: &[f32], dist_lin: &[f32], w: u32, h: u32) -> (f64, f64) {
    let ref_pixels: Vec<RGB<f32>> = ref_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let dist_pixels: Vec<RGB<f32>> = dist_lin
        .chunks(3)
        .map(|c| RGB::new(c[0], c[1], c[2]))
        .collect();
    let ref_img: Img<Vec<RGB<f32>>> = Img::new(ref_pixels, w as usize, h as usize);
    let dist_img: Img<Vec<RGB<f32>>> = Img::new(dist_pixels, w as usize, h as usize);
    let res = butteraugli_linear(
        ref_img.as_ref(),
        dist_img.as_ref(),
        &ButteraugliParams::default().with_compute_diffmap(true),
    )
    .expect("our butteraugli");
    let pnorm = res.pnorm(3.0).unwrap_or(0.0);
    (res.score as f64, pnorm)
}

fn main() {
    std::fs::create_dir_all(OUT_DIR).unwrap();
    let bench_dir = PathBuf::from(BENCH_TSV).parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&bench_dir).unwrap();

    let out = PathBuf::from(OUT_DIR);
    let ours_jxl = out.join("ours.jxl");
    let libjxl_jxl = out.join("libjxl.jxl");

    // Encode if not already present (we ran this before, but be resilient).
    if !ours_jxl.exists() {
        let s = Command::new(CJXL_RS)
            .args([
                "--effort",
                "9",
                "--distance",
                "4",
                SRC,
                ours_jxl.to_str().unwrap(),
            ])
            .status()
            .expect("cjxl-rs");
        assert!(s.success(), "cjxl-rs failed");
    }
    if !libjxl_jxl.exists() {
        let s = Command::new(CJXL)
            .args([
                "--effort",
                "9",
                "--distance",
                "4",
                SRC,
                libjxl_jxl.to_str().unwrap(),
            ])
            .status()
            .expect("cjxl");
        assert!(s.success(), "cjxl failed");
    }

    // Load linear-sRGB f32: ref + ours_recon + libjxl_recon.
    println!("[SA-F] loading reference (PNG -> linear sRGB f32)...");
    let (ref_lin, w, h) = load_png_linear(Path::new(SRC));
    println!("[SA-F] ref: {} x {} = {} pixels", w, h, ref_lin.len() / 3);

    println!("[SA-F] decoding ours.jxl (jxl-oxide linear)...");
    let (ours_recon, ow, oh) = decode_jxl_linear(&ours_jxl);
    assert_eq!((ow, oh), (w, h), "ours_recon size mismatch");

    println!("[SA-F] decoding libjxl.jxl (jxl-oxide linear)...");
    let (libjxl_recon, lw, lh) = decode_jxl_linear(&libjxl_jxl);
    assert_eq!((lw, lh), (w, h), "libjxl_recon size mismatch");

    // Write PFMs.
    let ref_pfm = out.join("ref.pfm");
    let ours_pfm = out.join("ours_recon.pfm");
    let libjxl_pfm = out.join("libjxl_recon.pfm");
    println!("[SA-F] writing PFMs...");
    write_pfm(&ref_pfm, &ref_lin, w, h);
    write_pfm(&ours_pfm, &ours_recon, w, h);
    write_pfm(&libjxl_pfm, &libjxl_recon, w, h);

    // 4-pair matrix:
    //   pair AA: (ref, ours_recon)    -- our crate scores
    //   pair AB: (ref, ours_recon)    -- libjxl scores  [SA-B's iter-0 ours number = 4.31]
    //   pair BA: (ref, libjxl_recon)  -- our crate scores
    //   pair BB: (ref, libjxl_recon)  -- libjxl scores  [SA-B's iter-0 cjxl number = 10.24]
    // (AA == AB) within ~1% --> metric at parity --> SA-B's claim REFUTED
    // (AA != AB) --> metric divergent
    println!("[SA-F] running our butteraugli (ref, ours_recon)...");
    let (ours_score_on_ours, ours_pnorm_on_ours) =
        run_ours_butteraugli(&ref_lin, &ours_recon, w, h);
    println!(
        "[SA-F]   ours/ours_recon: score={:.4} pnorm3={:.4}",
        ours_score_on_ours, ours_pnorm_on_ours
    );

    println!("[SA-F] running libjxl butteraugli_main (ref, ours_recon)...");
    let (libjxl_score_on_ours, libjxl_pnorm_on_ours) =
        run_libjxl_butteraugli(&ref_pfm, &ours_pfm).expect("libjxl on ours_recon");
    println!(
        "[SA-F]   libjxl/ours_recon: score={:.4} pnorm3={:.4}",
        libjxl_score_on_ours, libjxl_pnorm_on_ours
    );

    println!("[SA-F] running our butteraugli (ref, libjxl_recon)...");
    let (ours_score_on_libjxl, ours_pnorm_on_libjxl) =
        run_ours_butteraugli(&ref_lin, &libjxl_recon, w, h);
    println!(
        "[SA-F]   ours/libjxl_recon: score={:.4} pnorm3={:.4}",
        ours_score_on_libjxl, ours_pnorm_on_libjxl
    );

    println!("[SA-F] running libjxl butteraugli_main (ref, libjxl_recon)...");
    let (libjxl_score_on_libjxl, libjxl_pnorm_on_libjxl) =
        run_libjxl_butteraugli(&ref_pfm, &libjxl_pfm).expect("libjxl on libjxl_recon");
    println!(
        "[SA-F]   libjxl/libjxl_recon: score={:.4} pnorm3={:.4}",
        libjxl_score_on_libjxl, libjxl_pnorm_on_libjxl
    );

    // Also: sanity check (ref vs ref) should produce 0.0.
    println!("[SA-F] sanity (ref, ref) both sides...");
    let (ours_self, _) = run_ours_butteraugli(&ref_lin, &ref_lin, w, h);
    let (libjxl_self, _) = run_libjxl_butteraugli(&ref_pfm, &ref_pfm).expect("libjxl ref vs ref");
    println!(
        "[SA-F]   ours(ref,ref)={:.4} libjxl(ref,ref)={:.4}",
        ours_self, libjxl_self
    );

    // Write TSV.
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(BENCH_TSV)
        .unwrap();
    writeln!(
        f,
        "pair\trecon\tscorer\tmax_score\tpnorm3\trel_diff_vs_same_recon"
    )
    .unwrap();
    // For the rel_diff column: compare AA vs AB and BA vs BB.
    let rel_aa_ab =
        (ours_score_on_ours - libjxl_score_on_ours).abs() / libjxl_score_on_ours.max(1e-9);
    let rel_ba_bb =
        (ours_score_on_libjxl - libjxl_score_on_libjxl).abs() / libjxl_score_on_libjxl.max(1e-9);
    writeln!(
        f,
        "AA\tours_recon\tours\t{:.6}\t{:.6}\t{:.4}",
        ours_score_on_ours, ours_pnorm_on_ours, rel_aa_ab
    )
    .unwrap();
    writeln!(
        f,
        "AB\tours_recon\tlibjxl\t{:.6}\t{:.6}\t{:.4}",
        libjxl_score_on_ours, libjxl_pnorm_on_ours, rel_aa_ab
    )
    .unwrap();
    writeln!(
        f,
        "BA\tlibjxl_recon\tours\t{:.6}\t{:.6}\t{:.4}",
        ours_score_on_libjxl, ours_pnorm_on_libjxl, rel_ba_bb
    )
    .unwrap();
    writeln!(
        f,
        "BB\tlibjxl_recon\tlibjxl\t{:.6}\t{:.6}\t{:.4}",
        libjxl_score_on_libjxl, libjxl_pnorm_on_libjxl, rel_ba_bb
    )
    .unwrap();
    writeln!(
        f,
        "SANITY_ours_self\tref_as_dist\tours\t{:.6}\t-\t-",
        ours_self
    )
    .unwrap();
    writeln!(
        f,
        "SANITY_libjxl_self\tref_as_dist\tlibjxl\t{:.6}\t-\t-",
        libjxl_self
    )
    .unwrap();
    println!("[SA-F] TSV: {}", BENCH_TSV);

    let mut m = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(BENCH_META)
        .unwrap();
    writeln!(m, "# SA-F: butteraugli metric cross-test").unwrap();
    writeln!(m, "# Date: 2026-05-25").unwrap();
    writeln!(m, "# Source: {}", SRC).unwrap();
    writeln!(
        m,
        "# Encoders: cjxl-rs (ours) AND cjxl (libjxl) at -e 9 -d 4"
    )
    .unwrap();
    writeln!(
        m,
        "# Decoder for both .jxl files: jxl-oxide 0.12.5 with srgb_linear request"
    )
    .unwrap();
    writeln!(
        m,
        "# Our scorer: butteraugli crate `butteraugli_linear` with `ButteraugliParams::default()`"
    )
    .unwrap();
    writeln!(
        m,
        "# Libjxl scorer: butteraugli_main with `--colorspace RGB_D65_SRG_Rel_Lin --pnorm 3.0`"
    )
    .unwrap();
    writeln!(m).unwrap();
    writeln!(m, "## 4-pair matrix").unwrap();
    writeln!(
        m,
        "##   AA = our crate scoring (ref, ours_recon)    score={:.4}",
        ours_score_on_ours
    )
    .unwrap();
    writeln!(
        m,
        "##   AB = libjxl scoring     (ref, ours_recon)    score={:.4}   rel_diff vs AA = {:.2}%",
        libjxl_score_on_ours,
        rel_aa_ab * 100.0
    )
    .unwrap();
    writeln!(
        m,
        "##   BA = our crate scoring (ref, libjxl_recon)  score={:.4}",
        ours_score_on_libjxl
    )
    .unwrap();
    writeln!(
        m,
        "##   BB = libjxl scoring     (ref, libjxl_recon)  score={:.4}   rel_diff vs BA = {:.2}%",
        libjxl_score_on_libjxl,
        rel_ba_bb * 100.0
    )
    .unwrap();
    writeln!(m).unwrap();
    writeln!(m, "## Verdict logic").unwrap();
    writeln!(
        m,
        "##   If (AA ≈ AB) AND (BA ≈ BB) within 1% -> metric at parity, SA-B claim REFUTED"
    )
    .unwrap();
    writeln!(
        m,
        "##   If (AA != AB) significantly -> metric diverges, isolate to specific band/norm"
    )
    .unwrap();
    println!("[SA-F] meta: {}", BENCH_META);
}
