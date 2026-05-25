//! W44-AUDIT-8 Phase 4: per-DC-block dump on clic_22ea12 e7 d=4.
//!
//! Phase 3 (commit `1bddf371`) confirmed AC strategy + qac are at parity
//! on the worst CLIC cell (clic_22ea12 e7 d=4). The top-right region (-9.68
//! SSIM2 deficit) has 0/30 strategy mismatches; both encoders pick DCT64X64
//! with qac=8, nzeros=0 on flat sky/water. Decoded pixels come entirely
//! from DC + reconstruction.
//!
//! Phase 4 captures per-DC-block dumps on BOTH sides and joins on
//! (bx, by, channel). Diagnostic questions:
//!   - Are raw float DC values identical? (forward pipeline)
//!   - Are quantized DC values identical given identical raw? (quant step)
//!   - Are post-CfL X/B differences ours vs libjxl?
//!
//! Outputs:
//!   benchmarks/w44_audit_8_phase4_dc_dump_2026-05-24.{tsv,meta}
//!   /tmp/w44_audit_8_phase4_dc_dumps/<label>_e7_d4_{ours,cjxl}/dc_per_block_*.tsv
//!
//! Run:
//!   cargo run -p jxl-encoder --release \
//!     --features 'parallel butteraugli-loop ssim2-loop' \
//!     --example w44_audit_8_phase4_dc_dump
use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const IMG_CLIC_22EA12: &str =
    "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png";
const DISTANCE: f32 = 4.0;
const EFFORT: u8 = 7;

fn cjxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CJXL") {
        return PathBuf::from(p);
    }
    let dbg = PathBuf::from(
        "/home/lilith/work/jxl-efforts/libjxl--w44-76-per-block-debug-dump/build/tools/cjxl",
    );
    if dbg.exists() {
        return dbg;
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/cjxl")
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn encode_ours(rgb: &[u8], w: u32, h: u32) -> Option<(Vec<u8>, f64)> {
    let cfg = LossyConfig::new(DISTANCE)
        .with_effort(EFFORT)
        .with_threads(1);
    let start = std::time::Instant::now();
    let bytes = cfg.encode(rgb, w, h, PixelLayout::Rgb8).ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Some((bytes, ms))
}

fn encode_cjxl(src_png: &Path, dump_dir: &Path) -> Option<(Vec<u8>, f64)> {
    let tmp = std::env::temp_dir().join(format!(
        "w44_p4_cjxl_{}_e{}_d{}.jxl",
        src_png.file_stem().unwrap().to_string_lossy(),
        EFFORT,
        DISTANCE as i32
    ));
    let bin = cjxl_bin();
    let mut cmd = Command::new(&bin);
    cmd.arg(src_png)
        .arg(&tmp)
        .args(["-e", &EFFORT.to_string()])
        .args(["-d", &DISTANCE.to_string()])
        .args(["--num_threads", "1"])
        .arg("--quiet")
        .env("JXL_W44_AUDIT_8_P4_DUMP", dump_dir);
    let start = std::time::Instant::now();
    let status = cmd.status().ok()?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&tmp).ok()?;
    let _ = std::fs::remove_file(&tmp);
    Some((bytes, ms))
}

fn main() {
    let out_dir = PathBuf::from("/home/lilith/work/zen/jxl-encoder/benchmarks");
    let tsv_path = out_dir.join("w44_audit_8_phase4_dc_dump_2026-05-24.tsv");
    let dump_root = PathBuf::from("/tmp/w44_audit_8_phase4_dc_dumps");
    std::fs::create_dir_all(&dump_root).expect("mkdir dumps");

    let label = "clic_22ea12_WORST";
    let src_path = Path::new(IMG_CLIC_22EA12);
    let (rgb, w, h) = load_png(src_path).expect("load");
    eprintln!("=== {} ({}x{}) ===", label, w, h);

    // OURS with W44-AUDIT-8 P4 dump.
    let ours_dir = dump_root.join(format!("{}_e{}_d{}_ours", label, EFFORT, DISTANCE as i32));
    std::fs::create_dir_all(&ours_dir).unwrap();
    // SAFETY: single-threaded harness, child encode runs sequentially.
    unsafe {
        std::env::set_var("JXL_W44_AUDIT_8_P4_DUMP", &ours_dir);
    }
    let (ours_bytes, ours_ms) = encode_ours(&rgb, w, h).expect("encode ours");
    unsafe {
        std::env::remove_var("JXL_W44_AUDIT_8_P4_DUMP");
    }
    eprintln!("  ours: bytes={} ({:.0}ms)", ours_bytes.len(), ours_ms);
    eprintln!("  ours dump: {}", ours_dir.display());

    // CJXL with W44-AUDIT-8 P4 dump.
    let cjxl_dir = dump_root.join(format!("{}_e{}_d{}_cjxl", label, EFFORT, DISTANCE as i32));
    std::fs::create_dir_all(&cjxl_dir).unwrap();
    let (cjxl_bytes, cjxl_ms) = encode_cjxl(src_path, &cjxl_dir).expect("encode cjxl");
    eprintln!("  cjxl: bytes={} ({:.0}ms)", cjxl_bytes.len(), cjxl_ms);
    eprintln!("  cjxl dump: {}", cjxl_dir.display());

    // Index TSV.
    let mut tsv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tsv_path)
        .expect("open tsv");
    writeln!(
        tsv,
        "image\tencoder\teffort\tdistance\tbytes\tencode_ms\tdump_dir"
    )
    .unwrap();
    writeln!(
        tsv,
        "{}\tours\t{}\t{}\t{}\t{:.2}\t{}",
        label,
        EFFORT,
        DISTANCE,
        ours_bytes.len(),
        ours_ms,
        ours_dir.display()
    )
    .unwrap();
    writeln!(
        tsv,
        "{}\tcjxl\t{}\t{}\t{}\t{:.2}\t{}",
        label,
        EFFORT,
        DISTANCE,
        cjxl_bytes.len(),
        cjxl_ms,
        cjxl_dir.display()
    )
    .unwrap();

    eprintln!();
    eprintln!("Wrote {}", tsv_path.display());
    eprintln!();
    eprintln!("Run analyzer:");
    eprintln!(
        "  python3 benchmarks/w44_audit_8_phase4_analyze.py {} {}",
        ours_dir.display(),
        cjxl_dir.display(),
    );
}
