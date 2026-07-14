//! W44-AUDIT-9 / SA-G Fix C: multi-decoder roundtrip via jxl-rs +
//! jxl-oxide + djxl on 3 cells (1 smoke + 2 random). All three decoders
//! must decode the output of `EncoderStrategy::Libjxl` (where Fix C is
//! ON by default) on these cells without error.

use jxl_encoder::api::{EncoderStrategy, LossyConfig, PixelLayout};
use std::path::{Path, PathBuf};
use std::process::Command;

const CELLS: &[(&str, &str, u8, f32)] = &[
    (
        "clic_22ea12_e9_d4",
        "/home/lilith/work/codec-corpus/clic2025-1024/22ea12c903e41583b7c469cb86040157.png",
        9,
        4.0,
    ),
    (
        "cid22_1418519_e7_d3",
        "/home/lilith/work/codec-corpus/CID22/CID22-512/validation/1418519.png",
        7,
        3.0,
    ),
    (
        "gb82_terminal_e7_d2",
        "/home/lilith/work/codec-corpus/gb82-sc/terminal.png",
        7,
        2.0,
    ),
];

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?;
    let rgb = img.to_rgb8();
    Some((rgb.as_raw().clone(), rgb.width(), rgb.height()))
}

fn djxl_bin() -> PathBuf {
    if let Ok(p) = std::env::var("DJXL") {
        return PathBuf::from(p);
    }
    PathBuf::from("/home/lilith/work/jxl-efforts/libjxl/build/tools/djxl")
}

fn check_djxl(jxl_bytes: &[u8], name: &str) -> bool {
    let in_path = std::env::temp_dir().join(format!("sa_g_fix_c_djxl_in_{name}.jxl"));
    let out_path = std::env::temp_dir().join(format!("sa_g_fix_c_djxl_out_{name}.png"));
    std::fs::write(&in_path, jxl_bytes).expect("write tmp jxl");
    let status = Command::new(djxl_bin())
        .arg(&in_path)
        .arg(&out_path)
        .arg("--quiet")
        .status();
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    match status {
        Ok(s) if s.success() => true,
        Ok(s) => {
            eprintln!("  ✗ djxl exit={s:?} on {name}");
            false
        }
        Err(e) => {
            eprintln!("  ✗ djxl spawn err on {name}: {e}");
            false
        }
    }
}

fn check_jxl_oxide(jxl_bytes: &[u8], name: &str) -> bool {
    match jxl_oxide::JxlImage::builder().read(std::io::Cursor::new(jxl_bytes)) {
        Ok(img) => match img.render_frame(0) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("  ✗ jxl-oxide render failed on {name}: {e:?}");
                false
            }
        },
        Err(e) => {
            eprintln!("  ✗ jxl-oxide read failed on {name}: {e:?}");
            false
        }
    }
}

fn check_jxl_rs(jxl_bytes: &[u8], name: &str) -> bool {
    let in_path = std::env::temp_dir().join(format!("sa_g_fix_c_jxlrs_in_{name}.jxl"));
    let out_path = std::env::temp_dir().join(format!("sa_g_fix_c_jxlrs_out_{name}.png"));
    std::fs::write(&in_path, jxl_bytes).expect("write tmp jxl");
    let jxl_rs_bin = std::env::var("JXL_RS_BIN").unwrap_or_else(|_| {
        "/home/lilith/work/third-party/jxl-rs/target/release/jxl_cli".to_string()
    });
    let status = Command::new(&jxl_rs_bin)
        .arg(&in_path)
        .arg(&out_path)
        .status();
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    match status {
        Ok(s) if s.success() => true,
        Ok(s) => {
            eprintln!("  ✗ jxl-rs exit={s:?} on {name}");
            false
        }
        Err(e) => {
            eprintln!("  ✗ jxl-rs spawn err on {name}: {e}");
            false
        }
    }
}

fn main() {
    let mut all_pass = true;
    let mut total = 0;
    let mut passed = 0;

    for (name, path, effort, distance) in CELLS {
        eprintln!("\n→ {name} ({effort}, d={distance})");
        let path = Path::new(path);
        let (rgb, w, h) = match load_png(path) {
            Some(v) => v,
            None => {
                eprintln!("  ✗ load failed: {name}");
                all_pass = false;
                continue;
            }
        };
        let bytes = LossyConfig::new(*distance)
            .with_effort(*effort)
            .with_threads(1)
            .with_strategy(EncoderStrategy::Libjxl)
            .encode(&rgb, w, h, PixelLayout::Rgb8)
            .expect("encode");
        eprintln!("  encoded: {} bytes", bytes.len());

        for (decoder_name, check_fn) in [
            ("jxl-oxide", check_jxl_oxide as fn(&[u8], &str) -> bool),
            ("djxl", check_djxl),
            ("jxl-rs", check_jxl_rs),
        ] {
            total += 1;
            if check_fn(&bytes, name) {
                eprintln!("  ✓ {decoder_name} decoded {name}");
                passed += 1;
            } else {
                all_pass = false;
            }
        }
    }

    eprintln!("\n=== {passed}/{total} decoder checks pass ===");
    if !all_pass {
        std::process::exit(1);
    }
}
