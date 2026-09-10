//! The last known IQA-monotonicity violation: at extreme distance our BYTES go
//! non-monotone, and so do both quality oracles.
//!
//! Measured on `2400_textures` at e5 (where the perceptual loop is off by
//! default — `butteraugli_iters` is 0 for efforts 0..=7), 512 crop:
//! bytes 2,932 at d=14, 2,313 at d=15, **2,604 at d=17** — coarser request,
//! more bytes. cjxl v0.12 is monotone and smaller through the same points, so
//! the instability is ours and lives in base encoding rather than in the loop.
//!
//! This isolates it by disabling, one at a time, the features that can ADD data
//! independently of the quantiser.
use jxl_encoder::api::{LossyConfig, PixelLayout};

/// One isolation arm: a label and the knob it disables.
type Arm = (&'static str, Box<dyn Fn(LossyConfig) -> LossyConfig>);

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe <png> [size] [effort]");
    let n: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let e: u8 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let rgb = image::open(&path).expect("open").to_rgb8();
    let (x0, y0) = (
        rgb.width().saturating_sub(n) / 2,
        rgb.height().saturating_sub(n) / 2,
    );
    let src: Vec<u8> = image::imageops::crop_imm(&rgb, x0, y0, n, n)
        .to_image()
        .as_raw()
        .clone();

    let arms: Vec<Arm> = vec![
        ("default", Box::new(|c: LossyConfig| c)),
        (
            "patches OFF",
            Box::new(|c: LossyConfig| c.with_patches(false)),
        ),
        (
            "dots OFF",
            Box::new(|c: LossyConfig| c.with_dot_detection(false)),
        ),
        (
            "gaborish OFF",
            Box::new(|c: LossyConfig| c.with_gaborish(false)),
        ),
        ("epf 0", Box::new(|c: LossyConfig| c.with_epf_level(0))),
    ];

    println!("{} @ {n}x{n} e{e}", path.rsplit('/').next().unwrap());
    print!("{:>6}", "d");
    for (label, _) in &arms {
        print!("  {label:>13}");
    }
    println!("   (bytes; watch for a RISE as d grows)");

    for &d in &[10.0f32, 12.0, 14.0, 15.0, 17.0, 20.0] {
        print!("{d:>6}");
        for (_, f) in &arms {
            let cfg = f(LossyConfig::new(d).with_effort(e));
            let bytes = cfg
                .encode_request(n, n, PixelLayout::Rgb8)
                .encode(&src)
                .map(|v| v.len())
                .unwrap_or(0);
            print!("  {bytes:>13}");
        }
        println!();
    }
}
