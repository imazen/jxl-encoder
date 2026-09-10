//! Where does lossy e7's wall go? It is 2.54x cjxl v0.12 on the committed
//! baseline, the worst ratio on our ladder (e3 is 0.71x and e9 is 1.06x), so
//! the gap is specifically the e5-e7 band.
//!
//! Already ruled out: entropy coding. `__JXL_ENC_PHASE_TIMING=1` shows
//! `encode_two_pass` essentially identical at e5 and e7 (build_codes 3.1-3.7 ms,
//! pass2_write 1.5-1.8 ms at both), so the time is in pre-entropy analysis.
//!
//! Six things turn on at effort 7: patches, tree_learning, try_dct64,
//! chromacity_adjustment, cfl_two_pass, and dot detection. Five of the six are
//! reachable — `patches` and `dot_detection` from `LossyConfig` directly, and
//! `try_dct64` / `try_dct32` / `cfl_two_pass` / `chromacity_adjustment` through
//! the `__expert` `LossyInternalParams` escape hatch. `tree_learning` has no
//! override field, so it is reported as unreachable rather than guessed at.
//!
//! Run: `cargo run --release --features __expert --example e7_cost_attribution
//! -- <png> [size] [distance]`
use jxl_encoder::LossyInternalParams;
use jxl_encoder::api::{LossyConfig, PixelLayout};
use std::time::Instant;

type Arm = (&'static str, Box<dyn Fn(LossyConfig) -> LossyConfig>);

fn ip(f: impl FnOnce(&mut LossyInternalParams)) -> LossyInternalParams {
    let mut p = LossyInternalParams::default();
    f(&mut p);
    p
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe <png> [size] [distance]");
    let want: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let d: f32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let reps: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let rgb = image::open(&path).expect("open").to_rgb8();
    // Clamp the crop to the source: a 1440x900 screenshot cannot give a 1024
    // square without this, and `crop_imm` panics rather than shrinking.
    let n = want.min(rgb.width()).min(rgb.height());
    let (x0, y0) = ((rgb.width() - n) / 2, (rgb.height() - n) / 2);
    let src: Vec<u8> = image::imageops::crop_imm(&rgb, x0, y0, n, n)
        .to_image()
        .as_raw()
        .clone();

    let arms: Vec<Arm> = vec![
        (
            "e5 (reference)",
            Box::new(|c: LossyConfig| c.with_effort(5)),
        ),
        (
            // Issue #43 chunk 2a lifts `patches` to e5/e6 on Screenshot-class
            // content, so e5 is NOT patch-free on graphics — this arm prices
            // that lift.
            "e5, patches OFF",
            Box::new(|c: LossyConfig| c.with_effort(5).with_patches(false)),
        ),
        ("e7 default", Box::new(|c: LossyConfig| c.with_effort(7))),
        (
            "e7, patches OFF",
            Box::new(|c: LossyConfig| c.with_effort(7).with_patches(false)),
        ),
        (
            "e7, dots OFF",
            Box::new(|c: LossyConfig| c.with_effort(7).with_dot_detection(false)),
        ),
        (
            "e7, patches+dots OFF",
            Box::new(|c: LossyConfig| {
                c.with_effort(7)
                    .with_patches(false)
                    .with_dot_detection(false)
            }),
        ),
        (
            "e7, try_dct64 OFF",
            Box::new(|c: LossyConfig| {
                c.with_effort(7)
                    .with_internal_params(ip(|p| p.try_dct64 = Some(false)))
            }),
        ),
        (
            "e7, dct32+64 OFF",
            Box::new(|c: LossyConfig| {
                c.with_effort(7).with_internal_params(ip(|p| {
                    p.try_dct32 = Some(false);
                    p.try_dct64 = Some(false);
                }))
            }),
        ),
        (
            "e7, cfl_two_pass OFF",
            Box::new(|c: LossyConfig| {
                c.with_effort(7)
                    .with_internal_params(ip(|p| p.cfl_two_pass = Some(false)))
            }),
        ),
        (
            "e7, chromacity_adj OFF",
            Box::new(|c: LossyConfig| {
                c.with_effort(7)
                    .with_internal_params(ip(|p| p.chromacity_adjustment = Some(false)))
            }),
        ),
        (
            "e7, all reachable OFF",
            Box::new(|c: LossyConfig| {
                c.with_effort(7)
                    .with_patches(false)
                    .with_dot_detection(false)
                    .with_internal_params(ip(|p| {
                        p.try_dct32 = Some(false);
                        p.try_dct64 = Some(false);
                        p.cfl_two_pass = Some(false);
                        p.chromacity_adjustment = Some(false);
                    }))
            }),
        ),
    ];

    // `E7_ARM=<substring>` narrows to the matching arms, so a single arm can be
    // run under `__JXL_ENC_PHASE_TIMING=1` without ten encodes of noise around
    // it.
    let filter = std::env::var("E7_ARM").unwrap_or_default();
    let arms: Vec<Arm> = arms
        .into_iter()
        .filter(|(l, _)| filter.is_empty() || l.contains(filter.as_str()))
        .collect();
    assert!(!arms.is_empty(), "E7_ARM={filter:?} matched no arm");

    println!(
        "{} @ {n}x{n} (requested {want}), d={d}, min of {reps}",
        path.rsplit('/').next().unwrap()
    );
    // INTERLEAVE the arms inside each repeat. Block-ordered repeats (all reps
    // of arm A, then all of arm B) inverted the sign of a 8 % effect during
    // T3 on short cells, because the machine drifts over a run.
    // ...and ROTATE the starting arm per repeat. With a fixed order the first
    // arm pays the cold allocator/page-cache cost in EVERY repeat, which made
    // byte-IDENTICAL arms differ by 16 %.
    let mut best = vec![f64::INFINITY; arms.len()];
    let mut bytes = vec![0usize; arms.len()];
    // One untimed warm-up so no timed encode is the process's first.
    let _ = arms[0].1(LossyConfig::new(d))
        .encode_request(n, n, PixelLayout::Rgb8)
        .encode(&src)
        .expect("encode");
    for rep in 0..reps {
        for k in 0..arms.len() {
            let i = (k + rep) % arms.len();
            let f = &arms[i].1;
            let t = Instant::now();
            let enc = f(LossyConfig::new(d))
                .encode_request(n, n, PixelLayout::Rgb8)
                .encode(&src)
                .expect("encode");
            best[i] = best[i].min(t.elapsed().as_secs_f64() * 1000.0);
            bytes[i] = enc.len();
        }
    }
    let e5_ms = best[0];
    for (i, (label, _)) in arms.iter().enumerate() {
        println!(
            "  {label:<28} {:>9.1} ms  ({:>5.2}x e5)   {:>9} B",
            best[i],
            best[i] / e5_ms,
            bytes[i]
        );
    }
    println!("  (tree_learning has no LossyInternalParams override — unreachable here)");
}
