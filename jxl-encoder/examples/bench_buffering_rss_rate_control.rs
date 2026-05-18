//! Peak-RSS bench (rate-control path): encode a 3072×3072 photo via
//! [`VarDctEncoder::encode_with_rate_control_config`] under each
//! [`Buffering`] variant.
//!
//! Why a second bench: the default `LossyConfig::encode()` path goes
//! through `encoder.rs::encode_inner` which does inline precompute
//! (XYB / quant_field / mask1x1 / gaborish / CfL / AC strategy) and
//! never touches `EncoderPrecomputed::compute_with_budget`. The
//! streaming refactor #11 chunk-6 dispatch lives inside
//! `compute_with_budget_and_buffering`, so it only fires when
//! `encode_with_rate_control_config` is invoked. This bench exercises
//! that path so the per-region precompute (chunk 5) is reachable from
//! the user-facing API.
//!
//! Requires `--features rate-control`.
//!
//! Run with:
//!   cargo run --release -p jxl-encoder \
//!     --features rate-control \
//!     --example bench_buffering_rss_rate_control [variant] [w] [h]

#[cfg(feature = "rate-control")]
fn main() {
    use jxl_encoder::Buffering;
    use jxl_encoder::vardct::{RateControlConfig, VarDctEncoder};

    let variant_str = std::env::args().nth(1).unwrap_or_else(|| "full".into());
    let variant = match variant_str.as_str() {
        "full" | "0" => Buffering::FullBuffered,
        "threshold" | "1" => Buffering::Threshold2048,
        "buffered" | "2" => Buffering::BufferedOutput,
        "streaming" | "3" => Buffering::FullStreaming,
        "auto" | "-1" => Buffering::Auto,
        _ => panic!("unknown variant: {variant_str}"),
    };
    let w: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072);
    let h: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3072);

    // Same fbm-ish content as bench_buffering_rss to keep memory
    // workloads comparable.
    let mut pixels_u8 = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let v1 = ((x * 17 + y * 11) & 0xff) as u8;
            let v2 = ((x ^ y) & 0xff) as u8;
            let v3 = (((x as u64).wrapping_mul(31) ^ (y as u64).wrapping_mul(7)) & 0xff) as u8;
            pixels_u8[i] = v1;
            pixels_u8[i + 1] = v2;
            pixels_u8[i + 2] = v3;
        }
    }
    // sRGB → linear.
    let linear_rgb: Vec<f32> = pixels_u8
        .iter()
        .map(|&b| {
            let c = b as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();

    let mut enc = VarDctEncoder::new(2.0);
    enc.buffering = variant;
    let cfg = RateControlConfig::default();
    let t0 = std::time::Instant::now();
    let (bytes, iters) = enc
        .encode_with_rate_control_config(w, h, &linear_rgb, &cfg)
        .expect("rate-control encode failed");
    let dt = t0.elapsed();
    eprintln!(
        "variant={variant:?} bytes={} iters={iters} took={:.2}s",
        bytes.len(),
        dt.as_secs_f64()
    );
}

#[cfg(not(feature = "rate-control"))]
fn main() {
    eprintln!(
        "bench_buffering_rss_rate_control requires --features rate-control; \
         skipping."
    );
}
