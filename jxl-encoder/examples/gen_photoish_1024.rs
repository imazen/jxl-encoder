//! W45-RECON part 9: regenerate the `photoish_rgb_1024x1024` hash-lock
//! fixture as a PPM so both our encoder and cjxl can encode the identical
//! input for BCM/used_orders parity diffing.
//!
//! `cargo run --example gen_photoish_1024 -- /tmp/photoish_1024.ppm`

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/photoish_1024.ppm".into());
    let (w, h) = (1024usize, 1024usize);
    let hash = |x: i64, y: i64, c: i64| -> f32 {
        let mut v = (x.wrapping_mul(0x27d4_eb2d)
            ^ y.wrapping_mul(0x1656_67b1)
            ^ c.wrapping_mul(0x9e37_79b9)) as u64;
        v ^= v >> 33;
        v = v.wrapping_mul(0xff51_afd7_ed55_8ccd);
        v ^= v >> 29;
        ((v >> 40) as u32 as f32) / 16_777_216.0
    };
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                let mut acc = 0.0f32;
                let mut amp = 0.5f32;
                for oct in 0..6u32 {
                    let step = 1usize << (6 - oct);
                    let (gx, gy) = (x / step, y / step);
                    let (fx, fy) = (
                        (x % step) as f32 / step as f32,
                        (y % step) as f32 / step as f32,
                    );
                    let c0 = c as i64 * 7 + oct as i64 * 131;
                    let v00 = hash(gx as i64, gy as i64, c0);
                    let v10 = hash(gx as i64 + 1, gy as i64, c0);
                    let v01 = hash(gx as i64, gy as i64 + 1, c0);
                    let v11 = hash(gx as i64 + 1, gy as i64 + 1, c0);
                    let top = v00 + (v10 - v00) * fx;
                    let bot = v01 + (v11 - v01) * fx;
                    acc += amp * (top + (bot - top) * fy);
                    amp *= 0.55;
                }
                if x > 700 && y < 300 {
                    acc = acc * 0.35 + 0.6;
                }
                if (x as i64 - y as i64).abs() < 6 {
                    acc = 1.0 - acc;
                }
                acc += (hash(x as i64, y as i64, c as i64 * 977 + 31) - 0.5) * 0.08;
                out[(y * w + x) * 3 + c] = (acc.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    let mut ppm = format!("P6\n{} {}\n255\n", w, h).into_bytes();
    ppm.extend_from_slice(&out);
    std::fs::write(&out_path, &ppm).unwrap();
    eprintln!("wrote {}", out_path);
}
