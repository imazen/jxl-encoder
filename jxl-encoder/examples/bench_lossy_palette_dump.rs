//! Dumps lossy-palette encode outputs to /tmp for byte-exact comparison
//! across the Vec<Vec<...>> -> flat-stride refactor.

use jxl_encoder::api::{LosslessConfig, PixelLayout};

fn make_palette_image(seed: u32, w: u32, h: u32, num_colors: usize) -> Vec<u8> {
    let palette: Vec<[u8; 3]> = (0..num_colors)
        .map(|i| {
            let s = seed
                .wrapping_mul(2654435761)
                .wrapping_add(i as u32 * 0x9E3779B1);
            [s as u8, (s >> 8) as u8, (s >> 16) as u8]
        })
        .collect();
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let bx = x / 8;
            let by = y / 8;
            let idx = (bx
                .wrapping_mul(31)
                .wrapping_add(by.wrapping_mul(17))
                .wrapping_add(seed)) as usize
                % palette.len();
            let c = palette[idx];
            let n = ((x.wrapping_mul(7) ^ y.wrapping_mul(13)) & 7) as i16 - 3;
            for &ch in c.iter() {
                out.push((ch as i16 + n).clamp(0, 255) as u8);
            }
        }
    }
    out
}

fn dump(name: &str, pixels: &[u8], w: u32, h: u32) {
    let cfg = LosslessConfig::new()
        .with_lossy_palette(true)
        .with_ans(true);
    let out = cfg.encode(pixels, w, h, PixelLayout::Rgb8).expect("encode");
    let path = format!("/tmp/lossy_palette_dump_{name}.jxl");
    std::fs::write(&path, &out).expect("write");
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::Hasher;
    h.write(&out);
    let hash = h.finish();
    println!(
        "{name:<28} bytes={:>6} hash=0x{hash:016x}  -> {path}",
        out.len()
    );
}

fn main() {
    let cases: Vec<(&str, u32, u32, usize, u32)> = vec![
        ("256x256_8c_s1", 256, 256, 8, 1),
        ("256x256_16c_s2", 256, 256, 16, 2),
        ("256x256_32c_s3", 256, 256, 32, 3),
        ("256x256_64c_s4", 256, 256, 64, 4),
        // Multi-group 300x300, exercises the per-group call path
        ("300x300_8c_s5", 300, 300, 8, 5),
    ];
    for (name, w, h, k, s) in cases {
        let img = make_palette_image(s, w, h, k);
        dump(name, &img, w, h);
    }
}
