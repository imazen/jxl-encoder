//! W44-104 proxy probe: dump simple proxies for 11 gb82-sc screenshots
//! to identify a discriminator separating terminal-class (admit DCT64) from
//! codec_wiki-class (keep W44-65/W44-68 suppression).
//!
//! Inlines the zenanalyze-equivalent helpers because [`ZenanalyzeProxies`]
//! is `pub(crate)`. Source: `vardct/encoder.rs:678` `compute_srgb_u8`.

use std::path::PathBuf;

const SCREENSHOTS: &[&str] = &[
    "codec_wiki.png",
    "gmessages.png",
    "graph.png",
    "gui.png",
    "imac_dark.png",
    "imac_g3.png",
    "imac_g3_strip.png",
    "imessage.png",
    "terminal.png",
    "windows.png",
    "windows95.png",
];

/// Hasler-Süsstrunk M3 colourfulness (sRGB u8 input).
fn m3_colourfulness(pixels: &[u8], width: usize, height: usize, bpp: usize) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let n = (width * height) as f64;
    let mut sum_rg = 0.0f64;
    let mut sum_rg2 = 0.0f64;
    let mut sum_yb = 0.0f64;
    let mut sum_yb2 = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * bpp;
            let r = pixels[i] as f64;
            let g = pixels[i + 1] as f64;
            let b = pixels[i + 2] as f64;
            let rg = (r - g).abs();
            let yb = (0.5 * (r + g) - b).abs();
            sum_rg += rg;
            sum_rg2 += rg * rg;
            sum_yb += yb;
            sum_yb2 += yb * yb;
        }
    }
    let mean_rg = sum_rg / n;
    let mean_yb = sum_yb / n;
    let var_rg = (sum_rg2 / n - mean_rg * mean_rg).max(0.0);
    let var_yb = (sum_yb2 / n - mean_yb * mean_yb).max(0.0);
    let sigma = (var_rg + var_yb).sqrt();
    let mu = (mean_rg * mean_rg + mean_yb * mean_yb).sqrt();
    (sigma + 0.3 * mu) as f32
}

/// Fraction of 8×8 blocks where max(R)-min(R), max(G)-min(G), max(B)-min(B)
/// are all < 4. Matches zenanalyze `flat_color_blocks` Tier 1 feature.
fn flat_color_block_ratio(pixels: &[u8], width: usize, height: usize, bpp: usize) -> f32 {
    if width < 8 || height < 8 {
        return 0.0;
    }
    let bx = width / 8;
    let by = height / 8;
    if bx == 0 || by == 0 {
        return 0.0;
    }
    let mut flat = 0u32;
    let mut total = 0u32;
    for cy in 0..by {
        for cx in 0..bx {
            let mut rmin = 255u8;
            let mut rmax = 0u8;
            let mut gmin = 255u8;
            let mut gmax = 0u8;
            let mut bmin = 255u8;
            let mut bmax = 0u8;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = cy * 8 + dy;
                    let x = cx * 8 + dx;
                    let i = (y * width + x) * bpp;
                    let r = pixels[i];
                    let g = pixels[i + 1];
                    let b = pixels[i + 2];
                    rmin = rmin.min(r);
                    rmax = rmax.max(r);
                    gmin = gmin.min(g);
                    gmax = gmax.max(g);
                    bmin = bmin.min(b);
                    bmax = bmax.max(b);
                }
            }
            if rmax - rmin < 4 && gmax - gmin < 4 && bmax - bmin < 4 {
                flat += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        flat as f32 / total as f32
    }
}

/// Sobel-luma edge density. Matches `ZenanalyzeProxies::edge_density`.
fn edge_density(pixels: &[u8], width: usize, height: usize, bpp: usize) -> f32 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let luma = |x: usize, y: usize| -> f32 {
        let i = (y * width + x) * bpp;
        let r = pixels[i] as f32;
        let g = pixels[i + 1] as f32;
        let b = pixels[i + 2] as f32;
        0.299 * r + 0.587 * g + 0.114 * b
    };
    let mut edges = 0u32;
    let mut total = 0u32;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gx = luma(x + 1, y - 1) + 2.0 * luma(x + 1, y) + luma(x + 1, y + 1)
                - luma(x - 1, y - 1)
                - 2.0 * luma(x - 1, y)
                - luma(x - 1, y + 1);
            let gy = luma(x - 1, y + 1) + 2.0 * luma(x, y + 1) + luma(x + 1, y + 1)
                - luma(x - 1, y - 1)
                - 2.0 * luma(x, y - 1)
                - luma(x + 1, y - 1);
            let mag = (gx * gx + gy * gy).sqrt();
            if mag > 32.0 {
                edges += 1;
            }
            total += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        edges as f32 / total as f32
    }
}

fn main() {
    let base = PathBuf::from("/home/lilith/work/codec-corpus/gb82-sc");
    println!(
        "{:<24} {:>12} {:>12} {:>12} {:>10}",
        "name", "m3_colour", "fcbr", "edge_density", "WxH"
    );
    for name in SCREENSHOTS {
        let path = base.join(name);
        if !path.exists() {
            eprintln!("MISS {}", path.display());
            continue;
        }
        let img = image::open(&path).unwrap();
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let rgb_u8: Vec<u8> = rgb.as_raw().clone();
        let m3 = m3_colourfulness(&rgb_u8, w, h, 3);
        let fcbr = flat_color_block_ratio(&rgb_u8, w, h, 3);
        let ed = edge_density(&rgb_u8, w, h, 3);
        println!(
            "{:<24} {:>12.3} {:>12.4} {:>12.4} {:>4}x{:>4}",
            name, m3, fcbr, ed, w, h
        );
    }
}
