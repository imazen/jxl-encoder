//! Per-cell score equivalence between fast-ssim2 0.7.1 (what jxl-encoder is
//! pinned to) and 0.8.2 (current). Both versions are linked simultaneously via
//! renamed deps, so every cell is scored by both on the SAME input bytes in the
//! SAME process — no cross-run drift.
//!
//! Prints one line per cell so the comparison is PER-CELL, never aggregated: an
//! aggregate mean hides a real shift (the dssim-core 3.5.1 precedent moved all
//! 216 scores while the mean moved in the seventh decimal).

use imgref::Img;

fn degrade(src: &[[f32; 3]], w: usize, h: usize, kind: usize, strength: usize) -> Vec<[f32; 3]> {
    let mut out = src.to_vec();
    match kind {
        // Block quantization — codec-shaped (blocking + flat regions).
        0 => {
            let q = 1.0 / (1.0 + strength as f32 * 6.0);
            for y in 0..h {
                for x in 0..w {
                    let p = &mut out[y * w + x];
                    for c in 0..3 {
                        p[c] = ((p[c] / q).round()) * q;
                    }
                }
            }
        }
        // Box blur — detail loss.
        1 => {
            let r = strength;
            let mut tmp = out.clone();
            for y in 0..h {
                for x in 0..w {
                    let mut acc = [0.0f32; 3];
                    let mut n = 0.0f32;
                    for dy in y.saturating_sub(r)..(y + r + 1).min(h) {
                        for dx in x.saturating_sub(r)..(x + r + 1).min(w) {
                            for c in 0..3 {
                                acc[c] += src[dy * w + dx][c];
                            }
                            n += 1.0;
                        }
                    }
                    tmp[y * w + x] = [acc[0] / n, acc[1] / n, acc[2] / n];
                }
            }
            out = tmp;
        }
        // Deterministic additive noise.
        _ => {
            let a = strength as f32 * 0.012;
            for (i, p) in out.iter_mut().enumerate() {
                let n = (((i as u32).wrapping_mul(2654435761) >> 8) & 0xff) as f32 / 255.0 - 0.5;
                for c in 0..3 {
                    p[c] = (p[c] + n * a).clamp(0.0, 1.0);
                }
            }
        }
    }
    out
}

fn load(path: &str, crop: Option<(usize, usize)>) -> (Vec<[f32; 3]>, usize, usize) {
    let img = image::open(path).unwrap_or_else(|e| panic!("{path}: {e}")).to_rgb8();
    let (fw, fh) = (img.width() as usize, img.height() as usize);
    let (w, h) = crop.map(|(cw, ch)| (cw.min(fw), ch.min(fh))).unwrap_or((fw, fh));
    let raw = img.as_raw();
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * fw + x) * 3;
            v.push([
                raw[i] as f32 / 255.0,
                raw[i + 1] as f32 / 255.0,
                raw[i + 2] as f32 / 255.0,
            ]);
        }
    }
    (v, w, h)
}

fn main() {
    let sources: Vec<(&str, &str, Option<(usize, usize)>)> = vec![
        ("cid_1279330", "CID22/CID22-512/validation/1279330.png", None),
        ("cid_1420710", "CID22/CID22-512/validation/1420710.png", None),
        ("gb82_graph", "gb82-sc/graph.png", None),
        ("gb82_terminal", "gb82-sc/terminal.png", None),
        // Small + odd sizes: 0.8.1 notes `scales_n < NUM_SCALES` shifts the
        // score() weight walk at 64x64, so small cells are load-bearing here.
        ("crop_64", "CID22/CID22-512/validation/1279330.png", Some((64, 64))),
        ("crop_33x17", "CID22/CID22-512/validation/1279330.png", Some((33, 17))),
        ("crop_9x9", "CID22/CID22-512/validation/1279330.png", Some((9, 9))),
        ("crop_5x5", "CID22/CID22-512/validation/1279330.png", Some((5, 5))),
    ];
    let root = std::env::var("CORPUS").unwrap_or_else(|_| {
        format!("{}/work/codec-corpus", std::env::var("HOME").unwrap())
    });
    println!("cell\tw\th\tkind\tstrength\tv0_7_1\tv0_8_2\tabs_delta\trel_delta\tbits_equal");
    let kinds = ["quant", "blur", "noise"];
    for (name, rel, crop) in &sources {
        let (src, w, h) = load(&format!("{root}/{rel}"), *crop);
        for (ki, kname) in kinds.iter().enumerate() {
            for strength in 1..=4usize {
                let dist = degrade(&src, w, h, ki, strength);
                let sref = Img::new(src.clone(), w, h);
                let dref = Img::new(dist.clone(), w, h);
                let old = ssim_old::Ssimulacra2Reference::new(sref.as_ref())
                    .and_then(|r| r.compare(dref.as_ref()));
                let sref2 = Img::new(src.clone(), w, h);
                let dref2 = Img::new(dist, w, h);
                let new = ssim_new::Ssimulacra2Reference::new(sref2.as_ref())
                    .and_then(|r| r.compare(dref2.as_ref()));
                match (old, new) {
                    (Ok(a), Ok(b)) => {
                        let d = (a - b).abs();
                        let rel = if a.abs() > 1e-12 { d / a.abs() } else { 0.0 };
                        println!(
                            "{name}\t{w}\t{h}\t{kname}\t{strength}\t{a:.17}\t{b:.17}\t{d:.3e}\t{rel:.3e}\t{}",
                            a.to_bits() == b.to_bits()
                        );
                    }
                    (Err(e), Ok(b)) => println!(
                        "{name}\t{w}\t{h}\t{kname}\t{strength}\tERR({e:?})\t{b:.17}\tNA\tNA\tBEHAVIOUR_CHANGE"
                    ),
                    (Ok(a), Err(e)) => println!(
                        "{name}\t{w}\t{h}\t{kname}\t{strength}\t{a:.17}\tERR({e:?})\tNA\tNA\tBEHAVIOUR_CHANGE"
                    ),
                    (Err(_), Err(_)) => println!(
                        "{name}\t{w}\t{h}\t{kname}\t{strength}\tERR\tERR\tNA\tNA\tboth_err"
                    ),
                }
            }
        }
    }
}
