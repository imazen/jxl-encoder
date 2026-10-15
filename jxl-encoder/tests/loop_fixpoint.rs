//! The perceptual loop stops once an iteration leaves the quant field
//! unchanged. The remaining iterations would re-encode that same field, so
//! stopping must give the same bytes as running the whole schedule.
//!
//! Its own test binary: it flips the process-global `FIXPOINT_STOP` switch,
//! which would race the trace tests in the shared `it` binary.
#![cfg(feature = "butteraugli-loop")]

use std::sync::atomic::Ordering;

use jxl_encoder::vardct::__buttloop_overrides::FIXPOINT_STOP;
use jxl_encoder::{LoopFrame, LoopTrace, LossyConfig, PixelLayout};

fn textured(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let edge = if (x / 9 + y / 7) % 2 == 0 { 50 } else { 0 };
            let v = ((x * 3 + y * 5) % 180 + edge) as u8;
            [v, v.wrapping_add(31), v.wrapping_mul(3)]
        })
        .collect()
}

fn gradient(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y) * 127 / (w + h)) as u8,
            ]
        })
        .collect()
}

fn encode(
    px: &[u8],
    w: usize,
    h: usize,
    distance: f32,
    iters: u32,
    stop: bool,
) -> (Vec<u8>, Vec<LoopFrame>) {
    FIXPOINT_STOP.store(stop, Ordering::Relaxed);
    let trace = LoopTrace::new();
    let bytes = LossyConfig::new(distance)
        .with_effort(7)
        .with_butteraugli_iters(iters)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_loop_trace(&trace)
        .encode(px)
        .unwrap();
    FIXPOINT_STOP.store(true, Ordering::Relaxed);
    (bytes, trace.take())
}

fn same_frame(a: &LoopFrame, b: &LoopFrame) -> bool {
    a.iter == b.iter
        && a.score.to_bits() == b.score.to_bits()
        && a.quant_field == b.quant_field
        && a.tile_dist == b.tile_dist
        && a.diffmap == b.diffmap
        && a.recon == b.recon
}

#[test]
fn fixpoint_stop_keeps_the_bytes_and_drops_only_repeats() {
    let (w, h) = (96usize, 72usize);
    let iters = 8u32;
    let mut stopped_early = 0;
    for (name, px) in [("textured", textured(w, h)), ("gradient", gradient(w, h))] {
        for distance in [0.5f32, 1.0, 2.0, 4.0] {
            let cell = format!("{name} d{distance}");
            let (full, full_frames) = encode(&px, w, h, distance, iters, false);
            let (stop, stop_frames) = encode(&px, w, h, distance, iters, true);
            assert_eq!(stop, full, "{cell}: the fixed-point stop changed the bytes");

            let seeds = |f: &[LoopFrame]| {
                f.chunk_by(|a, b| a.seed.to_bits() == b.seed.to_bits())
                    .map(<[LoopFrame]>::to_vec)
                    .collect::<Vec<_>>()
            };
            let (full_seeds, stop_seeds) = (seeds(&full_frames), seeds(&stop_frames));
            assert_eq!(full_seeds.len(), stop_seeds.len(), "{cell}: seed count");
            for (fs, ss) in full_seeds.iter().zip(&stop_seeds) {
                assert_eq!(fs.len(), iters as usize + 1, "{cell}: the full schedule");
                assert!(ss.len() <= fs.len(), "{cell}");
                // The stopped run is a prefix of the full one.
                for (a, b) in ss.iter().zip(fs) {
                    assert!(same_frame(a, b), "{cell}: iteration {} differs", a.iter);
                }
                if ss.len() < fs.len() {
                    stopped_early += 1;
                    let last = ss.last().unwrap();
                    assert!(
                        last.iter >= 2,
                        "{cell}: stopped before the cur_pow = 0 regime"
                    );
                    assert_eq!(
                        last.next_quant_field.as_deref(),
                        Some(&last.quant_field[..]),
                        "{cell}: stopped although the field moved"
                    );
                    // Every dropped iteration repeats the last one exactly.
                    for f in &fs[ss.len()..] {
                        assert_eq!(f.quant_field, last.quant_field, "{cell}");
                        assert_eq!(f.score.to_bits(), last.score.to_bits(), "{cell}");
                        assert_eq!(f.diffmap, last.diffmap, "{cell}");
                    }
                } else {
                    assert_eq!(ss.last().unwrap().next_quant_field, None, "{cell}");
                }
            }
        }
    }
    assert!(
        stopped_early > 0,
        "no cell reached a fixed point; the test checks nothing"
    );
}
