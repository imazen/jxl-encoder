//! `EncodeRequest::with_loop_trace`: the perceptual loop records each scored
//! iteration without changing the encoded bytes.

use jxl_encoder::{LoopTrace, LossyConfig, PixelLayout};

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

#[test]
fn loop_trace_records_every_iteration_and_keeps_the_bytes() {
    let (w, h) = (96usize, 72usize);
    let px = textured(w, h);
    let iters = 3u32;
    let config = LossyConfig::new(1.5)
        .with_effort(7)
        .with_butteraugli_iters(iters);
    let plain = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .encode(&px)
        .unwrap();
    let trace = LoopTrace::new();
    let traced = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_loop_trace(&trace)
        .encode(&px)
        .unwrap();
    assert_eq!(traced, plain, "a trace must not change the bytes");

    let frames = trace.frames();
    assert!(!frames.is_empty(), "the loop ran {iters} iterations");
    let (xb, yb) = (w.div_ceil(8), h.div_ceil(8));
    for seed in frames.chunk_by(|a, b| a.seed.to_bits() == b.seed.to_bits()) {
        // The full schedule, or fewer iterations when the loop reached a
        // fixed point (its last frame's next field equals its own field;
        // `tests/loop_fixpoint.rs` checks the dropped iterations repeat it).
        let last = seed.last().unwrap();
        if !(seed.len() == iters as usize + 1 && last.next_quant_field.is_none()) {
            assert!(seed.len() <= iters as usize && last.iter >= 2);
            assert_eq!(
                last.next_quant_field.as_deref(),
                Some(&last.quant_field[..])
            );
        }
        for (k, f) in seed.iter().enumerate() {
            assert_eq!(f.iter, k as u32);
            assert_eq!(f.metric, "butteraugli");
            assert_eq!((f.width, f.height), (w as u32, h as u32));
            assert_eq!((f.xsize_blocks, f.ysize_blocks), (xb as u32, yb as u32));
            assert_eq!(f.quant_field.len(), xb * yb);
            assert_eq!(f.tile_dist.len(), xb * yb);
            assert_eq!(f.diffmap.len(), w * h);
            assert!(f.recon.iter().all(|p| p.len() == w * h));
            assert!(f.score.is_finite() && f.score >= 0.0 && f.target > 0.0);
            assert!(f.diffmap.iter().all(|v| v.is_finite() && *v >= 0.0));
            // The adjusted field is what the next iteration encodes with.
            if let Some(next) = seed.get(k + 1) {
                assert_eq!(f.next_quant_field.as_deref(), Some(&next.quant_field[..]))
            }
        }
    }
    assert!(
        frames.windows(2).any(|p| p[0]
            .next_quant_field
            .as_ref()
            .is_some_and(|q| q != &p[0].quant_field)),
        "the loop adjusted the field at least once"
    );
}
