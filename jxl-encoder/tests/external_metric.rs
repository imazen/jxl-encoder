//! `EncodeRequest::with_external_metric`: a caller's metric scores the
//! perceptual loop in butteraugli's place.
//!
//! Its own process: the full-schedule equivalence oracle temporarily disables
//! the global fixed-point switch. Other integration binaries keep their own
//! independent switch, and the tests here serialize access.
#![cfg(feature = "__external_metric")]

use jxl_encoder::api::PerceptualMetric;
use jxl_encoder::{
    EncodeError, ExternalLoopMetric, ExternalMetric, LoopTrace, LossyConfig, PixelLayout,
};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use jxl_encoder::vardct::__buttloop_overrides::FIXPOINT_STOP;

#[path = "common/external_butteraugli.rs"]
mod external_butteraugli;
use external_butteraugli::Butteraugli;

static TEST_LOCK: Mutex<()> = Mutex::new(());

struct FullSchedule(bool);

impl FullSchedule {
    fn new() -> Self {
        Self(FIXPOINT_STOP.swap(false, Ordering::Relaxed))
    }
}

impl Drop for FullSchedule {
    fn drop(&mut self) {
        FIXPOINT_STOP.store(self.0, Ordering::Relaxed);
    }
}

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

/// A metric that sees no error anywhere.
struct Blind;

impl ExternalLoopMetric for Blind {
    fn set_reference(
        &mut self,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: usize,
        _: usize,
    ) -> Result<(), String> {
        Ok(())
    }
    fn compare(
        &mut self,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: usize,
        width: usize,
        height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String> {
        diffmap.clear();
        diffmap.resize(width * height, 0.0);
        Ok(0.0)
    }
}

#[test]
fn butteraugli_as_an_external_metric_reproduces_the_builtin_loop() {
    let _lock = TEST_LOCK.lock().unwrap();
    // Compare the same full schedule on both paths. The separate
    // loop_fixpoint binary tests the optimized schedule and external bypass.
    let _schedule = FullSchedule::new();
    let (w, h) = (96usize, 72usize);
    let px = textured(w, h);
    let config = LossyConfig::new(1.5)
        .with_effort(7)
        .with_butteraugli_iters(3);
    let builtin_trace = LoopTrace::new();
    let builtin = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_loop_trace(&builtin_trace)
        .encode(&px)
        .unwrap();
    let calls = Arc::new(Mutex::new(0));
    let metric = ExternalMetric::new(Butteraugli {
        reference: None,
        calls: calls.clone(),
    });
    let trace = LoopTrace::new();
    let external = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_external_metric(&metric)
        .with_loop_trace(&trace)
        .encode(&px)
        .unwrap();
    assert_eq!(external, builtin, "same metric, same loop, same bytes");
    let (frames, builtin_frames) = (trace.frames(), builtin_trace.frames());
    assert_eq!(frames.len(), builtin_frames.len());
    assert_eq!(*calls.lock().unwrap() as usize, frames.len());
    for (a, b) in frames.iter().zip(&builtin_frames) {
        assert_eq!((a.metric, b.metric), ("external", "butteraugli"));
        assert_eq!(a.score.to_bits(), b.score.to_bits());
        assert_eq!(a.diffmap, b.diffmap);
        assert_eq!(a.tile_dist, b.tile_dist);
    }
}

#[test]
fn an_external_metric_steers_the_quant_field() {
    let _lock = TEST_LOCK.lock().unwrap();
    let (w, h) = (96usize, 72usize);
    let px = textured(w, h);
    let config = LossyConfig::new(1.5)
        .with_effort(7)
        .with_butteraugli_iters(3);
    let builtin = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .encode(&px)
        .unwrap();
    let trace = LoopTrace::new();
    let blind = config
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_external_metric(&ExternalMetric::new(Blind))
        .with_loop_trace(&trace)
        .encode(&px)
        .unwrap();
    // No block is ever above target, so every adjustment coarsens or keeps.
    let frames = trace.frames();
    assert!(!frames.is_empty());
    for f in &frames {
        assert!(f.tile_dist.iter().all(|&d| d == 0.0));
        if let Some(next) = &f.next_quant_field {
            assert!(next.iter().zip(&f.quant_field).all(|(n, q)| n <= q));
        }
    }
    assert!(
        blind.len() < builtin.len(),
        "{} vs {}",
        blind.len(),
        builtin.len()
    );
}

#[test]
fn an_external_metric_refuses_other_loop_metrics() {
    let _lock = TEST_LOCK.lock().unwrap();
    let (w, h) = (32u32, 32u32);
    let px = textured(w as usize, h as usize);
    let compiled = [
        (PerceptualMetric::Cvvdp, cfg!(feature = "cvvdp-loop")),
        (
            PerceptualMetric::Zensim,
            cfg!(any(feature = "zensim-loop", feature = "zensim-loop-gpu")),
        ),
    ];
    for (metric, compiled) in compiled {
        let config = LossyConfig::new(1.0).with_perceptual_metric(metric);
        let result = config
            .encode_request(w, h, PixelLayout::Rgb8)
            .with_external_metric(&ExternalMetric::new(Blind))
            .encode(&px);
        // Without that metric's loop feature compiled in, the resolver keeps
        // butteraugli and the external metric applies.
        if compiled {
            let error = result.unwrap_err();
            assert!(
                matches!(error.error(), EncodeError::InvalidConfig { .. }),
                "{error:?}"
            );
        } else {
            result.unwrap();
        }
    }
}
