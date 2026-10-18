//! The perceptual loop stops once an iteration leaves the quant field
//! unchanged. The remaining iterations would re-encode that same field, so
//! stopping must give the same bytes as running the whole schedule.
//!
//! Its own test binary: it flips the process-global `FIXPOINT_STOP` switch,
//! which would race the trace tests in the shared `it` binary.
#![cfg(feature = "butteraugli-loop")]

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use jxl_encoder::vardct::__buttloop_overrides::FIXPOINT_STOP;
use jxl_encoder::{LoopFrame, LoopTrace, LossyConfig, PixelLayout};

#[cfg(feature = "__external_metric")]
#[path = "common/external_butteraugli.rs"]
mod external_butteraugli;

static TEST_LOCK: Mutex<()> = Mutex::new(());

struct FixpointSwitch(bool);

impl FixpointSwitch {
    fn set(enabled: bool) -> Self {
        Self(FIXPOINT_STOP.swap(enabled, Ordering::Relaxed))
    }
}

impl Drop for FixpointSwitch {
    fn drop(&mut self) {
        FIXPOINT_STOP.store(self.0, Ordering::Relaxed);
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
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
    encode_at_effort(px, w, h, distance, iters, stop, 7)
}

fn encode_at_effort(
    px: &[u8],
    w: usize,
    h: usize,
    distance: f32,
    iters: u32,
    stop: bool,
    effort: u8,
) -> (Vec<u8>, Vec<LoopFrame>) {
    let _switch = FixpointSwitch::set(stop);
    let trace = LoopTrace::new();
    let bytes = LossyConfig::new(distance)
        .with_effort(effort)
        .with_butteraugli_iters(iters)
        .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
        .with_loop_trace(&trace)
        .encode(px)
        .unwrap();
    (bytes, trace.take())
}

fn same_frame(a: &LoopFrame, b: &LoopFrame) -> bool {
    a.seed.to_bits() == b.seed.to_bits()
        && a.iter == b.iter
        && a.metric == b.metric
        && a.target.to_bits() == b.target.to_bits()
        && a.score.to_bits() == b.score.to_bits()
        && same_bits(&a.quant_field, &b.quant_field)
        && same_bits(&a.tile_dist, &b.tile_dist)
        && same_bits(&a.diffmap, &b.diffmap)
        && a.recon.iter().zip(&b.recon).all(|(a, b)| same_bits(a, b))
}

#[test]
fn fixpoint_stop_keeps_the_bytes_and_drops_only_repeats() {
    let _lock = TEST_LOCK.lock().unwrap();
    let (w, h) = (96usize, 72usize);
    let mut stopped_early = 0;
    for (effort, iters, seed_count) in [(7, 8, 1), (11, 8, 2), (12, 16, 4)] {
        for (name, px) in [("textured", textured(w, h)), ("gradient", gradient(w, h))] {
            for distance in [0.5f32, 1.0, 2.0, 4.0] {
                let cell = format!("{name} e{effort} d{distance}");
                let (full, full_frames) =
                    encode_at_effort(&px, w, h, distance, iters, false, effort);
                let (stop, stop_frames) =
                    encode_at_effort(&px, w, h, distance, iters, true, effort);
                assert_eq!(stop, full, "{cell}: the fixed-point stop changed the bytes");

                let seeds = |f: &[LoopFrame]| {
                    f.chunk_by(|a, b| a.seed.to_bits() == b.seed.to_bits())
                        .map(<[LoopFrame]>::to_vec)
                        .collect::<Vec<_>>()
                };
                let (full_seeds, stop_seeds) = (seeds(&full_frames), seeds(&stop_frames));
                assert_eq!(
                    full_seeds.len(),
                    seed_count,
                    "{cell}: scheduled seed coverage"
                );
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
                        assert!(
                            same_bits(last.next_quant_field.as_deref().unwrap(), &last.quant_field),
                            "{cell}: stopped although the field moved"
                        );
                        // Every dropped iteration repeats the last one exactly.
                        for f in &fs[ss.len()..] {
                            let mut repeated = last.clone();
                            repeated.iter = f.iter;
                            assert!(same_frame(f, &repeated), "{cell}: omitted state changed");
                            if let Some(next) = &f.next_quant_field {
                                assert!(same_bits(next, &last.quant_field), "{cell}");
                            }
                        }
                    } else {
                        assert_eq!(ss.last().unwrap().next_quant_field, None, "{cell}");
                    }
                }
            }
        }
    }
    assert!(
        stopped_early > 0,
        "no cell reached a fixed point; the test checks nothing"
    );
}

#[test]
fn cancellation_after_a_scored_frame_preserves_the_destination() {
    use jxl_encoder::EncodeError;

    struct CancelAtFrame<'a>(&'a LoopTrace);
    impl enough::Stop for CancelAtFrame<'_> {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.frames().len() >= 3 {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }

    let _lock = TEST_LOCK.lock().unwrap();
    let px = gradient(96, 72);
    for enabled in [false, true] {
        let _switch = FixpointSwitch::set(enabled);
        let trace = LoopTrace::new();
        let stop = CancelAtFrame(&trace);
        let mut destination = vec![1, 3, 5, 7];
        let result = LossyConfig::new(4.0)
            .with_effort(7)
            .with_butteraugli_iters(8)
            .encode_request(96, 72, PixelLayout::Rgb8)
            .with_loop_trace(&trace)
            .with_stop(&stop)
            .encode_into(&px, &mut destination);
        assert!(matches!(result, Err(e) if matches!(e.error(), EncodeError::Cancelled)));
        assert_eq!(destination, [1, 3, 5, 7]);
        assert_eq!(trace.frames().len(), 3);
    }
    let (full, _) = encode(&px, 96, 72, 4.0, 8, false);
    let (stopped, _) = encode(&px, 96, 72, 4.0, 8, true);
    assert_eq!(stopped, full, "cancellation must not poison a later encode");
}

#[cfg(feature = "__internal_recon_hook")]
#[test]
fn reconstruction_capture_keeps_the_scheduled_terminal_frame() {
    use jxl_encoder::vardct::__recon_hook as recon_hook;

    struct CaptureReset;
    impl Drop for CaptureReset {
        fn drop(&mut self) {
            recon_hook::set_capture_enabled(false);
        }
    }

    let _lock = TEST_LOCK.lock().unwrap();
    let _reset = CaptureReset;
    recon_hook::set_capture_enabled(true);
    let (_, frames) = encode(&gradient(32, 32), 32, 32, 4.0, 8, true);
    assert_eq!(frames.len(), 9);
    assert!(frames.last().unwrap().next_quant_field.is_none());
    let captured = recon_hook::take_last().expect("scheduled final reconstruction");
    let terminal = frames.last().unwrap();
    assert_eq!((captured.iter, captured.iters), (8, 8));
    // Both the hook and trace expose the cropped logical planes.
    assert!(same_bits(&captured.r, &terminal.recon[0]));
    assert!(same_bits(&captured.g, &terminal.recon[1]));
    assert!(same_bits(&captured.b, &terminal.recon[2]));
}

#[cfg(feature = "__external_metric")]
#[test]
fn deterministic_external_metric_keeps_the_full_schedule_when_builtin_stops() {
    use jxl_encoder::ExternalMetric;
    use std::sync::Arc;

    let _lock = TEST_LOCK.lock().unwrap();
    let _switch = FixpointSwitch::set(true);
    let px = textured(96, 72);
    let config = LossyConfig::new(1.5)
        .with_effort(7)
        .with_butteraugli_iters(8);
    let builtin_trace = LoopTrace::new();
    let builtin = config
        .encode_request(96, 72, PixelLayout::Rgb8)
        .with_loop_trace(&builtin_trace)
        .encode(&px)
        .unwrap();
    let calls = Arc::new(Mutex::new(0));
    let metric = ExternalMetric::new(external_butteraugli::Butteraugli {
        reference: None,
        calls: calls.clone(),
    });
    let trace = LoopTrace::new();
    let external = config
        .encode_request(96, 72, PixelLayout::Rgb8)
        .with_external_metric(&metric)
        .with_loop_trace(&trace)
        .encode(&px)
        .unwrap();
    assert_eq!(external, builtin);
    let (frames, builtin_frames) = (trace.take(), builtin_trace.take());
    assert_eq!(frames.len(), 9);
    assert_eq!(*calls.lock().unwrap() as usize, frames.len());
    assert!(
        builtin_frames.len() < frames.len(),
        "fixture must stop early"
    );
    for (a, b) in frames.iter().zip(&builtin_frames) {
        assert_eq!((a.metric, b.metric), ("external", "butteraugli"));
        let mut expected = b.clone();
        expected.metric = "external";
        assert!(same_frame(a, &expected));
    }
    let last = builtin_frames.last().unwrap();
    assert!(last.iter >= 2);
    assert!(same_bits(
        last.next_quant_field.as_ref().unwrap(),
        &last.quant_field
    ));
    for frame in &frames[builtin_frames.len()..] {
        let mut expected = last.clone();
        expected.metric = "external";
        expected.iter = frame.iter;
        assert!(same_frame(frame, &expected));
    }
    assert!(frames.last().unwrap().next_quant_field.is_none());
}

#[cfg(feature = "__external_metric")]
#[test]
fn external_callbacks_keep_the_schedule_after_unchanged_and_saturated_fields() {
    use jxl_encoder::{ExternalLoopMetric, ExternalMetric};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    struct Stateful {
        calls: Arc<AtomicUsize>,
        delayed: bool,
    }
    impl ExternalLoopMetric for Stateful {
        fn set_reference(
            &mut self,
            _: &[f32],
            _: &[f32],
            _: &[f32],
            _: usize,
            _: usize,
        ) -> Result<(), String> {
            self.calls.store(0, Ordering::Relaxed);
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
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            // Iteration 2 is stationary in the delayed case, but iteration
            // 4 changes the metric and must still run. The other case
            // saturates the field while every block remains above target.
            let value = if self.delayed && call < 4 { 0.0 } else { 100.0 };
            diffmap.clear();
            diffmap.resize(width * height, value);
            Ok(f64::from(value))
        }
    }

    let _lock = TEST_LOCK.lock().unwrap();
    let px = textured(32, 32);
    for delayed in [false, true] {
        let run = |enabled| {
            let _switch = FixpointSwitch::set(enabled);
            let trace = LoopTrace::new();
            let calls = Arc::new(AtomicUsize::new(0));
            let metric = ExternalMetric::new(Stateful {
                calls: calls.clone(),
                delayed,
            });
            let bytes = LossyConfig::new(1.0)
                .with_effort(7)
                .with_butteraugli_iters(8)
                .encode_request(32, 32, PixelLayout::Rgb8)
                .with_external_metric(&metric)
                .with_loop_trace(&trace)
                .encode(&px)
                .unwrap();
            let frames = trace.take();
            assert_eq!(frames.len(), 9);
            assert_eq!(calls.load(Ordering::Relaxed), frames.len());
            assert!(frames.last().unwrap().next_quant_field.is_none());
            (bytes, frames)
        };
        let (full, full_frames) = run(false);
        let (enabled, enabled_frames) = run(true);
        assert_eq!(enabled, full);
        for (a, b) in enabled_frames.iter().zip(&full_frames) {
            assert!(same_frame(a, b));
        }
        if delayed {
            let frame = &enabled_frames[2];
            assert!(same_bits(
                frame.next_quant_field.as_ref().unwrap(),
                &frame.quant_field
            ));
            assert!(enabled_frames[4].score > frame.score);
            assert!(!same_bits(
                &enabled_frames[4].quant_field,
                &enabled_frames[5].quant_field
            ));
        } else {
            let frame = &enabled_frames[7];
            assert!(frame.tile_dist.iter().all(|&d| f64::from(d) > frame.target));
            assert!(same_bits(
                frame.next_quant_field.as_ref().unwrap(),
                &frame.quant_field
            ));
            assert!(same_bits(
                &enabled_frames[8].quant_field,
                &frame.quant_field
            ));
        }
    }
}
