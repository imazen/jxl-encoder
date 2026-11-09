//! Real encoder integration with a native prepared Butteraugli factory.
#![cfg(feature = "psy-loop")]
use jxl_encoder::loop_metric::PsyLoopMetric;
use jxl_encoder::{
    ColorEncoding, ExternalLoopMetric, ExternalMetric, LoopTrace, LossyConfig, PixelLayout,
    Primaries,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use zenpsy_api::{
    zenpixels::{PixelSlice, plane::*},
    *,
};
struct Factory(Arc<AtomicUsize>);
struct Prepared(butteraugli::ButteraugliReference);
struct Owned(butteraugli::ButteraugliResult);
fn specs() -> [ScoreSpec; 2] {
    [
        ScoreSpec::new("max", Direction::Lower)
            .with_perfect(0.0)
            .with_unit("butteraugli"),
        ScoreSpec::new("pnorm3", Direction::Lower)
            .with_perfect(0.0)
            .with_unit("butteraugli"),
    ]
}
struct Borrowed<'a> {
    values: [f64; 2],
    specs: [ScoreSpec; 2],
    spec: ScoreSpec,
    map: imgref::ImgRef<'a, f32>,
}
impl Evaluation for Borrowed<'_> {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.specs
            .iter()
            .zip(self.values)
            .map(|(s, v)| Score::new(s, v))
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        std::iter::once(MapView::new(
            &self.spec,
            AnyPlane::F32(
                PlaneRef::new(
                    *self.map.buf(),
                    self.map.width(),
                    self.map.height(),
                    self.map.stride(),
                )
                .unwrap(),
            ),
            MapGrid::new(self.map.width() as u32, self.map.height() as u32).unwrap(),
        ))
    }
}
impl Evaluation for Owned {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        // The owned path is unused by lending; native output is retained.
        static SPEC: ScoreSpec = ScoreSpec::new("max", Direction::Lower);
        std::iter::once(Score::new(&SPEC, self.0.score))
    }
}
fn planes<'a>(p: &PlanarSlice<'a>) -> [imgref::ImgRef<'a, f32>; 3] {
    core::array::from_fn(|c| {
        let AnyPlane::F32(v) = p.planes()[c] else {
            panic!("f32")
        };
        imgref::Img::new_stride(v.into_strided_samples(), v.width(), v.height(), v.stride())
    })
}
impl Metric for Factory {
    type Output = Owned;
    fn id(&self) -> MetricId {
        MetricId::new(
            "butteraugli",
            Implementation::new("butteraugli", "0.9.4", 1),
        )
    }
    fn evaluate(
        &mut self,
        _: &PixelSlice<'_>,
        _: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Owned> {
        fail(MetricError::new(ErrorKind::UnsupportedInput))
    }
}
impl PlanarMetric for Factory {
    fn evaluate_planar(
        &mut self,
        r: &PlanarSlice<'_>,
        d: &PlanarSlice<'_>,
        o: EvalOpts<'_>,
    ) -> MetricResult<Owned> {
        self.warm_planar_reference(r, o)?.evaluate_planar(d, o)
    }
}
impl PreparePlanarReference for Factory {
    type PlanarPrepared = Prepared;
    fn warm_planar_reference(
        &mut self,
        r: &PlanarSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<Prepared> {
        self.0.fetch_add(1, Ordering::SeqCst);
        butteraugli::ButteraugliReference::new_linear_planes(
            planes(r),
            butteraugli::ButteraugliParams::default()
                .with_intensity_target(80.0)
                .with_compute_diffmap(true),
        )
        .map(Prepared)
        .metric_err("butteraugli", ErrorKind::Backend)
    }
}
impl PlanarWarmReference for Prepared {
    type Output = Owned;
    fn evaluate_planar(&mut self, _: &PlanarSlice<'_>, _: EvalOpts<'_>) -> MetricResult<Owned> {
        panic!("encoder must use native lending")
    }
    fn with_evaluation(
        &mut self,
        d: &PlanarSlice<'_>,
        o: EvalOpts<'_>,
        f: &mut dyn FnMut(EvaluationRef<'_>),
    ) -> MetricResult<()> {
        self.0
            .compare_linear_planes_with(planes(d), o.stop(), |max, p3, map| {
                let result = Borrowed {
                    values: [max, p3],
                    specs: specs(),
                    spec: ScoreSpec::new("diffmap", Direction::Lower)
                        .with_perfect(0.0)
                        .with_unit("butteraugli"),
                    map,
                };
                f(EvaluationRef::new(&result));
            })
            .metric_err("butteraugli", ErrorKind::Backend)
    }
}
struct BorrowOnly(PsyLoopMetric<Factory>, Arc<AtomicUsize>);
impl ExternalLoopMetric for BorrowOnly {
    fn strict_failures(&self) -> bool {
        true
    }
    fn configure_color(&mut self, c: Option<&ColorEncoding>) -> Result<(), String> {
        self.0.configure_color(c)
    }
    fn set_reference(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        w: usize,
        h: usize,
    ) -> Result<(), String> {
        self.0.set_reference(r, g, b, w, h)
    }
    fn set_reference_with_stop(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        w: usize,
        h: usize,
        s: &dyn enough::Stop,
    ) -> Result<(), String> {
        self.0.set_reference_with_stop(r, g, b, w, h, s)
    }
    fn compare(
        &mut self,
        _: &[f32],
        _: &[f32],
        _: &[f32],
        _: usize,
        _: usize,
        _: usize,
        _: &mut Vec<f32>,
    ) -> Result<f64, String> {
        panic!("untraced encoder must not export a map Vec")
    }
    fn compare_with_map(
        &mut self,
        p: [imgref::ImgRef<'_, f32>; 3],
        s: &dyn enough::Stop,
        f: &mut dyn FnMut(imgref::ImgRef<'_, f32>),
    ) -> Option<Result<f64, String>> {
        self.1.fetch_add(1, Ordering::SeqCst);
        self.0.compare_with_map(p, s, f)
    }
}
struct FullSchedule(bool);
impl Drop for FullSchedule {
    fn drop(&mut self) {
        jxl_encoder::vardct::__buttloop_overrides::FIXPOINT_STOP.store(self.0, Ordering::Relaxed);
    }
}
#[test]
fn native_lending_steers_actual_encode_without_an_export_and_matches_native_trace() {
    let _schedule = FullSchedule(
        jxl_encoder::vardct::__buttloop_overrides::FIXPOINT_STOP.swap(false, Ordering::Relaxed),
    );
    let (w, h) = (97u32, 73u32);
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = ((i * 31 + (i / w) * 47) % 211 + 13) as u8;
            [v, v.wrapping_add(17), v.wrapping_add(39)]
        })
        .collect();
    let cfg = LossyConfig::new(1.5)
        .with_effort(7)
        .with_butteraugli_iters(3);
    let baseline_trace = LoopTrace::new();
    let baseline = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_loop_trace(&baseline_trace)
        .encode(&pixels)
        .unwrap();
    let prepared = Arc::new(AtomicUsize::new(0));
    let lends = Arc::new(AtomicUsize::new(0));
    let metric = ExternalMetric::new(BorrowOnly(
        PsyLoopMetric::new(Factory(prepared.clone()), "max", "diffmap"),
        lends.clone(),
    ));
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_external_metric(&metric)
        .encode(&pixels)
        .unwrap();
    assert_eq!(result, baseline);
    assert_eq!(prepared.load(Ordering::SeqCst), 1);
    assert!(lends.load(Ordering::SeqCst) > 0);
    let traced = ExternalMetric::new(PsyLoopMetric::new(
        Factory(Arc::default()),
        "max",
        "diffmap",
    ));
    let trace = LoopTrace::new();
    let result = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_external_metric(&traced)
        .with_loop_trace(&trace)
        .encode(&pixels)
        .unwrap();
    assert_eq!(result, baseline);
    let (a, b) = (trace.frames(), baseline_trace.frames());
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.score.to_bits(), b.score.to_bits());
        assert_eq!(a.diffmap, b.diffmap);
        assert_eq!(a.tile_dist, b.tile_dist);
    }
}
#[test]
fn unsupported_color_refuses_before_reference_preparation() {
    let count = Arc::new(AtomicUsize::new(0));
    let mut adapter = PsyLoopMetric::new(Factory(count.clone()), "max", "diffmap");
    let mut c = ColorEncoding::srgb();
    c.primaries = Primaries::P3;
    assert!(adapter.configure_color(Some(&c)).is_err());
    assert!(
        adapter
            .configure_color(Some(&ColorEncoding::bt2100_pq()))
            .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    fn send<T: Send>() {}
    send::<PsyLoopMetric<Factory>>();
}

#[test]
fn typed_map_and_color_refusals_fail_actual_encoding() {
    let (w, h) = (97u32, 73u32);
    let px: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 31 + i / 17) % 211 + 13) as u8)
        .collect();
    let cfg = LossyConfig::new(1.5)
        .with_effort(7)
        .with_butteraugli_iters(2);
    let count = Arc::new(AtomicUsize::new(0));
    let metric = ExternalMetric::new(PsyLoopMetric::new(Factory(count.clone()), "max", "missing"));
    let error = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_external_metric(&metric)
        .encode(&px)
        .unwrap_err();
    assert!(error.to_string().contains("requested common map"));
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let metric = ExternalMetric::new(PsyLoopMetric::new(Factory(count.clone()), "max", "diffmap"));
    let mut color = ColorEncoding::srgb();
    color.primaries = Primaries::P3;
    let error = cfg
        .encode_request(w, h, PixelLayout::Rgb8)
        .with_color_encoding(color)
        .with_external_metric(&metric)
        .encode(&px)
        .unwrap_err();
    assert!(error.to_string().contains("requires declared SDR"));
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn each_preparation_and_comparison_uses_its_current_stop() {
    struct Cancel;
    impl enough::Stop for Cancel {
        fn check(&self) -> Result<(), enough::StopReason> {
            Err(enough::StopReason::Cancelled)
        }
    }
    let count = Arc::new(AtomicUsize::new(0));
    let mut adapter = PsyLoopMetric::new(Factory(count.clone()), "max", "diffmap");
    let pixels: Vec<f32> = (0..64).map(|i| (i * 17 % 61) as f32 / 64.0).collect();
    assert!(
        adapter
            .set_reference_with_stop(&pixels, &pixels, &pixels, 8, 8, &Cancel)
            .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    adapter
        .set_reference_with_stop(&pixels, &pixels, &pixels, 8, 8, &enough::Unstoppable)
        .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let planes = [pixels.as_slice(); 3].map(|p| imgref::Img::new(p, 8, 8));
    assert!(
        adapter
            .compare_with_map(planes, &Cancel, &mut |_| panic!(
                "cancelled call lent a result"
            ))
            .unwrap()
            .is_err()
    );
    let mut calls = 0;
    adapter
        .compare_with_map(planes, &enough::Unstoppable, &mut |_| calls += 1)
        .unwrap()
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
