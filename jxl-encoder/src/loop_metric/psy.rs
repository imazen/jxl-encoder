//! Prepared, borrowed integration; target and reducer policy remain in JXL.
use super::ExternalLoopMetric;
use alloc::{
    borrow::Cow,
    string::{String, ToString},
    vec::Vec,
};
use zenpsy_api::{
    self as psy, EvalOpts, EvaluationRef, PreparePlanarReference, PreparedPlanarReference,
    zenpixels::{self as pixels, plane::*},
};

/// Adapt owned planar preparation to JXL's stateful refinement callback.
/// The selected scalar/map must declare native `butteraugli` distance units,
/// lower-is-better and zero identity. Other units, reduced/undefined maps and
/// HDR/wide-gamut source interpretation are refused, never approximated.
/// Map calibration for another metric belongs in a separately qualified adapter.
pub struct PsyLoopMetric<M: PreparePlanarReference> {
    metric: M,
    prepared: Option<PreparedPlanarReference<M::PlanarPrepared>>,
    score_key: Cow<'static, str>,
    map_key: Cow<'static, str>,
    layout: PlanarLayout,
    color: pixels::ColorContext,
}
impl<M: PreparePlanarReference> PsyLoopMetric<M> {
    /// Select the metric's qualified scalar and full-resolution distance map.
    pub fn new(
        metric: M,
        score_key: impl Into<Cow<'static, str>>,
        map_key: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            metric,
            prepared: None,
            score_key: score_key.into(),
            map_key: map_key.into(),
            layout: PlanarLayout::new(
                [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue]
                    .map(|r| PlaneSpec::new(r, PlaneEncoding::Float(FloatEncoding::NOMINAL)))
                    .to_vec(),
                pixels::sample::ChromaSubsampling::Yuv444,
                pixels::sample::ChromaSiting::Unknown,
            )
            .expect("RGB layout"),
            color: pixels::ColorContext::from_cicp(pixels::Cicp::new(1, 8, 0, true))
                .with_diffuse_white(pixels::hdr::DiffuseWhite::new(80.0)),
        }
    }
    fn prepare(
        &mut self,
        p: [imgref::ImgRef<'_, f32>; 3],
        stop: &dyn enough::Stop,
    ) -> Result<(), String> {
        self.prepared = None;
        let (w, h) = dimensions(p[0])?;
        let planes = views(p)?;
        let image = PlanarSlice::new(&planes, &self.layout, w, h, &self.color, None)
            .map_err(|e| e.to_string())?;
        let pulse = psy::how_far::WithStop::borrowed(&psy::how_far::NoPulse, stop);
        self.prepared = Some(
            psy::prepare_planar_reference(
                &mut self.metric,
                &image,
                EvalOpts::new().with_pulse(&pulse),
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(())
    }
    fn lend(
        &mut self,
        p: [imgref::ImgRef<'_, f32>; 3],
        stop: &dyn enough::Stop,
        consume: &mut dyn FnMut(imgref::ImgRef<'_, f32>),
    ) -> Result<f64, String> {
        let (w, h) = dimensions(p[0])?;
        let planes = views(p)?;
        let image = PlanarSlice::new(&planes, &self.layout, w, h, &self.color, None)
            .map_err(|e| e.to_string())?;
        let warm = self
            .prepared
            .as_mut()
            .ok_or_else(|| String::from("metric reference is not prepared"))?;
        let pulse = psy::how_far::WithStop::borrowed(&psy::how_far::NoPulse, stop);
        warm.with_evaluation(
            &image,
            EvalOpts::new().with_pulse(&pulse).with_map(&self.map_key),
            |view| select(view, &self.score_key, &self.map_key, consume),
        )
        .map_err(|e| e.to_string())
    }
}
fn dimensions(p: imgref::ImgRef<'_, f32>) -> Result<(u32, u32), String> {
    Ok((
        u32::try_from(p.width()).map_err(|e| e.to_string())?,
        u32::try_from(p.height()).map_err(|e| e.to_string())?,
    ))
}
fn views(p: [imgref::ImgRef<'_, f32>; 3]) -> Result<[AnyPlane<'_>; 3], String> {
    let [r, g, b] = p;
    fn plane(p: imgref::ImgRef<'_, f32>) -> Result<AnyPlane<'_>, String> {
        PlaneRef::new(*p.buf(), p.width(), p.height(), p.stride())
            .map(AnyPlane::F32)
            .map_err(|e| e.to_string())
    }
    Ok([plane(r)?, plane(g)?, plane(b)?])
}
fn select(
    view: EvaluationRef<'_>,
    score_key: &str,
    map_key: &str,
    consume: &mut dyn FnMut(imgref::ImgRef<'_, f32>),
) -> psy::MetricResult<f64> {
    let mut score = None;
    let mut map = None;
    view.visit_scores(|s| {
        if s.spec.key() == score_key {
            score = Some(s)
        }
    });
    view.visit_maps(|m| {
        if m.spec.key() == map_key {
            map = Some(m)
        }
    });
    let fail = || {
        psy::whereat::at(psy::MetricError::new(psy::ErrorKind::UnsupportedRequest).with_message("JXL requires a qualified finite, nonnegative full-resolution butteraugli-distance signal"))
    };
    let score = score.ok_or_else(fail)?;
    let map = map.ok_or_else(fail)?;
    for spec in [score.spec, map.spec] {
        if spec.direction() != psy::Direction::Lower
            || spec.perfect() != Some(0.0)
            || spec.unit() != "butteraugli"
        {
            return Err(fail());
        }
    }
    let AnyPlane::F32(p) = map.values else {
        return Err(fail());
    };
    if !score.value.is_finite()
        || score.value < 0.0
        || map.grid.origin_x() != 0
        || map.grid.origin_y() != 0
        || map.grid.cell_width() != 1
        || map.grid.cell_height() != 1
        || p.width() != map.grid.input_width() as usize
        || p.height() != map.grid.input_height() as usize
        || p.rows().flatten().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(fail());
    }
    consume(imgref::Img::new_stride(
        p.into_strided_samples(),
        p.width(),
        p.height(),
        p.stride(),
    ));
    Ok(score.value)
}
impl<M> ExternalLoopMetric for PsyLoopMetric<M>
where
    M: PreparePlanarReference + Send,
    M::PlanarPrepared: Send,
{
    fn strict_failures(&self) -> bool {
        true
    }
    fn compare_with_stop(
        &mut self,
        planes: [imgref::ImgRef<'_, f32>; 3],
        out: &mut Vec<f32>,
        stop: &dyn enough::Stop,
    ) -> Result<f64, String> {
        self.lend(planes, stop, &mut |map| {
            out.clear();
            for row in map.rows() {
                out.extend_from_slice(row)
            }
        })
    }
    fn configure_color(
        &mut self,
        color: Option<&crate::headers::color_encoding::ColorEncoding>,
    ) -> Result<(), String> {
        self.prepared = None;
        use crate::headers::color_encoding::{ColorSpace, Primaries, TransferFunction, WhitePoint};
        if color.is_some_and(|c| {
            c.want_icc
                || c.color_space != ColorSpace::Rgb
                || c.primaries != Primaries::Srgb
                || c.white_point != WhitePoint::D65
                || matches!(
                    c.transfer_function,
                    TransferFunction::Pq | TransferFunction::Hlg | TransferFunction::Unknown
                )
        }) {
            return Err(String::from(
                "zenpsy loop requires declared SDR sRGB/BT.709 primaries and D65; HDR/wide-gamut needs a qualified adapter",
            ));
        }
        Ok(())
    }
    fn set_reference(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        w: usize,
        h: usize,
    ) -> Result<(), String> {
        self.prepare(
            [r, g, b].map(|p| imgref::Img::new(p, w, h)),
            &enough::Unstoppable,
        )
    }
    fn set_reference_with_stop(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        w: usize,
        h: usize,
        stop: &dyn enough::Stop,
    ) -> Result<(), String> {
        self.prepare([r, g, b].map(|p| imgref::Img::new(p, w, h)), stop)
    }
    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        w: usize,
        h: usize,
        out: &mut Vec<f32>,
    ) -> Result<f64, String> {
        self.lend(
            [r, g, b].map(|p| imgref::Img::new_stride(p, w, h, stride)),
            &enough::Unstoppable,
            &mut |map| {
                out.clear();
                for row in map.rows() {
                    out.extend_from_slice(row)
                }
            },
        )
    }
    fn compare_with_map(
        &mut self,
        p: [imgref::ImgRef<'_, f32>; 3],
        stop: &dyn enough::Stop,
        consume: &mut dyn FnMut(imgref::ImgRef<'_, f32>),
    ) -> Option<Result<f64, String>> {
        Some(self.lend(p, stop, consume))
    }
}
