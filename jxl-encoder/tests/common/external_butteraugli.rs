use jxl_encoder::ExternalLoopMetric;
use std::sync::{Arc, Mutex};

/// Butteraugli through the external interface, with the parameters the
/// built-in CPU backend uses for SDR input.
pub(super) struct Butteraugli {
    pub(super) reference: Option<butteraugli::ButteraugliReference>,
    pub(super) calls: Arc<Mutex<u32>>,
}

impl ExternalLoopMetric for Butteraugli {
    fn set_reference(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        let params = butteraugli::ButteraugliParams::new()
            .with_intensity_target(80.0)
            .with_compute_diffmap(true);
        self.reference = Some(
            butteraugli::ButteraugliReference::new_linear_planar(
                r, g, b, width, height, width, params,
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(())
    }

    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        _width: usize,
        _height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String> {
        *self.calls.lock().unwrap() += 1;
        let reference = self.reference.as_ref().ok_or("no reference")?;
        reference
            .compare_linear_planar_into(r, g, b, stride, diffmap)
            .map(|(score, _)| score)
            .map_err(|e| e.to_string())
    }
}
