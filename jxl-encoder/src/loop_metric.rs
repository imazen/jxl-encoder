// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! A caller-supplied metric for the VarDCT perceptual quantization loop.
//!
//! **Unstable** (`__external_metric` feature): the trait and its contract may
//! change in any release.
//!
//! The loop normally scores each reconstruction with butteraugli. Attach an
//! [`ExternalMetric`] with [`crate::EncodeRequest::with_external_metric`] and
//! the loop scores with it instead, keeping everything else: the seeds, the
//! iteration count, the per-block reduction and the quant-field update.
//! Those were calibrated on butteraugli, so the metric's values must be on
//! butteraugli's scale: each 8×8 block's 16-norm, times 1.2, is compared with
//! the target distance, and the returned score (butteraugli returns its
//! diffmap's maximum) is compared with the target when the loop picks a seed.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;

/// A perceptual metric the quantization loop can score with.
pub trait ExternalLoopMetric: Send {
    /// Called once per encode, before any [`Self::compare`], with the source
    /// as planar linear RGB (`width × height` values per plane, row-major,
    /// nominally in `[0, 1]`).
    ///
    /// # Errors
    /// An error stops the loop; the encode keeps the initial quant field.
    fn set_reference(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
    ) -> Result<(), String>;

    /// Scores one reconstruction, planar linear RGB whose rows are `stride`
    /// values apart (`stride ≥ width`). Fills `diffmap` with `width × height`
    /// values, row-major, on butteraugli's scale, and returns the score.
    ///
    /// # Errors
    /// An error ends the current seed with the previous iteration's result,
    /// as a failed butteraugli comparison does.
    #[allow(clippy::too_many_arguments)]
    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        width: usize,
        height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String>;
}

/// A shared handle to an [`ExternalLoopMetric`]. Clones share one metric.
#[derive(Clone)]
pub struct ExternalMetric {
    inner: Arc<Mutex<Box<dyn ExternalLoopMetric>>>,
}

impl core::fmt::Debug for ExternalMetric {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExternalMetric").finish_non_exhaustive()
    }
}

impl ExternalMetric {
    /// Wraps `metric` for [`crate::EncodeRequest::with_external_metric`].
    pub fn new(metric: impl ExternalLoopMetric + 'static) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Box::new(metric))),
        }
    }

    pub(crate) fn with<T>(&self, f: impl FnOnce(&mut dyn ExternalLoopMetric) -> T) -> T {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        f(guard.as_mut())
    }
}
