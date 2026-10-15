// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Opt-in per-iteration record of the perceptual quantization loop, for
//! viewers that step through how the metric's map drives the quant field.
//!
//! Attach a [`LoopTrace`] with [`crate::EncodeRequest::with_loop_trace`].
//! Each loop iteration that scores a reconstruction appends one
//! [`LoopFrame`]; without a trace the loop does no extra work.

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;

/// One scored iteration of the perceptual quantization loop.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LoopFrame {
    /// The seed's `k_init_mul` (the loop may run several seeds and keep the
    /// best; frames of one seed share this value).
    pub seed: f64,
    /// Iteration index within the seed, from 0. The last iteration of a
    /// seed only scores; it has no [`Self::next_quant_field`].
    pub iter: u32,
    /// The metric that scored it: `"butteraugli"`, `"cvvdp"` or `"zensim"`.
    pub metric: &'static str,
    /// The reconstruction's score in the metric's units.
    pub score: f64,
    /// The per-block distance the loop steers toward, in the same units as
    /// [`Self::tile_dist`].
    pub target: f64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Blocks per row (8×8 pixels each).
    pub xsize_blocks: u32,
    /// Block rows.
    pub ysize_blocks: u32,
    /// The float quant field this iteration encoded with, one value per
    /// block, row-major (higher = finer quantization).
    pub quant_field: Vec<f32>,
    /// The quant field after this iteration's adjustment: what the next
    /// iteration encodes with. `None` on a seed's final, score-only pass.
    pub next_quant_field: Option<Vec<f32>>,
    /// Per-block distance from the metric's diffmap (the 16-norm of each
    /// block's values times the metric's tile constant), row-major; blocks
    /// inside a larger transform carry their first block's value. Blocks
    /// above [`Self::target`] get finer quantization next iteration.
    pub tile_dist: Vec<f32>,
    /// The metric's per-pixel diffmap of this reconstruction,
    /// `width × height`, row-major.
    pub diffmap: Vec<f32>,
    /// The reconstruction as linear-RGB planes `[R, G, B]`, each
    /// `width × height`, row-major.
    pub recon: [Vec<f32>; 3],
}

/// A shared, cloneable collector of [`LoopFrame`]s.
#[derive(Clone, Debug, Default)]
pub struct LoopTrace {
    frames: Arc<Mutex<Vec<LoopFrame>>>,
}

impl LoopTrace {
    /// An empty trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The frames recorded so far, in loop order.
    #[must_use]
    pub fn frames(&self) -> Vec<LoopFrame> {
        self.frames
            .lock()
            .map(|f| f.clone())
            .unwrap_or_else(|e| e.into_inner().clone())
    }

    /// Removes and returns the recorded frames.
    pub fn take(&self) -> Vec<LoopFrame> {
        match self.frames.lock() {
            Ok(mut f) => core::mem::take(&mut *f),
            Err(e) => core::mem::take(&mut *e.into_inner()),
        }
    }

    pub(crate) fn push(&self, frame: LoopFrame) {
        match self.frames.lock() {
            Ok(mut f) => f.push(frame),
            Err(e) => e.into_inner().push(frame),
        }
    }

    /// Sets the last frame's [`LoopFrame::next_quant_field`].
    pub(crate) fn set_next_quant_field(&self, qf: &[f32]) {
        let mut guard = match self.frames.lock() {
            Ok(f) => f,
            Err(e) => e.into_inner(),
        };
        if let Some(last) = guard.last_mut() {
            last.next_quant_field = Some(qf.to_vec());
        }
    }
}
