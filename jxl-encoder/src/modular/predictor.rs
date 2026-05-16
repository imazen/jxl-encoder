// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Predictor implementations for modular encoding.
//!
//! Predictors estimate the value of a pixel based on its neighbors.
//! The prediction residual (actual - predicted) is what gets entropy coded.

use super::channel::Channel;

/// Available predictors for modular encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Predictor {
    /// Always predicts 0.
    #[default]
    Zero = 0,
    /// Uses the left neighbor (west).
    Left = 1,
    /// Uses the top neighbor (north).
    Top = 2,
    /// Average of left and top.
    Average0 = 3,
    /// Select between left/top based on top-left.
    Select = 4,
    /// Gradient: left + top - topleft (clamped).
    Gradient = 5,
    /// Weighted average favoring left.
    Weighted = 6,
    /// Top-right neighbor.
    TopRight = 7,
    /// Top-left neighbor.
    TopLeft = 8,
    /// Left-left neighbor (2 pixels left).
    LeftLeft = 9,
    /// Average of west and north-west: (W + NW) / 2
    Average1 = 10,
    /// Average of north and north-west: (N + NW) / 2
    Average2 = 11,
    /// Average of north and north-east: (N + NE) / 2
    Average3 = 12,
    /// Weighted average: (6N - 2NN + 7W + WW + NE2 + 3NE + 8) / 16
    Average4 = 13,
}

impl Predictor {
    /// Number of simple predictors (excluding weighted/variable).
    pub const NUM_SIMPLE: usize = 14;

    /// Returns all simple predictors.
    pub fn all_simple() -> &'static [Predictor] {
        &[
            Predictor::Zero,
            Predictor::Left,
            Predictor::Top,
            Predictor::Average0,
            Predictor::Select,
            Predictor::Gradient,
            Predictor::Weighted,
            Predictor::TopRight,
            Predictor::TopLeft,
            Predictor::LeftLeft,
            Predictor::Average1,
            Predictor::Average2,
            Predictor::Average3,
            Predictor::Average4,
        ]
    }

    /// Predicts the value at (x, y) using this predictor.
    #[inline]
    pub fn predict(self, channel: &Channel, x: usize, y: usize) -> i32 {
        let neighbors = Neighbors::gather(channel, x, y);
        self.predict_from_neighbors(&neighbors)
    }

    /// Predicts from pre-gathered neighbor values.
    #[inline]
    pub fn predict_from_neighbors(self, n: &Neighbors) -> i32 {
        match self {
            Predictor::Zero => 0,
            Predictor::Left => n.w,
            Predictor::Top => n.n,
            Predictor::Average0 => (n.w + n.n) / 2,
            Predictor::Select => {
                // Select predictor (matches JXL spec):
                // p = W + N - NW
                // if abs(p - W) < abs(p - N) then W else N
                // Since p - W = N - NW and p - N = W - NW:
                // if abs(N - NW) < abs(W - NW) then W else N
                if n.n.abs_diff(n.nw) < n.w.abs_diff(n.nw) {
                    n.w
                } else {
                    n.n
                }
            }
            Predictor::Gradient => {
                // Clamped gradient: W + N - NW, clamped to [min(W,N), max(W,N)]
                let gradient = n.w.saturating_add(n.n).saturating_sub(n.nw);
                gradient.clamp(n.w.min(n.n), n.w.max(n.n))
            }
            Predictor::Weighted => {
                // Simplified weighted predictor (full version uses adaptive weights)
                // This is a placeholder - full weighted uses WP state
                let gradient = n.w.saturating_add(n.n).saturating_sub(n.nw);
                gradient.clamp(n.w.min(n.n), n.w.max(n.n))
            }
            Predictor::TopRight => n.ne,
            Predictor::TopLeft => n.nw,
            Predictor::LeftLeft => n.ww,
            Predictor::Average1 => (n.w + n.nw) / 2,
            Predictor::Average2 => (n.n + n.nw) / 2,
            Predictor::Average3 => (n.n + n.ne) / 2,
            Predictor::Average4 => {
                // AverageAll: (6*N - 2*NN + 7*W + WW + NEE + 3*NE + 8) / 16
                // where NEE = toprightright = pixel at (x+2, y-1)
                // Use i64 intermediates to prevent overflow (libjxl PR #4574)
                ((6i64 * n.n as i64 - 2 * n.nn as i64
                    + 7 * n.w as i64
                    + n.ww as i64
                    + n.nee as i64
                    + 3 * n.ne as i64
                    + 8)
                    / 16) as i32
            }
        }
    }
}

/// Neighbor values for prediction.
#[derive(Debug, Clone, Copy, Default)]
pub struct Neighbors {
    /// North (top) neighbor.
    pub n: i32,
    /// West (left) neighbor.
    pub w: i32,
    /// Northwest (top-left) neighbor.
    pub nw: i32,
    /// Northeast (top-right) neighbor.
    pub ne: i32,
    /// North-north (2 pixels above) neighbor.
    pub nn: i32,
    /// West-west (2 pixels left) neighbor.
    pub ww: i32,
    /// Northeast-east (top-right of top-right, pixel at x+2, y-1). Used by AverageAll predictor.
    pub nee: i32,
}

impl Neighbors {
    /// Gathers neighbor values from a channel, matching the JXL spec's edge handling.
    ///
    /// Edge clamping rules (from jxl-rs PredictionData::get_rows):
    /// - left: `x>0 ? row[x-1] : (y>0 ? top_row[0] : 0)`
    /// - top: `y>0 ? top_row[x] : left`
    /// - topleft: `x>0 && y>0 ? top_row[x-1] : left`
    /// - topright: `x+1 < width && y>0 ? top_row[x+1] : top`
    /// - leftleft: `x>1 ? row[x-2] : left`
    /// - toptop: `y>1 ? toptop_row[x] : top`
    #[inline]
    pub fn gather(channel: &Channel, x: usize, y: usize) -> Self {
        let width = channel.width();

        let w = if x > 0 {
            channel.get(x - 1, y)
        } else if y > 0 {
            channel.get(0, y - 1)
        } else {
            0
        };

        let n = if y > 0 { channel.get(x, y - 1) } else { w };

        let nw = if x > 0 && y > 0 {
            channel.get(x - 1, y - 1)
        } else {
            w
        };

        let ne = if x + 1 < width && y > 0 {
            channel.get(x + 1, y - 1)
        } else {
            n
        };

        let ww = if x > 1 { channel.get(x - 2, y) } else { w };

        let nn = if y > 1 { channel.get(x, y - 2) } else { n };

        let nee = if x + 2 < width && y > 0 {
            channel.get(x + 2, y - 1)
        } else {
            ne
        };

        Self {
            n,
            w,
            nw,
            ne,
            nn,
            ww,
            nee,
        }
    }

    /// Gathers neighbors with explicit row pointers for speed, matching JXL spec edge handling.
    #[inline]
    pub fn gather_fast(
        row: &[i32],
        prev_row: Option<&[i32]>,
        prev_prev_row: Option<&[i32]>,
        x: usize,
        _width: usize,
    ) -> Self {
        let w = if x > 0 {
            row[x - 1]
        } else if let Some(prev) = prev_row {
            prev[0]
        } else {
            0
        };

        let n = if let Some(prev) = prev_row {
            prev[x]
        } else {
            w
        };

        let nw = if x > 0 {
            if let Some(prev) = prev_row {
                prev[x - 1]
            } else {
                w
            }
        } else {
            w
        };

        let ne = if let Some(prev) = prev_row {
            if x + 1 < prev.len() { prev[x + 1] } else { n }
        } else {
            n
        };

        let ww = if x > 1 { row[x - 2] } else { w };

        let nn = if let Some(pp) = prev_prev_row {
            pp[x]
        } else {
            n
        };

        let nee = if let Some(prev) = prev_row {
            if x + 2 < prev.len() { prev[x + 2] } else { ne }
        } else {
            ne
        };

        Self {
            n,
            w,
            nw,
            ne,
            nn,
            ww,
            nee,
        }
    }
}

/// Number of sub-predictors in weighted predictor.
const NUM_WP_PREDICTORS: usize = 4;
/// Extra precision bits for weighted predictor.
const PRED_EXTRA_BITS: i64 = 3;
/// Rounding value for weighted predictor.
const PREDICTION_ROUND: i64 = ((1 << PRED_EXTRA_BITS) >> 1) - 1;

/// Division lookup table for fast approximate division by 1-64.
/// `DIVLOOKUP[i] = (1 << 24) / (i + 1)`
const DIVLOOKUP: [u32; 64] = [
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721,
    1525201, 1398101, 1290555, 1198372, 1118481, 1048576, 986895, 932067, 883011, 838860, 798915,
    762600, 729444, 699050, 671088, 645277, 621378, 599186, 578524, 559240, 541200, 524288, 508400,
    493447, 479349, 466033, 453438, 441505, 430185, 419430, 409200, 399457, 390167, 381300, 372827,
    364722, 356962, 349525, 342392, 335544, 328965, 322638, 316551, 310689, 305040, 299593, 294337,
    289262, 284359, 279620, 275036, 270600, 266305, 262144,
];

/// Parameters for the weighted predictor (from bitstream header).
#[derive(Debug, Clone, Copy)]
pub struct WeightedPredictorParams {
    /// Correction parameter for predictor 1.
    pub p1c: u32,
    /// Correction parameter for predictor 2.
    pub p2c: u32,
    /// Correction parameters for predictor 3.
    pub p3ca: u32,
    pub p3cb: u32,
    pub p3cc: u32,
    pub p3cd: u32,
    pub p3ce: u32,
    /// Weight multipliers for error weighting.
    pub w0: u32,
    pub w1: u32,
    pub w2: u32,
    pub w3: u32,
}

impl Default for WeightedPredictorParams {
    fn default() -> Self {
        // Default values from JXL spec
        Self {
            p1c: 16,
            p2c: 10,
            p3ca: 7,
            p3cb: 7,
            p3cc: 7,
            p3cd: 0,
            p3ce: 0,
            w0: 0xd,
            w1: 0xc,
            w2: 0xc,
            w3: 0xc,
        }
    }
}

impl WeightedPredictorParams {
    /// Get weight multiplier by index.
    pub fn w(&self, i: usize) -> u32 {
        match i {
            0 => self.w0,
            1 => self.w1,
            2 => self.w2,
            3 => self.w3,
            _ => panic!("Invalid weight index"),
        }
    }

    /// Returns true if all parameters are at default values.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Get parameter set by mode index (0–4), matching libjxl's PredictorMode().
    ///
    /// - Mode 0: Default (lossless16)
    /// - Mode 1: lossless8 variant
    /// - Mode 2: West-biased lossless8
    /// - Mode 3: North-biased lossless8
    /// - Mode 4: Generic/balanced
    pub fn for_mode(mode: u8) -> Self {
        match mode {
            0 => Self::default(),
            1 => Self {
                p1c: 8,
                p2c: 8,
                p3ca: 4,
                p3cb: 0,
                p3cc: 3,
                p3cd: 23,
                p3ce: 2,
                w0: 0xd,
                w1: 0xc,
                w2: 0xc,
                w3: 0xb,
            },
            2 => Self {
                p1c: 10,
                p2c: 9,
                p3ca: 7,
                p3cb: 0,
                p3cc: 0,
                p3cd: 16,
                p3ce: 9,
                w0: 0xd,
                w1: 0xc,
                w2: 0xd,
                w3: 0xc,
            },
            3 => Self {
                p1c: 16,
                p2c: 8,
                p3ca: 0,
                p3cb: 16,
                p3cc: 0,
                p3cd: 23,
                p3ce: 0,
                w0: 0xd,
                w1: 0xd,
                w2: 0xc,
                w3: 0xc,
            },
            _ => Self {
                p1c: 10,
                p2c: 10,
                p3ca: 5,
                p3cb: 5,
                p3cc: 5,
                p3cd: 12,
                p3ce: 4,
                w0: 0xd,
                w1: 0xc,
                w2: 0xc,
                w3: 0xc,
            },
        }
    }
}

impl PartialEq for WeightedPredictorParams {
    fn eq(&self, other: &Self) -> bool {
        self.p1c == other.p1c
            && self.p2c == other.p2c
            && self.p3ca == other.p3ca
            && self.p3cb == other.p3cb
            && self.p3cc == other.p3cc
            && self.p3cd == other.p3cd
            && self.p3ce == other.p3ce
            && self.w0 == other.w0
            && self.w1 == other.w1
            && self.w2 == other.w2
            && self.w3 == other.w3
    }
}

/// Floor log2 for non-zero values.
#[inline]
fn floor_log2_nonzero(x: u64) -> u32 {
    63 - x.leading_zeros()
}

/// Add extra precision bits.
#[inline]
fn add_bits(x: i32) -> i64 {
    (x as i64) << PRED_EXTRA_BITS
}

/// Compute error weight from accumulated error.
#[inline]
fn error_weight(x: u32, maxweight: u32) -> u32 {
    let shift = floor_log2_nonzero(x as u64 + 1) as i32 - 5;
    if shift < 0 {
        4u32 + maxweight * DIVLOOKUP[x as usize & 63]
    } else {
        4u32 + ((maxweight * DIVLOOKUP[(x as usize >> shift) & 63]) >> shift)
    }
}

/// Compute weighted average of predictions.
fn weighted_average(
    pixels: &[i64; NUM_WP_PREDICTORS],
    weights: &mut [u32; NUM_WP_PREDICTORS],
) -> i64 {
    let log_weight = floor_log2_nonzero(weights.iter().fold(0u64, |sum, el| sum + *el as u64));
    let weight_sum = weights.iter_mut().fold(0, |sum, el| {
        *el >>= log_weight - 4;
        sum + *el
    });
    let sum = weights
        .iter()
        .enumerate()
        .fold(((weight_sum >> 1) - 1) as i64, |sum, (i, weight)| {
            sum + pixels[i] * *weight as i64
        });
    (sum * DIVLOOKUP[(weight_sum - 1) as usize] as i64) >> 24
}

/// Full weighted predictor state for adaptive prediction.
/// Matches libjxl/jxl-rs implementation for encoding parity.
#[derive(Debug)]
pub struct WeightedPredictorState {
    /// Current predictions from each sub-predictor.
    prediction: [i64; NUM_WP_PREDICTORS],
    /// Final weighted prediction.
    pred: i64,
    /// Per-position error buffer (position-major layout).
    /// Layout: [pos0: p0,p1,p2,p3] [pos1: p0,p1,p2,p3] ...
    pred_errors_buffer: Vec<u32>,
    /// Prediction errors per position.
    error: Vec<i32>,
    /// Weighted predictor parameters.
    params: WeightedPredictorParams,
}

impl WeightedPredictorState {
    /// Creates a new weighted predictor state.
    pub fn new(params: &WeightedPredictorParams, xsize: usize) -> Self {
        let num_errors = (xsize + 2) * 2;
        Self {
            prediction: [0; NUM_WP_PREDICTORS],
            pred: 0,
            pred_errors_buffer: vec![0; num_errors * NUM_WP_PREDICTORS],
            error: vec![0; num_errors],
            params: *params,
        }
    }

    /// Creates a new weighted predictor state with an optional allocation
    /// budget. The two scratch buffers (`pred_errors_buffer` of
    /// `(xsize + 2) * 2 * NUM_WP_PREDICTORS` u32 and `error` of
    /// `(xsize + 2) * 2` i32) are width-driven and therefore charged
    /// against the per-encode cap. Returns
    /// [`crate::error::Error::AllocationLimit`] when the cap is exceeded;
    /// callers convert to `EncodeError` upstream.
    ///
    /// `budget = None` is zero-overhead — equivalent to [`Self::new`].
    pub(crate) fn new_with_budget(
        params: &WeightedPredictorParams,
        xsize: usize,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> crate::error::Result<Self> {
        let num_errors = (xsize + 2) * 2;
        let buffer_bytes = (num_errors as u64)
            .checked_mul(NUM_WP_PREDICTORS as u64)
            .and_then(|n| n.checked_mul(core::mem::size_of::<u32>() as u64))
            .ok_or(crate::error::Error::AllocationLimit {
                requested: u64::MAX,
                used: 0,
                cap: 0,
            })?;
        let error_bytes = (num_errors as u64)
            .checked_mul(core::mem::size_of::<i32>() as u64)
            .ok_or(crate::error::Error::AllocationLimit {
                requested: u64::MAX,
                used: 0,
                cap: 0,
            })?;
        crate::budget::MemoryBudget::reserve_permanent_opt(budget, buffer_bytes)?;
        crate::budget::MemoryBudget::reserve_permanent_opt(budget, error_bytes)?;
        Ok(Self {
            prediction: [0; NUM_WP_PREDICTORS],
            pred: 0,
            pred_errors_buffer: vec![0; num_errors * NUM_WP_PREDICTORS],
            error: vec![0; num_errors],
            params: *params,
        })
    }

    /// Creates with default parameters.
    pub fn with_defaults(xsize: usize) -> Self {
        Self::new(&WeightedPredictorParams::default(), xsize)
    }

    /// Creates with default parameters and an optional allocation budget.
    /// Counterpart to [`Self::new_with_budget`].
    pub(crate) fn with_defaults_and_budget(
        xsize: usize,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> crate::error::Result<Self> {
        Self::new_with_budget(&WeightedPredictorParams::default(), xsize, budget)
    }

    /// Get all predictor errors for a given position (contiguous in memory).
    #[inline(always)]
    fn get_errors_at_pos(&self, pos: usize) -> &[u32; NUM_WP_PREDICTORS] {
        let start = pos * NUM_WP_PREDICTORS;
        self.pred_errors_buffer[start..start + NUM_WP_PREDICTORS]
            .try_into()
            .unwrap()
    }

    /// Get mutable reference to all predictor errors for a given position.
    #[inline(always)]
    fn get_errors_at_pos_mut(&mut self, pos: usize) -> &mut [u32; NUM_WP_PREDICTORS] {
        let start = pos * NUM_WP_PREDICTORS;
        (&mut self.pred_errors_buffer[start..start + NUM_WP_PREDICTORS])
            .try_into()
            .unwrap()
    }

    /// Compute prediction and property value.
    /// Returns (prediction, max_error_property).
    #[inline]
    pub fn predict_and_property(
        &mut self,
        x: usize,
        y: usize,
        xsize: usize,
        neighbors: &Neighbors,
    ) -> (i64, i32) {
        let (cur_row, prev_row) = if y & 1 != 0 {
            (0, xsize + 2)
        } else {
            (xsize + 2, 0)
        };
        let pos_n = prev_row + x;
        let pos_ne = if x < xsize - 1 { pos_n + 1 } else { pos_n };
        let pos_nw = if x > 0 { pos_n - 1 } else { pos_n };

        // Get errors at neighboring positions
        let errors_n = self.get_errors_at_pos(pos_n);
        let errors_ne = self.get_errors_at_pos(pos_ne);
        let errors_nw = self.get_errors_at_pos(pos_nw);

        // Compute weights from errors
        let mut weights = [0u32; NUM_WP_PREDICTORS];
        for i in 0..NUM_WP_PREDICTORS {
            weights[i] = error_weight(
                errors_n[i]
                    .wrapping_add(errors_ne[i])
                    .wrapping_add(errors_nw[i]),
                self.params.w(i),
            );
        }

        // Convert neighbors to higher precision
        let n = add_bits(neighbors.n);
        let w = add_bits(neighbors.w);
        let ne = add_bits(neighbors.ne);
        let nw = add_bits(neighbors.nw);
        let nn = add_bits(neighbors.nn);

        // Get transmission errors from neighboring positions
        let te_w = if x == 0 {
            0
        } else {
            self.error[cur_row + x - 1] as i64
        };
        let te_n = self.error[pos_n] as i64;
        let te_nw = self.error[pos_nw] as i64;
        let sum_wn = te_n + te_w;
        let te_ne = self.error[pos_ne] as i64;

        // Find max absolute error for property
        let mut p = te_w;
        if te_n.abs() > p.abs() {
            p = te_n;
        }
        if te_nw.abs() > p.abs() {
            p = te_nw;
        }
        if te_ne.abs() > p.abs() {
            p = te_ne;
        }

        // Compute 4 sub-predictions with corrections
        self.prediction[0] = w + ne - n;
        self.prediction[1] = n - (((sum_wn + te_ne) * self.params.p1c as i64) >> 5);
        self.prediction[2] = w - (((sum_wn + te_nw) * self.params.p2c as i64) >> 5);
        self.prediction[3] = n
            - ((te_nw * (self.params.p3ca as i64)
                + (te_n * (self.params.p3cb as i64))
                + (te_ne * (self.params.p3cc as i64))
                + ((nn - n) * (self.params.p3cd as i64))
                + ((nw - w) * (self.params.p3ce as i64)))
                >> 5);

        // Compute weighted average
        self.pred = weighted_average(&self.prediction, &mut weights);

        // Apply clamping when errors have consistent signs
        if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
            let mx = w.max(ne.max(n));
            let mn = w.min(ne.min(n));
            self.pred = mn.max(mx.min(self.pred));
        }

        ((self.pred + PREDICTION_ROUND) >> PRED_EXTRA_BITS, p as i32)
    }

    /// Update error buffers after seeing actual value.
    #[inline]
    pub fn update_errors(&mut self, actual: i32, x: usize, y: usize, xsize: usize) {
        let (cur_row, prev_row) = if y & 1 != 0 {
            (0, xsize + 2)
        } else {
            (xsize + 2, 0)
        };
        let val = add_bits(actual);
        self.error[cur_row + x] = (self.pred - val) as i32;

        // Compute errors for all predictors
        let mut errs = [0u32; NUM_WP_PREDICTORS];
        for (err, &pred) in errs.iter_mut().zip(self.prediction.iter()) {
            *err = (((pred - val).abs() + PREDICTION_ROUND) >> PRED_EXTRA_BITS) as u32;
        }

        // Write to current position
        *self.get_errors_at_pos_mut(cur_row + x) = errs;

        // Update previous row position
        let prev_errors = self.get_errors_at_pos_mut(prev_row + x + 1);
        for i in 0..NUM_WP_PREDICTORS {
            prev_errors[i] = prev_errors[i].wrapping_add(errs[i]);
        }
    }

    /// Simple predict method for compatibility.
    pub fn predict(&mut self, x: usize, y: usize, xsize: usize, neighbors: &Neighbors) -> i32 {
        let (pred, _) = self.predict_and_property(x, y, xsize, neighbors);
        pred as i32
    }
}

impl Default for WeightedPredictorState {
    fn default() -> Self {
        Self::with_defaults(256)
    }
}

/// Packs a signed integer for entropy coding.
/// Converts signed values to unsigned using zig-zag encoding.
#[inline]
pub fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) * 2
    } else {
        ((-value) as u32) * 2 - 1
    }
}

/// Unpacks a zig-zag encoded value back to signed.
#[inline]
pub fn unpack_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value / 2) as i32
    } else {
        -((value / 2) as i32) - 1
    }
}

/// Estimate the total encoding cost of using a WP parameter set on the given channels.
///
/// Runs the weighted predictor over every pixel, computes residuals, and
/// estimates Shannon entropy + HybridUint extra bits as a cost proxy.
/// Matching libjxl's EstimateWPCost (enc_modular.cc:238-287).
pub fn estimate_wp_cost(channels: &[super::Channel], params: &WeightedPredictorParams) -> f64 {
    // Use 256-bin histogram for entropy estimation
    const NUM_BINS: usize = 256;
    let mut histogram = [0u32; NUM_BINS];
    let mut total_extra_bits = 0u64;
    let mut total_samples = 0u64;

    for channel in channels {
        let width = channel.width();
        let height = channel.height();
        if width == 0 || height == 0 {
            continue;
        }

        let mut wp_state = WeightedPredictorState::new(params, width);

        for y in 0..height {
            for x in 0..width {
                let pixel = channel.get(x, y);
                let neighbors = Neighbors::gather(channel, x, y);
                let prediction = wp_state.predict(x, y, width, &neighbors);

                let residual = pixel - prediction;
                let packed = pack_signed(residual);

                // Bin the packed residual for histogram
                let bin = if packed < NUM_BINS as u32 {
                    packed as usize
                } else {
                    // For large residuals, count extra bits needed
                    let bits = 32 - packed.leading_zeros();
                    total_extra_bits += bits as u64;
                    NUM_BINS - 1
                };
                histogram[bin] += 1;
                total_samples += 1;

                wp_state.update_errors(pixel, x, y, width);
            }
        }
    }

    if total_samples == 0 {
        return 0.0;
    }

    // Estimate Shannon entropy from histogram
    let total_f = total_samples as f64;
    let mut entropy = 0.0f64;
    for &count in &histogram {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * jxl_simd::fast_log2f(p as f32) as f64;
        }
    }

    // Total cost = entropy bits + extra bits for large values
    entropy * total_f + total_extra_bits as f64
}

/// Find the best WP parameter set by trying `num_sets` modes (0..num_sets).
///
/// Returns the best `WeightedPredictorParams` and whether it differs from default.
/// At effort 8 (kKitten): `num_sets=2` (modes 0-1).
/// At effort 9+ (kTortoise): `num_sets=5` (modes 0-4).
///
/// Mode evaluations are independent and parallelized via [`crate::parallel::parallel_map`]
/// when the `parallel` feature is enabled. At e9 with 5 modes on a multicore
/// machine this is a 2-3× speedup of the WP search phase — meaningful because
/// the search is a non-trivial fraction (3-6%) of total lossless wall-clock
/// at higher efforts on photo content. Deterministic: on tie cost we keep the
/// lowest mode index, matching the prior `<` (strict) sequential tie-break.
pub fn find_best_wp_params(channels: &[super::Channel], num_sets: u8) -> WeightedPredictorParams {
    if num_sets <= 1 {
        return WeightedPredictorParams::default();
    }

    let max_mode = num_sets.min(5);

    // Evaluate each mode independently. Each call constructs its own
    // `WeightedPredictorState` per channel internally — no shared mutable
    // state, safe to fan out.
    let costs: Vec<f64> = crate::parallel::parallel_map(max_mode as usize, |mode_i| {
        let params = WeightedPredictorParams::for_mode(mode_i as u8);
        estimate_wp_cost(channels, &params)
    });

    // Pick the lowest-cost mode; on tie, keep the lowest index. Use a manual
    // scan with `<` to preserve the same tie-break as the previous serial
    // loop (so the bit-identical hash-lock holds).
    let mut best_cost = f64::MAX;
    let mut best_mode = 0u8;
    for (i, &cost) in costs.iter().enumerate() {
        if cost < best_cost {
            best_cost = cost;
            best_mode = i as u8;
        }
    }

    WeightedPredictorParams::for_mode(best_mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictors() {
        let mut channel = Channel::new(4, 4).unwrap();

        // Set up a simple gradient pattern
        for y in 0..4 {
            for x in 0..4 {
                channel.set(x, y, (x + y * 4) as i32);
            }
        }

        // Test at position (2, 2)
        // Pattern: 0  1  2  3
        //          4  5  6  7
        //          8  9 [10] 11
        //         12 13 14 15

        let neighbors = Neighbors::gather(&channel, 2, 2);
        assert_eq!(neighbors.n, 6); // Top
        assert_eq!(neighbors.w, 9); // Left
        assert_eq!(neighbors.nw, 5); // Top-left
        assert_eq!(neighbors.ne, 7); // Top-right

        // Test Zero predictor
        assert_eq!(Predictor::Zero.predict_from_neighbors(&neighbors), 0);

        // Test Left predictor
        assert_eq!(Predictor::Left.predict_from_neighbors(&neighbors), 9);

        // Test Top predictor
        assert_eq!(Predictor::Top.predict_from_neighbors(&neighbors), 6);

        // Test Gradient predictor: 9 + 6 - 5 = 10, clamped to [6, 9] = 9
        assert_eq!(Predictor::Gradient.predict_from_neighbors(&neighbors), 9);
    }

    #[test]
    fn test_pack_signed() {
        assert_eq!(pack_signed(0), 0);
        assert_eq!(pack_signed(1), 2);
        assert_eq!(pack_signed(-1), 1);
        assert_eq!(pack_signed(2), 4);
        assert_eq!(pack_signed(-2), 3);
    }

    #[test]
    fn test_unpack_signed() {
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(unpack_signed(4), 2);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        for i in -1000..=1000 {
            assert_eq!(unpack_signed(pack_signed(i)), i);
        }
    }

    #[test]
    fn test_weighted_predictor_params_default() {
        let params = WeightedPredictorParams::default();
        assert_eq!(params.p1c, 16);
        assert_eq!(params.p2c, 10);
        assert_eq!(params.w0, 0xd);
        assert!(params.is_default());
    }

    #[test]
    fn test_weighted_predictor_state() {
        let xsize = 8;
        let mut wp = WeightedPredictorState::with_defaults(xsize);

        // Test prediction on uniform data
        let neighbors = Neighbors {
            n: 100,
            w: 100,
            nw: 100,
            ne: 100,
            nn: 100,
            ww: 100,
            nee: 100,
        };

        let (pred, _prop) = wp.predict_and_property(4, 2, xsize, &neighbors);
        // For uniform data, prediction should be close to 100
        assert!((pred - 100).abs() <= 2);

        // Update with actual value
        wp.update_errors(100, 4, 2, xsize);
    }

    #[test]
    fn test_weighted_predictor_adapts() {
        let xsize = 8;
        let mut wp = WeightedPredictorState::with_defaults(xsize);

        // Simulate processing a row with gradient pattern
        for x in 0..xsize {
            let actual = (x * 10) as i32;
            let neighbors = Neighbors {
                n: if x > 0 { ((x - 1) * 10) as i32 } else { 0 },
                w: if x > 0 { ((x - 1) * 10) as i32 } else { 0 },
                nw: if x > 1 { ((x - 2) * 10) as i32 } else { 0 },
                ne: (x * 10) as i32,
                nn: 0,
                ww: if x > 1 { ((x - 2) * 10) as i32 } else { 0 },
                nee: 0,
            };

            let (_pred, _prop) = wp.predict_and_property(x, 1, xsize, &neighbors);
            wp.update_errors(actual, x, 1, xsize);
        }
        // Just verify it doesn't panic
    }

    /// Reproduce jxl-rs golden-number test to verify bit-exactness.
    #[test]
    fn test_wp_matches_jxl_rs_golden() {
        struct SimpleRandom {
            out: i64,
        }
        impl SimpleRandom {
            fn new() -> Self {
                Self { out: 1 }
            }
            fn next(&mut self) -> i64 {
                self.out = self.out * 48271 % 0x7fffffff;
                self.out
            }
        }

        let mut rng = SimpleRandom::new();
        let params = WeightedPredictorParams {
            p1c: rng.next() as u32 % 32,
            p2c: rng.next() as u32 % 32,
            p3ca: rng.next() as u32 % 32,
            p3cb: rng.next() as u32 % 32,
            p3cc: rng.next() as u32 % 32,
            p3cd: rng.next() as u32 % 32,
            p3ce: rng.next() as u32 % 32,
            w0: rng.next() as u32 % 16,
            w1: rng.next() as u32 % 16,
            w2: rng.next() as u32 % 16,
            w3: rng.next() as u32 % 16,
        };
        let xsize = 8;
        let ysize = 8;
        let mut state = WeightedPredictorState::new(&params, xsize);

        // Helper: one step of predict + update
        let step = |rng: &mut SimpleRandom, state: &mut WeightedPredictorState| -> (i64, i32) {
            let x = rng.next() as usize % xsize;
            let y = rng.next() as usize % ysize;
            let neighbors = Neighbors {
                n: rng.next() as i32 % 256,  // top
                w: rng.next() as i32 % 256,  // left
                ne: rng.next() as i32 % 256, // topright
                nw: rng.next() as i32 % 256, // topleft
                nn: rng.next() as i32 % 256, // toptop
                ww: 0,
                nee: 0,
            };
            let res = state.predict_and_property(x, y, xsize, &neighbors);
            state.update_errors((rng.next() % 256) as i32, x, y, xsize);
            res
        };

        // Golden numbers from libjxl (verified in jxl-rs test)
        assert_eq!(step(&mut rng, &mut state), (135, 0), "step 1");
        assert_eq!(step(&mut rng, &mut state), (110, -60), "step 2");
        assert_eq!(step(&mut rng, &mut state), (165, 0), "step 3");
        assert_eq!(step(&mut rng, &mut state), (153, -60), "step 4");
    }
}
