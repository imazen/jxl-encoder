// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
    /// Average of average and gradient.
    Average1 = 10,
    /// Average of average and left.
    Average2 = 11,
    /// Average of average and top.
    Average3 = 12,
    /// Average of top and top-right.
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
                // Median-like selection
                if n.w.abs_diff(n.nw) < n.n.abs_diff(n.nw) {
                    n.n
                } else {
                    n.w
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
            Predictor::Average1 => {
                let avg = (n.w + n.n) / 2;
                let grad = n.w.saturating_add(n.n).saturating_sub(n.nw);
                (avg + grad) / 2
            }
            Predictor::Average2 => {
                let avg = (n.w + n.n) / 2;
                (avg + n.w) / 2
            }
            Predictor::Average3 => {
                let avg = (n.w + n.n) / 2;
                (avg + n.n) / 2
            }
            Predictor::Average4 => (n.n + n.ne) / 2,
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
}

impl Neighbors {
    /// Gathers neighbor values from a channel.
    #[inline]
    pub fn gather(channel: &Channel, x: usize, y: usize) -> Self {
        let x = x as isize;
        let y = y as isize;

        Self {
            n: channel.get_clamped(x, y - 1),
            w: channel.get_clamped(x - 1, y),
            nw: channel.get_clamped(x - 1, y - 1),
            ne: channel.get_clamped(x + 1, y - 1),
            nn: channel.get_clamped(x, y - 2),
            ww: channel.get_clamped(x - 2, y),
        }
    }

    /// Gathers neighbors with explicit row pointers for speed.
    /// Assumes y >= 1 for valid top row access.
    #[inline]
    pub fn gather_fast(
        row: &[i32],
        prev_row: Option<&[i32]>,
        prev_prev_row: Option<&[i32]>,
        x: usize,
        width: usize,
    ) -> Self {
        let w = if x > 0 { row[x - 1] } else { 0 };
        let ww = if x > 1 { row[x - 2] } else { 0 };

        let (n, nw, ne, nn) = if let Some(prev) = prev_row {
            let n = prev[x];
            let nw = if x > 0 { prev[x - 1] } else { 0 };
            let ne = if x + 1 < width { prev[x + 1] } else { 0 };
            let nn = prev_prev_row.map_or(0, |pp| pp[x]);
            (n, nw, ne, nn)
        } else {
            (0, 0, 0, 0)
        };

        Self {
            n,
            w,
            nw,
            ne,
            nn,
            ww,
        }
    }
}

/// Weighted predictor state for adaptive prediction.
#[derive(Debug, Clone)]
pub struct WeightedPredictorState {
    /// Prediction errors for weight adaptation.
    pub errors: [i32; 4],
    /// Current weights.
    pub weights: [i32; 4],
    /// Weight update parameters.
    pub params: WeightedPredictorParams,
}

/// Parameters for the weighted predictor.
#[derive(Debug, Clone, Copy)]
pub struct WeightedPredictorParams {
    /// Weight for predictor 0 (N + W - NW).
    pub p0: u32,
    /// Weight for predictor 1 (N).
    pub p1: u32,
    /// Weight for predictor 2 (W).
    pub p2: u32,
    /// Weight for predictor 3 (NW).
    pub p3: u32,
    /// Weight for N-NN prediction.
    pub w0: u32,
    /// Weight for W-WW prediction.
    pub w1: u32,
    /// Weight for NE-N prediction.
    pub w2: u32,
    /// Weight for NW-N prediction.
    pub w3: u32,
}

impl Default for WeightedPredictorParams {
    fn default() -> Self {
        Self {
            p0: 16,
            p1: 10,
            p2: 10,
            p3: 4,
            w0: 0,
            w1: 0,
            w2: 0,
            w3: 0,
        }
    }
}

impl WeightedPredictorState {
    /// Creates a new weighted predictor state with default parameters.
    pub fn new() -> Self {
        Self::with_params(WeightedPredictorParams::default())
    }

    /// Creates a new weighted predictor state with custom parameters.
    pub fn with_params(params: WeightedPredictorParams) -> Self {
        Self {
            errors: [0; 4],
            weights: [
                params.p0 as i32,
                params.p1 as i32,
                params.p2 as i32,
                params.p3 as i32,
            ],
            params,
        }
    }

    /// Predicts using the weighted predictor.
    pub fn predict(&self, neighbors: &Neighbors) -> i32 {
        // Base predictions
        let p0 = neighbors.n + neighbors.w - neighbors.nw; // Gradient
        let p1 = neighbors.n; // Top
        let p2 = neighbors.w; // Left
        let p3 = neighbors.nw; // Top-left

        // Compute weighted average
        let total_weight = self.weights.iter().sum::<i32>().max(1);
        let sum = self.weights[0] * p0
            + self.weights[1] * p1
            + self.weights[2] * p2
            + self.weights[3] * p3;

        sum / total_weight
    }

    /// Updates weights based on prediction error.
    pub fn update(&mut self, neighbors: &Neighbors, actual: i32) {
        let p0 = neighbors.n + neighbors.w - neighbors.nw;
        let p1 = neighbors.n;
        let p2 = neighbors.w;
        let p3 = neighbors.nw;

        // Errors for each predictor
        let errs = [
            (actual - p0).abs(),
            (actual - p1).abs(),
            (actual - p2).abs(),
            (actual - p3).abs(),
        ];

        // Update running error estimates and adjust weights
        for (i, &err) in errs.iter().enumerate() {
            self.errors[i] = (self.errors[i] * 7 + err) / 8;
        }

        // Recompute weights inversely proportional to errors
        let min_err = self.errors.iter().min().copied().unwrap_or(1).max(1);
        for i in 0..4 {
            self.weights[i] = (min_err * 16) / self.errors[i].max(1);
        }
    }
}

impl Default for WeightedPredictorState {
    fn default() -> Self {
        Self::new()
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
}
