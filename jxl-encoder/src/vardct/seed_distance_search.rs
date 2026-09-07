//! Bounded global scale search after a lifted quant-field seed converges.
//!
//! Larger quant-field values mean finer quantization. Bracket the requested
//! score by halving the scale, then bisect geometrically. The 10% stopping
//! band is an accuracy tolerance, not a content-fitted calibration.

pub(super) const MAX_STEPS: usize = 8;

pub(super) struct SeedDistanceSearch {
    target: f64,
    fine: Option<f32>,
    coarse: Option<f32>,
    pub(super) scale: f32,
    pub(super) best_scale: f32,
    pub(super) best_score: f64,
}

impl SeedDistanceSearch {
    pub(super) fn new(target: f64) -> Self {
        Self {
            target,
            fine: None,
            coarse: None,
            scale: 1.0,
            best_scale: 1.0,
            best_score: f64::INFINITY,
        }
    }

    pub(super) fn observe(&mut self, score: f64) -> Option<f32> {
        assert!(score.is_finite() && score >= 0.0);
        if (score - self.target).abs() < (self.best_score - self.target).abs() {
            self.best_scale = self.scale;
            self.best_score = score;
        }
        if score >= self.target && score <= self.target * 1.1 {
            self.best_scale = self.scale;
            self.best_score = score;
            return None;
        }
        if score < self.target {
            self.fine = Some(self.scale);
        } else {
            self.coarse = Some(self.scale);
        }
        self.scale = match (self.fine, self.coarse) {
            (Some(fine), Some(coarse)) => (fine * coarse).sqrt(),
            (Some(fine), None) => fine * 0.5,
            (None, Some(coarse)) => coarse * 2.0,
            (None, None) => unreachable!(),
        };
        Some(self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brackets_and_converges_from_both_sides() {
        for initial_loss in [0.1_f64, 0.5, 0.9, 1.05, 1.5, 3.0] {
            let mut search = SeedDistanceSearch::new(1.0);
            for _ in 0..=MAX_STEPS {
                let score = initial_loss / f64::from(search.scale);
                if search.observe(score).is_none() {
                    break;
                }
            }
            assert!((1.0..=1.1).contains(&search.best_score), "{initial_loss}");
            assert_eq!(
                search.best_score,
                initial_loss / f64::from(search.best_scale)
            );
        }
    }

    #[test]
    fn unreachable_plateau_retains_a_measured_candidate() {
        let mut search = SeedDistanceSearch::new(2.0);
        for _ in 0..=MAX_STEPS {
            assert!(search.observe(0.5).is_some());
        }
        assert_eq!(search.best_scale, 1.0);
        assert_eq!(search.best_score, 0.5);
    }
}
