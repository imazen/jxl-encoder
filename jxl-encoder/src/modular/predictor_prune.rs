// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Predictor-cost lower bound for early-skip at the root of the modular
//! tree-learning predictor selection (issue #23, chunk 1 of multi-day port).
//!
//! ## What this is
//!
//! `find_best_predictor` (the root call site in
//! `modular/tree_learn.rs::compute_best_tree_with_budget`) evaluates 14
//! candidate predictors against the full sample range and picks the one with
//! the lowest total estimated bits. Each evaluation builds a per-predictor
//! histogram via [`crate::modular::tree_learn::compute_predictor_entropy`] —
//! that is, the cost of the full residual symbols, weighted by sample counts,
//! plus an extra-bits term for the HybridUint nbits portion.
//!
//! For a non-trivial fraction of the 14 candidates, the extra-bits term
//! alone is already worse than the best total cost seen so far — in which
//! case the histogram build + entropy estimate is pure waste.
//!
//! This module provides a **provably sound** lower-bound primitive that
//! callers can use to early-skip a predictor before paying for the full
//! evaluation. The bound is half the work of the full pass (one slice
//! traversal instead of two) and never produces a false skip.
//!
//! ## Why the lower bound is sound
//!
//! The cost computed by `compute_predictor_entropy` is
//!
//! ```text
//!   total = estimate_bits(counts, total_count) + extra_bits_term
//! ```
//!
//! where:
//! - `estimate_bits` is libjxl's `EstimateBits` (`enc_ma.cc:54-71`) —
//!   `-Σ count[i] * log2(max(count[i]/total, 1/ANS_TAB_SIZE))`. Every term
//!   is a non-negative product (`count[i] >= 0` and
//!   `log2(prob)` is negated then negated again here for nonneg-ness because
//!   `prob ∈ (0, 1]` ⇒ `log2(prob) ≤ 0`). Therefore `estimate_bits ≥ 0`
//!   for any non-degenerate histogram, and identically zero in the
//!   `total == 0` degenerate case.
//! - `extra_bits_term = Σ (extra_bits[i] * sample_counts[i])`, both
//!   non-negative `u8` × `u32` products.
//!
//! Therefore `total ≥ extra_bits_term`. If `extra_bits_term ≥ best_so_far`,
//! the full evaluation cannot improve on `best_so_far` and the predictor
//! can be skipped without inspecting its histogram. This matches the
//! "TryPredictor"-style lower-bound discipline libjxl uses inside
//! `FindBestSplit` — strict-`<` semantics preserve byte-identical
//! tie-break behavior with the unconditional evaluation path.
//!
//! ## What this chunk ships (chunk 1)
//!
//! 1. `predictor_extra_bits_lower_bound` — the pure cost-lower-bound helper.
//! 2. `decide_predictor` — the skip/evaluate decision wrapper.
//! 3. Unit tests covering 5+ sample distributions.
//! 4. Standalone microbench (see `benches/predictor_prune_microbench.rs`).
//!
//! **Not wired into production yet** — chunk 2 will integrate this into
//! `find_best_predictor` after the primitive is proven on the microbench.
//!
//! ## libjxl reference
//!
//! - `lib/jxl/modular/encoding/enc_ma.cc:54-71` — `EstimateBits`. Verifies
//!   the entropy term is non-negative for `total > 0`.
//! - `lib/jxl/modular/encoding/enc_ma.cc:215-235` — the per-predictor
//!   histogram + `tot_extra_bits[pred]` pair from which the full cost is
//!   formed. The Rust equivalent lives in
//!   `modular/tree_learn.rs::compute_predictor_entropy`.

/// Decision returned by [`decide_predictor`].
///
/// `Skip` is only returned when the extra-bits lower bound for the candidate
/// is `>=` the current best cost. Because the lower bound is provably tight
/// (see module docs), `Skip` never hides a predictor that would have been
/// strictly better.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredictorDecision {
    /// The candidate cannot beat `best_cost`; histogram build can be elided.
    Skip,
    /// The candidate might beat `best_cost`; perform the full
    /// `compute_predictor_entropy` evaluation.
    EvaluateFully,
}

/// Compute the extra-bits lower bound on a predictor's total cost over a
/// contiguous sample range `[start..end)`.
///
/// Returns `Σ extra_bits[i] * sample_counts[i]` for `i in start..end`,
/// where `extra_bits[i]` is the HybridUint `nbits` for sample `i` under
/// the given predictor (the SoA slot
/// `TreeSamples::extra_bits[predictor_idx]`).
///
/// The output is in **bits**, expressed as `f64` so the caller can compare
/// directly against `compute_predictor_entropy`'s `f64` return value
/// without an intermediate cast.
///
/// ## Sound lower bound
///
/// For the same `(samples, start, end, predictor_idx)` tuple,
/// `compute_predictor_entropy(...) >= predictor_extra_bits_lower_bound(...)`
/// always holds (see module docs for the proof). This invariant is
/// exercised by [`tests::lower_bound_never_exceeds_full_cost`].
///
/// ## Cost
///
/// Single linear scan over two contiguous `u8`/`u32` slices, no histogram
/// allocation. Roughly half the bytes touched by
/// `compute_predictor_entropy` (which additionally builds the histogram
/// and runs `estimate_bits_u32`).
#[inline]
pub(crate) fn predictor_extra_bits_lower_bound(
    tokens: &[u8],
    ebits_for_token: &[u8; 256],
    sample_counts: &[u32],
    start: usize,
    end: usize,
) -> f64 {
    debug_assert!(start <= end, "start must not exceed end");
    debug_assert!(
        end <= tokens.len(),
        "end out of range for tokens: end={}, len={}",
        end,
        tokens.len()
    );
    debug_assert!(
        end <= sample_counts.len(),
        "end out of range for sample_counts: end={}, len={}",
        end,
        sample_counts.len()
    );
    debug_assert_eq!(
        tokens.len(),
        sample_counts.len(),
        "tokens / sample_counts must be parallel SoA slices"
    );

    let tk = &tokens[start..end];
    let sc = &sample_counts[start..end];
    let mut acc: u64 = 0;
    // Zip-iterate so the bounds check is elided by the matched zip pair.
    // Per-sample extra bits are a pure function of the token
    // (GATHER_EBITS_LUT) — the dedicated column no longer exists.
    for (&t, &c) in tk.iter().zip(sc.iter()) {
        acc += ebits_for_token[t as usize] as u64 * c as u64;
    }
    acc as f64
}

/// Decide whether a predictor candidate can be skipped given its
/// extra-bits lower bound and the best total cost seen so far.
///
/// Returns [`PredictorDecision::Skip`] iff `extra_bits_lb >= best_cost`.
/// Strict inequality on the *evaluate* side preserves
/// `find_best_predictor`'s `<` tie-break (lowest-index wins on equal cost).
///
/// `f64::NAN` is treated as "evaluate" — it can occur only when
/// `best_cost` is still the sentinel `f64::MAX` or the lower bound
/// computation overflows (it cannot; `u64` accumulation is bounded by
/// `n * 255 * u32::MAX < 2^64` for any plausible `n`). Defensive
/// handling included for completeness.
#[inline]
pub(crate) fn decide_predictor(extra_bits_lb: f64, best_cost: f64) -> PredictorDecision {
    if extra_bits_lb.is_nan() || best_cost.is_nan() {
        return PredictorDecision::EvaluateFully;
    }
    if extra_bits_lb >= best_cost {
        PredictorDecision::Skip
    } else {
        PredictorDecision::EvaluateFully
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror of `compute_predictor_entropy`'s scalar logic, kept here so
    /// the lower-bound tests don't need to materialize a full `TreeSamples`.
    /// Operates on three parallel slices (tokens, extra_bits, counts) and
    /// uses the same `min_prob = 1/4096` formula as
    /// `jxl_simd::estimate_bits_scalar_f64`.
    fn full_entropy_reference(
        tokens: &[u8],
        ebits_lut: &[u8; 256],
        sample_counts: &[u32],
        histogram_size: usize,
    ) -> f64 {
        let mut counts = vec![0u32; histogram_size];
        let mut total: u32 = 0;
        let mut tot_extra: u64 = 0;
        for (&t, &c) in tokens.iter().zip(sample_counts.iter()) {
            let tok = t as usize;
            if tok < histogram_size {
                counts[tok] += c;
                total += c;
            }
            tot_extra += ebits_lut[tok] as u64 * c as u64;
        }
        if total == 0 {
            return tot_extra as f64;
        }
        let total_f = total as f64;
        let min_prob = 1.0 / 4096.0;
        let mut bits: f64 = 0.0;
        for &c in counts.iter() {
            if c > 0 {
                let p = (c as f64 / total_f).max(min_prob);
                bits -= c as f64 * p.log2();
            }
        }
        bits + tot_extra as f64
    }

    /// The canonical invariant the primitive must satisfy.
    fn assert_lb_sound(
        tokens: &[u8],
        ebits_lut: &[u8; 256],
        sample_counts: &[u32],
        histogram_size: usize,
    ) {
        let lb =
            predictor_extra_bits_lower_bound(tokens, ebits_lut, sample_counts, 0, tokens.len());
        let full = full_entropy_reference(tokens, ebits_lut, sample_counts, histogram_size);
        assert!(
            lb <= full + 1e-9,
            "lower bound {} exceeded full cost {} (delta {})",
            lb,
            full,
            full - lb,
        );
    }

    // ---------------------------------------------------------------------
    // Case 1: uniform residuals, all extra_bits = 0.
    // The histogram is one big spike at token 7. Estimate_bits ≈ 0 (single
    // symbol → prob 1 → bits 0). extra_bits term = 0. So full ≈ 0 and
    // lb = 0 — bound is tight in this degenerate case.
    // ---------------------------------------------------------------------
    #[test]
    fn lb_uniform_zero_extra() {
        let tokens = vec![7u8; 256];
        let lut = [0u8; 256];
        let sample_counts = vec![1u32; 256];
        assert_lb_sound(&tokens, &lut, &sample_counts, 32);

        let lb = predictor_extra_bits_lower_bound(&tokens, &lut, &sample_counts, 0, 256);
        assert_eq!(lb, 0.0, "all-zero extra_bits must yield lb=0");

        let full = full_entropy_reference(&tokens, &lut, &sample_counts, 32);
        // Single-symbol histogram: each term is `count * (-log2(1)) = 0`.
        assert!(
            full.abs() < 1e-6,
            "single-symbol histogram should have entropy 0, got {}",
            full
        );
    }

    // ---------------------------------------------------------------------
    // Case 2: skewed distribution with non-trivial extra_bits. The lb
    // must be strictly less than the full cost because the entropy term
    // is positive.
    // ---------------------------------------------------------------------
    #[test]
    fn lb_skewed_with_extra_bits() {
        // 100 samples: 80 at token 0 (eb=2), 15 at token 1 (eb=4), 5 at
        // token 2 (eb=6).
        let mut tokens = Vec::with_capacity(100);
        let mut lut = [0u8; 256];
        lut[0] = 2;
        lut[1] = 4;
        lut[2] = 6;
        for _ in 0..80 {
            tokens.push(0);
        }
        for _ in 0..15 {
            tokens.push(1);
        }
        for _ in 0..5 {
            tokens.push(2);
        }
        let sample_counts = vec![1u32; 100];
        assert_lb_sound(&tokens, &lut, &sample_counts, 16);

        let lb = predictor_extra_bits_lower_bound(&tokens, &lut, &sample_counts, 0, 100);
        let expected_lb = (80 * 2 + 15 * 4 + 5 * 6) as f64;
        assert_eq!(lb, expected_lb, "lb computation must match scalar sum");

        let full = full_entropy_reference(&tokens, &lut, &sample_counts, 16);
        // Skewed 80/15/5 histogram has positive entropy; full > lb.
        assert!(
            full > lb + 1.0,
            "skewed distribution should leave a real entropy gap; lb={}, full={}",
            lb,
            full
        );
    }

    // ---------------------------------------------------------------------
    // Case 3: empty range — start == end. Lower bound and full cost
    // must both be 0.0, and the primitive must not panic.
    // ---------------------------------------------------------------------
    #[test]
    fn lb_empty_range() {
        let tokens = vec![0u8, 1, 2, 3];
        let mut lut = [0u8; 256];
        lut[..4].copy_from_slice(&[1, 2, 3, 4]);
        let sample_counts = vec![1u32; 4];
        let lb = predictor_extra_bits_lower_bound(&tokens, &lut, &sample_counts, 2, 2);
        assert_eq!(lb, 0.0, "empty range must produce lb=0");

        // Also: a non-empty buffer with empty range still works.
        let lb2 = predictor_extra_bits_lower_bound(&tokens, &lut, &sample_counts, 0, 0);
        assert_eq!(lb2, 0.0);
    }

    // ---------------------------------------------------------------------
    // Case 4: partial range — confirms the start/end slicing is honored.
    // Computes lb over a sub-range and verifies it matches the manual sum.
    // Also confirms the invariant holds for the same sub-range against a
    // full-cost reference restricted to the same window.
    // ---------------------------------------------------------------------
    #[test]
    fn lb_partial_range() {
        // tokens, extra_bits, sample_counts of length 10. Pick window [3..8).
        let tokens: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut lut = [0u8; 256];
        for (t, e) in lut.iter_mut().enumerate().take(10) {
            *e = (10 * (t + 1)) as u8;
        }
        let sample_counts: Vec<u32> = vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

        // Window: i ∈ [3..8) => contributions 40*2 + 50*2 + 60*2 + 70*2 + 80*2 = 600.
        let lb = predictor_extra_bits_lower_bound(&tokens, &lut, &sample_counts, 3, 8);
        assert_eq!(lb, 600.0);

        // Sound bound on the same window: build a reference cost restricted
        // to that window and confirm lb <= full.
        let window_tokens = &tokens[3..8];
        let window_sc = &sample_counts[3..8];
        let full = full_entropy_reference(window_tokens, &lut, window_sc, 16);
        assert!(
            lb <= full + 1e-9,
            "partial-range lb violated soundness: lb={}, full={}",
            lb,
            full
        );
    }

    // ---------------------------------------------------------------------
    // Case 5: high sample_counts (post-dedup case). Ensures u32 counts
    // multiplied by u8 extra_bits stay sound in u64 accumulation and
    // the bound still holds.
    // ---------------------------------------------------------------------
    #[test]
    fn lb_high_sample_counts_post_dedup() {
        // After dedup, a few unique samples carry large counts.
        let tokens = vec![0u8, 1, 2, 3];
        let mut lut = [0u8; 256];
        lut[..4].copy_from_slice(&[5, 7, 11, 13]);
        let sample_counts = vec![1_000_000u32, 250_000, 100_000, 50_000];
        assert_lb_sound(&tokens, &lut, &sample_counts, 16);

        let lb = predictor_extra_bits_lower_bound(&tokens, &lut, &sample_counts, 0, 4);
        let expected = (5u64 * 1_000_000 + 7 * 250_000 + 11 * 100_000 + 13 * 50_000) as f64;
        assert_eq!(lb, expected, "high-count u64 accumulation");
    }

    // ---------------------------------------------------------------------
    // Case 6: pseudo-random samples — broader stress that the invariant
    // holds across many randomized inputs (deterministic seed for repro).
    // ---------------------------------------------------------------------
    #[test]
    fn lb_pseudo_random_stress() {
        let n = 1024;
        let mut tokens = Vec::with_capacity(n);
        let mut sample_counts = Vec::with_capacity(n);
        // splitmix64 with a fixed seed for determinism; per-TOKEN ebits
        // (the production shape — ebits are a pure function of the token).
        let mut lut = [0u8; 256];
        for (t, e) in lut.iter_mut().enumerate() {
            *e = ((t.wrapping_mul(37) >> 2) & 0x0f) as u8;
        }
        let mut state: u64 = 0xdeadbeefcafef00d;
        for _ in 0..n {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            tokens.push((z & 0x1f) as u8);
            sample_counts.push(((z >> 16) & 0x7) as u32 + 1);
        }
        assert_lb_sound(&tokens, &lut, &sample_counts, 64);
    }

    // ---------------------------------------------------------------------
    // Case 7: decision wrapper — verifies skip/evaluate boundary.
    // ---------------------------------------------------------------------
    #[test]
    fn decide_predictor_skip_when_lb_at_or_above_best() {
        // Strict-< tie-break: when lb == best, skip (cannot strictly beat).
        assert_eq!(decide_predictor(100.0, 100.0), PredictorDecision::Skip);
        assert_eq!(decide_predictor(101.0, 100.0), PredictorDecision::Skip);
    }

    #[test]
    fn decide_predictor_evaluate_when_lb_below_best() {
        assert_eq!(
            decide_predictor(99.0, 100.0),
            PredictorDecision::EvaluateFully,
        );
        // Sentinel best: anything beats f64::MAX.
        assert_eq!(
            decide_predictor(1e18, f64::MAX),
            PredictorDecision::EvaluateFully,
        );
    }

    #[test]
    fn decide_predictor_nan_safe() {
        assert_eq!(
            decide_predictor(f64::NAN, 100.0),
            PredictorDecision::EvaluateFully,
        );
        assert_eq!(
            decide_predictor(100.0, f64::NAN),
            PredictorDecision::EvaluateFully,
        );
    }
}
