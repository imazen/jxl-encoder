// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at
// https://www.imazen.io/pricing
//
// Microbench: predictor-pruning lower-bound primitive for the modular
// tree-learning root predictor selection (issue #23, chunk 1).
//
// Compares two strategies at real-photo sample counts (200K / 1.35M / 3.2M
// post-dedup samples, mirroring the bench grid used in
// `dedup_samples_strategies.rs`):
//
// 1. `unconditional` - the production path: call the full
//    `compute_predictor_entropy` for every one of 14 candidate predictors.
// 2. `with_lb_skip` - the chunk-1 primitive: compute the extra-bits
//    lower bound first; skip the full evaluation if it cannot beat the
//    running best cost.
//
// Reports prune rate (predictors skipped / 14) and wall-clock time. The
// primitive is also exercised under a debug-side parity check that
// confirms the chosen best predictor matches the unconditional path
// byte-for-byte (no false skips → byte-identical bitstream invariant).

use core::hint::black_box;

use zenbench::Throughput;
use zenbench::prelude::*;

/// 14 candidate predictors. Matches `CANDIDATE_PREDICTORS.len()` in
/// `jxl-encoder/src/modular/tree_learn.rs`.
const NUM_PREDICTORS: usize = 14;

#[derive(Clone)]
struct PredictorSamples {
    /// extra_bits[pred][sample] — `u8` HybridUint nbits.
    extra_bits: Vec<Vec<u8>>,
    /// residual_tokens[pred][sample] — `u8` HybridUint token.
    residual_tokens: Vec<Vec<u8>>,
    /// sample_counts[sample] — `u32` post-dedup count.
    sample_counts: Vec<u32>,
    histogram_size: usize,
}

/// Generate synthetic samples that mimic the per-predictor cost spread
/// observed on real photos at e7:
/// - Predictor 0 (Zero): expensive (high extra_bits + flat histogram)
/// - Predictor 1 (Gradient): mid-cost
/// - Predictors 2-5 (W/N/AvgWN/Select): mid-low cost
/// - Predictor 6 (Weighted): often best on photos
/// - Predictors 7-13: assorted, varying
///
/// The exact numeric layout doesn't matter for the prune-rate measurement;
/// what matters is that some predictors have lower bounds above the best
/// total cost, so the bench exercises a realistic skip ratio.
fn make_samples(n: usize) -> PredictorSamples {
    // Deterministic PRNG (splitmix64) for repeatability.
    let mut state: u64 = 0x9e3779b97f4a7c15 ^ (n as u64);

    let next = |state: &mut u64| -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };

    let mut residual_tokens: Vec<Vec<u8>> =
        (0..NUM_PREDICTORS).map(|_| Vec::with_capacity(n)).collect();
    let mut extra_bits: Vec<Vec<u8>> = (0..NUM_PREDICTORS).map(|_| Vec::with_capacity(n)).collect();
    let mut sample_counts: Vec<u32> = Vec::with_capacity(n);

    // Per-predictor cost profile. eb_mean (mean extra-bits per sample) governs
    // the lower bound; eb_mean = 0 means the lb is 0 and the predictor will
    // never be skippable. Higher eb_mean → easier to prune.
    //
    // Indices loosely follow `CANDIDATE_PREDICTORS` ordering — first is
    // Weighted (best on photos), then Gradient, then the rest. Make the
    // first few cheap and the tail expensive to exercise a realistic mix.
    let eb_mean: [u8; NUM_PREDICTORS] = [2, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12];
    // Token spread: how flat each predictor's histogram is. Larger tok_mod
    // → flatter histogram → higher entropy → less compressible.
    let tok_mod: [u8; NUM_PREDICTORS] = [4, 6, 8, 10, 10, 12, 12, 14, 16, 16, 20, 24, 28, 32];

    for _ in 0..n {
        let r = next(&mut state);
        // After-dedup count: clustered around 1-3.
        let count = ((r & 0x3) as u32) + 1;
        sample_counts.push(count);
        for pred in 0..NUM_PREDICTORS {
            let h = next(&mut state);
            // Token: pseudo-Gaussian via two coin flips would be ideal, but
            // a uniform-mod is fine for bench shape.
            let tok = (h as u8) % tok_mod[pred].max(1);
            residual_tokens[pred].push(tok);
            // extra_bits centered on eb_mean with small jitter.
            let jitter = ((h >> 32) as u8) & 0x3;
            let eb = eb_mean[pred].saturating_add(jitter).min(20);
            extra_bits[pred].push(eb);
        }
    }

    // Histogram size = max token + 1 across all predictors.
    let mut max_tok: u8 = 0;
    for v in residual_tokens.iter() {
        for &t in v.iter() {
            if t > max_tok {
                max_tok = t;
            }
        }
    }
    let histogram_size = max_tok as usize + 1;

    PredictorSamples {
        extra_bits,
        residual_tokens,
        sample_counts,
        histogram_size,
    }
}

/// Reference `compute_predictor_entropy` (scalar, no SIMD dispatch). Mirror
/// of `jxl-encoder/src/modular/tree_learn.rs::compute_predictor_entropy`
/// using `f64` accumulation and the `min_prob = 1/4096` floor.
#[inline]
fn compute_predictor_entropy_ref(
    tokens: &[u8],
    extra_bits: &[u8],
    sample_counts: &[u32],
    histogram_size: usize,
    counts_buf: &mut [u32],
) -> f64 {
    counts_buf[..histogram_size].fill(0);
    let mut total: u32 = 0;
    let mut tot_extra: u64 = 0;
    for ((&t, &eb), &c) in tokens
        .iter()
        .zip(extra_bits.iter())
        .zip(sample_counts.iter())
    {
        let tok = t as usize;
        if tok < histogram_size {
            counts_buf[tok] += c;
            total += c;
        }
        tot_extra += eb as u64 * c as u64;
    }
    if total == 0 {
        return tot_extra as f64;
    }
    let total_f = total as f64;
    let min_prob = 1.0 / 4096.0;
    let mut bits: f64 = 0.0;
    for &c in &counts_buf[..histogram_size] {
        if c > 0 {
            let p = (c as f64 / total_f).max(min_prob);
            bits -= c as f64 * p.log2();
        }
    }
    bits + tot_extra as f64
}

/// Production-shape baseline: evaluate every predictor fully, pick min.
#[inline(never)]
fn find_best_unconditional(samples: &PredictorSamples) -> (usize, f64) {
    let mut counts_buf = vec![0u32; samples.histogram_size];
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;
    for pred in 0..NUM_PREDICTORS {
        let bits = compute_predictor_entropy_ref(
            &samples.residual_tokens[pred],
            &samples.extra_bits[pred],
            &samples.sample_counts,
            samples.histogram_size,
            &mut counts_buf,
        );
        if bits < best_bits {
            best_bits = bits;
            best_pred = pred;
        }
    }
    (best_pred, best_bits)
}

/// Chunk-1 candidate: extra-bits lower bound, skip fully-dominated predictors.
/// Returns `(best_pred, best_bits, num_skipped)`.
#[inline(never)]
fn find_best_with_lb_skip(samples: &PredictorSamples) -> (usize, f64, usize) {
    let mut counts_buf = vec![0u32; samples.histogram_size];
    let mut best_pred = 0;
    let mut best_bits = f64::MAX;
    let mut num_skipped = 0usize;
    for pred in 0..NUM_PREDICTORS {
        // Cheap lower bound. Identical formula to
        // `modular::predictor_prune::predictor_extra_bits_lower_bound`.
        let eb = &samples.extra_bits[pred];
        let sc = &samples.sample_counts;
        let mut lb_acc: u64 = 0;
        for (&e, &c) in eb.iter().zip(sc.iter()) {
            lb_acc += e as u64 * c as u64;
        }
        let lb = lb_acc as f64;
        // Strict-< tie-break: skip when lb >= best.
        if lb >= best_bits {
            num_skipped += 1;
            continue;
        }
        let bits = compute_predictor_entropy_ref(
            &samples.residual_tokens[pred],
            &samples.extra_bits[pred],
            &samples.sample_counts,
            samples.histogram_size,
            &mut counts_buf,
        );
        if bits < best_bits {
            best_bits = bits;
            best_pred = pred;
        }
    }
    (best_pred, best_bits, num_skipped)
}

/// One-shot parity + prune-rate report. Runs both strategies on each
/// sample count and writes a human-readable summary to stderr (zenbench
/// will surface it). Catches silent bugs (e.g. wrong best predictor)
/// before microbench numbers are reported.
fn print_parity_and_prune_stats() {
    eprintln!("--- predictor-prune parity + prune rate ---");
    for &n in &[200_000usize, 1_350_000, 3_200_000] {
        let samples = make_samples(n);
        let (b1, c1) = find_best_unconditional(&samples);
        let (b2, c2, skipped) = find_best_with_lb_skip(&samples);
        let prune_rate = (skipped as f64) / (NUM_PREDICTORS as f64) * 100.0;
        assert_eq!(b1, b2, "lb-skip chose a different predictor: bug in lb");
        // Relative cost tolerance: f64 accumulation order is identical,
        // so equality must hold.
        assert_eq!(
            c1.to_bits(),
            c2.to_bits(),
            "best-cost bits differ between unconditional and lb-skip",
        );
        eprintln!(
            "  n={:>9}: best_pred={:>2}, best_cost={:>14.1}, \
             skipped {}/{} predictors ({:.1}%)",
            n, b1, c1, skipped, NUM_PREDICTORS, prune_rate,
        );
    }
    eprintln!("--- end parity report ---");
}

fn bench_for_count<const N: usize>(suite: &mut Suite, label: &str) {
    let group_name = format!("find_best_pred_{label}");
    suite.group(&group_name, |g| {
        // Throughput = predictors-considered per call (14 candidates × N samples
        // is the work upper-bound; we report per-element for readability).
        g.throughput(Throughput::Elements(N as u64 * NUM_PREDICTORS as u64));

        g.bench("unconditional", |b| {
            b.with_input(|| make_samples(N)).run(|samples| {
                let out = find_best_unconditional(&samples);
                black_box(out);
                samples
            })
        });

        g.bench("with_lb_skip", |b| {
            b.with_input(|| make_samples(N)).run(|samples| {
                let out = find_best_with_lb_skip(&samples);
                black_box(out);
                samples
            })
        });

        g.baseline("unconditional");
        g.config().sort_by_speed(true);
    });
}

fn bench_predictor_prune(suite: &mut Suite) {
    // Print parity + prune-rate diagnostic before benches so it appears in
    // the captured log alongside zenbench's runtime numbers.
    print_parity_and_prune_stats();

    // Real-photo-scale sample counts (mirror dedup_samples_strategies.rs):
    //   0.26 MP -> ~200K samples
    //   1.05 MP -> ~1.35M samples
    //   4.19 MP -> ~3.2M samples
    bench_for_count::<200_000>(suite, "n=200K");
    bench_for_count::<1_350_000>(suite, "n=1.35M");
    bench_for_count::<3_200_000>(suite, "n=3.2M");
}

zenbench::main!(bench_predictor_prune);
