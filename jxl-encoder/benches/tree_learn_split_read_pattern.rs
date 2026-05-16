// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at
// https://www.imazen.io/pricing
//
// Microbench: quantifies the read-pattern delta between today's indexed
// access (find_best_split `collect` step at jxl-encoder/src/modular/
// tree_learn.rs:1951-1967) and the contiguous-read pattern enabled by the
// chunk-1 SplitTreeSamples primitive (`split_tree_samples_in_place` in
// jxl-encoder/src/modular/tree_learn_split.rs).
//
// Issue #40 thread (2026-05-15) instrumented `find_best_split` and showed
// the `collect` step at 51.9% of wall-clock for a 1024² photo at e7. The
// fix path (per libjxl's enc_ma.cc:119-138 `SplitTreeSamples<S>`) is to
// physically permute the SoA arrays so [begin..end) is contiguous in every
// underlying buffer. This bench measures whether the read-pattern speedup
// from contiguous-vs-random access is actually present at the sample
// scales the production code hits.

use core::hint::black_box;

use jxl_encoder::__bench_internals::tree_learn_split::{
    PartitionKey, SplittableSamples, split_tree_samples_in_place,
};
use zenbench::Throughput;
use zenbench::prelude::*;

/// Number of predictors in the production TreeSamples for non-squeeze mode.
const NUM_PREDICTORS: usize = 14;
/// Number of base properties in the production TreeSamples.
const NUM_PROPERTIES: usize = 16;

// Sample-count sweep (driven through bench_for_count::<N> below):
// - 10_000: small-photo node (e.g., a deep tree leaf on a 512² photo).
// - 100_000: representative node on a 1024² photo at the root or upper levels.
// - 1_000_000: root node of a 4096² photo (post-dedup).

/// Build a synthetic sample set with `n` rows.
///
/// All SoA arrays are filled with deterministic values so the bench is
/// reproducible. The property column we partition by is filled with
/// pseudo-random values in `[0, 255]`, half landing on each side of the
/// chosen split value.
fn make_samples(n: usize) -> Storage {
    let residual_tokens: Vec<Vec<u8>> = (0..NUM_PREDICTORS)
        .map(|p| (0..n).map(|i| ((i + p) & 0xff) as u8).collect())
        .collect();
    let extra_bits: Vec<Vec<u8>> = (0..NUM_PREDICTORS)
        .map(|p| {
            (0..n)
                .map(|i: usize| ((i.wrapping_mul(13).wrapping_add(p)) & 0xff) as u8)
                .collect()
        })
        .collect();
    let mut props: Vec<Vec<i32>> = (0..NUM_PROPERTIES)
        .map(|p| (0..n).map(|i| ((i + p) & 0xff) as i32).collect())
        .collect();
    let bucket_indices: Vec<Vec<u8>> = (0..NUM_PROPERTIES)
        .map(|p| {
            (0..n)
                .map(|i: usize| ((i.wrapping_mul(31).wrapping_add(p)) & 0xff) as u8)
                .collect()
        })
        .collect();
    let sample_counts: Vec<u32> = (0..n).map(|i| (i as u32 & 0xfff) + 1).collect();

    // Override props[0] with pseudo-random values for the partition column.
    let mut state: u32 = 0x12345678;
    for v in props[0].iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((state >> 24) & 0xff) as i32;
    }

    // Build an `indices` permutation that simulates the post-partition state
    // of today's find_best_split: a left/right partition where the left
    // entries' source-row offsets are pseudo-random in the underlying SoA.
    let mut indices: Vec<usize> = (0..n).collect();
    // Lehmer shuffle for deterministic but random-feeling order.
    let mut s: u32 = 0xcafebabe;
    for i in (1..n).rev() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let j = (s as usize) % (i + 1);
        indices.swap(i, j);
    }

    // Pick the split value so roughly half the rows land on each side.
    let split_val: i32 = 127;
    let left_count = props[0].iter().filter(|&&v| v <= split_val).count();

    Storage {
        residual_tokens,
        extra_bits,
        props,
        bucket_indices,
        sample_counts,
        indices,
        split_val,
        left_count,
        n,
    }
}

struct Storage {
    residual_tokens: Vec<Vec<u8>>,
    extra_bits: Vec<Vec<u8>>,
    props: Vec<Vec<i32>>,
    bucket_indices: Vec<Vec<u8>>,
    sample_counts: Vec<u32>,
    /// Pseudo-random index permutation simulating today's `find_best_split`
    /// `sorted_by_bucket[start..end]` indices (which carry source-row offsets
    /// into the SoA — the source of the random-read cost).
    indices: Vec<usize>,
    /// Partition split value for the bench (props[0] <= split_val).
    split_val: i32,
    /// Number of rows where `props[0][i] <= split_val`.
    left_count: usize,
    /// Total rows.
    n: usize,
}

/// "collect"-shape workload simulating today's per-predictor inner loop in
/// `find_best_split`. Iterates `indices[..]` and reads `tokens[idx]`,
/// `ebits[idx]`, `sample_counts[idx]` for ONE predictor pair (representative
/// of one (predictor, property) iteration of the inner sweep).
///
/// Returns an accumulator black-box'd against DCE.
#[inline(always)]
fn collect_indexed(storage: &Storage) -> u64 {
    let tokens = &storage.residual_tokens[0];
    let ebits = &storage.extra_bits[0];
    let sc = &storage.sample_counts;
    let mut histo: [u64; 256] = [0; 256];
    let mut eb_sum: u64 = 0;
    // The "left side" of the partition (the collect step touches every row).
    for &idx in &storage.indices {
        let tok = tokens[idx] as usize;
        let count = sc[idx] as u64;
        histo[tok] = histo[tok].wrapping_add(count);
        eb_sum = eb_sum.wrapping_add(ebits[idx] as u64 * count);
    }
    let mut acc: u64 = eb_sum;
    for &v in &histo {
        acc = acc.wrapping_add(v);
    }
    acc
}

/// Same workload but with contiguous reads. Simulates the post-permutation
/// state: `indices` is no longer needed; we iterate rows in order.
#[inline(always)]
fn collect_contiguous(storage: &Storage) -> u64 {
    let tokens = &storage.residual_tokens[0];
    let ebits = &storage.extra_bits[0];
    let sc = &storage.sample_counts;
    let mut histo: [u64; 256] = [0; 256];
    let mut eb_sum: u64 = 0;
    for i in 0..storage.n {
        let tok = tokens[i] as usize;
        let count = sc[i] as u64;
        histo[tok] = histo[tok].wrapping_add(count);
        eb_sum = eb_sum.wrapping_add(ebits[i] as u64 * count);
    }
    let mut acc: u64 = eb_sum;
    for &v in &histo {
        acc = acc.wrapping_add(v);
    }
    acc
}

/// In-place partition cost: how long does the SplitTreeSamples primitive take
/// to permute all SoA arrays for one split? This is the per-node amortized
/// cost we pay to enable contiguous reads on every downstream (predictor ×
/// property) inner loop within that node.
fn partition_in_place(storage: &mut Storage) {
    let pos = storage.left_count;
    let key = PartitionKey::Property {
        prop_idx: 0,
        val: storage.split_val,
    };
    let mut view = SplittableSamples {
        residual_tokens: &mut storage.residual_tokens,
        extra_bits: &mut storage.extra_bits,
        props: &mut storage.props,
        bucket_indices: &mut storage.bucket_indices,
        sample_counts: &mut storage.sample_counts,
        len: storage.n,
    };
    let _ = split_tree_samples_in_place(&mut view, 0, pos, storage.n, key);
}

fn bench_for_count<const N: usize>(suite: &mut Suite, label: &str) {
    let group_name = format!("collect_{label}");
    suite.group(&group_name, |g| {
        g.throughput(Throughput::Elements(N as u64));

        g.bench("indexed_random_reads", |b| {
            b.with_input(|| make_samples(N)).run(|storage| {
                let acc = collect_indexed(&storage);
                black_box(acc);
                storage
            })
        });

        g.bench("contiguous_reads", |b| {
            b.with_input(|| make_samples(N)).run(|storage| {
                let acc = collect_contiguous(&storage);
                black_box(acc);
                storage
            })
        });

        g.bench("partition_in_place_then_contiguous_reads", |b| {
            b.with_input(|| make_samples(N)).run(|mut storage| {
                partition_in_place(&mut storage);
                let acc = collect_contiguous(&storage);
                black_box(acc);
                storage
            })
        });

        g.bench("partition_in_place_only", |b| {
            b.with_input(|| make_samples(N)).run(|mut storage| {
                partition_in_place(&mut storage);
                storage
            })
        });

        g.baseline("indexed_random_reads");
        g.config().sort_by_speed(true);
    });
}

fn bench_read_patterns(suite: &mut Suite) {
    bench_for_count::<10_000>(suite, "n=10K");
    bench_for_count::<100_000>(suite, "n=100K");
    bench_for_count::<1_000_000>(suite, "n=1M");
}

zenbench::main!(bench_read_patterns);
