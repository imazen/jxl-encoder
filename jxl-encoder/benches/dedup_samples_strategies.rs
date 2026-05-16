// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at
// https://www.imazen.io/pricing
//
// Microbench: compares dedup_samples strategies for the tree-learning
// hot path (jxl-encoder/src/modular/tree_learn.rs:869-962).
//
// The current production code uses `sort_unstable_by` with a composite-key
// closure that walks 9 properties × Vec<Vec<u8>> + 14 predictors × 2
// Vec<Vec<u8>> indirections per comparison. The bench compares against:
//
// - `packed_key_sort`: materialize each sample's composite key into a
//   contiguous `Vec<[u8; K]>` once, sort indices by packed key. Cuts
//   per-cmp cache misses from ~42 to ~K/cacheline.
// - `hashmap_dedup`: libjxl-style hash-table dedup. Single pass over
//   samples, hash composite key, lookup-or-insert. O(n) expected.
//
// Output formats:
//   `current` - production dedup_samples
//   `packed_key_sort` - sort by packed composite-key bytes
//   `hashmap_dedup` - rustc_hash FxHashMap on composite key
//
// Real-photo-scale sample counts: 200K (~0.26 MP image), 1.35M (~1.05
// MP image), 3.2M (~4.19 MP image). These match the e7 numbers from the
// lossless_cliff_profile harness (commit pxnzysqk:e3cea6f0).

use core::hint::black_box;

use hashbrown::HashMap;
use zenbench::Throughput;
use zenbench::prelude::*;

/// Match production: 7 active properties (e7) + 2 ref-channel props.
const NUM_PROPERTIES_ACTIVE: usize = 9;
const NUM_PROPERTIES_TOTAL: usize = 16;
/// 14 candidate predictors (CANDIDATE_PREDICTORS).
const NUM_PREDICTORS: usize = 14;

/// Composite-key length: 9 prop buckets (u8) + 14 × (tok u8 + ebits u8) = 37 bytes.
/// Padded to 40 for alignment.
const PACKED_KEY_BYTES: usize = 40;

#[derive(Clone)]
struct Samples {
    /// residual_tokens[pred][sample] - u8.
    residual_tokens: Vec<Vec<u8>>,
    /// extra_bits[pred][sample] - u8.
    extra_bits: Vec<Vec<u8>>,
    /// bucket_indices[prop][sample] - u8.
    bucket_indices: Vec<Vec<u8>>,
    /// Composite-key property list (subset of bucket_indices indices).
    properties: Vec<usize>,
    num_pred: usize,
    num_samples: usize,
    /// Number of properties (= properties.len()).
    num_active_props: usize,
}

fn make_samples(n: usize, dup_fraction: f32) -> Samples {
    // Build samples where about `dup_fraction` of rows are duplicates.
    let unique_n = ((n as f32) * (1.0 - dup_fraction)).max(1.0) as usize;

    let mut residual_tokens: Vec<Vec<u8>> =
        (0..NUM_PREDICTORS).map(|_| Vec::with_capacity(n)).collect();
    let mut extra_bits: Vec<Vec<u8>> = (0..NUM_PREDICTORS).map(|_| Vec::with_capacity(n)).collect();
    let mut bucket_indices: Vec<Vec<u8>> = (0..NUM_PROPERTIES_TOTAL)
        .map(|_| Vec::with_capacity(n))
        .collect();

    // Generate `n` samples by drawing from a pool of `unique_n` patterns.
    for i in 0..n {
        let pattern = (i.wrapping_mul(0x9e3779b1)) % unique_n;
        // Hash pattern to pseudo-random bytes for each field.
        let mut h = pattern as u64;
        for pred in 0..NUM_PREDICTORS {
            h = h.wrapping_mul(0x100000001b3).wrapping_add(pred as u64);
            residual_tokens[pred].push((h >> 24) as u8);
            extra_bits[pred].push((h >> 32) as u8);
        }
        for prop in 0..NUM_PROPERTIES_TOTAL {
            h = h.wrapping_mul(0x100000001b3).wrapping_add(prop as u64);
            bucket_indices[prop].push((h >> 40) as u8);
        }
    }

    // Property indices used at e7 lossless (PROP_ORDER_NO_SQUEEZE_NO_GID first 7 + 2 ref).
    let properties: Vec<usize> = vec![0, 15, 9, 10, 11, 12, 13, 14, 16];
    // bucket_indices size: 16 base + maybe ref. Make sure indexes are in range.
    let mut bi = bucket_indices;
    while bi.len() <= 16 {
        bi.push(Vec::with_capacity(n));
        let new_idx = bi.len() - 1;
        let mut h: u64 = (new_idx as u64).wrapping_mul(0xdeadbeef);
        for _ in 0..n {
            h = h.wrapping_mul(0x100000001b3);
            bi[new_idx].push((h >> 16) as u8);
        }
    }

    Samples {
        residual_tokens,
        extra_bits,
        bucket_indices: bi,
        properties,
        num_pred: NUM_PREDICTORS,
        num_samples: n,
        num_active_props: NUM_PROPERTIES_ACTIVE,
    }
}

/// Compact all SoA arrays to the unique subset (matching production cost).
#[inline(never)]
fn compact_arrays(
    samples: &Samples,
    unique_indices: &[usize],
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut new_tokens: Vec<Vec<u8>> = Vec::with_capacity(samples.num_pred);
    let mut new_ebits: Vec<Vec<u8>> = Vec::with_capacity(samples.num_pred);
    for pred in 0..samples.num_pred {
        let old_t = &samples.residual_tokens[pred];
        let old_e = &samples.extra_bits[pred];
        new_tokens.push(unique_indices.iter().map(|&i| old_t[i]).collect());
        new_ebits.push(unique_indices.iter().map(|&i| old_e[i]).collect());
    }
    let mut new_bi: Vec<Vec<u8>> = Vec::with_capacity(samples.bucket_indices.len());
    for old_bi in samples.bucket_indices.iter() {
        if old_bi.is_empty() {
            new_bi.push(Vec::new());
        } else {
            new_bi.push(unique_indices.iter().map(|&i| old_bi[i]).collect());
        }
    }
    (new_tokens, new_ebits, new_bi)
}

/// CURRENT production implementation, copied from
/// jxl-encoder/src/modular/tree_learn.rs:869-962.
#[inline(never)]
fn dedup_current(samples: &Samples) -> (Vec<usize>, Vec<u32>) {
    use core::cmp::Ordering;

    let n = samples.num_samples;
    let num_pred = samples.num_pred;
    let properties = &samples.properties;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        for &prop_idx in properties {
            let bi = &samples.bucket_indices[prop_idx];
            if !bi.is_empty() {
                match bi[a].cmp(&bi[b]) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
        }
        for pred in 0..num_pred {
            match samples.residual_tokens[pred][a].cmp(&samples.residual_tokens[pred][b]) {
                Ordering::Equal => {}
                ord => return ord,
            }
            match samples.extra_bits[pred][a].cmp(&samples.extra_bits[pred][b]) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        Ordering::Equal
    });

    let mut unique_indices: Vec<usize> = Vec::with_capacity(n / 2);
    let mut counts: Vec<u32> = Vec::with_capacity(n / 2);

    let is_same = |a: usize, b: usize| -> bool {
        for &prop_idx in properties {
            let bi = &samples.bucket_indices[prop_idx];
            if !bi.is_empty() && bi[a] != bi[b] {
                return false;
            }
        }
        for pred in 0..num_pred {
            if samples.residual_tokens[pred][a] != samples.residual_tokens[pred][b] {
                return false;
            }
            if samples.extra_bits[pred][a] != samples.extra_bits[pred][b] {
                return false;
            }
        }
        true
    };

    unique_indices.push(order[0]);
    counts.push(1);
    for &curr in &order[1..] {
        let prev = *unique_indices.last().unwrap();
        if is_same(prev, curr) {
            *counts.last_mut().unwrap() += 1;
        } else {
            unique_indices.push(curr);
            counts.push(1);
        }
    }
    (unique_indices, counts)
}

/// PACKED-KEY SORT: materialize each sample's composite key into a
/// contiguous `[u8; K]`, then sort indices by packed key. Cuts per-cmp
/// cache misses from ~42 (scattered Vec<Vec<u8>>) to <=2 (single cacheline
/// read per key).
#[inline(never)]
fn dedup_packed_key_sort(samples: &Samples) -> (Vec<usize>, Vec<u32>) {
    let n = samples.num_samples;
    let num_pred = samples.num_pred;
    let properties = &samples.properties;
    let num_active_props = samples.num_active_props;

    // Pack key: [props..] then [tok_pred0, eb_pred0, tok_pred1, eb_pred1, ...]
    // Length = num_active_props + 2 * num_pred bytes.
    let key_len = num_active_props + 2 * num_pred;
    debug_assert!(key_len <= PACKED_KEY_BYTES);

    // Materialize packed keys.
    let mut keys: Vec<[u8; PACKED_KEY_BYTES]> = vec![[0u8; PACKED_KEY_BYTES]; n];
    for i in 0..n {
        let k = &mut keys[i];
        let mut off = 0;
        for &prop_idx in properties {
            let bi = &samples.bucket_indices[prop_idx];
            if !bi.is_empty() {
                k[off] = bi[i];
            }
            off += 1;
        }
        for pred in 0..num_pred {
            k[off] = samples.residual_tokens[pred][i];
            off += 1;
            k[off] = samples.extra_bits[pred][i];
            off += 1;
        }
    }

    // Sort indices by packed key (lexicographic on key bytes).
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        // Compare only the first `key_len` bytes; the rest are zero-padded.
        let ka = &keys[a as usize][..key_len];
        let kb = &keys[b as usize][..key_len];
        ka.cmp(kb)
    });

    // Walk sorted order, merge consecutive identical keys.
    let mut unique_indices: Vec<usize> = Vec::with_capacity(n / 2);
    let mut counts: Vec<u32> = Vec::with_capacity(n / 2);
    if n > 0 {
        unique_indices.push(order[0] as usize);
        counts.push(1);
        let mut prev_key_idx = order[0];
        for &curr in &order[1..] {
            if keys[curr as usize][..key_len] == keys[prev_key_idx as usize][..key_len] {
                *counts.last_mut().unwrap() += 1;
            } else {
                unique_indices.push(curr as usize);
                counts.push(1);
                prev_key_idx = curr;
            }
        }
    }
    (unique_indices, counts)
}

/// HASHMAP dedup: libjxl-style streaming hash dedup. Each sample's key
/// hashes to a slot; matching key bumps counts, new key gets a new
/// position. O(n) expected, no sort.
#[inline(never)]
fn dedup_hashmap(samples: &Samples) -> (Vec<usize>, Vec<u32>) {
    let n = samples.num_samples;
    let num_pred = samples.num_pred;
    let properties = &samples.properties;
    let num_active_props = samples.num_active_props;
    let _key_len = num_active_props + 2 * num_pred;

    // Materialize packed keys (same cost as packed-key sort, fair comparison).
    let mut keys: Vec<[u8; PACKED_KEY_BYTES]> = vec![[0u8; PACKED_KEY_BYTES]; n];
    for i in 0..n {
        let k = &mut keys[i];
        let mut off = 0;
        for &prop_idx in properties {
            let bi = &samples.bucket_indices[prop_idx];
            if !bi.is_empty() {
                k[off] = bi[i];
            }
            off += 1;
        }
        for pred in 0..num_pred {
            k[off] = samples.residual_tokens[pred][i];
            off += 1;
            k[off] = samples.extra_bits[pred][i];
            off += 1;
        }
    }

    // Hashbrown HashMap: fast non-crypto. Key = [u8; PACKED_KEY_BYTES], value = unique-idx.
    let mut map: HashMap<[u8; PACKED_KEY_BYTES], u32> = HashMap::with_capacity(n / 2);
    let mut unique_indices: Vec<usize> = Vec::with_capacity(n / 2);
    let mut counts: Vec<u32> = Vec::with_capacity(n / 2);

    for i in 0..n {
        let key = keys[i];
        match map.get(&key) {
            Some(&idx) => {
                counts[idx as usize] += 1;
            }
            None => {
                let idx = unique_indices.len() as u32;
                map.insert(key, idx);
                unique_indices.push(i);
                counts.push(1);
            }
        }
    }
    (unique_indices, counts)
}

/// Run the dedup_current and also compact arrays (matches what the
/// production function actually does). This is the full apples-to-apples
/// wall-clock comparison.
fn full_current(samples: &Samples) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u32>) {
    let (unique_indices, counts) = dedup_current(samples);
    let (t, e, b) = compact_arrays(samples, &unique_indices);
    (t, e, b, counts)
}

fn full_packed(samples: &Samples) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u32>) {
    let (unique_indices, counts) = dedup_packed_key_sort(samples);
    let (t, e, b) = compact_arrays(samples, &unique_indices);
    (t, e, b, counts)
}

fn full_hashmap(samples: &Samples) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u32>) {
    let (unique_indices, counts) = dedup_hashmap(samples);
    let (t, e, b) = compact_arrays(samples, &unique_indices);
    (t, e, b, counts)
}

fn bench_for_count<const N: usize, const DUP_BPM: u32>(suite: &mut Suite, label: &str) {
    // DUP_BPM = parts per 1000 (since const generics can't be f32). dup_frac =
    // DUP_BPM / 1000.0. e.g. 300 -> 30% duplicates.
    let group_name = format!("dedup_full_{label}_dup{DUP_BPM}");
    suite.group(&group_name, |g| {
        g.throughput(Throughput::Elements(N as u64));

        g.bench("current_sort_indirect", |b| {
            b.with_input(|| make_samples(N, DUP_BPM as f32 / 1000.0))
                .run(|samples| {
                    let out = full_current(&samples);
                    black_box(out);
                    samples
                })
        });

        g.bench("packed_key_sort", |b| {
            b.with_input(|| make_samples(N, DUP_BPM as f32 / 1000.0))
                .run(|samples| {
                    let out = full_packed(&samples);
                    black_box(out);
                    samples
                })
        });

        g.bench("hashmap_dedup", |b| {
            b.with_input(|| make_samples(N, DUP_BPM as f32 / 1000.0))
                .run(|samples| {
                    let out = full_hashmap(&samples);
                    black_box(out);
                    samples
                })
        });

        g.baseline("current_sort_indirect");
        g.config().sort_by_speed(true);
    });

    let dedup_only_group = format!("dedup_only_{label}_dup{DUP_BPM}");
    suite.group(&dedup_only_group, |g| {
        g.throughput(Throughput::Elements(N as u64));

        g.bench("current_sort_indirect", |b| {
            b.with_input(|| make_samples(N, DUP_BPM as f32 / 1000.0))
                .run(|samples| {
                    let out = dedup_current(&samples);
                    black_box(out);
                    samples
                })
        });

        g.bench("packed_key_sort", |b| {
            b.with_input(|| make_samples(N, DUP_BPM as f32 / 1000.0))
                .run(|samples| {
                    let out = dedup_packed_key_sort(&samples);
                    black_box(out);
                    samples
                })
        });

        g.bench("hashmap_dedup", |b| {
            b.with_input(|| make_samples(N, DUP_BPM as f32 / 1000.0))
                .run(|samples| {
                    let out = dedup_hashmap(&samples);
                    black_box(out);
                    samples
                })
        });

        g.baseline("current_sort_indirect");
        g.config().sort_by_speed(true);
    });
}

fn bench_dedup(suite: &mut Suite) {
    // Real-photo-scale sample counts derived from 2026-05-16 e7 profile:
    //   0.26 MP -> ~200K samples
    //   1.05 MP -> ~1.35M samples
    //   4.19 MP -> ~3.2M samples
    // DUP_BPM = 300 -> 30% duplicates (unique = ~70% of input). Photos
    // dedup to roughly pixel_fraction × total_pixels unique.
    bench_for_count::<200_000, 300>(suite, "n=200K");
    bench_for_count::<1_350_000, 300>(suite, "n=1.35M");
    bench_for_count::<3_200_000, 300>(suite, "n=3.2M");
}

zenbench::main!(bench_dedup);
