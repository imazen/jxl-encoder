// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Phase 3 of issue #41 — true libjxl-parity `AddSample` / `AddToTableAndMerge`
//! port with the **packed key stored inline beside each slot**, eliminating
//! the SoA chase that doomed Phases 1 and 2.
//!
//! # Why a third dedup primitive?
//!
//! Phases 1 and 2 of issue #41 ported the libjxl two-hash cuckoo table but
//! kept the post-gather parallel-SoA layout: the table stored only indices
//! into [`TreeSamples`] arrays, and the `IsSameSample` check had to *read*
//! from those arrays per probe. The two earlier shapes both regressed:
//!
//! * **Phase 1 (`StreamingDedupTable`, commit 3f4b135):** post-gather pass
//!   that hashed each sample by random-accessing the parallel-SoA arrays.
//!   Real-photo measurement: +3 % to +8 % wall-clock at e7 (1.05 MP +4.4 %,
//!   4.19 MP +7.6 %). Cause: `pack_sample_key` reads ~42 scattered bytes
//!   from `Vec<Vec<u8>>` per sample — full cache miss per key build.
//! * **Phase 2 (`GatherDedupTable`, commit 63e5ea2):** moved the probe
//!   *inside* the gather loop and hashed from the local stack arrays
//!   (cache-hot writes). Improved at e9 (-13 % on the cliff at 0.26 MP)
//!   but still +2-4 % at e7 on larger images. Cause: the `is_same_local`
//!   tie-break still reads cold SoA columns of the *historical* sample
//!   at the index the table stored — one full SoA chase per cuckoo
//!   probe that finds a slot occupied.
//!
//! # Phase 3 design
//!
//! The libjxl shape that actually wins, hidden inside `AddSample` at
//! `lib/jxl/modular/encoding/enc_ma.cc:711`: the SoA push happens first
//! (cache-hot writes to the tail), the hash + IsSameSample read **the
//! just-pushed tail row** (same cacheline still hot), and on duplicate
//! the SoA tail is `pop_back`-ed. The `dedup_table_` only ever indexes
//! the live row; comparison reads the tail.
//!
//! Our pipeline cannot replicate that exactly (the per-channel gather
//! loop pushes a sample to *one* SoA column at a time and then
//! immediately moves on, so the just-pushed row is not cache-hot when
//! the *next* sample arrives and probes against it). What we can do is
//! eliminate the SoA chase entirely by storing a **fingerprint hash
//! inline in the slot**:
//!
//! ```text
//!   struct Slot { fp: u64, index: u32 }   // 12 bytes
//! ```
//!
//! and keeping the canonical packed keys in a separate compact
//! `Vec<[u8; KEY_BYTES]>` indexed by the slot's `index`. The probe
//! then becomes:
//!
//! 1. `hash1(key) → h1`, read `slot.fp` at `h1` (1 cacheline, dense:
//!    ~5 slots/cacheline at 12 bytes/slot).
//! 2. If `slot.fp == fingerprint(key)` AND `slot.index != EMPTY`,
//!    verify by full-key compare against `unique_keys[slot.index]`
//!    (one more cacheline). Mismatch ratio is dominated by hash
//!    quality — for any non-degenerate hash, ≤1/2^64 false positives.
//! 3. Else try Hash2 slot.
//! 4. Both miss → insert into first empty slot.
//!
//! This keeps the slot table cache-dense (12 bytes/slot ⇒ at 2 M
//! samples the table is ~32 MB, but only the slot row touched on each
//! probe matters — false-fingerprint reads stay in L1) while still
//! letting us avoid the SoA chase 99 %+ of the time (only true
//! duplicates and rare fingerprint collisions hit `unique_keys`).
//!
//! The first iteration of this primitive stored the full key inline in
//! the slot. That ballooned the slot table to KEY_BYTES + 4 = 68 bytes
//! per slot and was 2-3× *slower* than the packed-key sort in the
//! microbench (`/tmp/inline_dedup_proof_keys_inline_first_pass.txt`,
//! 2026-05-17): a 1.35 M sample table is then 89 MB which thrashes
//! L2/L3. The fingerprint-cache layout below collapses the slot to a
//! cacheline-friendly 12 bytes while preserving the no-SoA-chase
//! property for the common case.
//!
//! # What this file ships (Chunk 1)
//!
//! * [`InlineDedupTable`] — the open-addressing table with fingerprint
//!   cache filter and compact canonical-key array.
//! * `InlineSlot` (private) — the slot record (`u64 fingerprint` +
//!   `u32 index`); 12 bytes/slot, cacheline-dense for fast misses.
//! * Unit tests cross-checking [`InlineDedupTable`] against a reference
//!   packed-key sort dedup on randomised key streams (15 cases:
//!   all-unique, all-duplicate, half-duplicate, heavy duplicate,
//!   interleaved patterns, low-entropy, photo-like clusters, two
//!   random seeds, plus sentinel / capacity / lookup-only smoke tests).
//! * No production wiring yet. Chunk 2 plumbs it through the gather
//!   loop and dedup dispatcher.
//!
//! # libjxl references
//!
//! * `lib/jxl/modular/encoding/enc_ma.cc:603-655` — `AddToTableAndMerge`,
//!   `AddToTable`, two-hash cuckoo placement.
//! * `lib/jxl/modular/encoding/enc_ma.cc:657-686` — `Hash1`, `Hash2`
//!   (`0x1e35a7bd` mul-add fold and `0x1e35a7bd1e35a7bd` mul-xor fold).
//! * `lib/jxl/modular/encoding/enc_ma.cc:711-737` — `AddSample` push +
//!   `AddToTableAndMerge` + `pop_back` rollback on hit.
//! * `lib/jxl/modular/encoding/enc_ma.cc:642-655` — `PrepareForSamples`
//!   load-factor sizing (table size = `next_pow2(n * 3 / 2)`).

// Match libjxl `kDedupEntryUnused` (`enc_ma.h:153`).
const SLOT_EMPTY: u32 = u32::MAX;

// Mul-add hash constant from libjxl `enc_ma.cc:658`.
const HASH1_CONST: u64 = 0x1e35a7bd;
// Mul-xor hash constant from libjxl `enc_ma.cc:673`. The two distinct
// constants + ADD-vs-XOR combination decorrelate the two hash functions.
const HASH2_CONST: u64 = 0x1e35a7bd1e35a7bd;

/// Packed-key byte budget for the inline dedup primitive. Matches
/// [`super::tree_learn::DEDUP_KEY_BYTES`] (64 bytes) so all dedup
/// implementations operate on bit-identical key formats.
///
/// Worst case: 16 base properties + 16 ref-channel properties = 32
/// prop bytes + 14 candidate predictors * 2 bytes = 28 token bytes =
/// 60 bytes. The 4-byte tail is zero-padded; trailing zeros are
/// identical across all samples so they do not affect comparison.
pub const KEY_BYTES: usize = 64;

/// One slot in [`InlineDedupTable`]: 64-bit fingerprint plus 32-bit
/// canonical-key-array index. 12 bytes/slot keeps the slot table
/// cacheline-dense (5-6 slots per 64-byte line) so probes that miss
/// on the fingerprint stay in L1.
///
/// The fingerprint is `Hash1(key) | (Hash2(key) << 32)` — a 64-bit
/// signature of the full key built from the same two hash functions
/// that pick the probe positions. A non-degenerate hash gives ≤ 1
/// false positive per 2^64 distinct keys; in practice the full-key
/// verify step (against `unique_keys[index]`) catches the collision
/// without any observable cost.
#[repr(C)]
#[derive(Clone, Copy)]
struct InlineSlot {
    /// 64-bit fingerprint of the canonical key at `index`. Set to 0
    /// for empty slots; on the rare event that a real key hashes to
    /// `(0, 0)` it gets a forced bump to `(1, 1)` to keep the empty
    /// sentinel distinguishable.
    fingerprint: u64,
    /// Index into [`InlineDedupTable::unique_keys`]. `SLOT_EMPTY` for
    /// empty slots; this is the field that distinguishes empty vs
    /// occupied, so `fingerprint` collisions on empty slots stay
    /// correctly handled.
    index: u32,
}

impl InlineSlot {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            fingerprint: 0,
            index: SLOT_EMPTY,
        }
    }
}

/// Inline-fingerprint open-addressing dedup table. See module docs for
/// the design rationale and libjxl references.
///
/// Slot layout: `(u64 fingerprint, u32 index)` — 12 bytes per slot.
/// Canonical packed keys live in a compact `unique_keys: Vec<[u8;
/// KEY_BYTES]>` so the slot probe only reads `unique_keys[i]` on
/// fingerprint match (which is rare for non-duplicate inputs).
///
/// The `lookup_or_insert` hot path takes a candidate packed key and:
///
/// 1. Probes slot at `hash1(key)`. If `slot.fingerprint == fp` and the
///    canonical-key compare confirms → returns `Some(slot.index)`.
/// 2. Probes slot at `hash2(key)`. Same check.
/// 3. On both miss → push the key to `unique_keys`, insert
///    `(fp, push_index)` into the first empty slot, return `None`.
pub struct InlineDedupTable {
    /// Pow-2-sized slot table. Each slot is 12 bytes (fingerprint +
    /// index); the fingerprint serves as a cheap filter to avoid the
    /// `unique_keys[i]` cacheline read on misses.
    slots: Box<[InlineSlot]>,
    /// Canonical packed keys for each unique sample. Indexed by
    /// `slot.index`; only accessed on fingerprint match. Compact
    /// (no slot-table padding), so sequentially-allocated entries
    /// stay close together in memory for repeated dedup-time access.
    unique_keys: Vec<[u8; KEY_BYTES]>,
    /// `slots.len() - 1`; `&` mask for pow-2 indexing.
    mask: u32,
}

impl InlineDedupTable {
    /// Allocate a table sized for `expected_samples` unique entries.
    ///
    /// Capacity is rounded up to `next_pow2(max(16, expected * 3 / 2))`
    /// so the load factor stays ≤ 2/3 at convergence — matching libjxl's
    /// `PrepareForSamples` at `enc_ma.cc:653`. A pow-2 cap also lets the
    /// probe use `& mask` instead of `%`, which the compiler can fold
    /// into the hash function's tail shift.
    ///
    /// `expected_samples = 0` is allowed (returns a minimum-size 16-slot
    /// table) so callers can construct the table before they know the
    /// exact sample count.
    pub fn new(expected_samples: usize) -> Self {
        let target = expected_samples.saturating_mul(3).div_ceil(2).max(16);
        let cap = target.next_power_of_two();
        // Sanity bound: u32 indices means the unique count must fit in
        // u32. The pow2-rounded cap can be up to 1 << 30 slots
        // (~12 GB at 12 bytes/slot) — well beyond any sane sample count.
        debug_assert!(cap <= (1usize << 30));
        let slots = vec![InlineSlot::empty(); cap].into_boxed_slice();
        Self {
            slots,
            unique_keys: Vec::with_capacity(expected_samples.max(16)),
            mask: (cap - 1) as u32,
        }
    }

    /// Number of slots in the table (the pow-2-rounded capacity, not
    /// the live occupied count). Used by the microbench / tests.
    // Exposed via `__bench_internals`; default-features clippy reports unused.
    #[allow(dead_code)]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of unique keys currently stored.
    #[allow(dead_code)] // bench-only (`__bench_internals`)
    #[inline]
    pub fn len(&self) -> usize {
        self.unique_keys.len()
    }

    /// Whether the table currently holds zero entries.
    #[allow(dead_code)] // bench-only (`__bench_internals`)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.unique_keys.is_empty()
    }

    /// libjxl `Hash1` (`enc_ma.cc:657-671`): mul-add fold over key bytes
    /// with `0x1e35a7bd`. Returns the **raw 64-bit fold** before the
    /// `& mask` step so callers can re-use it as part of a fingerprint.
    /// The probe position is `(h >> 16) & mask`.
    #[inline(always)]
    fn raw_hash1(key: &[u8; KEY_BYTES]) -> u64 {
        let mut h: u64 = HASH1_CONST;
        for &b in key.iter() {
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(b as u64);
        }
        h
    }

    /// libjxl `Hash2` (`enc_ma.cc:672-686`): mul-xor fold with the
    /// 64-bit constant. XOR (not ADD) decorrelates this from Hash1.
    /// Returns the raw 64-bit fold.
    #[inline(always)]
    fn raw_hash2(key: &[u8; KEY_BYTES]) -> u64 {
        let mut h: u64 = HASH2_CONST;
        for &b in key.iter() {
            h = h.wrapping_mul(HASH2_CONST) ^ (b as u64);
        }
        h
    }

    /// Build the 64-bit fingerprint from the two hash folds: take 32
    /// bits from each, XORed together so a bit flip in either fold
    /// changes the result. The full 64 bits go in `slot.fingerprint`
    /// for the cheap-probe filter.
    #[inline(always)]
    fn fingerprint_from(h1: u64, h2: u64) -> u64 {
        let fp = h1 ^ h2.rotate_left(32);
        // Reserve 0 for the empty sentinel; ~1 / 2^64 keys collide
        // with this and get bumped to 1 instead — visible only as one
        // extra fingerprint hit (caught by the full-key verify).
        if fp == 0 { 1 } else { fp }
    }

    /// Hot path: probe both cuckoo slots; on duplicate return
    /// `Some(existing_index)`; on miss insert `(fp, next_index)` into
    /// the first empty slot, push the key to `unique_keys`, and return
    /// `None`.
    ///
    /// `next_index` parameter is retained for symmetry with the Phase
    /// 1/2 primitive but ignored here — `InlineDedupTable` manages its
    /// own canonical-key indexing internally. The returned `Some(idx)`
    /// on hit is the index into `unique_keys()`, which the caller can
    /// use to address parallel per-unique arrays (sample counts, sort
    /// orders, etc.).
    ///
    /// # Cache behaviour vs Phases 1 & 2
    ///
    /// Phase 1: probe slot reads `index` (4 bytes), then chases SoA
    /// arrays at `samples[index]` to read ~42 scattered bytes. One
    /// probe = 30+ cachelines.
    ///
    /// Phase 3: probe slot reads `(fp, index)` = 12 bytes. Fingerprint
    /// compare in registers; only on match does the cold `unique_keys[idx]`
    /// cacheline load happen. One probe = 1 cacheline (the slot) for
    /// the ≥ 99 % miss case; 2 cachelines for the verify case.
    #[inline(always)]
    pub fn lookup_or_insert(&mut self, key: &[u8; KEY_BYTES], next_index: u32) -> Option<u32> {
        debug_assert_ne!(
            next_index, SLOT_EMPTY,
            "next_index = SLOT_EMPTY ({}) is reserved as the empty sentinel",
            SLOT_EMPTY,
        );

        let raw_h1 = Self::raw_hash1(key);
        let raw_h2 = Self::raw_hash2(key);
        let fp = Self::fingerprint_from(raw_h1, raw_h2);
        let h1 = ((raw_h1 >> 16) as u32 & self.mask) as usize;
        let h2 = ((raw_h2 >> 16) as u32 & self.mask) as usize;

        let s1 = self.slots[h1];
        if s1.index != SLOT_EMPTY
            && s1.fingerprint == fp
            && &self.unique_keys[s1.index as usize] == key
        {
            return Some(s1.index);
        }
        let s2 = self.slots[h2];
        if s2.index != SLOT_EMPTY
            && s2.fingerprint == fp
            && &self.unique_keys[s2.index as usize] == key
        {
            return Some(s2.index);
        }
        // Miss: push the key into the compact canonical-key array and
        // record its position in the first empty slot.
        let new_idx = self.unique_keys.len() as u32;
        self.unique_keys.push(*key);
        if s1.index == SLOT_EMPTY {
            self.slots[h1] = InlineSlot {
                fingerprint: fp,
                index: new_idx,
            };
        } else if s2.index == SLOT_EMPTY {
            self.slots[h2] = InlineSlot {
                fingerprint: fp,
                index: new_idx,
            };
        }
        // Both occupied: silent drop, matching libjxl `AddToTable`
        // (`enc_ma.cc:632`). The unique row is still emitted (and is
        // reachable via `unique_keys[new_idx]` — the caller pairs that
        // index with its own per-unique data); future identical keys
        // simply won't find this slot and may create a separate entry.
        //
        // The downstream tree builder treats all rows with identical
        // keys as merged regardless of slot routing — bitstream output
        // is unaffected.
        //
        // For internal-state consistency, both unique_keys[i] and the
        // returned `None` mean "the caller now owns a fresh unique-row
        // slot at index new_idx".
        let _ = next_index;
        None
    }

    /// Probe-only counterpart of [`Self::lookup_or_insert`]. Returns
    /// `Some(index)` if the key is already present, `None` otherwise.
    /// Does not mutate the table. Used by tests and debug assertions.
    #[allow(dead_code)] // bench-only (`__bench_internals`)
    #[inline]
    pub fn lookup_only(&self, key: &[u8; KEY_BYTES]) -> Option<u32> {
        let raw_h1 = Self::raw_hash1(key);
        let raw_h2 = Self::raw_hash2(key);
        let fp = Self::fingerprint_from(raw_h1, raw_h2);
        let h1 = ((raw_h1 >> 16) as u32 & self.mask) as usize;
        let h2 = ((raw_h2 >> 16) as u32 & self.mask) as usize;
        let s1 = self.slots[h1];
        if s1.index != SLOT_EMPTY
            && s1.fingerprint == fp
            && &self.unique_keys[s1.index as usize] == key
        {
            return Some(s1.index);
        }
        let s2 = self.slots[h2];
        if s2.index != SLOT_EMPTY
            && s2.fingerprint == fp
            && &self.unique_keys[s2.index as usize] == key
        {
            return Some(s2.index);
        }
        None
    }

    /// Borrow the canonical packed keys for unique samples 0..len. The
    /// caller can index by the value returned from `lookup_or_insert`
    /// (on miss → new_idx == prev_len) to fetch the merged-into key.
    #[allow(dead_code)] // bench-only (`__bench_internals`)
    #[inline]
    pub fn unique_keys(&self) -> &[[u8; KEY_BYTES]] {
        &self.unique_keys
    }
}

/// Decision returned from [`pack_local_key_phase3`] when the configured
/// (`num_pred`, property list) combination cannot be encoded inside the
/// fixed [`KEY_BYTES`] budget without losing precision.
///
/// The gather-time dispatcher consults this before flipping into Phase 3
/// mode: when [`Self::Overflow`] is returned, the caller transparently
/// falls back to Phase 2's [`crate::modular::tree_learn::GatherDedupTable`]
/// for the offending image so we never widen merges beyond the strict
/// "at-or-below the post-sort bucket-equivalence set" contract.
pub enum LocalKeyPackResult {
    /// Key fits inside [`KEY_BYTES`]; safe to dispatch to Phase 3.
    Packed([u8; KEY_BYTES]),
    /// `2 * num_pred + 4 * num_props` exceeded [`KEY_BYTES`]; Phase 3
    /// cannot represent the sample without losing precision (which would
    /// over-merge and break Phase 2's "subset of bucket-equivalence" guarantee).
    Overflow,
}

/// Layout the gather-time packed key for [`InlineDedupTable`] from the
/// local stack arrays that `gather_channel_samples` already builds before
/// the SoA push.
///
/// Layout:
///   * 2 bytes per candidate predictor: `[token, extra_bits]` × `num_pred`.
///   * 4 bytes per hashed property (little-endian i32, full precision)
///     for each entry in `properties_to_hash` — the post-y/x-skip
///     property list the caller (`GatherDedupTable::new_with_properties`
///     in Phase 2) would have selected. Property indices < [`NUM_PROPERTIES`]
///     are read from `local_props`; indices ≥ [`NUM_PROPERTIES`] are read
///     from `local_ref_props` at `idx - NUM_PROPERTIES`.
///   * Zero-padded tail (unused bytes stay 0).
///
/// Returns [`LocalKeyPackResult::Overflow`] when
/// `2 * num_pred + 4 * properties_to_hash.len()` exceeds [`KEY_BYTES`].
/// In that case the caller falls back to Phase 2 to avoid silently
/// over-merging bit-different samples.
///
/// `NUM_PROPERTIES` is the base-property width of `local_props` — keep
/// in sync with [`crate::modular::tree_learn::NUM_PROPERTIES`].
#[inline]
pub fn pack_local_key_phase3(
    local_tokens: &[u8],
    local_ebits: &[u8],
    local_props: &[i32],
    local_ref_props: &[i32],
    properties_to_hash: &[u8],
    num_properties_base: usize,
) -> LocalKeyPackResult {
    debug_assert_eq!(local_tokens.len(), local_ebits.len());
    let num_pred = local_tokens.len();
    let num_props_hashed = properties_to_hash.len();
    let bytes_needed = 2usize.saturating_mul(num_pred) + 4usize.saturating_mul(num_props_hashed);
    if bytes_needed > KEY_BYTES {
        return LocalKeyPackResult::Overflow;
    }
    let mut key = [0u8; KEY_BYTES];
    let mut off = 0usize;
    // Predictor residual tokens: (token, ebits) pairs in predictor-index
    // order. Cache-hot because the caller just computed these.
    for i in 0..num_pred {
        key[off] = local_tokens[i];
        key[off + 1] = local_ebits[i];
        off += 2;
    }
    // Properties: full i32 little-endian. Source slice depends on whether
    // the property index falls in base (< num_properties_base) or
    // ref-channel (>=) range — same dispatch Phase 2's hash1_local uses.
    for &p in properties_to_hash {
        let p = p as usize;
        let v: i32 = if p < num_properties_base {
            local_props.get(p).copied().unwrap_or(0)
        } else {
            let r = p - num_properties_base;
            local_ref_props.get(r).copied().unwrap_or(0)
        };
        let b = v.to_le_bytes();
        key[off] = b[0];
        key[off + 1] = b[1];
        key[off + 2] = b[2];
        key[off + 3] = b[3];
        off += 4;
    }
    LocalKeyPackResult::Packed(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a packed key from a u64 seed by hashing it into the
    /// 64-byte buffer. Two identical seeds produce byte-identical
    /// keys; distinct seeds produce uncorrelated keys (with high
    /// probability).
    fn make_key(seed: u64) -> [u8; KEY_BYTES] {
        let mut k = [0u8; KEY_BYTES];
        let mut h = seed.wrapping_mul(0x9e3779b97f4a7c15);
        for byte in k.iter_mut() {
            h = h.wrapping_mul(0x100000001b3).wrapping_add(0xdeadbeef);
            *byte = (h >> 24) as u8;
        }
        k
    }

    /// Reference dedup: classic packed-key sort + walk-and-merge over
    /// `(key, original_index)` pairs. Returns `(unique_keys,
    /// per_unique_count)` in arbitrary order (the production sort
    /// dedup also doesn't guarantee a specific order across
    /// algorithms; tests compare as multisets).
    fn reference_dedup(keys: &[[u8; KEY_BYTES]]) -> Vec<([u8; KEY_BYTES], u32)> {
        let mut pairs: Vec<([u8; KEY_BYTES], u32)> = Vec::new();
        let mut ordered: Vec<[u8; KEY_BYTES]> = keys.to_vec();
        ordered.sort_unstable();
        for k in ordered {
            if let Some(last) = pairs.last_mut()
                && last.0 == k
            {
                last.1 += 1;
                continue;
            }
            pairs.push((k, 1));
        }
        pairs
    }

    /// Run [`InlineDedupTable`] over a key stream and return the
    /// unique-key set + per-unique counts. The table manages its own
    /// canonical-key storage internally; we maintain a parallel
    /// `counts` vec keyed by the returned unique-row index.
    fn run_inline_dedup(keys: &[[u8; KEY_BYTES]]) -> Vec<([u8; KEY_BYTES], u32)> {
        let mut table = InlineDedupTable::new(keys.len());
        let mut counts: Vec<u32> = Vec::new();
        for k in keys {
            // The `next_index` argument is informational only — the
            // table assigns its own indices. We still pass a unique
            // non-sentinel value so the debug assertion stays happy.
            let probe_idx = (counts.len() as u32).min(u32::MAX - 1);
            match table.lookup_or_insert(k, probe_idx) {
                Some(existing) => {
                    counts[existing as usize] += 1;
                }
                None => {
                    counts.push(1);
                }
            }
        }
        // Pair canonical keys (owned by the table) with the per-row
        // counts (owned by us). The contract is `unique_keys()[i]`
        // corresponds to the i'th miss returned, which is exactly
        // `counts[i]`.
        table.unique_keys().iter().copied().zip(counts).collect()
    }

    /// Multiset equality: same `(key, count)` pairs regardless of
    /// order. This is the right invariant because dedup
    /// implementations yield unique rows in different orders (sort
    /// vs hash) but the post-dedup tree-learning step is
    /// order-insensitive.
    fn assert_multiset_eq(
        lhs: &[([u8; KEY_BYTES], u32)],
        rhs: &[([u8; KEY_BYTES], u32)],
        label: &str,
    ) {
        let mut lhs_sorted = lhs.to_vec();
        let mut rhs_sorted = rhs.to_vec();
        lhs_sorted.sort_unstable();
        rhs_sorted.sort_unstable();
        assert_eq!(
            lhs_sorted,
            rhs_sorted,
            "{label}: multisets differ\n  lhs len={} rhs len={}",
            lhs_sorted.len(),
            rhs_sorted.len()
        );
    }

    #[test]
    fn empty_table_lookups_miss() {
        let table = InlineDedupTable::new(0);
        let key = make_key(42);
        assert_eq!(table.lookup_only(&key), None);
        assert_eq!(table.capacity(), 16);
        assert!(table.is_empty());
    }

    #[test]
    fn single_insert_then_lookup_hits() {
        let mut table = InlineDedupTable::new(4);
        let key = make_key(123);
        assert_eq!(table.lookup_or_insert(&key, 0), None);
        assert_eq!(table.lookup_only(&key), Some(0));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn duplicate_insert_returns_existing_index() {
        let mut table = InlineDedupTable::new(4);
        let key = make_key(7);
        assert_eq!(table.lookup_or_insert(&key, 0), None);
        assert_eq!(table.lookup_or_insert(&key, 99), Some(0));
        assert_eq!(table.lookup_or_insert(&key, 100), Some(0));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn distinct_keys_get_distinct_indices() {
        let mut table = InlineDedupTable::new(64);
        for seed in 0..32u64 {
            let key = make_key(seed);
            assert_eq!(
                table.lookup_or_insert(&key, seed as u32),
                None,
                "seed {seed} unexpectedly matched an existing slot"
            );
        }
        for seed in 0..32u64 {
            let key = make_key(seed);
            // Either matches the original index, or matched no slot
            // because cuckoo placement dropped one. Both are valid;
            // we only require correctness of returned indices.
            if let Some(idx) = table.lookup_only(&key) {
                assert_eq!(idx, seed as u32, "seed {seed} returned wrong index");
            }
        }
    }

    /// Streaming randomized test: build a key stream from a smaller
    /// pool of unique patterns, run both the reference packed-key
    /// sort dedup and [`InlineDedupTable`], and verify the resulting
    /// `(key, count)` distributions agree on the invariants the
    /// production pipeline depends on.
    ///
    /// # Invariants checked
    ///
    /// The libjxl cuckoo table can silently fail to insert a key when
    /// both its hash slots are already occupied by other keys (see
    /// `lookup_or_insert` doc and `enc_ma.cc:632`). Such a key is
    /// still emitted as a unique row — it just cannot serve as a
    /// merge target for future identical keys. Those future identical
    /// keys may either land on the now-empty slot that the original
    /// silent-drop missed, or trigger their own silent drop, ending
    /// up as additional unique rows with the same key. The downstream
    /// tree builder is order- and split-insensitive: it cares only
    /// about `total_count(k) = Σ count` summed over all rows with
    /// key `k`. So the contract this test enforces is:
    ///
    /// 1. **No rows lost.** Sum of counts equals input length on both
    ///    paths.
    /// 2. **Same unique-key set.** Both paths emit identical sets of
    ///    distinct keys (the cuckoo path may split a key across rows,
    ///    but the *set of keys observed* is the same).
    /// 3. **Per-key total preserved.** For every distinct key, the
    ///    sort path's count equals the sum of the cuckoo path's
    ///    counts across all rows carrying that key.
    fn assert_dedup_agrees_with_sort(keys: &[[u8; KEY_BYTES]], case_label: &str) {
        let sort_out = reference_dedup(keys);
        let inline_out = run_inline_dedup(keys);

        let sort_total: u32 = sort_out.iter().map(|(_, c)| *c).sum();
        let inline_total: u32 = inline_out.iter().map(|(_, c)| *c).sum();
        assert_eq!(
            sort_total,
            keys.len() as u32,
            "{case_label}: sort dedup lost rows ({sort_total} vs {})",
            keys.len()
        );
        assert_eq!(
            inline_total,
            keys.len() as u32,
            "{case_label}: inline dedup lost rows ({inline_total} vs {})",
            keys.len()
        );

        use std::collections::{HashMap, HashSet};
        let sort_keys: HashSet<[u8; KEY_BYTES]> = sort_out.iter().map(|(k, _)| *k).collect();
        let inline_keys: HashSet<[u8; KEY_BYTES]> = inline_out.iter().map(|(k, _)| *k).collect();
        assert_eq!(
            sort_keys,
            inline_keys,
            "{case_label}: distinct-key sets differ (sort {}, inline {})",
            sort_keys.len(),
            inline_keys.len(),
        );

        let sort_totals: HashMap<[u8; KEY_BYTES], u32> =
            sort_out.iter().map(|(k, c)| (*k, *c)).collect();
        let mut inline_totals: HashMap<[u8; KEY_BYTES], u32> = HashMap::new();
        for (k, c) in &inline_out {
            *inline_totals.entry(*k).or_insert(0) += *c;
        }
        for (k, expected) in &sort_totals {
            let actual = inline_totals.get(k).copied().unwrap_or(0);
            assert_eq!(
                *expected, actual,
                "{case_label}: per-key total differs for {k:?} (sort {expected}, inline {actual})",
            );
        }
    }

    /// Strict-equality counterpart for cases where the test author
    /// has independently established that no cuckoo silent-drop will
    /// trigger (e.g., small key universe with provably non-conflicting
    /// hash positions, or all-duplicate stream). Use sparingly — most
    /// random streams should use [`assert_dedup_agrees_with_sort`].
    #[allow(dead_code)]
    fn assert_dedup_byte_identical(keys: &[[u8; KEY_BYTES]], case_label: &str) {
        let sort_out = reference_dedup(keys);
        let inline_out = run_inline_dedup(keys);
        assert_multiset_eq(&sort_out, &inline_out, case_label);
    }

    #[test]
    fn agrees_on_all_unique_stream() {
        let keys: Vec<[u8; KEY_BYTES]> = (0..256u64).map(make_key).collect();
        assert_dedup_agrees_with_sort(&keys, "all unique, n=256");
    }

    #[test]
    fn agrees_on_all_duplicate_stream() {
        let key = make_key(0xCAFE);
        let keys = vec![key; 1024];
        assert_dedup_agrees_with_sort(&keys, "all duplicate, n=1024");
    }

    #[test]
    fn agrees_on_half_duplicate_stream() {
        // 50 % duplicates: 512 distinct keys, each repeated twice.
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::with_capacity(1024);
        for seed in 0..512u64 {
            keys.push(make_key(seed));
            keys.push(make_key(seed));
        }
        assert_dedup_agrees_with_sort(&keys, "50% duplicate, n=1024");
    }

    #[test]
    fn agrees_on_heavy_duplicate_stream() {
        // 90 % duplicates: 102 distinct keys repeated to 1024 entries.
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::with_capacity(1024);
        let unique = 102usize;
        for i in 0..1024usize {
            keys.push(make_key((i % unique) as u64));
        }
        assert_dedup_agrees_with_sort(&keys, "~90% duplicate, n=1024");
    }

    #[test]
    fn agrees_on_interleaved_pattern() {
        // Round-robin between 8 patterns × 256 reps = 2048 inputs,
        // 8 unique outputs. Stress-tests probe-then-merge ordering.
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::with_capacity(2048);
        for rep in 0..256u64 {
            for pat in 0..8u64 {
                let _ = rep;
                keys.push(make_key(pat));
            }
        }
        assert_dedup_agrees_with_sort(&keys, "interleaved, 8 patterns × 256 reps");
    }

    #[test]
    fn agrees_on_random_seed_42() {
        let n = 4096usize;
        let unique = 1024usize;
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::with_capacity(n);
        let mut rng_state: u64 = 42;
        for _ in 0..n {
            rng_state = rng_state.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            let pattern = rng_state as usize % unique;
            keys.push(make_key(pattern as u64));
        }
        assert_dedup_agrees_with_sort(&keys, "random seed 42, n=4096, ~75% dup");
    }

    #[test]
    fn agrees_on_random_seed_2026() {
        let n = 8192usize;
        let unique = 2048usize;
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::with_capacity(n);
        let mut rng_state: u64 = 2026;
        for _ in 0..n {
            rng_state = rng_state.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            let pattern = rng_state as usize % unique;
            keys.push(make_key(pattern as u64));
        }
        assert_dedup_agrees_with_sort(&keys, "random seed 2026, n=8192, ~75% dup");
    }

    #[test]
    fn agrees_on_low_entropy_keys() {
        // Small alphabet per byte stresses the hash distribution.
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::new();
        for a in 0u8..8 {
            for b in 0u8..8 {
                for c in 0u8..8 {
                    let mut k = [0u8; KEY_BYTES];
                    k[0] = a;
                    k[1] = b;
                    k[2] = c;
                    // Repeat each 4× so the dedup has work to do.
                    keys.push(k);
                    keys.push(k);
                    keys.push(k);
                    keys.push(k);
                }
            }
        }
        assert_dedup_agrees_with_sort(&keys, "low-entropy 8^3 × 4 reps");
    }

    #[test]
    fn agrees_on_realistic_photo_like_stream() {
        // Mimic photo-data shape: a clutch of nearby pixels share
        // residual tokens but differ in one or two property bytes,
        // so the cuckoo table sees long runs of near-misses with
        // occasional exact matches.
        let mut keys: Vec<[u8; KEY_BYTES]> = Vec::with_capacity(16_384);
        let mut rng_state: u64 = 0xc0ffee;
        for _ in 0..4096 {
            rng_state = rng_state.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            let cluster = rng_state >> 32;
            let base = make_key(cluster);
            for jitter in 0..4u64 {
                let mut k = base;
                // Perturb 2 bytes per jitter → 4 keys per cluster,
                // some hitting the same packed key on collision.
                let off1 = (jitter * 3) as usize % KEY_BYTES;
                let off2 = (jitter * 7) as usize % KEY_BYTES;
                k[off1] = k[off1].wrapping_add(jitter as u8);
                k[off2] = k[off2].wrapping_add((jitter * 13) as u8);
                keys.push(k);
            }
        }
        assert_dedup_agrees_with_sort(&keys, "photo-like clusters, n=16384");
    }

    #[test]
    fn capacity_rounds_up_to_pow2() {
        // Capacity = next_pow2(max(16, ceil(expected * 3 / 2))).
        assert_eq!(InlineDedupTable::new(0).capacity(), 16);
        assert_eq!(InlineDedupTable::new(1).capacity(), 16);
        // 10 × 3 = 30; ceil(30 / 2) = 15; max(16, 15) = 16.
        assert_eq!(InlineDedupTable::new(10).capacity(), 16);
        // 11 × 3 = 33; ceil(33 / 2) = 17; next_pow2(17) = 32.
        assert_eq!(InlineDedupTable::new(11).capacity(), 32);
        // 12 × 3 = 36; ceil(36 / 2) = 18; next_pow2(18) = 32.
        assert_eq!(InlineDedupTable::new(12).capacity(), 32);
        // 1000 × 3 = 3000; ceil(3000 / 2) = 1500; next_pow2(1500) = 2048.
        assert_eq!(InlineDedupTable::new(1000).capacity(), 2048);
    }

    /// Sentinel-rejection test runs only in debug builds because the
    /// production hot path keeps `debug_assert_ne!` to stay branch-free
    /// in release. The misuse can never trigger in production code
    /// (sample counts are bounded well below `u32::MAX`); the test
    /// documents the contract for future call sites.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "next_index = SLOT_EMPTY")]
    fn rejects_sentinel_index_on_insert() {
        let mut table = InlineDedupTable::new(4);
        let key = make_key(1);
        let _ = table.lookup_or_insert(&key, u32::MAX);
    }
}
