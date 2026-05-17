// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Phase 4 of issue #41 — fused-pass `AddSample` primitive.
//!
//! # Motivation
//!
//! Phases 1-3 of issue #41 all share a per-sample shape:
//!
//! ```text
//!   Pass A:  compute residual tokens          → local_tokens[..num_pred]
//!            (also extra_bits[..num_pred])
//!   Pass B:  compute ref-channel properties   → local_ref_props[..4*max_refs]
//!   Pass C:  pack_local_key_phase3            → reads A + B, writes [u8; 64]
//!   Pass D:  Hash1(key) / Hash2(key)          → re-walks the packed key
//! ```
//!
//! Passes C and D each re-walk the same bytes A and B wrote, defeating the
//! cache-hot-write win that motivated Phases 2 and 3. Phase 4 collapses
//! C and D into the *compute pass itself*: as each byte is produced, it
//! is folded into the running hash state and copied into the canonical
//! key buffer in the same instruction stream.
//!
//! # Shape
//!
//! [`FusedHashKeyBuilder`] is a small `Copy` struct holding:
//!
//!  * `h1: u64` — libjxl `Hash1` accumulator (mul-add fold).
//!  * `h2: u64` — libjxl `Hash2` accumulator (mul-xor fold).
//!  * `canonical_key: [u8; KEY_BYTES]` — the same canonical key Phase 3's
//!    [`super::inline_dedup_table::pack_local_key_phase3`] produces.
//!  * `off: usize` — next write cursor into `canonical_key`.
//!
//! Callers feed bytes via three methods (deliberately small surface so the
//! caller's hot loop stays linear and LLVM keeps the state in registers):
//!
//!  * [`add_token_pair`] — push `(token, ebits)` for one predictor.
//!  * [`add_prop_i32`]   — push a single i32 property (4 bytes LE).
//!  * [`finalize`]       — return `(canonical_key, h1, h2)` so the caller
//!    can compute fingerprint + probe positions exactly as Phase 3 does.
//!
//! The hash math is bit-identical to
//! [`super::inline_dedup_table::InlineDedupTable::raw_hash1`] and
//! [`super::inline_dedup_table::InlineDedupTable::raw_hash2`]:
//!
//!  * `h1 := h1 * 0x1e35a7bd + byte`              (mul-add)
//!  * `h2 := h2 * 0x1e35a7bd1e35a7bd ^ byte`      (mul-xor)
//!
//! Both folds iterate over **exactly the canonical key bytes** the existing
//! primitive iterates over, in the same order, so an end-to-end
//! `FusedHashKeyBuilder` then `InlineDedupTable::lookup_only(&canonical_key)`
//! round-trip is byte-equivalent to passing the same key bytes through
//! Phase 3's `pack_local_key_phase3` then `lookup_or_insert`.
//!
//! # Cache cost vs Phase 3 (hypothesis to validate in microbench)
//!
//! Phase 3 (`gather_sim_phase3`):
//!  1. Compute 14 × 2 token/ebits bytes into 32-byte stack scratch
//!     (28 bytes used, 4 padding). 1 sequential write pass.
//!  2. Compute ~4 × 8 = 32 bytes of ref-channel properties into another
//!     stack scratch. 1 sequential write pass.
//!  3. `pack_local_key_phase3` reads (1) and (2), writes a third 64-byte
//!     buffer. 2 sequential reads + 1 sequential write.
//!  4. `raw_hash1` / `raw_hash2` each read the 64-byte buffer. 2 more
//!     sequential reads.
//!
//! Total: 5 read passes + 3 write passes over ~60 bytes/sample.
//!
//! Phase 4 (`FusedHashKeyBuilder`):
//!  1. Compute 14 × 2 token/ebits bytes; fold into h1/h2 + write to
//!     `canonical_key` in the same instruction. 1 write pass + 0
//!     reads (bytes go straight from register → hash + memory).
//!  2. Compute ~4 × 8 ref-channel property bytes; same fold-and-write.
//!
//! Total: 1 write pass + 0 read passes for the hash; the verify path
//! on cuckoo-slot match still reads the existing `unique_keys[i]`
//! cacheline. Read pressure on the gather-hot side drops from 5 passes
//! to 0, and the local scratch buffers can be eliminated entirely if
//! Chunk 2's wiring removes the dependents.
//!
//! # libjxl references
//!
//! * `lib/jxl/modular/encoding/enc_ma.cc:657-686` — `Hash1`, `Hash2`.
//! * `lib/jxl/modular/encoding/enc_ma.cc:711-737` — `AddSample` (push +
//!   probe + pop_back-on-hit).
//!
//! # Chunk 1 contract
//!
//! This file ships ONLY the primitive + microbench + unit tests. Chunk 2
//! wires it into [`super::tree_learn::gather_channel_samples`].
//!
//! # Chunk 1 results (NEGATIVE, 2026-05-17)
//!
//! The microbench in `benches/dedup_samples_strategies.rs` (groups
//! `dedup_photo_full_*` and `dedup_full_*`) shows that this primitive
//! is **slower** than Phase 3's `pack_local_key_phase3` +
//! `InlineDedupTable::lookup_or_insert` by 10-25 % on every cell
//! measured (8 cells: 200K / 1.35M samples × dup 300/600/800 ×
//! photo-like + synthetic). See `benchmarks/inline_addsample_microbench_2026-05-17.{txt,meta}`
//! and `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/lossless_phase4_inline_addsample_2026-05-17.md`
//! for the full numbers and root-cause hypotheses.
//!
//! The likely causes are (a) loss of vectorization when byte-write and
//! hash-fold interleave inside the same loop body and (b) the trailing
//! zero-byte fold in [`FusedHashKeyBuilder::finalize`] adding 8-32 extra
//! multiplies per sample to match `InlineDedupTable::raw_hash1` over the
//! full `[u8; 64]` buffer.
//!
//! **The primitive ships anyway** as a documented negative result: it
//! is correct (25 unit tests pass, byte-equivalent to Phase 3 for every
//! seed tested) and the future agent who picks up issue #41 may try a
//! bulk-byte API (`add_token_slice`/`add_prop_slice`) that preserves
//! vectorization while still folding the hash inline. Do NOT wire this
//! current shape into `gather_channel_samples`.

use super::inline_dedup_table::KEY_BYTES;

/// libjxl `Hash1` multiplier (`enc_ma.cc:658`).
const HASH1_CONST: u64 = 0x1e35a7bd;
/// libjxl `Hash2` multiplier (`enc_ma.cc:673`).
const HASH2_CONST: u64 = 0x1e35a7bd1e35a7bd;

/// Fused canonical-key + hash builder for the Phase 4 gather-time
/// dedup probe.
///
/// The struct is `Copy` so the caller's hot loop can pass it by value
/// without forcing a stack spill — LLVM keeps the small `(h1, h2, off)`
/// state in registers and inlines the byte-fold into the surrounding
/// pixel loop.
///
/// The `canonical_key` field is always written speculatively: even on a
/// cuckoo-table hit we'd otherwise need to compare against the historical
/// `unique_keys[i]` row, so the canonical key must exist somewhere. We
/// keep it as a stack-local buffer in the builder; the caller decides
/// whether to copy it into `unique_keys` (on a miss) or discard it (on a
/// hit).
///
/// # Capacity
///
/// `canonical_key` is sized to [`KEY_BYTES`] (64). Worst case:
/// 14 predictors × 2 bytes + 16 properties × 4 bytes = 92 bytes. That
/// overflows the budget; [`add_token_pair`] / [`add_prop_i32`] return
/// `Err(BuilderOverflow)` rather than silently truncating. The caller
/// must therefore either:
///
///  * Pre-check that `2 * num_pred + 4 * num_props_hashed ≤ KEY_BYTES`
///    at backend-selection time (the same gate Phase 3's dispatcher
///    uses in `gather_samples_strided_with_dedup_backend`), or
///  * Catch the overflow and fall back to a non-dedup gather path.
///
/// The Chunk 2 wiring will use the pre-check pattern so the overflow
/// branch never fires in production.
#[derive(Clone, Copy)]
pub struct FusedHashKeyBuilder {
    /// Running libjxl `Hash1` accumulator. Seed = `HASH1_CONST`.
    h1: u64,
    /// Running libjxl `Hash2` accumulator. Seed = `HASH2_CONST`.
    h2: u64,
    /// Canonical key bytes accumulated so far. Trailing slots [off..]
    /// stay zero, which matches libjxl's implicit zero-padding for
    /// short keys (every sample at a given configuration has the same
    /// `off` so trailing zeros don't perturb equality).
    canonical_key: [u8; KEY_BYTES],
    /// Next write position into `canonical_key`.
    off: usize,
}

/// Returned by [`FusedHashKeyBuilder::add_token_pair`] /
/// [`FusedHashKeyBuilder::add_prop_i32`] when the canonical-key buffer
/// would overflow [`KEY_BYTES`]. See the [`FusedHashKeyBuilder`] docs
/// for the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuilderOverflow;

/// Output of [`FusedHashKeyBuilder::finalize`].
///
/// The two raw hash values are returned as-is (NOT masked to the slot
/// table size) so the caller can derive both the slot positions
/// (`(h >> 16) & mask`) and the 64-bit fingerprint (XOR of `h1` and
/// `h2.rotate_left(32)`) without re-walking the canonical key.
#[derive(Clone, Copy)]
pub struct FinalizedKey {
    /// Canonical packed key, byte-identical to Phase 3's
    /// `pack_local_key_phase3` output for the same input sequence.
    pub canonical_key: [u8; KEY_BYTES],
    /// Raw `Hash1` fold (mul-add over the canonical key bytes).
    pub raw_h1: u64,
    /// Raw `Hash2` fold (mul-xor over the canonical key bytes).
    pub raw_h2: u64,
    /// Number of bytes actually written to `canonical_key`. Trailing
    /// bytes [bytes_written..KEY_BYTES] are zero. Exposed so the
    /// fingerprint-cache verify path can compare prefixes only.
    pub bytes_written: usize,
}

impl Default for FusedHashKeyBuilder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl FusedHashKeyBuilder {
    /// Initialize a fresh builder. Seeds match libjxl `Hash1`/`Hash2`
    /// (`enc_ma.cc:659, 674`).
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            h1: HASH1_CONST,
            h2: HASH2_CONST,
            canonical_key: [0u8; KEY_BYTES],
            off: 0,
        }
    }

    /// Append a `(token, ebits)` pair from one predictor. Writes 2 bytes
    /// to the canonical key and folds both into `h1` / `h2`. Returns
    /// `Err(BuilderOverflow)` if the buffer would exceed [`KEY_BYTES`].
    ///
    /// Bytes are folded in the order `[token, ebits]`, matching
    /// `pack_local_key_phase3` (`inline_dedup_table.rs:447-450`).
    #[inline(always)]
    pub fn add_token_pair(&mut self, token: u8, ebits: u8) -> Result<(), BuilderOverflow> {
        if self.off + 2 > KEY_BYTES {
            return Err(BuilderOverflow);
        }
        // Write to canonical_key + fold into both hashes in one go;
        // both reads of the byte hit the register the caller passed in.
        self.canonical_key[self.off] = token;
        self.canonical_key[self.off + 1] = ebits;
        self.h1 = self.h1.wrapping_mul(HASH1_CONST).wrapping_add(token as u64);
        self.h2 = self.h2.wrapping_mul(HASH2_CONST) ^ (token as u64);
        self.h1 = self.h1.wrapping_mul(HASH1_CONST).wrapping_add(ebits as u64);
        self.h2 = self.h2.wrapping_mul(HASH2_CONST) ^ (ebits as u64);
        self.off += 2;
        Ok(())
    }

    /// Append a single i32 property value. Writes 4 little-endian bytes
    /// to the canonical key and folds them into `h1` / `h2`. Returns
    /// `Err(BuilderOverflow)` if the buffer would exceed [`KEY_BYTES`].
    ///
    /// Bytes are folded LE, matching `pack_local_key_phase3`
    /// (`inline_dedup_table.rs:463-468`).
    #[inline(always)]
    pub fn add_prop_i32(&mut self, value: i32) -> Result<(), BuilderOverflow> {
        if self.off + 4 > KEY_BYTES {
            return Err(BuilderOverflow);
        }
        let b = value.to_le_bytes();
        self.canonical_key[self.off] = b[0];
        self.canonical_key[self.off + 1] = b[1];
        self.canonical_key[self.off + 2] = b[2];
        self.canonical_key[self.off + 3] = b[3];
        // Four byte-folds — unrolled so LLVM can interleave the
        // multiply-add and multiply-xor chains across registers.
        self.h1 = self.h1.wrapping_mul(HASH1_CONST).wrapping_add(b[0] as u64);
        self.h2 = self.h2.wrapping_mul(HASH2_CONST) ^ (b[0] as u64);
        self.h1 = self.h1.wrapping_mul(HASH1_CONST).wrapping_add(b[1] as u64);
        self.h2 = self.h2.wrapping_mul(HASH2_CONST) ^ (b[1] as u64);
        self.h1 = self.h1.wrapping_mul(HASH1_CONST).wrapping_add(b[2] as u64);
        self.h2 = self.h2.wrapping_mul(HASH2_CONST) ^ (b[2] as u64);
        self.h1 = self.h1.wrapping_mul(HASH1_CONST).wrapping_add(b[3] as u64);
        self.h2 = self.h2.wrapping_mul(HASH2_CONST) ^ (b[3] as u64);
        self.off += 4;
        Ok(())
    }

    /// Number of bytes accumulated so far (sum of `2 *
    /// add_token_pair_calls + 4 * add_prop_i32_calls`). Tests use this
    /// to assert layout symmetry with Phase 3.
    #[inline(always)]
    pub fn bytes_written(&self) -> usize {
        self.off
    }

    /// Finalize the builder and return the canonical key + raw hash
    /// folds. The builder is consumed (taken by value) to discourage
    /// reuse — a second sample needs a fresh builder so its hash state
    /// starts at the libjxl seeds.
    ///
    /// # Trailing zero fold
    ///
    /// The canonical key is a fixed [`KEY_BYTES`]-byte buffer, so
    /// [`super::inline_dedup_table::InlineDedupTable::raw_hash1`] and
    /// `raw_hash2` fold ALL 64 bytes — including the [self.off..]
    /// trailing zeros from short keys. For byte-equivalence with the
    /// existing primitive, finalize folds the trailing zeros now.
    ///
    /// Each zero-byte fold reduces to `h = h * C` (mul-add: `h * C + 0`;
    /// mul-xor: `h * C ^ 0`), so the loop is just `KEY_BYTES - off`
    /// multiplications per hash. At `num_pred = 14, num_props = 7` the
    /// off is 28+28 = 56 and the trailing fold is 8 iterations.
    ///
    /// # Why not skip the zero-fold?
    ///
    /// Within a single gather call site `off` is constant across all
    /// samples, so two samples with equal canonical keys would still
    /// hash to the same slot (the trailing `* C^k` is a uniform
    /// transform). We fold the zeros anyway so the fingerprint matches
    /// the existing `InlineDedupTable` primitive bit-for-bit. That
    /// keeps Chunk 2's wiring drop-in: the same `InlineDedupTable` can
    /// service Phase 3 and Phase 4 callers interchangeably without a
    /// per-backend slot-table separation.
    #[inline]
    pub fn finalize(mut self) -> FinalizedKey {
        // Fold trailing zeros so the hash matches raw_hash1/2 over the
        // full canonical_key buffer. Loop body specializes to
        //   h1 := h1 * HASH1_CONST
        //   h2 := h2 * HASH2_CONST
        // because every byte is zero.
        for _ in self.off..KEY_BYTES {
            self.h1 = self.h1.wrapping_mul(HASH1_CONST);
            self.h2 = self.h2.wrapping_mul(HASH2_CONST);
        }
        FinalizedKey {
            canonical_key: self.canonical_key,
            raw_h1: self.h1,
            raw_h2: self.h2,
            bytes_written: self.off,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::inline_dedup_table::{
        InlineDedupTable, LocalKeyPackResult, pack_local_key_phase3,
    };

    /// Phase 3's `raw_hash1` — duplicated here so the test asserts an
    /// independent computation, not a tautology against the same code.
    fn reference_hash1(key: &[u8; KEY_BYTES]) -> u64 {
        let mut h: u64 = HASH1_CONST;
        for &b in key.iter() {
            h = h.wrapping_mul(HASH1_CONST).wrapping_add(b as u64);
        }
        h
    }

    /// Phase 3's `raw_hash2` — same independent-computation rationale.
    fn reference_hash2(key: &[u8; KEY_BYTES]) -> u64 {
        let mut h: u64 = HASH2_CONST;
        for &b in key.iter() {
            h = h.wrapping_mul(HASH2_CONST) ^ (b as u64);
        }
        h
    }

    /// Build a `(local_tokens, local_ebits, props, ref_props, properties_to_hash)`
    /// tuple from a 64-bit seed in a way that produces clustered, real-photo-like
    /// samples: bytes vary by 1-2 LSBs across the same seed family, with
    /// occasional exact matches.
    fn sample_from_seed(
        seed: u64,
        num_pred: usize,
        num_props: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<i32>, Vec<u8>) {
        let mut tokens = Vec::with_capacity(num_pred);
        let mut ebits = Vec::with_capacity(num_pred);
        let mut props = Vec::with_capacity(num_props);
        let mut s = seed;
        // Splitmix-style: rotate state per byte so adjacent seeds produce
        // bytes that differ in 1-3 bit positions (clustered, not random).
        for _ in 0..num_pred {
            s = s.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0xdeadbeef);
            tokens.push((s >> 32) as u8 & 0x3f); // token alphabet 0..63
            s = s.wrapping_mul(0x9e3779b97f4a7c15);
            ebits.push((s >> 32) as u8 & 0x0f); // ebits 0..15
        }
        for _ in 0..num_props {
            s = s.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
            props.push((s >> 16) as i32);
        }
        let properties_to_hash: Vec<u8> = (0..num_props as u8).collect();
        (tokens, ebits, props, properties_to_hash)
    }

    #[test]
    fn empty_builder_finalizes_to_full_zero_key_hash() {
        let f = FusedHashKeyBuilder::new().finalize();
        assert_eq!(f.bytes_written, 0);
        assert_eq!(f.canonical_key, [0u8; KEY_BYTES]);
        // finalize folds the [0..KEY_BYTES] zero tail, so the hash equals
        // an all-zero-key reference fold — NOT the bare seed constant.
        // This is what keeps the fingerprint identical to Phase 3's
        // raw_hash1/2 over the full [u8; KEY_BYTES] buffer.
        assert_eq!(f.raw_h1, reference_hash1(&[0u8; KEY_BYTES]));
        assert_eq!(f.raw_h2, reference_hash2(&[0u8; KEY_BYTES]));
    }

    #[test]
    fn single_token_pair_matches_reference_hash() {
        let mut b = FusedHashKeyBuilder::new();
        b.add_token_pair(0x42, 0x07).unwrap();
        let f = b.finalize();
        assert_eq!(f.bytes_written, 2);
        // Canonical key: [0x42, 0x07, 0, 0, ...].
        assert_eq!(f.canonical_key[0], 0x42);
        assert_eq!(f.canonical_key[1], 0x07);
        for &b in &f.canonical_key[2..] {
            assert_eq!(b, 0);
        }
        // Independent hash recomputation must agree.
        assert_eq!(f.raw_h1, reference_hash1(&f.canonical_key));
        assert_eq!(f.raw_h2, reference_hash2(&f.canonical_key));
    }

    #[test]
    fn single_prop_i32_matches_reference_hash() {
        let mut b = FusedHashKeyBuilder::new();
        b.add_prop_i32(-12345).unwrap();
        let f = b.finalize();
        assert_eq!(f.bytes_written, 4);
        // LE encoding of -12345 = 0xCFFC_FFFF.
        let want = (-12345i32).to_le_bytes();
        for (i, &w) in want.iter().enumerate() {
            assert_eq!(f.canonical_key[i], w, "byte {i}");
        }
        for &b in &f.canonical_key[4..] {
            assert_eq!(b, 0);
        }
        assert_eq!(f.raw_h1, reference_hash1(&f.canonical_key));
        assert_eq!(f.raw_h2, reference_hash2(&f.canonical_key));
    }

    #[test]
    fn mixed_token_and_prop_matches_reference_hash() {
        let mut b = FusedHashKeyBuilder::new();
        b.add_token_pair(0x10, 0x03).unwrap();
        b.add_prop_i32(42).unwrap();
        b.add_token_pair(0xff, 0x00).unwrap();
        b.add_prop_i32(-1).unwrap();
        let f = b.finalize();
        assert_eq!(f.bytes_written, 12);
        // Canonical key bytes 0..12:
        // [tok0, ebit0, ...prop0_le_bytes..., tok1, ebit1, ...prop1_le_bytes...]
        let expected: [u8; 12] = [0x10, 0x03, 42, 0, 0, 0, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff];
        for (i, &w) in expected.iter().enumerate() {
            assert_eq!(f.canonical_key[i], w, "byte {i}");
        }
        assert_eq!(f.raw_h1, reference_hash1(&f.canonical_key));
        assert_eq!(f.raw_h2, reference_hash2(&f.canonical_key));
    }

    #[test]
    fn overflow_returns_err_and_does_not_corrupt_state() {
        let mut b = FusedHashKeyBuilder::new();
        // 14 predictors × 2 + 11 props × 4 = 72 bytes, exceeds 64.
        for i in 0..14 {
            b.add_token_pair(i as u8, (i >> 4) as u8).unwrap();
        }
        // 14 × 2 = 28 bytes used. 9 more props = 36 bytes → 64 total: OK.
        for i in 0..9 {
            b.add_prop_i32(i as i32).unwrap();
        }
        assert_eq!(b.bytes_written(), 64);
        // 10th prop pushes off to 68 > 64: overflow.
        assert_eq!(b.add_prop_i32(999).unwrap_err(), BuilderOverflow);
        // State unchanged — still 64 bytes written.
        assert_eq!(b.bytes_written(), 64);
        // Finalize still works.
        let f = b.finalize();
        assert_eq!(f.bytes_written, 64);
        assert_eq!(f.raw_h1, reference_hash1(&f.canonical_key));
    }

    #[test]
    fn token_overflow_at_boundary() {
        let mut b = FusedHashKeyBuilder::new();
        for _ in 0..32 {
            b.add_token_pair(0xab, 0xcd).unwrap();
        }
        // 32 × 2 = 64 bytes used; one more token pair would push to 66.
        assert_eq!(b.add_token_pair(0x00, 0x00).unwrap_err(), BuilderOverflow);
        assert_eq!(b.bytes_written(), 64);
    }

    /// Critical invariant test — proves the FusedHashKeyBuilder produces
    /// bit-identical output to Phase 3's `pack_local_key_phase3` +
    /// `raw_hash1`/`raw_hash2` for the same input sequence.
    fn assert_fused_matches_pack_for_seed(seed: u64, num_pred: usize, num_props: usize) {
        let (tokens, ebits, props, properties_to_hash) =
            sample_from_seed(seed, num_pred, num_props);

        // Build via Phase 3 packing.
        let phase3_key = match pack_local_key_phase3(
            &tokens,
            &ebits,
            &props,
            &[], // no ref_props in this test
            &properties_to_hash,
            num_props, // num_properties_base — all props are "base"
        ) {
            LocalKeyPackResult::Packed(k) => k,
            LocalKeyPackResult::Overflow => {
                panic!(
                    "test setup error: seed {seed} num_pred {num_pred} num_props {num_props} \
                     overflows the 64-byte budget"
                );
            }
        };

        // Build via Phase 4 fused builder.
        let mut b = FusedHashKeyBuilder::new();
        for (&t, &e) in tokens.iter().zip(ebits.iter()) {
            b.add_token_pair(t, e).unwrap();
        }
        // Properties_to_hash in this test = 0..num_props in order, so we
        // can just iterate the local props array. (The production gather
        // loop will iterate properties_to_hash itself; the contract is
        // that the caller controls the order, the builder just hashes
        // whatever bytes the caller hands it.)
        for &v in props.iter() {
            b.add_prop_i32(v).unwrap();
        }
        let phase4 = b.finalize();

        // 1. Canonical keys must be byte-identical.
        assert_eq!(
            phase4.canonical_key, phase3_key,
            "canonical key mismatch (seed {seed}, num_pred {num_pred}, num_props {num_props})"
        );

        // 2. Phase 4's raw hashes must match an independent recomputation
        //    over the same key bytes.
        assert_eq!(
            phase4.raw_h1,
            reference_hash1(&phase3_key),
            "h1 mismatch (seed {seed})"
        );
        assert_eq!(
            phase4.raw_h2,
            reference_hash2(&phase3_key),
            "h2 mismatch (seed {seed})"
        );
    }

    #[test]
    fn fused_matches_pack_for_16_real_photo_seeds() {
        // 7 active properties + 14 candidate predictors — same shape as
        // the prod e7 gather (modulo ref-channel props, which we omit
        // for this test because pack_local_key_phase3 routes them
        // through a separate slice).
        let configs = [
            (0xc0ffee_u64, 14, 7),
            (0xdeadbeef, 14, 7),
            (0x12345678, 14, 7),
            (0x87654321, 14, 7),
            (0xabcdef01, 14, 7),
            (0x10fedcba, 14, 7),
            (0xfeedface, 14, 7),
            (0xbadc0ffe, 14, 7),
            (0xf00dbabe, 14, 7),
            (0x5a5a5a5a, 14, 7),
            (0x00000001, 14, 7),
            (0xffffffff, 14, 7),
            // Off-default predictor / property counts (smaller efforts).
            (0x11111111, 2, 4),
            (0x22222222, 5, 6),
            (0x33333333, 7, 9),
            // Edge: num_pred = 1 (single predictor, smallest valid).
            (0x44444444, 1, 4),
        ];
        for (seed, num_pred, num_props) in configs {
            assert_fused_matches_pack_for_seed(seed, num_pred, num_props);
        }
    }

    /// Cross-check end-to-end: 200 samples through Phase 3's
    /// `InlineDedupTable::lookup_or_insert` produce the same unique set
    /// whether keys are built via `pack_local_key_phase3` or via
    /// `FusedHashKeyBuilder`. Catches any divergence in the canonical
    /// key (since both paths share the same table).
    #[test]
    fn fused_then_table_matches_pack_then_table_unique_set() {
        let num_pred = 14;
        let num_props = 7;

        let mut phase3_table = InlineDedupTable::new(200);
        let mut phase4_table = InlineDedupTable::new(200);

        for i in 0..200u64 {
            // Cluster seeds so duplicates and near-duplicates exist.
            let cluster = i / 4; // 50 distinct clusters, 4 samples each
            let (tokens, ebits, props, properties_to_hash) =
                sample_from_seed(cluster, num_pred, num_props);

            let p3_key = match pack_local_key_phase3(
                &tokens,
                &ebits,
                &props,
                &[],
                &properties_to_hash,
                num_props,
            ) {
                LocalKeyPackResult::Packed(k) => k,
                LocalKeyPackResult::Overflow => unreachable!(),
            };

            let mut b = FusedHashKeyBuilder::new();
            for (&t, &e) in tokens.iter().zip(ebits.iter()) {
                b.add_token_pair(t, e).unwrap();
            }
            for &v in props.iter() {
                b.add_prop_i32(v).unwrap();
            }
            let p4 = b.finalize();

            // Tables receive byte-identical keys, so unique counts match.
            let p3_hit = phase3_table.lookup_or_insert(&p3_key, i as u32);
            let p4_hit = phase4_table.lookup_or_insert(&p4.canonical_key, i as u32);
            assert_eq!(
                p3_hit, p4_hit,
                "lookup_or_insert mismatch at sample {i}: phase3 = {p3_hit:?}, phase4 = {p4_hit:?}"
            );
        }
        assert_eq!(phase3_table.len(), phase4_table.len());
        assert_eq!(phase3_table.unique_keys(), phase4_table.unique_keys());
    }

    /// `Default` trait should give the same state as `new()`.
    #[test]
    fn default_matches_new() {
        let a = FusedHashKeyBuilder::default().finalize();
        let b = FusedHashKeyBuilder::new().finalize();
        assert_eq!(a.canonical_key, b.canonical_key);
        assert_eq!(a.raw_h1, b.raw_h1);
        assert_eq!(a.raw_h2, b.raw_h2);
        assert_eq!(a.bytes_written, b.bytes_written);
    }

    /// Per-byte fold ordering matters — pushing the same bytes in
    /// different orders must produce different hashes (the hash is
    /// position-dependent because each fold step multiplies first).
    #[test]
    fn byte_order_affects_hash() {
        let mut a = FusedHashKeyBuilder::new();
        a.add_token_pair(0x01, 0x02).unwrap();
        a.add_token_pair(0x03, 0x04).unwrap();
        let fa = a.finalize();

        let mut b = FusedHashKeyBuilder::new();
        b.add_token_pair(0x03, 0x04).unwrap();
        b.add_token_pair(0x01, 0x02).unwrap();
        let fb = b.finalize();

        assert_ne!(fa.canonical_key, fb.canonical_key);
        assert_ne!(fa.raw_h1, fb.raw_h1);
        assert_ne!(fa.raw_h2, fb.raw_h2);
    }
}
