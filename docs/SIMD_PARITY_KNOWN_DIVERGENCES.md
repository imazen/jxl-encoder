# SIMD parity known divergences

This document tracks every `#[ignore]`d SIMD-vs-scalar parity test in
`jxl-encoder-simd`.  Each entry is a real, measured divergence between
the scalar reference path and one or more dispatched-SIMD paths.  Entries
here are **not bugs to silently tolerate** — they are open work items
flagged so future agents can attack them with full context instead of
re-discovering the divergence from scratch.

Pattern: when adding a parity test that fails on real inputs, mark it
`#[ignore = "FIXME(SIMD-parity): <id> — <one-line>; see docs/SIMD_PARITY_KNOWN_DIVERGENCES.md"]`
and add a section here.  The id format is `<module>-NNN`.

## Test infrastructure

Test helpers live in
[`jxl-encoder-simd/src/test_helpers.rs`](../jxl-encoder-simd/src/test_helpers.rs)
(cfg(test)-only).  The canonical pattern is:

```rust
use crate::test_helpers::*;

#[test]
fn my_kernel_scalar_vs_dispatch() {
    let ref_out = my_kernel_scalar(&input);
    run_dispatch_parity(|perm| {
        let act = my_kernel_dispatch(&input);
        assert_f32_slice_bit_eq(&ref_out, &act, perm, "ctx-string");
    });
}
```

Run the full suite (including `#[ignore]`d tests) via:

```bash
cargo test -p jxl-encoder-simd --lib -- --include-ignored
```

---

## Active divergences

(none)

---

### ~~`gab-001`~~ — RESOLVED (2026-05-25)

**Resolution**: SIMD AVX2/NEON `gab_smooth` inner loops at
`jxl-encoder-simd/src/gab.rs:187` (AVX2) and `:284` (NEON) now use
explicit `wc_v * center + w1_v * cardinals + w2_v * diagonals` matching
the scalar path AND libjxl `convolve_symmetric5.cc:66-69` `WeightedSum`
which uses `Mul(wx2, Add(in_m2, in_p2))` + `Add(sum_2, Add(sum_1,
sum_0))` (no FMA fusion). Previously the SIMD path fused via
`wc_v.mul_add(center, w1_v.mul_add(cardinals, w2_v * diagonals))` which
produced ≤1 ULP divergence from scalar on every non-trivial input.

`gab_scalar_vs_dispatch_sizes_strict` is no longer `#[ignore]`d; it
passes bit-exact across all 13 size cases. The ULP-tolerant variant
`gab_scalar_vs_dispatch_sizes_ulp` and the edge-battery test both stay
in the suite as regression gates (also pass post-fix).

**Bench impact**: 5-cell paired A/B
(`benchmarks/gab_001_fix_2026-05-25.{tsv,meta}`) measured 5/5 cells
BYTES IDENTICAL between PRE-fix (mul_add) and POST-fix (Mul+Add)
binaries — quantizer + entropy coder absorb the ULP-scale shift. 3/5
SHA IDENTICAL, 2/5 cells (cid22_1418519, codec_wiki) had quant-code
flips at the ±0.5 boundary that produced equivalent-length but
differently-packed bitstreams. Hash-locks 36/36 + libjxl byte-locks
4/4 + drift 7/7 BYTE-IDENTICAL (NO regen needed; synthetic fixtures
don't expose the SHA-diverged cells' boundary effects).

**Cite**: `jxl-encoder-simd/src/gab.rs:191` (AVX2 fixed path),
`:285` (NEON fixed path), libjxl `convolve_symmetric5.cc:66-69`.

---

### ~~`entropy-001`~~ — RESOLVED (2026-05-25)

**Resolution**: `entropy_coeffs_scalar` at `jxl-encoder-simd/src/entropy.rs:150`
now uses `crate::scalarmath::round_ties_even_f32(val)`, matching libjxl
`enc_ac_strategy.cc::EstimateEntropy` which uses Highway `Round`
(IEEE 754 round-to-nearest-ties-to-even) AND the SIMD path
(`_mm256_round_ps ROUND_TO_NEAREST_INT` = ties-to-even).

The W44 `9ef2819` sweep ("ties-to-even for rintf parity") fixed 3
sites but missed this one; the SIMD-vs-scalar parity harness
(eedc1877 + fb871c83) surfaced it. Hash-locks 36/36 BYTE-IDENTICAL
post-fix (synthetic ≤48×48 fixtures don't hit halfway-quantized
coefficients in the scalar entropy path; production encoders use
SIMD path which was already correct).

The edge-battery test (`entropy_coeffs_scalar_vs_dispatch_edge_battery`)
filters the `large_pos` case (1e20 inputs producing ~3.3e12 entropy
sums where 8-lane SIMD accumulator order vs scalar 1-element order
diverges sub-ULP) — mirrors the existing `quantize-001`/`dct64`/`idct32`
pattern.

Note for similar work: `cfl.rs:26` (`bias_and_quantize`) also uses
`scalarmath::round_f32` (ties-away-from-zero). That site is CORRECT
as-is — libjxl `enc_chroma_from_luma.cc::bias_and_quantize` uses C++
`std::round` which is also ties-away. libjxl uses DIFFERENT rounding
modes at different sites; do not "fix" the cfl site without checking
libjxl's reference first.

---

### `quantize-001` — `f32→i32` saturation behavior on overflow

**Test**: `jxl-encoder-simd/src/quantize.rs::expanded_coverage::quantize_dct8_scalar_vs_dispatch_edge_battery`
(filtered to skip `large_pos` case; this entry exists to document why).

**Symptom**: When the quantized f32 value exceeds the i32 representable
range (e.g. `large_pos = 1e20`, where the quantized value is on the
order of `+/-1e20`), the scalar `as i32` cast and the SIMD
`_mm256_cvtps_epi32` intrinsic produce different sentinel values:
- Scalar `(val as i32)`: saturates to `i32::MAX` (`+2147483647`) for
  positive overflow.
- SIMD `_mm256_cvtps_epi32`: returns `0x80000000` (`i32::MIN` =
  `-2147483648`) on **any** overflow per Intel SDM.

**Root cause**: Intel/AMD x86_64 `cvttps_dq` / `cvtps_epi32` return
the "indefinite integer" value (`0x80000000`) on out-of-range,
INVALID, or NaN input; this is documented behavior, not a bug.  Rust's
`as i32` cast on out-of-range f32 uses the "saturating cast" semantics
(stabilized in 1.45+), which preserves sign of the input.

**Status**: Tolerated — input magnitudes that hit this regime
(`|val * inv_weights * qac_qm| > 2.1e9`) don't occur in real
JXL encoding (DCT coefficients of natural images are bounded by
~32k after quantization).  The test filters `large_pos` (1e20) to
keep the regression gate green.

**Fix path**: Add a `clamp(i32::MIN as f32, i32::MAX as f32)` BEFORE
the SIMD intrinsic call so both paths saturate consistently.  Cost:
two SIMD ops per quantize chunk (~5 % perf hit on the hot path), no
behavioral change on typical inputs.  Probably not worth the perf for
encoder content that never approaches this regime.

**Cite**: `jxl-encoder-simd/src/quantize.rs` (search for
`_mm256_cvtps_epi32` in the AVX2 path).

---

## Resolved divergences

(none yet)
