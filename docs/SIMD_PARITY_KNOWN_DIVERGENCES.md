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

### `gab-001` — `gab_smooth` FMA association

**Test**: `jxl-encoder-simd/src/gab.rs::tests::gab_scalar_vs_dispatch_sizes_strict`

**Symptom**: SIMD (AVX2 and NEON) output differs from `gab_smooth_scalar`
by ≤1 ULP on most pixels.  Triggered by any non-trivial input distribution
(observed on `rand_a`, `rand_b`, `ramp` cases).

**Root cause**: The SIMD path fuses
`w_center*center + (w1*cardinals + w2*diagonals)` via two `mul_add`
intrinsics (1 rounding per FMA, 2 roundings total).  The scalar path
performs explicit `*` then `+` operations
(`w_center * center + w1 * (top + bottom + left + right) + w2 * (tl + tr + bl + br)`,
which is 4 muls + 6 adds with explicit Rust precedence = up to 10
roundings).  Cumulative rounding diverges in the LSB.

**Status**: Tolerated.  ULP-tolerant variant
`gab_scalar_vs_dispatch_sizes_ulp` ships as a regression gate (8 ULP +
1e-5 abs floor); the strict variant is `#[ignore]`d as documentation.

**Fix path**: Either (a) rewrite `gab_smooth_scalar` to use
`scalarmath::mul_add_f32` in the same chained pattern as the SIMD path
(would close the divergence at the cost of slightly slower scalar
fallback on non-FMA hardware), or (b) drop the SIMD `mul_add` calls in
favor of explicit `*`+`+` (would close the divergence at the cost of
slightly slower SIMD path on FMA hardware).  Either changes encoder
output bytes and requires hash-lock regen.

**Cite**: `jxl-encoder-simd/src/gab.rs:187` (SIMD mul_add fusion),
`:96` (scalar `*`+`+` expression).

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
