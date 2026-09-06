# JXL encoder ARM audit, 2026-09-06

Coverage so far: 17 kernel groups. The entropy group's original scalar arm
is a historical formula with different accumulation order, so it is not a
measurement of today's production fallback. Whole-image encoder measurements
and a corrected entropy-dispatch comparison are still pending.

Apple M4 Pro, macOS, Rust 1.98.0 / LLVM 22, runtime dispatch without
`target-cpu=native`, four build/Rayon/OMP threads, `nice -n 19`. The baseline
source is `7a7751e6`. All 16 other groups favored NEON; results and paired
confidence intervals are in [the baseline log](jxl-encoder-tiers.log).
Several microbenchmarks have substantial variance; exact ratios should not
be treated as stable hardware characteristics.

## Fixed bounds for 16-point DCT batches

The NEON 16-point batch helper stayed out of line with dynamically sized
slices and a row offset. Its assembly had 29 bounds-panic call sites and
598 lines. Passing each four-row batch as fixed `[f32; 64]` arrays exposes
the bounds inside the helper, leaving zero bounds-panic calls and 350 lines.
The arithmetic, lane order, scaling, and transposes are unchanged.

The 16×16 DCT mean moved from 581.8 ns to 413.9 ns; scalar measured 722.3 ns
and 767.4 ns respectively. These are separate builds/runs, not paired
before/after statistics. Compare the [before assembly](jxl-dct16-before.asm),
[after assembly](jxl-dct16-fixed.asm), and [new timings](jxl-dct16-fixed.log).
This change is `06b9751c`, also used by the 16×8 and 8×16 forward transforms.
Those rectangular shapes have not yet been remeasured after this change.

An earlier `#[inline(always)]` experiment changed no assembly: archmage-macros
0.9.28's `rite_single_impl` explicitly filters inline attributes and inserts
`#[inline]`. The fixed-array solution does not depend on stronger inlining.
The unsuccessful experiment's [measurement](jxl-dct16-inline.log) is retained.

Validation after the fixed-array change: 188 SIMD library tests passed,
clippy passed with `-D warnings`, and 60 encoder hash-lock integration tests
passed with no expected-value changes. The corresponding logs are committed
alongside this report. No whole-image speedup is claimed for this kernel change.
