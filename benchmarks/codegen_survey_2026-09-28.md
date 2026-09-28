# Codegen survey — 2026-09-28 (r7900x, Zen4 x86_64)

Post-#110 sweep for codegen-level wins in `jxl-encoder`/`jxl-encoder-simd`:
callgrind attribution, `cargo asm`, target-cpu deltas, PGO, LTO, trampoline
elision. Host: r7900x (24c Zen4), release `--features parallel`. Workload:
`nature1024.ppm` lossy e7/d1.0 (default Zenjxl strategy) and
`nature_2048.png --size 512 --depth u8 --effort 9` lossless keep-best.

## Landed

**`50baf1ac` — transpose fusion in AVX2 2D DCT/IDCT kernels**
(`dct32/64`, `idct32/64`, 10 functions). The two-pass kernels zeroed two
scratch buffers and ran a full elementwise transpose between passes. Pass 1's
scatter now writes the transposed layout with contiguous 8-lane stores; where
a kernel began by un-transposing input or re-transposing between buffers,
`gather_col` on a transposed buffer is algebraically a contiguous slice of the
source — those became `f32x8::from_slice` loads.

- lossy e7 wall: 66.5 → 64.4 ms median (~−3%); memset Ir 509M → 369M
- 197/197 `jxl-encoder-simd` tests pass on x86-64; encoder lossless output
  byte-identical (sha256); lossy output size identical (122448 B)

**`a15b3a13` — same fusion for NEON + wasm128** (all 32/64 kernels; scalar
left as the parity reference). Bigger win at 4-wide: arm64 `dct_32x32`
937→733 ns/call (−22%), `dct_64x64` 6216→4859 ns/call (−22%), checksums
bit-identical, 194/194 tests pass on aarch64, wasm32 compiles clean.

## Measured and rejected

| Change | Result |
|---|---|
| `estimate_bits_u32_pair` (share SIMD prologue across the find_best_split L/R call pair) | −120M Ir on lossless-e9, **0 wall** — sweep is load/store-bound, not issue-bound. Reverted; 120-line macro body not worth ~0.4% Ir. |
| PGO (`-Cprofile-generate`/`-Cprofile-use`, 3-cell training set) | lossy −3%, **terminal-e9 +4% regression** — single-corpus PGO perturbs layout unfavorably. Needs a broad corpus to evaluate fairly; not adoptable as-is. |
| `lto=thin` + `codegen-units=1` | lossy ~−3%, lossless ±0 — marginal; real but small. |
| `target-cpu=native` / `x86-64-v3` | **no measurable delta** on any cell — the arcane dispatch layer already saturates the SIMD-critical kernels; the scalar remainder is integer/memory-bound. Also: valgrind cannot emulate the AVX-512 native binary (SIGILL) and `perf` counters are restricted on r7900x. |

## Clean bill of health

- `estimate_bits_u32_impl_v3` asm: tight interleaved AVX2 (2 accumulators,
  polynomial log2, no scalar fallthrough in the loop). ~15-instr constant-
  splat prologue per call is inherent to the `#[target_feature]` boundary —
  hoisting it requires making the *caller* `#[arcane]`/`#[rite]` per the
  crate convention; measured upside ~0.4% Ir, ~0 wall. Not worth it.
- `HashChain::find_match` asm: 386 instrs, zero panic/bounds branches in the
  chain-walk loop; ~20 field spills from the fat struct is the only bloat.
- `tree_learn::find_best_split` inner zip loops already autovectorize
  (documented asm post-mortems inline).
- No missing `#[cold]` candidates found: error paths are unit-enum
  constructions (`Error::Cancelled`), no formatting machinery inline in hot
  fns; `check_stop` is `#[inline(always)]` no-op on `None`.

## Where remaining cost lives (callgrind, lossy e7 zen)

- `patches_cc_min_starts` + `find_text_like_patches` closures: ~250M Ir —
  BFS/DFS graph traversal, inherently scalar; not a dispatch candidate.
- `transform_blocks_into` (~136M), `compute_srgb_u8` (~100M),
  `count_zero_coefficients` (~80M): scalar loops that v3 didn't speed up —
  bound by integer work / memory, not FP SIMD.
- `scratch_buf` zero-init of pass-1 `tmp` survives (the other buffer is gone);
  eliminating it needs `MaybeUninit` (blocked by `#![forbid(unsafe_code)]`)
  or a caller-passed scratch arena — API surgery across ~168 call sites,
  ~1-2% upside. Defer.

## Lossless e9 profile (30.5G Ir)

`find_best_split` ~25% + `build_node_tensor` ~9% — both memory-bound histogram
construction. `estimate_bits_u32` dispatch overhead ≈ 119M Ir (0.4%);
`summon()` is a cached atomic load per project notes. Further wins there are
algorithmic (fewer split evaluations), not codegen.
