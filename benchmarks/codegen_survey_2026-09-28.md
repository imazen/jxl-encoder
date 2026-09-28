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

# Continuation — 2026-09-28 (i265, Zen5-class x86_64, 20c)

Second sweep on i265. Workload changed: `jxl-encoder/tests/images/frymire.png`
(1118×1105, ~1.24 MP photo) lossy e7/d1.0 `--features parallel` — nature1024.ppm
was not transferable, so absolute numbers are NOT comparable to the r7900x
rows above. callgrind Ir totals: 6,324M → 6,030M. Wall: paired-interleaved
15-run A/B, base median 115.3 ms → exp 103.8 ms (**−10.0%**, min −14.6%,
exp won all 15 pairs). Lossless e9 frymire: 4.88 s → 4.83 s (neutral).
Both outputs sha256-identical on lossy-e7 and lossless-e9.

## Landed (i265)

**gaborish 5×5 kernel — row pre-slicing** (`gaborish_5x5_impl`, shared
magetypes body so v3/v4/neon/wasm128 all get it). The inner loop issued 25
`f32x8::from_slice` calls per 8-pixel iteration, each carrying a
`len >= 8` bounds check LLVM could not elide — `slice::index` attribution
**129.5M → 10.1M Ir (−119M)**. Fix: slice the five stencil rows + output
row once per `y` (`len == width` known), so `x < width - 10` proves every
`row[x..]` load. Store via `out_row[x..x+8]` likewise.

**`gaborish_5x5_into` (out-of-place entry point) + strip writes direct to
`out`** (`parallel_chunks_mut` helper added to `parallel.rs`, rayon
`par_chunks_mut` / sequential `chunks_mut`). The strip-parallel path used to
`to_vec()` the halo-extended input AND let `gaborish_5x5_channel` copy it
into scratch, then `drain`+`extend` the kept rows — ~4 full-strip copies.
Now: kernel reads `raw` in place and writes a strip tmp; kept rows
`copy_from_slice` into `out`. memcpy **170.6M → 85.5M Ir (−85M)**. Same fix
applied to `compute_mask1x1_strip_parallel`.

**`transform_blocks_into` env-probe hoist.** `JXL_QAC_DUMP` /
`JXL_COEFF_IN_DUMP` were read with `std::env::var_os` **per block** — getenv
is a linear environ scan (~800 Ir each). Hoisted to once per call:
getenv **30.6M → 0.57M Ir**.

**`patches_cc_min_starts` phase-1 root tracking + row slices.** Each
foreground pixel unioned W/NW/N/NE neighbors with a fresh `find_local(gi)`
per union — but `gi`'s root while its own unions run is just the running
min (its entry was self-initialized one instruction earlier). Track `root`
locally; skip up to 4 find walks per pixel. Plus `is_background` row slices
killing per-pixel bounds checks. `closure#1`: **112.5M → 74.7M Ir (−38M)**.

**BFS `eval_chunk` — plane/weight hoisting + interior fast path.**
`weighted_distance_to_color_idx` reloads a (ptr,len) slice header per
channel per call; inlined it with `p0/p1/p2` + `cw0..2` hoisted, and added
an `interior` predicate (all 8 neighbors in-bounds) skipping the per-k
coordinate range checks. `closure#3`: **203.5M → 155.7M Ir (−48M)**.

**`replay_cc` stack entries → (x,y) pairs.** The flat u32 index encoding
paid `pi % stride` + `pi / stride` (a hardware `div`) per pop; (x,y) pairs
recover `pi` with one multiply. Small on photo input (the single giant
foreground CC rejects on size quickly) — larger on screenshot corpora
where many small CCs survive. Byte-identical traversal order.

## Measurement notes

- **callgrind massively overstates `memset`/`memcpy` ERMS cost**: glibc's
  `__memset_avx2_unaligned_erms` uses `rep stosb`/`rep movsb`; callgrind
  counts ~8 Ir per *byte*. The ~436M Ir memset residue (DCT-kernel
  `scratch_buf` zero-init, ~4 KB per call) is real but the wall cost is
  ~50-100 ns per call — the survey's "~1-2%" estimate is an Ir-space
  overstatement. The correct fix remains caller-passed scratch (~168 call
  sites) — deferred again on wall-ROI grounds, not Ir.
- `compute_srgb_u8` (~117M Ir: `row_sums` + `flat_blocks` + Sobel luma)
  needs a stride-3 deinterleave to SIMD — **no gather/shuffle exists in
  the magetypes portable surface** (`i32x8` has no `shuffle`, `u8x32` no
  gather). Can't express it under `#![forbid(unsafe_code)]` without
  archmage-side support. Left as-is; a per-(bpp,offsets) const-generic
  specialization is the remaining scalar lever (~10-20M Ir estimate).
- `histogram_distance_reuse` (~134M Ir incl. zip + tail copy): work is
  already zip/SIMD-shaped; cost is call-count-driven (clustering O(n²)
  pairs) — algorithmic.
