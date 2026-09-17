# Encoder performance: state of play and handoff

Last updated 2026-09-17. Every number here is copied from a committed benchmark
record, named inline — none is recalled or estimated. Where the record and this
page disagree, **the record wins**; fix this page.

This page is for picking the work up cold. Read **Traps** before measuring
anything: most of the time lost on 2026-09-10 went to harnesses that produced a
confident wrong number, not to the code.

---

## 1. Where we stand against cjxl v0.12

Process wall, ours/cjxl, median over 4 images × 2 distances, `<1.00` = we are
faster. Source: [`benchmarks/ladder_vs_cjxl_cbrt_2026-09-10.meta`](../benchmarks/ladder_vs_cjxl_cbrt_2026-09-10.meta)
(224 cells). That run predates the `compute_pre_erosion` fix below, so the
threads=8 rows at ≥ 1 MP are slightly pessimistic.

| size | e3 t1 | e3 t8 | e5 t1 | e5 t8 | e7 t1 | e7 t8 | e9 t1 | e9 t8 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 64 | 0.86 | 0.83 | 0.88 | 0.87 | 0.88 | 0.81 | 0.68 | 0.61 |
| 256 | 0.93 | 0.90 | 0.94 | 1.04 | 0.84 | 0.77 | 0.85 | 0.51 |
| 1024 | 1.17 | 1.00 | 0.85 | **1.11** | 0.86 | 0.98 | 1.04 | 0.46 |
| 2048 | 1.14 | 0.88 | 0.89 | **1.30** | 0.85 | 0.89 | 1.06 | 0.44 |

**The two bands still above parity:**

1. **e5 at threads=8** — 1.30 at 2048², 1.11 at 1024². The loosest cell on the
   ladder. It is a *thread-scaling* problem, not per-pixel cost: the same effort
   is 0.85-0.89 at threads=1. §3 is the analysis.
2. **e3 at threads=1** — 1.14-1.17 at ≥ 1 MP. Improved from 1.23-1.25 by the
   cube-root change; still above parity.

≤ 256 px is flat and should stay flat: at 4-12 ms the encode is fixed cost, and
per-pixel work has nothing to bite on. e9 t1 at 1.04-1.06 is perceptual-loop
bound, not kernel bound.

**Reproducing the grid:** `scripts/make_ladder_crops.sh` regenerates the exact
inputs (verified byte-identical against the originals: all 10 crops SHA256-equal),
then

```
scripts/ladder_vs_cjxl.py --ours target/release/cjxl-rs \
  --cjxl <ABSOLUTE path to cjxl v0.12> --out <tsv> --reps 5 <crops...>   # --reps 3 for 2048
```

Build `cjxl-rs` from `jxl-encoder-cli` (it enables `parallel`). The cjxl arm must
be **v0.12** — see CLAUDE.md "Reference Implementations"; on this Mac it is
`.ci-libjxl/tools/cjxl` in the primary checkout. Pass an absolute path (Trap 3).

---

## 2. What landed on 2026-09-10

All byte-identical unless stated. Each links its A/B record, which carries the
method, the byte-identity evidence and the regression test.

| change | effect | record |
|---|---|---|
| patches BFS already-background prefilter | BFS −21…−35 %, patch scan −16…−28 %, bytes and all 15 scan counters identical | [patches_bfs_prefilter_ab](../benchmarks/patches_bfs_prefilter_ab_2026-09-10.meta) |
| skip per-token value-frequency maps below e9 | median e3 0.853×, e5 0.951×, e7 0.977×, e9 1.000× (e9 is the control — maps still built there) | [value_freq_skip_ab](../benchmarks/value_freq_skip_ab_2026-09-10.meta) |
| `per_block_modulations` strip-parallel | step 6.5×; process wall e5 0.982×, e7 0.981× at t8 | [per_block_modulations_parallel_ab](../benchmarks/per_block_modulations_parallel_ab_2026-09-10.meta) |
| `fuzzy_erosion` strip-parallel (even `region_h` only — see record) | e5 0.982×, e7 0.988× at t8 | [fuzzy_erosion_parallel_ab](../benchmarks/fuzzy_erosion_parallel_ab_2026-09-10.meta) |
| HF groups within a DC group encoded in parallel | e3 0.891×, e5 0.952×, e7 0.966×, e9 0.982× at t8; best 0.793× | [hf_group_parallel_ab](../benchmarks/hf_group_parallel_ab_2026-09-10.meta) |
| AC groups within a DC group tokenised in parallel | e3 0.972×, e5 0.984×, e7 0.988×, e9 0.990× | [ac_tokenize_parallel_ab](../benchmarks/ac_tokenize_parallel_ab_2026-09-10.meta) |
| XYB cube root selectable: `cbrt_lowp` (zen) / libjxl-exact (Libjxl strategy) | e3 −7…−8 % of ours/cjxl at ≥ 1 MP. **Moves bytes** at e9 only: median +0.000 %, 50/112 identical, max +2.39 % | [ladder_vs_cjxl_cbrt](../benchmarks/ladder_vs_cjxl_cbrt_2026-09-10.meta), [xyb_cbrt_selection](../benchmarks/xyb_cbrt_selection_2026-09-10.meta), [cbrt_candidates](../benchmarks/cbrt_candidates_2026-09-10.meta) |
| hand-written NEON forward-XYB tier | **a wash on the shipped path** (LowP median 0.989); 0.936 on MidP. Landed as requested, not as a speedup | [xyb_neon_handwritten](../benchmarks/xyb_neon_handwritten_2026-09-10.meta) |
| `compute_pre_erosion` strip-parallel | `quant_field` 14.40 → 12.40 ms at t8; t1 unchanged; bytes identical on 72 cells (threads 1/4/8) | [pre_erosion_parallel_ab](../benchmarks/pre_erosion_parallel_ab_2026-09-10.meta) |

Corpus-level checks behind the whole set: 504 lossy + 336 lossless cells
byte-identical to the pre-session build ([session_byte_identity](../benchmarks/session_byte_identity_2026-09-10.meta));
560 encodes across 10 thread counts and 216 odd/degenerate shapes with zero
nondeterminism; peak RSS unchanged ([parallelism_memory_determinism](../benchmarks/parallelism_memory_determinism_2026-09-10.meta));
no nesting penalty on multi-DC-group images ([multi_dc_group_ab](../benchmarks/multi_dc_group_ab_2026-09-10.meta)).

---

## 3. e5 at threads=8: where the time actually goes

Source: [`benchmarks/e5_t8_phase_scaling_2026-09-10.md`](../benchmarks/e5_t8_phase_scaling_2026-09-10.md).
nature 2048², d1.0, e5. Values are ms per encode.

| phase | t=1 | t=8 | speedup | % of t=8 |
|---|--:|--:|--:|--:|
| **encode_inner total** | 411.8 | 106.0 | 3.88× | 100 % |
| acstrat | 239.4 | 46.6 | **5.14×** | 44.0 % |
| entropy | 51.0 | 22.3 | **2.29×** | 21.0 % |
| quant_field | 31.4 | 15.1 | **2.08×** | 14.2 % |
| xform | 56.5 | 10.5 | 5.38× | 9.9 % |
| gaborish | 13.0 | 4.2 | 3.10× | 4.0 % |

**`acstrat` is not the lever**, despite being the largest share (an older note in
CLAUDE.md said it was). It is the best-scaling large phase. The cell is held back
by `entropy` and `quant_field`; at `acstrat`'s scaling those two would cost ~15 ms
instead of 37.4 — roughly the whole gap to cjxl.

Inside `entropy`:

| sub-phase | t=1 | t=8 | speedup |
|---|--:|--:|--:|
| build_codes | 17.3 | 10.2 | **1.70×** |
| pass2_write | 16.4 | 5.6 | 2.93× |
| ac_tok | 9.5 | 4.8 | 1.98× |
| co (coeff orders) | 7.6 | 1.4 | 5.43× |

`build_codes` parallelises only as `join(build_dc, build_ac)` with AC dominating,
which is exactly why it measures 1.70×. Inside the AC build (64 groups, 6930
contexts): Phase A `accumulate_groups_parallel` ~3.4 ms (already parallel),
Phase B 7.16 ms, of which `cluster_histograms` 6.82 ms. See §4 — only about half
of that is addressable.

Inside `quant_field` at t=8, after the pre-erosion fix: `mask1x1_for_pre_scale`
2.00 ms (3.66×), and **~8.3 ms still unattributed** between it and
`compute_quant_field_float` — `median_mask1x1`, `quantize_quant_field`, the
pre-scale block and the pixel-loss dispatch live there.

---

## 4. Tried and rejected — do not repeat without reading the record

| idea | why it failed | record |
|---|---|---|
| Parallelise `cluster_histograms`' main walk | Walk 2.2× faster, `entropy` −1.5 ms, but encode **+0.6 ms**: a code-LAYOUT penalty of +2.4 ms in `acstrat`, reproduced by a layout probe with the new branch disabled. Also: the assignment half is byte-locked sequential (it mutates clusters as it walks, as libjxl does), so the block is worth ≤ ~3.5 ms, not 6.8 | [cluster_parallel_ab](../benchmarks/cluster_parallel_ab_2026-09-10.md) |
| Native 128-bit (`f32x4`) hand-written NEON XYB | 1.24-1.39× **slower**: `cbrt_lowp`'s `to_array()` window SLP-vectorises better at 8 lanes than 4 | [xyb_neon_handwritten](../benchmarks/xyb_neon_handwritten_2026-09-10.meta) |
| Vectorise the Kahan cube-root bit-hack (exact `divu3`) | 0.36 vs 0.29 ms/Mvalue — LLVM's scalar `/3` over the array window is already cheaper than 14 vector ops. Kept as `bench_cbrt::cbrt_lowp_vecguess_batch` | same |
| Route aarch64 forward XYB to the scalar path | scalar beat SIMD 1.37× on Mac — and SIMD is **4.3-5.3× faster** on three x86 microarchitectures. Would have been a 5× regression on the primary platform | [xyb_forward_kernel_platforms](../benchmarks/xyb_forward_kernel_platforms_2026-09-10.meta) |

---

## 5. Ranked targets for the next box

In measured order of expected value. None has been started.

1. **Attribute the ~8.3 ms inside `quant_field` at t=8.** Measurement, not a fix.
   Add sub-timers between `mask1x1_for_pre_scale` and
   `compute_quant_field_float` in `vardct/encoder.rs` (`_t_quant_field` block,
   ~line 4297-4735). Expect `median_mask1x1` and `quantize_quant_field` to be
   serial. Do this before writing code.
2. **`ac_tok`** — 4.8 ms at t8, 1.98×.
3. **`pass2_write`** — 5.6 ms at t8, 2.93×.
4. **`build_codes`' DC/AC split.** The DC build is small and the AC build is
   serial inside; a better split than `join(dc, ac)` is the structural route.
   `cluster_histograms` itself is capped (§4).
5. **Patch scan waste** — [patches_scan_yield](../benchmarks/patches_scan_yield_2026-09-10.pointer.md):
   542 scans, **433 (80 %) produced no byte change**, 2,842 ms of 42,228 ms
   (6.7 %). On patents, manuscript-text, ai-clipart and textures *every* scan was
   wasted. A cheap pre-scan predictor of "patches will pay" is the lever. This
   changes a decision, so it needs an RD check, not just byte identity.
6. **e3 at threads=1** — per-pixel cost. [e3_profile](../benchmarks/e3_profile_2026-09-10.md)
   has the flat profile, taken BEFORE the cube-root and value-freq changes:
   `convert_rows_to_xyb` 15.3 %, `AccumulatedAnsData::add_tokens` 12.8 %,
   `transform_blocks_into` 9.2 %, `write_tokens_ans` 8.8 %,
   `ANSEncodingHistogram::from_histogram_cached` 4.0 %. Both later changes hit the
   top two symbols, so **re-profile before choosing** — that ordering is stale.

Every item except 5 should be byte-identical. If a hash lock moves, the change is
wrong, not the lock.

---

## 6. How to measure

**Phase timers.** Build with `--features "parallel,__env_var_diagnostics"` and set
`__JXL_ENC_PHASE_TIMING=1`. Prints `encode_inner: total= xyb= quant_field= …`,
`encode_two_pass: … build_codes= pass2_write=`, and setup phases.

**Loop harness.** [`examples/effort_loop_profile.rs`](../jxl-encoder/examples/effort_loop_profile.rs):
```
effort_loop_profile <png> <size> <effort> <distance> <seconds> [threads]
```
Keeps one encode looping so a sampler (`sample <pid>` / `perf record`) can attach.

**A/B protocol that survived:**
- Build BOTH arms as binaries and alternate them inside each repeat, flipping the
  leading arm every repeat. Block-ordered repeats inverted the sign of an 8 %
  effect once.
- Take min-per-encode within a run, then look at **median and range across
  runs**, not just min. Two A/Bs today differed only because one used min.
- Report the phase you changed AND the total. A phase can improve while the total
  regresses (§4).
- **Run a layout probe** for anything under ~3 ms (Trap 5).
- For effects smaller than run-to-run drift, compile both bodies into ONE binary
  and call them back to back ([xyb_neon_handwritten](../benchmarks/xyb_neon_handwritten_2026-09-10.meta)
  has the recipe): cross-binary A/Bs drifted ±4 % on untouched control arms.

**Proving bytes didn't move.** Hash locks run at threads=1 — they cannot see a
bug in a parallel-only path. SHA256-compare `cjxl-rs` output against the
pre-change binary across images × efforts × **threads {1,4,8}**. For a split
whose correctness is non-obvious, also add a parity test that FORCES the split
(see `pre_erosion_strip_parallel_matches_whole_image`) — deriving strip count
from `effective_threads()` makes it vacuous on a single-threaded runner.

---

## 7. Traps (each cost real time on 2026-09-10)

1. **`effort_loop_profile` used to ignore thread count.** The encoder takes
   threads from `LossyConfig::with_threads`, NOT `RAYON_NUM_THREADS`. A t1 vs t8
   run reported 400.0 ms/encode for both. Fixed: `threads` is now an argument.
2. **`parallel` is not a jxl-encoder default feature** — only the CLI enables it.
   `cargo run --example` encodes single-threaded regardless of `with_threads`.
   The example now refuses `threads > 1` without the feature. Any new harness
   needs the same guard.
3. **`ladder_vs_cjxl.py` used to exit 0 with every cell failed** — a relative
   `--cjxl` path from the wrong directory gave a header-only TSV and 224 `FAILED`
   lines on stderr. Fixed: it smoke-encodes both binaries first and exits 1 on
   any failed cell. Use absolute paths anyway.
4. **Arithmetic composition of two measured ratios is not a measurement.** One
   composition predicted 1.19× where the ladder measured 1.25×; the fixes overlap.
5. **Code layout moves unrelated phases by several ms.** Adding a function to
   `cluster.rs` slowed `acstrat` (a different module, running earlier) by 2.4 ms
   with the new code never executing. The only way to see it is a layout probe:
   build the change with its new path disabled by a constant and measure. If the
   "regression" (or "win") survives, the A/B was measuring layout.
6. **Per-iteration allocation inside a parallel region.** A fresh scratch `Vec`
   per chunk per outer iteration was 536 allocations per call and cost 2.4 ms.
   Hoist per-worker scratch out of the outer loop.
7. **aarch64-only conclusions.** See §4 — scalar-beats-SIMD on the Mac inverted
   to a 5× loss on every x86 box. Measure x86 before shipping any kernel-dispatch
   change. Hosts in CLAUDE.md "Resource Discipline"; `r5900xt` is the quiet Zen 3
   box (AVX2, no AVX-512).
8. **`debug_assert` in a release A/B is compiled out.** A mis-mapped strip that
   computed an extra row was silently truncated by `copy_from_slice` and passed
   its parity test. Shape checks guarding indirect addressing belong in real
   `assert!`.
9. **Lossy-e9 peak RSS is bimodal** (~1016 / ~1067 MB on nature 2048²). A
   max-of-3 delta on that cell caught opposite modes and reported a false +4.3 %
   memory regression. Use interleaved repeats and medians.
10. **Lossless output depends on thread count by design** at e ≤ 7
    (`SectionedTrees::Auto`, `parallel.rs`). Not a bug; don't chase it.

---

## 8. Correctness constraints any speedup must keep

- **Cross-tier SIMD bit-identity.** `hash_lock_expected.txt` is one sidecar and CI
  runs it on x86_64 and aarch64. All five `#[magetypes]` kernels in
  `jxl-encoder-simd` are now pinned bitwise across tiers by `*_is_bit_identical_*`
  tests. magetypes' SCALAR backend implements `mul_add` unfused, which is why four
  kernels have hand-written `_scalar` tiers — see the Resolved Bugs entry in
  CLAUDE.md and `docs/SIMD_PARITY_KNOWN_DIVERGENCES.md`. The general fix belongs
  upstream in magetypes and is an owner decision.
- **wasm32 cannot be bit-identical** (WASM SIMD has no FMA) — `xyb-001`. The
  bitwise tests are `#[ignore]`d on wasm32 only.
- **`EncoderStrategy::Libjxl` must stay libjxl-exact**: its byte lock
  (`strategy_libjxl_byte_lock`, 5 cells) must not move. The XYB cube root is
  selected per strategy for exactly this reason.
- Byte-identical changes still need the multi-DC-group and odd-shape checks —
  2048² has exactly ONE DC group, so most of the grid cannot see a DC-group
  ordering bug.
