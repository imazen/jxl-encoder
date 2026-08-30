# JPEG XL Encoder (Rust) - Claude Code Instructions

## Project Overview

This is a work-in-progress Rust implementation of a JPEG XL encoder.

**Reference Target: Full libjxl** — We started by porting libjxl-tiny as a stepping stone,
but now target full libjxl quality and feature parity. libjxl-tiny was useful for initial
correctness verification, but is no longer the reference for quality comparisons or new features.

## Reference Implementations

- **libjxl (C++)**: `~/work/jxl-efforts/libjxl` - **PRIMARY** reference encoder/decoder
  - Use cjxl for quality comparisons and RD benchmarks
  - Use djxl for decode verification
- **libjxl-tiny (C++)**: `~/work/libjxl-tiny` - Historical stepping stone (DO NOT USE FOR REFERENCE)
  - Was used for initial port verification, now superseded
  - See [docs/archive/LIBJXL_TINY_PORT.md](docs/archive/LIBJXL_TINY_PORT.md) for historical port details
- **jxl-rs (Rust decoder)**: `~/work/jxl-rs` - **PRIMARY** Rust decoder for roundtrip tests
  - GitHub: https://github.com/lilith/jxl-rs (more conformant and complete)
- **jxl-oxide (Rust decoder)**: `~/work/jxl-efforts/jxl-oxide` - Alternative Rust decoder

## IMPORTANT: Reference Target is libjxl, NOT libjxl-tiny

**libjxl-tiny was a stepping stone. It is no longer the reference for quality or features.**

Quality comparisons should use cjxl (full libjxl) as the baseline. libjxl-tiny produces
lower quality at the same distance parameter due to different quantization constants and
lack of advanced features (error diffusion, better cost models, etc.).

**DO NOT:**
- Compare our output with libjxl-tiny for quality assessment
- Use libjxl-tiny constants or algorithms without checking full libjxl

**DO:**
- Compare RD curves against cjxl at effort 5-7 (Hare/Wombat/Squirrel)
- Read full libjxl source for algorithm details
- Use djxl for decode verification (works for both)

## IMPORTANT: Decoder Testing Priority

**ALWAYS use jxl-rs as the primary decoder for roundtrip validation tests.**

1. **jxl-rs** (`~/work/jxl-rs`) - Use FIRST for all roundtrip tests
2. **djxl** (libjxl CLI) - Use for compatibility verification with reference implementation
3. **jxl-oxide** - Use as secondary/alternative decoder

When adding or modifying roundtrip tests, ensure BOTH jxl-rs and djxl are tested.
Never omit jxl-rs from decoder validation.

## CRITICAL: PNG Color Metadata Causes Bogus Butteraugli Scores

**This has wasted days of investigation TWICE. Read this before any butteraugli comparison.**

**The problem**: `butteraugli_main` (libjxl CLI) uses the CMS to linearize input images.
Different PNG color metadata → different linearization → different scores, even for
identical pixel data. This produces up to **2x score inflation** that looks like a quality bug.

**How it happens**:
- Most codec-corpus PNGs have `gAMA=0.45455 + cHRM` chunks (no `sRGB` chunk)
- `butteraugli_main` linearizes these with pure gamma 2.2: `pixel^2.2`
- Our JXL files declare `TransferFunction::Srgb` (the correct thing to do)
- `butteraugli_main` linearizes JXL with sRGB TF: linear segment below 0.04045, then `((x+0.055)/1.055)^2.4`
- These differ significantly in darks — gamma 2.2 vs sRGB TF diverge by up to 5% near black
- Result: `butteraugli_main source.png our.jxl` compares gamma-2.2-linearized vs sRGB-linearized → inflated score

**The fix — three valid approaches**:
1. **Use Rust butteraugli** (always applies sRGB TF consistently to both images) — PREFERRED
2. **Strip PNG metadata** before comparison: `convert source.png -strip stripped.png`
3. **Add sRGB chunk** to source: both images linearize with sRGB TF

**What does NOT work**:
- Comparing `butteraugli_main source.png our.jxl` directly (metadata mismatch)
- Decoding our JXL to PFM and comparing (PFM is assumed nonlinear sRGB by butteraugli_main)
- Assuming inflated scores mean quality is bad (they might just mean metadata mismatch)

**History**:
- Feb 15, 2026: wasted a session investigating "2x worse external scores" — was metadata mismatch
- Previously in butteraugli crate: similar TF mismatch caused months of wrong parity conclusions

**The old comparison scripts at `/tmp/run_cmp3.sh` (was /tmp — wiped; use ~/tmp
for any rebuild) used `butteraugli_main` and are UNRELIABLE unless source PNGs
are metadata-stripped first.**

**The correct way to compare quality against cjxl:** `just quality-compare` — uses in-process
Rust butteraugli on both encoders' output, decoded via jxl-oxide in linear RGB. Completely
immune to PNG metadata issues.

## Multi-Metric Perceptual Backend (2026-05-25)

The iterative quantization loop ("buttloop") is now **metric-agnostic**.
It is driven by a pluggable `PerceptualBackend` trait (`vardct/perceptual_backend.rs`)
with three selectable metrics, chosen explicitly by the caller:

```rust
LossyConfig::new(distance)
    .with_perceptual_metric(PerceptualMetric::Butteraugli) // default
    // or Cvvdp, or Zensim
    .with_perceptual_device(PerceptualDevice::Auto)        // Auto | Cpu | Gpu
```

**Metrics + verdicts** (from the 7-backend × 1,134-cell tracking sweep,
`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`):

| Metric | Cargo features | Pareto-front | Verdict |
|---|---|---|---|
| **Butteraugli** | (default) | 67.7% | DEFAULT — Pareto-optimal, all W44 calibration built on it |
| **CVVDP** | `cvvdp-loop` (+ `cvvdp-loop-cpu`, `cvvdp-loop-tighten`) | **95.4%** | OPT_IN (improved) — after Phase 8 refit; -7.4% bytes, -27% wall vs B |
| **Zensim** | `zensim-loop` (+ `zensim-loop-gpu`) | 65.3% | OPT_IN — calibration over-loose; Phase 8-zensim refit pending |

**Invariants** (binding, verified by tests on every change):
- Default path (butteraugli, no opt-in) is BYTE-IDENTICAL regardless of
  which metric cargo features are compiled (hash-locks 36/36).
- `EncoderStrategy::Libjxl` ALWAYS forces Butteraugli via
  `resolve_perceptual_metric()` — strict cjxl-parity (byte-lock 4/4),
  regardless of `with_perceptual_metric()`.
- Per-content-class auto-dispatch is intentionally OUT OF SCOPE — the
  user explicitly selects the metric.

**Score direction**: the trait normalizes every metric to butteraugli
direction (smaller = better). CVVDP JOD → `(10 - jod).clamp(0, 10)`;
zensim 0-100 → `(100 - score).clamp(0, 100)`.

**Per-metric tuning**: each backend ships its own block-reducer constants.
The critical one is `K_TILE_NORM` (the 16th-power-norm premultiplier in
the per-block reducer): Butteraugli 1.2, CVVDP **0.16** (Phase 8g refit —
the single change that took cvvdp 40%→95% Pareto), Zensim 1.2
(butter-parity placeholder; Phase 8-zensim will refit). CVVDP also has
`CVVDP_DIFFMAP_RENORM_SCALE = 0.018` (Phase 8c — derived as
`(target_c/target_b) × (mean_b/mean_c)`, NOT `mean_b/mean_c`).

**Reference docs**: `docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` (API),
`docs/RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` (what a metric needs),
`docs/CVVDP_FORK_DECISION.md` + `docs/ZENSIM_FORK_DECISION.md` (verdicts),
`docs/CVVDP_W44_GATE_TRANSFER.md` (which W44 gates transfer vs need
re-calibration per-metric). The metric CPU/GPU crates live in
`~/work/zen/zenmetrics/crates/{cvvdp-gpu,cvvdp-cpu,zensim-gpu}` and
`~/work/zen/zensim/`.

**Queued follow-ons**: Phase 8-zensim (K_TILE_NORM refit, 65%→85%+),
Phase 7-zensim (docs), zensim-gpu GPU-native diffmap kernels (currently
CPU-fallback), cvvdp-cpu structural perf (strip-pipeline + f16 for
150ms→50ms at 1024²).

## ACTIVE WORK PROGRAM 2026-08-30 — five tasks, root causes established

Owner-directed after the #45/#76/#96 program closed. **Every one of these
touches this repo, so they are strictly sequential — claim `.workongoing`,
finish, release.** Root causes below are MEASURED, not hypothesised; do not
re-derive them from scratch, but do re-verify any premise you are about to
build on (this file has been wrong before).

### T1 — e≤9 resampling domain fix + re-lock (APPROVED byte move)

Root cause and evidence: see "ACTIVE 2026-08-30: e≤9 `with_resampling(2)`
downsamples in the WRONG domain" under Known Bugs. Short form: the decoder
upsamples XYB *before* the inverse colour transform, libjxl's encoder
downsamples the opsin image, and our e≤9 box/sharper wrappers downsample linear
RGB. Averaging does not commute with the XYB nonlinearity, so the round trip
cannot compose — butteraugli 19.68/30.20 vs 7.20/10.66 domain-correct.
The e≥10 iterative path already got this fix in `e3a54d7e`; **copy its shape**
(convert → box/sharper per opsin plane → invert), do not invent a second one.
**The owner has approved moving e≤9 `resampling>1` bytes.** No hash lock covers
those cells today, so ADD e≤9 r2 lock cells as part of the fix, and regenerate
via the repo's regen workflow with in-process decode of every stream. e1–e9
locks for `resampling == 1` must NOT move — if any non-resampling cell moves,
stop and find out why before relocking.

### T2 — zensim secant min-|Δln L| guard

`vardct/zensim_loop.rs`. The controller measures elasticity
`ε̂ = Δln L / Δln S` from the last two iterates. **Sign convention that is easy
to get backwards: higher quant_field = MORE bits = LESS loss, so ε̂ is
NEGATIVE**; the secant fires only when `ε̂ < −1e-6`. Known flaw: when
consecutive iterates are nearly equal, Δln L is tiny, ε̂ is unreliable, and the
step overshoots (measured: t=70 went 71.0 → 61.8 at iter 2). The existing
`|Δln S| > 1e-6` guard does not catch this because it guards the wrong axis.
Fix = a min-|Δln L| guard falling back to the power law when the loss barely
moved. Evidence + the A/B methodology: `benchmarks/zensim_secant_2026-08-25.md`.
Note the loop never entropy-codes, so its traces have no bytes column — per-
iterate bytes only come from separate budget-capped full encodes.

### T3 — e7 sectioned wall, via content-adaptive K (corpus expansion REQUIRED)

Root cause: at e7 t=1 the 11.5 s total is 8.5 s per-group tree learn, of which
**`find_best_split` is 4.7 s — our split search evaluates K=8 kept predictors
where libjxl learns over 2** (`enc_modular.cc:642-644`, `Predictor::Best` =
{Weighted, Gradient}). ~4× the search work, and sectioned runs it *per group*
rather than once whole-image, so it multiplies by group count exactly where
sectioned mode is meant to be winning memory. e9 is inside the 1.3× bar
(1.08×/1.10×) only because its surrounding encode is expensive enough to bury
the same absolute cost.
Already measured and rejected as blanket defaults: prune K=6 (−4…−13 % wall,
≤ +0.20 % bytes), K=4 (up to +1.35 % bytes on palette content) —
`benchmarks/jxl_sectioned_prune_k_2026-08-28.{tsv,meta}`. The single-worker
fork-engine bypass already landed (−3.5 % learn wall).
**Owner directive: content-adaptive K is approved, but the gate must be fitted
on a corpus that actually spans the content space** — expand the low-resolution
sampling of imazen-26 to more images *and crops thereof*. Per the workspace
sweep discipline: cluster to pick representative sources rather than sampling
randomly (random over-represents the modal class and drops the outliers a gate
fails on), log-spaced sizes, skip upscaling, and hold out a validation split.
A gate fitted on a handful of images will look excellent and generalise badly.

### T4 — libjxl mimic mode toward byte-exactness

`EncoderStrategy::Libjxl` exists already. The goal is **not** cosmetic parity:
byte-exactness is the instrument for (a) quantifying where our perf gap to
libjxl actually comes from and (b) surfacing bugs in zen mode that only show up
as an unexplained divergence. Start from `docs/LIBJXL_PARITY_TRACKING.md`
(component-by-component, source-grounded against libjxl `d089091a`) and
`docs/LIBJXL_DIVERGENCES.md`. Known structural gaps recorded there include the
predictor count (theirs 2, ours 7–9 via probe-tree pruning, 14 historically),
Bernoulli vs fixed-stride sampling (density at parity, aliasing behaviour
differs), and the three §A fidelity gaps from the e11 TectonicPlate work
(kNoWP transcribed but unwired; fixed-predictor trials single-leaf vs their
restricted-candidate tree learn; sequential vs RunOnPool). **Every divergence
found must land in `LIBJXL_DIVERGENCES.md` — that file is mandatory per the
section below.** Where we deliberately differ because we beat them on RD, say
so explicitly rather than "fixing" it into a regression.

### T5 — make the sectioned estimator accurate

Issue #99 item 3. `estimate_encode_sectioned`'s e9 additive term is
**36 MiB/worker, calibrated on photo 1024²** (measured slope 31.6). imac_dark
e9 t=12 measures **4.3 MiB/worker** — TYP = 847.9 MB = **2.52×** the measured
cell, just over the 2.5× tightness bar in
`sectioned_estimate_covers_measured_cells_2026_08_27`. The pin is deliberately
held stale-high with an in-code comment; lowering it to the honest measurement
fails the bar.
**The conflict to resolve:** a headroom-aware additive term (subtracting the
measured envelope−content-floor slack, 68−48 B/px) restores tightness on every
rgb cell but makes the model **non-strictly-monotone in threads** at large px —
which `heuristics::tests::sectioned_estimate_shape`,
`api_tests::encode_preflight_sectioned::floor_is_the_minimum_over_one_and_two_workers`
and `walks_down_on_the_sectioned_per_thread_term` all assert, and on which the
admission thread-walk-down contract is built. Either refine differently, or
change the contract deliberately with its tests updated as a designed change —
**not** by weakening an assertion to make a number fit.
Bundle with #99 item 2 if you touch the model: the RCT trial wave
(`select_best_rct` holding nine whole-image i32 clones = 193 MiB, the designed
`RCT_TRIAL_WAVE = 2` transient) is now what SETS the sectioned peak, and the
per-trial fold at `effective_threads() == 1` would move the floors again. Move
the model once, not twice.

### Cross-cutting facts worth not rediscovering

- **`cargo fmt --all` reaches into sibling path-dep repos** and will silently
  reformat them (confirmed live on 2026-08-29: it edited `../zenav1-aom`).
  Always scope: `cargo fmt -p <pkg>`.
- **CI cancellation is high in this workspace** (measured 2026-08-30: 44 % of
  the last 25 CI runs on this repo's main, and 72–80 % in zenmetrics/zensim/
  zenanalyze/zenpipe). A run that ends `cancelled` is an ABSENCE of a verdict,
  not a weak pass — and a real failure can hide among the cancellations because
  a red Lint job fail-fasts the matrix. Always check the job-level tally, and
  re-run if your commit's own run was superseded.
- **`butteraugli 0.9.4` DOES gate this crate.** A 2026-08-29 claim that 0.9.3
  suffices was measured false: `ButteraugliReference::estimated_reference_bytes`
  was introduced after `v0.9.3` and is referenced from
  `tests/it/alloc_budget.rs`. A lib-only compile succeeds against 0.9.3 and
  looks like proof — it is not.
- **Published `jxl-encoder 0.3.1` does not build today** under fresh
  resolution (`magetypes 0.9.28` breaks published `jxl-encoder-simd 0.3.0`);
  workaround `cargo update -p magetypes --precise 0.9.23`. The 0.4.0 release
  fixes it. Path-dep local builds never exercise what a registry consumer gets.

## Current Status

The VarDCT encoder implements every algorithmic component libjxl evaluates
through effort 9 (19/27 AC strategies — the remaining 8 are never selected by
libjxl either), full parametric quantization weights, the complete adaptive
quantization pipeline, the butteraugli quantization loop at e8+, and the
modular (lossless) path beats cjxl e7 on CLIC photos. JPEG-in-JXL transcode
sits at +0.115 % vs cjxl e7 with 200/200 byte-exact reconstruction.

Expert knobs measured ALWAYS-WORSE are tabled in
[docs/EXPERT_KNOBS_MEASURED_WORSE.md](docs/EXPERT_KNOBS_MEASURED_WORSE.md)
(same verdicts carried as **MEASURED** warnings on the field rustdocs).

Where we intentionally diverge from libjxl, the row lives in
[docs/LIBJXL_DIVERGENCES.md](docs/LIBJXL_DIVERGENCES.md) (single source of
truth, drift-tested in CI). Measure current RD with `just quality-compare`
(in-process Rust butteraugli + SSIM2, metadata-immune); never trust dated
tables — re-measure. Historical status snapshots (Feb 2026 quality-gap
tables, the libjxl-tiny upgrade roadmap) are archived in
[docs/CODE-HISTORY.md](docs/CODE-HISTORY.md).

The Feb-2026 deep-status snapshots that used to follow here ("Remaining
Gaps vs Full libjxl" strategy/calibration/entropy tables + the "What Works"
checklist) were archived to [docs/CODE-HISTORY.md](docs/CODE-HISTORY.md)
("Archived from CLAUDE.md 2026-07-19" section) — re-measure, don't trust them.

## Resolved Bugs

See [docs/CODE-HISTORY.md](docs/CODE-HISTORY.md) for full chronological bug narrative.

### RESOLVED 2026-08-01: 108 MP encodes kernel-OOM-killed 32 GiB boxes — QUADRATIC per-tile AC-strategy peak + dishonest flat pre-flight

zensysbench fleet (2026-07-31): 108 MP lossy e5/e7 encodes OOM-killed 32 GiB
boxes at ~29.9 GiB RSS **even at threads=1** (and a 40 GiB cgroup at 24
threads in 18 s). Measured root cause (grid
`benchmarks/jxl_encode_mem_threads_2026-08-01.{tsv,meta}` + heaptrack):
`compute_ac_strategy_for_tiles` allocated a FULL-IMAGE `AcStrategyMap`
(n_blocks bytes) per 64×64 tile and `parallel_map` collected all of them
alive until the merge → **n_tiles × n_blocks = px²/262144 bytes** of
transient peak (3.81e-6 B/px², matched the measured size curve to 1 %):
~8.8 of 9.6 GiB at 48 MP, ~44.5 GiB at 108 MP — the true t=1 peak exceeded
44 GiB (cgroup-censored); the fleet kills were box-size censoring, and the
"parallel blowup" was the same peak reached faster, NOT a per-thread term
(RSS is thread-flat at 12–108 MP; the only measured slope is lossy e7's
2.5 MB/thread). Two fixes (byte-identical, hash-locks 53/53 + Libjxl
byte-lock 5/5 + rd-regression green):
1. `AcStrategyMap` backing-store **window** (`new_dct8_tile`): per-tile
   scratch maps are tile-rect-sized (≤ 64 B). Safe because every
   `process_tile` access is tile-bounded and the merge reads only the tile
   rect; `idx()` debug-asserts the window. 108 MP e5 t=1: ≥ 44 GiB → 5.65
   GiB; 48 MP: 9.93 → 2.49 GiB. Do NOT revert to full-image scratch maps.
2. The flat `w×h×40` pre-flight (six copies, effort/path/thread-blind — it
   ADMITTED those encodes) → ONE `encode_preflight` (api.rs) at all five
   entry points: calibrated `heuristics::estimate_encode_threaded`
   admission, budget-driven thread walk-down (fits cap AND MemAvailable
   ×0.8; availability only reduces threads), rejection only when the t=1
   estimate exceeds the cap, decision recorded in `EncodeStats`
   (`budget_peak_bytes` / `threads_used` / `estimated_peak_bytes`).
   Recalibrated on the 12–108 MP corpus post-fix
   (`…_postfix_2026-08-01.tsv`): lossy e≥7 band 135 B/px (q30 regime was
   never swept; measured up to 122), lossless base 92 / tree 540, lossy
   γ=2.5 MB/thread UNCLAMPED past max_useful_threads. At default caps this
   now cleanly rejects ≳50 MP lossy-e≤6 / ≳30 MP lossy-e7 / ≳15.7 MP
   lossless-e7 (they estimate past 4/8 GiB) — callers with real budgets set
   `Limits::with_max_memory_bytes`.
Remaining unguarded mass (post-fix heaptrack, 48 MP e5): per-group AC token
vectors from `tokenize_ac_group` (~1.15 GB ≈ 24 B/px, linear, all groups'
tokens alive before ANS build) — covered by the pre-flight envelope, not
individually budget-reserved. Budget guards track ~61 % of the 108 MP peak.
Tests: `tests/it/alloc_budget.rs` (honest tight-cap + default-cap
rejections, 16→3 thread-walk-down success), `api_tests::
encode_preflight_walkdown`, `heuristics::estimate_covers_measured_large_sizes_2026_08`.

### RESOLVED 2026-06-23: #93 butteraugli buttloop OOMs sweeps — precompute evaded the budget (NOT a pool leak)

`zenmetrics sweep --codec zenjxl --plan modes_full` OOM-killed Hetzner
cpx51 boxes (31+ GB RSS) encoding many varied-size renditions. The issue
blamed `butteraugli::image::BufferPool` "growing unbounded across
encodes." **Measurement overturned that diagnosis** — there is NO leak:
- `examples/pool_growth_probe.rs` (sequential multi-encode, VmRSS per
  encode): at constant 512² e8 threads=8 over 300 encodes the RSS *floor*
  is flat (44→45.7 MB, oscillating 33–60), peak ~85–99 MB. The earlier
  "growth" was pure current-image-size tracking. Provenance:
  `benchmarks/issue93_pool_*_2026-06-23.{tsv,meta}`.
- Each jxl-encoder encode builds a **fresh** `ButteraugliReference` with a
  **fresh** `BufferPool` (`precompute.rs:361`, `BufferPool::new()`),
  count-capped at `MAX_POOL_BUFFERS = 8`, **dropped at encode end**. The
  pool does not persist across encodes; heaptrack only attributed bytes to
  `BufferPool::take` because that is the malloc *site*. The 32 GB is
  concurrency × per-encode butteraugli precompute, not accumulation.

**Real root cause**: the buttloop runs under the default 4 GiB lossy
`MemoryBudget` cap, but the `ButteraugliReference`'s multi-resolution
psycho pyramid + masks (the buttloop's dominant allocation — heaptrack:
6.35 GB) is allocated *inside* butteraugli, so it was invisible to the
budget. `perceptual_loop.rs` reserved only the `4*3*n` planar ref planes,
NOT the precompute → the cap could not catch the largest allocation →
OOM-kill instead of a graceful `EncodeError`. Same shape as the 2026-06-13
12 MP HDR fix, inverted (that over-counted; this under-counted).

**Fix — LANDED** (butteraugli + jxl, authorized cross-repo; verified
in-tree 2026-07-19: `ButteraugliReference::estimated_reference_bytes` defined at
butteraugli `src/precompute.rs:789` — pinned by
`estimated_reference_bytes_matches_precompute` — consumed at
`jxl-encoder/src/vardct/perceptual_loop.rs:1066`, tests in
`jxl-encoder/tests/it/alloc_budget.rs`):
1. butteraugli (additive, no behavior change): `ButteraugliReference::`
   `estimated_reference_bytes(w,h,&params)` (a-priori precompute cost) +
   `precompute_bytes`/`memory_bytes` introspection, pinned exactly to the
   real allocation by `estimated_reference_bytes_matches_precompute`.
2. jxl guard (`perceptual_loop.rs`): reserve `estimated_reference_bytes`
   on the budget before `set_reference`, scoped to the CpuButteraugli path,
   held for the backend lifetime. The 4 GiB cap now bounds the buttloop.
   Pure accounting — hash-locks 48/48 + Libjxl byte-lock 10/10 unchanged.
3. fallible alloc: the buttloop ref planes honor the runtime
   `Limits::fallible_alloc()` toggle (`budget::vec_f32_zeroed_fallible`).
Tests: `tests/it/alloc_budget.rs` `issue_93_buttloop_precompute_charged_
against_budget` (multi-group 768² e8) + `butteraugli_precompute_estimate_
is_exposed_and_scales`. Does NOT bound the *concurrent sum* across a
sweep's parallel encodes — that is a sweep-harness (zenmetrics) concern;
the jxl fix bounds per-encode so a single oversized rendition fails
gracefully instead of killing the box.


Key patterns to watch for when working on this codebase:
- **Transpose/layout bugs**: DCT output is transposed for square blocks, not for ROWS<COLS. Always verify against C++ `ComputeScaledDCT`.
- **Bit alignment cascades**: One wrong-width field shifts all subsequent reads. Decoder errors appear far from the actual bug.
- **Quantization direction inversions**: Using `1/weight` vs `weight`, or forward vs inverse scale. Values end up squared-off from correct.
- **Block context map**: Decoder uses `order_id` (0-12), not `strategy_code` (0-26). Account for X↔Y channel swap.
- **Edge handling**: Encoder and decoder must agree on boundary pixel fallbacks (0 vs clamped neighbors).

## MANDATORY: Maintain `docs/LIBJXL_DIVERGENCES.md` with every change

**Every commit that touches a gate, tolerance, constant, or algorithm choice that differs from libjxl MUST update `docs/LIBJXL_DIVERGENCES.md`.** No exceptions.

This is the SINGLE SOURCE OF TRUTH for where our encoder diverges from libjxl reference. It prevents:
- Re-investigating already-ruled-out divergences (W44-93 try_dct64, W44-102 cfl_two_pass)
- Losing track of intentional content-aware lifts (the W44-91/96/98/99 zenanalyze stack)
- Re-introducing fixed bugs (W44-3 EPF Pass-1, W44-8 patches DC quant, etc.)
- False-completion claims about parity that don't account for active KNOWN-BUG clusters

**When to update**:
- Adding/changing/removing any gate condition (`effort >= N`, distance threshold, content discriminator)
- Adding/changing/removing any numeric constant (entropy_mul, kFavor, threshold, multiplier)
- Adding/changing/removing any algorithm choice (which TreeKind, which clustering strategy, which search policy)
- Moving a divergence from ACTIVE/KNOWN-BUG → RESOLVED (do NOT delete the row; move to Section G)

**How to update**:
1. Find the relevant section (A: effort gates, B: content-aware discriminators, C: cost-model constants, D: algorithm choices, E: opt-in APIs, F: known-bug clusters, G: resolved)
2. Add/edit the row with: site, ours-value, libjxl-value, status, commit SHA
3. If discriminator-shape: include the EXACT predicate (e.g. `m_colourfulness >= 80 AND fcbr < 0.01`)
4. Commit the doc update in the SAME commit as the code change (or as an immediate follow-up if scope is too tight)

**W44-193 (2026-05-22)**: the 24 production gates that used to live as hand-written `EncoderImprovementsCustom` / `ResolvedImprovements` / 5 ctors / `apply_env_var_fallbacks` in `api.rs` are now generated by a single `strategy_def!{}` invocation in [`jxl-encoder/src/gate_registry.rs`](jxl-encoder/src/gate_registry.rs). **Every gate carries `divergence_section` + `divergence_row_ref` metadata inline** (emitted as `__CUSTOM_DIVERGENCE_<GATE>: &str` compile-time consts). When adding or changing a gate, update both:

1. The `strategy_def!{}` metadata in `gate_registry.rs` (per-gate `divergence_section` + `divergence_row_ref`, plus the per-strategy value blocks).
2. The matching `ALL_DIVERGENCE_ENTRIES` row immediately below in `gate_registry.rs` (carries the gate-name → (section, row_ref, raw) tuple harvested by the W44-194 drift test).
3. The matching row in `docs/LIBJXL_DIVERGENCES.md` (the W44-194 drift test enforces FINDABILITY: the chunk's W-code referenced by the macro must appear in the table; full auto-generation is deferred per W44-190 RFC §G5).
4. Bump `EXPECTED_DIVERGENCE_GATE_COUNT` in [`jxl-encoder/tests/it/divergence_table_drift.rs`](jxl-encoder/tests/it/divergence_table_drift.rs) when adding/removing a gate.

**W44-194 (2026-05-22)**: two new CI gates enforce the maintenance rule end-to-end:

- [`jxl-encoder/tests/it/divergence_table_drift.rs`](jxl-encoder/tests/it/divergence_table_drift.rs) — anchor-based drift test on the macro-emitted metadata vs the table. Run via `cargo test -p jxl-encoder --features "__expert __internals" --test it divergence_table_drift`. Catches gate add/remove/rename without table sync.
- [`jxl-encoder/tests/it/strategy_libjxl_byte_lock.rs`](jxl-encoder/tests/it/strategy_libjxl_byte_lock.rs) — per-cell SHA256 byte lock for 10 fixtures encoded with `EncoderStrategy::Libjxl`. Run via `cargo test -p jxl-encoder --features __expert --test it strategy_libjxl_byte_lock`. Catches Libjxl-strategy byte drift on any gate-value flip with a per-cell diff message. Regen via `UPDATE_LIBJXL_BYTE_LOCK=1 cargo test --features __expert --test it strategy_libjxl_byte_lock`.

**Sub-agent prompt requirement**: When spawning a sub-agent for any code-change chunk, the prompt MUST include reading `docs/LIBJXL_DIVERGENCES.md` AND `jxl-encoder/src/gate_registry.rs` in "inputs to read FIRST" AND a requirement to update the relevant row(s) + macro metadata before commit. Sub-agents that ship without updating both are failing the chunk's acceptance. The W44-194 drift test will catch most omissions on the next CI run, but pre-commit awareness is still the right discipline.

**Verification**: `git log --oneline -- docs/LIBJXL_DIVERGENCES.md jxl-encoder/src/gate_registry.rs` should show updates roughly synchronized with commits touching `effort.rs`, `vardct/encoder.rs`, `vardct/perceptual_loop.rs` (formerly `butteraugli_loop.rs`), `vardct/ac_strategy_search.rs`, `vardct/dc_tree_learn.rs`, `modular/tree_learn.rs`, or any cost-model constant table.

## Research methodology (binding for tuning chunks)

Empirical encoder-tuning chunks (W44-216 onward) follow nine rules distilled from the W44-218→W44-221 retrospective. Canonical doc: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/research_methodology_9_rules_2026-05-22.md`. Highlights:

1. **Ablation-first** — kitchen-sink GBR with ALL axes (params + features + effort + distance) before pre-imposing structure. The W44-218 R²=0.08 honest-stop was caused by dropped confounder axes, not corpus density.
2. **Pre-registered pipeline** — every new sweep gets `scripts/zenjxl-tuning-sweep/run_all_analyses.py <merged.parquet> <out>` run once, producing kitchen-sink GBR + per-pair baseline + ANOVA + PDPs + SVD basis + Pareto coverage in one job. Add new stages to the pipeline rather than writing one-offs.
3. **Parallel hypothesis chunks** — data-only chunks that don't consume each other's source changes spawn concurrently in one message.
4. **Bigger sweeps, fewer of them** — target 30-50K cells per sweep, ≤$30, all 6 params × all axes upfront.
5. **Persistent ML scratchpad** per sweep at `benchmarks/sweeps/<id>/analysis/notebook/` with `_arrays/` cache.
6. **Hypothesis ledger** at `docs/HYPOTHESIS_LEDGER.md` — every chunk's acceptance memo updates ≥1 entry or notes "no belief change."
7. **Kill heartbeat noise** — silent ScheduleWakeup re-arm on stale `/loop` ticks; no preamble text.
8. **Skip speculative algebraic forms** — for empirical surfaces, use non-parametric fit + SVD basis discovery + post-hoc semantic naming, not pre-imposed "additive/multiplicative/gated" templates.
9. **Use vast.ai interruptible** — `--bid_price` is ~50-70% cheaper than on-demand; fleet already takes eviction hits handled by chunk-rescue. Reference: `scripts/zenjxl-tuning-sweep/launch_w44_216_fleet.sh` (env-toggleable via `INTERRUPTIBLE=0` for smoke pods).

When spawning a sub-agent for a tuning chunk, the prompt MUST include reading the methodology memo + `docs/HYPOTHESIS_LEDGER.md` in "inputs to read FIRST" and acceptance criteria MUST include updating the ledger.

## Known Bugs (ACTIVE)

### ACTIVE 2026-08-30: e≤9 `with_resampling(2)` downsamples in the WRONG domain (linear RGB, decoder upsamples XYB)

**Status**: ACTIVE — flagged during #45; e≤9 fix needs an owner decision
(it moves e≤9 bytes, which the ladder-shift job froze).

[PROVEN] The decoder's 2× upsampling stage runs on the XYB planes BEFORE
the inverse color transform (libjxl `dec_cache.cc` adds
`GetUpsamplingStage` ahead of `GetXYBStage`; jxl-oxide matches), and
libjxl's encoder downsamples the OPSIN image (`enc_frame.cc:742`). Our
e≤9 resampling path (box 4×/8× + sharper 2×, `api.rs` both call sites)
downsamples **linear RGB pre-XYB** — a domain mismatch. Measured
(in-process Rust butteraugli, jxl-oxide `srgb_linear` decode, d1.0 r2,
CID22-512): sharper-on-linear e9 scores bfly **19.68 / 30.20** where the
XYB-domain iterative e10 scores **7.20 / 10.66** on the same cells
(cjxl e10: 7.19 / 9.90). Repro:
`cargo test -p jxl-encoder --test it effort_ladder_tiers::iterative_downsample_cjxl_differential -- --ignored --nocapture`.
The e≥10 iterative path was fixed in `e3a54d7e` (runs in XYB); the fix
for e≤9 is the same shape (convert → box/sharper per opsin plane →
invert) but changes every e≤9 `resampling>1` stream. No hash-lock covers
e≤9 resampling cells. Next step: owner decision on moving e≤9 r2 bytes,
then port the domain fix to the box + sharper wrappers and add e≤9 r2
lock cells.

### RESOLVED 2026-06-13: 12 MP HDR encode failed at 2 GiB cap — budget over-count + too-low default

12 MP PQ-HDR e9 d4 failed with `memory budget exceeded: requested 144 MB on
top of 2.13 GB (cap 2147483648)`. Measured: real peak RSS is ~2.14 GB
(byte-identical output 2820230) — NORMAL for VarDCT at 12 MP (libjxl `cjxl`
v0.12 uses 1.23 GB at e7 → **3.42 GB** at e9 on the same file; we're lighter
at e9). Two real defects: (1) the `MemoryBudget` tracker over-counted —
the buttloop's `TransformOutput`/recon/VDP2-ref buffers used
`reserve_permanent` so the reservations persisted into the post-loop entropy
phase after the buffers dropped (and the loop's `TransformOutput`
double-counted the base one — they never coexist), inflating the budget peak
to 2.99 GB vs 2.14 GB real; (2) `Limits::DEFAULT_MAX_MEMORY_BYTES` was 2 GiB
— too low for ≥ 11 MP. Fix: all four sites → RAII `BudgetGuard` (peak
2.99 → 2.55 GB, −438 MB, byte-identical; hash-locks 48/48, byte-lock 5/5);
default cap 2 GiB → 4 GiB. The `issue_54` / `w44_audit_2` no-spurious-OOM
tests were pinned to an explicit 2 GiB cap so the 4 GiB default doesn't blunt
them. Do NOT scale the default cap with image dimensions — that defeats the
DoS bound (a huge upload would get a huge cap). For trusted batch, callers
set `Limits::with_max_memory_bytes` higher (or `Some(u64::MAX)`).
**Follow-up 2026-06-14: the default cap is now PATH-AWARE** — lossy stays
4 GiB, lossless is 8 GiB (`DEFAULT_MAX_MEMORY_BYTES_LOSSLESS` +
`default_max_memory_bytes(is_lossless)`), because lossless tree-learning is
~440 B/px (≈ 5 GB at 12 MP) so a flat 4 GiB rejected ordinary ≥ 9 MP
lossless encodes. Both remain fixed ceilings (still not dimension-scaled);
the 5 internal budget-cap sites select by path; explicit
`with_max_memory_bytes` still wins. Commit nvupmply.
**Update 2026-08-28 (#96):** the lossless band is re-anchored on a
three-class grid after the August memory reductions — e7–e8 128 B/px,
e9 160, e6 92, alpha +72, effort-dependent intercepts
(`benchmarks/jxl_lossless_band_2026-08-28.{tsv,meta}`); the "~440 B/px"
figure above is the 2026-06 measurement. At the 8 GiB lossless cap the
whole-image path now admits up to ~65 MP at e7.

### RESOLVED 2026-06-11: `gpu-butteraugli` did not compile — two cubecl universes

The workspace patched crates.io `cubecl-*` to the lilith/cubecl git fork
while zenmetrics' GPU crates had migrated to the renamed
`zenforks-cubecl-*` 0.10.1 crates.io publication: `Butteraugli<R>`
bounded on zenforks' `Runtime`, our `CudaRuntime` came from the git
lineage → E0277/E0599. Fixed by dropping the git patch entries and
re-aliasing the member dep `cubecl = { package = "zenforks-cubecl",
version = "0.10.1" }` — the same aliases zenmetrics uses. All GPU
features (`gpu-butteraugli`, `zensim-loop`, `cvvdp-loop`) and the default
workspace now compile; hash-locks 40/40.

### RESOLVED 2026-06-10: env-var runtime-override tests flaked the `it` binary under parallelism

`content_class_dispatch_with_patches_false_respects_opt_out` failed in
full parallel runs, passed in isolation: 9 files mutated process env
(`set_var`/`remove_var`) while dispatch-sensitive roundtrip tests read
those overrides live mid-encode. Fixed structurally: the 9 mutator
modules moved to the separate `env_overrides` test binary
(`tests/env_overrides/`, process-isolated — cargo runs test binaries
serially, nextest gives one process per test) with an in-binary
`env_serial()` mutex taken by every `#[test]` (42 fns). Same isolation
contract as the standalone tuning-override targets. New env-mutating
tests go in `env_overrides`, never in `it`. nightly.yml corpus-gate
invocation carries `--test env_overrides` alongside `--test it`.

### RESOLVED 2026-06-10: `just rd-regression` red — was a one-time CID22 auto-download

The red run's "failed to open" cells were the 5 CID22-512 images, which
`codec_corpus::Corpus::get("CID22/CID22-512/training")` downloaded DURING
that first failing run; the immediate re-run is green 2/2 (13.9 s warm).
frymire's post-2c drift is FAVORABLE and within tolerances — no baseline
regen needed. (The initial diagnosis blaming a missing
`clic2025/validation/` dir was wrong for rd-regression — that path was only
referenced by ignored/visual tests + the fresh_encode example; the corpus
renamed it to `clic2025/training/`, and all 15 stale refs were re-pointed
the same day.)

## Investigation Notes

Dated investigation narratives live in [docs/CODE-HISTORY.md](docs/CODE-HISTORY.md)
(chronological archive — full mechanisms, per-cell tables, acceptance gates,
verbatim DO-NOT lists). This section keeps only (a) the distilled binding
constraints and (b) live follow-ons. When an investigation closes, append the
full entry to CODE-HISTORY.md and add one line here if it leaves a binding
constraint.

### Binding constraints from closed investigations

Each line is enforceable; the named entry in docs/CODE-HISTORY.md carries the
measurement and the full DO-NOT list. Do not relitigate these without new
measurement at equal or better coverage.

**Cost model / AC strategy**
- No global `entropy_mul` lifts: variant Z/W global dct32 values in [1.20, 1.34],
  `dct16x32` > 1.30 (high-colour) or above 1.349 globally, all regress measured
  cells; only per-image discriminators ship. Don't move the calibrated
  discriminator thresholds (W44-98 m3=25, W44-96 ed=0.7/fcbr=0.01, W44-91 m3=80
  band d∈[3,5]) without re-running their bisections. (W44-91/94/95/96/98/99/207)
- `try_dct64: effort >= 7` stays — widening to e5 OOMs imac_g3 e9 and costs photo
  SSIM2; the W44-35 smart-dispatch window is the sanctioned route. (W44-93)
- AFV evaluation policy and per-call cost are AT PARITY with libjxl (16,384
  calls/variant verified) — do not respawn AFV-distribution chunks. (W44-60)
- coeff_orders: no single-scalar `savings_factor` recalibration — measured 6.4×
  weaker than the W44-201/205 per-bucket gates and additive EV below threshold;
  a future per-bucket `bits_per_zero: [f32; 8]` needs per-bucket measurement.
  (W44-206)
- Custom-order cost-gate stays for VarDCT; the JPEG path uses
  `compute_custom_orders_unconditional` (libjxl `is_nondefault`-only). (EX-J29)
- `try_dct4x8_afv` is already default-on at e ≥ 6 for EVERY strategy
  (libjxl parity; pin-probe byte-identical) — "enable AFV at e6+" specs are
  structural no-ops. The 2c Screenshot 8×8-class lift at e5 is banded to
  d ∈ [1.0, 2.0]: full-range mean missed the bytes bar (d=0.5 / d≥4 the
  block buys quality, not bytes) — don't widen the band without re-running
  the 2026-06-10 A/B. (issue #43 chunk 2c, ae62c219)

**Perceptual loop / butteraugli**
- Buttloop screenshot qf-seed scale: gate ≥ d=3.5 (main band) plus the
  W44-108 m3<24 sub-band at d∈[2,3.5) (wedge-2 lowered 30→24), AND — since
  task #12 (2026-07-14) — ALL LOW-COLOUR (m3<24) firing on BOTH bands
  additionally requires `mask1x1_p25 ≥ 95` via the `(m3 ≥ 24 OR p25 ≥ 95)`
  term (`BUTTLOOP_QF_SEED_SCALE_SUB_BAND_MIN_P25`, both the e5-7 resolver
  AND the e8+ buttloop `auto_gate_fires`). The m3 axis LEAKS pure-white-bg
  low-colour product PHOTOS at m3<24 (beard-oil 9290 m3=23.4, is_screenshot
  median=100 → +112% sub-band / +136% main-band vs cjxl, distance-
  monotonicity violation); p25 separates them (photos p25<90, low-colour
  SHIP screenshots p25≥98.9) where m3/fcbr/edge/luma_var cannot. HIGH-colour
  cells (m3≥24) short-circuit the p25 check → main-band behaviour byte-
  identical. Don't re-widen to a flat d≥2.0 (codec_wiki e8 d=3 regression),
  don't lower the 4× scale constants without re-running the W44-105 sweep,
  and don't raise the p25 floor above 95 without re-checking imac_dark
  (98.9, the worst SHIP cell). HIGH-colour main-band: a blanket exclude was
  HYPOTHESISED then REFUTED (ledger #31, task #13,
  `benchmarks/qfseed_mainband_rd_efficiency_2026-07-14.*`) — on the true RD
  curve (matched-quality efficiency, NOT the matched-distance byte-ratio)
  high-colour screenshots contain genuine WINS (w8221 2.32×, w8102 1.55×);
  the apparent misfire is the lift overshooting the requested distance to a
  point ON cjxl's curve (deliberate screenshot design), NOT RD waste. So the
  main band keeps firing on all is_screenshot (minus W44-176 + AUDIT-6 m3≥80).
  Colour does NOT separate wins from losses; luma_var/detail is the candidate
  future discriminator (RD losses cluster at lv≤1223, wins lv≥3162) pending
  the dense Pareto-scored SHIP-validated re-baseline-screenshot-gates sweep.
  (W44-105/107/108, wedge 2, task #12/#13)
- No per-block mask1x1 bimodal scaling at e5-e7 — cjxl is NOT bimodal below e8;
  candidate only for the e8+ W44-105 path. LOW threshold bisection [70, 95] is
  conclusive. (W44-145)
- CPU butteraugli strip-tile is framework-only behind `strip-tile-butteraugli`
  (default OFF, measured slower at every size); keep the `*_strip` primitives +
  50-image parity test as the harness for a future true-tile refactor. Hot
  kernels are at LLVM's autoversion ceiling — no further inner-loop SIMD chunks.
  (W44-PHASE3-B7d, B6)
- Do not cite "FMA precision" or "XYB precision" for byte movements or parity
  wedges — the A11 XYB single-FMA fix was byte-identical on every cell; bytes
  move for structural reasons. (W44-RECON-DEEP/A11, W44-66)
- CfL: Mode C (libjxl math + LS warm start) is byte-identical to LS-only on
  measured cells — opt-in API only; `EncoderStrategy::Libjxl` keeps
  `cfl_newton_libjxl_parity = true` (byte-locked). The screenshot x=0-start
  route (Phase 3) is strictly worse under the Zenjxl cost model — opt-in only,
  don't default-flip either field. (W44-AUDIT-5 P2/P3)

**Quantization / k_ac_quant**
- `K_AC_QUANT` stays 0.765 (libjxl parity) on EVERY default path. The 0.65
  default-flip is RULED OUT (2026-05-25, 29/36 cells fail ±0.30 SSIM2) AND
  the issue #25 follow-on B content-aware smooth-photo gate is RULED OUT
  (2026-06-10, 198-cell A/B: photos pass 2/126 cells, no ZenanalyzeProxies
  threshold separates pass from fail at any margin). Don't re-spawn proxy-
  discriminator chunks for k_ac_quant; the per-cell ±0.30 SSIM2 / +2 %
  butteraugli budget is binding. Remaining routes: picker-oracle re-train
  with SSIM2 axis (follow-on A) or learned per-image dispatch via the
  `LossyInternalParams::k_ac_quant` opt-in (follow-on C). (issue #25,
  CODE-HISTORY 2026-06-10)

**Tier-2 knobs / sweeps**
- Never default-flip `Tier2Knobs::auto_for_distance` or raw per-stratum optima
  on screen strata: `k1 < 0.5` or `k2 < 1.0` on screen/{very_high,high}
  re-incurs the W44-105 SHIP-cell catastrophe (-4.9 to -5.1 SSIM2). The
  membership test pins this. Next-sweep optimizers must score Pareto
  (bytes AND SSIM2) and pre-validate candidate optima on SHIP cells.
  (W44-PHASE4-S2-validate, refit-c1/c2, W44-228c1)
- Sweep infra: keep the launcher corpus pre-flight AND the worker 3-retry fetch
  (both load-bearing, different failure classes); retries stay ≤ 5. Tier-2
  validation must check BOTH SHIP-cell protection and anchor-arm shifts.
  (W44-PHASE4-S1h, S2f)
- `encoded_bytes` is u32 in sweep sidecars — cast to int64 before diff
  arithmetic. (S2f)

**DC tree**
- `DC_TREE_VARIABLE_TRIAL_MIN_EFFORT = 8` and
  `DC_TREE_VARIABLE_PREDICTOR_FULL_MIN_EFFORT = 9` are libjxl-parity gates;
  the +0.7-1.6 % byte cost at e8 is the parity cost of `Predictor::Best`,
  not a regression. Don't move either without re-measuring both wall and
  bytes. (W44-171/172)

**JPEG-in-JXL path**
- DC stream uses kWPFixedDC (`Predictor::Weighted`); never revert to gradient
  (`JPEG_GRADIENT_DC` is A/B-only) and never swap
  `collect_dc_tokens_wp_region_jpeg` for the full-res VarDCT collector — the
  per-channel subsampling shifts are load-bearing. Accurate ANS population cost
  stays JPEG-scoped (global default-flip changes modular-lossless output).
  (EX-J31/J30)
- `frame_header.all_default` is STRUCTURALLY IMPOSSIBLE for JPEG frames
  (kSkipAdaptiveDCSmoothing flag, kYCbCr color_transform, loop_filter
  non-defaults); never experiment with `flags = 0` — decoder DC smoothing
  breaks byte-exact reconstruction. (all_default note)
- EX-J15 per-(channel × band) block contexts RULED OUT (+0.05 % on 200 files;
  16-ctx wire cap + per-cluster header cost); keep the
  `jpeg_dc_quantile_ex_j15` helper + env hook as the A/B harness; don't add
  more block contexts. (EX-J15)
- Pad-bit tracking: only `decode_coefficients_with_jbrd_metadata` carries the
  tracking cost; never default-enable in zenjpeg's legacy entry, never
  unconditionally set `has_zero_padding_bit`. (task #11)
- LZ77 on JPEG AC streams: RLE all-or-nothing is win-neutral, greedy regresses;
  libjxl does call ApplyLZ77 — the overhead exceeds savings on post-CfL AC.
  Skip without new evidence. (10-agent diff, A9)
- Lossy recompression: PreserveJxl wins only at gentle targets (crossover ≈
  zensim 88-89 for photos; bpp-based routing REFUTED at N=12); pixel path wins
  medium/aggressive. Never compare at aggressive targets without the `px_valid`
  flag; chroma ≤ 1.5× luma; scale-proportional deadzone stays (strict Pareto
  win); closed-loop targeting requires encode-measure-adjust, never fixed
  scale. Productization home is zenjxl (scorer callback), NOT jxl-encoder.
  (JPEG lossy router)

**Modular / lossless**
- Owned-clone tree-learn fallback regresses at every measured size (the
  2026-05-17 audit numbers are stale); keep the dispatch infra as the A/B
  harness, default stays borrowed-view. (issue #42)
- Parent-histogram subtraction (hist-sub, c19815ff) pays ONLY on the
  lossless-photo split path: post-W44-171/172 the lossy cells' whole
  find_best_split ceiling is 0-3.9 % of wall (don't respawn hist-sub-for-
  lossy), and lossless SCREENSHOTS are not split-bound (1.4-6.9 % — their
  cost is collect/gather/WP; see Live follow-ons). Single-sample RSS deltas
  on WSL2 swing ±120 MB from glibc adaptive-arena policy — pin
  `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072` for memory A/Bs.
  (issue #64 chunk 1, benchmarks/perf_hist_sub{,_lossless}_2026-06-10.meta)
- Lossless-graphics gap vs cjxl (#74 "match always") — RE-MEASURED
  2026-08-21 (`benchmarks/noaa_lossless_refresh_2026-08-21.tsv`, 22
  train-legal NOAA docs, HEAD b21cba67): **at e7 we now WIN every doc**
  (−1.1 % to −75 %, 5336 −11.8 %, 5340 −9.5 %, 5342 −2.9 %; the July
  "+29.6 % e7 residual" claim is stale — property-pre-quant parity thread
  CLOSED, no work needed). At e5 a small tail still loses: 5332 +11.9 %,
  5342 +9.5 %, 5338 +7.2 %, 5326 +8.1 %, 5336 +0.5 % (self-repair
  holding). Historical record (2026-07-14, pre-self-repair-arc): sparse
  docs > ~80 % top_color_pct lost 5336 +26 %, 5340 +6 %, 5342 +9 %;
  lower won (5334 −53 %, 5324 −25 %), palette-friendly won big (8222
  499-colour −68 %). Four global levers RULED
  OUT (2026-07-14, don't re-try): context-map ANS+LZ77 writer is ALREADY at
  parity; histogram cap 128→256 is a NO-OP (clustering picks cost-optimal
  ≤128; CLUSTERS_LIMIT=256); global tree-threshold sweep helps ONLY the 5336
  outlier while regressing photos + not helping 5340/5342; trial-encode-two-
  trees is LOW-EV (1/13 img, ~1 pp). 5336 attribution: over-split global tree
  (2059 nodes / 10.9 KB LfGlobal vs cjxl 312 B) that ALSO codes residuals
  +21 % worse. A zenanalyze proxy PREDICTS the loss but there is NO winning
  alternative modular strategy to route to (unlike lossy Libjxl↔Zenjxl) —
  fix needs modular predictor/context RD on high-dominance content, not a
  knob. (#74, HYPOTHESIS_LEDGER #24, benchmarks/lossless_graphics_gap_probe_2026-07-14.*)
- cjxl v0.11.2 fails ("Getting pixel data failed") on some Display-P3 /
  Android-EXIF PNGs (e.g. imazen-26 8025) → produces NO file. Sweeps that
  stat() a reused temp path then compare vs a STALE prior output (observed:
  a bogus 8025 "+265 %" that was 8248's cjxl size). Record cjxl_fail, never
  compare; filter scoreboard cjxl-dominant counts for this. (2026-07-14)

**Process**
- Gate changes: update `gate_registry.rs` macro metadata + ALL_DIVERGENCE_ENTRIES
  + LIBJXL_DIVERGENCES.md row + EXPECTED_DIVERGENCE_GATE_COUNT together (the
  W44-194 drift test enforces). New gates go in the macro, not hand-written
  api.rs fields. (W44-193/194)

### Live follow-ons

- **#24 lossless-docs stride-aliasing — MECHANISM FOUND + COST-BASED SELF-REPAIR
  DEFAULT-ON for the lossless path (2026-07-15, ledger #33, task #14).** Root cause (source-verified
  vs libjxl): our modular tree-learning sample gather uses a FIXED stride
  (`compute_gather_stride_from_profile`) that ALIASES against periodic document
  text-line spacing → non-representative sample → catastrophic tree over-split
  (5336 noaa-leslie e5 +36.9 % vs cjxl; the same content compresses to +0.3 %
  pixel-exact under a non-aliasing sample). libjxl `CollectPixelSamples`
  (`enc_ma.cc:972`) samples RANDOMLY (geometric), which cannot alias. Two opt-in
  fixes shipped, default BYTE-IDENTICAL: (1) `JXL_TREE_SAMPLE_RANDOM` (d0a304aa,
  `tree_learn.rs`) = unconditional randomized gap — fixes 5336 but regresses
  photos +0.7pp (NOT clean). (2) `JXL_TREE_SELF_REPAIR` (task #14,
  `section.rs` global tree learner + `encode.rs` predicates) = learn tree_a
  (fixed) + tree_r (de-aliased re-gather), keep whichever codes the ACTUAL
  residuals cheaper → STRICT byte win, ZERO regressions (5336 +36.9→+0.3 %, all
  13 photos byte-IDENTICAL). **Two refutations (do NOT re-try): node count does
  NOT discriminate aliasing** (5336 641 vs 613 nodes; photos over-split to
  1500-7859) and **ideal per-context entropy does NOT** (`estimate_token_cost`
  sees 1.7 % of the real 27 %) — ONLY the ≤96-histogram-clustered ANS cost
  (`build_entropy_code_ans_with_options`) reproduces the gap; a clustered/ideal
  ratio > 1.05 is the cheap aliasing pre-filter. **NOW DEFAULT-ON (task #14 flip,
  wins rd AND rdtime)**: (a) perf-opt (67d1dc91) caches the ratio gate's
  clustered_a build and reuses it in the downstream Step-4
  `build_entropy_code_ans_with_options` when tree_a wins (LZ77 off at e5/e6 ⇒
  byte-exact; photo wall 1.3×→1.12×), so non-aliased content pays ~0; (b) it is a
  plain `EffortProfile::tree_self_repair` field (`true` in `lossless_reference`,
  `false` in `lossy_reference`), `JXL_TREE_SELF_REPAIR=0/1` a runtime override.
  **NOT a gate-registry gate** — the modular/lossless path reads NO
  strategy/`ResolvedImprovements` (lossless output is STRATEGY-INVARIANT;
  `EncoderStrategy::Libjxl` makes no distinct lossless bitstream — we already
  beat cjxl lossless, there is no libjxl-lossless parity target), so "off for
  Libjxl" is vacuous and a per-strategy gate would be non-functional; drift count
  stays 32. **NO hash-lock regen needed**: every fixture is lossless-e7/e9
  (stride-2, no fire), lossy (field OFF), or below the 256-node floor → 53/53
  byte-identical; 5/5 Libjxl byte-lock (LossyConfig) untouched. Validated (release,
  default features, no env): 5336 e5 288293→211098 (−26.8 %, djxl AE=0 pixel-exact),
  photos byte-identical, `=0` recovers 288293. **e7 residual — ×0.1 lever
  REFUTED at source (2026-07-15)**: self-repair fires only at e5-e6 (tree-lift
  stride ≥ 8); e7's base path is stride-2. The prior "fix e7 with a
  `tree_sample_fraction` ×0.1" idea is WRONG — libjxl's `GatherTreeData`
  (enc_encoding.cc:105/629, `pixel_fraction = min(1.0, nb_repeats)`, NO ×0.1) is
  the tree-learning sample gatherer and OURS MATCHES it; the ×0.1
  (`CollectPixelSamples`, enc_ma.cc:973) feeds only `PreQuantizeProperties`
  (property BOUNDS), not the tree-learn gather. Applying ×0.1 would DIVERGE +
  under-sample → regress. So the e7 5336 +29.6 % residual is NOT a tree-sample-
  density divergence; only un-refuted e7 thread is property-pre-quant density
  parity (source-verify first). `benchmarks/lossless_{stride_alias,self_repair_ab}_2026-07-15.*`.
- **cfl_two_pass line-art over-fit — FIXED via keep-best CfL Pass-2 guard
  (2026-07-14, #74 #1 SDR gap).** Aliased line-art (imazen-26
  `7000-plots/aliased-*`) lost +19..52 % to cjxl at **e7 d0.5** while our OWN e6
  BEAT cjxl (−0.7 %) — an effort-monotonicity violation. ROOT CAUSE (proven,
  `benchmarks/cfl_two_pass_lineart_ablation_2026-07-14.{tsv,meta}`, harness
  `examples/line_art_cfl_probe.rs`): the effort-≥7 CfL Pass-2 refit
  (`chroma_from_luma::refine_cfl_map`) minimizes an L2 chroma residual then
  UNCONDITIONALLY overwrites the Pass-1 multiplier; on aliased color edges the
  L2-optimal multiplier is dominated by HF aliased-edge energy → a chroma field
  harder to entropy-code than Pass-1 (min L2 residual ≠ min coded bits).
  **DO NOT try a content-proxy gate on the 4 proxies** (m3/fcbr/edge/luma_var) —
  RULED OUT: same-seed aliased/anti-aliased pairs are proxy-IDENTICAL but
  opposite (7022 −16.5 % vs 7076 SAME SEED +0.1 %; fcbr ranges fully overlap).
  The discriminator is ALIASING, invisible to those proxies. **FIX SHIPPED
  (task #10): the keep-best CfL Pass-2 guard** — `EffortProfile.cfl_keep_best`
  (Section C gate, ON for Zenjxl/Aggressive/LeanFaster at e≥7, OFF for
  `EncoderStrategy::Libjxl`). Per tile, `refine_cfl_map` compares the Pass-1 vs
  Pass-2 multiplier by an ANS-shaped chroma-AC coding-cost proxy (dead-zoned
  residual count + `log2` magnitude, `CFL_COST_DEADZONE=0.58`,
  `CFL_KEEP_BEST_MARGIN=0.02` confidence moat) over the ACTUAL AC-strategy
  coefficients and keeps the cheaper. Content-AGNOSTIC (no proxy discriminator).
  Quality-neutral: CfL recon error is bounded by ±0.5 quant-step regardless of
  the multiplier (signaled + inverted exactly at decode) → decoded butteraugli
  equal-or-better on all 5 quality-checked cells. A/B
  (`benchmarks/cfl_keep_best_lineart_ab_2026-07-14.{tsv,meta}`): aliased
  line-art saves mean **+17.7 %** (max +31.3 %), ≤+1.5 % elsewhere, 0/24
  real-content imgs regress. **DON'T disable it or move `CFL_KEEP_BEST_MARGIN`
  below 0.02** without re-running the A/B — the margin makes the tile decision
  platform-stable (float-noise ≪ 2 % moat) and avoids proxy-noise flips that
  regressed ~190-byte synthetic fixtures (Huffman/16-bit/error-diffusion, +1..2
  bytes, documented-acceptable). Libjxl byte-lock 5/5 UNCHANGED; hash-locks
  regenerated. Ledger #28, LIBJXL_DIVERGENCES.md `EffortProfile.cfl_keep_best`.
- **HDR e8/e9 buttloop over-refinement — FIXED (2026-07-15, #74 task #11,
  ledger #34).** SEPARATE from and LARGER than the e5/e7 diffuse gap below (the
  2026-07-14 scoreboard sampled only e5/e7 = no loop, so it MISSED this). At
  e8/e9 the perceptual quantization loop OVER-REFINES HDR PQ/HLG +100..500 %
  bytes vs cjxl across d1..d4 (worst at low-d: 1493 d1 base 9306 → loop 52729,
  5.7×). Root cause (source-verified): the default HDR loop steers on VDP2-lite
  (`HdrLoss::Auto`→`Vdp2`, `hdr_metrics.rs:176`), but Step 6
  (`perceptual_loop.rs:2448`) reduces VDP2's diffmap with butteraugli's
  `k_tile_norm=1.2` and compares against butteraugli's `effective_target=4` — a
  SCALE MISMATCH → td_median ~8 vs 4 → the loop cranks quant far past target.
  BOTH metrics over-refine (forced-Butteraugli 53712 > Vdp2 41079 on 1230.q1
  e9 d4), so it's structural, not a single-metric miscalibration. **Two REFUTED
  sub-hypotheses (don't re-chase)**: intensity_target is NOT the driver
  (`JXL_METRIC_INTENSITY_OVERRIDE` {80..10000} byte-identical — butteraugli-params
  is dead code on the default Vdp2 path; our butteraugli opsin MATCHES libjxl:
  log gamma + ×intensity_target + fixed kGlobalScale); libjxl does NOT scale its
  loop target for HDR. **Our NO-LOOP base BEATS/MATCHES cjxl e9 on 6/6 crops** (d4
  wins bytes AND VDP2), so the loop is pure harm. **FIX**:
  `is_hdr_pq_hlg(layout, color_encoding) AND resolved hdr_loss == Vdp2` →
  `enc.butteraugli_iters = 0` (api.rs encode_lossy + streaming) = reproduces
  `--no-butteraugli` EXACTLY, verified byte-identical on 18 cells; SDR untouched
  (hash-locks 48/48, Libjxl byte-lock 5/5, drift green; NOT a gate-registry gate).
  Explicit `with_hdr_loss(Butteraugli)` on HDR keeps the loop (escape hatch,
  documented caveat). Test `hdr_vdp2_chunk4_auto::default_hdr_path_skips_buttloop`
  (single- + multi-group, non-vacuous). Divergence LIBJXL_DIVERGENCES.md §B;
  `benchmarks/hdr_buttloop_blowup_2026-07-15.*`, `hdr_loop_rd_sweep_2026-07-15.tsv`.
  FUTURE (low EV): a VDP2-native block reducer + VDP2-scale target could re-enable
  an HDR loop, but its best case only matches the base that already wins.
- **HDR-lossy RD gap — CHARACTERIZED as a HARD diffuse RD-efficiency gap
  (2026-07-14, #74 task #11, ledger #29).** 25 scoreboard real losses: cjxl-rs
  spends +2..5 % bytes for HDR quality 3-metric-TIED (pq_butteraugli/VDP2-lite/
  CVVDP) at e5/e7 d2-4 on PQ 512² crops. **RULED OUT — the calibration-nudge
  hypothesis**: the distance knob is EXHAUSTED (`benchmarks/hdr_rd_sweep_1230q1_
  2026-07-14.log`) — sweeping OUR d4.0→5.5, at iso-vdp2-quality ours is still
  **+4.5 %** bytes vs cjxl; coarsening degrades vdp2/cvvdp faster than it saves
  bytes. cjxl's HDR RD curve genuinely dominates ours; this is NOT a quant offset
  a distance-nudge closes. (pq_bfly is a NOISY max-norm — non-monotonic across
  distance — use vdp2/cvvdp as arbiters.) **Section split** (`benchmarks/hdr_
  section_split_2026-07-14.tsv`, jxl-oxide `--with-offset`, 4 cells): LfGlobal
  +12..27 %, DC +3..7 %, AC +1..4 % — **95 % of the gap is in DC+AC COEFFICIENT
  coding (diffuse), only ~10 % LfGlobal signaling**. Our LfGlobal `context_tree`
  = 983 bits (~123 B) at 512² on both SDR and HDR
  (`benchmarks/hdr_lfglobal_dctree_2026-07-14.tsv`). **CORRECTION — the DC tree
  is ALREADY at libjxl parity; do NOT chase a "compact DC tree" lever**: at
  effort 4-7 the encoder uses `build_wp_fixed_dc_tree` (`dc_tree_learn.rs:2751`),
  our BYTE-FOR-BYTE port of libjxl's kWPFixedDC `MakeFixedTree` (same
  `WP_FIXED_DC_CUTOFFS` + same `min_gap = 8*(14-log_px)` size pruning). The tree
  DOES shrink with size (313 tokens at 512², 82 at 64²; e5==e7). The static
  313-token `CONTEXT_TREE_TOKENS` (`context_tree.rs:33`) is only the
  `write_context_tree` fallback these encodes never hit. So the LfGlobal +97 B
  vs cjxl is NOT a bloated-tree holdover — it is UNATTRIBUTED signaling (our
  `dc_entropy_code` DC ANS histograms = 222 B and/or the AC-metadata prefix), not
  separable from cjxl's tree-vs-entropy split without cjxl's sub-breakdown.
  `dc_tree_learning` goes the WRONG way (DC_TREE_VARIABLE at e8 COSTS +0.7-1.6 %).
  **VERDICT: NO clean lever** — DC tree at parity, LfGlobal ~10 % + unattributed,
  the 90 % is diffuse DC/AC coefficient coding. Only next-attempt: a DC/AC
  coefficient-coding RD study on PQ content (histogram clustering + context
  modeling + DC ANS efficiency) — a multi-week research item, not a chunk.
  **UPDATE 2026-07-15 (post e8/e9 fix, ledger #29+#34): the residual e5 gap is a
  distance-calibration OVERSHOOT, not a diffuse coding deficit.** With the
  buttloop-skip fix, e9 WINS (−3..−6 %), e7 ~parity, and only e5 remains (+3.2 %
  matched-distance). But at e5 d2 OURS is FINER quality than cjxl on 12/12 crops
  at the same distance (overshoot); at iso-VDP2-quality ours is RD-competitive
  (1493 −27 %, others ~parity), so the +3.2 % is largely TRADEOFF_QUALITY (same
  shape as ledger #31/#32 screenshots), which the 3-metric guard demotes from
  REAL_LOSS. The 1493 "+47 % AC" is the overshoot spending on finer AC, NOT coding
  inefficiency. The part-A iso-quality deficit (+4.5 %) is confined to AGGRESSIVE
  d4–5.5. **No clean RD lever remains**; a matched-distance lean-out would be a
  content-adaptive e5-HDR quant coarsening (dense HDR-scoped sweep, lateral RD
  move, no free bytes). `benchmarks/hdr_e5_overshoot_2026-07-15.*`.
- **Buttloop memory reduction — measured options (2026-06-23, #93 follow-up).**
  Two threads investigated at the user's request:
  1. **jxl↔butteraugli XYB buffer-sharing is NOT viable** (definitive,
     source-grounded — don't re-attempt). butteraugli's XYB
     (`opsin.rs::opsin_dynamics_image`) is a **spatially-adaptive** opsin: it
     blurs RGB at σ=1.2, derives a per-pixel sensitivity from the BLURRED
     neighborhood, applies it to the original RGB, and scales by
     `intensity_target`. jxl's encoder XYB (`vardct/xyb.rs::convert_rows_to_xyb`)
     is **pointwise** + intensity-scaled. Different transforms → butteraugli
     needs the linear RGB regardless (for the blur), so the encoder's XYB
     planes can't feed it. **Linear-RGB planar is already the shared interface**
     (`compare_linear_planar_into`); the GPU backend already uploads linear
     planes directly (W44-phase3-B4). The only redundant pure-copy is jxl's
     reference deinterleave (`linear_rgb` → `ref_r/g/b`), and eliminating it is
     a net loss with today's API (`new_linear` interleaved CLONES the source
     via `rgb.to_vec()`, vs `new_linear_planar`'s `source: None`). Net: no
     meaningful cross-crate copy win.
  2. **Strip-wise butteraugli is a real memory lever** (measured,
     `imazen/butteraugli` `benchmarks/strip_vs_full_mem_2026-06-23` via the new
     `strip_vs_full_mem` harness): one-shot strip vs full is **2.84–2.97×
     smaller RSS** (3.10→1.09 GiB @16 MP, 6.47→2.26 GiB @36 MP), **BIT-IDENTICAL
     score**, and ~equal-or-faster one-shot. The catch: the buttloop is
     **warm-ref** (precompute once, compare 2–4×) — full amortizes the
     precompute across iters, standalone strip re-walks it, which is why the
     buttloop-wired strip-tile was "measured slower" (W44-PHASE3-B7d). The
     unblock is a **warm-ref strip** (cache per-strip reference precompute,
     reuse across the 2–4 compares) — the unfinished true-tile refactor. For
     the memory-bound `modes_full` sweep this is the highest-leverage per-encode
     win; for latency-sensitive single encodes, weigh the warm-ref re-walk cost.
     heaptrack (`benchmarks/issue93_*`, 12 MP e8) confirms butteraugli's
     working set (precompute + per-compare opsin-blur/diffmap/malta) is ~40–50 %
     of the e8 peak — a strip path bounds all of it.

- **Encoder memory follow-ups (from the 2026-06-13 12 MP HDR cap fix).**
  Three measured-but-unfixed items, deprioritized vs the cap+over-count fix
  that shipped:
  1. `estimate_peak_memory_bytes` — RESOLVED 2026-06-14 by calibration, NOT
     a guessed constant. The old term-by-term model modelled only the
     dimension-driven planes (linear_rgb + XYB + quant_ac) and MISSED the
     entropy-coding/transient working set → ~4× under on lossy, ~14× under on
     lossless (it treated tree-learning as ~8 B/px; measured ~440). Replaced
     by `crate::heuristics::estimate_encode` (new module, zenwebp per-codec
     pattern): `EncodeEstimate { peak_memory_bytes_min / peak_memory_bytes
     (typical) / _max, time_ms, output_bytes }`, model `input + fixed +
     bpp(path, effort)·pixels` with effort STEP jumps (lossy buttloop e≥8,
     lossless tree-learning e≥7) + content mult (min 0.85 / max 1.8). Both
     configs' `estimate_peak_memory_bytes` delegate to `..._max`; new
     `estimate_encode(w,h,layout)` exposes the full breakdown. Constants
     calibrated from the MARGINAL working set (mem_probe `VmHWM` delta,
     12 MP-anchored — no extrapolation): lossy 75/87/300 B/px (e5/e7/e8+),
     lossless 88/135/440 B/px (e5/e6/e7+). Provenance
     `benchmarks/mem_peak_calibrate_libharness_2026-06-14.tsv`; harness
     `scripts/mem_peak_calibrate.py` + `examples/mem_probe.rs`. Bit depth
     barely moves it (8 vs 16-bit ≈ 75 vs 72 B/px — f32 internals dominate,
     only the input buffer carries bpp). Commits ntszwlux (module) +
     ltqvptqw (rewire) + 44a929b6 (refinement). The e8/e10 points, the RGBA
     alpha term, and the content-spread percentiles were settled by a
     focused 7-class × 1024² × e7/8/9/10 × rgb/rgba grid (112 cells, ~4 min,
     benchmarks/mem_peak_quick_2026-06-14.tsv + scripts/mem_peak_fit.py):
     lossy e8/e9/e10 are one flat band at 12 MP (lossy is sub-linear in
     size); lossless is ~size-independent, adds an e10 band (~620 B/px,
     +35 %) + a +230 B/px alpha term (lossy+alpha is noise). estimate_encode
     now takes has_alpha. No further sweep warranted — a ≥50-img/class run
     was tried and explicitly cut as overkill; the 7-class grid pins every
     band and MULT_MAX stays 1.8 as the conservative cap.
  2. e7 real-RSS gap vs cjxl — LARGELY ADDRESSED 2026-06-13. Root cause
     (heaptrack peak attribution): `compute_epf_sharpness` ran its 2-3
     candidates via `parallel_map`, each cloning base_recon + scratch
     (~432 MB/candidate at 12 MP) → ~1.3 GB held at once, the single
     biggest real-memory consumer. Fixed: sequential candidates + reused
     buffers + strip-parallel `compute_block_l2_errors` (byte-identical,
     hash-locks 48/48). 12 MP e7 RSS 2.07 → 1.55 GB (−27%, gap to cjxl
     1.23 GB roughly halved); e9 2.15 → 1.61 GB. Wall +7 % e7 / +3 % e9
     (sequential apply_epf loses candidate overlap; the residual gap to
     cjxl is the entropy-coder/token materialization, still unattributed).
  3. VDP2 / butteraugli ref-plane dedup — SHIPPED 2026-06-13 (byte-
     identical). The VDP2-lite metric is in jxl-encoder
     (`hdr_vdp2_lite.rs`), NOT zenmetrics (earlier note was wrong). Added
     `compare_vdp2_interleaved` (reads the ref straight from interleaved
     `linear_rgb` — bit-identical to the planar path, pinned by
     `vdp2_planar_and_interleaved_bit_identical`), so the VDP2 path no
     longer deinterleaves 3 planar reference planes; the butteraugli path
     now builds them only transiently for `set_reference` and drops them
     before the loop. Frees ~144 MB of buttloop-PHASE working set on both
     paths. NOTE: headline peak RSS barely moved (e9 1.61 → 1.59 GB, −22
     MB) — the encoder's peak is the EPF sharpness search (a different
     phase), not the buttloop, so removing buttloop-phase buffers doesn't
     lower the high-water mark. The win is reduced memory pressure during
     the loop + removed redundancy, not peak. Further PEAK reduction must
     target the EPF search (base_recon + recon clone + step0 buffers).
- **Allocation-count vs libjxl — TWO NAIVE FIXES RULED OUT 2026-06-13.**
  We do ~1.45 M allocs at 12 MP e9 d4 vs libjxl's 831 k (1.75×), but
  *temporary* allocs already match (104 k ≈ 105 k), so the excess is
  long-lived tiny-byte allocations — likely NOT on the wall-time critical
  path (un-profiled). Heaptrack alloc-COUNT attribution: entropy
  `AccumulatedAnsData` (merge/add_tokens/Histogram) ~65 %, `dot_detection`
  14 %, DC `find_best_split` 12 %. Two byte-identical "plug-in" fixes were
  tried and BOTH REGRESSED, do not re-try:
  (a) `value_freqs`/`lz77_freqs` `BTreeMap<u32,u32>` → `hashbrown::HashMap`:
  **+11 %** allocs. Rust's `BTreeMap` is a B-tree (~11 entries/node) so a
  small map = 1 node alloc; hashbrown resizes its buffer 3-4× as it grows.
  For "many small freshly-built maps" BTreeMap wins.
  (b) pre-reserving each accumulator `Histogram`'s counts to
  `ANS_MAX_ALPHABET_SIZE` (256): **+132 %** allocs. `AccumulatedAnsData::new`
  builds `num_contexts` histograms PER GROUP but most contexts are EMPTY in
  a given 256² group — lazy `Vec::new()` costs 0 allocs for an empty
  histogram; pre-reserving forces a 1 KB alloc on every empty one.
  The real lever is architectural — the sparse per-group accumulator is
  rebuilt per group; a pool/reuse (the existing `Histogram::clear`/
  `copy_from` infra retains capacity, but `BTreeMap::clear` frees nodes) or
  a sparse representation, NOT a data-structure swap. Gate any such work on
  a callgrind malloc-fraction profile first — the temporary-alloc parity
  says the allocator is probably not the bottleneck.
- **Lossless screenshots wall — ARC COMPLETE 2026-06-10** (#41 closing
  ledgers on the issue): B1 gather row staging, B2 batched traversal,
  lane-per-predictor, WP fusion, radix cmp (chunk 1, `a47fabc4`),
  estimate_cost LUT, capacity reserves all SHIPPED byte-identical; WP
  batching / inline-dedup-≥8MP / pair-sort / rayon-entry variants
  measured-REJECTED with committed data. **MSD radix bucketing (chunk 2)
  was NOT shipped** (corrected 2026-06-13): left as an unfinished orphan
  WIP from 2026-06-10 (recovered + labeled `STRANDED WIP`; sibling to the
  rejected pair-sort, never benched to conclusion) — only chunk-1's
  inline comparator landed. Do not cite bucketing as
  byte-identical-shipped. Day deltas: lossless e7 −8…−12 % on screens/docs
  (more on photos with hist-sub), e5 −9…−12 %, lossy e3/e4 ≈ −25 %
  (dump env-hook gates + classifier skip). Bench via
  `scripts/bench_lossless_ab.py` on
  `benchmarks/lossless_bench_set_2026-06-10.tsv` (43 picks, feed
  `bench_input`). Remaining symbols are pinned core work — see
  `perf_gather_profile_2026-06-10.meta` addenda + the #41 close-out.
- **Lossy-low hygiene rule (2026-06-10)**: DIAGNOSTIC dump env hooks go
  behind the `__env_var_diagnostics` cargo feature (compiled out of
  default builds entirely — six dump modules + the inline dump sites now
  are; dump-driver examples declare `required-features`). Inside the
  feature, keep the once-presence OnceLock gate so set-but-unused hooks
  stay cheap. BEHAVIOUR-override env hooks (gate fallbacks, dispatch
  disables, buttloop scales) stay runtime — the override test contract
  and A/B harnesses run against production builds. Pre-encode analysis
  passes MUST be gated/lazy on their consumers' bands (ZenanalyzeProxies
  precedent: 24 % of e3 CPU computed-and-discarded). New hooks that
  probe env or sweep pixels per block/image need a profile cell at
  lossy e3 before landing.
- **BestSplit side-costs rider — SHIPPED 2026-06-10** (byte-identical;
  six engine sites consume sweep-carried best_l/r_cost; permanent
  debug-asserts verify carried == recomputed at every site). Quiet-machine
  wall: lossless photos -1.3..-1.8 % at 1T (e7+e9), -0.8 % 8T; controls
  within noise (`benchmarks/perf_bestsplit_rider_2026-06-10.quiet.tsv`).
- **Dispatch chunks 2b/2d** (issue #43): 2d = fine_grained_step at e9 on
  ≥4 MP (Pareto sweep first); 2b = DCT64 distance-gate expansion on medium
  (measure picks first). 2a + 2c shipped.
- **imazen-26 re-baselining**: validate the 2c screenshot lift on the
  8000/8100 strata; longer-term re-baseline the screenshot-class gates
  (W44-105/107/108 thresholds were calibrated on gb82-sc's 10 images only).
  The k-means lossless set covers 3/5 web-screenshot viewports — add
  2880×1800 retina cells explicitly when validating screenshot gates.
- **JPEG lossy productization** (relative/inferred quality targeting + the
  quality-threshold router): build in `~/work/zen/zenjxl` with a scorer
  callback; jxl-encoder keeps only the PreserveJxl coeff path +
  `--jpeg-coarsen` + `coarsen_policy`. Harness to port:
  `benchmarks/jpeg_lossy_closed_loop_2026-05-28.py`.
- **Residual JPEG-transcode gap** +0.115 % vs cjxl-e7 (200-file): distributed
  micro-overhead, no single structural lever. Localize any future gap with
  `jxl-oxide info --with-offset` section diffs first.
- **Phase 8-zensim**: K_TILE_NORM refit (65 % → 85 %+ Pareto target);
  zensim-gpu GPU-native diffmap kernels (currently CPU-fallback).
- **cvvdp-cpu structural perf**: strip-pipeline + f16 (150 ms → 50 ms at 1024²)
  in the zenmetrics repo.
- Open issues: #64 (DC-tree hot-path: hist-sub chunk 1 SHIPPED, -20.7 %
  mean on lossless photos; remaining = the screenshots item above +
  MABSplit), #43 (dispatch 2b/2d), #41 (gather/collect/WP — option C
  list), #25 (k_ac_quant — follow-ons A/C only; default-flip AND the
  follow-on B smooth-photo proxy gate are both RULED OUT, see
  "Quantization / k_ac_quant" above), #45 (e10/e11 smart modes), #24
  (lossless e9 picker — re-baseline EV after hist-sub dropped e9 walls
  10-38 %).

### Reference findings (stable)

- **CfL on DC/LLF**: AC-only CfL is CORRECT — the decoder's
  `LowestFrequenciesFromDC` overwrites LLF after DequantLane, so
  coefficient-level CfL on LLF is discarded; full CfL scores SSIM2 ≈ -40.
- **EX-J1 Steiner ANS + EX-J2 per-context LZ77 distance contexts**: both
  wire-format-impossible (alias method spec-mandated; single distance context
  hardcoded in every decoder). Don't re-investigate; details in CODE-HISTORY.md.
- **Picker oracle sweeps (2026-04-30)**: TSVs archived at
  `/mnt/v/output/jxl-encoder/picker-oracle-2026-04-30/` (165k lossless +
  610k lossy rows); reproducible via `examples/{lossless,lossy}_pareto_calibrate`.
- **alpha_distance**: bit-exact MAE parity with cjxl `--responsive=0`; the gap
  vs cjxl-default is the unported Squeeze-on-extras path (multi-week), then
  ChannelCompact for extras, then the entropy-coder gap on alpha residuals.

## Build Commands

```bash
# Build
cargo build

# Test
cargo test

# Clippy
cargo clippy -- -D warnings

# Format
cargo fmt

# RD regression test (6 images x 2 distances, ~3 min debug)
just rd-regression

# Regenerate hash lock sidecar after intentional encoding changes
just update-hashes

# Compare quality vs cjxl (CSV-backed, CID22 images, Rust butteraugli + ssim2)
just quality-compare

# 6-panel visual comparison (ours/cjxl side by side, errors below, delta bottom-left)
just compare-visual source.png ours_decoded.png cjxl_decoded.png 4.0 [output_dir]
```

## Pre-Commit Checklist

Run before every commit:
```bash
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo test
```
(`--all-targets` is load-bearing: test/example/bench targets carry their own
feature unification — the 2026-07 zenjpeg nightly break was only visible
there, and CI enforces the same flags.)

Run `just rd-regression` after any change to encoding, quantization, or entropy coding
to verify no quality/size regressions.

## Workspace Structure

```
jxl-encoder-rs/
├── jxl_encoder/             # Main encoder library
│   ├── src/
│   │   ├── api.rs             # Public API (LossyConfig, LosslessConfig, EncodeRequest)
│   │   ├── bit_writer.rs      # Bitstream writing
│   │   ├── entropy_coding/    # ANS, Huffman, HybridUint, LZ77, tokens
│   │   ├── headers/           # File and frame headers
│   │   ├── icc.rs             # ICC profile encoding
│   │   ├── image/             # Image buffer types
│   │   ├── modular/           # Modular (lossless) encoder + FrameEncoder
│   │   ├── vardct/            # VarDCT (lossy) encoder
│   │   │   ├── perceptual_loop.rs      # Metric-agnostic quantization loop (was butteraugli_loop.rs)
│   │   │   ├── perceptual_backend.rs   # PerceptualBackend trait + construct_backend dispatch
│   │   │   ├── cvvdp_backend.rs        # Gpu/CpuCvvdpBackend (feature cvvdp-loop)
│   │   │   ├── zensim_backend.rs       # Gpu/CpuZensimBackend (feature zensim-loop)
│   │   │   ├── cvvdp_targets.rs        # CVVDP per-distance JOD calibration table
│   │   │   └── zensim_targets.rs       # Zensim per-distance calibration table
│   │   └── error.rs           # Error types
└── jxl_encoder_cli/         # Command-line tool (cjxl-rs)
```

**Note**: `vardct/butteraugli_loop.rs` was renamed to `perceptual_loop.rs`
in the multi-metric refactor (2026-05-25). The historical Investigation
Notes below still reference the old name — those are dated records, not
live file paths.

## Porting Guidelines

### Reading libjxl Encoder Code

Key files to port from `libjxl/lib/jxl/`:
- `enc_bit_writer.cc/h` - BitWriter (DONE)
- `enc_ans.cc/h` - ANS entropy encoder
- `enc_huffman.cc/h` - Huffman encoder
- `enc_modular.cc/h` - Modular (lossless) encoder
- `enc_frame.cc/h` - Frame assembly
- `enc_group.cc/h` - Group encoding
- `enc_transforms.cc/h` - Color transforms
- `enc_ac_strategy.cc/h` - AC strategy for VarDCT
- `enc_xyb.cc/h` - XYB color space conversion

### Matching Patterns with jxl-rs Decoder

- Use similar module structure to jxl-rs decoder
- Match error types and Result patterns
- Reuse types from decoder where possible (headers, color encoding)
- BitWriter should be symmetric with BitReader

### Test Strategy

1. Unit tests for individual components
2. **Round-trip tests with jxl-rs** (PRIMARY): encode -> decode with jxl-rs crate
3. **Round-trip tests with djxl**: encode -> decode with libjxl CLI for reference compatibility
4. Round-trip tests with jxl-oxide: encode -> decode as secondary validation
5. Parity tests: compare byte output with libjxl reference
6. Use test images from `~/work/codec-corpus/`

**Corpus note (2026-06-10)**: `~/work/codec-corpus/imazen-26/` is the new
stratified 21-class corpus (6.6 GB, per-folder `MANIFEST.tsv` + unified
`CORPUS-MANIFEST.tsv` with dims/format/license/provenance). For
SCREENSHOT-class measurement prefer it over gb82-sc's 10 retro images:
`8000-lilith-mobile-screenshots/` (32 modern mobile captures, 1080×2520
class) + `8100-lilith-web-screenshots/<viewport>/` (370 web captures at 5
viewport sizes, 375×667 → 2880×1800). It also adds content classes no
prior bench covered: document scans (NPS/EPA/NOAA), patents, manuscripts,
plots, renders, textures, AI illustrations. Keep gb82-sc cells in benches
for continuity with the W44-era baselines; add imazen-26 strata for
coverage. Stratify by folder, don't random-sample the modal class.
For LOSSLESS perf benches use the pre-selected k-means set
`benchmarks/lossless_bench_set_2026-06-10.tsv` (43 picks / 13 core across
23 strata ≤16 MP: PNG classes + the JPEG photo classes pre-decoded to
stripped PNGs under `/mnt/v/input/jxl-encoder/lossless-bench-imazen26-png/`
— feed the `bench_input` column, never raw .jpg; provenance + tier rule +
caveats in its `.meta`; regenerate via
`scripts/select_lossless_bench_imazen26.py`).

**CRITICAL**: All roundtrip validation tests MUST include jxl-rs. Do not create tests
that only use jxl-oxide or only use djxl - always include jxl-rs as well.

**CRITICAL: multi-group variants are REQUIRED in every tested path** (user
directive 2026-06-11). Every lock/roundtrip surface that exercises an
encode path MUST include a >256px (multi-group) cell alongside any
single-group fixtures — the entire multi-group lossless path had ZERO
byte-lock coverage until 2026-06-11, which is exactly where #68's two e9
desyncs and the #69 LZ77/palette drops hid. Procedural PRNG fixtures keep
committed bytes at zero (see `hash_lock_features.rs`'s 512x512 cells:
lossless noise e7/e9, LZ77-firing tiled e8, palette blocky e7, lossy
VarDCT e5/e7, RGBA-with-alpha, bilevel gray). New paths get a multi-group
cell in the same commit that adds the path.

**CRITICAL: jxl-oxide linear output**: When using jxl-oxide for metric computation
(butteraugli, SSIM2), ALWAYS request linear output:
```rust
let mut image = jxl_oxide::JxlImage::builder().read(reader)?;
image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
    jxl_oxide::RenderingIntent::Relative,
));
```
Without this, jxl-oxide applies sRGB gamma (because our file header signals Srgb
transfer function), producing sRGB f32 values. Feeding sRGB to butteraugli_linear()
or double-gamma-encoding for SSIM2 produces garbage metrics (butteraugli 49-60,
SSIM2 -30 to -50). This bug was silently present from the transfer function fix
(d8c34ff) until caught in a334b28.

### CRITICAL: No Synthetic-Only Quality Tests

**Synthetic images (gradients, solid colors, checkerboards) mask real bugs.**

The `raw_quant=1` bug is a perfect example:
- Synthetic tests: SSIM2 63-85 (PASSING)
- Real photos: SSIM2 23 (4x worse than libjxl)

**Rules:**
1. **Quality validation tests MUST use real photos** from `~/work/codec-corpus/`
2. Synthetic images are OK for unit tests (DCT correctness, bit-exactness)
3. Synthetic images are OK for decode-only tests (does it parse without error?)
4. **Quality thresholds MUST be validated on real photos**, not synthetic images
5. When a synthetic test passes but real photos fail, the synthetic test is LYING
6. **ANS/entropy tests MUST use real photos or complex real-world distributions**.
   Gradients and other synthetic content produce degenerate histograms that let ANS
   "cheat" — the omit_pos bug only manifested on CLIC photos with many symbols at
   the same logcount, never on gradients. Synthetic images for ANS are only OK for
   basic "does it parse" smoke tests, never for correctness validation.

**Mandatory quality test:**
```bash
cargo test test_save_broken_image -- --ignored --nocapture
# Must produce SSIM2 > 70 on real photo (currently broken due to raw_quant=1)
```

## CRITICAL: Patterns of Mistakes to Avoid

**MANDATORY READING before making any changes to this codebase.**

Analysis of 69 commits from Dec 28, 2025 - Jan 3, 2026 reveals systematic patterns of mistakes that have caused significant wasted effort and looping. These patterns MUST be avoided going forward.

### Mistake Pattern 1: False Positive Tests (HIGHEST SEVERITY)

**What happened (multiple times):**
- Commit 66a8934 (Jan 1): "test: add comprehensive VarDCT decoder validation tests"
- Commit 4e4f0ef (Jan 3): "docs: correct false claims about VarDCT working"
- Commit 83605ed (Jan 3): "fix: correct VarDCT tests to verify rendering, not just parsing"
- Commit bf6f0a2 (Jan 3): "fix: make all lossy VarDCT tests actually render frames"

**The mistake:** Tests called `JxlImage::builder().read()` which ONLY parses headers, never `render_frame()` which actually decodes pixels. Result: claimed "356 tests pass, VarDCT works!" when VarDCT was completely broken.

**Why this is severe:** False confidence led to documentation claims, commit messages, and wasted debugging time investigating wrong theories.

**Rules to prevent:**
1. **NEVER** declare success based on parsing alone - ALWAYS call `render_frame()` for image decoders
2. **NEVER** trust test counts - verify what tests actually test
3. **ALWAYS** manually verify claimed functionality before documenting it as working
4. Use `test_helpers.rs` standardized roundtrip functions that enforce full decode path
5. When adding validation, add it to ALL existing tests, not just new ones

**Detection:**
- If tests pass but manual testing fails, tests are false positives
- If commit message says "fix: make tests actually test X", previous tests were false positives
- If docs say something works but files don't decode, tests lied

### Mistake Pattern 2: Re-Investigating Already-Documented Bugs

**What happened:**
- Commit 8491735 (Jan 2): Fixed `TransformId=3` error for lossless Modular (missing num_dist field)
- Commit 9d4141d (Jan 3): INVESTIGATION.md documented `TransformId=3` error for VarDCT lossy: "8x8 lossy: FAILS (InvalidEnum TransformId=3)"
- Commit 8874f01 (Jan 3, same day): "Found" TransformId=3 error "again", wrote "Investigation continues for the TransformId error" as if discovering it for the first time

**The mistake:** Did not read existing docs or git history before "investigating." Re-discovered the exact same bug that was documented hours earlier.

**Why this is severe:** Wasted time on duplicate investigation. No progress made despite spending time.

**Rules to prevent:**
1. **BEFORE investigating a bug, read:**
   - `CLAUDE.md` (Known Bugs, Resolved Bugs sections)
   - `docs/CODE-HISTORY.md` (chronological bug history)
   - Recent git log (`git log --oneline --since="3 days ago" -30`)
   - `git log --grep="<error message>" --all`
2. **IF bug is already documented:**
   - Read what was already tried
   - Continue from where previous investigation left off
   - Update existing documentation, don't create duplicate sections
3. **NEVER** claim to have "found" a bug without checking if it's already known

**Detection:**
- If investigation notes say "Found bug X" but git history shows bug X documented earlier, it's duplicate work
- If commit message references an error that appears in earlier commits, check if it's already investigated

### Mistake Pattern 3: Creating Buggy Infrastructure to Prevent Bugs

**What happened:**
- Commit 9d4141d (Jan 3): Created `test_helpers.rs` to "prevent false positive loops"
  - Created `parse_encoding_mode()` that searches `for start_bit in 30..70`
  - This had a bug: found file header's `num_extra_channels=0` (bit 31) instead of frame header's `all_default` (bit 40)
  - Result: detected Modular when bitstream was actually VarDCT
- Commit 8874f01 (Jan 3, hours later): "fix: correct encoding mode detection in test_helpers"
  - Changed search range to `38..70` to skip file header
  - This "fixed" the bug in the solution that was supposed to prevent bugs

**The mistake:** Created test infrastructure without thoroughly testing it. The infrastructure itself had a false positive bug!

**Why this is severe:** Infrastructure bugs are worse than regular bugs because they give false confidence and affect all future tests. The solution to prevent mistakes became a source of mistakes.

**Rules to prevent:**
1. **Test the test infrastructure:**
   - When creating test helpers, test them against known-good and known-bad bitstreams
   - Create reference files with cjxl to validate parsing logic
   - Don't assume helper code is correct just because it's "simple"
2. **Validate with external truth:**
   - Use djxl, jxl-oxide, or jxl-rs to decode files and verify our parser agrees
   - Compare parser results against specification examples
3. **Don't use new infrastructure immediately:**
   - Test it standalone first
   - Verify it catches bugs it's supposed to catch
   - Verify it doesn't have false positives/negatives

**Detection:**
- If test infrastructure has bugs fixed shortly after creation, it wasn't tested properly
- If "fix:" commits appear for test helpers, those helpers were shipped broken

### Mistake Pattern 4: Documentation Claims Without Verification

**What happened:**
- Multiple commits claimed VarDCT "works" or "is complete"
- Commit 4e4f0ef explicitly titled: "docs: correct false claims about VarDCT working"
- Claims appeared in:
  - Status docs: "VarDCT: ✓ Complete"
  - Commit messages: "feat: complete VarDCT AC coefficient encoding pipeline"
  - Code comments

**The mistake:** Updated documentation to claim success based on tests passing, without manual verification that the feature actually works end-to-end.

**Why this is severe:** False documentation wastes everyone's time (including future self). Reading docs that say something works, then discovering it doesn't, destroys trust in all documentation.

**Rules to prevent:**
1. **NEVER claim something works without:**
   - Manual testing with reference decoder (djxl)
   - Visual inspection of decoded output for image codecs
   - Comparison with reference encoder output (cjxl)
   - Both jxl-rs AND jxl-oxide decoding successfully
2. **Use accurate status markers:**
   - ✓ Complete: Fully working, tested with multiple decoders, matches reference
   - ⚠ Partial: Some functionality works, but not all cases
   - ⚙ In Progress: Implementation exists but not tested
   - ✗ Broken: Implementation exists but known to fail
   - ❌ Not Started: No implementation
3. **When correcting false claims:**
   - Update ALL locations (docs, comments, commit messages can't be changed)
   - Document WHY the claim was false (what test was inadequate)
   - Document WHY the claim was false in CLAUDE.md

**Detection:**
- If commit says "correct false claims", previous documentation lied
- If docs say "complete" but bugs exist, documentation is premature
- If different docs contradict each other, at least one is wrong

### Mistake Pattern 5: Multiple Corrections of Same Issue

**What happened (same day, Jan 3):**
- Commit 4cef0e1: "fix: correct VarDCT modular substream encoding"
- Commit 07cfdaf: "fix: correct VarDCT modular substream encoding for jxl-oxide"
- Commit 44d8d58: "fix: correct modular frame header and prefix code encoding"

**The mistake:** Fixed part of the problem, committed, then discovered the fix was incomplete, fixed more, committed again. Multiple "fix: correct X" commits for the same X indicates incomplete understanding before first fix attempt.

**Why this is severe:** Each commit should be a complete fix for an issue. Multiple correction commits indicate:
- Didn't understand the full problem before attempting fix
- Didn't test the fix thoroughly before committing
- Possibly making changes without understanding (trial and error)

**Rules to prevent:**
1. **Before fixing a bug:**
   - Understand the FULL scope (trace all consumers of wrong data)
   - Write a failing test that reproduces the bug
   - Understand WHY the bug exists, not just WHERE
2. **After fixing a bug:**
   - Verify fix with multiple decoders (jxl-rs, jxl-oxide, djxl)
   - Test edge cases, not just the specific case that failed
   - Check if other code makes the same mistake
3. **Batch related fixes:**
   - If you discover "fix was incomplete" within hours/days, you didn't understand it
   - Better to spend more time understanding before committing partial fix

**Detection:**
- Multiple commits with "fix: correct X" for same X
- Commits saying "fix: improve X" shortly after "feat: add X"
- Commit message says "was inverted" or "was using wrong Y" (didn't verify before first commit)

### Mistake Pattern 6: Investigation Loop - Same Error, Different Names

**What happened:**
- Jan 2: `TransformId=3` error in lossless (missing num_dist) - FIXED
- Jan 3 (multiple commits):
  - "investigate: correct VarDCT bug analysis - ALL sizes fail"
  - "investigate: document VarDCT single-group bug"
  - "investigate: root cause analysis for VarDCT AC coefficient loss"
  - All finding variations of the same underlying issue: VarDCT bitstream is wrong

**The mistake:** Created multiple investigation documents, status files, and commit messages about what's fundamentally the same bug (VarDCT encoder produces invalid bitstream), just observed in different ways.

**Why this is severe:** Makes it impossible to track what's actually been tried. Investigation notes become noise instead of signal.

**Rules to prevent:**
1. **Use ONE place for investigation state:**
   - CLAUDE.md "Known Bugs" section is the single source of truth
   - Don't create STATUS.md, NOTES.md, INVESTIGATION.md, etc.
   - Use dated entries with clear status labels
2. **Link related errors:**
   - If seeing multiple symptoms (UnexpectedEof, InvalidEnum, byte corruption), they may be the same root cause
   - Document the connection: "This may be related to issue from YYYY-MM-DD"
3. **Update in place:**
   - If investigation reveals new info, UPDATE the existing section
   - Don't create "investigate: correct VarDCT bug analysis" - just update the analysis

**Detection:**
- Multiple files documenting same issue (scattered .md files all about the same bug)
- Multiple commits with "investigate:" prefix in same day for same component
- Commit message says "correct X analysis" meaning previous analysis was wrong

### Mistake Pattern 7: Not Reading Code Before Claiming Understanding

**What happened:**
- Commit a1b8fc4: "fix: use correct global_scale_float for quantization (was inverted)"
- Commit 1083e63: "fix: correct LZ77 channel boundaries and prediction function"
- Commit 3f0bace: "fix: use Huffman codes from build_and_store_huffman_tree"

**The mistake:** Code had fundamental errors that should have been caught by reading it before claiming it was complete:
- Used inverted quantization scale
- Used wrong prediction function despite documenting the right one
- Generated Huffman codes but didn't use the codes that were generated

**Why this is severe:** These are not edge case bugs - they're fundamental logic errors that make the code completely wrong. They should never have been committed in the first place.

**Rules to prevent:**
1. **Before committing implementation:**
   - Read through the code line by line
   - Verify variable names match their semantics (scale vs inverse_scale)
   - Check that documentation matches code
   - Verify all computed values are actually used
2. **For porting from reference:**
   - Read the reference implementation COMPLETELY
   - Don't assume "similar" code does the same thing
   - Verify matching inputs produce matching outputs (parity tests)
3. **Red flags to catch:**
   - Variable named X but used as inverse_X
   - Function returns value that's never used
   - Comment says "use X" but code uses Y

**Detection:**
- Commit message says "was inverted" or "was wrong" for basic logic
- "fix: use X" implies previous code computed X but didn't use it
- Bug found by reviewer/testing that should have been obvious from reading code

### Mistake Pattern 8: Bitstream Tracing Added Too Late

**What happened:**
- VarDCT implemented across multiple commits (Jan 1)
- Bitstream tracing added Jan 3 (commits 24421d7, 6c11635, 543f1dc)
- By this time, VarDCT already broken and being debugged

**The mistake:** Implemented complex bitstream encoding without the ability to inspect what was being written. Only added tracing after bugs were discovered and debugging was difficult.

**Why this is severe:** Debugging bitstream issues without seeing what's written is extremely difficult. Tracing should be built in from the start, not retrofitted.

**Rules to prevent:**
1. **Add tracing FIRST when implementing bitstream code:**
   - Before writing first `writer.write()`, add `trace_write!` infrastructure
   - Use the existing trace macros: `trace_write!`, `trace_section!`, `trace_note!`
   - Make tracing zero-cost with feature flag (already implemented)
2. **Never remove tracing:**
   - Keep `trace_write!` even after code works
   - This is debugging infrastructure for future issues
   - Zero cost when feature disabled
3. **Use tracing during development:**
   - Run tests with `--features trace-bitstream` to see what's written
   - Compare trace output with reference encoder
   - Verify bit positions match expected layout

**Detection:**
- If debugging commit adds tracing, tracing should have existed from the start
- If can't explain where bytes come from, need more tracing

## Proof-by-Tests Investigation Methodology (MANDATORY)

**Do not guess. Build a stack of invariant tests that accumulate until the bug is proven.**

The ANS omit_pos bug was found this way: Layer 1 (ANS symbol roundtrip) passed →
Layer 2 (histogram serialization roundtrip) failed → root cause pinpointed immediately.
Guessing would have taken days longer.

### Rules

1. **Layer your invariants from coarsest to finest:**
   - Layer 0: Does it compile? Do existing tests pass?
   - Layer 1: Does each component roundtrip in isolation? (encode → decode → compare)
   - Layer 2: Does serialization roundtrip? (write to bits → read back → compare)
   - Layer 3: Does the full pipeline produce valid output? (encode → external decoder)
   - Layer 4: Is the output correct? (quality metrics on real photos)

2. **Each layer MUST be a test that stays in the codebase:**
   - Not a one-off printf. A `#[cfg(debug_assertions)]` check or a `#[test]` function.
   - If you add a diagnostic check that finds a bug, keep it as a permanent invariant.
   - Gate verbose output behind `#[cfg(feature = "debug-tokens")]`, not behind nothing.

3. **When a layer passes, record that fact and move to the next layer:**
   - Don't re-investigate passing layers. The test proves they work.
   - Focus effort on the first failing layer — that's where the bug lives.

4. **Never skip to guessing before exhausting invariant layers:**
   - If you find yourself saying "maybe it's X", write a test that proves or disproves X.
   - If you can't write a test, you don't understand the problem well enough yet.

5. **Real data only for integration layers (3+):**
   - Synthetic data hides bugs (see: ANS omit_pos, raw_quant=1).
   - Use CLIC 2025 photos or `~/work/codec-corpus/` for any test above Layer 2.

## Invariant Preservation Across Sessions (MANDATORY)

**Every finding and proof-narrowing of invariants MUST be recorded in CLAUDE.md.**

Context compaction loses knowledge. The only way to preserve it is to write it down and commit it.

### Rules

1. **Commit findings immediately:**
   - When a layer passes, record it in CLAUDE.md "Investigation Notes" with the test name
   - When a layer fails, record what was ruled out
   - Include the commit hash where the test was added

2. **Format for tracking features under development:**
   ```markdown
   ### Feature: <name> (IN PROGRESS)

   #### Proven Layers
   - [x] Layer 1: Transform roundtrip (`test_dct_4x8_roundtrip`, commit abc123)
   - [x] Layer 1: Quant weights match libjxl (`test_dct4x8_quant_weights`, commit def456)
   - [ ] Layer 2: Tokenization roundtrip (IN PROGRESS)
   - [ ] Layer 3: External decoders
   - [ ] Layer 4: Quality on real photos

   #### Ruled Out
   - Transpose bug: verified output layout matches C++ (see test_dct4x8_layout)
   - DC extraction: spatial ordering confirmed correct (see test_dc_from_dct_4x8)

   #### Open Questions
   - Strategy selection threshold needs tuning after Layer 4
   ```

3. **After context compaction:**
   - FIRST action: read CLAUDE.md
   - Resume from the first unchecked layer
   - Do NOT re-investigate proven layers

4. **Commit atomically:**
   - Each layer proven = one commit with test + CLAUDE.md update
   - Message format: `test: prove Layer N for <feature> - <what was proven>`

5. **Clean up completed features:**
   - Move completed feature tracking from "Investigation Notes" to appropriate sections
   - Keep the record of what was proven (tests are the permanent proof)

## Investigation Documentation (MANDATORY)

**CLAUDE.md is the single source of truth for all debugging investigations.**

Do NOT create separate INVESTIGATION.md, STATUS.md, or similar files.
Active bugs go in "Known Bugs (ACTIVE)". Resolved bugs go in "Resolved Bugs".
Historical context lives in `docs/CODE-HISTORY.md`.

### Rules

1. **Keep CLAUDE.md up to date at ALL times** - Update immediately when you discover something
2. **Move resolved items promptly** - Keep "Known Bugs" lean, move fixes to "Resolved Bugs"
3. **Label findings by confidence level:**
   - `[PROVEN]` - Verified with evidence (include proof: test output, hex dump, etc.)
   - `[LIKELY]` - Strong evidence but not conclusive
   - `[SUSPICION]` - Educated guess, needs investigation
   - `[THREAD]` - Investigation path to explore
   - `[RULED OUT]` - Investigated and disproven (explain why)
   - `[RESOLVED]` - Issue was fixed (link to commit)

### Format

```markdown
### YYYY-MM-DD: Issue Title

**Status**: [ACTIVE|RESOLVED|BLOCKED]

**Summary**: Brief description of the problem.

**Findings**:
- [PROVEN] X causes Y (proof: `cargo test foo` output shows...)
- [SUSPICION] Could be related to Z
- [THREAD] Need to check if W affects this

**What's Been Tried**:
- Tried A - didn't work because...
- Tried B - partial success, revealed...

**Next Steps**:
1. Investigate X
2. Test Y with Z
```

### Why This Exists

Investigation loops have wasted weeks of effort. Proper documentation prevents:
- Re-discovering the same bug
- Re-trying failed approaches
- Losing context between sessions
- Multiple people investigating the same issue

## Bitstream Tracing (NEVER REMOVE)

**The `trace_write!`, `trace_section!`, `trace_note!`, and `trace_bytes!` macros are MANDATORY instrumentation. NEVER remove them.**

### Why This Exists

VarDCT debugging has consumed months of effort. These macros provide zero-cost tracing (compiled out without `--features trace-bitstream`) that shows exactly what's written to the bitstream.

### Rules

1. **NEVER remove trace macros** - They are critical debugging infrastructure
2. **ALWAYS add tracing when writing new bitstream code** - Every `writer.write()` should use `trace_write!`
3. **Use sections for structure** - `trace_section!(begin/end ...)` to show hierarchy
4. **Include semantic descriptions** - Explain what values mean, not just what they are

### Usage

```bash
# Enable tracing for debugging
cargo test --features trace-bitstream -- --nocapture 2>&1 | tee trace.log

# Normal build (zero cost - tracing compiled out)
cargo build
```

### Conversion Pattern

```rust
// WRONG - no tracing
writer.write(2, 0)?;

// CORRECT - with tracing
trace_write!(writer, 2, 0, "frame_type", "RegularFrame")?;
```

### Output Format

```
[bit_pos] SECTION.field: value (n_bits bits) = 0bXXXX // description
```

## Buffer Padding Rule

Always pad and align buffers to the working tile/block size upfront, with edge replication,
rather than adding bounds checks and scalar fallback paths throughout the processing code.
Wasting a few bytes of memory is cheaper than scattered branches and prevents entire classes
of off-by-one / OOB bugs. The adaptive_quant OOB bug (Jan 31, 2026) was caused by operating
on unpadded dimensions — the C++ reference pads first and never worries about it again.

## Notes

- The encoder produces little-endian bitstreams (LSB first within bytes)
- JXL signature is 0xFF 0x0A
- Group size is 256x256 pixels
- Block size is 8x8 for DCT

### SIMD Target Feature Boundaries (Feb 13, 2026)

The performance bottleneck with SIMD dispatch is NOT `summon()` (it's a cached atomic load, ~1.3ns).
The bottleneck is the **target feature mismatch at call boundaries**. When a function without
`#[target_feature]` calls a function with `#[target_feature]`, LLVM cannot inline across that
boundary — costing up to 4-6x on hot loops (measured in archmage benchmarks).

**Fix**: Expose concrete `_avx2(token, ...)` / `_neon(token, ...)` / `_scalar(...)` variants
from jxl_simd. Have jxl_encoder callers be `#[arcane]` functions that accept a concrete token
type, then call the matching variant directly. The token type IS the performance benefit —
it carries `#[target_feature]` through the call chain.

**Do NOT** create a `SimdDispatch` struct that wraps `Option<Token>`. That just moves the
dispatch inside the function and doesn't solve the boundary problem.

### SIMD parity testing (Phase S1-S4, 2026-05-25)

Every kernel in `jxl-encoder-simd` MUST have a scalar-vs-dispatch parity test that
exercises (a) all relevant tail-loop boundary sizes and (b) the f32_edge_battery
input distribution. The canonical pattern lives in
[`jxl-encoder-simd/src/test_helpers.rs`](jxl-encoder-simd/src/test_helpers.rs).

Use the three-line wrapper:
```rust
let ref_out = my_kernel_scalar(&input);
run_dispatch_parity(|perm| {
    let act = my_kernel(&input);
    assert_f32_slice_bit_eq(&ref_out, &act, perm, "context");
});
```

For kernels that cannot be bit-exact (FMA association, reduction-tree order),
use `assert_f32_slice_close_ulps_abs` with a documented tolerance and absolute
floor. Known divergences are tracked at
[`docs/SIMD_PARITY_KNOWN_DIVERGENCES.md`](docs/SIMD_PARITY_KNOWN_DIVERGENCES.md);
any new `#[ignore]` MUST add an entry there.

Motivation: SA-G commit `7d383785` found CfL Newton SIMD diverged from libjxl
on real inputs because existing tests only checked scalar-vs-dispatch on 1-2
fixed cases. The test_helpers module forces coverage of tail-loop boundaries
at every SIMD width (f32x4, f32x8, f32x16) plus edge-value input distributions
(zeros, denormals, alternating sign, large/small magnitudes).

**Archmage annotation rules**:
- `#[arcane]`: Top-level SIMD entry points. Adds `#[target_feature]`.
- `#[rite]`: Inner helpers called from `#[arcane]`. Adds `#[target_feature]` + `#[inline]`.
  Requires a token parameter (macro derives features from token type).
- `#[inline(always)]`: Only for helpers WITHOUT a token parameter (pure scalar code that
  gets inlined into the `#[arcane]` caller). Works because inlining plain code INTO a
  `#[target_feature]` context is fine; the reverse is not.
- **Never call `#[arcane]` from `#[arcane]`** — use `#[rite]` for function-to-function calls
  within SIMD code.

### Enhanced Clustering Cost Model Discovery (Jan 31, 2026)

**Finding:** Enhanced clustering with pair merge refinement produces ~0.5% LARGER
files when using Huffman entropy coding. The fast clustering algorithm (k-means-like
without refinement) is already near-optimal for Huffman.

**Root Cause Analysis:**
1. Fast clustering uses histogram distance = `merged_data_cost - sum(individual_data_costs)`
2. This correctly measures the DATA cost increase from merging
3. For Huffman, header cost savings from merging are minimal (~1-2 bits per merge)
4. The pair merge refinement finds "beneficial" merges based on cost model, but
   the actual file is larger due to:
   - Context map encoding overhead
   - Suboptimal tree sharing across contexts with different distributions

**Cost Model Details:**
- Shannon entropy underestimates Huffman cost by 2-3% (integer code lengths)
- Implemented `compute_huffman_data_cost()` using actual `create_huffman_tree()`
- Header cost for Huffman: simple tree (1-4 symbols) ~4+n*8 bits, complex tree ~40+n*2.5 bits
- ANS header cost: ~5 bits per symbol for frequency table

**ANS:** Enhanced clustering is beneficial with ANS (larger header cost per symbol).
Enabled at effort >= 9 for VarDCT via `enhanced_clustering_vardct`.

### DC Tree Learning (Feb 4, 2026)

WP-based DC tree with AC metadata prefix subtree. Property 1 (stream_id) split at root
with dynamic splitval=num_dc_groups routes DC groups to WP subtree and AC metadata to
its own subtree. Works for any number of DC groups (fixed Apr 2026 — previously hardcoded
splitval=2 caused decode failures for images >2048px wide, imazen/jxl-encoder#3).

**Files**: `dc_tree_learn.rs` (tree building), `context_tree.rs` (bitstream writing)


### Modular parity + lossless status (Feb 2026) — archived

The "Modular Encoder Parity vs libjxl (Feb 6, 2026)" and "Lossless
Compression Status (Feb 16, 2026)" deep-status sections were archived to
[docs/CODE-HISTORY.md](docs/CODE-HISTORY.md) ("Archived from CLAUDE.md
2026-07-19" section).

## API Convergence TODOs

See `/home/lilith/work/zendiff/API_COMPARISON.md` for full cross-codec comparison.

**Three-layer pattern: EncoderConfig → EncodeRequest<'a> → Encoder (streaming only)**

**No backwards compatibility required** — we have no external users. Just bump the 0.x major version for breaking changes. No deprecation shims or legacy aliases — delete old APIs. Prefer one obvious way to do things — no duplicate entry points. Minimize API surface for forwards compatibility. Avoid free functions — use methods on types (Config, Request, Decoder) instead.

**Builder convention**: `with_` prefix for consuming builder setters, bare-name for getters.

**Licensing**: AGPL v3 / Commercial dual license. Cargo.toml uses `license = "AGPL-3.0-or-later"`. README must include the standard licensing text (see codec-design README).

**Project standards**: `#![forbid(unsafe_code)]` with default features. no_std+alloc (minimum: wasm32). CI with codecov. README with badges and usage examples. As of Rust 1.92, almost everything is in `core::` (including `Error`) — don't assume `std` is needed. Use `wasmtimer` crate for timing on wasm. Fuzz targets required (decode, roundtrip, limits, streaming). Codecs must be safe for malicious input on real-time image proxies — no amplification, bound memory/CPU, periodic DoS/security audits.

- [x] Split `EncoderOptions` into `LossyConfig` / `LosslessConfig` (compile-time invalid state prevention)
- [x] Add `EncodeRequest<'a>` intermediate layer
- [x] One-shot via `request.encode()`/`encode_into()`/`encode_to()`
- [x] Add `PixelLayout` enum (replace method-name-based dispatch)
- [x] Rename `Error` → `EncodeError` (new `api::EncodeError`)
- [x] Change dimensions from `usize` to `u32` (in new API)
- [x] Add `Limits` struct (all fields `Option<u64>`, default None = no limit)
- [x] Add `ImageMetadata` struct for ICC/EXIF/XMP on request (type exists, not wired yet)
- [x] Add `Quality` enum with `Distance(f32)` and `Percent(u32)`
- [x] Add `&dyn Stop` cancellation (from `enough` crate)
- [x] Adopt `with_` prefix convention for all builder setters on Config/Request
- [x] Bare-name getters on both Config types (distance(), effort(), ans(), etc.)
- [x] Fluent encode shortcuts on Config types (encode(), encode_into())
- [x] Remove free functions (encode_lossless_rgb8 etc.) per "avoid free functions"
- [x] `PixelLayout::bytes_per_pixel()` public + const, `is_linear()`, `has_alpha()`
- [x] CLI updated to use new API (LosslessConfig/LossyConfig/PixelLayout)
- [x] Hide old `EncoderOptions` + `Encoder` API (#[doc(hidden)], no root re-exports)
- [x] Add streaming `LossyEncoder`/`LosslessEncoder` with `push_rows()`/`finish()`/`finish_into()`/`finish_to()`
- [x] `encode_to()`/`finish_to()` std-only (gated behind `feature = "std"`)
- [x] Add `At<>` error location tracking (from `whereat` crate)
- [x] Add `EncodeStats` for encode metrics
- [x] Memory estimation on both config types — shipped as `estimate_peak_memory_bytes()` + `estimate_encode()` delegating to `heuristics::estimate_encode` (calibrated 2026-06-14)
- [x] Wire `ImageMetadata` (ICC/EXIF/XMP) through to actual encoder output
  - ICC: embedded in codestream via PredictICC + Huffman entropy, lossy + lossless paths
  - EXIF/XMP: container format boxes (already working)
- [ ] Add probing: `ImageInfo::from_bytes(&[u8])` static probe with `PROBE_BYTES` constant
- [ ] Two-phase decoder: `build()` parses header → `info()` inspects → `decode()` continues without re-parsing
- [x] Support `Rgba8` and `Bgra8` for lossless encode (alpha preserved)
- [x] Support `Bgr8` and `Bgra8` pixel layouts (R↔B swap)
- [x] Lossy+alpha: encode alpha as modular extra channel alongside VarDCT RGB
- [ ] Support `Bgra8` for decode (future — no decoder yet)
