# RFC: zensim fork — phased ship plan for adding zensim as third `PerceptualBackend`

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-25
**Status**: SCOPING (forward-looking; companion to `RFC_ZENSIM_BUTTLOOP_AUDIT.md` + `RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`)

This RFC sketches the phased ship plan for adding zensim as a third perceptual metric backend alongside butteraugli and cvvdp, mirroring the cvvdp arc's 7-phase + Phase 8 structure. Each phase brief below is sketched (≥ 80 LOC), not finalized — they're starting briefs for downstream agents to expand into full `RFC_ZENSIM_PHASE<N>_BRIEF.md` documents at chunk-spawn time.

The cvvdp arc spans 10 commits across `8c6e91cc..689ba0df` (`docs/RFC_CVVDP_FORK.md` §9). zensim mirrors the structure but starts closer to Phase 4 because the CPU diffmap + linear-planes APIs are already shipped (per RFC #2 §6.1).

## §1. Phase summary

| Phase | Scope | Crate(s) touched | Effort | Status |
|---    |---    |---               |---     |---     |
| **0 (prereq)** | Multi-metric API refactor per RFC #3 (PerceptualMetric enum) | jxl-encoder | 1-2 days | not started |
| **1** | zensim-gpu diffmap + linear-planes API + CPU `_into` wrap | zenmetrics + jxl-encoder | 2-3 weeks | not started |
| **2** | PerceptualBackend trait alignment (verify; likely no-op) | jxl-encoder | 1 day | not started |
| **3** | `CpuZensimBackend` + `GpuZensimBackend` impls + opt-in API | jxl-encoder | 1 week | not started |
| **4** | Buttloop wiring + per-distance target table + initial benches | jxl-encoder | 1 week | not started |
| **5** | CPU/GPU split refinement + optional zensim profile selector | jxl-encoder | 1-2 days | not started |
| **6** | 6-backend tracking sweep + Pareto decision per RFC #1 §8 | jxl-encoder + corpus | 1 week | not started |
| **7** | Docs closeout (if OPT_IN_ONLY) | jxl-encoder | 1 day | not started |
| **8 (conditional)** | Pareto refit: 8a diagnosis → 8c renorm → 8d tighten → 8g block-constants | jxl-encoder | 2-3 weeks | conditional on Phase 6 |

**Total budget**: 5-7 weeks if Phase 6 lands ≥ 85% Pareto-front; 8-10 weeks if Phase 8 refit needed (based on cvvdp arc precedent).

## §2. Phase 0 — Multi-metric API refactor (PREREQUISITE)

**Sibling workspace**: `~/work/zen/jxl-encoder--multi-metric-api`.
**Branch**: `multi-metric-api-refactor` (NOT cvvdp-fork-rfc; this is a separate landing).

### Scope

Implements the API design from `RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`:

1. New `PerceptualMetric` + `PerceptualDevice` enums in `api.rs`.
2. New `LossyConfig::with_perceptual_metric` / `with_perceptual_device` / `with_perceptual_target_score` setters.
3. New `LossyConfig::resolve_perceptual_metric` resolver with Libjxl strict-parity short-circuit.
4. Hard-rename: DELETE `with_gpu_butteraugli` + `with_cvvdp_loop` + `with_cvvdp_use_cpu`. Per CLAUDE.md "no backwards-compat hacks." Keep `with_cvvdp_bytes_tighten` (cvvdp-specific tuning knob, not metric-selection).
5. Refactor `construct_backend(width, height, cpu_params, intensity_target, MetricSelection)` — collapse 4 trailing bools into one struct.
6. Refactor dispatch in `vardct/perceptual_loop.rs::butteraugli_refine_quant_field` to consume `PerceptualMetric` via resolver.
7. Update existing tests:
   - `cvvdp_loop_smoke.rs` migrated to new setter shape.
   - `cvvdp_cpu_backend_smoke.rs` migrated.
   - `strategy_libjxl_byte_lock.rs` extended with `PerceptualMetric::{Cvvdp, Butteraugli} × PerceptualDevice::{Auto, Cpu, Gpu}` × 4 fixtures (24 byte-locks).
8. CHANGELOG breaking-change entry under `[Unreleased] / QUEUED BREAKING CHANGES`.

### Inputs to read FIRST

1. `docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` — the design spec.
2. `docs/RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §7.3 — Libjxl invariant.
3. `jxl-encoder/src/vardct/perceptual_backend.rs:1232-1457` — current `construct_backend` to refactor.
4. `jxl-encoder/src/api.rs:6552-6838` — current cvvdp/gpu setters to delete.
5. `jxl-encoder/src/vardct/perceptual_loop.rs:2157-2167` — current `effective_metric_target_distance` dispatch.

### Acceptance gates

- (a) `cargo build` PASS on every feature combo (default, `gpu-butteraugli`, `cvvdp-loop`, `cvvdp-loop-cpu`, all combinations).
- (b) `cargo test --lib`: pre-Phase-0 count + 0 (zero tests removed; some setter unit tests rewritten).
- (c) Hash-locks 36/36 BYTE-IDENTICAL at default features (no encode behaviour change).
- (d) `strategy_libjxl_byte_lock` extended to 36 byte-locks (4 fixtures × 3 metrics × 3 devices) — all PASS.
- (e) `divergence_table_drift` 7/7 PASS.
- (f) Multi-decoder roundtrip ≥ 4 cells × {butter, cvvdp} × {CPU, GPU when available} — all PASS.
- (g) CHANGELOG breaking-change entries reference the deleted setters + their replacements.
- (h) Pushed to `multi-metric-api-refactor` bookmark on origin.

### DO NOT

- DO NOT keep `#[deprecated]` shims for the deleted setters (CLAUDE.md "no backwards-compat hacks; we have no external users").
- DO NOT change `BlockReducerConstants` dispatch yet — Phase 4 (zensim) does that.
- DO NOT change the `PerceptualBackend` trait surface — it's already correct.
- DO NOT change `cvvdp_bytes_tighten` setter — it's a tuning knob, not metric-selection.
- DO NOT cite "FMA precision" for any byte movement.

### Chunk N+1 plan

Phase 1 (zensim-gpu diffmap + linear-planes API) can fire in parallel with Phase 0 — they touch different crates (`zenmetrics/crates/zensim-gpu/` for Phase 1 vs `jxl-encoder/` for Phase 0). Spawn both with sibling-workspace ownership.

---

## §3. Phase 1 — zensim-gpu diffmap + linear-planes + CPU `_into` wrap

**Sibling workspace**: `~/work/zen/zenmetrics--zensim-diffmap` (zenmetrics-side) + `~/work/zen/jxl-encoder--zensim-cpu-wrap` (jxl-encoder-side, if Phase 0 hasn't landed yet; otherwise stack on `multi-metric-api-refactor`).

### Scope — zenmetrics-side (large; ~2-3 weeks)

Mirror cvvdp-gpu's `8b658b40` commit shape (cvvdp diffmap API + linear-planes entry points). Add to `~/work/zen/zenmetrics/crates/zensim-gpu/`:

1. **`src/kernels/diffmap.rs`** — new module. Kernels:
   - `bilinear_upsample_band` — upsample one per-scale SSIM error band to base resolution.
   - `multi_scale_fuse` — sum per-scale upsampled bands with per-scale-channel weights.
   - `post_processing` — optional `apply_contrast_masking` + `sqrt` per `DiffmapOptions`.
   - Host-side reference scalar impl for parity testing.
2. **`src/pipeline.rs`** — new methods on `Zensim<R>`:
   - `score_with_diffmap(ref_srgb, dist_srgb, diffmap_out: &mut Vec<f32>) -> Result<Score>`.
   - `score_with_warm_ref_diffmap(dist_srgb, diffmap_out) -> Result<Score>`.
   - `score_from_linear_planes(ref_rgb_planes, dist_rgb_planes) -> Result<Score>`.
   - `score_from_linear_planes_with_diffmap(ref_planes, dist_planes, diffmap_out) -> Result<Score>`.
   - `warm_reference_from_linear_planes(ref_planes) -> Result<()>`.
   - `score_from_linear_planes_with_warm_ref(dist_planes) -> Result<Score>`.
   - `score_from_linear_planes_with_warm_ref_diffmap(dist_planes, diffmap_out) -> Result<Score>`.
3. **`src/opaque.rs`** — mirror all 7 methods on `ZensimOpaque` via the `ZensimInner` trait extension.
4. **`tests/diffmap_invariants.rs`** — 5 invariant tests per RFC #1 §2.1:
   - identity → zero diffmap (1e-7).
   - non-negative.
   - monotone in distortion (synthetic-perturbation pair).
   - spatial localization (block-perturbation).
   - warm-ref invariance (cold vs warm path within 1e-7).
5. **`tests/diffmap_dispatch.rs`** — GPU end-to-end tests, 5 cells covering Cuda + cubecl-cpu backends.
6. **`examples/diffmap_overhead.rs`** — wall-time bench at 256² / 512² / 1024² / 2048².
7. **`crates/zensim-gpu/benchmarks/diffmap_overhead_<date>.{tsv,meta}`** — bench output.
8. **`crates/zensim-gpu/docs/DIFFMAP_DIVERGENCES.md`** — RFC §3 divergence note (strict `score == minkowski_norm(diffmap)` aspirational, not delivered).

### Scope — jxl-encoder-side (small; ~1 day)

Add CPU `_into` wrap shape for the existing `compute_with_ref_and_diffmap_linear_planar`. Two paths:

A. **Option B from RFC #2 §3.2.1** (recommended): wrap inside `CpuZensimBackend`. `backend.compare_with_reference` calls the existing API, then copies `result.diffmap()` into the caller's `diffmap_out` via `diffmap_out.clear(); diffmap_out.extend_from_slice(result.diffmap())`. Adds 1 `O(W*H)` memcpy per iter — ~1-2 ms at 1024², negligible.

B. **Option A** (cleaner but adds API surface): file a zensim issue asking for `compute_with_ref_and_diffmap_linear_planar_into(_, _, _, _, _, _, scratch: &mut ZensimScratch, diffmap_out: &mut Vec<f32>) -> Result<ZensimResult>`. Defer to a zensim 0.3.0 follow-on; ship Option B in Phase 1 to unblock the rest of the arc.

### Inputs to read FIRST

1. `docs/RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §2 (the 5 invariants) + §5 (warm-ref API shape).
2. `docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md` §3.1 (the 3 GPU gaps).
3. `~/.claude/projects/.../cvvdp_gpu_diffmap_api_shipped_2026-05-24.md` — the cvvdp-gpu Phase 1 memo. Same shape.
4. `~/work/zen/zenmetrics/crates/cvvdp-gpu/src/kernels/diffmap.rs` — kernel structure to mirror.
5. `~/work/zen/zenmetrics/crates/cvvdp-gpu/src/pipeline.rs` lines 252-... — the 7 new pub methods to mirror.
6. `~/work/zen/zensim/zensim/src/diffmap.rs:706-795` — CPU `compute_with_ref_and_diffmap_linear_planar`. Match its multi-scale fusion semantics.

### Acceptance gates

zenmetrics-side:
- (a) `cargo build -p zensim-gpu` default features PASS.
- (b) `cargo build -p zensim-gpu --no-default-features --features 'cpu pixels'` PASS.
- (c) `cargo test -p zensim-gpu`: all existing tests PASS + 5 invariants + 5 dispatch + 7 kernel-unit PASS.
- (d) Identity → all-zero diffmap pinned to 1e-7 absolute on 16×16 sRGB-bytes pair.
- (e) GPU-CPU pointwise parity pinned to 1e-6 absolute on 24×24 synthetic pair.
- (f) `score_with_diffmap` JOD-equivalent matches `score` to 1e-4 absolute.
- (g) `score_from_linear_planes` matches sRGB-bytes path to 1e-3 score units.
- (h) Diffmap overhead bench TSV committed.
- (i) Pushed to zenmetrics master via `jj git push`.
- (j) Sibling workspace cleaned up after push (`jj workspace forget zenmetrics--zensim-diffmap` + `rm -rf`).

jxl-encoder-side:
- (k) Field `LossyConfig::cvvdp_use_cpu` etc unaffected by `CpuZensimBackend` wrap.
- (l) Bench `examples/zensim_diffmap_into_wrap_bench.rs` shows 0 alloc growth across iters.

### DO NOT

- DO NOT modify `zensim/` (the CPU crate) source from Phase 1. The CPU `_into` wrap lives in jxl-encoder.
- DO NOT change the `ZensimProfile` default — keep V0_3 per RFC #2 §5.
- DO NOT add HDR support to zensim — separate workstream.
- DO NOT cite "FMA precision" for any score drift on the diffmap path vs the scalar path.

### Chunk N+1 plan

Phase 2 (trait alignment verification) can fire after Phase 1 zenmetrics-side lands. Phase 3 (backend impl) needs both Phase 1 zenmetrics + Phase 0 jxl-encoder before it can spawn.

---

## §4. Phase 2 — PerceptualBackend trait alignment

**Sibling workspace**: `~/work/zen/jxl-encoder--zensim-trait-verify`.
**Branch**: `multi-metric-api-refactor` (stacked on Phase 0).

### Scope (~1 day)

Verify the `PerceptualBackend` trait at `vardct/perceptual_backend.rs:325-381` is shape-correct for zensim. Per RFC #3 §3, this is expected to be a no-op — the cvvdp Phase 2 (`8c6e91cc`) generalized the trait specifically because the same shape works for any metric.

Tasks:
1. Read trait. Confirm zensim's wrapped APIs (CPU `compute_with_ref_and_diffmap_linear_planar` + GPU Phase 1's new `score_from_linear_planes_with_warm_ref_diffmap`) match the trait's `set_reference + compare_with_reference` shape.
2. Confirm `BackendCompareResult { score: f64 }` is sufficient for zensim's score output (`raw_distance` direction or `(100 - score)` normalization — see RFC #1 §1.1).
3. Write a smoke test that constructs a `Box<dyn PerceptualBackend>` from `CpuZensimBackend` (stub) and verifies the trait object dispatch works. Don't ship the full impl yet — that's Phase 3.

### Inputs to read FIRST

1. `jxl-encoder/src/vardct/perceptual_backend.rs:325-381`.
2. `RFC_CVVDP_FORK.md` §2.1 (Phase 2 rationale).
3. `~/.claude/projects/.../cvvdp_fork_phase2_perceptual_rename_shipped_2026-05-24.md` (cvvdp Phase 2 memo).

### Acceptance gates

- (a) Smoke test compiles + passes (stub impl of `CpuZensimBackend`).
- (b) Hash-locks 36/36 BYTE-IDENTICAL.
- (c) Single commit; small (< 100 LOC change).

### DO NOT

- DO NOT modify the trait surface. If you find a gap, that means RFC #1 §5.1 missed something — file a follow-on issue, don't change the trait surface mid-stream.

### Chunk N+1 plan

Phase 3 (backend impl) fires next.

---

## §5. Phase 3 — `CpuZensimBackend` + `GpuZensimBackend` impls + opt-in API

**Sibling workspace**: `~/work/zen/jxl-encoder--zensim-backend-impl`.
**Branch**: `multi-metric-api-refactor` (stacked on Phase 2).

### Scope (~1 week)

Mirror cvvdp Phase 3 + Phase 5 (commits `57757ff8` + `206b874e`). Add to `jxl-encoder/`:

1. **`Cargo.toml`** — register optional `zensim = { version = "0.3", optional = true }` + `zensim-gpu = { version = "0.0.x", default-features = false, features = ["cuda", "cubecl-types"], optional = true }`. New cargo features:
   - `zensim-loop = ["dep:zensim"]` (CPU)
   - `zensim-loop-gpu = ["zensim-loop", "dep:zensim-gpu", "dep:cubecl"]` (GPU; implies CPU)
2. **`src/vardct/zensim_backend.rs`** — new file ~500 LOC mirroring `cvvdp_backend.rs`:
   - `pub(crate) mod cpu { pub(crate) struct CpuZensimBackend { ... } }`
   - `pub(crate) mod gpu { pub(crate) struct GpuZensimBackend { ... } }`
   - Both implement `PerceptualBackend`.
   - CPU backend wraps `zensim::Zensim::new(ZensimProfile::PreviewV0_3)` + `PrecomputedReference` + `compute_with_ref_and_diffmap_linear_planar` per RFC #2 §3.2.
   - GPU backend wraps `zensim_gpu::ZensimOpaque` + Phase 1's `warm_reference_from_linear_planes` + `score_from_linear_planes_with_warm_ref_diffmap`.
   - `catch_unwind` around CUDA init per RFC #1 §7.1.
   - Score direction normalization: `let score = (100.0 - zensim_score).clamp(0.0, 100.0)` — produces butter-direction score (smaller=better, 0=identical).
   - Diffmap renorm: apply `ZENSIM_DIFFMAP_RENORM_SCALE` (initial value 1.0; fit Phase 4).
   - Phase 8b-style diffmap dump hook (env `JXL_PHASE8B_DIFFMAP_DUMP` already exists; extend `maybe_dump_diffmap_stats` callers to tag `"Z_CPU"` / `"Z_GPU"`).
3. **`src/vardct/mod.rs`** — register the new module.
4. **`src/vardct/perceptual_backend.rs`** — extend `construct_backend` dispatch with `PerceptualMetric::Zensim` branch.
5. **`src/api.rs`** — extend `PerceptualMetric` enum already-added in Phase 0 with `Zensim` variant (no-op if Phase 0 already added it; otherwise add now). Add `with_zensim_profile(Option<ZensimProfile>)` + resolver if Phase 5 will support per-config profile choice (otherwise defer).
6. **`tests/zensim_backend_smoke.rs`** — gated `cvvdp-loop`-style smoke tests (5 cells × invariants).
7. **`docs/LIBJXL_DIVERGENCES.md`** — Section E new row for zensim opt-in.

### Inputs to read FIRST

1. `RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` §1-§4 (the API surface this implements).
2. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/cvvdp_backend.rs` — read in full; mirror structure exactly.
3. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/perceptual_backend.rs:1232-1457` — `construct_backend` dispatch to extend.
4. `~/.claude/projects/.../cvvdp_fork_phase3_backend_impl_blocked_local_ff_2026-05-24.md` — cvvdp Phase 3 memo. Note the "BLOCKED on local FF" pattern; coordinate with Phase 1 zenmetrics-side to ensure the local checkout is at-or-after the zensim-gpu Phase 1 commit.
5. `~/.claude/projects/.../cvvdp_fork_phase5_cpu_integration_shipped_2026-05-24.md` — cvvdp Phase 5 (CPU integration) memo. Same shape for `CpuZensimBackend`.
6. `~/work/zen/zensim/zensim/src/metric.rs:704` — confirm `score()` direction; `(100.0 - score)` is the normalization.

### Acceptance gates

- (a) Build PASS on every feature combo: default, `zensim-loop`, `zensim-loop-gpu`, `__expert butteraugli-loop cvvdp-loop zensim-loop-gpu parallel`.
- (b) `cargo test --lib`: default 1437/1437 PASS, zensim features +8-12 unit tests PASS.
- (c) Hash-locks 36/36 BYTE-IDENTICAL at default features.
- (d) Hash-locks 36/36 BYTE-IDENTICAL with `zensim-loop` compiled (opt-in not engaged).
- (e) `strategy_libjxl_byte_lock` PASS with full feature set (Libjxl strict-parity invariant).
- (f) `divergence_table_drift` PASS.
- (g) Multi-decoder roundtrip ≥ 5 cells × {jxl-rs, djxl, jxl-oxide} on `PerceptualMetric::Zensim` opt-in encodes — all 15/15 PASS.
- (h) Section E divergence row added.
- (i) Pushed to `cvvdp-fork-rfc` (or a sibling bookmark; user picks).
- (j) Sibling workspace cleaned up post-merge.

### DO NOT

- DO NOT default-on `PerceptualMetric::Zensim` — opt-in only. Default-flip is a Phase 7 OR Phase 8f decision per RFC #1 §8.4.
- DO NOT set the CVVDP_DIFFMAP_RENORM_SCALE-equivalent away from 1.0 yet — Phase 4 fits this.
- DO NOT change the cvvdp opt-in API surface beyond what Phase 0 already did.
- DO NOT modify the `zensim` or `zensim-gpu` source from inside jxl-encoder.

### Chunk N+1 plan

Phase 4 (buttloop wiring + target table + bench) fires next.

---

## §6. Phase 4 — Buttloop wiring + per-distance target table + initial benches

**Sibling workspace**: `~/work/zen/jxl-encoder--zensim-loop-wiring`.
**Branch**: `cvvdp-fork-rfc` or `zensim-fork-rfc` (split at user request).

### Scope (~1 week)

Mirror cvvdp Phase 4 (commit `32581839`). Tasks:

1. **`benchmarks/zensim_target_calibration_seed_<date>.txt`** — pre-computed seed from a quick 10-cell butteraugli-default baseline sweep. Methodology per RFC #1 §1.3:
   - Run buttloop-driven butteraugli baseline on 10 representative cells (mix of CID22 photos + GB82-SC screenshots, distances ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}).
   - Score every output with zensim post-hoc.
   - Median zensim score per distance × 1.00 (NO 1.05× tightening initially — the cvvdp arc showed tightening overshoots; start neutral, calibrate via Phase 8c renorm if needed).
   - Convert direction: `target_score = 100.0 - median_zensim_score`.
2. **`jxl-encoder/src/vardct/zensim_targets.rs`** — new file. Mirror `cvvdp_targets.rs` shape:
   - `pub(crate) static ZENSIM_DISTANCE_TARGETS: &[(f32, f32); 7]` — seeded from Step 1.
   - `pub(crate) fn zensim_target_score_for_distance(d: f32) -> f32` — linear interpolation + clamp.
   - 7 unit tests (sort, monotone, endpoints, clamp, interp, well-behaved, 7-entry guard).
3. **`vardct/perceptual_loop.rs`** — extend `effective_metric_target_distance` dispatch with `PerceptualMetric::Zensim` branch per RFC #3 §2.2.
4. **`vardct/perceptual_loop.rs`** — extend `block_reducer_constants_for_backend` dispatch with `ZENSIM_BLOCK_CONSTANTS` (initial `k_tile_norm: 1.2`, butter-parity; Phase 8g-style refit if needed).
5. **`tests/zensim_loop_smoke.rs`** — 5 cells × 5 invariants (byte-identical to default when off, decode-cleanly when on, Libjxl strict-parity, public API roundtrip).
6. **`examples/zensim_loop_smoke_bench.rs`** — 20-cell B vs Z_CPU vs Z_GPU bench at e=8 (mirror cvvdp Phase 4 bench).
7. **`benchmarks/zensim_loop_smoke_<date>.{tsv,meta}`** — bench output.
8. **`docs/LIBJXL_DIVERGENCES.md`** — Section E row extended with Phase 4 SHIPPED narrative.

### Inputs to read FIRST

1. `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §1.3 (target table seed methodology) + §3 (per-block reducer compatibility).
2. `RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` §2.2 (dispatch shape) + §7 (per-metric constants tables).
3. `~/.claude/projects/.../cvvdp_fork_phase4_loop_wiring_shipped_2026-05-24.md` — cvvdp Phase 4 memo. Same shape.
4. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/cvvdp_targets.rs` — mirror structure exactly.
5. `~/work/zen/jxl-encoder/benchmarks/cvvdp_jod_calibration_seed_2026-05-24.txt` — seed format to mirror.

### Acceptance gates

- (a) Build PASS on every feature combo from Phase 3.
- (b) `cargo test --lib`: +7 (target table unit tests) + 5 smoke tests PASS.
- (c) Hash-locks 36/36 BYTE-IDENTICAL at default (zensim opt-in not engaged).
- (d) Libjxl strict-parity PASS with zensim opt-in flag set.
- (e) Multi-decoder roundtrip 5 cells × 3 decoders PASS.
- (f) Bench TSV committed.
- (g) Section E divergence row updated.

### DO NOT

- DO NOT apply a 1.05× tightening factor to the initial target table. The cvvdp arc showed it overshoots; start neutral. Phase 8c renorm closes any per-block predicate misalignment.
- DO NOT change the `K_TILE_NORM` literal away from 1.2 yet — Phase 8g-style refit is a separate chunk.
- DO NOT add per-distance refinement to the renorm/block constants — single-value is sufficient until measurement proves otherwise.

### Chunk N+1 plan

Phase 5 (CPU/GPU split refinement + profile selector) is OPTIONAL — skip if Phase 4 lands clean. Phase 6 (tracking sweep) is mandatory.

---

## §7. Phase 5 — CPU/GPU split refinement + optional zensim profile selector

**Sibling workspace**: `~/work/zen/jxl-encoder--zensim-cpu-gpu-split`.
**Branch**: `cvvdp-fork-rfc` (stacked on Phase 4).

### Scope (~1-2 days, OPTIONAL)

Two thin chunks:

1. **`LossyConfig::with_zensim_profile(Option<ZensimProfile>)`** — caller picks the active zensim profile. Default `None` → `PreviewV0_3` (the audit's recommendation in RFC #2 §5). Use cases:
   - Caller wants the SOTA Tuner profile despite the 2× wall cost.
   - Caller wants to A/B compare profiles for their corpus.

2. **CPU/GPU dispatch matrix refinement** — per RFC #3 §4. The Phase 3 stub dispatch is mechanical; Phase 5 can refine if measurement shows specific cells benefit from CPU-only (e.g. small images < 128² where GPU launch overhead dominates).

### Inputs to read FIRST

1. `RFC_ZENSIM_BUTTLOOP_AUDIT.md` §5 (profile selection).
2. `RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` §4 (dispatch).
3. `~/.claude/projects/.../cvvdp_fork_phase5_cpu_integration_shipped_2026-05-24.md` — cvvdp Phase 5 memo.

### Acceptance gates

- (a) `with_zensim_profile` setter + resolver round-trip via unit test.
- (b) Hash-locks unchanged.
- (c) Bench `examples/zensim_profile_compare_bench.rs` shows expected profile differences (V0_3 vs V0_5Tuner at d=1.0/2.0/3.0).

### DO NOT

- DO NOT default-flip the profile to V0_5Tuner — V0_3 has higher CID22 SROCC (0.9367 vs 0.9278) and 2× lower wall cost. V0_5Tuner is opt-in.

### Chunk N+1 plan

Phase 6 (tracking sweep) fires next.

---

## §8. Phase 6 — 6-backend tracking sweep + Pareto decision

**Sibling workspace**: `~/work/zen/jxl-encoder--zensim-tracking-sweep`.
**Branch**: `cvvdp-fork-rfc` (stacked on Phase 5).

### Scope (~1 week)

Mirror cvvdp Phase 6 (commit `8b5a13a7`). Expand from 4-backend to 6-backend sweep:

| cell tag | encoder backend          | feature flag       |
|---       |---                       |---                 |
| `B`      | butteraugli (CPU)        | (default)          |
| `B_GPU`  | butteraugli (GPU)        | `gpu-butteraugli`  |
| `C_GPU`  | cvvdp (GPU)              | `cvvdp-loop`       |
| `C_CPU`  | cvvdp (CPU)              | `cvvdp-loop-cpu`   |
| **`Z_GPU`** | **zensim (GPU)**       | **`zensim-loop-gpu`** |
| **`Z_CPU`** | **zensim (CPU)**       | **`zensim-loop`**  |

Corpus: same as cvvdp Phase 6 — 54 distinct images (CID22 + GB82-SC + W44-PHASE4-S1 extras) × 7 distances × 3 efforts = 1,134 cells per backend. Total: 6 × 1,134 = 6,804 cells.

Per cell, compute every metric:
- `score_butter_cpu`, `score_butter_gpu`
- `score_cvvdp_gpu`, `score_cvvdp_cpu`
- `score_zensim_cpu` (new — via `Zensim::compute_with_ref` on the encoded output)
- `score_zensim_gpu` (new, parity check)
- `score_ssim2` (legacy reference)

Decision rule per RFC #1 §8 (REVERT / DEFAULT_FLIP / OPT_IN_ONLY):

1. REVERT if any decoder roundtrip fails. Hard gate.
2. DEFAULT_FLIP if zensim strictly Pareto-dominates butteraugli on ≥ 1 corpus + within 5% on others.
3. OPT_IN_ONLY otherwise.

Tasks:

1. Extend `examples/cvvdp_track_baseline.rs` with `--backend {B|B_GPU|C_GPU|C_CPU|Z_CPU|Z_GPU}` flag.
2. Extend `scripts/cvvdp_pareto_analysis.py` to include zensim columns.
3. Run all 6 sweeps; populate TSV.
4. Apply RFC §5.4 decision rule.
5. `docs/ZENSIM_FORK_DECISION.md` — verdict + supporting data + Phase 7-or-8 recommendation.
6. Multi-decoder spotcheck — 20 zensim-encoded cells × 3 decoders = 60 cells.

### Inputs to read FIRST

1. `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §8 (Pareto-position validation).
2. `~/.claude/projects/.../cvvdp_fork_phase6_tracking_sweep_shipped_2026-05-24.md` — cvvdp Phase 6 memo. Same shape.
3. `~/work/zen/jxl-encoder/docs/CVVDP_FORK_DECISION.md` — verdict format.
4. `~/work/zen/jxl-encoder/scripts/cvvdp_pareto_analysis.py` — analyzer to extend.
5. `~/work/zen/jxl-encoder/scripts/cvvdp_phase6_finalize.sh` — finalize helper.

### Acceptance gates

- (a) ≥ 1,000 cells per backend populated (6 × 1,000 = 6,000+ rows).
- (b) Pareto analysis script committed + runnable + produces TSV + meta.
- (c) Decision memo committed with explicit recommendation tied to bench numbers.
- (d) `cargo fmt --check` PASS on harness + spotcheck examples.
- (e) Multi-decoder roundtrip 20 cells × 3 decoders = 60 PASS.
- (f) Decision memo + bench data committed.
- (g) Pushed to branch.

### DO NOT

- DO NOT default-flip zensim ON without ≥ 85% Pareto-front-pct.
- DO NOT cite "FMA precision" for any zensim-vs-butteraugli delta.
- DO NOT re-investigate transient panic rows without first checking if they cluster on specific image sizes.
- DO NOT spawn a Phase 7 default-flip agent if verdict is OPT_IN_ONLY. Phase 7 OPT_IN brief is the right next chunk.

### Chunk N+1 plan

If verdict is DEFAULT_FLIP: Phase 7 default-flip + docs cascade. If OPT_IN_ONLY: Phase 7 OPT_IN brief. If Pareto-pct < 85%: Phase 8a diagnosis → 8c renorm → 8d tighten → 8g block-constants refit.

---

## §9. Phase 7 — Docs closeout (OPT_IN_ONLY shape)

**Sibling workspace**: `~/work/zen/jxl-encoder--zensim-optin-closeout`.
**Branch**: `cvvdp-fork-rfc` (stacked on Phase 6).

### Scope (~1 day)

Mirror cvvdp Phase 7 (commit `9d4e4999`). Zero encoder source touched — pure docs cascade.

Tasks:

1. **`CHANGELOG.md`** — append zensim section under `[Unreleased]`:
   - **Added**: `zensim-loop` + `zensim-loop-gpu` cargo features (Phase 3); `PerceptualMetric::Zensim` variant (Phase 0); buttloop wiring + zensim targets seed (Phase 4); Phase 6 sweep + verdict.
   - **Changed**: (Phase 0's API rename if not already documented).
   - **Documentation**: this RFC + decision memo + 5 phase memos.
   - **Measured**: 6-backend × 1,134-cell sweep + Pareto analyzer outputs + multi-decoder spotcheck.

2. **`README.md`** — new section "## zensim-driven quantization loop (experimental opt-in)" mirroring cvvdp's. ≤ 400 words. Covers what zensim is (multi-scale XYB SSIM + per-codec calibration), how to opt in, builder API, code example, when zensim wins (e.g. JXL-to-JXL comparisons per RFC #2 §6.1), when butteraugli wins (default everywhere else), wall-time numbers from Phase 1+4 benches.

3. **`docs/RFC_ZENSIM_FORK_PLAN.md`** §9 — status flipped from SCOPING → SHIPPED-OPT-IN. Append 6-phase commit chain.

### Inputs to read FIRST

1. `~/.claude/projects/.../cvvdp_fork_phase7_optin_closeout_shipped_2026-05-24.md` — cvvdp Phase 7 memo. Same shape.
2. `~/work/zen/jxl-encoder/CHANGELOG.md` — cvvdp section to mirror.
3. `~/work/zen/jxl-encoder/README.md` — cvvdp section to mirror.
4. `~/work/zen/jxl-encoder/docs/ZENSIM_FORK_DECISION.md` — Phase 6 verdict.

### Acceptance gates

- (a) `cargo fmt --check` PASS (no Rust source touched).
- (b) `cargo build` PASS (default features).
- (c) CHANGELOG entries reference commit SHAs (Phase 1 zenmetrics + Phase 0-6 jxl-encoder commits).
- (d) README section word count ≤ 400.
- (e) RFC status flipped.
- (f) Hash-locks 36/36 BYTE-IDENTICAL.
- (g) `cargo doc -p jxl-encoder --no-deps` PASS.

### DO NOT

- DO NOT flip default-on zensim without re-running Phase 6 verdict with the OPT_IN_ONLY → ALWAYS_ON-when-feature-compiled criterion + a broader corpus sweep.
- DO NOT propose merging the branch into main without explicit user instruction.

### Chunk N+1 plan

Done unless Phase 8 fires (conditional on Phase 6 Pareto-pct < 85%).

---

## §10. Phase 8 (CONDITIONAL) — Pareto refit

Fires only if Phase 6 verdict is OPT_IN_ONLY with Pareto-pct < 85%. Mirrors cvvdp Phase 8 exactly:

### §10.1. Phase 8a — Pareto position diagnosis

Mirror cvvdp Phase 8 RFC `docs/RFC_CVVDP_PHASE8_PARETO_TARGETING.md`. Diagnose which intervention is needed:

- **Intervention A** (per-block reducer constant refit): if `bad_rate` distribution gap is the dominant cause.
- **Intervention B** (replace reducer): if Intervention A plateaus.
- **Intervention C** (bytes-tighten exit pass): if convergence consistently overshoots target.
- **Intervention BX** (per-distance scale): if single-value fits leave a residual gap.

### §10.2. Phase 8b/8c — Diffmap renormalization

If Intervention A is the path:

1. Capture diffmap distribution via `JXL_PHASE8B_DIFFMAP_DUMP` env hook (already exists; `maybe_dump_diffmap_stats` callers tag `"Z_CPU"`/`"Z_GPU"`).
2. Compute `ZENSIM_DIFFMAP_RENORM_SCALE` per RFC #1 §4.2 formula: `(target_z / target_b) * (mean_b / mean_z)`.
3. Apply in-place in `compare_with_reference` before returning.

### §10.3. Phase 8d — Bytes-tighten exit pass

If convergence overshoots target:

1. Add `LossyConfig::with_zensim_bytes_tighten(Option<bool>)` (mirror cvvdp's `cvvdp_bytes_tighten`).
2. Cargo feature `zensim-loop-tighten = ["zensim-loop"]`.
3. Post-convergence multiplicative bump on `quant_field_float` with re-score + accept/reject.
4. Mirror cvvdp Phase 8d structure at `cvvdp_fork_phase8d_bytes_tighten_shipped_2026-05-25.md`.

### §10.4. Phase 8g — Per-block reducer constants refit

If §8c renorm + §8d tighten plateau below 85%:

1. Capture `tile_dist` distribution per backend via `JXL_PHASE8G_TILE_DIST_DUMP` env hook.
2. Compute bad-rate per (distance, iter) per backend.
3. Fit `ZENSIM_BLOCK_CONSTANTS::k_tile_norm` per RFC #1 §3.2 methodology.
4. Apply via `block_reducer_constants_for_backend(PerceptualMetric::Zensim)`.

### §10.5. Phase 8f — Decision memo update

If Phase 8c+8d+8g cumulative Pareto-pct ≥ 85%, queue Phase 8f: update `docs/ZENSIM_FORK_DECISION.md` from OPT_IN_ONLY → ALWAYS_ON-when-feature-compiled + broader-corpus validation sweep.

### §10.6. Effort estimate

cvvdp Phase 8c → 8d → 8g took ~3 weeks (`0ab5a53d` → `7876ba95` → `689ba0df`). Same shape for zensim. Don't budget Phase 8 unless Phase 6 demands it.

---

## §11. Cross-phase invariants

These hold across every phase. Re-stated for the avoidance of doubt:

1. **Hash-locks 36/36 BYTE-IDENTICAL at default features** — every phase verifies this in its CI gate. zensim opt-in cannot regress the default path.
2. **`strategy_libjxl_byte_lock` PASS with all feature combos** — Libjxl strict-parity is mandatory.
3. **Multi-decoder roundtrip ≥ 4 cells × 3 decoders per phase** — every phase that touches encoder behaviour must validate against jxl-rs + djxl + jxl-oxide.
4. **`divergence_table_drift` PASS** — every gate added to the encoder must have a Section E row in `docs/LIBJXL_DIVERGENCES.md`.
5. **`docs/HYPOTHESIS_LEDGER.md` updated when each phase ships** — per CLAUDE.md research methodology rule #6.
6. **NO "FMA precision" citations** — per W44-66 user correction (CLAUDE.md). Every byte movement has a structural cause.
7. **Sibling-workspace discipline** — every chunk spawns in `~/work/zen/jxl-encoder--<chunk-name>/`, cleans up on merge.
8. **Atomic commit + push** — every phase lands as ONE commit pushed to the branch. No half-shipped state.

## §12. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---  |---         |---     |---         |
| Phase 1 GPU diffmap kernels harder than cvvdp's | Medium | High (2-3 week slip) | Same engineer who shipped cvvdp-gpu Phase 1 owns this; pattern proven |
| zensim CPU wall > 250 ms/iter at 1024² | Medium | High (forces opt-in regardless of Pareto) | Phase 1 closing bench catches this early; can ship opt-in-only |
| Phase 6 Pareto-pct < 60% (worst case) | Low | High (Phase 8 multi-week) | cvvdp arc precedent: 40% → 85% via Phase 8c+8d+8g. Same recipe should work for zensim |
| zensim profile choice (V0_3 vs V0_5Tuner) materially changes verdict | Low | Medium | Phase 5 ships profile selector; Phase 6 sweeps both |
| HDR encoders need zensim and zensim doesn't support PQ/HLG | Medium | Low (current butter behaviour same) | Document as known limitation; zensim short-circuits to butter on non-SDR per RFC #2 §3.4 |
| Migration breaking-change cascade hits imageflow / dotnet bindings | Low | Medium | RFC #3 §6 explicit hard-rename; CHANGELOG + comms before publish |
| Multi-metric API refactor (Phase 0) introduces hash-lock drift | Low | High | Default `PerceptualMetric::Butteraugli` + `PerceptualDevice::Auto` produces identical dispatch to pre-Phase-0; gate is mechanical |

## §13. Out of scope (for this RFC)

- Replacing SSIMULACRA2 in non-buttloop contexts (legacy tuning sweeps stay on SSIM2 metric).
- Per-codec affine calibration in the buttloop (zensim's `compute_with_codec_hint` is an explicit-call API; the buttloop's compare path stays codec-agnostic).
- Per-image content-class auto-dispatch (RFC #3 §5 surfaces the explicit-choice mandate).
- Adding HDR support to zensim (separate upstream workstream).
- Replacing cvvdp with zensim (both ship; caller picks).
- Streaming / animation buttloop zensim integration (cvvdp Phase 4-7 left this out; same precedent here).
- Per-window quality probe inside the buttloop (zensim's `compute_with_ref_into` + window-pre-slice pattern at `zensim/src/lib.rs:37-78` is a separate API surface, not the buttloop driver).
- Reverting any W44 cost-model gates (orthogonal; calibrated against butter, stays butter-calibrated).

## §14. Cross-references

- **Spec**: `docs/RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` (the contract every backend satisfies).
- **Audit**: `docs/RFC_ZENSIM_BUTTLOOP_AUDIT.md` (zensim vs the contract).
- **Architecture**: `docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` (the API design this RFC implements).
- **Precedent**: `docs/RFC_CVVDP_FORK.md` + `docs/CVVDP_FORK_DECISION.md` + the 6 cvvdp Phase 1-7 + Phase 8c/d/g memos at `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_*.md`.
- **Calibration scripts**: `scripts/cvvdp_pareto_analysis.py` + `scripts/cvvdp_pareto_diagnosis.py` + `scripts/cvvdp_calibration_seed.py` (all generalize to zensim with column rename).
- **Decision template**: `docs/CVVDP_FORK_DECISION.md` (Phase 6 verdict format to mirror).
- **Gate transfer audit template**: `docs/CVVDP_W44_GATE_TRANSFER.md` (one row per W44-gate × metric impact).
