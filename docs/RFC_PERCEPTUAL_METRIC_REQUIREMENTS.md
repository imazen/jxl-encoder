# RFC: What a perceptual metric needs to drive the buttloop AND sit on the Pareto front

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-25
**Status**: SCOPING (research-only; this RFC is a prerequisite for `RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` + `RFC_ZENSIM_BUTTLOOP_AUDIT.md` + `RFC_ZENSIM_FORK_PLAN.md`)

This RFC consolidates the lessons of the cvvdp fork's 7-phase + Phase 8a/b/c/d/g arc (2026-05-24..05-25, branch `cvvdp-fork-rfc@origin`) into a contract specification: **what must a perceptual metric provide for it to be a usable backend for the jxl-encoder buttloop AND produce Pareto-competitive bytes-vs-quality output?**

The cvvdp arc shipped one new metric (cvvdp-gpu + cvvdp-cpu) end-to-end and surfaced every gap between "shipped scalar+diffmap API" and "shipped Pareto-competitive replacement for butteraugli". Every requirement below is anchored in a measured finding or commit from that arc — no speculation. Future metric integrations (zensim being the next candidate) need to satisfy these contracts before they can flip from OPT_IN_ONLY to default-on.

## §1. Scalar score contract

The buttloop calls the metric ~3-5× per encode (one per iter at e=8+; ~2 iters at e=8, ~4 at e=9 in `vardct/perceptual_loop.rs::butteraugli_refine_quant_field_inner_seed`). Each call returns a `BackendCompareResult { score: f64 }` (`vardct/perceptual_backend.rs:54-58`) that drives:

1. **Termination / early-out**: the seed picker's `accept_bound = K_BUTTERAUGLI_ACCEPT_FACTOR * effective_metric_target_distance` test (`perceptual_loop.rs:2235`).
2. **Per-iter qac refinement direction**: the seed-vs-iter monotonicity check in the inner seed loop.
3. **Best-seed pick** across mean_qf seed candidates.

### §1.1. Direction convention

**Smaller = better** is the buttloop's hard requirement. Established by W44-126 (`vardct/perceptual_loop.rs:2235`, `K_BUTTERAUGLI_ACCEPT_FACTOR * target`). Every backend must normalize to this direction at the trait boundary.

For metrics whose natural output is larger=better (cvvdp JOD ∈ [0, 10] where 10 = perfect; zensim score ∈ [0, 100] where 100 = identical), normalize via `score = (max_score - native).clamp(0.0, max_score)` (cvvdp recipe at `cvvdp_backend.rs:304-305`, mirror-applies to zensim).

The normalization MUST be applied INSIDE the backend's `compare_with_reference` impl, BEFORE returning `BackendCompareResult`. Do NOT push direction conversion into the buttloop body — it would be a per-metric branching surface that has no business at the call site.

### §1.2. Numerical range that buttloop convergence reasons about

Butteraugli's score is unbounded but typically lives in `[0, 5]` for d=0.5..5.0 encodes (W44 cost-model calibration assumes this band). cvvdp's normalized `(10 - JOD)` lives in `[0, 0.3]` at typical distances per `cvvdp_targets.rs:61-69`. **The buttloop's `target_distance` field is a butteraugli-direction number** — the comparison surface needs explicit per-metric calibration; see §1.3.

The score MUST be `f64` (preserves 1e-7 precision across reduction stages — GPU vs CPU divergence floor per W44-RECON-DEEP/A7). Returning `f32` loses the bottom 10× of detail at the convergence threshold.

The score MUST be finite. NaN/Inf at the trait boundary is a contract violation — the buttloop's `accept_bound` comparison silently mis-fires on NaN. Backends MUST `clamp(0.0, MAX)` defensively even when the underlying math should produce a finite value (e.g. cvvdp's JOD on identical-plus-noise inputs can drift above 10.0 by 1e-7; the clamp at `cvvdp_backend.rs:305` exists for this reason).

### §1.3. Distance-target calibration lookup

The buttloop's caller-facing `target_distance: f32` is a **butteraugli-native** parameter (libjxl convention; d=1.0 is the "visually lossless threshold"). Every metric backend that doesn't natively produce butteraugli-direction scores needs a **per-distance target lookup table** that maps `target_distance` → metric-native target.

Reference: `vardct/cvvdp_targets.rs` (cvvdp Phase 4, `48b6ead1` commit, 7-entry table at d ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0} with linear interpolation + endpoint clamps). The dispatch site:

```rust
// perceptual_loop.rs:2157-2167
let effective_metric_target_distance: f32 = {
    if cvvdp_loop_active && !use_vdp2 {
        super::cvvdp_targets::cvvdp_target_score_for_distance(target_distance)
    } else {
        target_distance  // butteraugli pass-through
    }
};
```

The same shape applies to every future metric: `zensim_target_score_for_distance(d) -> f32` lives next to `cvvdp_target_score_for_distance`.

**How to seed the table** (cvvdp arc methodology, `cvvdp_targets.rs:23-41`):

1. Run a baseline sweep with butteraugli as the loop driver on a representative corpus (CID22-512 + GB82-SC + CLIC).
2. Score every output cell with the NEW metric (post-hoc).
3. For each distance band, compute the median NEW-metric score across the corpus.
4. Convert to direction-normalized form via §1.1.
5. (Optional) Multiply by a small tightening factor (cvvdp used 1.05× per `cvvdp_targets.rs:38`) so the new loop converges slightly stricter than what butteraugli achieves at the same nominal distance. This is empirically calibrated; the cvvdp table was overshoot in Phase 6 (verdict OPT_IN_ONLY), corrected by Phase 8c renorm.

**WARNING — the 1.05× tightening is NOT a recipe to copy blindly.** The Phase 6 sweep showed that cvvdp's tighter target table caused +155% to +286% bytes overhead on CID22 photos at the same nominal distance (`CVVDP_FORK_DECISION.md` headline). The Phase 8c diffmap renorm (§4) closed most of that gap by attacking the per-block reducer dispatch math — NOT the target table. Future metrics should aim for the target table to produce the same nominal byte budget as butteraugli on a held-out corpus, then bisect the diffmap renorm scale (§4) to close the per-block gap.

### §1.4. Identity → zero

The metric's native scalar score MUST equal "perfect quality" when reference == distorted, to within 1e-7 absolute precision per `cvvdp-gpu` invariant test (`score_with_diffmap_identity_yields_zero_diffmap`, zenmetrics master `8b658b4`). After direction normalization (§1.1), the buttloop sees a score of 0 on identical input — this is what makes the early-out work.

## §2. Diffmap contract — 5 PRACTICAL invariants + 1 ASPIRATIONAL

The buttloop's per-block reducer (`vardct/perceptual_loop.rs:2790-2843`) reduces the per-pixel diffmap to per-8×8-block `tile_dist` values via:

```rust
let td = k_tile_norm * (sum_v16 / pixels).sqrt().sqrt().sqrt().sqrt() as f32;
// where sum_v16 = Σ v^16 over the 64 pixels in the block.
```

This 16th-power-norm reduction is **metric-agnostic in shape** but **structurally calibrated to butteraugli's diffmap distribution** (see §3 for the K_TILE_NORM problem). Every metric must produce a diffmap satisfying the contract below.

### §2.1. The 5 PRACTICAL invariants (every impl MUST hold)

Established by Agent A (cvvdp-cpu) honest-stop at `cvvdp_cpu_port_shipped_2026-05-24.md:64-80`, ratified in `RFC_CVVDP_FORK.md:126-156`, byte-aligned across cvvdp-cpu + cvvdp-gpu via `cvvdp_gpu_diffmap_api_shipped_2026-05-24.md:36-42`.

1. **Identity → zero**: `diffmap(ref, ref)` is all-zero to 1e-7 absolute on every pixel. Test: `tests/diffmap_invariants.rs::score_with_diffmap_identity_yields_zero_diffmap`. Required because the buttloop's first iter on an identical seed must terminate with `tile_dist == 0` at every block.

2. **Non-negative**: every diffmap pixel `≥ 0`. The reducer's `v^16` would amplify a negative pixel's magnitude before sign-loss; downstream `(Σ v^16 / n)^(1/16)` must operate on a strictly-positive signal for the metric-agnostic math to mean anything.

3. **Monotone in distortion**: for two distorted images D1, D2 where D1 differs from ref strictly more than D2 (in any normed sense), `mean(diffmap_D1) ≥ mean(diffmap_D2)`. Without this the buttloop's per-iter direction can invert and never converge.

4. **Spatial localization**: a pixel-block perturbation in input produces a diffmap response concentrated near that block (not a global average). The per-block 8×8 median+MAD heuristic depends on it.

5. **Warm-reference invariance**: `score_with_diffmap(ref, dist)` and `warm_reference(ref); score_with_warm_ref_diffmap(dist)` produce the same diffmap byte-for-byte (or within 1e-7 if the warm path takes a different reduction order). Required because the buttloop calls `set_reference` once per encode-cell and `compare_with_reference` 3-5× per cell; the warm path must be numerically equivalent.

These five invariants are TESTABLE per cvvdp-gpu's `tests/diffmap_invariants.rs` + cvvdp-cpu's `tests/` (6 tests). Every new metric MUST ship the same shape: 5-8 unit tests gated under the metric's cargo feature, named to mirror the cvvdp ones.

### §2.2. ASPIRATIONAL strict invariant — deferred

The original `RFC_CVVDP_FORK.md` drafted:

> JOD == 10.0 - minkowski_p_norm(diffmap, p = params.beta_sch)

This strict form would let callers reason about diffmap norms == metric scores. **Agent A's cvvdp-cpu port measured it isn't achievable without restructuring the cvvdp pool order** — see `cvvdp_cpu_port_shipped_2026-05-24.md:79-81` + `crates/cvvdp-gpu/docs/DIFFMAP_DIVERGENCES.md`. Intermediate per-band/per-channel Minkowski folds don't cleanly factor back into a single base-res per-pixel pool; the canonical cvvdp algorithm pools per-channel then per-band, not the other way round.

**Per "honest-stop > false completion": ship the looser invariants and document strict as aspirational.** Future metric integrations are expected to honest-stop in the same way if the strict form doesn't drop out for free.

## §3. Per-block reducer compatibility

The 16th-power-norm in §2's preamble is **the single biggest invisible coupling between a metric and the W44 cost model**. The buttloop's per-block reducer was calibrated against butteraugli's per-pixel signal distribution shape — a narrow-peak distribution where few pixels carry most of the magnitude. Other metrics produce broader distributions where many pixels share moderate magnitude.

The reducer's `(Σ v^16 / n)^(1/16)` is **near-max-norm** on the narrow-peak case (butteraugli) but acts like a **flatter average** on the broad distribution (cvvdp). Result: post-direction-normalization, post-target-table-calibration, the cvvdp loop STILL drives qac UP on ~80% of blocks per iter (bad-block tightening) while butteraugli drives qac DOWN on ~90% of blocks (good-block loosening). Same nominal distance target → opposite per-iter dynamics → +155% to +286% bytes overhead.

### §3.1. The `K_TILE_NORM` per-metric constant

cvvdp Phase 8g (`689ba0df`, `cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md`) introduced a per-metric constants struct in `vardct/perceptual_loop.rs:1155-1283`:

```rust
pub(crate) struct BlockReducerConstants { pub k_tile_norm: f32 }
pub(crate) const BUTTER_BLOCK_CONSTANTS: BlockReducerConstants = BlockReducerConstants {
    k_tile_norm: 1.2,  // libjxl TileDistMap literal
};
#[cfg(feature = "cvvdp-loop")]
pub(crate) const CVVDP_BLOCK_CONSTANTS: BlockReducerConstants = BlockReducerConstants {
    k_tile_norm: 0.16,  // Phase 8g fit, 7.5× smaller than butter
};
pub(crate) fn block_reducer_constants_for_backend(cvvdp_loop_active: bool)
    -> BlockReducerConstants;
```

**Every new metric needs its own table row.** zensim will need `ZENSIM_BLOCK_CONSTANTS`. The dispatch site `perceptual_loop.rs:2796` extends naturally to a 3-way dispatch when zensim lands.

### §3.2. Bad-rate distribution as the structural test

Per Phase 8g methodology (`cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md:99-156`):

1. Capture per-iter `tile_dist` distribution per backend via env-gated dump `JXL_PHASE8G_TILE_DIST_DUMP=<path>` (the hook at `perceptual_loop.rs:1301-1313`).
2. For each (distance, iter) compute `bad_rate = fraction of blocks where tile_dist > effective_metric_target`.
3. At parity, `bad_rate_new ≈ bad_rate_butter` (within 0.01) at every distance band on the calibration corpus.
4. Compute per-distance scale to align `td_p95` to `effective_target`. The single-value fit is the median scale across distances (rounded to 2 sig figs).
5. Re-run the bad-rate capture; verify parity is achieved.

The Phase 8g fit derivation table (`cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md:130-141`):

| d   | cvvdp_p95 | target   | scale_to_align |
|---  |---        |---       |---             |
| 0.5 | 0.028     | 0.0029   | 0.104          |
| 1.0 | 0.105     | 0.0238   | 0.227          |
| 2.0 | 0.49      | 0.0724   | 0.148          |
| 3.0 | 1.39      | 0.1336   | 0.096          |

→ median 0.13 × 1.2 = 0.156, shipped **0.16**.

For zensim, the same harness recipe applies; the resulting constant will be its own value. The single-value fit is the Phase 1 shipping choice; per-distance lookup is a follow-on chunk (see §3.3).

### §3.3. WARNING — wrong-direction trap

cvvdp Phase 8g hit a near-fatal wrong-direction trap (`cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md:163-179`):

> Initial instinct: "scale cvvdp's `k_tile_norm` UP to match butter's `td_mean` because cvvdp's mean is 14.7× smaller". That would have HURT — increasing `k_tile_norm` scales tile_dist UP, pushing MORE blocks above the target → MORE bad blocks → MORE bytes.

The correct direction is to scale DOWN when the metric's distribution has more mass above the target. **Always look at the RATIO to TARGET, not the absolute distribution magnitude.** The mean-ratio is a red herring when the 16th-power norm transforms distributions differently per backend.

Encoded as a Phase 8g DO-NOT (`cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md:241-265`):

> 1. DO NOT flip CVVDP_BLOCK_CONSTANTS.k_tile_norm above 1.2 (BUTTER_BLOCK_CONSTANTS.k_tile_norm). The `cvvdp_block_constants_k_tile_norm_below_butter` invariant test enforces this.
> 2. DO NOT lower below 0.05 — bad_rate saturates at 0 (disables refinement entirely).

Future metric integrations should ship the same kind of invariant unit test (`metric_block_constants_within_reasonable_bounds`) so accidental refactors can't violate the calibration.

## §4. Diffmap renormalization factor

Separately from the per-block reducer constant (§3), the per-pixel diffmap itself needs renormalization at the trait boundary. This is what cvvdp Phase 8c shipped (`0ab5a53d`, `cvvdp_fork_phase8c_diffmap_renorm_shipped_2026-05-25.md`).

### §4.1. The Phase 8c formula

```rust
// vardct/perceptual_backend.rs:111
pub(crate) const CVVDP_DIFFMAP_RENORM_SCALE: f32 = 0.018;
```

Applied in-place on `diffmap_out` AFTER the metric compute, BEFORE returning to the buttloop (`cvvdp_backend.rs:330-335`):

```rust
let renorm = resolved_cvvdp_diffmap_renorm_scale();
if (renorm - 1.0).abs() > f32::EPSILON {
    for v in diffmap_out.iter_mut() { *v *= renorm; }
}
```

### §4.2. How to derive the scale

The right scale **aligns the BAD-BLOCK PREDICATE, not just the diffmap mean** (`cvvdp_fork_phase8c_diffmap_renorm_shipped_2026-05-25.md:60-74`).

The buttloop's bad-block predicate is `tile_dist / effective_metric_target > 1`. `effective_metric_target` differs between backends (butter: `target_b = distance`; cvvdp: `cvvdp_target_score_for_distance(distance)`). The right scale satisfies:

```
(mean_c * scale) / target_c ≈ mean_b / target_b
```

i.e.

```
scale = (target_c / target_b) * (mean_b / mean_c)
```

**The Phase 8c wrong-direction correction** (`cvvdp_fork_phase8c_diffmap_renorm_shipped_2026-05-25.md:76-84`):

> My initial commit shipped scale 0.660 derived from `1 / median(mean_c/mean_b)`. That formula IGNORED the target ratio. The Phase 8c first re-bench showed bytes WORSE than pre-renorm. After observing the regression I re-derived the scale using the corrected formula (target ratio × mean ratio). 0.018 was 37× smaller than 0.660 and empirically validated by the second re-bench.

**DO NOT** re-derive from mean-ratio alone — encoded as a Phase 8c DO-NOT (line 134-137). Same trap applies to every future metric: the target table (§1.3) and the renorm scale (§4) are coupled through the bad-block predicate; deriving one without the other gives the wrong answer.

### §4.3. Distribution capture harness

The env-gated `JXL_PHASE8B_DIFFMAP_DUMP=<path>` hook in `vardct/perceptual_backend.rs:240-299` writes one TSV row per `compare_with_reference` call with `mean / median / p25 / p75 / p95 / max` of the diffmap. Zero production cost when env unset; load-bearing for every future per-metric refit.

Future metric integrations should ship a `<metric>_phaseN_diffmap_distribution.rs` example that drives the same harness across 5 fixtures × 4 distances, capturing pre- and post-renorm distributions. Same shape as `cvvdp_phase8b_diffmap_distribution.rs`.

## §5. Warm-reference API shape

The buttloop's per-cell pattern (`vardct/perceptual_loop.rs::butteraugli_refine_quant_field_inner_seed`):

```
backend.set_reference(ref_r, ref_g, ref_b, w, h)?;       // once per cell
for iter in 0..iters {
    backend.compare_with_reference(dist_r, dist_g, dist_b, padded_w, w, h, &mut diffmap_vec)?;
    // ...qac refinement using returned score + diffmap...
}
```

### §5.1. Trait surface (mandatory)

```rust
pub(crate) trait PerceptualBackend: core::fmt::Debug {
    fn name(&self) -> &'static str;
    fn set_reference(&mut self, ref_r: &[f32], ref_g: &[f32], ref_b: &[f32],
                     width: usize, height: usize) -> Result<()>;
    fn compare_with_reference(
        &mut self,
        dist_r: &[f32], dist_g: &[f32], dist_b: &[f32],
        padded_width: usize, width: usize, height: usize,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<BackendCompareResult>;
    fn divergence_status(&self) -> Option<(f64, bool)> { None }
}
```

Source: `vardct/perceptual_backend.rs:325-381`. Every metric backend implements this trait exactly. **NO method signature changes** when adding a new metric.

### §5.2. Per-cell init cost vs per-iter cost

`set_reference` runs ONCE per encode cell (~3-5 iter compares). It can be 10× more expensive than `compare_with_reference` and still amortize cleanly. The pattern in cvvdp-gpu: `warm_reference_from_linear_planes` runs the reference's full DKL transform + Laplacian pyramid + per-band CSF; subsequent compares only run the distorted-side pyramid + masking + pool.

If the metric crate doesn't have a warm-ref API yet, the wrapping backend MUST cache the reference planes locally and pass them to a one-shot compare API on every iter (cvvdp-cpu's Phase 5 workaround, `cvvdp_fork_phase5_cpu_integration_shipped_2026-05-24.md:58-72`, ~5-10% per-iter overhead vs proper warm-ref). This is the worst-case shape; metric authors should add warm-ref APIs (Phase 1-style of the source-metric repo).

### §5.3. Linear-planes input (skip sRGB conversion)

The buttloop hands the backend **linear-RGB f32 planes**, NOT sRGB-u8. The reference is the same opsin-input the encoder used; the distorted is the reconstruction in linear space (`vardct/perceptual_loop.rs:~2400`).

Reference: butteraugli-gpu W44-PHASE3-B4 (`w44_phase3_b4_gpu_linear_planes_2026-05-23.md`) eliminated the host-side linear→sRGB-u8 LUT + GPU-side sRGB→linear kernel, recovering 11-21% wall on 6 of 6 cells. cvvdp-gpu followed the same pattern via Agent B's `*_from_linear_planes_*` API surface (`cvvdp_gpu_diffmap_api_shipped_2026-05-24.md:14-30`).

**A metric whose only API takes sRGB-u8 is shippable but slow.** zensim falls in this category for the GPU side today — see `RFC_ZENSIM_BUTTLOOP_AUDIT.md`. Phase 1 of the zensim fork is "add linear-planes input to zensim-gpu" + ensure the CPU side's existing `compute_with_ref_and_diffmap_linear_planar` survives the integration.

### §5.4. Caller-owned diffmap Vec (B7a buffer recycling)

The `&mut Vec<f32>` parameter on `compare_with_reference` (introduced by W44-PHASE3-B7a, `w44_phase3_b7_cpu_buffer_recycling_2026-05-23.md`) is **load-bearing**. The buttloop allocates the Vec once per cell (outside the iter loop), passes it by `&mut` on every iter, and the backend's `resize(_, 0.0)` is amortized O(1) once the high watermark is reached. Without it: 16 MB / encode of fresh allocation on 1024² + GC pressure on the 8-core rayon pool.

Every new metric impl must follow the same shape: `diffmap_out.clear(); diffmap_out.resize(W*H, 0.0); fill_with_metric_data(diffmap_out);`. The cvvdp-gpu pattern at `cvvdp_backend.rs:270-287` is the canonical reference.

### §5.5. Stride convention

- Reference planes (`set_reference`): **tight stride** == width. The contract doesn't take a stride arg; it's implicit.
- Distorted planes (`compare_with_reference`): **`padded_width` stride** (one block-row of padding). The backend MAY need to copy strided rows into a tight scratch buffer before passing to its metric crate's API (see `GpuButteraugliBackend::copy_strided_row_into_scratch` at `perceptual_backend.rs:817-838`).

The metric crate's native API SHOULD accept stride directly — zensim's `compute_with_ref_and_diffmap_linear_planar(planes, width, height, stride, ...)` is the right shape (`zensim/src/diffmap.rs:706-714`); the wrapping backend then passes `padded_width` directly without the row-by-row copy.

## §6. Wall-time budget

The buttloop dominates encode wall at e=8+ (W44-PHASE3-A6 measurement: ~20-30% of total wall, of which ~30-50% is the per-iter metric compute). To keep encode wall sub-second on 1024², the per-iter metric compute floor is ~50ms.

### §6.1. Measured wall-times

| metric              | impl                | wall @ 1024²    | source                                                |
|---                  |---                  |---              |---                                                    |
| butteraugli CPU     | rayon + AVX2/512    | ~150 ms / iter  | W44 baseline                                          |
| butteraugli GPU     | CUDA via CubeCL     | ~20 ms / iter   | W44-PHASE3-B1+B4 (`w44_phase3_b1_*` memo)             |
| cvvdp GPU           | CUDA via CubeCL     | ~20-25 ms warm  | `cvvdp_fork_phase5_cpu_integration_shipped:118`       |
| cvvdp CPU           | rayon + scalar      | ~222 ms warm    | `cvvdp_cpu_port_shipped:38`                           |
| zensim CPU          | rayon + AVX2        | TBD (audit §2)  | `RFC_ZENSIM_BUTTLOOP_AUDIT.md`                        |
| zensim GPU          | CUDA via CubeCL     | TBD (audit §2)  | `RFC_ZENSIM_BUTTLOOP_AUDIT.md`                        |

### §6.2. Practical floor for production-default-on

- ≤ 50 ms/iter @ 1024² for the metric to be a viable default-on backend at e=8.
- ≤ 100 ms/iter is acceptable for opt-in mode (cvvdp-cpu shipped at 222 ms, 1.55× butter CPU at e=8; opt-in only).
- > 250 ms/iter forces opt-in + a known DO-NOT note about wall hit on the marketing path.

The cvvdp-cpu integration shipped at 222 ms (`cvvdp_fork_phase5_cpu_integration_shipped:117-128`) — explicitly opt-in, GPU-first default policy. Future metric crates should target ≤ 50 ms before claiming a default-flip.

### §6.3. Diffmap overhead is part of the budget

Adding a per-pixel diffmap output to a metric pipeline isn't free. cvvdp-gpu measured +9.5% (256²), +31.7% (512²), +24.1% (1024²) wall vs the score-only path (`cvvdp_gpu_diffmap_api_shipped_2026-05-24.md:78-81`). The benchmark TSV `crates/cvvdp-gpu/benchmarks/diffmap_overhead_2026-05-24.tsv` captures the measurement.

Future metric integrations should ship the same `diffmap_overhead_<date>.tsv` so the user can see what they're paying for the per-pixel signal.

## §7. Defense-in-depth (mandatory)

### §7.1. `catch_unwind` around GPU init

Established by W44-PHASE3-B1 (`w44_phase3_b1_*` memo). Pattern at `perceptual_backend.rs:758-770`:

```rust
let client = match std::panic::catch_unwind(|| Runtime::client(&Default::default())) {
    Ok(c) => c,
    Err(_) => return None,  // CUDA missing/failed; caller falls back
};
```

Every GPU backend's `try_new` MUST wrap CubeCL init in `catch_unwind`. cvvdp-gpu's variant at `cvvdp_backend.rs:137-144` is the canonical reference. **Encoder must never crash on a missing CUDA driver** — silent fallback to the next dispatch tier is the contract.

### §7.2. Silent CPU fallback chain

The dispatch shape at `perceptual_backend.rs:1232-1457` cascades:

1. `cvvdp_requested + cvvdp_use_cpu_requested` → CPU cvvdp first
2. `cvvdp_requested + GPU OK` → GPU cvvdp
3. `cvvdp_requested + GPU FAIL + cvvdp-loop-cpu compiled` → CPU cvvdp fallback
4. `gpu_butteraugli_requested + GPU OK` → GPU butteraugli
5. `gpu_butteraugli_requested + GPU FAIL` → CPU butteraugli fallback
6. Default → CPU butteraugli

Future metric integrations extend this chain. The `eprintln!` one-shot fallback warning (`perceptual_backend.rs:1383-1389`) is the user-visible signal that a backend silently downgraded.

### §7.3. EncoderStrategy::Libjxl strict-parity bypass

**Mandatory and load-bearing.** `EncoderStrategy::Libjxl` ALWAYS routes to butteraugli regardless of any opt-in field. Enforced at the resolver level via `LossyConfig::resolve_cvvdp_loop` (`api.rs:6676-6681`):

```rust
pub(crate) fn resolve_cvvdp_loop(&self) -> bool {
    if matches!(self.strategy, EncoderStrategy::Libjxl) {
        return false;
    }
    self.cvvdp_loop.unwrap_or(false)
}
```

The W44-126 user-signoff invariant ("all-divergence parity for Libjxl strategy") cascades: every new metric opt-in must mirror the same short-circuit. The byte-lock test `tests/strategy_libjxl_byte_lock.rs` (W44-194) enforces this at CI time.

**Test pattern for new metrics**: ship a smoke test that constructs `LossyConfig::default().with_strategy(EncoderStrategy::Libjxl).with_<metric>_loop(Some(true))` and asserts bytes are byte-identical to `LossyConfig::default().with_strategy(EncoderStrategy::Libjxl)`. cvvdp Phase 4's `cvvdp_loop_smoke::libjxl_strategy_byte_identical_regardless_of_cvvdp_loop` is the canonical reference (5 cells × 3 cvvdp_loop values = 15 assertions).

## §8. Pareto-position validation (the ship-decision gate)

Per `RFC_CVVDP_FORK.md` §5.4, the decision rule for any new metric backend is:

1. **REVERT** if any decoder roundtrip breaks. Hard gate.
2. **DEFAULT_FLIP** if the new metric strictly Pareto-dominates butteraugli on at least one corpus + comes within 5% on the others.
3. **OPT_IN_ONLY** otherwise.

### §8.1. Methodology (Phase 6 shape)

Source: `cvvdp_fork_phase6_tracking_sweep_shipped_2026-05-24.md` + `scripts/cvvdp_pareto_analysis.py` + `docs/CVVDP_FORK_DECISION.md`.

1. **Corpus**: CID22-512 validation (41 photos held out from training) + GB82-SC (11 screenshots) + W44-PHASE4-S1 extras (2 lossless). Actual: 54 distinct images × 7 distances × 3 efforts = 1,134 cells per backend.
2. **Backends**: minimum 4 (butter CPU + butter GPU + new-metric GPU + new-metric CPU). When zensim ships, expand to 6: + zensim GPU + zensim CPU.
3. **Score every output with every metric**: per (image, distance, effort, encode-backend) cell, compute `score_butter_cpu`, `score_butter_gpu`, `score_cvvdp_gpu`, `score_cvvdp_cpu`, `score_zensim`, `score_ssim2` (the legacy reference).
4. **Per-(corpus, metric) Pareto frontier on (bytes, score)**. Count Pareto wins per backend.
5. **Per-backend wall p50/p95**.
6. **Per-distance mean bytes** + per-distance score deltas.

### §8.2. Pareto-front-pct binary metric

Phase 6 used `pareto_front_pct = fraction of cells where the backend is on the Pareto frontier of (bytes, metric)`. cvvdp Phase 6 verdict: butter 100% / 92.1% / 91.7% / 89.5% for (B / butter_cpu / butter_gpu / cvvdp_gpu) on GB82-SC butter_cpu metric. The 5pp gap below leader triggered OPT_IN_ONLY.

### §8.3. Bytes-at-equal-score continuous metric

The Pareto-front-pct metric is BINARY (on/off frontier). The continuous metric `bytes-at-cvvdp_9.99` / `bytes-at-cvvdp_9.98` (`cvvdp_fork_phase8d_bytes_tighten_shipped:13-29`) captures the actual quantitative improvement:

> Phase 8d (C_GPU_v3) closed -55% of the gap to B at the perceptually-demanding 9.99 cvvdp target.

Future metric Pareto evaluations should report BOTH the binary Pareto-front-pct AND continuous bytes-at-equal-score. The binary metric is the ship-gate; the continuous one is the progress-tracker.

### §8.4. Decision shapes from the cvvdp arc

The cvvdp arc landed across all three decisions in sequence:

- Phase 6: **OPT_IN_ONLY** (Pareto position 40.3% — well below 85% threshold). `CVVDP_FORK_DECISION.md`.
- Phase 8c (diffmap renorm): 40.3% → 60.0% (still OPT_IN_ONLY).
- Phase 8d (bytes-tighten): 60.0% binary unchanged; continuous bytes-at-9.99 closed 55% of gap. Still OPT_IN_ONLY but proxy-progressing.
- Phase 8g (per-block constants refit): 60.0% → **85.0%** (meets ≥85% ship target; Phase 8f decision-memo update queued).

**Pattern for future metric integrations**: don't expect a Phase 1-7 arc to produce a default-flip-worthy verdict. The cvvdp lesson is that the per-block reducer constant + diffmap renorm + target table are all coupled; you need 3-4 measurement phases AFTER the initial integration before the Pareto-position-pct hits the ship gate. Budget accordingly.

## §9. Synthesis — the contract checklist

Every metric integration MUST tick the following 14 boxes before opt-in shipment, plus 6 more for default-flip:

### §9.1. Opt-in ship gates (14)

- [ ] (G1) Scalar score: f64, smaller=better at trait boundary, finite, clamped.
- [ ] (G2) Identity → zero diffmap to 1e-7 absolute (unit test).
- [ ] (G3) Non-negative diffmap (unit test).
- [ ] (G4) Monotone in distortion (unit test or invariant-check on synthetic pairs).
- [ ] (G5) Spatially localized (unit test on block-perturbation).
- [ ] (G6) Warm-ref invariance to 1e-7 (unit test).
- [ ] (G7) Per-distance target table (`<metric>_targets.rs`) with ≥ 7 entries, monotone, well-behaved.
- [ ] (G8) Per-block reducer constant table (`<METRIC>_BLOCK_CONSTANTS`) with fit derivation + bad-rate parity verification.
- [ ] (G9) Diffmap renorm scale (`<METRIC>_DIFFMAP_RENORM_SCALE`) with target-aware derivation per §4.2.
- [ ] (G10) Warm-ref API (`set_reference` + `compare_with_reference`) with `&mut Vec<f32>` diffmap output.
- [ ] (G11) Linear-planes input (skip sRGB conversion).
- [ ] (G12) `catch_unwind` around GPU init in `try_new`.
- [ ] (G13) `EncoderStrategy::Libjxl` strict-parity bypass at resolver level.
- [ ] (G14) Multi-decoder roundtrip ≥ 10 cells × 3 decoders (djxl + jxl-rs + jxl-oxide).

### §9.2. Additional default-flip ship gates (6)

- [ ] (D1) Pareto-front-pct ≥ 85% across all corpora × all metrics (per §8.2).
- [ ] (D2) Wall-time ≤ 50 ms/iter @ 1024² (per §6.2).
- [ ] (D3) Hash-locks 36/36 byte-identical at default features (zero regen).
- [ ] (D4) `strategy_libjxl_byte_lock` 4/4 byte-identical (Libjxl invariant).
- [ ] (D5) Section E divergence row + W44 gate transfer audit (see `docs/CVVDP_W44_GATE_TRANSFER.md` shape).
- [ ] (D6) CHANGELOG + README + RFC status update (Phase 7-style closeout).

### §9.3. The cvvdp arc as the empirical proof of contract

Every gate above is anchored in one specific commit / memo from the cvvdp arc. The full chain:

1. RFC drafted: `4722c5ac` (Phase 1).
2. Trait extraction: `8c6e91cc` (Phase 2).
3. GPU backend impl: `57757ff8` (Phase 3).
4. Buttloop wiring + target table: `32581839` + seed `9c1cfa3e` (Phase 4).
5. CPU backend impl: `206b874e` (Phase 5).
6. Tracking sweep + Pareto decision: `8b5a13a7` (Phase 6).
7. OPT_IN docs closeout: `9d4e4999` (Phase 7).
8. Diffmap renorm fix: `0ab5a53d` (Phase 8c).
9. Bytes-tighten exit pass: `7876ba95` (Phase 8d).
10. Per-block constant refit: `689ba0df` (Phase 8g).

Total: 10 commits, ~3,000 LOC, ≥85% Pareto-front-pct on the calibration corpus. This RFC's contract is what makes Phase 11 (zensim) plannable as a discrete chunk-set rather than open-ended R&D.

## §10. Out of scope (for this RFC)

- Replacing SSIMULACRA2 in non-buttloop contexts. SSIM2 remains the legacy reference metric for tuning sweeps (`benchmarks/cjxl_parity_ledger_*.tsv`).
- Animation / video buttloop (cvvdp-gpu is still-image only per `RFC_CVVDP_FORK.md` §8).
- Replacing the per-block 16th-power-norm reducer with a metric-shaped one (Phase 8i candidate per `cvvdp_fork_phase8g_block_constants_shipped:223-226`). The current dispatch via `BlockReducerConstants` is sufficient for cvvdp and likely sufficient for zensim.
- W44 cost-model gate re-calibration per metric (`docs/CVVDP_W44_GATE_TRANSFER.md` audit). Each metric integration inherits the butteraugli-calibrated gates; per-metric refits are a separate multi-phase effort if needed.
- Per-image / content-class metric dispatch (e.g. "use cvvdp for screenshots, butteraugli for photos"). The current architecture is per-encode-call; per-image is a future API surface (`RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md` §6 may surface this).
