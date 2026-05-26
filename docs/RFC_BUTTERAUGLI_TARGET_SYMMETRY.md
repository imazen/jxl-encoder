# RFC: Butteraugli target-symmetry — closing the implicit-identity gap

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-26
**Status**: SCOPING (research + spec only; no Rust code in this RFC)

The cvvdp arc (`8c6e91cc..689ba0df`, 10 commits) shipped a Pareto-competitive
second perceptual backend AND surfaced a structural gap in the multi-metric
API: **butteraugli is the implicit-identity case in `with_perceptual_target_score`
handling**. cvvdp ships an explicit JOD → distance calibration table at
[`vardct/cvvdp_targets.rs`](../jxl-encoder/src/vardct/cvvdp_targets.rs);
butteraugli has nothing because its native unit IS the distance number. The
result is that `LossyConfig::with_perceptual_target_score(Some(score))`
silently does nothing on the default (butteraugli) path — it's plumbed
through to [`MetricSelection.target_score`](../jxl-encoder/src/vardct/perceptual_backend.rs)
but the `'lookup` dispatch in [`perceptual_loop.rs:2350-2386`](../jxl-encoder/src/vardct/perceptual_loop.rs)
never consults it. The user-facing setter is a no-op for the default metric.

This RFC fixes the asymmetry. It scopes a `butteraugli_targets.rs`
calibration table (Phase 1), audits the API-shape mismatch with the cvvdp/zensim
7-method family (Phase 2), and scopes a `DiffmapStats` introspection API
(Phase 3). The phased ship plan + sketched briefs live in
[`RFC_BUTTERAUGLI_FORK_PLAN.md`](RFC_BUTTERAUGLI_FORK_PLAN.md).

The underlying contract — every metric MUST provide an inverse mapping
`target_score → effective_distance` for symmetric API behaviour — is
appended to [`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md)
as §9.

## §1. The targeting asymmetry

The buttloop's caller-facing `target_distance: f32` is a butteraugli-native
parameter (libjxl convention; d=1.0 ≈ visually lossless threshold). The
buttloop's INTERNAL convergence target is a metric-native score with
`smaller=better` direction normalization (per
[`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §1.1).

For non-butteraugli metrics, that translation requires a per-distance
lookup table. cvvdp Phase 4 (commit
[`32581839`](https://github.com/imazen/jxl-encoder/commit/32581839),
memo `cvvdp_fork_phase4_loop_wiring_shipped_2026-05-24.md`) shipped exactly
that: [`cvvdp_targets.rs:117-313`](../jxl-encoder/src/vardct/cvvdp_targets.rs)
defines a `DisplayCalibration { display, entries: [(f32, f32); 7] }` row
per `DisplayConfig` variant and a linear-interp lookup
`cvvdp_target_score_for_distance_and_display(d, display) -> f32`. zensim's
Phase 4 (`RFC_ZENSIM_FORK_PLAN.md` §6) is queued to mirror the pattern.

### §1.1. Two-column comparison

| Direction | cvvdp                                          | butteraugli (today)      |
|---        |---                                             |---                       |
| metric → distance (`with_perceptual_target_score(Some(score))`) | NOT IMPLEMENTED (TODO Phase 1) | NOT IMPLEMENTED (this RFC, Phase 1) |
| distance → metric (`with_distance(d)` drives loop) | `cvvdp_target_score_for_distance_and_display(d, display)` (Phase 4) | implicit identity (pass-through at [`perceptual_loop.rs:2385`](../jxl-encoder/src/vardct/perceptual_loop.rs)) |

The current dispatch site:

```rust
// vardct/perceptual_loop.rs:2350-2386 (verified at HEAD a3e937d3)
let effective_metric_target_distance: f32 = 'lookup: {
    #[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
    {
        if self.zensim_loop && !use_vdp2 {
            break 'lookup super::zensim_targets::zensim_target_score_for_distance(
                target_distance,
            );
        }
    }
    #[cfg(feature = "cvvdp-loop")]
    {
        if self.cvvdp_loop && !use_vdp2 {
            break 'lookup
                super::cvvdp_targets::cvvdp_target_score_for_distance_and_display(
                    target_distance,
                    self.target_display,
                );
        }
    }
    // NOTE: butteraugli arm — no `super::butteraugli_targets::...` call.
    target_distance
};
```

The trailing `target_distance` (line 2385) is the butteraugli identity arm —
it's the source of the implicit-identity gap.

### §1.2. The `with_perceptual_target_score` no-op cluster

`LossyConfig::with_perceptual_target_score(Option<f32>)` was added by
multi-metric Phase 0 ([`api.rs:6803-6816`](../jxl-encoder/src/api.rs)). The
setter stores the value on the config, the resolver
([`api.rs:6895-6897`](../jxl-encoder/src/api.rs)) plumbs it into
`MetricSelection`, the dispatch site
([`perceptual_backend.rs:1392`](../jxl-encoder/src/vardct/perceptual_backend.rs))
binds it to `_` to silence the unused-field warning, and the buttloop body
hard-codes `target_score: None` on the second `MetricSelection` it builds
internally ([`perceptual_loop.rs:1792-1807`](../jxl-encoder/src/vardct/perceptual_loop.rs)):

```rust
// perceptual_loop.rs:1792-1807 — verified at HEAD a3e937d3
let selection = MetricSelection {
    metric,
    device,
    // Per-distance target override is plumbed via
    // `perceptual_target_score` on the API; the
    // buttloop body consumes it through the
    // `effective_metric_target_distance` lookup below.
    target_score: None,           // <-- hard-coded None; comment lies about lookup-below
    target_display: self.target_display,
};
```

The comment claims "the buttloop body consumes it through the
`effective_metric_target_distance` lookup below" — but lines 2350-2386
contain NO reference to `target_score`, `perceptual_target_score`, or any
caller-supplied override. **The wiring is half-shipped.** The setter on
`LossyConfig` exists, the field on `MetricSelection` exists, but the
consumption site in the buttloop body never reads it. Unit tests at
[`api.rs:16774-16777`](../jxl-encoder/src/api.rs) verify the setter
round-trips on the config but no integration test verifies the buttloop
acts on it.

The same gap exists for cvvdp and zensim today: `with_perceptual_target_score`
is documented as "override the metric's per-distance target table" but the
`'lookup` block always consults `cvvdp_targets` / `zensim_targets` even
when the caller sets a target score. Fixing the butteraugli case in this
RFC also fixes the cvvdp + zensim cases (the dispatch lives in the same
six-line block).

### §1.3. The empirical "butteraugli identity" is not bit-exact

The buttloop achieves butter score ≈ distance only on average. Per-image
the achieved score at each nominal distance varies materially. Mining the
Phase 6 tracking sweep `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
for `backend=B` rows (162 cells per distance):

| nominal `target_distance` | butter median achieved | p25     | p75     | p95     |
|---                        |---                     |---      |---      |---      |
| 0.5                       | 0.7223                 | 0.6449  | 0.7878  | 0.8962  |
| 1.0                       | 1.2245                 | 1.1519  | 1.2858  | 1.3935  |
| 1.5                       | 1.7285                 | 1.6135  | 1.8230  | 2.0123  |
| 2.0                       | 2.1936                 | 1.9614  | 2.3471  | 2.5734  |
| 3.0                       | 2.9616                 | 2.6421  | 3.2037  | 3.4590  |
| 4.0                       | 3.7608                 | 3.3719  | 3.9843  | 4.3332  |
| 5.0                       | 4.4004                 | 3.9147  | 4.6956  | 5.0572  |

The buttloop's convergence target IS `K_BUTTERAUGLI_ACCEPT_FACTOR ×
target_distance` (per
[`perceptual_loop.rs:2450`](../jxl-encoder/src/vardct/perceptual_loop.rs))
with the per-block 16th-power-norm pool reduced to per-tile `tile_dist`
values. The median achieved score is **higher than the target** at d=0.5
(0.72 vs 0.5) and **lower than the target** at d=5.0 (4.40 vs 5.0). The
identity is a useful approximation, not a bit-exact mapping.

This matters for two reasons:

1. A caller asking `with_perceptual_target_score(Some(0.5))` reasonably
   expects the encoded output to have butter ≈ 0.5. With today's
   implicit-identity arm, they would get `target_distance = 0.5` → achieved
   butter ≈ 0.72 (44% higher). A calibrated `butter_target_score_for_distance`
   lookup would convert the caller's score-target back to a `target_distance`
   that achieves the requested median.
2. The cvvdp `Phase 8c` diffmap-renorm formula
   ([`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §4.2)
   `scale = (target_c / target_b) × (mean_b / mean_c)` uses `target_b` as
   the butteraugli-anchor — and `target_b` IS the distance number (today's
   identity arm). If butteraugli's calibration table inverts that mapping
   (target_score → distance) for symmetric API, the `target_b` in the
   renorm formula stays the distance number (it's defined per the formula
   as "the buttloop's actual convergence target", which is `target_distance`
   per the K_BUTTERAUGLI_ACCEPT_FACTOR math). The renorm formula is
   untouched by Phase 1.

## §2. Butteraugli's current API surface (verified)

End-to-end grep against `~/work/butteraugli/butteraugli/src/precompute.rs`
+ `~/work/zen/zenmetrics/crates/butteraugli-gpu/src/pipeline.rs` at HEAD
confirms butteraugli is **functionally complete** for every method the
`PerceptualBackend` trait
([`vardct/perceptual_backend.rs:325-381`](../jxl-encoder/src/vardct/perceptual_backend.rs))
requires. The 7-method shape that cvvdp/zensim follow is naturally satisfied
by butteraugli's existing `ButteraugliReference` + `ButteraugliParams` API
+ `butteraugli-gpu`'s warm-ref / batch / diffmap surface:

### §2.1. CPU side (`butteraugli` crate)

| Role | Function | Source line |
|---   |---       |---          |
| Cold one-shot, sRGB    | `pub fn butteraugli(ref_srgb, dist_srgb, w, h, params) -> Result<ButteraugliResult>` | `lib.rs:555` |
| Cold one-shot, linear  | `pub fn butteraugli_linear(ref_linear, dist_linear, w, h, params) -> Result<ButteraugliResult>` | `lib.rs:618` |
| Warm-ref setup, sRGB   | `pub fn ButteraugliReference::new(rgb, w, h, params) -> Result<Self>` | `precompute.rs:117` |
| Warm-ref setup, linear | `pub fn ButteraugliReference::new_linear(rgb, w, h, params) -> Result<Self>` | `precompute.rs:163` |
| Warm-ref setup, planar | `pub fn ButteraugliReference::new_linear_planar(planes, w, h, stride, params) -> Result<Self>` | `precompute.rs:266` |
| Warm-ref compare, sRGB | `pub fn ButteraugliReference::compare(rgb) -> Result<ButteraugliResult>` | `precompute.rs:380` |
| Warm-ref compare, linear | `pub fn ButteraugliReference::compare_linear(rgb) -> Result<ButteraugliResult>` | `precompute.rs:405` |
| Warm-ref compare, planar | `pub fn ButteraugliReference::compare_linear_planar(planes, w, h, stride) -> Result<ButteraugliResult>` | `precompute.rs:435` |
| Warm-ref compare + recycle, planar | `pub fn ButteraugliReference::compare_linear_planar_into(planes, w, h, stride, diffmap_out) -> Result<f64>` | `precompute.rs:490` |
| Diffmap opt-in (params) | `ButteraugliParams::with_compute_diffmap(true)` + `result.diffmap()` | `lib.rs:276`/`484` |

The `compare_linear_planar_into` shape directly satisfies the trait's
`&mut Vec<f32>` recycling pattern from
[`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §5.4.

### §2.2. GPU side (`butteraugli-gpu` crate)

| Role | Function | Source line |
|---   |---       |---          |
| Cold one-shot | `Butteraugli::compute(ref_srgb, dist_srgb) -> Result<GpuButteraugliResult>` | `pipeline.rs:730` |
| Cold + options | `Butteraugli::compute_with_options(ref_srgb, dist_srgb, opts) -> Result<...>` | `pipeline.rs:737` |
| Warm-ref setup, sRGB | `Butteraugli::set_reference(ref_srgb) -> Result<()>` | `pipeline.rs:849` |
| Warm-ref setup, sRGB + options | `Butteraugli::set_reference_with_options(ref_srgb, opts) -> Result<()>` | `pipeline.rs:867` |
| Warm-ref setup, linear-planes | `Butteraugli::set_reference_from_linear_planes(ref_r, ref_g, ref_b) -> Result<()>` | `pipeline.rs:994` |
| Warm-ref setup, linear-planes + options | `..._with_options(...)` | `pipeline.rs:1013` |
| Warm-ref compare, sRGB | `Butteraugli::compute_with_reference(dist_srgb) -> Result<GpuButteraugliResult>` | `pipeline.rs:1058` |
| Warm-ref compare, linear-planes | `Butteraugli::compute_with_reference_from_linear_planes(dist_r, dist_g, dist_b) -> Result<...>` | `pipeline.rs:1101` |
| Diffmap copy (recycle) | `GpuButteraugliResult::copy_diffmap_to(&self, dst: &mut [f32]) -> Result<()>` | `pipeline.rs:2444` |
| Diffmap copy (alloc)   | `GpuButteraugliResult::copy_diffmap()` | `pipeline.rs:2423` |
| Has-cached-reference query | `has_cached_reference()` | `pipeline.rs:2429` |
| Multi-res setup (B5b divergence detector) | `Butteraugli::new_multires(...)` | `pipeline.rs:571` |
| Strip-tile mode (W44-PHASE3-B7d framework) | `Butteraugli::new_multires_strip(...)` | `pipeline.rs:605` |

The GPU side ships the full warm-ref + linear-planes + `_into`-style
diffmap recycling matrix. The B5b divergence detector at
[`perceptual_backend.rs:548-628`](../jxl-encoder/src/vardct/perceptual_backend.rs)
is butteraugli-only and depends on this matrix.

### §2.3. Conclusion — nothing structurally missing for the trait contract

Butteraugli already satisfies every gate in
[`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md)
§9.1 (G1-G14). The CPU + GPU backends at
[`vardct/perceptual_backend.rs:445-1224`](../jxl-encoder/src/vardct/perceptual_backend.rs)
have been shipping for months and are the reference implementation that
cvvdp Phase 2 (`8c6e91cc`) generalized the trait from.

What's missing is the **symmetric API behaviour around
`with_perceptual_target_score`** — a calibration table whose direction
goes `metric_score → effective_distance`. cvvdp ships the inverse direction
(`distance → metric_score`), and only that direction. Butteraugli has
neither — its native unit is distance, so the "distance → distance" trivial
mapping has always been the identity. Adding the "score → distance"
inverse mapping closes the gap.

## §3. Required new capabilities (RANKED by leverage)

| Gap | Severity | Effort | Phase |
|---  |---       |---     |---    |
| `target_score → effective_distance` inverse mapping (closes the no-op) | **HIGH** (blocks symmetric API behaviour for the default metric) | medium (table + linear interp + dispatch wiring) | **Phase 1** |
| API-shape audit: butteraugli's `compare_linear_planar_into` vs cvvdp/zensim's `_with_warm_ref_diffmap` naming family | LOW (wrappers paper over it in the backend impl) | small-medium (rename impacts every downstream caller) | Phase 2 (OPTIONAL — recommend status quo) |
| `DiffmapStats` introspection API on `ButteraugliReference` | LOW (Phase 8c/8g calibration was done with ad-hoc env-gated TSV dumps; future metric integrations would benefit from a stable API) | small (one method, one struct) | Phase 3 (OPTIONAL — only needed if a 4th metric joins) |

The Phase 1 work is the only ship-required item. Phase 2 + Phase 3 are
contingent on future demand; neither blocks any current functionality.

## §4. Design — three implementation options for the target_score inverse

The task: given a caller-supplied `target_score: f32` in butter direction
(smaller=better, identity-mapped to a butteraugli max-norm value), compute
an `effective_distance: f32` that the buttloop drives the convergence
loop with. The buttloop already consumes `effective_metric_target_distance`
at line 2386; the design only needs to populate the butteraugli arm.

### §4.1. Option A — Static calibration table (RECOMMENDED)

Mirror [`cvvdp_targets.rs`](../jxl-encoder/src/vardct/cvvdp_targets.rs)
exactly. Ship a new `vardct/butteraugli_targets.rs` module:

```rust
// vardct/butteraugli_targets.rs — Phase 1, RFC_BUTTERAUGLI_TARGET_SYMMETRY
//
// Closes the implicit-identity gap on the butteraugli arm of the buttloop's
// `effective_metric_target_distance` dispatch. Identical shape to
// `cvvdp_targets.rs`; the table direction is INVERTED — cvvdp's table goes
// `distance → score`, butteraugli's goes `score → distance` (the inverse).
// The buttloop site at `perceptual_loop.rs:2350-2386` consumes the
// resulting `effective_distance` and propagates through `accept_bound =
// K_BUTTERAUGLI_ACCEPT_FACTOR * effective_distance` exactly as today.

pub(crate) const BUTTERAUGLI_DISTANCE_BANDS: usize = 7;

/// 7-entry (butter_score_target, effective_distance) calibration. Seeded
/// from `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` `backend=B`
/// rows. Direction: stricter score (smaller value) → smaller
/// effective_distance (the buttloop converges tighter). Table is sorted
/// ascending in BOTH columns (monotone), so linear interp + clamp are
/// well-defined.
pub(crate) static BUTTERAUGLI_SCORE_TARGETS: &[(f32, f32); BUTTERAUGLI_DISTANCE_BANDS] = &[
    // (achieved butter median, nominal distance) — inverse of the
    // baseline lookup; caller passes the achieved target, we return
    // the distance that drives the loop to converge there.
    (0.7223, 0.5),
    (1.2245, 1.0),
    (1.7285, 1.5),
    (2.1936, 2.0),
    (2.9616, 3.0),
    (3.7608, 4.0),
    (4.4004, 5.0),
];

/// Symmetric to `cvvdp_target_score_for_distance`: caller passes a
/// butter-direction `target_score`, lookup returns the `effective_distance`
/// the buttloop drives convergence with.
pub(crate) fn butteraugli_effective_distance_for_target_score(target_score: f32) -> f32 {
    // ... linear interp + endpoint clamp, identical body to
    // cvvdp_target_score_for_distance_and_display ...
}

/// Identity-arm shim preserved for the existing implicit-identity path.
/// When the caller does NOT set `perceptual_target_score`, the buttloop
/// continues to use `target_distance` directly as before (byte-identical
/// to pre-Phase-1).
pub(crate) fn butteraugli_target_score_for_distance(d: f32) -> f32 { d.max(0.0) }
```

**Why this is the right shape**:

1. Mirrors the `cvvdp_targets.rs` pattern exactly — same module shape,
   same const seven-band structure, same linear-interp lookup. Easy to
   diff-review and easy to extend.
2. Default behaviour (no `perceptual_target_score`) is byte-identical to
   today — the identity arm is preserved as a const fn.
3. Two unit tests (monotonicity + endpoints + interp midpoint) are enough
   for the Phase 1 acceptance.
4. The Phase 1 sweep methodology is already done — the data lives in
   `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`, the seed
   medians are computed in §1.3 of this RFC.

**Costs**:

1. The table is per-corpus-median; per-image variance can be 30-50%
   (p25-p75 range at d=2.0 is 1.96-2.35). A caller wanting a specific
   achieved butter on a specific image will see imprecision — but that's
   ALSO true for cvvdp Phase 4's table, and cvvdp shipped anyway. The
   table is a best-effort calibration, not a guarantee.
2. The 7 distance bands are heuristic; per-distance refinement is a
   future chunk (no current demand).

### §4.2. Option B — Outer binary-search wrapper

Add `LossyConfig::with_target_score_iterate(bool)` opt-in. When set,
the encoder:

1. Trial-encodes at the static-table `effective_distance` (Phase 1's lookup
   gives the initial guess).
2. Decodes + measures actual achieved butter via in-process butteraugli.
3. Adjusts `effective_distance` by golden-section search (or bisection)
   until achieved butter is within ε of the target.
4. Re-encodes with the converged distance.

**Pros**: Exact target achievement on any single image. Lifts per-image
variance from "30-50% off the corpus median" to "≤ ε".

**Cons**: 4-6× wall cost (each iteration is a full encode + decode +
metric pass at ~1024² = ~500ms × 4-6 = 2-3 seconds per encode). The
buttloop already does ~3-5 internal iters at e≥8; this wraps a SECOND
iteration loop around it. For most callers, the static-table imprecision
is the right trade — 2-3 seconds per encode is unacceptable in any
production pipeline.

**Defer**: ship as Phase 2 or Phase 3, opt-in only, gated behind a new
cargo feature `butteraugli-target-iterate` to keep the dependency on the
in-process butteraugli decode-and-score path off the default build.

### §4.3. Option C — Live inverse via runtime stats

Track `(achieved_score, effective_distance)` pairs across encodes in a
thread-local rolling buffer. The lookup at site 2386 consults the rolling
mean to derive the inverse mapping live. This adapts to the caller's
actual corpus over time.

**Pros**: Self-calibrating; tracks corpus drift if the caller's content
mix changes.

**Cons**:

1. Most complex. Requires shared state across encodes (thread-local OK,
   but per-Encoder-instance is wrong for the iter shape).
2. Cold-start problem: the first ~50 encodes have no data to base the
   inverse on, so they fall back to Phase 1's static table anyway.
3. Reproducibility break — encodes from the same input produce different
   bytes depending on what came before. Caller can't reason about
   byte-output without freezing the inverse history.
4. The cvvdp arc didn't ship anything like this; no precedent to mirror.

**Defer indefinitely**. Document as a future research direction in §7.

### §4.4. Recommended pick

**Option A in Phase 1.** Static calibration table, seeded from the existing
Phase 6 tracking TSV. Mirrors the cvvdp_targets pattern; ~80 LOC of new
module + ~10 LOC of dispatch wiring + 7 unit tests. Phase 1 acceptance is
clear (the no-op gets closed; default-path bytes byte-identical).

**Option B as a stretch Phase 4+ (post-Phase-3) candidate** if/when a
caller surfaces the precision requirement. Not blocking.

**Option C deferred indefinitely.**

The combined "score → distance via Phase 1 + optional Phase 4 binary-search
refinement" covers every reasonable caller need without locking in a
research-grade live-inverse system the encoder doesn't have evidence to
support.

## §5. API-shape alignment audit

Butteraugli's method names diverge from cvvdp/zensim's. The trait surface
papers over this — every backend impl maps the metric-crate's names to
the trait's `set_reference + compare_with_reference` shape — but the
mismatch is visible if you cross-grep the three crates.

### §5.1. Comparison table

| Trait method role | cvvdp-gpu                                                  | zensim-gpu (Phase 1)                                         | butteraugli + butteraugli-gpu                                      |
|---                |---                                                         |---                                                           |---                                                                 |
| Warm-ref setup (linear) | `Cvvdp::warm_reference_from_linear_planes(ref_r, ref_g, ref_b)` | `Zensim::warm_reference_from_linear_planes(ref_r, ref_g, ref_b)` | CPU: `ButteraugliReference::new_linear_planar(planes, w, h, stride, params)` / GPU: `Butteraugli::set_reference_from_linear_planes(ref_r, ref_g, ref_b)` |
| Warm-ref compare (linear, NO diffmap) | `Cvvdp::score_from_linear_planes_with_warm_ref(dist_r, dist_g, dist_b)` | `Zensim::score_from_linear_planes_with_warm_ref(dist_r, dist_g, dist_b)` | CPU: `ButteraugliReference::compare_linear_planar(planes, w, h, stride)` / GPU: `Butteraugli::compute_with_reference_from_linear_planes(dist_r, dist_g, dist_b)` |
| Warm-ref compare (linear, WITH diffmap, recycled) | `Cvvdp::score_from_linear_planes_with_warm_ref_diffmap(dist_r, dist_g, dist_b, diffmap_out)` | `Zensim::score_from_linear_planes_with_warm_ref_diffmap(dist_r, dist_g, dist_b, diffmap_out)` | CPU: `ButteraugliReference::compare_linear_planar_into(planes, w, h, stride, diffmap_out)` (with `params.with_compute_diffmap(true)`) / GPU: `Butteraugli::compute_with_reference_from_linear_planes(...)` then `result.copy_diffmap_to(diffmap_out)` |
| Cold one-shot (no warm ref) | `Cvvdp::score_with_diffmap(ref_srgb, dist_srgb, diffmap_out)` | `Zensim::score_with_diffmap(ref_srgb, dist_srgb, diffmap_out)` | `butteraugli_linear(ref_linear, dist_linear, w, h, params)` + diffmap via `result.diffmap()` |
| Cold one-shot, linear-planes | `Cvvdp::score_from_linear_planes(...)` | `Zensim::score_from_linear_planes(...)` | (No direct equivalent — build a `ButteraugliReference::new_linear_planar` + `compare_linear_planar` pair instead. The CPU backend impl follows this pattern.) |

### §5.2. Verdict — keep status quo (Option 3, no rename)

The naming difference is real but **the wrapping backend impls already
absorb it** at
[`vardct/perceptual_backend.rs:445-1224`](../jxl-encoder/src/vardct/perceptual_backend.rs).
The `CpuButteraugliBackend::set_reference` impl calls
`ButteraugliReference::new_linear_planar`; the `compare_with_reference`
impl calls `compare_linear_planar_into`. The trait surface is uniform;
the metric-crate APIs differ.

**Reasons NOT to rename butteraugli's methods**:

1. **No external caller pain.** Butteraugli has stable downstream users
   (the jxl-encoder backend; the diff-validation suite at
   `~/work/zen/zenmetrics/butteraugli-gpu/tests/`; ad-hoc tooling). All
   of them use the existing names. Renaming would force a coordinated
   update across crates without measurable benefit.
2. **The names ARE descriptive in butteraugli's domain.** "Compare" vs
   "Score" is a stylistic choice; "linear_planar" vs "from_linear_planes"
   is a phrasing variant; "_into" vs "_with_warm_ref_diffmap" is shorter
   AND more idiomatic Rust. cvvdp's longer names were chosen because
   cvvdp doesn't have a separate `Reference` struct to anchor "warm" on —
   the name carries the warmness. Butteraugli encodes warmness in the
   `ButteraugliReference` type itself, so the method names don't repeat
   it.
3. **The wrapping layer is the right place to absorb the difference.**
   Both `CpuButteraugliBackend` and `GpuButteraugliBackend` are ~5-20 LOC
   each on the warm-ref + compare methods; the wrapping is mechanical
   and already correct.

**One scoped fix worth considering** (NOT in Phase 1 scope; flagged as a
potential Phase 2 micro-chunk): butteraugli's `compare_linear_planar_into`
returns `Result<f64>` (just the score), and the diffmap is computed
in-place inside the `ButteraugliReference`'s internal cache. The trait's
`compare_with_reference` returns `Result<BackendCompareResult>` with score
and the `diffmap_out` `&mut Vec<f32>` is populated by the impl. The
butteraugli CPU wrap at
[`perceptual_backend.rs:~525-585`](../jxl-encoder/src/vardct/perceptual_backend.rs)
copies from the internal cache into `diffmap_out` after the call. That
extra copy is ~8 KB / cell at 1024² (16 KB f32 for one channel, but
butteraugli's diffmap is single-plane), measured at ~0.05 ms — not worth
a rename to avoid.

### §5.3. If renaming is ever needed: the migration path

Should a future workstream want to rename butteraugli's CPU API to match
the cvvdp/zensim 7-method family:

1. Bump butteraugli crate to 0.2.0 (breaking change per semver — CLAUDE.md
   "no backwards-compat hacks").
2. Add new names alongside old (one release).
3. Remove old names in 0.3.0.

The change set in butteraugli would be ~10 method renames + ~20 docstring
updates + tests + CHANGELOG. Downstream impact is just the 2 backend
impls in jxl-encoder + the diff-validation suite in zenmetrics. NOT
recommended; the value is purely aesthetic, the cost is real.

**Recommendation: defer this Phase 2 audit. Document as out-of-scope. The
status-quo wrapper layer is the right shape.**

## §6. Diffmap-distribution introspection API

Phase 8c (`0ab5a53d`, memo `cvvdp_fork_phase8c_diffmap_renorm_shipped_2026-05-25.md`)
and Phase 8g (`689ba0df`, memo `cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md`)
derived cvvdp's `CVVDP_DIFFMAP_RENORM_SCALE = 0.018` and
`CVVDP_BLOCK_CONSTANTS.k_tile_norm = 0.16` by comparing butteraugli's and
cvvdp's per-pixel diffmap distributions across the calibration corpus.
The capture mechanism was ad-hoc:

```rust
// vardct/perceptual_backend.rs:370-430 (verified at HEAD a3e937d3)
#[cfg(feature = "std")]
pub(crate) fn maybe_dump_diffmap_stats(
    backend_name: &str,
    compare_call_idx: u32,
    width: usize,
    height: usize,
    diffmap: &[f32],
    score: f64,
) {
    let path = match std::env::var("JXL_PHASE8B_DIFFMAP_DUMP") { ... };
    // Compute mean / median / p25 / p75 / p95 / max, write TSV row.
}
```

This is env-gated, zero-cost-when-off, and writes one TSV row per
`compare_with_reference` call. cvvdp + butteraugli + zensim backends all
invoke this hook (cvvdp at
[`perceptual_backend.rs:641-648`](../jxl-encoder/src/vardct/perceptual_backend.rs);
butteraugli + zensim at similar `compare_with_reference` exit points).

### §6.1. The Phase 3 candidate — `DiffmapStats` introspection

The ad-hoc env-gated TSV is fine for one-off calibration but **brittle
under repeated use**:

1. Every Phase 8c/8g run re-derives the stats from scratch (the TSV is
   single-process; new sweeps overwrite).
2. The math (sort for percentiles, `O(N log N)` per call) is
   re-implemented at every consumer site.
3. The dump-on-compare side-effect couples the env var to the buttloop
   body — a caller programmatically running a bench loop has to set/unset
   the env var around their code.
4. No way for a programmatic harness to ask "what was the diffmap
   distribution on this last compare?" — must parse the TSV after the
   fact.

A clean Phase 3 API would add:

```rust
// butteraugli/src/lib.rs
pub struct DiffmapStats {
    pub n_pixels: usize,
    pub mean: f64,
    pub median: f32,
    pub p25: f32,
    pub p75: f32,
    pub p95: f32,
    pub max: f32,
}

impl DiffmapStats {
    /// Fraction of pixels above the threshold. Useful for bad-rate
    /// parity tests per RFC §3.2.
    pub fn bad_rate_at(&self, threshold: f32) -> f32 { ... }
}

impl ButteraugliReference {
    /// Returns the stats of the most recent compare's diffmap. None if
    /// the reference was just built (no compare yet) OR if the params
    /// were not configured with `with_compute_diffmap(true)`.
    pub fn diffmap_stats(&self) -> Option<DiffmapStats> { ... }
}
```

Same shape for `butteraugli-gpu::Butteraugli` + `GpuButteraugliResult`.

### §6.2. Phase 3 acceptance

- (a) `DiffmapStats` struct + `bad_rate_at` + `diffmap_stats()` getter
  shipped on both CPU (`butteraugli`) and GPU (`butteraugli-gpu`) crates.
- (b) Hash-locks 36/36 BYTE-IDENTICAL (the stats API is read-only).
- (c) Unit tests assert stats are consistent across the same compare
  (cold sRGB path vs warm-ref linear path produce the same diffmap →
  same stats).
- (d) Bench example `examples/diffmap_distribution_audit.rs` produces a
  TSV identical in shape to the env-gated dump.

### §6.3. When Phase 3 should land

Phase 3 is OPTIONAL for the current encoder shipment. It's worth doing
when:

- A 4th metric joins the lineup and the Phase 8c-style calibration needs
  to be re-run.
- A regression sweep wants programmatic access to per-iter diffmap
  distributions (e.g. proving that a W44 cost-model gate change doesn't
  shift the distribution in a way that would invalidate the existing
  `CVVDP_DIFFMAP_RENORM_SCALE`).
- The env-gated TSV is replaced by a programmatic capture in a long-running
  benchmark harness.

Until then, **the env-gated dump is sufficient**. The Phase 8c + 8g arcs
both shipped using it without API friction. Phase 3 is a "would be nice
to have" item, not a blocker.

**Recommendation: scope Phase 3 in this RFC, defer landing until a
demand surfaces. Phase 1 is the only mandatory work.**

## §7. Phased ship plan (summary; RFC #2 has the details)

| Phase | Scope | Effort | Status |
|---    |---    |---     |---     |
| **1** | `vardct/butteraugli_targets.rs` static table + `resolve_butteraugli_effective_distance` + buttloop dispatch wiring + 7 unit tests + integration test verifying `with_perceptual_target_score(Some(b))` actually drives the loop | 2-3 days | mandatory |
| 2 (optional) | API-shape alignment in `butteraugli` crate (rename to match cvvdp/zensim 7-method family) | 1 day + downstream coord | **recommend SKIP** — value is aesthetic, cost is real |
| 3 (optional) | `DiffmapStats` introspection API on `ButteraugliReference` + `GpuButteraugliResult` | 1-2 days | defer until demand surfaces |
| 4 (stretch) | Outer binary-search wrapper for `with_target_score_iterate(true)` exact-target convergence | 1 week | defer indefinitely; opt-in only if needed |

Phase 1 is fully scoped in [`RFC_BUTTERAUGLI_FORK_PLAN.md`](RFC_BUTTERAUGLI_FORK_PLAN.md)
with a ≥ 100-LOC brief.

## §8. Acceptance gates (Phase 1)

- (a) `cargo build` PASS on every feature combo:
  default, `butteraugli-loop` (default on), `gpu-butteraugli`, `cvvdp-loop`,
  `cvvdp-loop-cpu`, `zensim-loop`, `zensim-loop-gpu`, all combinations.
- (b) `cargo test --lib`: pre-Phase-1 count + 7 new tests (Phase 1 table
  unit tests) + 1-2 integration tests (`with_perceptual_target_score`
  actually drives convergence) — all PASS.
- (c) Hash-locks 36/36 BYTE-IDENTICAL at default features (`with_perceptual_target_score
  = None` is the default; Phase 1 must not move the default-path bytes).
- (d) `strategy_libjxl_byte_lock` 4/4 BYTE-IDENTICAL with the full feature
  stack compiled (Libjxl strategy forces `PerceptualMetric::Butteraugli`
  AND short-circuits `perceptual_target_score`; see §10).
- (e) `divergence_table_drift` 7/7 PASS.
- (f) Multi-decoder roundtrip ≥ 5 cells × {jxl-rs, djxl, jxl-oxide} using
  `with_perceptual_target_score(Some(0.5))`, `Some(1.0)`, `Some(2.0)` —
  all decode cleanly, all 15+ PASS.
- (g) New `butteraugli_targets_drift` test asserts table monotonicity in
  BOTH columns + endpoint clamps + interp midpoint matches average of
  endpoints to 1e-6.
- (h) New integration test
  `with_perceptual_target_score_actually_drives_loop` encodes the same
  fixture twice — once with `with_distance(1.0)` (default
  `perceptual_target_score: None`), once with
  `with_distance(5.0).with_perceptual_target_score(Some(1.2245))` (the
  table value at d=1.0). Asserts the second encode's bytes are within
  ±10% of the first's. (Loose tolerance because the table is per-corpus-
  median; tighter assertions belong in the corpus-wide regression sweep.)
- (i) CHANGELOG `[Unreleased] / Added`: "Phase 1 of butteraugli target
  symmetry: `LossyConfig::with_perceptual_target_score(Some(score))` now
  drives the butteraugli buttloop's effective convergence distance via
  `vardct/butteraugli_targets.rs` calibration table. Closes the implicit-
  identity gap left over from multi-metric Phase 0 (commit `23da77b1`)."

## §9. Out of scope (for this RFC)

- **NOT recalibrating butteraugli's own algorithm.** The Phase 1 table
  is a CALIBRATION layer between caller-facing scores and the buttloop's
  effective distance. The butteraugli algorithm itself (per-iter compare,
  diffmap math, K_TILE_NORM = 1.2) is untouched.
- **NOT changing the default `PerceptualMetric::Butteraugli` behaviour.**
  When `perceptual_target_score` is `None` (the default), Phase 1 is
  byte-identical to today's encoder. The identity arm at
  [`perceptual_loop.rs:2385`](../jxl-encoder/src/vardct/perceptual_loop.rs)
  stays — Phase 1 only adds a SECOND arm that fires when the caller
  explicitly opts in via `with_perceptual_target_score(Some(...))`.
- **NOT touching the cvvdp Phase 4 calibration table.** `cvvdp_targets.rs`
  ships as-is. Phase 1 only adds `butteraugli_targets.rs` alongside it.
- **NOT touching W44 cost-model gates.** The 587+ gates documented in
  `docs/LIBJXL_DIVERGENCES.md` are calibrated against butteraugli direction
  (smaller=better) and `accept_bound = K_BUTTERAUGLI_ACCEPT_FACTOR ×
  effective_metric_target_distance`. Phase 1 changes what
  `effective_metric_target_distance` resolves to ONLY when
  `perceptual_target_score` is set — the W44 gate evaluation is downstream
  of that and consumes the resulting `effective_metric_target_distance`
  with no awareness of why it has that value.
- **NOT touching the display-config backfill arc** (RFC
  [`RFC_DISPLAY_CONFIG_BACKFILL.md`](RFC_DISPLAY_CONFIG_BACKFILL.md), Phase
  1 + 2 shipped). butteraugli does NOT have a display-config dimension —
  it's a single-display metric. The Phase 1 table is 1-dimensional
  (`score → distance`); the cvvdp table is 2-dimensional (`distance ×
  display → score`). Phase 1 of THIS RFC sits orthogonal to the display-
  config backfill arc.
- **NOT auto-content-classifier dispatch.** Per
  [`RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`](RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md) §5
  the metric selection is caller-driven. Phase 1 of THIS RFC just
  symmetrizes the in-metric API; it doesn't change which metric fires.

## §10. EncoderStrategy::Libjxl invariant

**Mandatory and load-bearing.** `EncoderStrategy::Libjxl` strict cjxl-
parity requires byte-identical output regardless of any opt-in field.
Per [`api.rs:6834-6842`](../jxl-encoder/src/api.rs), the resolver:

```rust
pub(crate) fn resolve_perceptual_metric(&self) -> PerceptualMetric {
    if matches!(self.strategy, EncoderStrategy::Libjxl) {
        return PerceptualMetric::Butteraugli;
    }
    // ... feature-gate dispatch ...
}
```

forces butteraugli regardless of caller intent. Phase 1 must extend this:
**`with_perceptual_target_score(Some(...))` on a Libjxl-strategy config
MUST also be ignored**. Otherwise an caller setting a target_score under
Libjxl would change the buttloop's effective distance from the libjxl-
matching identity to a calibrated value, breaking the byte-lock.

Phase 1 design: add a resolver method

```rust
impl LossyConfig {
    pub(crate) fn resolve_perceptual_target_score(&self) -> Option<f32> {
        if matches!(self.strategy, EncoderStrategy::Libjxl) {
            return None;  // FORCED — strict parity
        }
        self.perceptual_target_score
    }
}
```

and consume it via `resolve_metric_selection` (the existing
`MetricSelection` builder at
[`api.rs:6891-6904`](../jxl-encoder/src/api.rs)). The `target_score:
self.perceptual_target_score` line becomes `target_score:
self.resolve_perceptual_target_score()`.

Test: extend `tests/strategy_libjxl_byte_lock.rs` (W44-194) to encode 4
fixtures × `with_perceptual_target_score`-{None, Some(0.5), Some(2.0)} =
12 byte-lock cells. All must be byte-identical to the
`with_perceptual_target_score = None` baseline.

## §11. Cross-references

- **Spec**: [`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §9 (NEW — appended in this chunk; codifies the symmetric target-score contract every metric must satisfy).
- **Architecture**: [`RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`](RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md) §1.2 (`with_perceptual_target_score` setter), §2 (per-metric target-score semantics).
- **Phased ship plan**: [`RFC_BUTTERAUGLI_FORK_PLAN.md`](RFC_BUTTERAUGLI_FORK_PLAN.md) (NEW — sketches Phase 1 brief at ≥ 100 LOC).
- **Pattern**: [`vardct/cvvdp_targets.rs`](../jxl-encoder/src/vardct/cvvdp_targets.rs) (the calibration-table shape Phase 1 mirrors).
- **Seed data**: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` (B-rows; mining methodology in §1.3 of this RFC).
- **Phase memos** for the cvvdp arc:
  - Phase 4 (`32581839`, `cvvdp_fork_phase4_loop_wiring_shipped_2026-05-24.md`) — the calibration table pattern.
  - Phase 8c (`0ab5a53d`, `cvvdp_fork_phase8c_diffmap_renorm_shipped_2026-05-25.md`) — diffmap renorm derivation methodology.
  - Phase 8f (`CVVDP_FORK_DECISION.md`) — verdict template.
  - Phase 8g (`689ba0df`, `cvvdp_fork_phase8g_block_constants_shipped_2026-05-25.md`) — per-metric K_TILE_NORM refit methodology.
- **Multi-metric API surface**: `multi_metric_api_phase0_shipped_2026-05-25.md` (the Phase 0 commit that added `with_perceptual_target_score`).
- **Diffmap-stats env hook**: [`vardct/perceptual_backend.rs:370-430`](../jxl-encoder/src/vardct/perceptual_backend.rs) (`maybe_dump_diffmap_stats` + `JXL_PHASE8B_DIFFMAP_DUMP`).
- **Buttloop convergence math**: [`vardct/perceptual_loop.rs:2450`](../jxl-encoder/src/vardct/perceptual_loop.rs) (`accept_bound = K_BUTTERAUGLI_ACCEPT_FACTOR * effective_metric_target_distance`).
