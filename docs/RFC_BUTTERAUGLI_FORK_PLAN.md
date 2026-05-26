# RFC: Butteraugli target-symmetry — phased ship plan

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-26
**Status**: SCOPING (companion to [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md))

This RFC sketches the phased ship plan for closing the butteraugli
target-symmetry gap surfaced by the cvvdp arc. Mirrors
[`RFC_ZENSIM_FORK_PLAN.md`](RFC_ZENSIM_FORK_PLAN.md) exactly in shape:
phase summary table, then per-phase briefs with pre-flight + file
ownership + acceptance gates + DO-NOT + chunk-N+1.

The full design rationale lives in
[`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md);
this RFC is the actionable phased schedule.

## §1. Phase summary

| Phase | Scope | Crate(s) touched | Effort | Status |
|---    |---    |---               |---     |---     |
| **1 (mandatory)** | `vardct/butteraugli_targets.rs` calibration table + buttloop dispatch wiring + Libjxl strict-parity short-circuit + 7 unit tests + 2 integration tests | jxl-encoder | 2-3 days | not started |
| 2 (optional) | API-shape rename audit in `butteraugli` crate (cosmetic alignment with cvvdp/zensim 7-method family). **Recommend SKIP** — value aesthetic, cost real. | butteraugli + butteraugli-gpu + jxl-encoder + zenmetrics validation suite | 1 day + downstream coord | not started |
| 3 (optional) | `DiffmapStats` introspection API on `ButteraugliReference` + `GpuButteraugliResult`. **Recommend DEFER** — env-gated dump (`JXL_PHASE8B_DIFFMAP_DUMP`) is sufficient until a 4th metric joins. | butteraugli + butteraugli-gpu | 1-2 days | not started |
| 4 (stretch) | Outer binary-search wrapper for `with_target_score_iterate(true)` exact-target convergence. **Recommend DEFER INDEFINITELY** — 4-6× wall cost not worth precision gain for ~all callers. | jxl-encoder | 1 week | not started |

**Total budget**: 2-3 days for Phase 1 only. Phase 2-4 are demand-driven.

The Phase 1 chunk could fire in parallel with the zensim Phase 4
calibration chunk (different files: `butteraugli_targets.rs` vs
`zensim_targets.rs`).

## §2. Phase 1 — butteraugli calibration table (MANDATORY)

**Sibling workspace**: `~/work/zen/jxl-encoder--butteraugli-targets-table`.
**Branch**: `main` (per CLAUDE.md "All work happens through jj on top of
a colocated git repo, directly on main").

### §2.1. Scope

Mirror [`RFC_CVVDP_PHASE4_BRIEF.md`](RFC_CVVDP_PHASE4_BRIEF.md) in shape;
ports the calibration-table pattern to butteraugli's score → distance
inverse direction. Specifically:

1. **`jxl-encoder/src/vardct/butteraugli_targets.rs`** (NEW, ~150 LOC):
   - `pub(crate) const BUTTERAUGLI_DISTANCE_BANDS: usize = 7;`
   - `pub(crate) struct ButteraugliCalibration { entries: [(f32, f32); 7] }`
     — pair `(butter_score_target, effective_distance)`. Direction
     INVERTED from `cvvdp_targets.rs` (cvvdp goes
     distance→score; butteraugli goes score→distance).
   - `pub(crate) const BUTTERAUGLI_SCORE_TARGETS: &[(f32, f32); 7]` seeded
     from `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` B-rows
     (see [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §1.3
     for the median values).
   - `pub(crate) fn butteraugli_effective_distance_for_target_score(target_score: f32) -> f32`
     — linear interp + endpoint clamps. Identical body shape to
     `cvvdp_target_score_for_distance_and_display`.
   - Identity-arm shim:
     `pub(crate) fn butteraugli_target_score_for_distance(d: f32) -> f32 { d.max(0.0) }`
     — preserves the pre-Phase-1 identity behaviour for callers that
     do NOT set `perceptual_target_score`.

2. **`jxl-encoder/src/vardct/mod.rs`** — register the new module.

3. **`jxl-encoder/src/vardct/perceptual_loop.rs`** — extend the dispatch
   at line 2350-2386 with a butteraugli arm that consults the new lookup
   when `perceptual_target_score` is set. The new arm fires AFTER the
   zensim/cvvdp branches (since both zensim_loop + cvvdp_loop are
   mutually exclusive with butteraugli being the active metric):
   ```rust
   // INSERTED before the trailing identity arm:
   if let Some(score_target) = self.perceptual_target_score {
       break 'lookup
           super::butteraugli_targets::butteraugli_effective_distance_for_target_score(
               score_target,
           );
   }
   target_distance  // identity arm — unchanged
   ```
   Phase 1 adds the field `perceptual_target_score: Option<f32>` to
   `VarDctEncoder` (it does not exist today — verified via grep) and
   threads it through `propagate_resolved_metric_to_encoder` from
   `LossyConfig::resolve_metric_selection`.

4. **`jxl-encoder/src/api.rs`**:
   - Add `resolve_perceptual_target_score(&self) -> Option<f32>` method
     with the EncoderStrategy::Libjxl strict-parity short-circuit (per
     [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §10).
   - Update `resolve_metric_selection` to consume
     `resolve_perceptual_target_score()` instead of
     `self.perceptual_target_score` directly.
   - Fix the misleading comment at
     [`api.rs:6643-6650`](../jxl-encoder/src/api.rs) — the doc currently
     says "[`perceptual_target_score`] overrides the metric's per-distance
     target table"; Phase 1 makes that doc accurate.

5. **`jxl-encoder/src/vardct/perceptual_backend.rs`**:
   - Remove the `let _ = selection.target_score;` no-op binding at
     [`perceptual_backend.rs:1392`](../jxl-encoder/src/vardct/perceptual_backend.rs) —
     it becomes dead code once Phase 1 wires the consumer site.
   - Update the doc comment on `MetricSelection.target_score` to say
     "consumed via `perceptual_target_score` field on `VarDctEncoder`,
     populated by `propagate_resolved_metric_to_encoder`."

6. **`jxl-encoder/src/vardct/perceptual_loop.rs:1792-1807`** —
   `MetricSelection` builder: change the hard-coded `target_score: None`
   to `target_score: self.perceptual_target_score`. Update the
   misleading comment that says "the buttloop body consumes it through
   the `effective_metric_target_distance` lookup below" to match the
   shipped wiring.

### §2.2. Pre-flight (inputs to read FIRST)

1. **`docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`** — the design rationale
   for this Phase 1.
2. **`docs/RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`** §9 (NEW; the
   appended symmetric-targeting contract that Phase 1 makes butteraugli
   satisfy).
3. **`docs/RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`** §1.2 and §2 — the
   `with_perceptual_target_score` API surface and the per-metric
   target-score dispatch shape.
4. **`jxl-encoder/src/vardct/cvvdp_targets.rs`** — the calibration-table
   pattern to mirror. Read in full; copy the docstring shape, copy the
   `zip_entries` helper, copy the `display_calibration` lookup shape
   (skipping the per-display dimension because butteraugli is
   single-display).
5. **`jxl-encoder/src/vardct/perceptual_loop.rs:2350-2386`** — the
   `'lookup` block to extend.
6. **`jxl-encoder/src/api.rs:4640-4660, 5095-5100, 6800-6817, 6885-6904`** —
   the existing `perceptual_target_score` field + setter + getter +
   `resolve_metric_selection` builder. Phase 1 needs the new
   `resolve_perceptual_target_score` resolver inserted into this chain.
7. **`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`** — the
   seed data. Mine `backend=B` rows for per-distance butter medians per
   [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §1.3.
8. **`jxl-encoder/tests/strategy_libjxl_byte_lock.rs`** — the W44-194
   byte-lock test to extend with Phase 1's `with_perceptual_target_score`
   variants.
9. **`~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase4_loop_wiring_shipped_2026-05-24.md`** — the cvvdp
   Phase 4 memo. Same shape; use it as the model for the Phase 1 commit
   message + memo.
10. **`scripts/cvvdp_pareto_analysis.py`** — the analysis pattern for
    deriving per-distance medians from a tracking TSV. Phase 1 can either
    bake the values into the const (RECOMMENDED — they're already in §1.3
    of the RFC) or commit a small `scripts/butteraugli_calibration_seed.py`
    that re-derives them on demand. Recommend baking + comment-linking
    to the TSV for provenance.

### §2.3. File ownership

**OWNED by the Phase 1 chunk** (write/modify):
- `jxl-encoder/src/vardct/butteraugli_targets.rs` (NEW)
- `jxl-encoder/src/vardct/mod.rs` (1-line module registration)
- `jxl-encoder/src/vardct/perceptual_loop.rs` (~5 LOC + comment fixes)
- `jxl-encoder/src/vardct/perceptual_backend.rs` (~3 LOC removal + comment fix)
- `jxl-encoder/src/api.rs` (~15 LOC new resolver method + 1-line consumer update + 1 docstring fix)
- `jxl-encoder/src/vardct/encoder.rs` (add `perceptual_target_score: Option<f32>` field + default + propagation arg — ~5 LOC)
- `jxl-encoder/tests/strategy_libjxl_byte_lock.rs` (extend test matrix)
- `jxl-encoder/tests/perceptual_target_score_drives_loop.rs` (NEW
  integration test) — see §2.4(h)
- `CHANGELOG.md` (Phase 1 [Unreleased] / Added entry)
- `docs/LIBJXL_DIVERGENCES.md` Section E (new row tracking the Phase 1
  resolver path)

**DO NOT TOUCH**:
- `butteraugli` or `butteraugli-gpu` crate source — Phase 1 is encoder-
  only. The metric algorithm + API are unchanged. (Phase 3 would touch
  them; Phase 1 does not.)
- `vardct/cvvdp_targets.rs` — Phase 1 leaves cvvdp's calibration
  table alone.
- `vardct/zensim_targets.rs` — same.
- Any W44 gate in `effort.rs` / `gate_registry.rs` /
  `vardct/encoder.rs` cost-model logic — Phase 1 only changes the
  `effective_metric_target_distance` resolution, NOT the downstream
  consumption.
- `EncoderStrategy::Libjxl` resolver — Phase 1 ADDS a new
  `resolve_perceptual_target_score` method WITH the strict-parity
  short-circuit; it does NOT modify the existing
  `resolve_perceptual_metric` short-circuit.

### §2.4. Acceptance gates

- (a) **Build PASS** on every feature combo: default, `butteraugli-loop`
  (default on), `gpu-butteraugli`, `cvvdp-loop`, `cvvdp-loop-cpu`,
  `cvvdp-loop-tighten`, `zensim-loop`, `zensim-loop-gpu`, and all
  combinations. Specifically:
  ```bash
  cargo build -p jxl-encoder
  cargo build -p jxl-encoder --features 'butteraugli-loop'
  cargo build -p jxl-encoder --features '__expert butteraugli-loop cvvdp-loop cvvdp-loop-cpu zensim-loop ssim2-loop parallel'
  ```
- (b) **`cargo test --lib`**: pre-Phase-1 count + 7 new butteraugli_targets
  unit tests + 2 new integration tests (see (g) + (h)) — all PASS.
- (c) **Hash-locks 36/36 BYTE-IDENTICAL** at default features.
  `cargo test --features '__expert butteraugli-loop parallel ssim2-loop'
  --test hash_lock_features` PASS. Default
  `perceptual_target_score: None` keeps the identity arm — zero byte
  movement on the default path.
- (d) **`strategy_libjxl_byte_lock` 4/4 BYTE-IDENTICAL** with full
  feature stack compiled AND with `with_perceptual_target_score(Some(0.5))`,
  `Some(1.0)`, `Some(2.0)` set. Test matrix expands from 4 fixtures to
  4 × 4 = 16 byte-locks (3 explicit scores + None baseline). All
  byte-identical thanks to the Phase 1 resolver short-circuit per
  [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §10.
- (e) **`divergence_table_drift` 7/7 PASS** — Phase 1 adds one row to
  Section E of `docs/LIBJXL_DIVERGENCES.md`.
- (f) **Multi-decoder roundtrip ≥ 5 cells × 3 decoders**: encode 5
  fixtures with `with_perceptual_target_score(Some(0.7))`, `Some(1.2)`,
  `Some(2.0)`, `Some(3.0)`, `Some(4.0)`; decode each with jxl-rs +
  djxl + jxl-oxide. All 15 must succeed. Test lives in
  `tests/butteraugli_target_score_decoder_roundtrip.rs` (new file).
- (g) **`butteraugli_targets_drift` test**: asserts table monotonicity
  in BOTH columns + endpoint clamps + interp midpoint matches average
  of endpoints to within 1e-6. Same shape as
  `cvvdp_targets::tests::rows_sorted_ascending_by_distance` +
  `cvvdp_target_score_per_display_is_monotone`. ~7 individual `#[test]`
  fns.
- (h) **`with_perceptual_target_score_actually_drives_loop` integration
  test**: encode the same fixture twice — once with `with_distance(1.0)`
  (default `perceptual_target_score: None`), once with
  `with_distance(5.0).with_perceptual_target_score(Some(1.2245))` (the
  Phase 1 table value at d=1.0). Assert the second encode's bytes are
  within ±10% of the first's. Loose tolerance because the table is per-
  corpus-median; tighter assertions belong in a follow-on regression
  sweep. (Test lives in `tests/perceptual_target_score_drives_loop.rs`.)
- (i) **CHANGELOG entry** under `[Unreleased] / Added`:
  ```
  ### Added
  - Phase 1 of butteraugli target symmetry:
    `LossyConfig::with_perceptual_target_score(Some(score))` now drives
    the butteraugli buttloop's effective convergence distance via the
    new `vardct/butteraugli_targets.rs` calibration table. Closes the
    implicit-identity gap left over from multi-metric Phase 0
    (commit `23da77b1`). Default path (`with_perceptual_target_score
    = None`) is byte-identical to pre-Phase-1. See
    `docs/RFC_BUTTERAUGLI_TARGET_SYMMETRY.md` Phase 1.
  ```
- (j) **One atomic commit** pushed to main via
  `jj git push --change @-`. Per CLAUDE.md "All work happens through jj
  on top of a colocated git repo, directly on main."
- (k) **Investigation Notes memo** at
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/butteraugli_targets_phase1_shipped_<date>.md`.
- (l) **`docs/HYPOTHESIS_LEDGER.md` updated** — add a belief entry
  recording the per-corpus-median seed methodology + the ±10% per-image
  variance found in §1.3 of the RFC.
- (m) **`.workongoing` marker cleaned up** on completion (`rm
  .workongoing`).
- (n) **Sibling workspace cleaned up** post-merge: `jj workspace
  forget butteraugli-targets-table` + `rm -rf
  ~/work/zen/jxl-encoder--butteraugli-targets-table`. Per CLAUDE.md
  cleanup-on-merge mandate.

### §2.5. Honest-stop discipline

The following are LEGITIMATE Phase 1 honest-stops (NOT failures):

1. **If the per-corpus-median table produces > ±15% per-image variance
   on the integration test fixtures**: the table may need per-content-class
   stratification. Honest-stop, document, ship the table as opt-in only
   with a known-limitation note in the docstring.
2. **If `with_perceptual_target_score(Some(0.5))` produces bytes < 10%
   different from `with_distance(0.5)` on the integration test**: the
   table calibration is too close to identity (the buttloop already
   over-shoots at low distances per
   [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §1.3
   shows median butter 0.72 at d=0.5). May need a tightness factor
   similar to cvvdp's 1.05× from
   [`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §1.3.
   Honest-stop, document the factor's empirical justification, ship.
3. **If the Libjxl byte-lock breaks under any `Some(score)` variant**:
   the Phase 1 resolver short-circuit isn't firing. Diagnose
   (likely `MetricSelection.target_score` consumer path), fix, do NOT
   ship until 16/16 byte-locks pass.

The following are NOT honest-stops (must be fixed to ship):

- Hash-lock drift on default features (gate (c)).
- `EncoderStrategy::Libjxl` byte-lock failure on `with_perceptual_target_score`
  variants (gate (d)).
- Multi-decoder roundtrip failure (gate (f)).
- Phase 1 needs to TOUCH butteraugli crate source (it should not; if
  it does, the design is wrong).

### §2.6. DO NOT

1. **DO NOT make the table per-display.** Butteraugli is a single-display
   metric. The display-config backfill arc shipped cvvdp's per-display
   tables but butteraugli stays 1-dimensional. Adding a `DisplayConfig`
   dimension would be a structural change to butteraugli itself —
   out of scope.
2. **DO NOT add a 1.05× tightening factor without measurement.** The
   cvvdp arc tried this in Phase 4 and the Phase 6 verdict showed it
   overshot (`CVVDP_FORK_DECISION.md`). If Phase 1's integration test
   shows the table needs adjustment, honest-stop and surface the
   empirical justification — do not hand-tune.
3. **DO NOT remove the identity arm at
   [`perceptual_loop.rs:2385`](../jxl-encoder/src/vardct/perceptual_loop.rs).**
   The arm is the default behaviour when `perceptual_target_score = None`.
   Phase 1 ADDS a SECOND arm that fires when the caller opts in; it does
   NOT replace the existing arm.
4. **DO NOT cite "FMA precision"** for any byte movement. Per W44-66
   user correction in CLAUDE.md, every byte movement has a structural
   cause; "FMA precision" is never a valid root cause.
5. **DO NOT ship Phase 1 without the Libjxl strict-parity short-circuit
   on `perceptual_target_score`.** A caller setting a target_score under
   `EncoderStrategy::Libjxl` MUST get butteraugli-identity behaviour
   (target_score is silently dropped). The W44-126 invariant requires
   it; the byte-lock test at (d) enforces it.
6. **DO NOT default-flip `LossyConfig::perceptual_target_score` to
   `Some(<value>)`.** The default MUST stay `None` (identity arm) to
   preserve hash-lock byte-identity on the default path.
7. **DO NOT touch the W44-126 / W44-194 byte-lock fixtures.** If they
   break under Phase 1, the implementation is wrong — fix the
   implementation, not the fixtures.
8. **DO NOT propose Phase 2 + Phase 3 + Phase 4 as part of the Phase 1
   chunk.** Phase 1 stands alone; subsequent phases are demand-driven
   and may never fire.
9. **DO NOT mine seed data from a different sweep TSV.** The
   `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` `backend=B`
   rows are the canonical reference (1,134 cells across CID22 + GB82-SC
   + W44-PHASE4-S1 extras). Drift from this seed without re-measuring
   would invalidate the calibration provenance.

### §2.7. Chunk N+1 plan

Phase 1 stands alone. After it lands:

- **If Phase 2 is queued**: spawn a separate chunk at
  `~/work/zen/butteraugli--rename-7-method-family`. See §3 below for
  the Phase 2 brief sketch. **Recommend SKIP** unless a downstream
  caller surfaces specific pain.
- **If Phase 3 is queued**: spawn a separate chunk at
  `~/work/zen/butteraugli--diffmap-stats-api`. See §4 below. Recommend
  DEFER until a 4th metric integration surfaces demand.
- **If Phase 4 is queued**: spawn at
  `~/work/zen/jxl-encoder--butteraugli-target-iterate-wrapper`. See §5
  below. Recommend DEFER INDEFINITELY.

Phase 1 does NOT block any other in-flight work; the cvvdp + zensim
arcs can proceed independently. Phase 1 fixes a no-op on the default
metric; it doesn't change cvvdp/zensim behaviour.

---

## §3. Phase 2 brief (OPTIONAL — recommend SKIP)

**Sibling workspace**: `~/work/zen/butteraugli--rename-7-method-family`.
**Branch**: `butteraugli@main` (butteraugli crate; coordinate with
zenmetrics + jxl-encoder downstream).

### §3.1. Scope

Cosmetic API alignment in the `butteraugli` crate to match the
cvvdp/zensim 7-method family naming:

| Old name                                     | Proposed new name                                |
|---                                           |---                                               |
| `ButteraugliReference::new_linear_planar`    | `ButteraugliReference::warm_reference_from_linear_planes` |
| `ButteraugliReference::compare_linear_planar` | `ButteraugliReference::score_from_linear_planes_with_warm_ref` |
| `ButteraugliReference::compare_linear_planar_into` | `ButteraugliReference::score_from_linear_planes_with_warm_ref_diffmap` |
| `butteraugli_linear` (free fn) | `butteraugli::score_from_linear_planes` (or move to associated fn) |

### §3.2. Why we recommend SKIP

Per [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §5:

1. The wrapping layer at
   [`vardct/perceptual_backend.rs:445-1224`](../jxl-encoder/src/vardct/perceptual_backend.rs)
   already absorbs the difference cleanly.
2. butteraugli's existing names ARE descriptive in its domain. The
   `Reference` struct already encodes "warmness"; the method names
   don't repeat it.
3. The rename forces a breaking change on butteraugli + butteraugli-gpu
   + jxl-encoder + zenmetrics validation suite. ~10 method renames + ~20
   docstrings + tests + CHANGELOG. Real cost, aesthetic benefit.

### §3.3. If Phase 2 ever fires

Pre-flight + acceptance gates would mirror Phase 1 in shape:

- **Inputs**: butteraugli crate full source; cvvdp_backend.rs +
  zensim_backend.rs in jxl-encoder for the naming pattern to mirror.
- **Acceptance**:
  - `butteraugli` crate bumps to 0.2.0 (breaking).
  - Old names deleted (no deprecation shim per CLAUDE.md).
  - jxl-encoder + zenmetrics + any other downstream caller updates
    in the same PR.
  - Hash-locks 36/36 BYTE-IDENTICAL (rename is name-only; no
    behavioural change).
  - CHANGELOG breaking-change entry under
    `QUEUED BREAKING CHANGES` if other 0.2.0 changes haven't shipped
    yet.

### §3.4. DO NOT (Phase 2)

- DO NOT add `#[deprecated]` shims (CLAUDE.md "no backwards-compat
  hacks").
- DO NOT touch butteraugli's algorithm — Phase 2 is pure rename.
- DO NOT mix Phase 2 with Phase 3 — they touch overlapping files but
  for different reasons; ship separately for diff-review clarity.

---

## §4. Phase 3 brief (OPTIONAL — recommend DEFER)

**Sibling workspace**: `~/work/zen/butteraugli--diffmap-stats-api`.
**Branch**: `butteraugli@main`.

### §4.1. Scope

Add programmatic introspection of the per-pixel diffmap distribution.
Replaces the ad-hoc `JXL_PHASE8B_DIFFMAP_DUMP` env-gated TSV with a
stable API:

```rust
// butteraugli/src/lib.rs (NEW)
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
    pub fn bad_rate_at(&self, threshold: f32) -> f32 { ... }
}

impl ButteraugliReference {
    pub fn diffmap_stats(&self) -> Option<DiffmapStats> { ... }
}

// butteraugli-gpu/src/pipeline.rs (NEW)
impl GpuButteraugliResult {
    pub fn diffmap_stats(&self) -> Option<DiffmapStats> { ... }
}
```

The `Option` is `None` when:
- The reference was just constructed (no compare yet).
- The params were NOT configured with `with_compute_diffmap(true)`.

### §4.2. Why we recommend DEFER

Per [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §6:

1. The env-gated dump shipped successfully through Phase 8c + 8g without
   API friction. No current demand for programmatic access.
2. Phase 3 only adds value when a 4th metric joins or a long-running
   harness needs distribution stats without env-var management. Neither
   is imminent.
3. Adding the API to butteraugli + butteraugli-gpu is mechanical but
   binds the crates to a new public surface that has to be maintained
   forward.

### §4.3. If Phase 3 ever fires

Pre-flight + acceptance gates:

- **Inputs**:
  - [`vardct/perceptual_backend.rs:370-430`](../jxl-encoder/src/vardct/perceptual_backend.rs) —
    the env-gated TSV implementation to lift into the crates.
  - [`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §4.3 —
    the canonical schema (mean / median / p25 / p75 / p95 / max).
- **Acceptance**:
  - `DiffmapStats` shipped on CPU + GPU.
  - Hash-locks 36/36 BYTE-IDENTICAL (read-only API).
  - Unit tests assert stats consistency across cold + warm paths
    (same diffmap → same stats).
  - `cvvdp_backend.rs` + `zensim_backend.rs` updated to use the new
    API (NOT mandatory; the env-gated dump can stay as a parallel
    facility).
  - `jxl-encoder/examples/diffmap_distribution_audit.rs` (NEW) — bench
    that uses the API to produce a TSV identical in shape to the
    env-gated dump.

### §4.4. DO NOT (Phase 3)

- DO NOT replace the env-gated dump until the programmatic API has
  shipped + the cvvdp_backend / butteraugli_backend / zensim_backend
  consumers have migrated. The env-gated dump is the existing
  calibration tool; removing it before the replacement is in place
  would block future calibration runs.
- DO NOT add `DiffmapStats` to cvvdp-gpu or zensim-gpu in Phase 3. It's
  a Phase 3 of THIS RFC (butteraugli target-symmetry); the other crates
  get their own phased rollout as separate chunks if/when demand
  surfaces.

---

## §5. Phase 4 brief (STRETCH — recommend DEFER INDEFINITELY)

**Sibling workspace**: `~/work/zen/jxl-encoder--butteraugli-target-iterate-wrapper`.
**Branch**: `main`.

### §5.1. Scope

Add `LossyConfig::with_target_score_iterate(bool)` opt-in for exact
target-score convergence via outer binary search:

1. Trial-encode at the Phase 1 lookup's `effective_distance`.
2. Decode + measure actual butter via in-process butteraugli.
3. Adjust `effective_distance` by golden-section search.
4. Repeat until achieved butter is within ε of `target_score` or 5
   iterations (whichever first).
5. Re-encode with the converged distance.

### §5.2. Why we recommend DEFER INDEFINITELY

Per [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §4.2:

1. **4-6× wall cost**: each iteration is a full encode + decode + metric
   pass at ~1024² ≈ 500 ms × 5 = 2.5 seconds per encode. Unacceptable
   in any production pipeline.
2. **The Phase 1 static table is "good enough" for ~all callers.** Per-
   image variance of ±30-50% from the median target is meaningful but
   not pathological. A caller wanting tighter convergence would more
   likely just lower `target_distance` directly.
3. **No cvvdp / zensim precedent.** Neither metric arc shipped a
   target-iterate wrapper. Adding it for butteraugli (the metric that
   needs it LEAST because its native unit IS distance) would be
   structurally asymmetric.

### §5.3. If Phase 4 ever fires

Pre-flight + acceptance gates:

- **Cargo feature**: new `butteraugli-target-iterate` (default OFF). Keeps
  the in-process butteraugli decode-and-score dependency off the
  default build.
- **API**: `LossyConfig::with_target_score_iterate(true)` opt-in
  alongside `with_perceptual_target_score(Some(...))` (both must be
  set for the wrapper to fire).
- **Acceptance**:
  - `cargo build` PASS without the new feature.
  - `cargo build --features butteraugli-target-iterate` PASS.
  - Hash-locks 36/36 BYTE-IDENTICAL at default features.
  - Integration test: `with_perceptual_target_score(Some(1.2)).with_target_score_iterate(true)`
    on 5 fixtures produces output within ±0.05 of the target butter
    (verified via in-process score).
  - Wall-time bench shows the expected 4-6× cost.

### §5.4. DO NOT (Phase 4)

- DO NOT default-on `target_score_iterate`. It's an opt-in for callers
  who can spend 2-3 seconds per encode for ±0.05 precision.
- DO NOT remove the Phase 1 lookup. Phase 4 USES the Phase 1 table as
  its initial guess for the binary search.
- DO NOT add Phase 4 to cvvdp or zensim. Phase 4 is butteraugli-specific
  because butteraugli's identity arm makes the iterate-wrapper most
  natural for it; cvvdp/zensim would need their own metric-specific
  decode-and-score paths that don't exist today.

---

## §6. Cross-phase invariants

These hold across every phase. Re-stated for the avoidance of doubt
(same shape as `RFC_ZENSIM_FORK_PLAN.md` §11):

1. **Hash-locks 36/36 BYTE-IDENTICAL at default features** — every phase
   verifies this in its CI gate. butteraugli opt-in cannot regress the
   default path.
2. **`strategy_libjxl_byte_lock` PASS with all feature combos** — Libjxl
   strict-parity is mandatory. Phase 1 explicitly extends the test
   matrix to cover `with_perceptual_target_score` variants.
3. **Multi-decoder roundtrip ≥ 5 cells × 3 decoders per phase** — every
   phase that touches encoder behaviour must validate against jxl-rs +
   djxl + jxl-oxide.
4. **`divergence_table_drift` PASS** — every gate added to the encoder
   must have a Section E row in `docs/LIBJXL_DIVERGENCES.md`.
5. **`docs/HYPOTHESIS_LEDGER.md` updated when each phase ships** — per
   CLAUDE.md research methodology rule #6.
6. **NO "FMA precision" citations** — per W44-66 user correction (CLAUDE.md).
   Every byte movement has a structural cause.
7. **Sibling-workspace discipline** — every chunk spawns in
   `~/work/zen/jxl-encoder--<chunk-name>/`, cleans up on merge.
8. **Atomic commit + push** — every phase lands as ONE commit pushed
   to main. No half-shipped state.

## §7. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---  |---         |---     |---         |
| Phase 1 calibration table's per-image variance breaks integration test (h) | Low | Medium | Loose ±10% tolerance per RFC §8; honest-stop documented in §2.5 |
| Phase 1 resolver doesn't fire under `EncoderStrategy::Libjxl` correctly | Low | High (byte-lock breaks) | Extend byte-lock test matrix to 16 cells; gate (d) is a hard CI gate |
| Phase 1 introduces hash-lock drift on default features | Low | High | `perceptual_target_score: None` default keeps identity arm; gate (c) catches any regression |
| Phase 2 cosmetic rename cascades through downstream crates without coord | Medium | Medium | Skip Phase 2 by default; if it fires, coordinate butteraugli + butteraugli-gpu + jxl-encoder + zenmetrics in one PR |
| Phase 3 `DiffmapStats` API surface needs revision after first use | Medium | Low (API is small) | Defer until a 4th metric integration surfaces real demand |
| Phase 4 binary-search wrapper diverges from butteraugli's deterministic output | Medium | High (reproducibility break) | Defer indefinitely; if shipped, gate behind a feature + opt-in flag |
| A caller setting `with_perceptual_target_score(Some(NaN))` makes the table return NaN | Low | Medium | Phase 1 input validation: `target_score.is_finite() && target_score > 0.0` else fall back to identity arm |

## §8. Out of scope (for this RFC)

- Multi-metric content-class auto-dispatch — per
  [`RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`](RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md)
  §5 the metric selection is caller-driven.
- Replacing the per-block 16th-power-norm reducer with a metric-shaped
  one. Stays butteraugli-calibrated; Phase 1 only changes the per-
  distance target lookup, not the reducer math.
- Per-image content-class butteraugli calibration. Phase 1 ships a
  corpus-median table; per-class stratification is a future research
  question.
- HDR-aware butteraugli targeting. butteraugli is SDR-only today; HDR
  is a separate workstream.
- Reverting any W44 cost-model gates. Orthogonal; calibrated against
  butter, stays butter-calibrated.
- Animation / streaming buttloop integration. Phase 1 is still-image
  only (matching cvvdp Phase 4 + zensim Phase 4 precedent).
- Replacing SSIMULACRA2 in non-buttloop contexts (legacy tuning sweeps
  stay on SSIM2).
- Adding `DiffmapStats` to cvvdp-gpu / zensim-gpu — Phase 3 of THIS
  RFC ships butteraugli's; cross-metric extensions are separate
  workstreams.

## §9. Cross-references

- **Design**: [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) (the
  rationale + 3 implementation options + recommended pick).
- **Contract**: [`RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md`](RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md) §9
  (NEW — symmetric target-score handling, the baseline every metric
  must satisfy).
- **Architecture**: [`RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md`](RFC_MULTI_METRIC_PERCEPTUAL_BACKEND.md) §1.2
  + §2 (the `with_perceptual_target_score` API surface + per-metric
  dispatch).
- **Pattern to mirror**: [`vardct/cvvdp_targets.rs`](../jxl-encoder/src/vardct/cvvdp_targets.rs)
  (the calibration-table shape).
- **Phase 4 cvvdp memo**: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase4_loop_wiring_shipped_2026-05-24.md` (the cvvdp
  Phase 4 commit shape to mirror for Phase 1 of THIS RFC).
- **Multi-metric Phase 0 memo**: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/multi_metric_api_phase0_shipped_2026-05-25.md` (the
  commit that added `with_perceptual_target_score` as a no-op).
- **Phase 6 tracking TSV**: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
  (the seed data for Phase 1's calibration table; B-rows give the
  butter per-distance medians per
  [`RFC_BUTTERAUGLI_TARGET_SYMMETRY.md`](RFC_BUTTERAUGLI_TARGET_SYMMETRY.md) §1.3).
- **Libjxl invariant**: [`tests/strategy_libjxl_byte_lock.rs`](../jxl-encoder/tests/strategy_libjxl_byte_lock.rs)
  (W44-194; Phase 1 extends the test matrix).
