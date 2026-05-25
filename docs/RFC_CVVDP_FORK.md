# RFC: cvvdp fork — replace butteraugli buttloop with ColorVideoVDP

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-24
**Status**: SHIPPED-OPT-IN (Phase 6 verdict 2026-05-24, see [`CVVDP_FORK_DECISION.md`](CVVDP_FORK_DECISION.md))

The jxl-encoder buttloop iteratively refines the quantization field by
calling butteraugli once per iter to get (a) a global perceptual score
that drives termination and (b) a per-pixel diffmap that drives the
per-block 8×8 median/MAD adjustment to qac. This RFC plans a fork that
swaps butteraugli for **ColorVideoVDP** (cvvdp) — a more recent
perceptual metric from Mantiuk et al. (Cambridge) — while preserving
the buttloop's iterative shape.

The user mandate: "fork this to use cvvdp instead of butteraugli,
cvvdp-gpu for now, but have an agent make a maximally optimized cpu
port of cvvdp and add diffmap support to replace buttloop. eval with
cvvdp instead of butter. track improvements." (2026-05-24).

## §1. Why cvvdp

- More recent perceptual model (Mantiuk et al., 2024); explicit DKL
  colour space, multi-band Laplacian pyramid with per-band CSF, and
  cross-channel masking. Butteraugli's pipeline is from 2017.
- Already has a GPU implementation in this repo at
  `~/work/zen/zenmetrics/crates/cvvdp-gpu/` (~10K LOC, parity-tested
  against pycvvdp v0.5.4).
- Better correlation with subjective MOS on recent codec comparison
  data (we'll verify with the tracking benchmark; this RFC doesn't
  presume the conclusion).

## §2. Architecture

### §2.1. Trait abstraction

The W44-PHASE3-B1 work introduced `ButteraugliBackend` (a trait with
`set_reference` + `compare_with_reference(...) -> (score, diffmap)`).
Rename → **`PerceptualBackend`**. The shape is correct for cvvdp:

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
    // ...
}
```

Implementors:

| backend                  | feature flag           | status        |
|---                       |---                     |---            |
| `CpuButteraugliBackend`  | `butteraugli-loop` (existing) | shipped W44-PHASE3-B1 |
| `GpuButteraugliBackend`  | `gpu-butteraugli`      | shipped W44-PHASE3-B1+B4 |
| `GpuCvvdpBackend`        | `cvvdp-loop` (NEW)     | this RFC      |
| `CpuCvvdpBackend`        | `cvvdp-loop` (NEW)     | this RFC, agent-owned |

`BackendCompareResult::score` semantics stay the same number-line: smaller
= better. CVVDP outputs JOD (10 = perfect, lower = worse). We invert sign
or use `10.0 - jod` as the score so the buttloop's `score < target` early-
termination logic is unchanged.

### §2.2. Feature gate + default

`--features cvvdp-loop` enables the cvvdp backend. Default remains
butteraugli (the 6+ months of W44 calibration is built around it; we
won't lose it). `LossyConfig::with_cvvdp_loop(bool)` selects the
metric at encode time when the feature is compiled in.

`EncoderStrategy::Libjxl` ALWAYS uses butteraugli (strict cjxl-parity
invariant). The W44-228 byte-locks stay byte-identical regardless of
the cvvdp feature compile state — same defense-in-depth pattern as
W44-PHASE3-B5-flip.

### §2.3. Score mapping (JOD ↔ buttloop semantics)

CVVDP's JOD scale: 10 = no perceptual difference, decreasing as
difference grows. Buttloop's `target_distance` is a butteraugli max-
norm threshold (smaller = stricter; ~1.0 for d=1.0 encodes,
exponentially higher at lower distances).

Mapping for the buttloop:

- Internal `score` = `(10.0 - jod).clamp(0.0, 10.0)` (so 0 = perfect,
  higher = worse — matches butteraugli direction).
- Per-distance target via a calibration table (`vardct/cvvdp_targets.rs`):
  starts as the JOD-equivalent of each distance level's butteraugli target.
  The tracking benchmark (§5) is what calibrates this table.

We do NOT preserve butteraugli's exact numeric meaning — that would
defeat the point of switching metrics. We DO preserve the buttloop's
iterative shape: refine qac, score, compare to target, refine again.

### §2.4. Per-block 8×8 distance from diffmap

The buttloop's `compute_per_block_distance` reduces the per-pixel
diffmap to 8×8-block median+MAD weights. This logic is **metric-
agnostic** — it just needs a per-pixel signal where larger = worse.
The cvvdp diffmap (§3) satisfies that contract.

## §3. Diffmap recipe (algorithmic contract)

CVVDP natively emits a scalar JOD. We extend it to also emit a per-
pixel signal whose Minkowski-p norm gives the JOD. Recipe (binding
for BOTH the cvvdp-gpu extension and the cvvdp-cpu port):

1. Pipeline runs as usual through masking. After masking, we have
   per-pixel masked-error per (band, DKL channel). Bands have
   different spatial resolutions (the Laplacian pyramid).
2. **Upsample each band to base resolution** via bilinear
   interpolation. (Match scipy `zoom(order=1)` / OpenCV
   `INTER_LINEAR` semantics.)
3. **Sum across bands** → one per-pixel f32 per DKL channel (3 planes:
   R-G, B-Y, L).
4. **Pool across DKL channels** using the SAME Minkowski-p exponent
   and channel weights from `CvvdpParams`. Output a single `W×H`
   row-major f32 plane.

### Invariants (PRACTICAL — Agent A 2026-05-24 honest-stop)

The original RFC drafted a strict
`JOD == 10.0 - minkowski_p_norm(diffmap, p = params.beta_sch)`
invariant. **Agent A's cvvdp-cpu port measured that this strict form
isn't achievable without restructuring the cvvdp pool order**
(intermediate per-band/per-channel Minkowski folds don't cleanly
factor back into a single base-res per-pixel pool — the canonical
cvvdp algorithm pools per-channel then per-band, not the other way
round). Per "honest-stop > false completion": ship the looser-but-
honest invariants and document the strict form as aspirational.

**Five PRACTICAL invariants** (every impl MUST hold):

1. **Identity → zero**: `diffmap(ref, ref) == 0` to 1e-7 absolute on
   every pixel.
2. **Non-negative**: every diffmap pixel is ≥ 0.
3. **Monotone in distortion**: for two distorted images D1, D2 where
   D1 differs from ref strictly more than D2 (in any normed sense),
   `mean(diffmap_D1) ≥ mean(diffmap_D2)`.
4. **Spatial localization**: a pixel-block perturbation in input
   produces a diffmap response concentrated near that block (i.e.
   the diffmap isn't a global average — the buttloop's per-block
   median/MAD heuristic depends on this).
5. **Warm-ref invariance**: `score_with_diffmap(ref, dist)` and
   `warm_reference(ref); score_with_warm_ref_diffmap(dist)` produce
   the same diffmap byte-for-byte (or within 1e-7 if the warm path
   takes a different reduction order).

These are TESTABLE in both cvvdp-gpu and cvvdp-cpu — match Agent A's
6 invariant tests in `cvvdp-cpu/tests/diffmap_invariants.rs`.

### Strict invariant (aspirational, deferred)

The original `JOD == 10 - minkowski_p_norm(diffmap)` invariant is a
strong design property — if it holds, callers can reason about diffmap
norms == metric scores. Restructuring the cvvdp pool order to
materialize this is a multi-week chunk; deferred until/unless the
buttloop integration shows a need.

**Reference for upstream**: pycvvdp v0.5.4 doesn't expose a per-pixel
diffmap API; this recipe is our extension. Document any further
divergence from upstream in
`zenmetrics/crates/cvvdp-gpu/docs/DIFFMAP_DIVERGENCES.md` (or the
cvvdp-cpu equivalent).

## §4. Phased ship plan

### Phase 1 — diffmap API in cvvdp-gpu

Sibling: `~/work/zen/zenmetrics--cvvdp-diffmap/`.

- Extend `Cvvdp<R>` (and `CvvdpOpaque`) with `score_with_diffmap` and
  `score_with_warm_ref_diffmap` methods.
- Add `from_linear_planes` family (avoids host sRGB pack overhead —
  same shape as butteraugli-gpu's W44-PHASE3-B4 work).
- New CubeCL kernel(s) for the bilinear band-upsample + sum step.
- Unit tests: identity → zero diffmap, scalar/diffmap invariant
  (§3), one parity test against pycvvdp `get_diff_map=True`.
- Acceptance: cargo test PASS on all backend features (cuda + cpu +
  cubecl-cpu); golden parity for the JOD scalar unchanged; new
  diffmap tests pass.

### Phase 2 — `PerceptualBackend` trait extraction

Sibling: `~/work/zen/jxl-encoder--cvvdp-fork/`.

- Rename `ButteraugliBackend` → `PerceptualBackend` in
  `vardct/butteraugli_backend.rs`. File can be renamed
  `vardct/perceptual_backend.rs`; keep a `pub(crate) use` shim so
  existing imports work for one commit, then delete the shim.
- Existing tests stay byte-identical (the trait rename has zero
  runtime effect).
- Acceptance: `cargo test` PASS; `hash_lock_features` 36/36
  byte-identical; `strategy_libjxl_byte_lock` 4/4 PASS.

### Phase 3 — `CvvdpBackend` impl

- New file `vardct/cvvdp_backend.rs` mirroring `butteraugli_backend.rs`.
- Wraps `cvvdp_gpu::CvvdpOpaque` (default) — uses
  `score_from_linear_planes_with_diffmap` from Phase 1.
- Feature-gated `cvvdp-loop` (and `cvvdp-loop-cpu` for the CPU port
  variant when ready — Phase 5).
- Defense-in-depth: cubecl init wrapped in `catch_unwind`; on failure
  emit one `eprintln!` and fall back to butteraugli (analogous to
  the GPU butteraugli backend's behaviour).

### Phase 4 — buttloop wiring

- Rename `vardct/butteraugli_loop.rs` → `vardct/perceptual_loop.rs`
  (one-commit shim; symbol-rename is mechanical).
- `LossyConfig::with_cvvdp_loop(bool)` field, plumbed through all 3
  encode entry sites (still / streaming / animation).
- `EffortProfile` / `EncoderStrategy` plumbing: cvvdp opt-in stays
  OFF for `Libjxl`; configurable on `Zenjxl` / `LeanFaster` /
  `Aggressive` / `Custom`.
- Calibration table `vardct/cvvdp_targets.rs` seeded from a quick
  10-cell sweep; refined by §5.
- Acceptance: with `cvvdp-loop` OFF (default) every existing test
  byte-identical; with `cvvdp-loop` ON, encodes complete on a 20-cell
  smoke without panic, decode round-trips via jxl-rs + djxl.

### Phase 5 — cvvdp-cpu integration (when agent ships)

- `CpuCvvdpBackend` wrapping the cvvdp-cpu crate (agent's deliverable).
- Default cvvdp backend switches from GPU to CPU when
  `cvvdp-loop-cpu` feature is on (mirrors the butteraugli GPU/CPU
  split).
- Wall-time regression budget vs butteraugli CPU buttloop at e8 d=1.0
  on the W44-228 ledger: ≤2× without further optimisation; ≤1.5× as
  the stretch target. If we miss, the GPU backend stays the default
  for the cvvdp path.

### Phase 6 — tracking benchmark + decision

See §5. Goal: a single TSV + meta giving the user the data to decide
whether to (a) flip default to cvvdp-loop, (b) keep butteraugli
default but ship cvvdp-loop as an opt-in flag, or (c) revert.

## §5. Tracking benchmark methodology

File: `benchmarks/cvvdp_vs_buttloop_tracking_<YYYY-MM-DD>.tsv`. Updated
continuously as the fork matures; each session appends rows. Meta
file documents commit, hostname, methodology.

### §5.1. Corpus

- **CID22-512 validation** (41 photos held out from training).
- **GB82-SC** (11 screenshots).
- **W44-PHASE4-S1 extras** (2 lossless additions:
  baby-lossless, bulb-lossless — the rest of S1's 27-image pick
  overlaps with CID22 + GB82-SC).

**Actual: 54 distinct images** (per Agent D's measurement 2026-05-24;
the original RFC's 101-image / 2,121-cell figure was aspirational
based on a larger intended W44-S1 sample). For each, sweep ⌊distance
∈ {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}⌋ × effort ∈ {5, 7, 8} = 21 cells
per image, **1,134 cells per backend**.

### §5.2. Backends to compare

| cell tag    | encoder backend     | feature flag           |
|---          |---                  |---                     |
| `B`         | butteraugli (CPU)   | (default)              |
| `B_GPU`     | butteraugli (GPU)   | `gpu-butteraugli`      |
| `C_GPU`     | cvvdp (GPU)         | `cvvdp-loop`           |
| `C_CPU`     | cvvdp (CPU port)    | `cvvdp-loop`           |

### §5.3. Score every output with every metric

For each (image, distance, effort, backend) cell, compute:

- `score_butter_cpu` — Rust butteraugli on linear sRGB
- `score_butter_gpu` — butteraugli-gpu (parity check)
- `score_cvvdp_gpu` — cvvdp-gpu JOD
- `score_cvvdp_cpu` — cvvdp-cpu JOD (parity check vs GPU)
- `score_ssim2` — SSIMULACRA2 (legacy reference)

### §5.4. Direction-of-improvement table

For each (image, distance, effort) we report:

- **Wins-vs-loses table**: of all cells where one backend produces
  smaller bytes AND every metric is within ±0.5 SSIM2 of the other,
  count the wins per backend.
- **Pareto frontier**: bytes vs each metric. The backend that pushes
  out the frontier wins.
- **Wall-time delta**: ms/MP per backend, mean + p95.

Decision rule (preserves the CLAUDE.md anti-shipping-on-one-metric
discipline):

- Ship cvvdp default-ON only if it strictly Pareto-dominates
  butteraugli on at least one corpus + comes within 5% on the others.
- Otherwise: ship as opt-in flag; document trade-offs in CHANGELOG.

## §6. Acceptance gates (cumulative across phases)

- (a) `cargo build` PASS on every feature combo: default, `cvvdp-loop`,
  `cvvdp-loop gpu-butteraugli`, `cvvdp-loop butteraugli-loop`,
  `cvvdp-loop cvvdp-loop-cpu` (when Phase 5 lands).
- (b) `cargo test --lib` PASS at every phase.
- (c) Hash-locks 36/36 byte-identical with `--features default`
  (cvvdp opt-in must not regress the default path).
- (d) `strategy_libjxl_byte_lock` 4/4 PASS — Libjxl strategy must
  ALWAYS use butteraugli regardless of feature compile state.
- (e) `divergence_table_drift` PASS — every new gate / divergence
  documented in `docs/LIBJXL_DIVERGENCES.md`.
- (f) Multi-decoder roundtrip (djxl + jxl-rs + jxl-oxide) on ≥2 cells
  per phase.
- (g) Tracking benchmark TSV + meta committed for each phase landing.
- (h) `docs/HYPOTHESIS_LEDGER.md` updated when each phase ships
  (per CLAUDE.md research methodology rule #6).

## §7. Risks + mitigations

- **JOD ↔ buttloop target calibration drift**: the per-distance JOD
  threshold table is empirical and may shift after corpus expansion.
  Mitigation: keep `vardct/cvvdp_targets.rs` data-driven (regen from
  the tracking benchmark), don't hard-code.
- **CPU port misses perf target**: Phase 5 may ship with the GPU
  backend as the only viable cvvdp path. Acceptable — butteraugli CPU
  stays the default, cvvdp-gpu the opt-in.
- **Diffmap recipe diverges from pycvvdp**: §3 is THIS RFC's contract,
  not pycvvdp's. If upstream gets a `get_diff_map=True` option that
  differs, document it in a divergence note.
- **W44 calibration loss**: the W44-91/96/98/99/107/167/172 etc.
  butteraugli-specific cost-model lifts will not transfer one-for-one
  to cvvdp. Either keep them gated on `!cvvdp-loop` OR re-calibrate
  per-cell on the cvvdp tracking corpus. Decision deferred to §5.

## §8. Out of scope (for this RFC)

- Replacing SSIMULACRA2 in any non-buttloop contexts. SSIM2 stays the
  decision metric for tuning sweeps until §5 says otherwise.
- Animation / video buttloop (cvvdp-gpu is still-image only; video
  axis is "intentionally out of scope for v0").
- Reverting any W44 cost-model gates. Those are orthogonal to the
  metric.

## §9. Status notes

- 2026-05-24: RFC drafted. Sibling workspaces claimed
  (`jxl-encoder--cvvdp-fork`, `zenmetrics--cvvdp-diffmap`,
  `zenmetrics--cvvdp-cpu-port`). CPU port agent spawned (background,
  agent id `a1a609e3ce6facf30`). Tracking benchmark skeleton TSV
  created.
- 2026-05-24: 6-phase implementation arc landed end-to-end on the
  `cvvdp-fork-rfc` branch:
  - Phase 1 (RFC + scope): this document.
  - Phase 2 (`8c6e91cc`): `ButteraugliBackend` trait renamed →
    `PerceptualBackend` per §2.1; existing CPU/GPU butteraugli
    backends preserved.
  - Phase 3 (`57757ff8`): `GpuCvvdpBackend` impl + `cvvdp-loop` cargo
    feature + `LossyConfig::with_cvvdp_loop` builder API +
    `resolve_cvvdp_loop` Libjxl-strict invariant.
  - Phase 4 (`32581839`, seed `9c1cfa3e`): buttloop wiring + JOD
    calibration seed table at `vardct/cvvdp_targets.rs`.
  - Phase 5 (`206b874e`): CPU CVVDP backend integration via
    `cvvdp-loop-cpu` feature + `with_cvvdp_use_cpu` builder API.
    Output byte-identical to GPU backend; ~1.55× wall vs butteraugli
    CPU at e=8.
  - Phase 6 (`8b5a13a7`): 4-backend × 1,134-cell tracking sweep +
    Pareto analysis. **Verdict: OPT_IN_ONLY** per RFC §5.4
    (see [`CVVDP_FORK_DECISION.md`](CVVDP_FORK_DECISION.md)). cvvdp
    Pareto-wins on its own metric on CID22 photos (100% of e=8 cells
    score higher on `cvvdp_gpu`, mean +0.005 to +0.31 JOD across
    distances); butteraugli leads on bytes-at-same-distance everywhere
    because the cvvdp target table is uncalibrated (+155% to +286%
    file size at the same `distance` parameter on CID22 photos).
    60/60 multi-decoder roundtrips PASS (jxl-oxide + djxl + jxl-rs);
    REVERT branch closed.
- 2026-05-24: Phase 7 docs closeout. cvvdp + cvvdp-loop-cpu features
  shipped behind opt-in API; CHANGELOG + README + this status line
  updated. Distance-table calibration left as future work per
  RFC §2.3 TBD and `CVVDP_FORK_DECISION.md` §11. The opt-in is
  callable today via `LossyConfig::with_cvvdp_loop(Some(true))` plus
  `--features cvvdp-loop` (GPU) or `--features cvvdp-loop-cpu` (CPU).
- 2026-05-25: Phase 8 cumulative interventions (8a-8g) landed via
  `0ab5a53d` + `7876ba95` + `689ba0df`. Diffmap renormalization
  (8c) + bytes-tighten exit pass (8d, gated behind `cvvdp-loop-tighten`
  cargo feature) + per-block reducer constants refit (8g
  `k_tile_norm = 0.16`) close the +200% bytes overhead the Phase 6
  verdict flagged. Phase 8f validation sweep on full 1,134-cell corpus
  shows the combined `C_GPU_v4` stack at **97.7% Pareto-front position**
  (vs B 69.6%, vs original C_GPU 32.5%), **-7.4% mean bytes vs B**,
  **-27% mean wall vs B**, and **30/30 multi-decoder roundtrip PASS**.
  Strict RFC §5.4 within-5pp gate fails on 6 (corpus, metric) cells
  (CID22 butter by < 1pp, GB82-SC ALL metrics by 5.7-13.1pp largely
  due to 29 OOM cells from the tighten pass on large screenshots).
  Verdict stays **OPT_IN_ONLY (improved)**: the cvvdp-fork ships the
  same opt-in features as Phase 7 but now strictly improves on
  butteraugli at the macro level when compiled in. See
  [`CVVDP_FORK_DECISION.md`](CVVDP_FORK_DECISION.md) §13 for the full
  Phase 8f decision-rule application.
