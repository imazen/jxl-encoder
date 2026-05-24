# RFC: cvvdp fork — replace butteraugli buttloop with ColorVideoVDP

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-24
**Status**: SCOPING

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

**Critical invariant** (test in both impls):

```
JOD_scalar  ==  10.0 - minkowski_p_norm(diffmap_flat, p = params.beta_sch)
```

within 1e-5 relative. If they disagree, one of them is wrong.

For matched inputs (ref ≡ dist), the diffmap must be all zeros to
1e-7 absolute (no false-floor).

**Reference for upstream**: pycvvdp has a `get_diff_map=True` debug
flag we can mirror against; the official semantics are documented in
`Mantiuk et al. 2024`. If our recipe differs from upstream's, this
RFC's recipe is the contract — the JXL buttloop's per-block heuristic
was designed against butteraugli's pre-pool spatial signal; we mirror
that shape.

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

- **CID22-512** (41 photos held out from training).
- **GB82-SC** (10 screenshots).
- **W44-PHASE4-S1 corpus subset** (50 stratified samples from the
  recent S1 sweep — covers the production-relevant content range).

Total: 101 images. For each, sweep ⌊distance ∈ {0.5, 1.0, 1.5, 2.0,
3.0, 4.0, 5.0}⌋ × effort ∈ {5, 7, 8} = 21 cells per image, 2,121
cells per backend.

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
- TBD: Phase 1 ship target, Phase 2 ship target, ...
