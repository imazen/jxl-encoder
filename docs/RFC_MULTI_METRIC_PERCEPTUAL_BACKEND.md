# RFC: Multi-metric `PerceptualBackend` architecture — explicit per-encode metric selection

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-25
**Status**: SCOPING (architectural design; informs `RFC_ZENSIM_FORK_PLAN.md` + future metric integrations)

This RFC designs the architecture for jxl-encoder's metric-selection API after zensim joins butteraugli and cvvdp as a third `PerceptualBackend` impl. The user mandate (2026-05-25):

> "zensim's latest version is comparable to ssim2 and cvvdp now. We want it added to the lineup with explicit per-encode user control."

The current API surface (post-cvvdp Phase 8g) exposes one boolean opt-in per metric — `with_gpu_butteraugli(bool)`, `with_cvvdp_loop(Option<bool>)`, `with_cvvdp_use_cpu(Option<bool>)`. Adding zensim under this pattern would mean four overlapping setters with implicit dispatch precedence. Not maintainable.

This RFC proposes a unified `PerceptualMetric` enum + a single explicit-choice setter `with_perceptual_metric(metric: PerceptualMetric)` that preserves the existing setters as `#[doc(hidden)] #[deprecated]` shims for one release cycle, then deletes them. Per CLAUDE.md "no backwards-compat hacks" — we have no external users; the API churn is contained.

## §1. The unified API surface

### §1.1. New enum + builder

```rust
// jxl-encoder/src/api.rs

/// Which perceptual metric drives the iterative quantization loop
/// when [`LossyConfig::butteraugli_iters`] > 0.
///
/// All three metrics share the same [`PerceptualBackend`] trait surface
/// (`vardct/perceptual_backend.rs`) — a per-cell `set_reference` followed
/// by per-iter `compare_with_reference` calls that return a scalar score
/// + per-pixel diffmap. The metric choice is fixed for the full encode;
/// per-iter switching is not supported.
///
/// **Default**: [`Self::Butteraugli`]. The other two metrics are opt-in
/// per their respective cargo features.
///
/// **`EncoderStrategy::Libjxl` invariant**: when the active strategy is
/// [`EncoderStrategy::Libjxl`], the metric is forced to [`Self::Butteraugli`]
/// regardless of this field (per W44-126 strict cjxl-parity invariant).
/// Set via [`LossyConfig::resolve_perceptual_metric`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PerceptualMetric {
    /// Butteraugli (max-norm score; smaller=better; calibrated against
    /// libjxl reference encoder). The default. Pareto-optimal on every
    /// corpus measured in `docs/CVVDP_FORK_DECISION.md`. CPU always
    /// available via the `butteraugli-loop` cargo feature (default on).
    /// GPU acceleration via the `gpu-butteraugli` cargo feature.
    Butteraugli,

    /// CVVDP (Mantiuk et al. 2024; JOD-direction normalized to butter-
    /// direction at trait boundary). Pareto-tied with butteraugli at 85%
    /// front position after Phase 8g (`689ba0df`) per-block reducer
    /// refit. Opt-in via the `cvvdp-loop` (GPU) or `cvvdp-loop-cpu` (CPU)
    /// cargo features. See `docs/RFC_CVVDP_FORK.md`.
    Cvvdp,

    /// zensim (multi-scale XYB SSIM + edge + HF + trained per-codec
    /// affine; score-direction normalized to butter-direction at trait
    /// boundary). Opt-in via the `zensim-loop` (CPU) or `zensim-loop-gpu`
    /// (GPU) cargo features. See `docs/RFC_ZENSIM_FORK_PLAN.md`.
    Zensim,
}

impl Default for PerceptualMetric {
    fn default() -> Self { Self::Butteraugli }
}

/// Compute-device preference for the active perceptual metric. Both
/// `Cpu` and `Gpu` are opt-in subject to their respective cargo features;
/// `Auto` (default) prefers GPU when both backends compiled in AND the
/// CUDA runtime initialises successfully, falling back to CPU otherwise.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PerceptualDevice {
    /// "Prefer GPU when available; fall back to CPU otherwise."
    #[default]
    Auto,

    /// Force CPU. Required for reproducibility (CPU paths have no GPU
    /// reduction-order variance — see W44-RECON-DEEP/A7).
    Cpu,

    /// Force GPU. Errors out (or silent-falls-back to CPU per metric's
    /// cargo features) if CUDA is unavailable.
    Gpu,
}
```

### §1.2. `LossyConfig` builder methods

```rust
// jxl-encoder/src/api.rs

impl LossyConfig {
    /// Set which perceptual metric drives the iterative quantization
    /// loop. Default: [`PerceptualMetric::Butteraugli`].
    ///
    /// Choosing a non-default metric requires the corresponding cargo
    /// feature to be compiled in. Without the feature, the dispatch
    /// silently falls back to butteraugli (with a one-shot eprintln
    /// warning the first time the caller picks an unbuilt metric).
    pub fn with_perceptual_metric(mut self, metric: PerceptualMetric) -> Self {
        self.perceptual_metric = metric;
        self
    }

    /// Currently configured metric (default: `Butteraugli`).
    pub fn perceptual_metric(&self) -> PerceptualMetric { self.perceptual_metric }

    /// Set the compute-device preference for the active metric. Default:
    /// [`PerceptualDevice::Auto`].
    pub fn with_perceptual_device(mut self, device: PerceptualDevice) -> Self {
        self.perceptual_device = device;
        self
    }

    /// Currently configured device (default: `Auto`).
    pub fn perceptual_device(&self) -> PerceptualDevice { self.perceptual_device }

    /// Override the metric's per-distance target table. When `None`
    /// (default), the metric's built-in calibration table from
    /// `vardct/<metric>_targets.rs` drives the loop's target. When
    /// `Some(score)`, the loop targets that score directly in the
    /// metric's score-direction (smaller=better).
    ///
    /// Use when calibrating against a non-standard quality requirement
    /// (e.g. matching a specific reference encoder's output). Default
    /// `None` is the right choice for ~all production callers.
    pub fn with_perceptual_target_score(mut self, score: Option<f32>) -> Self {
        self.perceptual_target_score = score;
        self
    }

    pub fn perceptual_target_score(&self) -> Option<f32> {
        self.perceptual_target_score
    }
}
```

### §1.3. Resolver method (Libjxl invariant + feature-gate fallback)

```rust
impl LossyConfig {
    /// Resolve the effective perceptual metric, honouring the
    /// `EncoderStrategy::Libjxl` strict-parity invariant (W44-126) +
    /// the per-metric cargo-feature gates.
    ///
    /// Returns the metric that will ACTUALLY drive the loop, which
    /// may differ from [`Self::perceptual_metric`] when:
    ///   - The active strategy is [`EncoderStrategy::Libjxl`] →
    ///     ALWAYS `Butteraugli` (strict parity).
    ///   - The configured metric's cargo feature is not compiled in →
    ///     falls back to `Butteraugli` with a one-shot warning.
    pub(crate) fn resolve_perceptual_metric(&self) -> PerceptualMetric {
        if matches!(self.strategy, EncoderStrategy::Libjxl) {
            return PerceptualMetric::Butteraugli;
        }
        let requested = self.perceptual_metric;
        match requested {
            PerceptualMetric::Butteraugli => PerceptualMetric::Butteraugli,
            PerceptualMetric::Cvvdp => {
                #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
                { PerceptualMetric::Cvvdp }
                #[cfg(not(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu")))]
                { PerceptualMetric::Butteraugli }
            }
            PerceptualMetric::Zensim => {
                #[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
                { PerceptualMetric::Zensim }
                #[cfg(not(any(feature = "zensim-loop", feature = "zensim-loop-gpu")))]
                { PerceptualMetric::Butteraugli }
            }
        }
    }

    pub(crate) fn resolve_perceptual_device(&self) -> PerceptualDevice {
        self.perceptual_device
    }
}
```

### §1.4. Example caller code

```rust
use jxl_encoder::api::{LossyConfig, PerceptualMetric, PerceptualDevice};

// Default — butteraugli on the best available device.
let cfg = LossyConfig::default().with_distance(1.0);
// equivalent to:
// .with_perceptual_metric(PerceptualMetric::Butteraugli)
// .with_perceptual_device(PerceptualDevice::Auto)

// Opt into cvvdp on GPU; fall back to CPU if CUDA missing.
let cfg = LossyConfig::default()
    .with_distance(1.0)
    .with_perceptual_metric(PerceptualMetric::Cvvdp)
    .with_perceptual_device(PerceptualDevice::Auto);

// Opt into zensim, force CPU for reproducibility.
let cfg = LossyConfig::default()
    .with_distance(1.0)
    .with_perceptual_metric(PerceptualMetric::Zensim)
    .with_perceptual_device(PerceptualDevice::Cpu);

// Opt into cvvdp but override the target score (e.g. user wants the
// JOD-direction-normalized score to converge to 0.05 = ~9.95 JOD).
let cfg = LossyConfig::default()
    .with_distance(1.0)  // bitstream target_distance
    .with_perceptual_metric(PerceptualMetric::Cvvdp)
    .with_perceptual_target_score(Some(0.05));
```

The compile-time matrix mirrors the existing cvvdp setup:

| caller wants  | cargo features needed              |
|---            |---                                 |
| Butteraugli   | `butteraugli-loop` (default on)    |
| Butteraugli + GPU | `gpu-butteraugli`              |
| Cvvdp + GPU only  | `cvvdp-loop`                   |
| Cvvdp + CPU only  | `cvvdp-loop-cpu`               |
| Cvvdp + auto      | `cvvdp-loop` + `cvvdp-loop-cpu`|
| Zensim + GPU only | `zensim-loop-gpu`              |
| Zensim + CPU only | `zensim-loop`                  |
| Zensim + auto     | `zensim-loop` + `zensim-loop-gpu`|

## §2. Per-metric target-score semantics

The buttloop's caller-facing `target_distance: f32` parameter is **butteraugli-native** (libjxl convention, d=1.0 ≈ visually lossless). Every metric's per-distance target table maps butter `target_distance` → metric-native `effective_metric_target_distance` (in direction-normalized score units, smaller=better).

### §2.1. Three target tables

```rust
// vardct/butter_targets.rs (NEW — extracted from current pass-through)
/// Butteraugli's target IS the distance directly (identity table). Lookup
/// is `target_distance.max(0.0)` — no calibration needed.
pub(crate) fn butter_target_score_for_distance(d: f32) -> f32 { d.max(0.0) }

// vardct/cvvdp_targets.rs (EXISTING — Phase 4 commit `32581839`)
pub(crate) static CVVDP_DISTANCE_TARGETS: &[(f32, f32); 7] = &[
    (0.50, 0.0029), (1.00, 0.0238), (1.50, 0.0461),
    (2.00, 0.0724), (3.00, 0.1336), (4.00, 0.2149), (5.00, 0.3005),
];
pub(crate) fn cvvdp_target_score_for_distance(d: f32) -> f32 { /* linear interp */ }

// vardct/zensim_targets.rs (NEW — built by RFC #4 Phase 4 baseline sweep)
pub(crate) static ZENSIM_DISTANCE_TARGETS: &[(f32, f32); 7] = &[
    // TBD — Phase 4 seed methodology mirrors cvvdp's
    // cvvdp_jod_calibration_seed_2026-05-24.txt shape: median zensim
    // score across the corpus at each distance, direction-normalized,
    // optional 1.05× tightening factor (cvvdp arc empirically showed
    // tightening overshoots; recommend starting at 1.00 = no tightening,
    // calibrate via Phase 8c renorm if Pareto-pct < 85%).
    (0.50, 0.0), (1.00, 0.0), (1.50, 0.0),  // placeholder values
    (2.00, 0.0), (3.00, 0.0), (4.00, 0.0), (5.00, 0.0),
];
pub(crate) fn zensim_target_score_for_distance(d: f32) -> f32 { /* linear interp */ }
```

### §2.2. Dispatch in `perceptual_loop.rs`

The current dispatch at `vardct/perceptual_loop.rs:2157-2167`:

```rust
let effective_metric_target_distance: f32 = {
    if cvvdp_loop_active && !use_vdp2 {
        super::cvvdp_targets::cvvdp_target_score_for_distance(target_distance)
    } else {
        target_distance
    }
};
```

Refactored to dispatch on resolved metric:

```rust
use crate::api::PerceptualMetric;
let effective_metric_target_distance: f32 = {
    if let Some(override_score) = self.perceptual_target_score {
        override_score  // explicit caller override
    } else {
        match self.resolved_metric {
            PerceptualMetric::Butteraugli =>
                super::butter_targets::butter_target_score_for_distance(target_distance),
            #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
            PerceptualMetric::Cvvdp if !use_vdp2 =>
                super::cvvdp_targets::cvvdp_target_score_for_distance(target_distance),
            #[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
            PerceptualMetric::Zensim =>
                super::zensim_targets::zensim_target_score_for_distance(target_distance),
            _ =>
                super::butter_targets::butter_target_score_for_distance(target_distance),
        }
    }
};
```

The `use_vdp2` short-circuit preserves the existing vdp2 path (butteraugli's `compute_for_vdp` mode); cvvdp dispatch is skipped when vdp2 is in play (which the cvvdp Phase 4 wiring already documented at `cvvdp_fork_phase4_loop_wiring_shipped_2026-05-24.md:32-43`).

## §3. Trait evolution — no changes needed

The `PerceptualBackend` trait at `vardct/perceptual_backend.rs:325-381` is **already shape-correct** for all three metrics. cvvdp Phase 2 (`8c6e91cc`) generalized it from `ButteraugliBackend` → `PerceptualBackend` exactly because the same shape works for any per-pixel perceptual metric. zensim integration extends the existing 4 impls to 6:

| Existing impl              | feature flag                       |
|---                         |---                                 |
| `CpuButteraugliBackend`    | `butteraugli-loop` (default on)    |
| `GpuButteraugliBackend`    | `gpu-butteraugli`                  |
| `GpuCvvdpBackend`          | `cvvdp-loop`                       |
| `CpuCvvdpBackend`          | `cvvdp-loop-cpu`                   |
| **`CpuZensimBackend`** (new) | **`zensim-loop`**                |
| **`GpuZensimBackend`** (new) | **`zensim-loop-gpu`**            |

The new impls live in `vardct/zensim_backend.rs` (mirror `cvvdp_backend.rs` shape). NO changes to:
- The `PerceptualBackend` trait signature.
- The `BackendCompareResult { score: f64 }` shape.
- The buttloop body (it calls trait methods through `Box<dyn PerceptualBackend>`).

The W44-PHASE3-B5b divergence detector (`vardct/perceptual_backend.rs:548-628`) stays butteraugli-specific (it shadows a CPU butteraugli compare against a GPU butteraugli compare to detect reduction-order divergence). cvvdp + zensim don't currently have a divergence-detector equivalent; if cross-impl drift becomes a concern, the trait's `divergence_status` extension point already exists.

## §4. Dispatch — `construct_backend` extended for 3-way metric

The current dispatch at `vardct/perceptual_backend.rs:1232-1457` takes:

```rust
pub(crate) fn construct_backend(
    width: u32, height: u32,
    cpu_params: butteraugli::ButteraugliParams,
    intensity_target: f32,
    gpu_requested: bool,            // gpu-butteraugli
    cvvdp_requested: bool,
    cvvdp_use_cpu_requested: bool,
) -> Box<dyn PerceptualBackend>;
```

This 7-arg signature is already at the edge of maintainability (the test `construct_backend_cpu_when_cvvdp_not_requested` passes 5 trailing bools). Adding zensim under the same pattern would force a 9-arg signature.

**Refactor**: collapse the dispatch hints into a single struct:

```rust
/// Resolved metric-selection state passed to `construct_backend`. Built
/// by `LossyConfig::resolve_perceptual_metric_selection`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct MetricSelection {
    /// Which metric to construct. The Libjxl strict-parity short-circuit
    /// has already been applied — `Butteraugli` here means "construct a
    /// butteraugli backend" unconditionally.
    pub metric: PerceptualMetric,
    /// Device preference. `Auto` follows the dispatch matrix (try GPU,
    /// fall back to CPU); `Cpu` / `Gpu` force.
    pub device: PerceptualDevice,
}

pub(crate) fn construct_backend(
    width: u32, height: u32,
    cpu_params: butteraugli::ButteraugliParams,
    intensity_target: f32,
    selection: MetricSelection,
) -> Box<dyn PerceptualBackend>;
```

Dispatch body (pseudocode):

```rust
match selection.metric {
    PerceptualMetric::Cvvdp => {
        // Phase 5 dispatch matrix: prefer CPU when Device::Cpu, else GPU first.
        #[cfg(feature = "cvvdp-loop-cpu")]
        if selection.device == PerceptualDevice::Cpu {
            if let Some(c) = CpuCvvdpBackend::try_new(w, h) { return c; }
        }
        #[cfg(feature = "cvvdp-loop")]
        if let Some(c) = GpuCvvdpBackend::try_new(w, h) { return c; }
        #[cfg(feature = "cvvdp-loop-cpu")]
        if let Some(c) = CpuCvvdpBackend::try_new(w, h) { return c; }
        // Fall through to butter dispatch.
    }
    PerceptualMetric::Zensim => {
        #[cfg(feature = "zensim-loop-gpu")]
        if selection.device != PerceptualDevice::Cpu {
            if let Some(g) = GpuZensimBackend::try_new(w, h) { return g; }
        }
        #[cfg(feature = "zensim-loop")]
        if let Some(c) = CpuZensimBackend::try_new(w, h) { return c; }
        // Fall through to butter dispatch.
    }
    PerceptualMetric::Butteraugli => { /* current butter dispatch */ }
}
// Butter dispatch (same as today):
#[cfg(feature = "gpu-butteraugli")]
if selection.device != PerceptualDevice::Cpu {
    if let Some(g) = GpuButteraugliBackend::try_new(w, h, intensity_target, ...) {
        return g;
    }
}
Box::new(CpuButteraugliBackend::new(cpu_params))
```

The refactor is mechanical; the existing 7-arg dispatch tests at `perceptual_backend.rs:1559-1627` translate 1:1.

## §5. Default-strategy rules across the three metrics

The user mandate: "user explicitly controlling which is used." This translates to: NO content-class-based auto-dispatch. The default metric is butteraugli; opting into a different metric is an explicit caller decision.

### §5.1. Per-`EncoderStrategy` rules

| Strategy            | Default metric    | Caller override allowed?         |
|---                  |---                |---                               |
| `Default` (= Zenjxl) | Butteraugli       | YES (via `with_perceptual_metric`) |
| `Zenjxl`            | Butteraugli       | YES                              |
| `LeanFaster`        | Butteraugli       | YES                              |
| `Aggressive`        | Butteraugli       | YES                              |
| `Libjxl`            | Butteraugli (FORCED) | **NO** (strict parity invariant) |
| `Custom`            | Butteraugli       | YES                              |

The Libjxl invariant is the only hard constraint. Every other strategy respects the caller's metric choice.

### §5.2. Why butteraugli stays the default

Three reasons (in priority order):

1. **Pareto-optimality**: cvvdp Phase 6 + Phase 8g verdict (`docs/CVVDP_FORK_DECISION.md`) is that butteraugli sits on the Pareto front of every (corpus, metric) pair tested. Even after Phase 8g closed the cvvdp gap to 85%, butteraugli is at 100% on multiple corpora.

2. **Calibration depth**: the W44-1..W44-228 cost-model gates (every entry in `docs/LIBJXL_DIVERGENCES.md` Section B/C) are calibrated against butteraugli. Switching default would invalidate that calibration — multi-week W44-gate-transfer work per `docs/CVVDP_W44_GATE_TRANSFER.md`.

3. **Predictability**: callers across the imageflow ecosystem (RIAPI, server, dotnet, node, go bindings) ship with the assumption that "JXL encoder with distance=1.0 produces a butteraugli=1.0 file." Changing default would break that expectation cascade.

### §5.3. When a non-default metric is the right choice

The choice is **caller-driven**. Surfacing the right metric for the workload is the caller's responsibility — RFC #3 doesn't impose a "use zensim for screenshots, butteraugli for photos" auto-dispatch. The reasons:

- **Different metrics measure different things.** zensim's per-codec affine calibration is specifically valuable for JXL-to-JXL comparisons (vs cvvdp's general-perception calibration vs butteraugli's libjxl-derived calibration). Cross-codec users might prefer cvvdp.
- **Different ground-truth corpora.** zensim was trained on 344k JXL-inclusive pairs; cvvdp on Mantiuk et al.'s subjective-MOS dataset. The "right" metric depends on what the caller is trying to optimize against.
- **Wall-time vs Pareto tradeoff.** A caller who wants the smallest possible files at the lowest wall picks GPU butteraugli (default + GPU). A caller who wants tighter perceptual-quality calibration on a specific corpus picks the corpus-matching metric.

Auto-dispatch on content class would lock the encoder into a single canonical "best metric per content class" choice that's hard to reverse. The explicit-choice API keeps that decision in the caller's hands.

## §6. Backwards compatibility — hard rename per CLAUDE.md

Per CLAUDE.md "no backwards-compat hacks — bump the 0.x major version for breaking changes." The current API surface has 4 metric-related setters; the new surface has 3:

| Old setter                           | New equivalent                                         |
|---                                   |---                                                     |
| `with_gpu_butteraugli(bool)`         | `with_perceptual_device(PerceptualDevice::Gpu)` (with metric `Butteraugli`) |
| `with_cvvdp_loop(Option<bool>)`      | `with_perceptual_metric(PerceptualMetric::Cvvdp)`     |
| `with_cvvdp_use_cpu(Option<bool>)`   | `with_perceptual_device(PerceptualDevice::Cpu)`       |
| `with_cvvdp_bytes_tighten(Option<bool>)` | `with_cvvdp_bytes_tighten(Option<bool>)` (KEPT — it's a cvvdp-specific tuning knob, not metric-selection) |

### §6.1. Migration shape

**Option A — hard rename in one minor (recommended)**:

- Add the new enum + setters in jxl-encoder 0.2.0.
- Delete the old setters (no `#[deprecated]` shim) in the same release.
- CHANGELOG documents the migration in detail.
- Per CLAUDE.md "we have no external users" — this is the right shape.

**Option B — one-release deprecation cycle**:

- Add the new enum + setters in 0.2.0.
- Mark old setters `#[doc(hidden)] #[deprecated(note = "use with_perceptual_metric / with_perceptual_device")]`.
- Implement old setters as wrappers around the new ones for one release.
- Delete in 0.3.0.

**Recommendation**: Option A. The cvvdp setters have only been in tree since 2026-05-24 (cvvdp Phase 3 commit `57757ff8`); breaking them in 0.2.0 is fine. The `with_gpu_butteraugli` setter has been in tree since W44-PHASE3-B5-flip (2026-05-23, ~3 days older); same call.

**`cvvdp_bytes_tighten` stays as-is.** It's a cvvdp-specific tuning knob (Phase 8d), not part of the metric-selection surface. Keep it; document that it's only effective when `perceptual_metric == Cvvdp`. The current `resolve_cvvdp_bytes_tighten` already short-circuits when `resolve_cvvdp_loop` returns false; the same shape works with `resolve_perceptual_metric() == Cvvdp`.

### §6.2. Field migration in `LossyConfig`

Current state (api.rs:4275-4905):

```rust
gpu_butteraugli: bool,           // default cfg!(feature = "gpu-butteraugli")
cvvdp_loop: Option<bool>,        // None
cvvdp_use_cpu: Option<bool>,     // None
cvvdp_bytes_tighten: Option<bool>, // None
```

New state:

```rust
perceptual_metric: PerceptualMetric,    // Butteraugli (default)
perceptual_device: PerceptualDevice,    // Auto (default)
perceptual_target_score: Option<f32>,   // None
cvvdp_bytes_tighten: Option<bool>,      // None (UNCHANGED)
```

Net field count: 4 → 4. Zero new fields. The semantic clarity goes from "boolean explosion" to "explicit metric + device enums".

### §6.3. Default behaviour invariant

`LossyConfig::default()` MUST produce byte-identical encode output across the migration. The default state:

- Old: `gpu_butteraugli: cfg!(feature = "gpu-butteraugli")`, `cvvdp_loop: None`, `cvvdp_use_cpu: None`.
- New: `perceptual_metric: Butteraugli`, `perceptual_device: Auto`.

Both resolve to "butteraugli backend, GPU when available else CPU." The `Auto` device matches the old `cfg!(feature = "gpu-butteraugli")` default — same backend selection. Hash-locks must stay 36/36 byte-identical across the migration.

## §7. Per-metric per-block constant tables

The `BlockReducerConstants` table at `vardct/perceptual_loop.rs:1160-1248` currently has 2 entries (butter + cvvdp). zensim adds a third:

```rust
pub(crate) struct BlockReducerConstants { pub k_tile_norm: f32 }

pub(crate) const BUTTER_BLOCK_CONSTANTS: BlockReducerConstants =
    BlockReducerConstants { k_tile_norm: 1.2 };

#[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
pub(crate) const CVVDP_BLOCK_CONSTANTS: BlockReducerConstants =
    BlockReducerConstants { k_tile_norm: 0.16 };

#[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
pub(crate) const ZENSIM_BLOCK_CONSTANTS: BlockReducerConstants =
    BlockReducerConstants {
        k_tile_norm: /* TBD — Phase 4 fit per RFC #1 §3.2 */,
    };
```

Dispatcher updated to 3-way:

```rust
pub(crate) fn block_reducer_constants_for_backend(metric: PerceptualMetric)
    -> BlockReducerConstants
{
    match metric {
        PerceptualMetric::Butteraugli => BUTTER_BLOCK_CONSTANTS,
        #[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
        PerceptualMetric::Cvvdp => CVVDP_BLOCK_CONSTANTS,
        #[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
        PerceptualMetric::Zensim => ZENSIM_BLOCK_CONSTANTS,
        _ => BUTTER_BLOCK_CONSTANTS,  // unreachable except in feature-off builds
    }
}
```

Same shape for the diffmap renorm scale (`vardct/perceptual_backend.rs:111`):

```rust
#[cfg(any(feature = "cvvdp-loop", feature = "cvvdp-loop-cpu"))]
pub(crate) const CVVDP_DIFFMAP_RENORM_SCALE: f32 = 0.018;

#[cfg(any(feature = "zensim-loop", feature = "zensim-loop-gpu"))]
pub(crate) const ZENSIM_DIFFMAP_RENORM_SCALE: f32 =
    /* TBD — Phase 4 fit per RFC #1 §4.2 */;
```

The env override hooks (`JXL_CVVDP_K_TILE_NORM` / `JXL_CVVDP_DIFFMAP_RENORM_SCALE`) get zensim siblings (`JXL_ZENSIM_K_TILE_NORM` / `JXL_ZENSIM_DIFFMAP_RENORM_SCALE`). Bench-only; not for production callers.

## §8. CLI exposure (`jxl-encoder-cli`)

The current CLI doesn't expose any metric-selection flag (per `cvvdp_fork_phase6_tracking_sweep_shipped:160-163`). The new flags should land as part of the multi-metric migration:

```bash
cjxl-rs input.png output.jxl \
    --distance 1.0 \
    --perceptual-metric {butteraugli|cvvdp|zensim} \  # default butteraugli
    --perceptual-device {auto|cpu|gpu}                # default auto
```

The CLI flag parser maps to the new enums directly. Feature-gated per the same cargo features as the library API.

## §9. EncoderStrategy::Libjxl invariant preservation

**Mandatory and load-bearing.** Tested by `tests/strategy_libjxl_byte_lock.rs` (W44-194) — 4 fixtures must produce byte-identical output regardless of any opt-in flag.

The new dispatch shape preserves this via:

```rust
impl LossyConfig {
    pub(crate) fn resolve_perceptual_metric(&self) -> PerceptualMetric {
        if matches!(self.strategy, EncoderStrategy::Libjxl) {
            return PerceptualMetric::Butteraugli;  // FORCED
        }
        // ... feature-gate dispatch ...
    }
}
```

The byte-lock test must be extended with new opt-in shapes:

```rust
// tests/strategy_libjxl_byte_lock.rs
#[test]
fn libjxl_byte_lock_with_zensim_opt_in() {
    let pinned = encode_with_strategy(EncoderStrategy::Libjxl, /* default cfg */);
    let with_zensim = encode_with_strategy_and_metric(
        EncoderStrategy::Libjxl, PerceptualMetric::Zensim);
    assert_eq!(pinned.bytes, with_zensim.bytes,
        "Libjxl strategy MUST be byte-identical regardless of metric choice");
}
```

Same shape for `PerceptualMetric::Cvvdp` + the device matrix. The test grows from 4 fixtures × 1 path to 4 fixtures × 9 paths (3 metrics × 3 devices) = 36 byte-locks.

## §10. `EncodeStats` — surface the active backend

The current `EncodeStats` (api.rs) doesn't surface which metric/backend the encode actually used. Adding it makes debugging silent-fallback paths possible:

```rust
pub struct EncodeStats {
    // ... existing fields ...

    /// Which perceptual metric drove the buttloop (may differ from
    /// `LossyConfig::perceptual_metric()` when a silent fallback fired
    /// — e.g. caller requested cvvdp but cargo feature not compiled,
    /// or GPU init failed).
    pub active_perceptual_metric: PerceptualMetric,

    /// Which device. `Auto` resolves to either `Cpu` or `Gpu` here;
    /// you can read this to verify GPU actually fired.
    pub active_perceptual_device: PerceptualDevice,

    /// Backend's `name()` (e.g. `"cpu"`, `"gpu-cuda"`, `"cvvdp-cpu"`,
    /// `"zensim-gpu-cuda"`, `"gpu-cuda-fallback-cpu"` from W44-PHASE3-B5b).
    pub active_perceptual_backend_name: &'static str,
}
```

This is a Phase 3-or-later API surface addition; not strictly required for opt-in shipment but useful for debugging.

## §11. Migration plan summary

1. **Phase A** (1-2 days, prereq for zensim Phase 3): land `PerceptualMetric` + `PerceptualDevice` enums + `LossyConfig` builder migration + `construct_backend` signature refactor. CHANGELOG breaking-change entry. Hash-locks 36/36 byte-identical (default behaviour unchanged).

2. **Phase B** (concurrent with zensim Phase 1-3): zensim ships using the new API surface from day one. NO old-setter shim ever exists; the migration is atomic.

3. **Phase C** (after zensim Phase 7 closeout): CLI flags `--perceptual-metric` + `--perceptual-device` land in `jxl-encoder-cli`.

**The migration is decoupled from the zensim integration timeline.** Phase A could ship before zensim Phase 1 starts (clean refactor on top of current cvvdp surface); Phase B onwards lands as zensim arc progresses.

## §12. Out of scope (for this RFC)

- Per-image content-class metric dispatch — the user explicitly wants explicit control, NOT auto-dispatch.
- Per-iter metric switching (would invalidate the buttloop's convergence assumptions).
- Cross-metric ensemble (e.g. "score = α·butter + β·cvvdp + γ·zensim") — multi-metric averaging is a research question beyond the buttloop's iterative refinement shape.
- Replacing SSIMULACRA2 in non-buttloop contexts (legacy tuning sweeps stay on SSIM2).
- Animation buttloop multi-metric (still-image only per cvvdp RFC §8).
- HDR-aware metric dispatch (HDR buttloop currently butter-only; per-metric HDR is a future workstream).
- Surfacing per-metric calibration tables in the public API (`zensim_targets` / `cvvdp_targets` stay `pub(crate)`).
