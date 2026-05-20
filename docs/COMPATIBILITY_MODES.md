# Compatibility Modes — Design Doc (W44-125)

**Status**: DESIGN ONLY — no implementation in this commit.
**Author**: agent W44-125
**Date**: 2026-05-20

---

## 1. Motivation

Through the W22-1 → W44-124 arc the encoder has accumulated ~12 caller-visible
opt-in knobs that each enable, suppress, or tweak one or more divergences
against libjxl reference. Today they ship as scattered `with_*_hint(Option<bool>)`
setters on [`LossyConfig`](../jxl-encoder/src/api.rs) — every new chunk that
introduces a content-aware gate creates another `Option<bool>` field, and the
matrix of "what knobs do I set to get cjxl-strict bitstreams?" / "what knobs do
I set to get our maximum-RD photo bundle?" / "what knobs do I set for the
screenshot-class improvements but not the photo-class ones?" is unwieldy and
under-specified.

The caller-visible problem has three faces:

1. **No cjxl-strict mode.** A caller who wants to verify libjxl parity for
   conformance testing has to know every `with_*_hint` setter individually and
   pin each to the matching "off" value. The set is moving — every chunk adds a
   knob — so a `git pull` can change what "strict cjxl" means.

2. **No coherent bundling.** Some improvements compose (e.g. W44-91 photo +
   W44-117 EPF screenshot seed are orthogonal); others are mutually exclusive
   (e.g. EPF sharpness map can be the legacy uniform-4 seed OR the W44-117
   one-shot seed OR a hypothetical per-iter recompute — they're three picks of
   one knob, not three independent bools). The `Option<bool>` field shape can't
   model that — it admits invalid configurations like "force seed AND force
   uniform-4 AND force per-iter recompute".

3. **No release-grade default presets.** Today the encoder defaults to "all the
   improvements". Callers wanting libjxl-faithful output ("cjxl mirror") or a
   conservative "only the ported algorithm fixes, none of the content-aware
   gates" bundle have to set knobs one at a time.

This doc proposes ONE top-level enum [`EncoderCompatibility`] with named
preset variants (`CjxlMirror`, `Conservative`, `Default`, `Aggressive`) and a
`Custom(EncoderImprovementsCustom)` variant for fine-grained control. The
`Custom` payload uses nested enums to encode mutually-exclusive picks (e.g.
[`EpfSharpnessSeed`] is exactly one of `LegacyUniform4`, `W44_117_OneShot`, or
`Disabled`).

The [`LossyConfig`] keeps the existing `with_*_hint` setters as a low-level
escape hatch but adds a higher-level `with_compatibility(EncoderCompatibility)`
setter that resolves to the same internal fields. `pub(crate)` resolution
methods on `EncoderCompatibility` translate the high-level pick into the
specific bools/floats consumed at each call site, so behaviour is identical
either way the caller drives it.

Per project policy (no external users; we can break the 0.x API freely), no
deprecation shim or backwards-compat wrapper is required — we will iterate.

---

## 2. Current divergence inventory

Every active divergence pulled from [`docs/LIBJXL_DIVERGENCES.md`](LIBJXL_DIVERGENCES.md)
sections A–E. RESOLVED rows (section G) are out of scope; AT-PARITY rows are
listed for completeness because callers may still want to opt OUT of them in
`CjxlMirror` mode (they're at parity *today* but a future divergence might
introduce a knob — the inventory keeps the surface honest).

### 2.1 Effort-gate divergences (Section A)

| Site | Current API | Mutually-exclusive group | One-sentence |
|---|---|---|---|
| `effort.rs:1027` `cfl_two_pass` gate at `effort >= 7` (libjxl `effort >= 5`) | none (effort-level constant) | (none — single bool) | We delay the CFL second-pass refine to e7+; libjxl runs it at e5+. |
| `effort.rs` `try_dct64` gate at `effort >= 7` (libjxl no effort gate) | `LossyConfig::with_dct_suppress_hint`, `with_dct32_keep_hint` (caller can force-allow / force-suppress) | DCT64-class search policy | We gate DCT64 evaluation by effort and content; libjxl always evaluates if not in fast-decode tier. |
| `effort.rs` `epf_dynamic_sharpness` gate at `effort >= 6` (libjxl no effort gate) | `LossyConfig::with_epf_dispatch` | EPF sharpness policy | We adaptively skip the per-block sharpness search on smooth regions at e>=6; libjxl always runs it. |

### 2.2 Content-aware discriminator gates (Section B)

These are SUPERSETS of libjxl behaviour — libjxl uses one global path; we add
narrow content-aware lifts on top.

| Site | Predicate | Current API | Mutually-exclusive group | One-sentence |
|---|---|---|---|---|
| W22-1 `screenshot_lift_hint` | `median(mask1x1) > 95` AND `content_aware_entropy_mul == true` | `with_content_aware_entropy_mul(bool)` + `with_screenshot_lift_hint(Option<bool>)` | Screenshot entropy-mul table | Lifts `IDENTITY`/`DCT2X2`/`AFV`/`DCT4X8` entropy_mul on screenshot-class to suppress small-transform artefacts at sharp glyph edges. |
| W44-29 `high_d_photo_smooth_suppressed` | `d >= 4.0` AND `median(mask1x1) < SMOOTH_THRESHOLD` | `with_high_d_photo_hint(Option<bool>)` | High-d photo entropy-mul table | Lowers `entropy_mul[DCT16X16]/[DCT32X32]` on smooth photos at d>=4 to close the F-D residual byte gap vs cjxl. |
| W44-65/68 `dct_suppress_hint` auto | `median(mask1x1) >= 99.5` | `with_dct_suppress_hint(Option<bool>)` | DCT64 search policy | Auto-suppresses DCT64-class search on screenshot-class content. |
| W44-91 `high_d_photo_smooth_zenanalyze` | W44-29 outer AND `m_colourfulness >= 80` AND `fcbr < 0.01` AND `d ∈ [3.0, 5.0]` | (no explicit hint — sub-gate of W44-29) | High-d photo entropy-mul table | Narrows the W44-29 admission to admit 1189261-class only. |
| W44-96 `high_d_photo_smooth_suppressed_z` | `d >= 4.5` AND `mask1x1 < 50` AND `edge_density >= 0.7` AND `fcbr < 0.01` | (no explicit hint — sub-gate of W44-29) | High-d photo entropy-mul table | DCT32X32 lift for {1420710, 1531677} class at d>=4.5. |
| W44-98 `high_d_photo_smooth_suppressed_z_high_colour` | W44-96 outer AND `m3_colourfulness >= 25.0` | (no explicit hint — sub-gate of W44-96) | High-d photo entropy-mul table | DCT16X32 lift 1.30 for 1420710 (HIGH colour). |
| W44-99/100 `high_d_photo_smooth_suppressed_z_low_colour` | W44-96 outer AND `m3_colourfulness < 25.0` | (no explicit hint — sub-gate of W44-96) | High-d photo entropy-mul table | DCT16X32 lift 1.23 for 1531677 (LOW colour). |
| W44-105/107/108 `BUTTLOOP_QF_SEED_SCALE` (4×) | `is_screenshot AND (d >= 3.5 OR (m3 < 30 AND d >= 2.0))` AND `butteraugli_iters > 0` | (no explicit hint — env-var `JXL_BUTTLOOP_INITIAL_QF_SCALE`) | Butteraugli loop qf-seed scaling | 4× pre-scale of the butteraugli loop's qf seed on screenshot-class to clear the W44-105 SSIM2 gap. |
| W44-109 `resolved_adaptive_quant_qf_seed_scale` (2× e5/e6, 3× e7) | `effort ∈ [5,7] AND butteraugli_iters == 0 AND is_screenshot AND (d >= 3.5 OR (m3 < 30 AND d >= 2.0))` | (no explicit hint — env-var `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`) | Adaptive-quant qf seed scaling at e<8 | Mirrors W44-105 at lower effort where buttloop unavailable; pre-scales `quant_field_float` at adaptive-quant time. |
| W44-117/118/120 EPF sharpness one-shot seed | `is_screenshot AND target_distance >= 1.0 AND profile.epf_dynamic_sharpness AND mask1x1.is_some()` AND buttloop runs | (no explicit hint — env-var `JXL_W44_117_DISABLE`, `JXL_W44_120_EPF_SEED_MIN_DISTANCE`) | EPF sharpness seed for buttloop | Computes `compute_epf_sharpness` once before the loop on the initial reconstruction; reuses the map for every iter's `apply_epf` (closes the buttloop-vs-decoder EPF mismatch on screenshots). |
| W44-123 `dct32_keep_hint` | (default `None` = follow W44-68 drop) — opt-in `Some(true)` | `with_dct32_keep_hint(Option<bool>)` | DCT32 search policy | Decouples `try_dct32` from `try_dct64` so caller can drop DCT64 (W44-65) but keep DCT32 evaluation for codec_wiki-class smooth screen content. |

### 2.3 Cost-model constant divergences (Section C)

All AT PARITY today *except* the W44-109 constants. Listed for completeness so
`CjxlMirror` can confirm strict parity.

| Constant | Ours | libjxl | Currently divergent? | Bound to |
|---|---|---|---|---|
| `entropy_mul[*]` reference table | per Section C | per Section C | At parity | (table swap is gated by Section 2.2 above) |
| `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E5_E6` | 2.0 | n/a | YES | W44-109 chain gate |
| `DEFAULT_ADAPTIVE_QUANT_SCREENSHOT_QF_SEED_SCALE_E7` | 3.0 | n/a | YES | W44-109 chain gate |
| `BUTTLOOP_QF_SEED_SCALE` | 4.0 | n/a | YES | W44-105 chain gate |
| `W44_120_EPF_SEED_MIN_DISTANCE` | 1.0 | n/a | YES | W44-117/118/120 gate |
| EPF Pass-1 indexing, AdjustQuantBlockAC, K_AC_QUANT, global_scale, kPow, K_INFO_LOSS_MUL, etc. | match | match | At parity | (n/a) |

### 2.4 Algorithm-choice divergences (Section D)

| Component | Status | Current API | Mutually-exclusive group | One-sentence |
|---|---|---|---|---|
| DC tree | AT PARITY (LearnTree at e>=4 + WP + per-stream override) | none (effort-driven) | (none) | Fully ported. |
| ANS histogram strategy | AT PARITY | none | (none) | Fully ported. |
| TryMergeAcs(DCT64X32) non-aligned pass | AT PARITY | none | (none) | Fully ported. |
| BlockCtxMap 15-cluster default | KNOWN-BUG (DISABLED at default) | none | (none) | Upstream `cluster_histograms` produces different histograms; the 15-cluster path regresses. Issue #59. |
| Modular tree learning fallback (4 unported TreeKinds) | INTENTIONAL (LOW EV) | none | (none) | Not implemented; only fires at `decoding_speed_tier >= 1`. |
| DCT64 selection on smooth screen content | INTENTIONAL (W44-65 gate) | `with_dct_suppress_hint` | DCT64 search policy | See Section 2.2 W44-65/68 row. |
| DCT32 family search on W44-65 screen-class | INTENTIONAL (W44-68 default, W44-123 opt-in to decouple) | `with_dct32_keep_hint` | DCT32 search policy | See Section 2.2 W44-123 row. |
| Buttloop internal recon vs decoder pipeline | INTENTIONAL on photos; CLOSED on screenshots d>=1.0 (W44-117/118/120) | env-var only | EPF sharpness seed for buttloop | See Section 2.2 W44-117/118/120 row. |

### 2.5 Per-API opt-in surface today (Section E)

Direct map of every `with_*_hint` / `with_*_dispatch` on `LossyConfig`. This is
the surface that the new enum will SUBSUME.

| API | Default | Caller can pick | Mutually-exclusive group |
|---|---|---|---|
| `with_content_aware_entropy_mul(bool)` | `false` | `true`/`false` | Screenshot entropy-mul gate enable |
| `with_screenshot_lift_hint(Option<bool>)` | `None` (auto via mask1x1 if `content_aware_entropy_mul`) | `Some(true)` force on, `Some(false)` force off | Screenshot entropy-mul override |
| `with_high_d_photo_hint(Option<bool>)` | `None` (auto via mask1x1 + distance) | `Some(true)` / `Some(false)` | High-d photo entropy-mul override |
| `with_smooth_photo_dct64_hint(Option<bool>)` | `None` (auto via RGB classifier) | `Some(true)` / `Some(false)` | Smooth-photo DCT64 admission override |
| `with_dct_suppress_hint(Option<bool>)` | `None` (auto via mask1x1 > 95) | `Some(true)` / `Some(false)` | DCT64 search policy override |
| `with_dct32_keep_hint(Option<bool>)` | `None` (follow W44-68 drop) | `Some(true)` / `Some(false)` | DCT32 search policy override (composes with `with_dct_suppress_hint`) |
| `with_pixel_loss_dispatch(PixelLossDispatch)` | `AlwaysOn` | `AlwaysOff`, `Auto` | Pixel-domain loss policy |
| `with_single_pass_entropy_dispatch(SinglePassEntropyDispatch)` | `AlwaysTwoPass` | `AlwaysSinglePass`, `Auto` | Two-pass entropy policy |
| `with_epf_dispatch(EpfDispatch)` | `AlwaysSelect` | `AlwaysDefault`, `Auto` | Per-block EPF sharpness search policy |
| `with_patches_dispatch(PatchesDispatch)` | `Auto` | `AlwaysScan`, `NeverScan` | Patches detector dispatch |
| `with_content_class(Option<ImageContentClass>)` | `None` | caller-supplied content class | Effort profile content adaptation |

Two env-var-only knobs not yet on the public API:

| Env var | Default | Effect |
|---|---|---|
| `JXL_W44_117_DISABLE=1` | unset | Forces legacy uniform-4 EPF sharpness seed in buttloop |
| `JXL_W44_120_EPF_SEED_MIN_DISTANCE=<f32>` | 1.0 | Overrides the W44-117 distance gate |
| `JXL_BUTTLOOP_INITIAL_QF_SCALE=<f32>` | 4.0 | Overrides the W44-105 4× seed scale |
| `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE=<f32>` | 2.0/3.0 (per effort) | Overrides the W44-109 pre-scale |

These should become first-class API knobs as part of this work (one of the open
questions in §7).

---

## 3. Categorisation

Six logical groups emerge from the table:

1. **Screenshot-class improvements** (lift entropy_mul, suppress DCT64,
   buttloop seed scale, EPF sharpness one-shot seed, W44-109 adaptive-quant
   pre-scale, W44-123 keep-DCT32 opt-in).
   - W22-1 lift
   - W44-65/68 DCT64 suppression
   - W44-105/107/108 BUTTLOOP_QF_SEED_SCALE
   - W44-109 adaptive-quant qf pre-scale
   - W44-117/118/120 EPF sharpness one-shot seed
   - W44-123 dct32_keep_hint

2. **Photo-class improvements** (entropy-mul lowering on high-d smooth photos,
   plus the zenanalyze-narrowed sub-class gates).
   - W44-29 high_d_photo_smooth_suppressed
   - W44-91 zenanalyze sub-gate (1189261)
   - W44-96 _z sub-gate (1420710 / 1531677)
   - W44-98 _z_high_colour (1420710)
   - W44-99/100 _z_low_colour (1531677)
   - W44-35/34 smooth-photo DCT64 admission

3. **Dispatch / perf policies** (orthogonal to the above; control when to
   run/skip expensive stages without changing what's emitted when they DO
   run).
   - `EpfDispatch`
   - `PixelLossDispatch`
   - `SinglePassEntropyDispatch`
   - `PatchesDispatch`

4. **Effort-gate divergences** (single-direction shifts of the effort
   threshold at which a feature engages).
   - `cfl_two_pass` (we e7+, libjxl e5+)
   - `try_dct64` effort gate (we e7+, libjxl no gate)
   - `epf_dynamic_sharpness` effort gate (we e6+, libjxl no gate)

5. **Algorithm / strategy choices** (where the picks are mutually exclusive
   and would benefit from a nested enum).
   - **EPF sharpness seed** for buttloop:
     `{ LegacyUniform4, W44_117_OneShot, Disabled }`
   - **Buttloop qf seeding**: `{ Off, FixedScale(f32), Auto }`
   - **Adaptive-quant qf pre-scale** (W44-109): `{ Off, Auto, FixedScale(f32 per effort) }`

6. **Effort-profile dispatch / content adaptation** (`with_content_class`,
   `adapt_to_image_content`).

The new `EncoderCompatibility` enum has to cover groups 1, 2, 4, 5 (the
"divergence" surface). Groups 3 and 6 are perf knobs / dispatch knobs and stay
as separate setters — they don't affect the bitstream in `CjxlMirror` mode
because Auto modes default to libjxl-faithful output, the variants are gated
by the underlying enable bit, and the perf knobs compose cleanly with
compatibility presets. (See open question #2 — there's a case for absorbing
group 3 too.)

---

## 4. Proposed enum structure

### 4.1 Top-level

```rust
/// Encoder behaviour bundle controlling which of our W44-* improvements over
/// libjxl reference are active.
///
/// **Default**: [`EncoderCompatibility::Default`] — the production bundle
/// we ship today. Equivalent to leaving every `with_*_hint` setter at its
/// current default value.
///
/// Set via [`LossyConfig::with_compatibility`]. Individual
/// `LossyConfig::with_*_hint` / `with_*_dispatch` setters called AFTER
/// `with_compatibility` override the matching field on the resolved
/// `EncoderImprovements`; this mirrors the
/// `with_perceptual_optimizations(false).with_gaborish(true)` precedence
/// pattern (see [`LossyConfig::with_perceptual_optimizations`]).
#[derive(Clone, Debug, PartialEq)]
pub enum EncoderCompatibility {
    /// **Strict libjxl-parity mode.** Disables every W44-* improvement that
    /// causes a bitstream divergence vs libjxl reference. Verified against
    /// `cjxl` output as a regression gate.
    ///
    /// Use cases:
    /// - Conformance testing — does our decoder agree with cjxl on our
    ///   output?
    /// - libjxl bitstream comparison — A/B against libjxl byte-for-byte.
    /// - Cardinal-rule "leave nothing unported" verification baseline.
    ///
    /// Behaviour:
    /// - All Section B content-aware gates: off
    /// - W44-105/107/108 buttloop qf seed scale: 1.0 (no scaling)
    /// - W44-109 adaptive-quant pre-scale: 1.0
    /// - W44-117/118/120 EPF sharpness one-shot seed: uniform-4 (legacy)
    /// - W44-65/68 DCT64 suppression: `try_dct64` per libjxl-effort gate
    /// - W44-123 dct32_keep_hint: no-op (W44-65 doesn't fire)
    /// - All `with_*_hint(Option<bool>)` setters: `None`, with the auto
    ///   gates inside the encoder pinned to OFF
    ///
    /// NOTE: this does NOT close ALL divergences — Section A effort-gate
    /// divergences (`cfl_two_pass` e7 vs libjxl e5, `try_dct64` effort
    /// gate) and Section D `BlockCtxMap 15-cluster default` KNOWN-BUG
    /// remain; those need separate code-level ports. `CjxlMirror`
    /// closes every divergence the API CAN close.
    CjxlMirror,

    /// **Conservative.** Enables the *photo-class* algorithm improvements
    /// (W44-29/91/96/98/99/100 entropy-mul lowering, W44-34/35 smooth-photo
    /// DCT64 admission) AND the ported algorithm fixes that are at-parity
    /// today, but DISABLES the screenshot-class lifts and the buttloop /
    /// EPF-seed corrections.
    ///
    /// Behaviour:
    /// - Photo-class gates: auto (default for those gates)
    /// - Screenshot-class gates: off (W22-1, W44-65/68, W44-105 chain,
    ///   W44-109 chain, W44-117/118/120 chain, W44-123)
    /// - Effort-gate divergences: ours (not libjxl)
    Conservative,

    /// **Default.** What we ship today. Equivalent to the all-Nones
    /// behaviour of the `with_*_hint(Option<bool>)` setters, which routes
    /// through the per-image auto discriminators.
    ///
    /// Behaviour:
    /// - Every Section B gate: auto (mask1x1 / zenanalyze discriminators
    ///   fire as documented)
    /// - W44-105/107/108 buttloop chain: 4.0 / fires per gate
    /// - W44-109 adaptive-quant chain: 2.0/3.0 / fires per gate
    /// - W44-117/118/120 EPF chain: fires on is_screenshot at d>=1.0
    /// - W44-65/68 DCT64 suppression: auto via mask1x1 > 95
    /// - W44-123 dct32_keep_hint: None (defer to W44-68 drop)
    Default,

    /// **Aggressive.** All the Default behaviours PLUS opt-in to:
    /// - W44-123 DCT32 keep on codec_wiki-class screenshots
    ///   (`dct32_keep_hint = Some(true)`).
    /// - Future opt-in improvements as they ship.
    ///
    /// Caller is expected to be using a richer content classifier
    /// (zenanalyze features) to avoid the W44-123 graph/imessage
    /// regression.
    Aggressive,

    /// **Custom.** Caller picks every dial individually. See
    /// [`EncoderImprovementsCustom`].
    Custom(Box<EncoderImprovementsCustom>),
}

impl Default for EncoderCompatibility {
    fn default() -> Self {
        Self::Default
    }
}
```

### 4.2 The `Custom` payload — fine-grained struct

The custom variant uses a struct of nested enums. Mutually-exclusive picks
become nested enums; orthogonal booleans stay as `Option<bool>` (None = auto
discriminator). This is the minimum surface that prevents invalid
configurations at the type level.

```rust
/// Fine-grained per-divergence picks. Use with
/// [`EncoderCompatibility::Custom`] when none of the named presets fit.
///
/// Every field has a `Default` impl that matches
/// [`EncoderCompatibility::Default`]. Construct via
/// `EncoderImprovementsCustom::default().with_*` builders for a fluent
/// experience.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderImprovementsCustom {
    // ── Screenshot-class entropy-mul lifts ─────────────────────────
    /// W22-1 screenshot lift table (lifts IDENTITY/DCT2X2/AFV/DCT4X8).
    pub screenshot_entropy_mul: ScreenshotEntropyMulPolicy,

    // ── Photo-class entropy-mul lowering ───────────────────────────
    /// W44-29 + nested sub-gates (W44-91/96/98/99/100).
    pub high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy,

    // ── DCT-class search admission ─────────────────────────────────
    /// W44-65/68 DCT64-class suppression on screenshot content.
    pub dct64_search_policy: Dct64SearchPolicy,
    /// W44-123 DCT32-class search retention (only matters when
    /// `dct64_search_policy` would otherwise drop the DCT32 family
    /// together).
    pub dct32_search_policy: Dct32SearchPolicy,
    /// W44-34/35 smooth-photo DCT64 admission (orthogonal to dct64_search_policy
    /// above — that one is screenshot suppression; this one is photo admission
    /// inside the small-image-pixel gate).
    pub smooth_photo_dct64_admission: SmoothPhotoDct64Policy,

    // ── Butteraugli loop qf seeding (e>=8) ─────────────────────────
    /// W44-105/107/108 — pre-scale the buttloop's initial qf seed on
    /// screenshot-class content at high d.
    pub buttloop_qf_seed: ButtloopQfSeedPolicy,

    // ── Adaptive-quant qf seeding (e ∈ [5,7]) ──────────────────────
    /// W44-109 — mirror of W44-105 at lower effort where buttloop is
    /// unavailable.
    pub adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy,

    // ── EPF sharpness seed for buttloop ────────────────────────────
    /// W44-117/118/120 — one-shot `compute_epf_sharpness` seed.
    pub buttloop_epf_sharpness_seed: EpfSharpnessSeed,

    // ── Future expansion ───────────────────────────────────────────
    // Non-exhaustive — new divergences add fields here. The derive of
    // `Default` keeps existing callers source-compatible across versions
    // (within a 0.x major bump).
}
```

### 4.3 Nested enums (mutually exclusive picks)

```rust
/// W22-1 screenshot entropy-mul lift.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScreenshotEntropyMulPolicy {
    /// **Default in [`EncoderCompatibility::Default`].** Auto-fire via
    /// `median(mask1x1) > 95` on screenshot-class content.
    #[default]
    Auto,
    /// Force the lift on regardless of content. Caller asserts the image
    /// is screenshot-class.
    ForceOn,
    /// Suppress the lift even when mask1x1 would fire it. Equivalent to
    /// the W22-1 `Some(false)` override.
    ForceOff,
    /// Disable the gate entirely (the `content_aware_entropy_mul` bit is
    /// false). [`EncoderCompatibility::CjxlMirror`] uses this.
    Disabled,
}

/// W44-29 + nested sub-gates (W44-91 / W44-96 / W44-98 / W44-99 / W44-100).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HighDPhotoEntropyMulPolicy {
    /// **Default.** Auto-fire via `d >= 4.0 AND mask1x1 < SMOOTH_THRESHOLD`
    /// with the W44-91/96/98/99/100 zenanalyze sub-discriminators
    /// composing on top.
    #[default]
    Auto,
    /// Force the lowering on regardless of content/distance.
    ForceOn,
    /// Suppress the lowering even when the auto gate would fire.
    ForceOff,
    /// Disable the gate entirely.
    Disabled,
}

/// W44-65/68 DCT64-class search admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dct64SearchPolicy {
    /// **Default.** Auto-suppress via `median(mask1x1) >= 99.5` on
    /// screenshot-class.
    #[default]
    Auto,
    /// Force-suppress regardless of content. Equivalent to
    /// `with_dct_suppress_hint(Some(true))`.
    ForceSuppress,
    /// Force-allow DCT64 evaluation everywhere. Equivalent to
    /// `with_dct_suppress_hint(Some(false))`. CjxlMirror uses this.
    ForceAllow,
}

/// W44-123 DCT32-class search retention. Composes with [`Dct64SearchPolicy`]:
/// only matters when DCT64 has been suppressed (auto or forced) AND the
/// underlying W44-68 default would also drop `try_dct32`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dct32SearchPolicy {
    /// **Default.** Follow W44-68 (`try_dct32` dropped together with
    /// `try_dct64` when W44-65 fires).
    #[default]
    FollowDct64Suppression,
    /// When DCT64 is suppressed (W44-65 fires), KEEP `try_dct32 = true`.
    /// Useful on codec_wiki-class smooth screen content where
    /// DCT16X16 → DCT32X32 splitting is the dominant win.
    KeepWhenDct64Suppressed,
}

/// W44-34/35 smooth-photo DCT64 admission inside the
/// `pixels < 500_000 AND distance < 2.0` gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SmoothPhotoDct64Policy {
    /// **Default.** Auto-admit via the smooth-photo classifier (edge
    /// density + flat block ratio + HF energy).
    #[default]
    Auto,
    /// Force-admit on the gated cell.
    ForceAdmit,
    /// Force-skip the admission (preserves pre-W44-35 behaviour).
    /// CjxlMirror uses this.
    ForceSkip,
}

/// W44-105/107/108 buttloop qf seed scaling (e>=8).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd)]
pub enum ButtloopQfSeedPolicy {
    /// **Default.** Auto-fire via the W44-105/107/108 gate
    /// (`is_screenshot AND (d >= 3.5 OR (m3 < 30 AND d >= 2.0))`),
    /// scale = 4.0.
    #[default]
    AutoScale4,
    /// Custom scale (replaces the 4.0 default but keeps the same gate
    /// predicate). 1.0 == off.
    AutoScale(f32),
    /// Force-fire the scale on every encode at given factor.
    ForceScale(f32),
    /// Off — never scale (`scale == 1.0`). CjxlMirror uses this.
    Off,
}

/// W44-109 adaptive-quant qf pre-scale (e ∈ [5,7]).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum AdaptiveQuantQfSeedPolicy {
    /// **Default.** Auto-fire on screenshot-class content at e ∈ [5,7]
    /// with the per-effort scales (2.0 at e5/e6, 3.0 at e7).
    #[default]
    AutoScalePerEffort,
    /// Custom per-effort scales (replaces the 2.0/3.0 defaults).
    AutoScaleCustom { e5_e6: f32, e7: f32 },
    /// Off — never pre-scale. CjxlMirror uses this.
    Off,
}

/// W44-117/118/120 EPF sharpness seed for buttloop.
///
/// Models the buttloop's internal `apply_epf` sharpness map source.
/// Mutually exclusive — exactly one of the three picks. This is the
/// canonical example of a nested-enum mutually-exclusive group: the
/// `Option<bool>` shape we ship today admits invalid states like
/// "force_seed AND force_uniform4 AND per_iter_recompute" — the enum
/// shape makes those unrepresentable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum EpfSharpnessSeed {
    /// **Default.** W44-117 one-shot `compute_epf_sharpness` on the
    /// initial reconstruction, with the W44-118 `is_screenshot` gate
    /// AND W44-120 `target_distance >= 1.0` gate. Falls back to
    /// `LegacyUniform4` on photos and on screenshots at d < 1.0.
    #[default]
    AutoW44_117 { min_distance: f32 }, // default { min_distance: 1.0 }
    /// Pre-W44-117 behaviour: uniform sharpness = 4 across the whole
    /// frame inside the buttloop. CjxlMirror uses this.
    LegacyUniform4,
    /// Future-shape pick — recompute `compute_epf_sharpness` per buttloop
    /// iter. Bench so far shows this regresses (Mode D in W44-118
    /// bisect); reserved for future investigation.
    #[doc(hidden)]
    PerIterRecompute,
}
```

### 4.4 Resolution path — `pub(crate)` methods

Each call site that today reads a `with_*_hint` field instead reads from a
resolved struct. The resolution happens once per encode, at the boundary
between `LossyConfig` and the internal `VarDctEncoder`.

```rust
/// Fully-resolved per-divergence flags consumed by the internal encoder.
///
/// Built once per encode by [`EncoderCompatibility::resolve`] from the
/// caller-supplied compatibility variant + any individual `with_*_hint`
/// setters that override the preset.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedImprovements {
    pub(crate) screenshot_entropy_mul: ScreenshotEntropyMulPolicy,
    pub(crate) high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy,
    pub(crate) dct64_search_policy: Dct64SearchPolicy,
    pub(crate) dct32_search_policy: Dct32SearchPolicy,
    pub(crate) smooth_photo_dct64_admission: SmoothPhotoDct64Policy,
    pub(crate) buttloop_qf_seed: ButtloopQfSeedPolicy,
    pub(crate) adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy,
    pub(crate) buttloop_epf_sharpness_seed: EpfSharpnessSeed,
}

impl EncoderCompatibility {
    /// Resolve to the internal per-divergence flag struct.
    ///
    /// `overrides` carries any individual `with_*_hint` calls the caller
    /// made AFTER `with_compatibility` — those win field-by-field, mirroring
    /// the `with_perceptual_optimizations` precedence pattern.
    pub(crate) fn resolve(
        &self,
        overrides: &CompatibilityOverrides,
    ) -> ResolvedImprovements {
        let base = match self {
            Self::CjxlMirror => ResolvedImprovements::cjxl_mirror(),
            Self::Conservative => ResolvedImprovements::conservative(),
            Self::Default => ResolvedImprovements::default_bundle(),
            Self::Aggressive => ResolvedImprovements::aggressive(),
            Self::Custom(c) => ResolvedImprovements::from_custom(c),
        };
        overrides.apply_to(base)
    }
}

impl ResolvedImprovements {
    fn cjxl_mirror() -> Self {
        Self {
            screenshot_entropy_mul: ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy::Disabled,
            dct64_search_policy: Dct64SearchPolicy::ForceAllow,
            dct32_search_policy: Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission: SmoothPhotoDct64Policy::ForceSkip,
            buttloop_qf_seed: ButtloopQfSeedPolicy::Off,
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::Off,
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::LegacyUniform4,
        }
    }
    fn conservative() -> Self {
        Self {
            screenshot_entropy_mul: ScreenshotEntropyMulPolicy::Disabled,
            high_d_photo_entropy_mul: HighDPhotoEntropyMulPolicy::Auto,
            dct64_search_policy: Dct64SearchPolicy::ForceAllow,
            dct32_search_policy: Dct32SearchPolicy::FollowDct64Suppression,
            smooth_photo_dct64_admission: SmoothPhotoDct64Policy::Auto,
            buttloop_qf_seed: ButtloopQfSeedPolicy::Off,
            adaptive_quant_qf_seed: AdaptiveQuantQfSeedPolicy::Off,
            buttloop_epf_sharpness_seed: EpfSharpnessSeed::LegacyUniform4,
        }
    }
    fn default_bundle() -> Self { /* every field at its enum's #[default] */ Default::default() }
    fn aggressive() -> Self {
        let mut r = Self::default_bundle();
        r.dct32_search_policy = Dct32SearchPolicy::KeepWhenDct64Suppressed;
        r
    }
    fn from_custom(c: &EncoderImprovementsCustom) -> Self {
        Self {
            screenshot_entropy_mul: c.screenshot_entropy_mul,
            high_d_photo_entropy_mul: c.high_d_photo_entropy_mul,
            dct64_search_policy: c.dct64_search_policy,
            dct32_search_policy: c.dct32_search_policy,
            smooth_photo_dct64_admission: c.smooth_photo_dct64_admission,
            buttloop_qf_seed: c.buttloop_qf_seed,
            adaptive_quant_qf_seed: c.adaptive_quant_qf_seed,
            buttloop_epf_sharpness_seed: c.buttloop_epf_sharpness_seed,
        }
    }
}

/// Per-field overrides set via the existing `with_*_hint` setters AFTER
/// `with_compatibility` is called. Field-by-field precedence over the
/// compatibility preset's resolved value.
pub(crate) struct CompatibilityOverrides {
    pub(crate) screenshot_lift_hint: Option<bool>,
    pub(crate) high_d_photo_hint: Option<bool>,
    pub(crate) smooth_photo_dct64_hint: Option<bool>,
    pub(crate) dct_suppress_hint: Option<bool>,
    pub(crate) dct32_keep_hint: Option<bool>,
    // Any future `with_*_hint` adds a field here.
}
```

Call-site read pattern (consumer):

```rust
// Existing call-site (vardct/encoder.rs:2985):
//   let w44_65_suppress_dct64 = match self.dct_suppress_hint { ... };
//
// New call-site (resolved once at enc construction):
//   let w44_65_suppress_dct64 = match self.improvements.dct64_search_policy {
//       Dct64SearchPolicy::Auto         => mask1x1_median >= 99.5,
//       Dct64SearchPolicy::ForceSuppress => true,
//       Dct64SearchPolicy::ForceAllow    => false,
//   };
```

The resolver runs once before AC strategy selection. Internal fields stay as
the fully-resolved enum picks — the call sites become a single `match` on each
pick rather than chained `if/else if` ladders over `Option<bool>` values, and
mutually-exclusive states are unrepresentable.

### 4.5 New `LossyConfig` setter

```rust
impl LossyConfig {
    /// Set the encoder compatibility bundle. Default
    /// [`EncoderCompatibility::Default`] reproduces what we ship today.
    ///
    /// Individual `with_*_hint` / `with_*_dispatch` setters called AFTER
    /// this one override the matching field on the resolved
    /// [`ResolvedImprovements`] (mirrors the
    /// [`Self::with_perceptual_optimizations`] precedence pattern).
    pub fn with_compatibility(mut self, c: EncoderCompatibility) -> Self {
        self.compatibility = c;
        self
    }
    pub fn compatibility(&self) -> &EncoderCompatibility { &self.compatibility }
}
```

---

## 5. Migration path

No external users (per CLAUDE.md), so we DELETE rather than deprecate. The
`with_*_hint` setters stay (as a low-level escape hatch and for the
"override after `with_compatibility`" pattern), but most callers should
migrate to `with_compatibility`.

| Old API call | New equivalent |
|---|---|
| (no call — defaults) | (no call — `EncoderCompatibility::Default`) |
| `with_content_aware_entropy_mul(true)` | `with_compatibility(EncoderCompatibility::Default)` (already on by default in Default bundle? — see open question #3) |
| `with_content_aware_entropy_mul(false)` | `with_compatibility(EncoderCompatibility::Conservative)` OR `with_compatibility(EncoderCompatibility::CjxlMirror)` |
| `with_dct_suppress_hint(Some(true))` | `EncoderCompatibility::Custom({ dct64_search_policy: ForceSuppress, ... })` |
| `with_dct_suppress_hint(Some(false))` | `EncoderCompatibility::CjxlMirror` (or override `dct64_search_policy: ForceAllow` in Custom) |
| `with_dct32_keep_hint(Some(true))` | `EncoderCompatibility::Aggressive` |

Env vars `JXL_W44_117_DISABLE`, `JXL_W44_120_EPF_SEED_MIN_DISTANCE`,
`JXL_BUTTLOOP_INITIAL_QF_SCALE`, `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE`: KEEP
as harness-sweep overrides (they sit at the bottom of the resolution stack,
below the API).

---

## 6. Call-site impact estimate

Read sites for the resolved struct (touched in implementation, NOT this design
chunk):

| File | LOC change | Sites |
|---|---|---|
| `jxl-encoder/src/api.rs` | +400 (new enum definitions, Default impls, builder), -50 (rewire `with_*_hint` setters to update overrides struct) | enum defs, `LossyConfig::with_compatibility`, `resolve()` |
| `jxl-encoder/src/vardct/encoder.rs` | ~10 sites changed (~50 LOC), each replacing an `Option<bool>` match with a 3-arm enum match | `w44_65_suppress_dct64`, `w22_1_lift`, `w44_29_lift`, `w44_123_keep_dct32`, smooth-photo DCT64, etc. |
| `jxl-encoder/src/vardct/butteraugli_loop.rs` | ~3 sites, ~40 LOC | EPF sharpness seed, buttloop qf seed scale |
| `jxl-encoder/src/vardct/bitstream.rs` | ~2 sites, ~20 LOC | EPF dispatch routing |
| `jxl-encoder/src/effort.rs` | ~5 LOC | EntropyMulTable resolve (table-swap entry point) |
| `jxl-encoder-cli/src/main.rs` | ~30 LOC | new `--compatibility cjxl-mirror|conservative|default|aggressive` flag |
| `jxl-encoder/src/vardct/encoder.rs` tests | ~20 LOC | rewrite existing `with_*_hint` test cases against `with_compatibility` |

Total: ~570 LOC across 6 files. Implementation is straightforward — the
hard part was already done in the W44-* arc (the divergence inventory and
the per-site auto-gate predicates).

---

## 7. Open questions for user

Before any implementation chunk:

1. **Variant naming.** `Default` shadows `std::default::Default`. Options:
   - Keep `EncoderCompatibility::Default` (allowed but slightly confusing).
   - Rename to `Production`, `Shipping`, `Recommended`, `Standard`?
   - **Recommendation**: `EncoderCompatibility::Production` (signals "what
     we ship") and reserve `Default` for the `impl Default`.

2. **Should perf dispatches absorb into the enum?** Today
   `EpfDispatch`/`PixelLossDispatch`/`SinglePassEntropyDispatch`/
   `PatchesDispatch` are separate setters and don't change bitstream output
   on the `Auto` setting beyond what the underlying gate enables. They're
   speed knobs, not divergence knobs. Should they be moved INTO the
   `Custom` payload as an `EncoderPerfDispatch` sub-struct, OR stay as
   independent setters? Trade-off:
   - **Absorb**: one config struct to rule them all; perf and quality
     bundles compose cleanly (e.g. "Conservative bundle + AlwaysOff pixel
     loss for max speed").
   - **Keep separate**: perf knobs are conceptually orthogonal to the
     parity-vs-improvement bundle. A caller wanting `CjxlMirror` may still
     want `EpfDispatch::Auto` for speed.
   - **Recommendation**: keep separate. They're not divergence-class.

3. **Where does `Default` sit?** Today `with_content_aware_entropy_mul`
   defaults to `false` — the W22-1 lift is opt-in. But many other
   divergences (W44-65, W44-105 chain, W44-109 chain, W44-117 chain) are
   default-on. So `EncoderCompatibility::Default` should mirror "all the
   stuff we ship by default" which includes some opt-out and some opt-in.
   Question: do we flip `content_aware_entropy_mul` to default-on in the
   Default bundle, or keep it opt-in?
   - **Recommendation**: keep it opt-in for backwards compat with existing
     `Default` hash-locks. `Aggressive` is the variant that flips it on.

4. **Should `Conservative` enable `epf_dynamic_sharpness` auto-skip?**
   That's a Section A effort-gate divergence (we e6+, libjxl no gate). It's
   a SUPERSET of libjxl (skipping work libjxl does), not a behaviour
   divergence. Trade-off: leaving it on in `Conservative` saves CPU but
   diverges from libjxl effort gating semantics.
   - **Recommendation**: leave on in `Conservative` (it's a perf-only
     superset); turn off in `CjxlMirror`.

5. **`EpfSharpnessSeed::AutoW44_117 { min_distance: f32 }`** — should this
   be a free-form `f32` or a presets enum? The W44-120 bisection swept
   {0.8, 1.0, 1.2, 1.5} and picked 1.0 as pareto-optimal. Future tuning
   may want {0.5, 0.7, 0.9, 1.1}. Free-form is more flexible.
   - **Recommendation**: free-form `f32` with a `DEFAULT_MIN_DISTANCE`
     constant.

6. **Should `with_*_hint` setters update a `CompatibilityOverrides`
   struct, or update fields directly on `LossyConfig`?** The current code
   stores each hint as `Option<bool>` on `LossyConfig` and the consumer
   reads it directly. Trade-off:
   - **Update overrides struct**: cleaner separation, all overrides live
     in one place, easy to query "did the caller override anything?".
   - **Direct fields**: less invasive change, existing tests keep working.
   - **Recommendation**: update overrides struct — the cleaner shape is
     worth the test rewrites.

7. **CLI flag shape.** `--compatibility cjxl-mirror` is the obvious one.
   For `Custom`, callers would set individual `--*-hint` flags. Question:
   are there enough callers wanting hand-tuned Custom configs at the CLI
   level to justify the surface area, or is `Custom` API-only and the CLI
   only exposes the named variants?
   - **Recommendation**: CLI exposes only the four named variants;
     `Custom` is API-only (it's a power-user knob and there's no clean
     CLI grammar for nested enums).

8. **`CjxlMirror` scope.** The doc says "closes every divergence the API
   CAN close" but leaves Section A effort-gate divergences and the
   KNOWN-BUG `BlockCtxMap 15-cluster default` as out-of-scope. Should
   `CjxlMirror` ALSO flip those? They require code-level ports (not just
   API knob changes); they should be tracked as separate chunks.
   - **Recommendation**: scope as designed — `CjxlMirror` covers the
     API-reachable divergences only. File issues for the rest.

9. **Section 2.5 env-var-only knobs.** Should `JXL_W44_117_DISABLE`,
   `JXL_W44_120_EPF_SEED_MIN_DISTANCE`, `JXL_BUTTLOOP_INITIAL_QF_SCALE`,
   `JXL_W44_109_ADAPTIVE_QUANT_QF_SCALE` become first-class API knobs as
   part of this work, OR stay env-var-only?
   - **Recommendation**: promote to API. The `EncoderImprovementsCustom`
     struct already has the slot — wire them through. Keep env-var
     fallback for harness sweeps.

---

## 8. Test plan

1. **Unit**: `EncoderCompatibility::resolve()` for each named variant
   returns the expected `ResolvedImprovements` (table-driven test, one row
   per variant × one field per row).
2. **Unit**: per-field override precedence — `with_compatibility(X)
   .with_dct_suppress_hint(Some(...))` produces the expected resolved
   value.
3. **Hash-lock**: `EncoderCompatibility::Default` produces BYTE-IDENTICAL
   output to today's pre-W44-125 `LossyConfig::new()` on every hash-lock
   fixture (36/36).
4. **Hash-lock**: `EncoderCompatibility::CjxlMirror` produces a different
   bitstream than `Default` on screenshot-class inputs (we expect divergence
   — pin the CjxlMirror outputs as new hash-locks for regression).
5. **Roundtrip**: `CjxlMirror` output decodes successfully via jxl-rs +
   djxl + jxl-oxide on 8 CLIC photos + 10 GB82-SC screenshots.
6. **A/B sweep**: `CjxlMirror` vs `cjxl --effort 7 --distance 4` on the
   cjxl-parity ledger images — verify total byte/SSIM2/butteraugli deltas
   are within MEASUREMENT NOISE on the cells that should be at parity
   (i.e. excluding Section A effort-gate-driven divergences and the
   KNOWN-BUG cells).

---

## 9. Implementation chunk plan (after design approval)

Out of scope for THIS doc — listed for forward planning only:

1. **Chunk A**: Land the type definitions in `api.rs` behind
   `__expert` feature flag. No `LossyConfig` field, no call-site wiring.
   ~200 LOC. Test: type definitions compile and `Default` impls
   round-trip.
2. **Chunk B**: Add `LossyConfig::compatibility` field +
   `with_compatibility` setter. Wire `resolve()` into encoder
   construction. Internal fields still drive the call sites. ~50 LOC.
3. **Chunk C**: Rewire one call site at a time (W44-65 first as it's
   the most-touched), replacing `Option<bool>` matches with
   `Dct64SearchPolicy` matches. One commit per call site, ~10 LOC each.
   ~10 commits.
4. **Chunk D**: Delete `with_*_hint` fields from `LossyConfig` and move
   them into `CompatibilityOverrides`. Update tests. ~50 LOC.
5. **Chunk E**: Add `--compatibility` CLI flag. ~30 LOC.
6. **Chunk F**: Promote env-var-only knobs to API (per open question
   #9). ~80 LOC.

Total: ~450-600 LOC across ~14 commits. Each chunk is independently
shippable.

---

## 10. References

- `docs/LIBJXL_DIVERGENCES.md` — authoritative divergence inventory
- `jxl-encoder/src/api.rs:3486-3508` — `SinglePassEntropyDispatch` (reference
  for the dispatch enum shape)
- `jxl-encoder/src/api.rs:4826-4838` — `with_perceptual_optimizations`
  (reference for the bundling-setter pattern with override precedence)
- W44-65 memory: `w44_65_dct_suppress_default_on_2026-05-19.md`
- W44-117/118/120 memory: cited in `docs/LIBJXL_DIVERGENCES.md` Section D row
- W44-123 memory: in this doc's git log
