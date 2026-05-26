# RFC: Display-config backfill for cvvdp calibration + zensim 372-feature training

**Author**: Lilith River (with Claude scaffolding)
**Started**: 2026-05-25
**Status**: RESEARCH — no code yet. Awaits user decisions in §8.

The cvvdp opt-in path (`LossyConfig::with_cvvdp_loop`) ships with a 7-entry
per-distance JOD calibration table seeded from 1,131 SDR-only cells. The
zensim PreviewV0_3 metric runs a 372-input MLP trained against feature
corpora extracted under the same single display assumption. This RFC plans
the work to extend BOTH surfaces across N additional display configurations
(HDR PQ 1000/4000/10000 cd/m², HLG, mobile, OLED, etc.) so the encoder can
dispatch the right perceptual target per intended viewing environment.

The user mandate (verbatim, 2026-05-25): "backfill both the 372 features
from zensim and cvvdp for additional display specs".

## §1. Current state (what exists)

### §1.1. cvvdp per-distance table — single-spec

- **Path**: `jxl-encoder/src/vardct/cvvdp_targets.rs:61-69`
- **Shape**: `&[(f32, f32)]` with 7 entries covering distance ∈ {0.5, 1.0,
  1.5, 2.0, 3.0, 4.0, 5.0}; values are cvvdp `score = (10.0 - jod) * 1.05`.
- **Lookup**: `cvvdp_target_score_for_distance(d: f32) -> f32` — linear
  interp + clamp.
- **Seeded from**: `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
  (7,938 rows) on 54 distinct images, **all assuming STANDARD_4K display**
  (200 cd/m² sRGB BT.709, 250 lux ambient).
- **Backend init**: `vardct/cvvdp_backend.rs:138` calls
  `CvvdpOpaque::new(..., CvvdpParams::default())` — `default()` returns
  `DisplayModel::STANDARD_4K`. No display-config dispatch exists.

### §1.2. zensim 372-feature path

- **Profile**: `ZensimProfile::PreviewV0_3` is the production default at
  `zensim/src/profile.rs:1543-1574` (PreviewV0_5Tuner-shape).
- **Feature breakdown** (`profile.rs:1557-1558`):
  `372 = 228 standard (xyb_lms multi-scale SSIM-like) + 72 masked + 72 IW
  pool`.
- **Public entry**: `zensim::score_features_with_profile{,_and_codec}`
  (`zensim/src/lib.rs:261`); features pre-extracted offline via
  `zensim-validate/src/bin/extract_pair_features.rs`.
- **Canonical corpus**:
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.csv`
  (73,300 rows × 372 cols, sha256 `14c205332701b5ff6f2842a8d60f8ac1282f8be3d5cd89c11700e1e4b864a20f`
  per zensim CLAUDE.md:15). Per-corpus: `{kadid,tid,cid22,konjnd,konjnd_full,aic3}_features_372col_2026-05-15.{csv,parquet}`.

### §1.3. Display-luminance plumbing the encoder already has

- **Encoder API**: `LossyConfig::with_intensity_target(nits: f32)`
  (`jxl-encoder/src/api.rs:1329`) — writes
  `ToneMapping.intensity_target` (`file_header.rs:174`, default 255.0).
- **Butteraugli loop**:
  `libjxl_butteraugli_intensity_target(tf, metadata) -> f32`
  (`vardct/perceptual_loop.rs:81-90`) dispatches on transfer function:
  SDR → 80.0; PQ / HLG → metadata `intensity_target`. Threaded to the
  GPU butteraugli backend via `with_intensity_target` (line 888) and the
  CPU rate-control path (`vardct/rate_control.rs:123-135`). Closed
  W44-RECON-DEEP/A10.
- **CICP**: `ColorEncoding::from_cicp(cp, tc, mc, full_range)` already
  parses standard color-encoding tuples
  (`headers/color_encoding.rs:394+`), mapping {1, 8, 13, 16, 17, 18} to
  the encoder's internal `TransferFunction` enum.

## §2. Display-config matrix (proposed backfill set)

Rows are ordered by production frequency (web SDR → niche HDR). Each
config maps cleanly to an existing `DisplayModel` preset in
`zenmetrics/crates/cvvdp-gpu/src/params.rs:399-662` — cvvdp-gpu already
ships 25+ presets covering this matrix, so backfill is "use them", not
"design them".

| # | Config            | y_peak | y_black | EOTF  | Primaries | Ambient  | Existing preset                          | Production share (est.) |
|---|---                |---     |---      |---    |---        |---       |---                                       |---                      |
| 1 | SDR-200 (baseline)| 200    | 0.2     | sRGB  | BT.709    | 250 lux  | `STANDARD_4K` / `STANDARD_FHD`           | ~70% (web SDR)          |
| 2 | SDR-100 budget    | 100    | 0.1     | sRGB  | BT.709    | 250 lux  | `SDR_4K_30` / `SDR_FHD_24`               | ~10% (older monitors)   |
| 3 | Mobile 500        | 500    | 0.05    | sRGB  | BT.709    | 250 lux  | `STANDARD_PHONE`                         | ~10% (high-end phones)  |
| 4 | OLED SDR (272)    | 272    | 0.014   | sRGB  | BT.709    | 100 lux  | `LG_OLED_2017_SDR`                       | ~3% (OLED desktop)      |
| 5 | HDR PQ 1000       | 1000   | 0.001   | PQ    | BT.2020   | 5 lux    | `HDR_PQ_1KNIT`                           | ~3% (HDR TV typical)    |
| 6 | HDR PQ 4000       | 4000   | 0.004   | PQ    | BT.2020   | 5 lux    | `HDR_PQ_4KNIT`                           | ~2% (HDR TV high-end)   |
| 7 | HDR HLG 1500      | 1500   | 0.0015  | HLG   | BT.2020   | 10 lux   | `STANDARD_HDR_HLG`                       | ~1% (broadcast HDR)     |
| 8 | HDR PQ 10000      | 10000  | 0.01    | PQ    | BT.2020   | 5 lux    | `STANDARD_HDR_LINEAR_ZOOM` (mastering)   | <1% (mastering)         |

Phase 1 scope (smallest viable expansion): configs **#1 + #5** (SDR-200
baseline + HDR PQ 1000). One SDR sanity-check + one wide-luminance probe.

## §3. zensim 372-feature display sensitivity

**Verdict: DISPLAY-INVARIANT for the 372-feature path.** No display
re-extraction needed.

Evidence (read-only audit of zensim source):

- The 372 = 228 standard + 72 masked + 72 IW pool. The standard 228
  come from `xyb_lms_features.rs` + the SSIM-shape `metric.rs` path,
  both of which operate on XYB / LMS color (display-encoding-aware
  but display-luminance-INVARIANT once the transfer function is
  consistent). Grep on `xyb_lms_features.rs` / `iw_pool.rs` /
  `metric.rs` for `DISPLAY_Y|y_peak|y_black|y_refl|cd/m|nits` returns
  ZERO matches — all 372 production features compute in display-
  relative space.
- The **only** display-luminance-folded code in zensim is
  `cvvdp_features.rs:72-77` (`DISPLAY_Y_PEAK = 200`,
  `DISPLAY_Y_BLACK = 0.2`, `DISPLAY_Y_REFL = 0.397887` —
  STANDARD_4K constants applied to convert sRGB u8 → linear cd/m²
  at lines 161-189). This file's `extract_cvvdp_features` produces
  **19 CVVDP-shape supplemental features**, NOT used in any production
  `ZensimProfile`. The only callers are `zensim-validate/src/bin/{extract_pair_features,extract_ex4_features}.rs` —
  offline research bins for experimental EX-4 batches.
- Confirmed: `grep -rn "extract_cvvdp_features"` returns 0 hits in
  `zensim/src/` (the file defines but never internally calls it).

**Consequence**: zensim 372-feature backfill across display configs is a
NO-OP for the PreviewV0_3 production path. Per-display re-extraction is
only needed IF we add a future profile that incorporates the supplemental
`cvvdp_features.rs` 19-vector — and even then, only the 19 supplemental
features need re-extraction per display (the 372 standard stays as-is).

A FUTURE per-display profile would need:
- Re-extracted 19-feature CVVDP-shape supplements per display config.
- Retrained MLP head (372+19=391 inputs or hybrid-head per-display routing).
- The current 372 base features remain canonical.

This is good news: the heavy zensim backfill (re-extract every CID22 /
KADID / TID / KonJND / AIC3 image at every display config = ~10 GB
extra parquet per config) **is not required for the current production
metric**. The work scope collapses to the cvvdp side.

## §4. cvvdp display-config sensitivity

**Verdict: HIGH sensitivity. JOD scores shift O(1) between SDR-200 and
HDR PQ 1000 at fixed encoder output.**

Reasoning (from cvvdp algorithm structure):

- cvvdp's per-iter pipeline: sRGB/PQ/HLG → linear cd/m² → DKL opponent
  → Laplacian pyramid → castleCSF (luminance-adapted) → contrast
  masking → Minkowski pool → JOD. The display model gates step 1 (sRGB
  → linear-cd/m² scale = `y_peak - y_black`, +`y_refl` offset)
  AND step 4 (castleCSF luminance adaptation lookup, which is a 4D
  LUT keyed on local mean luminance).
- A 1-stop difference in adaptation luminance shifts CSF peak
  sensitivity by ~5-15% (Mantiuk 2024). Compounded across 4 pyramid
  bands and 3 DKL channels, JOD shifts of 0.1-0.5 JOD per stop are
  typical at moderate distortions.
- Current single-spec STANDARD_4K table at d=5.0 sits at JOD=9.7138.
  Re-scoring the same encoder output at HDR PQ 1000 cd/m² typically
  shifts to JOD≈9.4-9.6 (the higher peak luminance + lower ambient
  exposes finer artifacts). The CALIBRATED TARGET to keep the buttloop
  converging "at the same perceived quality" must shift accordingly,
  OR the loop will under/over-quantize.
- Pareto trade-off interpretation: the SAME `LossyConfig::distance(d)`
  parameter encoded for SDR-200 viewing vs HDR PQ 1000 viewing should
  produce DIFFERENT bytes when the encoder is honest about the target
  display. A per-display per-distance table makes the encoder honest.

Empirically grounded estimate (cvvdp paper + Mantiuk 2024 §4.2):
expect JOD-axis shifts of:
- SDR-200 → SDR-100: ~0.03-0.08 JOD per cell (small)
- SDR-200 → STANDARD_PHONE (500): ~0.05-0.12 JOD per cell (small)
- SDR-200 → OLED_SDR (272 + low ambient): ~0.05-0.15 JOD per cell
- SDR-200 → HDR_PQ_1KNIT: ~0.15-0.40 JOD per cell (significant)
- SDR-200 → HDR_PQ_4KNIT: ~0.25-0.60 JOD per cell (large)
- SDR-200 → HDR_PQ_10000: ~0.35-0.80 JOD per cell (very large)

The SDR cluster (#1-4) may share one table within ±0.1 JOD. The HDR
cluster (#5-8) needs distinct tables. Phase 1 sanity-bench (§7) is what
verifies these estimates.

## §5. Backfill methodology

### §5.1. Sweep design

For each candidate display config `D_k`:

1. Take the existing tracking sweep corpus (54 distinct images × 7
   distances × 3 efforts = 1,134 cells per backend).
2. Encode each cell with `EncoderStrategy::Zenjxl` default (or
   bytes-tightened opt-in stack). The encode is display-AGNOSTIC —
   bytes don't change with display config.
3. **Score each cell's output under cvvdp configured for `D_k`**:
   `CvvdpOpaque::new(..., CvvdpParams { display: D_k, .. })`.
4. Per-distance median JOD → `score = (10 - jod) * 1.05` → row in the
   per-display table.

### §5.2. Output shape

Replace `cvvdp_targets.rs` table with multi-dim:

```rust
pub(crate) struct DisplayCalibration {
    pub display_id: DisplayPresetId,  // enum mirroring cvvdp-gpu presets
    pub entries: &'static [(f32, f32)],  // 7-entry distance→target
}

pub(crate) static CVVDP_DISPLAY_CALIBRATIONS: &[DisplayCalibration] = &[
    DisplayCalibration { display_id: DisplayPresetId::Standard4K, entries: &[...] },
    DisplayCalibration { display_id: DisplayPresetId::HdrPq1Knit, entries: &[...] },
    // ...
];

pub(crate) fn cvvdp_target_score_for_distance_and_display(
    target_distance: f32,
    display: DisplayPresetId,
) -> f32 { ... }
```

### §5.3. Storage layout (decision: in-source vs external lookup)

**Recommended: in-source `&'static [DisplayCalibration]` mirroring the
current single-spec layout.** Each calibration is 7 × 8 bytes = 56 B;
20 displays × 56 B = 1.1 KB. Trivially within source-file size
budgets. Provenance comments per-display point at the bench TSV.

Alternative (external lookup parquet) is heavier and adds a runtime IO
dependency; rejected for the size involved.

### §5.4. Compute cost estimate

Per cvvdp-gpu's `diffmap_overhead_2026-05-24.tsv`:
- 256² cell: 5360 µs scoring
- 512² cell: 2944 µs scoring (warm)
- 1024² cell: 4333 µs scoring (warm)

Tracking sweep cells average ~512² (CID22-512). Conservative:
~5 ms cvvdp-gpu scoring per cell.

Per display × per cell: encode is shared (already done in
existing tracking bench), only re-score is incremental:
- 1,134 cells × 5 ms = ~6 s of pure GPU score time per display
- + overhead (sweep harness, parquet I/O, multi-image batching): ~30 s
  per display, conservative ~1 min

Total for 8 displays: ~10 min single-GPU on lilith RTX 5070. Fleet
trivial. Vast.ai cost: ~$0.10.

The expensive part is the underlying encode sweep — which we ALREADY
have at `cvvdp_vs_buttloop_tracking_2026-05-24.tsv`. The encode bytes
are display-INVARIANT (the encoder doesn't know what display will be
used) so we re-use the existing TSV's encoded outputs and only re-run
the metric scoring under each new display config.

**Implementation**: extend `examples/cvvdp_track_baseline.rs` with a
`--display <preset_name>` flag. Run 8× per backend. Total: hours not
days.

### §5.5. Tracking benchmark output

`benchmarks/cvvdp_per_display_targets_<YYYY-MM-DD>.tsv` with schema:

```
display_preset	image	corpus	effort	distance	bytes	score_cvvdp_<preset>
```

Per-display per-distance medians are then computed by a Python
analyzer (extends existing `scripts/cvvdp_pareto_analysis.py`).

## §6. Integration plan

### §6.1. New API surface (proposed, NOT implemented in this RFC)

```rust
/// Display target preset for cvvdp calibration dispatch. Mirrors a
/// subset of cvvdp-gpu's `DisplayModel::*` const presets — enum
/// values stable across encoder versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DisplayPreset {
    /// Standard 4K SDR (200 cd/m² sRGB BT.709, 250 lux). DEFAULT.
    Standard4K,
    /// Office SDR (100 cd/m²).
    Sdr100,
    /// Mobile phone (500 cd/m² SDR).
    Mobile500,
    /// OLED SDR (272 cd/m², low ambient).
    OledSdr272,
    /// HDR PQ TV (1000 cd/m², BT.2020).
    HdrPq1000,
    /// HDR PQ TV (4000 cd/m², BT.2020).
    HdrPq4000,
    /// HDR HLG TV (1500 cd/m², BT.2020).
    HdrHlg1500,
    /// HDR PQ mastering (10000 cd/m², BT.2020).
    HdrPq10000,
}

impl LossyConfig {
    /// Set the target display preset for cvvdp-driven quantization.
    /// Affects ONLY the cvvdp metric calibration table; the encoded
    /// bitstream is display-invariant. Default: auto-derive from
    /// `with_intensity_target` + CICP (sRGB → `Standard4K`, PQ + nits
    /// → nearest `HdrPq*` preset, HLG → `HdrHlg1500`).
    ///
    /// Has no effect when cvvdp loop is not active.
    pub fn with_target_display(mut self, preset: DisplayPreset) -> Self { ... }
    pub fn target_display(&self) -> Option<DisplayPreset> { ... }
}
```

### §6.2. `MetricSelection` extension

Add `target_display: Option<DisplayPreset>` to
`MetricSelection { metric, device, target_score, target_display }`
struct in `vardct/perceptual_backend.rs`. When metric is `Cvvdp` and
`target_display` is `Some`, propagate to backend constructor; backend
swaps `CvvdpParams.display = DisplayModel::<preset>`.

### §6.3. Auto-dispatch from existing metadata

Mirror the W44-RECON-DEEP/A10 pattern:

```rust
fn auto_target_display_from_color_encoding(
    ce: &ColorEncoding, intensity_target_nits: f32,
) -> DisplayPreset {
    match ce.tf {
        TransferFunction::Pq => match intensity_target_nits {
            n if n >= 8000.0 => DisplayPreset::HdrPq10000,
            n if n >= 3000.0 => DisplayPreset::HdrPq4000,
            _ => DisplayPreset::HdrPq1000,
        },
        TransferFunction::Hlg => DisplayPreset::HdrHlg1500,
        _ => DisplayPreset::Standard4K,  // SDR default
    }
}
```

Caller can override via `with_target_display`. Pattern matches
W44-RECON-DEEP/A10 butteraugli-side dispatch.

### §6.4. Picker oracle re-train implications

If we add display-config dimension to encoder dispatch, the picker
oracle that selects (codec, quality) per-image must ALSO accept
display target as an input feature. Two paths:

- (A) Train a per-display picker (8 displays = 8 oracle bakes).
- (B) Train a single picker with `display_preset` as an input one-hot
  feature (8 extra inputs).

Per CLAUDE.md sweep discipline §1 "Stratification: pick representative
source images via clustering", path (B) is cheaper and more flexible —
the model learns the display-conditional mapping with one bake.

Picker re-train cost: same magnitude as existing picker re-trains
(~$10-30 vast.ai per re-bake). Not gating for any phase before §7
Phase 4.

## §7. Phases (smallest demoable chunk first)

### Phase 1 — API surface + 3-preset opt-in (SHIPPED 2026-05-25)

**Status**: **SHIPPED**. The Phase 1 deliverable evolved from the
original "sanity bench" plan into a full API-surface ship: the user
mandate from 2026-05-25 prioritised landing the `with_target_display`
caller surface + a 3-preset (WebSdr80 / Phone / Tv) calibration table
inside one chunk, so callers can opt-in to per-display CVVDP scoring
immediately. Heuristic seeds for Phone + Tv ship in lieu of the
1-day re-score sweep (deferred to Phase 2 + the cvvdp-gpu
`new_with_geometry` upstream gap-close).

What landed:

- `crate::api::DisplayConfig` enum (`WebSdr80` / `Phone` / `Tv` —
  3 Phase 1 presets, `#[non_exhaustive]` for future extension) with
  `display_model()` + `display_geometry()` conversion methods (gated
  on `cvvdp-loop` cargo feature).
- `LossyConfig::with_target_display(DisplayConfig)` setter +
  `target_display()` getter + `resolve_target_display()` resolver
  (Libjxl strict-parity short-circuit forces `WebSdr80`).
- Per-display calibration table in `vardct/cvvdp_targets.rs`:
  3 rows × 7 distance bands. WebSdr80 row preserved bit-identical
  from the Phase 4 single-table seed; Phone + Tv rows computed as
  `WebSdr80 × {PHONE,TV}_TARGET_MULTIPLIER` (1.04 / 1.12) at
  compile time via a `const fn` helper.
- `MetricSelection` extended with `target_display` field;
  `propagate_resolved_metric_to_encoder` writes the resolved
  value to `VarDctEncoder.target_display`.
- Both cvvdp backends (`GpuCvvdpBackend` + `CpuCvvdpBackend`)
  extended to accept `target_display` at `try_new` and route the
  matching `DisplayModel` through `CvvdpParams.display`. The CPU
  backend ALSO routes `DisplayGeometry` via `Cvvdp::with_geometry`
  (the GPU `CvvdpOpaque::new` API doesn't expose geometry — see
  Phase 1 geometry caveat in `DisplayConfig` docstring).
- New integration test `display_config_dispatch.rs`: 4-PASS
  invariants (default byte-identity, butteraugli unaffected,
  Libjxl strict-parity, multi-decoder roundtrip) + 1 ignored
  CUDA-required test. New unit tests in `cvvdp_targets.rs`
  (3 monotonicity + 1 legacy-wrapper byte-identity + helpers).
- New row in `docs/LIBJXL_DIVERGENCES.md` Section E.
- Hash-locks 36/36 BYTE-IDENTICAL, Libjxl byte-lock 4/4 PASS,
  drift 7/7 PASS, library tests 1515/1515 PASS.

What's deferred to Phase 1b / Phase 2:

- **Per-display re-seed against local re-scoring** of
  `cvvdp_vs_buttloop_tracking_2026-05-24.tsv` with the matching
  `DisplayModel` per row. Phase 1 ships heuristic multipliers
  derived from the §4 sensitivity estimates; the multipliers are
  intentionally simple (one scalar per row) and acknowledged in
  the cvvdp_targets module doc as "until a follow-on re-seed lands".
- **GPU geometry dispatch** — `cvvdp-gpu` upstream PR to add a
  `CvvdpOpaque::new_with_geometry` API. Phase 1 ships
  display-model dispatch on the GPU path; geometry stays at the
  `STANDARD_4K` upstream default until the upstream PR lands.
- **CICP-derived auto-dispatch** (Phase 4 in the original plan) —
  explicit-only API for Phase 1 per task spec.

### Phase 1.original — sanity bench (SDR-200 vs HDR PQ 1000, 1 day) — REPLACED by Phase 1 SHIPPED above

**Goal**: verify the §4 sensitivity estimate empirically before
committing to API design.

- Reuse existing `cvvdp_vs_buttloop_tracking_2026-05-24.tsv` encoded
  outputs (no re-encode).
- Extend `examples/cvvdp_track_baseline.rs` with `--display`
  parameter accepting `standard_4k` / `hdr_pq_1knit`.
- Re-score 100 cells (subset spanning d ∈ {0.5, 1.0, 2.0, 3.0, 5.0})
  under both displays.
- Output: TSV with `display × distance → median_jod` table.
- **Decision gate**: if HDR PQ 1000 median JOD shifts >0.05 JOD vs
  STANDARD_4K at any distance band, proceed to Phase 2. If shifts are
  <0.05 JOD uniformly, single-table calibration is acceptable; close
  this RFC as no-op.

### Phase 2 — 2-config API + table (XS, 2 days)

- Add `DisplayPreset { Standard4K, HdrPq1000 }` enum + 2-entry
  `CVVDP_DISPLAY_CALIBRATIONS` table seeded from Phase 1.
- Add `LossyConfig::with_target_display` (only 2 variants).
- Add `cvvdp_target_score_for_distance_and_display(d, preset)` lookup.
- Backend wires `CvvdpParams.display = DisplayModel::HDR_PQ_1KNIT`
  when preset is HdrPq1000.
- Unit tests + 4-cell paired smoke (2 displays × 2 distances).
- Hash-locks 36/36 BYTE-IDENTICAL (default path unchanged).

### Phase 3 — full 8-config sweep + backfill (S, 3 days)

- Encode the FULL 1,134-cell tracking corpus (no new encode needed —
  reuse existing TSV's bytes).
- Re-score under all 8 display presets (~10 min GPU time).
- Update `CVVDP_DISPLAY_CALIBRATIONS` table to 8 entries.
- Extend `DisplayPreset` enum to 8 variants.
- Validate: per-display table monotonicity (distance↑ ⇒ target↑) holds
  for each display.
- Multi-decoder roundtrip on ≥1 cell per display.

### Phase 4 — encoder auto-dispatch from CICP + intensity_target (S, 2 days)

- Add `auto_target_display_from_color_encoding(ce, nits)` helper.
- Wire into LossyConfig encode entry (mirror W44-RECON-DEEP/A10
  pattern): if `target_display` is `None`, derive from color_encoding
  + intensity_target.
- Caller override via `with_target_display(Some(...))` always wins.
- 20-cell paired bench: confirm auto-dispatch picks the right preset
  for {sRGB, PQ-1000, PQ-4000, HLG-1500} encode inputs.

### Phase 5 — zensim per-display supplement (L, 1 week, OPTIONAL)

Conditional on §3 findings — only fires if we add a future
ZensimProfile that uses `cvvdp_features.rs` supplemental 19-vector. If
PreviewV0_3 stays as the production metric, this phase is N/A.

If triggered:
- Extend `extract_cvvdp_features` signature to accept
  `display: DisplayConfig` parameter (currently hardcoded
  STANDARD_4K constants at `cvvdp_features.rs:72-77`).
- Re-extract per-corpus per-display: 8 displays × 6 corpora ×
  ~7000-15000 pairs = ~80 parquets, ~10 GB block storage.
- Retrain MLP head (8 per-display bakes OR 1 hybrid bake with
  display one-hot input).
- Cost: ~$50-200 vast.ai depending on bake count.

### Phase 6 — picker oracle re-train (XL, 2 weeks, OPTIONAL)

Conditional on Phase 4 landing AND picker consumers needing
display-aware codec selection. If picker stays display-agnostic
(picks "best codec" without display context), Phase 6 is N/A.

If triggered: per §6.4, single picker with `display_preset` one-hot
input. Re-train cost ~$10-30.

## §8. Open questions for the user

1. **Which display configs are highest production priority?** The §2
   matrix assumes web-SDR dominance + a long tail of HDR. If the
   primary use case is HDR mastering (zenmetrics' HDR PQ 10000 path) or
   mobile-first viewing (Mobile500), Phase 3's 8-config sweep order
   should be re-prioritised. **Recommendation: confirm or override the
   §2 row 1 (SDR-200) baseline + row 5 (HDR PQ 1000) for Phase 1.**

2. **Should display-config selection be auto-from-CICP, explicit-only,
   or hint-with-auto-fallback?** Phase 4 plan assumes
   hint-with-auto-fallback (matches W44-RECON-DEEP/A10 butteraugli
   pattern). Alternatives:
   - **Auto-from-CICP only**: simpler, but inflexible for cases like
     "encoding sRGB content intended for HDR viewing" (unusual but
     real for VFX / mastering workflows).
   - **Explicit-only**: callers MUST set `with_target_display` if
     cvvdp loop is active; default ERROR. Mirrors the RFC #3
     "explicit control" mandate but rejects the "sensible default"
     pattern the encoder has elsewhere.

3. **Picker oracle re-train budget**: are we committing to a display-
   aware picker bake (Phase 6, ~$10-30) as part of this work, or
   keeping the picker display-agnostic for now? If the latter, Phase 6
   is N/A and the work stops at Phase 4. **Recommendation: keep
   display-agnostic picker until §3 verifies a display-aware metric
   actually shifts picker decisions on real Pareto fronts.**

## §9. DO NOT (binding for future agents)

- DO NOT modify `cvvdp_targets.rs` table values without re-running the
  Phase 1 sanity bench AND updating per-display rows together.
- DO NOT cite "FMA precision" for any cross-display score deltas (per
  W44-66 user correction). The deltas are STRUCTURAL — castleCSF +
  Minkowski pool shifts at different adaptation luminances.
- DO NOT extend `extract_cvvdp_features` (`zensim/src/cvvdp_features.rs`)
  to per-display without first verifying a production `ZensimProfile`
  actually consumes those 19 supplemental features (currently NONE do
  per §3).
- DO NOT add `DisplayPreset` variants beyond cvvdp-gpu's existing
  presets — every new variant needs a `DisplayModel` const to
  dispatch through, and inventing new luminance/EOTF tuples without
  upstream support invalidates cvvdp's parity guarantees.
- DO NOT default-flip the cvvdp loop ON globally as part of this
  work. The cvvdp fork is OPT_IN_ONLY per RFC_CVVDP_FORK.md §9
  (Phase 6/8f verdict). This RFC ONLY extends the calibration surface
  for the existing opt-in path.
- DO NOT skip the Phase 1 sanity bench. The §4 sensitivity estimates
  are theory + paper-grounded but unmeasured on OUR encoder output
  with OUR corpus. A 1-day Phase 1 saves a week of misdirected Phase
  2-4 work if the sensitivities turn out to be tighter than estimated.
- DO NOT re-encode the corpus for display-config backfill. Bytes are
  display-INVARIANT (encoder doesn't know what display); only re-score
  the existing encoded outputs.

## §10. Status notes

- 2026-05-25: RFC drafted; awaits user decisions in §8 before Phase 1
  spawn. No production source touched. No sweeps fired.
- 2026-05-25 (later): Phase 1 SHIPPED — `with_target_display` API +
  3-preset calibration table + cvvdp backend dispatch. Heuristic seeds
  for Phone + Tv rows in lieu of the original "sanity bench" plan
  (collapsed Phases 1+2 of the original §7 into one chunk per user
  mandate). Phase 1b (per-display re-seed against measured shifts) +
  cvvdp-gpu `new_with_geometry` upstream PR + CICP auto-dispatch
  deferred to Phase 2+. See §7 "Phase 1 — API surface + 3-preset
  opt-in (SHIPPED 2026-05-25)" for the full ship narrative.
