# W44 cost-model gate transfer to cvvdp — scoping audit

**Date**: 2026-05-24
**Author**: Lilith River (with Claude scaffolding)
**Status**: SCOPING — first-pass categorization, not yet per-gate verified.

The jxl-encoder W44 wave shipped ~80 divergences from libjxl in
`docs/LIBJXL_DIVERGENCES.md`. Many were calibrated against butteraugli's
specific measurement behaviour. When the cvvdp fork lands, some of those
calibrations transfer directly and some need re-calibration on the cvvdp
metric. This memo categorizes the gates so Phase 5+ (cvvdp + W44 gate
revalidation) has a concrete inventory.

**Categories**:

- **M-AGNOSTIC** — pure entropy_mul / strategy-search table lifts. Don't
  touch the buttloop. Transfer 1:1; bytes change is structural, not
  metric-driven.
- **B-SEED** — buttloop seed-scale / iter-count adjustments. Tuned against
  butteraugli's convergence behaviour. Need re-calibration on cvvdp.
- **B-DIFFMAP** — EPF sharpness seed driven from butteraugli's per-pixel
  diffmap. Depends on the diffmap's spatial signal characteristics. Need
  recipe-match validation against cvvdp's RFC §3 diffmap.
- **D-PROXY** — discriminator predicates (mask1x1, m3, fcbr, edge_density,
  luma_var). Computed at encode entry on sRGB source. Metric-agnostic;
  transfer 1:1.
- **D-TARGET** — distance-band gates that fire when `target_distance` is
  in a range. The `target_distance` value is butteraugli's calibrated
  parameter; cvvdp uses the same field but its effective "perceptual
  difficulty" mapping is different. Re-calibrate distance bands per cvvdp
  difficulty.

## Inventory (first pass)

### Section A — Effort-gate divergences

| Row | Category | Action under cvvdp |
|---|---|---|
| `cfl_two_pass` (W44-133) | M-AGNOSTIC | Transfer 1:1 |
| `try_dct64` (W44-93) | M-AGNOSTIC | Transfer 1:1 |
| `epf_dynamic_sharpness` (W44-133) | M-AGNOSTIC | Transfer 1:1 |
| buttloop gate (`effort >= 8`) | B-SEED | Re-evaluate: cvvdp may converge faster/slower than butteraugli — iter count may want different effort gate |
| DC tree `kLearn` (W44-171) | M-AGNOSTIC | Transfer 1:1 |
| DC predictor set (W44-172) | M-AGNOSTIC | Transfer 1:1 |
| AC modular tree-learn `Variable` | M-AGNOSTIC | Transfer 1:1 |
| CfL Pass-2 LS-at-low-effort (W44-197) | M-AGNOSTIC | Transfer 1:1 |
| CfL Pass-1 LS/Newton (W44-195) | M-AGNOSTIC | Transfer 1:1 |

### Section B — Content-aware discriminator gates

| Row | Category | Action under cvvdp |
|---|---|---|
| W44-29 `high_d_photo_smooth_suppressed` | D-PROXY (mask1x1) + D-TARGET (d>=3) | Predicate fires same way; the inner entropy_mul table is M-AGNOSTIC; transfer 1:1, watch for d-band drift |
| W44-164 `auto_classify_content_class_from_layout` | D-PROXY | Transfer 1:1 |
| W44-65 `dct_suppress_hint` | D-PROXY | Transfer 1:1 |
| W44-91 `high_d_photo_smooth_zenanalyze` | D-PROXY + D-TARGET (d ∈ [3.0, 5.0]) | Predicate transfers; **d-band needs re-validation on cvvdp** |
| W44-151+152 `high_d_photo_smooth_p25_admit` | D-PROXY + D-TARGET | Same — re-validate d-band |
| W44-96/148/154 variant Z | D-PROXY + D-TARGET (d >= 4.5) | Re-validate d-band |
| W44-98/99/100 variant Z sub-discriminators (HC/LC) | D-PROXY | Transfer 1:1 |
| W44-156 `variant_z_d_high` | D-TARGET (d > 5.5) | Re-validate d-band on cvvdp |
| W44-166 `variant_z_admit` (mask_p25 >= 85) | D-PROXY + D-TARGET (d >= 4.5) | Re-validate d-band |
| W44-105 `BUTTLOOP_QF_SEED_SCALE` (4×) | **B-SEED** | **Re-calibrate scale on cvvdp**. Butteraugli's screenshot-class measurement bias was the root cause; cvvdp's bias may be smaller/larger. |
| W44-107/108 seed gate tightening (d >= 3.5) | B-SEED + D-TARGET | Re-validate both the scale AND the d-band |
| W44-109 `adaptive_quant_qf_seed_scale` (e5/e6/e7 lift) | B-SEED (extends W44-105 to no-buttloop efforts) | Re-evaluate — cvvdp may not need this lift if its iter-0 estimate is more accurate |
| W44-176 `terminal_class_exclude` | B-SEED + D-PROXY (luma_var ∈ [1500, 2200], fcbr >= 0.70) | Re-calibrate — terminal-class needed exclusion because W44-109 lift was net-negative pareto vs cjxl on butteraugli; cvvdp's measurement may not need exclusion at all |
| W44-117/118/120 EPF sharpness seed | **B-DIFFMAP** | **Most-affected by metric switch.** EPF seed compute uses butteraugli diffmap shape directly. With cvvdp diffmap (RFC §3) the spatial signal characteristics differ; recipe-match validate; possibly re-design. |
| W44-140 EPF seed distance-fade | B-DIFFMAP + D-TARGET | Depends on above; re-validate |
| W44-142 EPF seed codec_wiki suppress | B-DIFFMAP + D-PROXY | Depends on above; re-validate |
| W44-124 `dct32_keep_hint` auto | D-PROXY | Transfer 1:1 |
| W44-169 `compute_iters_narrow` (d ∈ [4, 5], e8+) | B-SEED + D-TARGET | Re-validate — buttloop iter narrowing depends on cvvdp's iter convergence behaviour |
| W44-135/143 `dct32_keep` distance gate | D-PROXY + D-TARGET | Re-validate d-band |

### Section C — Cost-model constants

(All M-AGNOSTIC — every entropy_mul lift, dead-zone threshold, kFavor,
mul8x8 multiplier transfers 1:1. The cost model operates on the encoder's
internal entropy estimates, not on the metric.)

### Section D — Algorithm choices

(All M-AGNOSTIC — clustering strategies, search policies, tree-kinds.
Metric-agnostic.)

### Section E — Opt-in APIs

(Mostly metric-agnostic; per-API audit deferred.)

### Section F — KNOWN-BUG clusters

Several KNOWN-BUG entries name a specific butteraugli-bias-induced
overshoot (e.g. "terminal e8 d=4 SSIM2 -1.93 + bytes +33%"). On cvvdp,
these bugs may resolve themselves OR shift to different cells. **Treat
the entire KNOWN-BUG cluster as "needs full re-measurement under cvvdp"**.

## Estimated work per category (Phase 5+ planning)

- **M-AGNOSTIC + D-PROXY** (≈45 of 80 rows): no work. Stay byte-identical
  under cvvdp since they don't touch the buttloop.
- **D-TARGET** (≈15 rows): per-row 4-cell A/B sweep to find the new
  d-band that gives the same effect under cvvdp. ~1 day of agent work
  per cluster of related rows.
- **B-SEED** (≈8 rows): each needs a 20-cell paired A/B sweep (cvvdp on
  vs cvvdp on + W44-N gate firing). 1-3 days per row.
- **B-DIFFMAP** (≈3 rows, all EPF-related): the biggest unknown. The
  W44-116/117 work proved the EPF seed mechanism is load-bearing on
  butteraugli's specific spatial signal. Under cvvdp, the EPF seed
  mechanism may need a different recipe entirely. Estimate: 2-3 weeks
  of measurement + redesign.

## Recommended order for Phase 5+ post-cvvdp-landing

1. Land cvvdp opt-in (Phase 3+4). Default off; W44 gates fire as today
   under butteraugli; cvvdp callers see the M-AGNOSTIC + D-PROXY effects.
2. Run the tracking benchmark (Phase 6) at default settings to surface
   which W44 gates over/underperform under cvvdp out-of-the-box.
3. Re-calibrate D-TARGET d-bands per the benchmark surface.
4. Tackle B-SEED gates in order of pareto impact (W44-105 / W44-109 /
   W44-176 likely top priority).
5. Tackle B-DIFFMAP cluster last — may require RFC §3 recipe iteration.

## DO NOT (binding for Phase 5+ agents)

- DO NOT delete any W44 gate just because it doesn't transfer to cvvdp.
  Gate either keeps the butteraugli-only path (with `cvvdp_loop` check)
  OR ships a cvvdp-equivalent variant.
- DO NOT cite "FMA precision" for any byte drift under cvvdp.
- DO NOT recalibrate D-TARGET d-bands by hand-eye; use the tracking
  benchmark TSV per the methodology in RFC §5.
- DO NOT touch the `EncoderStrategy::Libjxl` invariant — Libjxl ALWAYS
  uses butteraugli, ALWAYS uses the W44 gates as calibrated, ALWAYS
  byte-locks 4/4.

## Open questions (file as next-chunk hypotheses)

- **Q1**: Does cvvdp converge faster than butteraugli in the buttloop
  (allowing W44-169's iter narrowing to be wider)?
- **Q2**: Does cvvdp produce a per-pixel signal in the same value range
  as butteraugli's diffmap (so the W44-117 EPF seed compute can reuse
  its existing threshold)?
- **Q3**: Does the W44-105 4× seed scale buy ANYTHING on cvvdp's
  screenshot measurements? Or does cvvdp not have the screenshot-bias
  butteraugli has?

The tracking benchmark answers Q1/Q2/Q3 empirically once we have ≥50
cells of cvvdp data on the W44-105/W44-169-firing cluster.
