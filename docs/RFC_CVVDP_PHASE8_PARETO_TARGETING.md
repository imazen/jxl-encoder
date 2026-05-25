# Phase 8 RFC: make cvvdp diffmaps Pareto-optimal

**Date**: 2026-05-24
**Author**: Lilith River (with Claude scaffolding)
**Status**: SCOPING (revised 2026-05-24)

User mandate (2026-05-24, follow-on to Phase 7):
> "FIGURE OUT HOW TO SIT ON THE PARETO FRONT AND TARGET A CVVDP VALUE BUT
> STILL HAVE OPTIMAL RD TRADEOFFS. ... cvvdp is the language too. The
> entire point is to make cvvdp diffmaps WORK and work well."

cvvdp is the QUALITY LANGUAGE AND the LOOP DRIVER. The user is targeting
a cvvdp value AND using cvvdp's per-pixel diffmap to steer the qac field.
Phase 8's job is to make the resulting encoded output sit on the bytes-
vs-cvvdp Pareto front — fixing the cvvdp diffmap heuristic itself, NOT
routing around it via butteraugli.

A REVERTED-DIRECTION RFC drafted earlier today suggested using butteraugli
as the loop driver. The user explicitly rejected that: "cvvdp is the
language too. the entire point is to make cvvdp diffmaps WORK and work
well." This RFC is the corrected direction.

## §1. Phase 8a diagnosis (the data) — UNCHANGED

`scripts/cvvdp_pareto_diagnosis.py` analyzed Phase 6's 4,536-row TSV.

| backend | Pareto-front position (bytes vs cvvdp_gpu, e=8) |
|---|---|
| `B` (butteraugli) | **93.6%** of cells on front |
| `C_GPU` (cvvdp) | **40.3%** of cells on front |

At target cvvdp_jod 9.95, C_GPU uses 1.726× the bytes of B for the
same achieved cvvdp_jod. Per-distance: cvvdp competitive at d=0.5
(92.6% on front), degrades to 13-22% at d=2-5. The cvvdp loop is
**Pareto-dominated by butteraugli on cvvdp's own metric** for the
mid- to high-distance regime. The data tells us the cvvdp loop's
qac field is over-encoding — spending bytes that don't buy cvvdp.

## §2. Root cause: per-block diffmap reducer is butteraugli-calibrated

The buttloop (`vardct/perceptual_loop.rs`) reduces per-pixel diffmap
to per-8×8-block weights via median+MAD, then applies qac adjustments
proportional to those weights. The constants in that machinery were
tuned against butteraugli's value distribution + spatial signal shape
(narrow peaks where small regions are perturbed). cvvdp's diffmap has
a structurally different distribution:

1. **Broader spatial support** — cvvdp's Laplacian pyramid + per-band
   CSF + cross-channel masking smear the per-pixel signal across wider
   regions than butteraugli's near-pointwise max-norm signal.
2. **Different value range** — cvvdp's post-pool per-pixel quantity
   lives in a different numerical range than butteraugli's diffmap
   (which is a max-norm of psycho-adapted error blocks).
3. **W44 cost-model gates** (W44-105 4× seed scale, W44-109 e5/e7
   lifts, W44-117 EPF seed cluster) fire on diffmap-derived statistics
   that mean different things under cvvdp's distribution.

Net effect: when the cvvdp loop "asks" the per-block reducer "which
blocks need refinement?", the answer is shaped for butteraugli's
distribution and over-allocates qac to blocks that cvvdp wouldn't
prioritize. Then to converge to the JOD target, the loop has to
over-quantize OTHER regions — spending bytes that don't help cvvdp.

## §3. Three intervention paths

### §3.1. INTERVENTION A — Diffmap renormalization (least invasive)

Before the per-block reducer fires, normalize the cvvdp diffmap to
butteraugli-equivalent value range + distribution shape. The W44
heuristics fire as designed; the cvvdp signal becomes a "drop-in"
replacement.

Mechanism:
- Per-image (or per-tile) compute a scaling factor `s = mean(butter_diffmap) / mean(cvvdp_diffmap)` from a one-shot offline calibration
- OR: per-iter compute `s` from the diffmap's own statistics (relative scaling, no butteraugli involvement)
- Scale cvvdp_diffmap by `s` before feeding to `compute_per_block_distance`

Pros: minimal code change (one multiply); reuses W44 calibration 1:1
Cons: relative scaling assumes shape-compatibility; if shape differs
      structurally, normalization alone won't fix it

### §3.2. INTERVENTION B — Per-block reducer recalibration

Read what the per-block reducer's constants actually are (median + MAD
thresholds, qac adjust magnitudes) and re-tune them for cvvdp's
diffmap distribution. Likely 5-10 named constants.

Mechanism:
- Capture diffmap distribution per backend (B vs C_GPU) on a calibration
  corpus
- Fit per-block heuristic constants such that cvvdp's distribution
  drives equivalent qac adjustments to butteraugli's
- Ship two constant tables: `BUTTER_BLOCK_CONSTANTS` + `CVVDP_BLOCK_CONSTANTS`
- Switch at runtime based on which backend is active

Pros: structurally correct; respects cvvdp's distribution
Cons: multi-day calibration work; risk of overfitting; requires bench
      sweeps per change

### §3.3. INTERVENTION C — Per-iter bytes-tightening (Pareto-aware exit)

Modify the loop's exit condition: instead of "stop when JOD ≤ target",
add "AND no qac block can be loosened without crossing the target".
Forces the loop to give back bytes whenever cvvdp_jod has headroom.

Mechanism:
- After the loop's primary convergence step, run a "tighten" pass:
  for each block, try increasing qac by 1; if JOD still ≤ target +
  ε, accept the tightening (gives back bytes); if not, revert
- Repeat until no block can be tightened further

Pros: structurally Pareto-optimal by construction
Cons: extra iterations cost wall (each tighten-probe needs a cvvdp
      score), adds compute proportional to block count

### §3.4. Combined approach

A + C: cheap diffmap normalization PLUS bytes-tightening exit pass.
Captures most of the win at modest implementation cost. B remains as
follow-on if A+C doesn't reach Pareto-parity.

## §4. Phased ship plan (revised)

### Phase 8a — Pareto diagnosis (SHIPPED) ✓

Already on `cvvdp-fork-rfc` at commit `a4a89c1f`.
`scripts/cvvdp_pareto_diagnosis.py` documents the gap.

### Phase 8b — Diffmap distribution analysis

Capture per-pixel diffmap stats from both backends on the same input.
Output: histogram TSV + comparison plot. Identifies:
- Scale ratio (cvvdp mean / butter mean) per image class
- Shape divergence (KS test on distributions)
- Spatial-correlation structure (does cvvdp's broader signal redistribute
  the "important blocks" set?)

Foreground analytical chunk; ~1 day. Output informs §4.3.

### Phase 8c — Intervention A: diffmap renormalization

Implement the scaling factor + apply pre-reducer. Test:
- Re-run Phase 6 tracking sweep on the same 1,134 cells with cvvdp loop
- Pareto-front position should improve from 40.3% toward 80%+
- If not: fall back to §3.2 / §3.3

### Phase 8d — Intervention C: bytes-tighten exit

Add the post-convergence tightening pass. Bench wall hit (should be
≤ 30%; if more, document + ship gated). Re-run Pareto sweep.

### Phase 8e — Combined Intervention A + C

Validate that A and C compose well (no interference). Final Pareto
position should be ≥ 85% on front for the cvvdp-loop default.

### Phase 8f — Decision

If 8c+8d+8e gets cvvdp ≥ 85% Pareto-front position: ship as the new
cvvdp-loop default. Flip Phase 7's OPT_IN_ONLY verdict to ALWAYS_ON
when `--features cvvdp-loop` is compiled. Decision memo update.

If 8e plateaus below 85%: spawn Phase 8g (Intervention B per-block
constant re-fit, multi-day work).

### Phase 8g — Intervention B fallback

Per-block reducer constant re-fit. Only if 8e doesn't hit 85%.

## §5. Acceptance gates (per phase)

- Hash-locks 36/36 BYTE-IDENTICAL at default features (every phase)
- `EncoderStrategy::Libjxl` byte-locked 4/4 with all cvvdp features compiled
- Multi-decoder roundtrip (jxl-rs + djxl + jxl-oxide) per phase
- Phase-6-shape Pareto sweep at each phase, comparing C_GPU Pareto-front-pct
- Bench TSV committed to repo

## §6. Out of scope

- Phase 8 calibration table (target_jod → distance lookup) — drafted earlier
  today as `scripts/cvvdp_build_calibration_table.py` + `.snippet`. **Retained
  as fallback / OPT-OUT path** if Phase 8c-8g cannot reach Pareto-front
  parity. NOT the primary direction per user mandate.
- Replacing cvvdp's pool-order to materialize the strict diffmap
  invariant (Phase 1 honest-stop)
- Re-calibrating individual W44 gates for cvvdp (Phase 5+ work
  documented in `CVVDP_W44_GATE_TRANSFER.md`)
- Per-image content classification (Phase 8a default-blend uses
  global stats)

## §7. Status notes

- 2026-05-24: Phase 8a Pareto diagnosis SHIPPED. Numbers
  conclusive: cvvdp-loop Pareto-dominated 60% of cells.
- 2026-05-24: RFC drafted+revised after user redirect on direction.
  cvvdp diffmap is the loop driver; this RFC's interventions all
  preserve that.
- TBD: Phase 8b diffmap distribution analysis.
- TBD: Phase 8c-8g implementation.

## §8. The "cvvdp as language too" framing

The user's phrase locks in: cvvdp is the user-facing quality language
AND the internal loop signal. The encoder commits to cvvdp end-to-end.
The bytes-overhead diagnosis means cvvdp's loop is currently wasteful
on bytes; the fix is to make it efficient WITHIN the cvvdp-driven
discipline. That's what §3 interventions do — they all keep cvvdp's
diffmap as the steering signal; they just teach the per-block reducer
how to read cvvdp's distribution correctly.
