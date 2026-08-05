# zensim loop 2/3-SHOT — SOTA-944 candidate panel (2026-08-05)

Fifth study in the loop series; the direct extension of
`zensim_loop_23shot_2026-08-01.md` (protocol inherited verbatim). **Purpose:** the
sota944 campaign's wave-11 candidate has never entered a loop-targeting study — the
last "uses" axis on its freeze surface. This study adds the candidate as one arm to
the existing 2/3-shot panel and regenerates the machine summary the zensim gauntlet
reads.

**Question:** does `W10L9_s4003_packed` (the 944-class pruned candidate) hit
decoded-judged targets within ±2.0 (its OWN units) at loop budget k=2/k=3 at least
as well as the incumbents (v47A, shippedB)?

## Candidate + controls

- **Candidate**: `/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin`
  (165,696 B; wave-11 battery-selected, k=8-confirmed recipe; PRUNED:
  `caller_input_width()` = 944, internal `n_inputs()` = 667 via
  `FeatureTransform::Drop`; dial mono 99.32 %, M3a 0.862). Board name
  `W10L9_s4003_packed`. Arm key **`W10L9_base`** (baseline mode — the same loop
  rule as every `*_base` arm; the score model is the only variable).
- **Controls**: the committed 2026-08-01 panel rows — v47A_base, B_base, bvls_base,
  blend2L_base, v47A_h3g20c135, outer_zensimA, outer_ssim2 — CARRIED, not re-run,
  gated by the substrate probe below.

## Registered outcomes (pre-committed before any run)

On k2/k3 emit-best within-±2 counts (and the medians as tie-breaks), candidate vs
v47A_base and B_base:

- **(a) beats both** ⇒ the candidate is the loop-recommended model; the last freeze
  axis fills in green.
- **(b) parity** (within ±1 cell of the better incumbent, medians comparable) ⇒
  fine — the candidate is no worse where B was the incumbent.
- **(c) worse** ⇒ the loop axis stays with the incumbents and the freeze surface
  says so. Honest nulls stand; nothing ships or swaps on this study alone.

Also recorded (cheap, registered): the candidate's per-compare loop cost vs v47A
from the SAME-session fresh runs (`loop_ms`, `ms_per_compare` medians — the probe
re-runs v47A k3 fresh, so both sides are measured on this box today). The zensim
perf work (forward ~25× faster; pruning −25 % forward) is NOT assumed to show up
end-to-end — the loop's 944 route pays a second walk (see mechanics), and the
number reported is the measured end-to-end one.

## Integration mechanics (jxl-encoder changes, registered before running)

The existing `bake:` mount extracts 372-class features and would fail the forward
for a 944-class bake — and the loop swallows compare errors
(`Err(_) => return Ok(current_params)`), silently emitting the seed. The changes:

1. **Width probe fixed + loud**: `rd_infer_n_inputs` probes
   `[156, 228, 300, 372, 720, 924, 944]` smallest-first over a 944-long zero
   vector (smallest-first preserves the ≤372 arms' resolved widths exactly;
   `bake:` mount PANICS if no width is accepted — no silent seed-emit).
2. **`JXL_ZENSIM_MODEL_MAP` loud guard**: unknown values PANIC (previously fell
   through to baseline silently — the loop69 hazard); map-steering arms with a
   folded-class (≥720) bake PANIC (the fused compare is 372-class only —
   registered limitation, not silently downgraded).
3. **Folded-class score route** (`rd_n_in ≥ 720`): per iterate, the score comes
   from the canonical folded extraction — 944 → `compute_folded720_append2_features`,
   924 → `compute_folded720_append_features`, 720 → `compute_folded720_features` —
   over (source, recon) as `LinearF32Rgba` StridedBytes, forwarded through
   `zensim::score_features_with_profile` (full bundle: transforms incl. Drop,
   spline, extrapolation; sizes by `caller_input_width()` — the metric.rs
   caller-width fix class, verified). The redistribution map stays the standard
   Trained diffmap computed from a no-mlp profile whose walk config (weights,
   num_scales=4, blur 5/1) is bit-identical to the mlp mounts' — the map and loop
   rule are IDENTICAL to every `*_base` arm; only the controlling/judging score
   model differs. Cost consequence (registered): the folded route runs the v1
   diffmap walk AND the 944 extraction per compare — there is no fused 944 entry
   (C3a fusion is v1/372-class) — so the candidate's per-compare cost carries a
   structural second pass, reported as measured.
4. **Judge**: `zensim_diffmap_rd`'s judge mounts the same bake; for folded-class
   bakes it judges via the same canonical folded extraction on (ref, decoded) u8
   sources + `score_features_with_profile`. Same-scorer judging contract
   unchanged.
5. `Cargo.toml`: zensim dep gains `feature-regime-v2` (compile-time only; no
   shipped-path behavior change — gated by the substrate probe).

## Cells

Standard 9-ref × {70, 80, 88} matrix (city/dog/girl 576², CID22-512 val
1025469/1418519/1189261, gb82-sc 576² crops sc_wiki/sc_gui/sc_imessage; effort 8;
seed distance 2.5/1.5/1.5; `JXL_ZENSIM_TARGET_TOL=-1`, full budget, no early stop).

**FRESH (this study):** `W10L9_base` × {k2_last, k2_best, k3_last, k3_best} = 4
runs = 108 encodes + the 27-encode probe. k3_last is run fresh (the candidate has
no mm-study rows to derive from).

**CARRIED:** every 2026-08-01 summary entry, provenance preserved.

## Gates (in-run, hard unless marked)

- **SUBSTRATE PROBE** (hard): re-run `v47A_base_k3` (emit-last, 27 cells) on the
  CURRENT substrate; `analyze_23shot.py verify` against the committed mm rows —
  27/27 cells equal (achieved_decoded/abs_err/bytes) and 108/108 trace compares
  equal. This simultaneously (i) validates carrying the 2026-08-01/mm rows across
  the substrate move, and (ii) is the R0-identity gate proving the integration
  changes above do NOT alter the 372-class loop. Any mismatch ⇒ STOP (fresh re-run
  of all control arms required; derivations invalid).
- **Engagement** (hard): candidate runs are baseline-mode — `JXL_ZENSIM_ATTR_PROBE`
  files must have 0 lines; trace files must carry exactly 27×(k+1) compare rows.
- **Emit-best k2/k3 A/B** (report): count of candidate bitstreams differing
  best-vs-last (0 = argmin==last everywhere, legal at low k).
- **Summarize anchors** (hard): the analyze owner re-derives every carried number
  and asserts the published mm F1@k3 + outer j3 counts; candidate rows must be
  27/27 per mode with `abs_err` consistent with |achieved−target| ≤ 5e-3.

## Outputs

- `benchmarks/zensim_loop_23shot_sota944_2026-08-05.tsv` — fresh per-cell rows
  (probe + candidate; target_ab schema with `run` prefix, as 2026-08-01).
- `benchmarks/zensim_loop_23shot_summary_2026-08-05.json` — the regenerated
  machine summary (schema `zensim_loop_23shot_summary.v1`): all 2026-08-01 models
  (provenance carried) + `W10L9_base` (4 fresh modes). The zensim gauntlet's
  `--loop-targeting` default moves to this file; `LOOP_BAKE_MAP` gains
  `W10L9_base → W10L9_s4003_packed`.
- Runner: `scripts/zensim-loop-eff/run_23shot_sota944.sh`; stats owner:
  `scripts/zensim-loop-eff/analyze_23shot.py` (extended with `--extra-arm`/
  `--extra-cells` — the owner is extended, never forked).

## Results

Runs 2026-08-05 (~135 fresh encodes: 27 probe + 108 candidate, nice'd, ~90 s).
Substrate: jxl `792378e1`+this-study's integration commit, zensim `17770775`
(path dep, clean tree at origin/main). Data:
`zensim_loop_23shot_sota944_2026-08-05.tsv` (fresh cells + probe) + the two mm
TSVs + the 2026-08-01 TSV for carried entries; machine summary
`zensim_loop_23shot_summary_2026-08-05.json`.

**Gates — all green:**

- **SUBSTRATE PROBE PASS** — fresh `v47A_base_k3_lastprobe` vs committed mm
  rows: 27/27 cells equal (achieved_decoded/abs_err/bytes), 108/108 trace
  compares equal. Carried controls valid AND the folded-class integration
  changes provably did not alter the 372-class loop (the R0-identity read).
- **Engagement PASS** — candidate probe files 0 lines on all 4 runs; traces
  exactly 81/81/108/108 rows (27×(k+1)).
- **Caller-width check (the task's first-priority hazard) — CONFIRMED + FIXED**:
  the pre-existing probe (`[372, 300, 228, 156]`) returns 0 for the pruned
  944-class bake, and the pre-existing mount would then have silently emitted
  seed-quality bitstreams through the compare-error swallow. The smallest-first
  probe resolves the candidate at its CALLER width 944 (not its internal 667);
  the full-bundle forward (`score_features_with_profile` →
  `caller_input_width()` sizing) scores it correctly — in-loop vs decoded-judged
  transfer is tight (med |Δ| ≈ 0.1, e.g. city/t80 80.53 in-loop vs 80.55
  decoded).
- **Emit-best A/B (report)** — candidate bitstreams differing best-vs-last:
  k2 6/27, k3 5/27 (oscillating-controller class, like B). The banked iterates
  changed neither census nor median (no mover crossed ±2; the median cell was
  unmoved — med_bytes did move, k2 23.3K→23.8K, confirming distinct rows).
- **Analyze-owner regression** — the extended `analyze_23shot.py` byte-reproduces
  the 2026-08-01 summary (all fields identical; only the self-referential
  `source` filename differs when written under a new name).

### The panel table — cells within ±2.0 (of 27) | median |err| | median bytes

Candidate fresh; incumbents carried (ᶜ) from the 2026-08-01 summary
(substrate-probe-verified). Full carried table: `zensim_loop_23shot_2026-08-01.md`.

| model | k2 emit-last | k2 emit-best | k3 emit-last | k3 emit-best |
|---|--:|--:|--:|--:|
| v47A_base ᶜ     | 12/27 · 3.04 · 22.8K | 12/27 · 3.04 · 22.8K | 13/27 · 2.28 · 22.2K | 13/27 · 2.28 · 22.2K |
| B_base ᶜ        | 13/27 · 2.23 · 26.4K | **14/27** · 1.94 · 26.4K | 14/27 · 1.89 · 26.3K | 15/27 · 1.89 · 26.3K |
| **W10L9_base**  | 10/27 · 2.63 · 23.3K | 10/27 · 2.63 · 23.8K | **15/27 · 1.82** · 22.9K | **15/27 · 1.82** · 22.9K |

Per-target (emit-best, within-±2 of 9 | median):

| model | t70 k2 | t80 k2 | t88 k2 | t70 k3 | t80 k3 | t88 k3 |
|---|--:|--:|--:|--:|--:|--:|
| v47A_base  | 3/9 · 3.27 | 5/9 · 1.70 | 4/9 · 2.22 | 3/9 · 2.93 | 5/9 · 1.03 | 5/9 · 1.98 |
| B_base     | 6/9 · 1.91 | 5/9 · 1.62 | 3/9 · 2.34 | 6/9 · 1.24 | 5/9 · 0.70 | 4/9 · 2.16 |
| W10L9_base | 2/9 · 3.03 | 4/9 · 2.26 | 4/9 · 2.50 | 3/9 · 2.76 | 5/9 · 1.40 | **7/9** · 1.81 |

Class split (emit-best): candidate photo 9/18 (k2) → **14/18 (k3, the best
photo census of any inner arm**; B 13/18, v47A 12/18); nonphoto 1/9 both
budgets — the sc_gui/sc_wiki t70/t80 cells overshoot for EVERY arm (achieved
86-91 at t70; the 1.35-clamp controller cannot push screen content down to
target in 2-3 steps from these seeds — the k2 worst-cell list is entirely this
class plus cid1418519/t70).

### Which registered outcome fired

**Split verdict — (c) at budget 2, (b)-plus at budget 3; NOT (a) overall.**

- **k2 emit-best: candidate 10/27 < v47A 12/27 < B 14/27** — worse than both
  incumbents (outcome c). The 2-shot loop recommendation stays with shippedB.
- **k3 emit-best: candidate 15/27 = B 15/27 > v47A 13/27**, with the best
  inner-arm median (1.816 vs B 1.894) — registered tie-break favors the
  candidate (outcome b, leaning a). At **k3 emit-last the candidate leads
  outright: 15/27 vs B 14/27 / v47A 13/27** — the best inner-arm 3-shot
  as-emitted census in the series.
- Freeze-surface sentence: *the loop axis is filled — at budget 2 the
  incumbent B remains the recommendation (14 vs 10); at budget 3 the candidate
  ties B's census with the best median, the best photo census (14/18), and the
  best near-lossless-band census (t88 7/9, vs 4/9 B / 5/9 v47A) — the
  candidate's k3 t88 strength lands exactly in the HF weak zone the campaign
  targets.* Honest nulls stand; nothing ships or swaps on this study alone.

### Loop cost (same-session, 576²-class, medians over 27 cells)

| run | ms/compare | loop_ms | encode_ms |
|---|--:|--:|--:|
| v47A_base_k3 (probe, fresh today) | 34.6 | 138.6 | 200.2 |
| W10L9_base_k3_last | 51.8 | 207.0 | 276.3 |
| W10L9_base_k2_last | 55.5 | 166.5 | 221.5 |

**Candidate/v47A: 1.50× per compare, 1.38× whole-encode (k3).** The folded
route pays the registered structural second pass (v1 diffmap walk for the map
+ the 944 streaming extraction for the score) and STILL lands at 1.5× — the
C5 streaming foldapp extraction is the reason this is not 2-3×. The zensim
forward-perf work (~25× faster forwards, pruning −25%) does not dominate
end-to-end because extraction, not the MLP forward, is the per-compare cost —
measured, as registered, not assumed. A fused folded-class compare (score +
map in one walk) is the obvious future lever if the candidate's k3 profile
motivates loop use.

## Limitations (registered)

- Own-units caveat as 2026-08-01: within-±2 counts are in each arm's own metric
  units; candidate-vs-incumbent counts compare *target-hitting in own units*, not
  a shared perceptual scale.
- The candidate's loop route pays a structural second pass (no fused 944 entry);
  its cost number is an integration-surface fact, not a model-capability fact.
- The candidate cannot run map-steering arms (h3 etc.) — 372-class fused only;
  guarded loudly, listed as the map-steering axis's registered gap.
- 576²-class fixtures only; loop-dynamics study, not a perf calibration.
- In-loop features come from LinearF32 recon; judge features from decoded u8 —
  same relationship every arm has (the emit-best study's judged−internal transfer
  caveat applies to the emit-best selection channel).
