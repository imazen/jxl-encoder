# zensim loop 2/3-SHOT sweep — dashboard feed (2026-08-01)

Fourth study in the loop series (after `zensim_attr_loop69_2026-07-29.md`,
`zensim_diffmap_efficiency_2026-07-31.md`, `zensim_loop_metric_matrix_2026-07-31.md`,
`zensim_emit_best_2026-07-31.md`). **Purpose:** the full model dashboard (zensim's
summer gauntlet) gains a "which models do best at 2-shot and 3-shot jxl targeting"
panel — loop budget k=2 and k=3, within ±2.0 of target, decoded-judged. This study
fills the exact missing cells and emits the machine summary the dashboard reads.

**Question:** per loop model, how many of the standard 27 cells does the loop land
within ±2.0 of target (decoded-judged, the model's OWN units) at encode budget 2 and
3, under both emission rules — and at what byte cost.

## Substrate (recorded before running)

- jxl-encoder `main@origin` = `1e98748e` — **no encoder changes in this study**; it
  uses only landed features (`--iters k`; `JXL_ZENSIM_EMIT_BEST=1` from the emit-best
  study, its R0/G-TRAJ/G-EMIT gates green there).
- zensim read-only path dep at `2f0cec29` (clean tree; moved since the mm study's
  `402f4e63` — see the SUBSTRATE PROBE below, which turns "should be identical"
  into a measurement).
- Loop semantics unchanged: `--iters N` = compares at indices 0..=N (N+1 compares;
  index 0 = seed state); the emitted bitstream corresponds to the LAST measured
  iterate (emit-last) or the INTERNALLY-closest-to-target iterate
  (`JXL_ZENSIM_EMIT_BEST=1`, latest-tie); the loop never entropy-codes.
- Judged = decode the emitted bitstream, score with the SAME metric that drove the
  loop (bake judge for bake arms; zenmetrics ssim2 for ssim2-outer). ±2.0 is in each
  arm's OWN units — cross-arm counts are not unit-comparable (the mm study's F5 is
  the shared-scale read).
- Bakes (sha-verified sizes as mm): v47A `v47_strict_qat_native_2026-05-27.bin`
  (27,316 B), shippedB `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`
  (7,325 B), v02bvls `v02_bvls_NO_shaping_2026-05-28.bin` (8,622 B), blend2L
  `mlp_2L_diverse_H128_2026-07-15.bin` (271,900 B).

## Cells: fresh vs derived (the study's shape)

Standard 9-ref × {70, 80, 88} matrix (#69 fixtures: city/dog/girl 576², CID22-512
val 1025469/1418519/1189261, gb82-sc 576² crops sc_wiki/sc_gui/sc_imessage;
effort 8; seed distance 2.5/1.5/1.5).

**FRESH (this study, 15 runs = 405 encodes + 27 probe encodes):** arms
{v47A_base, B_base, bvls_base, blend2L_base, v47A_h3g20c135(g20 c1.35, the
#70-item-1 gate winner)} × {k2 emit-last, k2 emit-best, k3 emit-best}. Runner
`scripts/zensim-loop-eff/run_23shot.sh` (fixtures/envs/labels mirror `run_mm.sh` /
`run_emitbest.sh`); raw cells committed as `zensim_loop_23shot_2026-08-01.tsv`
(probe run included, labeled `v47A_base_k3_lastprobe`).

**DERIVED (no re-run, from committed mm-study TSVs):**

- k3 emit-last per arm = `zensim_mm_cells_2026-07-31.tsv` rows `run == <arm>_k3`
  (the mm study's as-emitted k3 cells are exactly k3 emit-last).
- outer 2/3-shot = `zensim_mm_outer_2026-07-31.tsv`: j2/j3 = the as-emitted full
  encode at `outer_iter == 2/3` (seed + j controller steps — the step-count parallel
  of inner k=j); j2_best/j3_best = min |judged − t| over `outer_iter ≤ j`
  (decoded-judged by construction; latest-tie, matching the emit-best tie rule).
- `latest` arm SKIPPED by design: the mm mount-equivalence gate proved it 27/27
  byte-identical to B_base (decoded max |Δ| = 0.0000) — its 2/3-shot numbers ARE
  B_base's rows.

**SUBSTRATE PROBE (the cross-substrate honesty gate):** the mm study forbids merging
prior-study TSVs across substrate moves. Before deriving anything, this study re-ran
the mm `v47A_base_k3` arm (emit-last, all 27 cells) on the CURRENT substrate and
diffed it against the committed mm rows: **27/27 cells equal on
achieved_decoded/abs_err/bytes and 108/108 trace compares equal** (`analyze_23shot.py
verify`). The loop is numerically identical across jxl `475f50ad+instr → 1e98748e`
and zensim `402f4e63 → 2f0cec29` — the derivations above are measurements on the
same function, not cross-substrate hope. (zensim's #70 perf work gated "maps
bit-identical cross-version"; this confirms the loop scores too.)

## Gates (in-run)

- Engagement: h3 runs' `JXL_ZENSIM_ATTR_PROBE` = exactly k×27 lines (54 at k2, 81
  at k3); every baseline run = 0 lines.
- Emit-best k2 A/B REPORT (not a hard gate — at low k the internal argmin is often
  the last compare, in which case best==last bitstreams are correct behavior):
  count of k2 best-vs-last bitstream diffs per arm, reported below.
- The substrate probe above (hard gate: any mismatch = derivations invalid, stop).

## Outputs

- `benchmarks/zensim_loop_23shot_2026-08-01.tsv` — fresh per-cell rows
  (run + the target_ab schema, as the mm cells TSV).
- `benchmarks/zensim_loop_23shot_summary_2026-08-01.json` — the machine summary the
  zensim gauntlet reads (`--loop-targeting`): per model × mode
  `{within2, n_cells: 27, med_abs_err, med_bytes}` + per-entry provenance
  (fresh-run vs derived-from-which-TSV) + a units/semantics `notes` field.
  Every number re-derives exactly from the committed TSVs via
  `scripts/zensim-loop-eff/analyze_23shot.py summarize` (median = numpy.median;
  within2 recomputed from |achieved_decoded − target| and asserted equal to the
  TSV's abs_err; published-doc anchors asserted: mm F1@k3 counts and outer j3
  counts must reproduce or the derivation aborts).

## Results

Data: `zensim_loop_23shot_2026-08-01.tsv` (fresh cells + probe, this dir) +
the two mm TSVs for derived entries; machine summary
`zensim_loop_23shot_summary_2026-08-01.json`. Runs 2026-08-01 (~432 encodes,
~4 min nice'd). Substrate as registered: jxl `1e98748e`, zensim `2f0cec29`.

**Gates — all green:**

- **SUBSTRATE PROBE PASS** — fresh `v47A_base_k3_lastprobe` vs committed mm
  `v47A_base_k3`: 27/27 cells equal (achieved_decoded/abs_err/bytes), 108/108
  trace compares equal. Derivations valid.
- **Engagement PASS** — h3 probes exactly 54/54/81 lines (k2_last/k2_best/
  k3_best); all 12 baseline-run probes 0.
- **Emit-best k2 A/B (report)** — bitstreams differing best-vs-last:
  v47A 3/27, B 6/27, bvls 5/27, blend2L 5/27, h3g20c135 2/27 (the rest have
  argmin == last compare — correct low-k behavior, per the emit-best study's P4).

### The panel table — cells within ±2.0 (of 27) | median |err| | median bytes

Fresh unless marked D (derived from the mm TSVs; substrate-probe-verified).

| model | k2 emit-last | k2 emit-best | k3 emit-last | k3 emit-best |
|---|--:|--:|--:|--:|
| v47A_base       | 12/27 · 3.04 · 22.8K | 12/27 · 3.04 · 22.8K | 13/27 · 2.28 · 22.2K ᴰ | 13/27 · 2.28 · 22.2K |
| B_base (shippedB) | 13/27 · 2.23 · 26.4K | 14/27 · 1.94 · 26.4K | 14/27 · 1.89 · 26.3K ᴰ | **15/27** · 1.89 · 26.3K |
| bvls_base       | 6/27 · 5.35 · 31.2K | 7/27 · 5.24 · 31.2K | 6/27 · 3.26 · 32.6K ᴰ | 8/27 · 3.14 · 32.6K |
| blend2L_base    | 8/27 · 2.76 · 23.1K | 8/27 · 2.76 · 23.1K | 11/27 · 2.18 · 22.5K ᴰ | 11/27 · 2.18 · 23.1K |
| v47A_h3g20c135  | 12/27 · 2.84 · 22.5K | 12/27 · 2.84 · 22.5K | 13/27 · 2.05 · 21.7K ᴰ | 13/27 · 2.05 · 21.7K |
| outer_zensimA ᴰ | — | 12/27 · 3.09 · 22.9K (j2) | — | 14/27 · 1.94 · 21.9K (j3) |
| outer_ssim2 ᴰ   | — | 11/27 · 2.63 · 22.2K (j2) | — | **16/27** · 1.62 · 21.2K (j3) |

(Outer j2/j3 = as-emitted at that iterate; best-of-≤j equals as-emitted on every
outer entry — the outer controller's judged error never regretted an earlier
iterate at these budgets, consistent with the mm F1 outer rows. Outer wall cost
is ~4× inner per step — mm F4.)

### Headline conclusions

1. **2-shot ranking (emit-best): shippedB 14/27 > v47A 12/27 = v47A-h3 12/27 >
   outer-zensimA 12/27 (j2) > outer-ssim2 11/27 (j2) > blend2L 8/27 >
   v02bvls 7/27.** The linear B dial is the best 2-shot targeting model — its
   oscillating controller gets close fast and emit-best banks the closest pass
   (girl/t70 3.21 → 1.53 is the flipped cell).
2. **3-shot ranking (emit-best): outer-ssim2 16/27 (own units) > shippedB
   15/27 > outer-zensimA 14/27 > v47A 13/27 = v47A-h3 13/27 > blend2L 11/27 >
   v02bvls 8/27.** Among INNER (single-encode) arms shippedB leads; the outer
   arms buy their counts with 4 full encodes (~4× wall). Cross-arm counts are
   own-units (registered caveat).
3. **Emit-best is worth +1..+2 cells at budget 2-3 for the oscillating linear
   dials and nothing for the one-sided MLP dials** — B k2 13→14, k3 14→15
   (cid1418519/t70 2.63→1.94); bvls k2 6→7, k3 6→8 (sc_imessage/t70+t80);
   v47A/h3/blend2L unchanged (their k≤3 trajectories are monotone approaches —
   argmin is the last compare on 24-25/27 cells). Bytes cost of emit-best at
   these budgets: median 0 (same-bytes medians), consistent with the emit-best
   study's neutral P3.
4. **Budget 2 → 3 is worth +1..+3 cells everywhere** (B 14→15, v47A 12→13,
   h3 12→13, blend2L 8→11, outer-zensimA 12→14, outer-ssim2 11→16): the
   mm-study's "budget-limited, not tolerance-limited" verdict extends down to
   k=2 — no arm's 2-shot census reaches its k6 census (18-22/27).
5. **h3 steering does not change the 2/3-shot census vs v47A-baseline (12/13
   both), but improves the medians** (k3 2.28 → 2.05; k2 3.04 → 2.84) **and
   stays byte-cheaper at equal achieved** (mm F3: 0.976 ratio) — consistent
   with the mm study's verdict that h3's win is bytes + fine accuracy, not
   census, at small budgets.
6. **v02bvls remains the census floor at every budget** (6-8/27): the
   compressed top-end (dial p95 ≈ 87.4, mm headline 6) makes t88 structurally
   unreachable regardless of budget or emission rule.

## Limitations

- Emit-best selects on the loop's INTERNAL score (registered design of the landed
  feature); the judged outcome of that selection is what this study reports —
  decoded-judged either way, but the selection channel carries the emit-best
  study's judged−internal transfer caveat (±0.13 med photo, worse on nonphoto).
- Own-units caveat as registered: within-±2 counts compare each arm in its own
  metric's units.
- Outer budgets count controller steps (j full encodes after the seed); an outer
  "2-shot" spends ~4× the wall of an inner k=2 run (mm F4) — the panel is a
  targeting-accuracy read, not a cost-normalized one.
- 576²-class fixtures only; loop-dynamics study, not a perf calibration.
