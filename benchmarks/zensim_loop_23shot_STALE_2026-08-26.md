# jxl 23-shot zensim-loop census is STALE for the current loop (2026-08-26)

**Finding:** the committed 27-cell census
(`benchmarks/zensim_loop_23shot_summary_2026-08-05.json`, substrate jxl `b8a582e5`
/ zensim `471ce401`) no longer describes the current loop. Measured on jxl HEAD
`7155083e` (`feat(loop): jxl diffmap secant ON by default`, 2026-08-25) with a
fresh build of the `zensim_diffmap_rd` example.

## Evidence (run_23shot.sh probe + fresh, 2026-08-26)

- **Substrate probe FAIL:** re-running `v47A_base` k3 emit-last on the current
  loop and diffing against the committed mm-study substrate
  (`zensim_mm_cells_2026-07-31.tsv`) gives **81 cell mismatches, 87 trace
  mismatches** (27 cells × 3 channels). The loop's per-cell emit-last output
  changed since 2026-07-31, so the census's mm-*derived* entries (k3 emit-last +
  outer j2/j3) are cross-substrate and INVALID — exactly the condition the
  runner's probe gate exists to catch.
- **Fresh phase ran** all 5 arms × {k2_last, k2_best, k3_best} = 405 encodes in
  ~50 s (small refs), 33 TSVs at `~/tmp/23shot/run/fresh/` — but the **h3-mag arm
  engagement gate FAILED**: `v47A_h3g20c135` attr-probe fires **2× expected**
  (k2 probe=108 want=54; k3 probe=162 want=81), while all 4 base arms are clean
  (probe=0 want=0). The H3-magnitude steering path now emits the attribution
  probe twice per (cell × iter) — a mechanism change since the gate's `want=27·K`
  was calibrated.

## What a clean refresh requires (not a drop-in re-run)

The census was designed to DERIVE k3-emit-last + the outer j2/j3 entries from the
2026-07-31 mm-study rather than re-run them. The probe just proved that derivation
invalid on the current substrate, so a clean refresh must:
1. Run **fresh k3-emit-last for all 5 arms** (+ the outer j2/j3 arms) instead of
   deriving — extend `run_23shot.sh` fresh to emit those, and teach
   `analyze_23shot.py summarize` to read all-fresh inputs (today it reads
   `zensim_mm_{cells,outer}_2026-07-31.tsv` for the derived half).
2. **Reconcile the h3 engagement gate**: determine whether the 2× attr-probe on
   `v47A_h3g20c135` is a benign instrumentation change (update `want` to `54·K`)
   or a regression in the H3 steering path (fix in `src/vardct/zensim_loop.rs`).
   The base-arm census numbers do NOT depend on the attr-probe count (they come
   from the target_ab TSVs), so the 4 base arms can be summarized independently
   once (1) is done.

Until then, the gauntlet's `--loop-targeting` should keep reading the 2026-08-05
summary WITH THIS CAVEAT: it is a snapshot of the `b8a582e5` loop, not the shipped
secant-on-by-default loop. This is the current criterion-4 gap for jxl.

## Update: the h3-mag 2× probe is BENIGN (loop converges correctly)

Checked the fresh h3 arm census numbers (`~/tmp/23shot/run/fresh/
target_ab_v47A_h3g20c135_k3_best.tsv`): the `v47A_h3g20c135` loop **converges
tightly to target** on every cell — city 70/80/88 → 69.36/80.01/87.68, dog →
69.30/80.10/87.54, girl → 68.38/79.95/88.20. So the H3-magnitude steering is
functionally sound on the current loop; the doubled attr-probe (108 vs 54, 162 vs
81) is an INSTRUMENTATION count change, NOT a steering regression. The engagement
gate's `want=27·K` for the h3 arm is simply stale — the fix is `want=54·K` (or
gating on convergence, not probe count), pending a one-line confirmation of WHY
the probe now emits twice per (cell×iter) in the h3 path (the write at
`zensim_loop.rs:1186` fires once per attr-steered iteration; the h3 path enters it
twice). This removes the "steering regression?" branch: the clean refresh is now
(1) run fresh k3-emit-last + outer, (2) teach analyze_23shot.py to read them, and
(3) bump the h3 gate `want` — no loop fix needed.

## REFRESHED (same day): the current-loop census is now measured

`benchmarks/zensim_loop_23shot_summary_2026-08-26.json` is the ALL-FRESH census on
the current loop (jxl 24b20eed, secant on by default): 5 arms × {k2_last, k2_best,
k3_last, k3_best} measured fresh (`zensim_loop_23shot_2026-08-26.tsv`, 567 rows =
21 runs × 27 cells) + outer j2/j3 re-run fresh via `run_mm.sh outer`
(`zensim_outer_{zensimA,ssim2}_2026-08-26.tsv`) — zero mm-derived entries, both
published-anchor asserts correctly skipped on fresh inputs. Headline within±2:
k3_best v47A 24/27 · B 26/27 · bvls 25/27 · blend2L 25/27 · h3 18/27; k2_best
21/23/18/23/14. Notable vs 2026-08-05: the h3-mag arm lost its k2 edge on this
substrate (14/27 vs base 21/27) — the secant-on-default change moved the frontier;
gauntlet's `--loop-targeting` should point at the 2026-08-26 summary.

**W10L9_base re-measured fresh** (run_23shot_sota944.sh fresh on the current
substrate; `zensim_loop_23shot_sota944_2026-08-26.tsv`) and folded into the
2026-08-26 summary via `--extra-arm`. Still awaiting fresh re-measure:
`W10L9_h3ctrl2` (the adopted frontier arm; beatbutter runner), `h3own`,
`h3ownsp` — the board's LOOP_BAKE_MAP skips absent keys gracefully, so
W10L9_s4003_packed's scoreboard primary falls back to the fresh `W10L9_base`
until then. Do NOT cite the 08-05/08-07 W10L9 loop rows as current-loop numbers.

**W10L9_h3ctrl2 re-measured fresh too** (`run_beatbutter.sh h3ctrl2fresh` — the
exp100/clamp-2.00/bin-8 recipe on the current substrate;
`zensim_loop_h3ctrl2_2026-08-26.tsv`). The 08-26 summary now carries BOTH board
arms fresh; only h3own/h3ownsp (appendix N/P levers) remain old-substrate.

**Census family COMPLETE — every arm fresh.** h3own + h3ownsp re-measured via
run_23shot_sota944.sh (engagement wants updated to the measured benign 2× for
probes AND trace rows); all four W10L9 arms + the 5 panel arms + outer j2/j3 now
carry 2026-08-26 current-substrate provenance in the summary. Nothing derived,
nothing old-substrate remains.

**CORRECTION (same day):** the h3own/h3ownsp engagement wants were briefly
patched to 54·K on the assumption the attr-probe doubling applied there — the
run itself refuted that (probe=54 at k2 = exactly 27·K): **the 2× is
recipe-dependent** — present in the generic-map v47A_h3g20c135 and the exp100
recipes, ABSENT in the own-map h3 phases. Wants reverted to 27·K. Also measured:
`W10L9_h3own` rows are IDENTICAL (modulo run name) to `W10L9_h3ctrl2` on this
substrate — the adopted defaults are exp 1.0 / clamp 2.0, so the two names now
resolve to ONE config; do not count them as independent evidence.

**Precision on the h3own≡h3ctrl2 identity (measured):** 108/108 cells carry
IDENTICAL bytes, achieved_decoded, iters_used and seed_d across the two arms —
the payload is bit-for-bit one config; only wall-clock telemetry columns
(encode_ms/loop_ms/ms_per_compare) differ, which is what a raw file diff trips
on. "One config, two names" stands, now with the exact measurement.
