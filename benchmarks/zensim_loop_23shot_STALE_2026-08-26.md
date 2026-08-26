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
