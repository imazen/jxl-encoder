# Revisit queue — decisions made WITHOUT zenbench on a busy system (2026-06-11 overnight)

**User directive (2026-06-11, verbatim intent): flag every decision this
session that was not made with zenbench and properly repeated, as possibly
wrong, to revisit once the system calms.**

The overnight box hosted concurrent activity (remote orchestration, capture
tails, background loops; loadavg 1–4 across windows). None of tonight's
wall/share measurements used zenbench (interleaved paired execution, paired
statistics, regression gates) — they were ad-hoc loops (median-of-2/3,
process-serial, not interleaved) or single perf runs. Byte-based evidence is
load-immune and NOT flagged.

## FLAGGED — possibly wrong, re-run zenbench-grade on a quiet box

| # | decision | evidence used | risk | quiet re-run shape |
|---|---|---|---|---|
| 1 | **#64: gather_samples SIMD next-chunk MEASURED-REJECTED** | perf cycles self-shares, 1 run/cell, 2 photo cells (`perf_gather_profile_2026-06-10.meta` ADDENDUM 4) | The ranking (gather never top) held on two days/two binaries, but shares swung ±10 pp on the SAME cell (find_best_split 15.1 %→25.3 %) — that swing may itself be load. A quiet profile could lift gather above the 5 % bar (it was 4.2/4.8 %). | 3+ perf runs/cell on idle box (loadavg<1), clic097 + tokyo12mp + 1 screen cell; recompute shares; if gather ≥5 % on ≥2 photo cells AND ranks top-3, the rejection flips. |
| 2 | **#24 EV table: e9 = 5.49× e7, per-class savings ranking** | ad-hoc median-of-3, threads 8, busy box (`ev_wall_tables_2026-06-11.tsv` ev24 rows) | Bytes deltas are sound; the 5.49× ratio and any picker threshold derived from walls are not anchor-grade. | zenbench-style interleaved paired e7/e9 per cell, ≥5 iters, quiet box; medians + min; re-derive ratio. |
| 3 | **#45 wall tables: e6→e7 = 7.41× cliff, ~2.2×/tier, e10/e11 absolutes** | ad-hoc median-of-2 (walls45 rows, same file) | Tier-ratio SHAPE is probably robust; absolutes (e10 210.7 s / e11 446.5 s) are not. RFC budgets must not anchor on these. | same as #2 across e1..e11, 5-cell set, ≥3 iters. |
| 4 | **MABSplit Phase-0 verdict's supporting clause** ("find_best_split ceiling is only 12.8–25.3 % of wall") | single perf runs | The PRIMARY evidence (margin_rel distributions) is deterministic bit-cost — sound. Only the wall-ceiling framing is flagged; a quiet profile could show a larger ceiling, which would RAISE the value of revisiting MAB ideas (the margins still argue against property-pruning specifically). | fold into #1's quiet profile. |
| 5 | **Oracle `encode_ms` columns** (arm-zen/arm-big TSVs) | self-contended ARM boxes, 4–7 worker threads on 4–8 cores | Bytes + butteraugli/ssim2 are sound (deterministic); encode_ms must not feed any wall model. aarch64 bytes additionally carry the ±tiny cross-ISA caveat (#70). | n/a for training (drop the column); local quiet re-measure if a wall model ever needs it. |
| 6 | **2c revalidation wall columns** (`/tmp/2c_revalidation.tsv` → benchmarks) | min-per-(cell,arm) of 2 process-serial passes, busy box | The 2c verdict consumes BYTES + SSIM2 (sound). Ignore its wall/encode_ms columns entirely. | only if a wall claim is ever attached to 2c. |

## NOT flagged (byte/deterministic evidence, load-immune)

- #69 item 1 LZ77 verdict (129/129 byte-identical; −87.8 % synthetic cell).
- #69 item 2 palette verdict (plots −30 % ×3 efforts; 126/129 identical;
  gray ≡ ChannelCompact byte-proof).
- #69 item 3 patches verdict (37.3 % A/B bytes).
- MABSplit margin_rel distributions (deterministic bit costs).
- #70 cross-ISA divergence facts (byte counts + hashes).
- gpu-butteraugli cubecl unification (compile-only).
- Hash-lock / byte-lock / suite results (deterministic).

## Standing rule going forward

Wall-clock dispositions require: zenbench (or zenbench-grade interleaved
paired methodology), loadavg < 1, nothing else running, ≥3 iterations, and
the repo's quiet-box discipline. Ad-hoc medians under load are
exploration-only.
