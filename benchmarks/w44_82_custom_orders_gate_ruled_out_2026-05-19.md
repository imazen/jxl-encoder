# W44-82 — Custom-Orders Cost-Gate Removal RULED OUT (issue #60)

**Date**: 2026-05-19
**Workspace**: `~/work/zen/jxl-encoder--w44-82-custom-orders-gate`
**Status**: [RULED OUT] — gate is load-bearing, restored to source.
**Hypothesis source**: W44-81 audit item #1
  (`benchmarks/w44_81_residual_mechanism_survey_2026-05-19.md`).
**Time**: ~3 hours read + measure + write-up.

## TL;DR

The W44-81 audit predicted removing the Lehmer-cost-vs-AC-savings gate
in `compute_custom_orders` (`vardct/coeff_order.rs:410-487`, added by
commit `70b1a18`) would close 4-8 F-D OPEN cells at -100 to -400 B
each. **Measured paired bench shows the opposite**: removing the gate
regresses bytes by +1595 B total across 20 cells (5 photos × e7 ×
d∈{3,4,5,6}), with +1437 B on the 16 OPEN cells specifically. **0
OPEN→FIXED flips, 1 FIXED→OPEN regression**. The gate is correctly
defensive — at high d on smooth content the Lehmer code overhead
really does exceed the per-position zero-rate savings, and libjxl is
implicitly paying that cost.

## Measurement

`benchmarks/w44_82_custom_orders_gate_ab_2026-05-19.tsv` lists bytes
with gate removed; comparison vs W44-79 baseline ledger
(commit `0fb4854c` HEAD, gate ON):

| image    | d   | old (gate on) | new (gate off) | Δ      | Δ%      | cjxl  | new vs cjxl | status        |
|---       |---  |---:           |---:            |---:    |---:     |---:   |---:         |---            |
| 1420710  | 3.0 | 36505         | 36589          | +84    | +0.230% | 36567 | +0.060%     |               |
| 1420710  | 4.0 | 26783         | 28465          | +1682  | +6.280% | 27929 | +1.919%     | [OPEN]        |
| 1420710  | 5.0 | 24589         | 24672          | +83    | +0.338% | 23643 | +4.352%     | [OPEN]        |
| 1420710  | 6.0 | 21212         | 21256          | +44    | +0.207% | 20653 | +2.920%     | [OPEN]        |
| 1531677  | 3.0 | 32086         | 32424          | +338   | +1.053% | 32698 | -0.838%     | [OPEN→FIXED]  |
| 1531677  | 4.0 | 24614         | 25078          | +464   | +1.885% | 24918 | +0.642%     |               |
| 1531677  | 5.0 | 21128         | 21063          | -65    | -0.308% | 20620 | +2.148%     | [OPEN]        |
| 1531677  | 6.0 | 18227         | 18141          | -86    | -0.472% | 17705 | +2.463%     | [OPEN]        |
| 1189261  | 3.0 | 30496         | 30481          | -15    | -0.049% | 29197 | +4.398%     | [OPEN]        |
| 1189261  | 4.0 | 24416         | 24458          | +42    | +0.172% | 23751 | +2.977%     | [OPEN]        |
| 1189261  | 5.0 | 19871         | 20076          | +205   | +1.032% | 19385 | +3.565%     | [OPEN]        |
| 1189261  | 6.0 | 17043         | 17074          | +31    | +0.182% | 16884 | +1.125%     |               |
| 1025469  | 3.0 | 16323         | 16324          | +1     | +0.006% | 16641 | -1.905%     | [OPEN]        |
| 1025469  | 4.0 | 13452         | 13031          | -421   | -3.130% | 13468 | -3.245%     | [FIXED→OPEN]  |
| 1025469  | 5.0 | 10901         | 10789          | -112   | -1.027% | 11126 | -3.029%     | [OPEN]        |
| 1025469  | 6.0 | 9614          | 9145           | -469   | -4.878% | 9848  | -7.139%     | [OPEN]        |
| 1418519  | 3.0 | 10940         | 10930          | -10    | -0.091% | 11256 | -2.896%     | [OPEN]        |
| 1418519  | 4.0 | 9030          | 9041           | +11    | +0.122% | 9351  | -3.315%     | [OPEN]        |
| 1418519  | 5.0 | 7695          | 7670           | -25    | -0.325% | 8185  | -6.292%     | [OPEN]        |
| 1418519  | 6.0 | 6792          | 6605           | -187   | -2.753% | 7233  | -8.682%     | [OPEN]        |

**Aggregate**:
- Total bytes Δ: **+1595 B**
- OPEN cells (16) Δ: **+1437 B**
- OPEN→FIXED flips: **1** (1531677 d=3)
- FIXED→OPEN regressions: **1** (1025469 d=4)
- Hash-lock impact: 4 lossy sidecars (`lossy_defaults_rgb_48x48_noise`,
  `lossy_with_noise`, `lossy_with_lz77_rle`, `lossy_with_lz77_greedy`)
  all regress by +47 B (3254 vs 3207) — confirms gate is doing real
  work on the small noise fixture as well.

## Acceptance gates (per task brief)

- (a) Gate behavior change documented: ✅ (Option A — removed entirely)
- (b) F-D cells: at least 4 OPEN→FIXED OR avg -100 B per cell on 13
  photo OPEN cells: **FAIL** — only 1 OPEN→FIXED, +95 B avg on the
  16-cell OPEN subset.
- (c) NO FIXED→OPEN regression on photo OR screenshot OR windows95:
  **FAIL** — 1025469 d=4 flipped FIXED→OPEN.

Both core gates failed. Gate restored to source. **Issue #60 closed as
RULED OUT (not as shipped)**.

## Why the audit was wrong

W44-81 inferred that the +22442 ZeroDensity tokens observed in W44-75
were caused by our scan order being natural while cjxl's was the
post-quant-stable custom order. The reasoning was: more positions
scanned before `nzeros` hits 0 → more ZeroDensity-context tokens per
block.

**That inference doesn't hold** on these particular cells: the Lehmer
code overhead at the chosen bucket sizes (DCT16x16=256 positions,
DCT32x32=1024 positions) is significant (~30-60 bits per channel for
the end marker + Lehmer entries on a sparse permutation). At d=4-6 on
smooth photos, the marginal AC-savings per block from custom orders is
sub-bit on most buckets, so the per-bucket aggregate is roughly
balanced. Our gate correctly identifies when savings < cost.

The +22442 ZeroDensity token gap from W44-75 must be attributed to a
DIFFERENT mechanism — most likely W44-81's #2 (BlockCtxMap
clustering), #4 (kAvoidEntropyOfTransforms double-count in
find_best_16x16_transform), or #5 (max_histograms cap). W44-83+
should investigate those before re-touching the order gate.

## Why libjxl appears to "always emit" custom orders

Two reasons our removal didn't match cjxl's output:

1. **Small-image gate**: libjxl's `ComputeUsedOrders`
   (`enc_coeff_order.cc:62`) returns `used_orders = 0` when
   `xsize_blocks < 5 && ysize_blocks < 5`. Our 512×512 cells (64×64
   blocks) don't trigger this, but it's the structural reason the
   `is_nondefault` gate alone is sufficient in libjxl: small images
   bypass the path entirely.

2. **Implicit RD cost**: libjxl pays the Lehmer-code overhead on all
   non-small images regardless of marginal AC savings. The cjxl
   measurement at these specific F-D cells shows cjxl's overall bytes
   are LARGER on some of these cells than what our gate produces
   (1025469 d=4-6 and 1418519 d=3-6 are net WINS for ours, including
   FIXED→OPEN cells), suggesting the gate is a small Pareto win on
   smooth photos.

## What changed in source

No production code changed at conclusion — the gate was restored to
its pre-W44-82 state with an updated comment block citing the W44-82
ruling and pointing to the bench data. The diagnostic example
(`examples/w44_82_custom_orders_gate_ab.rs`) and bench TSV/meta are
shipped so future agents don't re-investigate.

## Files committed

- `jxl-encoder/src/vardct/coeff_order.rs` (comment-only update on the
  gate; logic unchanged)
- `jxl-encoder/examples/w44_82_custom_orders_gate_ab.rs` (new)
- `jxl-encoder/Cargo.toml` (register the new example)
- `benchmarks/w44_82_custom_orders_gate_ab_2026-05-19.tsv` + `.meta`
- `benchmarks/w44_82_compare_script.py`
- `benchmarks/w44_82_custom_orders_gate_ruled_out_2026-05-19.md` (this
  file)

## Recommended next chunk

W44-81 audit item #4 — verify and (if confirmed) remove the
`kAvoidEntropyOfTransforms` double-count in `find_best_16x16_transform`
(`vardct/ac_strategy_search.rs:2495-2502`). Per W44-81 note this is
NOT verified at line-by-line parity yet; the 32x32 fork was audited
but the 16x16 fork was not. Expected EV -100 to -500 B on F-D cells,
5-10 LOC.

## Cross-references

- W44-81 audit (the hypothesis source):
  `benchmarks/w44_81_residual_mechanism_survey_2026-05-19.md`
- W44-79 baseline ledger:
  `benchmarks/cjxl_parity_ledger_2026-05-19_w44_79.tsv`
- libjxl reference: `enc_coeff_order.cc:37-237`
  (`~/work/jxl-efforts/libjxl/lib/jxl/`)
- Original gate add: commit `70b1a18`
