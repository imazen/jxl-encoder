# Shipped-loop redistribution vs independent judges — RD gate (2026-08-27)

The LOOPS production-gate criterion demands **RD >= baseline under judges that
are never the steering metric**. For six of the seven encoder lines the gate
holds BY CONSTRUCTION (pure-native-dial loops; ruling in zensim
`benchmarks/loops_production_gates_2026-08-27.md`). jxl-encoder is the ONE
line whose SHIPPED loop steers encoder internals with zensim-derived signal:
the Trained-diffmap per-tile redistribution (`zensim_loop.rs`: alpha 0.25,
factor cap 1.15, ON by default). Its independent-judge evidence dates from the
2026-03-08 tuning sweep (+1.262 SSIM2 @ +1.10% size e7-zen4; e8-zen4 −4.61%
bytes at SSIM2 parity, 20 images). This wave re-measures on the CURRENT loop.

## Design (FROZEN pre-run)

- **Arms differ in ONE env var** (zero code change):
  - **R** (shipped): defaults — redistribution active (α=0.25, cap 1.15).
  - **N** (neutralized): `ZENSIM_FACTOR_MAX=1.0` — the clamp
    `(1+kα·r).clamp(1/fmax, fmax)` pins every factor to 1.0
    (zensim_loop.rs:1574/1619/1671 verified), quant field untouched;
    outer controller + everything else identical.
- **Cells**: corpus9 (the S4 census corpus, 9 SDR refs) × t ∈ {70,80,88},
  `--arms baseline --iters 3` (k3), `JXL_ZENSIM_EMIT_BEST=1`,
  `JXL_ZENSIM_TARGET_TOL=-1`, v47 bake (the loop's pinned Profile A) — the
  exact S4-census loop shape.
- **Judges (never zensim)**: ssim2 (GPU) + butteraugli-max (GPU) on
  (ref, decoded-best) pairs; bytes from the loop manifest.
- **Paired per-cell deltas** Δ = R − N: Δssim2 (higher better),
  Δbutter_max (lower better), Δbytes% ; plus Δ|zensim target err| as context
  (the benefit axis, NOT a gate).

## Gates (FROZEN)

- **G-V**: 54/54 cells emit decoded output + judge cleanly.
- **G-RD1** (aggregate, per judge): NOT RD-dominated — reject if
  (median Δjudge is worse AND median Δbytes% > 0). Worse = Δssim2 < 0,
  Δbutter_max > 0.
- **G-RD2** (cell level, per judge): strictly-dominated cells (judge worse
  AND bytes larger) ≤ 25%.
- PASS = both gates for BOTH judges. FAIL → the redistribution default
  becomes a PROPOSAL to revisit (user-gated; no default flip here).

## Results — FULL PASS (all four gates, both judges)

**Runtime amendment (pre-judging, recorded):** judges ran the CPU
implementations (`ssim2`, `butteraugli`) — this distro has no CUDA userspace
(the fleet's GPU-only rule is a criterion-1 DATA hygiene rule; for a paired
local A/B the CPU implementations compute the same metrics and any impl bias
cancels across arms). Metric identity unchanged from registration.

n=27 paired cells (G-V PASS; 27/27 both arms emitted + judged):

| paired median | value | reading |
|---|---|---|
| ΔSSIM2 (R−N) | **+0.138** | redistribution IMPROVES the independent judge |
| Δbutter_max | +0.0128 | marginal worsening (lower-better metric) |
| Δbutter_pnorm3 | +0.0025 | ~noise |
| Δbytes % | **−0.91%** | redistribution SHRINKS output |
| Δ\|zensim err\| | +0.031 | target-hitting unchanged |

- **G-RD1 ssim2 PASS** (better judge AND fewer bytes — RD-POSITIVE, not merely
  not-dominated); **G-RD1 butter PASS** (bytes decreased).
- **G-RD2**: strictly-dominated cells 0/27 (ssim2), 3/27 (butter_max) vs the
  ≤25% (6.75) bar — PASS both.
- Modern confirmation of the 2026-03-08 tuning-sweep result (+1.262 SSIM2 @
  +1.10% size; e8 −4.61% bytes at parity): the shipped zensim redistribution
  does not game zensim at the independent judges' expense — it buys ssim2
  quality per byte. Per-cell table:
  `benchmarks/zensim_loop_rd_deltas_2026-08-27.tsv`; raw scores + decoded
  arms: `/mnt/v/output/jxl-encoder/rd-judges-2026-08-27/`.

