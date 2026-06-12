# GOAL: Strict per-scenario Pareto dominance over cjxl

**Adopted 2026-06-12.** Tracking issue: imazen/jxl-encoder#74.

> For every cell of the canonical scenario matrix, jxl-encoder must be
> **no worse on any axis and better on at least one** of:
>
> 1. **Encoded bytes**
> 2. **Perceptual quality** — butteraugli AND SSIM2 must both agree
>    (two-metric guard; HDR cells use PQ-EOTF butteraugli @ display nits)
> 3. **Wall time** at matched thread count (initial budget: ≤ 1.2× cjxl
>    at e ≤ 7, ≤ 1.0× at e ≥ 8; ratchet down as cells close)
>
> verified by decode through **djxl + jxl-rs + jxl-oxide**. A cell is
> WON only when all axes hold. The goal is **zero cells where cjxl
> dominates us**, then maximize cells where we dominate cjxl.

## Why this formulation

- **Per-cell, never on averages.** Averages hide cells: smooth-sky HDR
  sat at +57 % bytes while photo means looked fine. cjxl's strength is
  that it is never embarrassing anywhere; beating it means inheriting
  that property.
- **Pareto, not single-axis.** "Smaller" alone is trivial (drop
  quality); "faster" alone is trivial (drop effort). Wins must hold the
  other axes — the QuantizeWP flip is the template: −22 % bytes on the
  worst cell at quality still at-or-better than cjxl.
- **Two-metric quality guard** blocks metric-gaming (repo doctrine).

## Scenario matrix

| Axis | Cells |
|---|---|
| Content | photo, screenshot/UI, document/scan, plot/line-art, illustration, texture, **smooth-gradient/sky** (own class — historical loss locus) |
| Depth/color | 8-bit sRGB, 16-bit, PQ HDR, HLG HDR, gray, +alpha, premultiplied |
| Mode | lossy d0.25–5 (dense, low-q included per sweep discipline), lossless, JPEG-transcode, animation, progressive |
| Effort | e1–e3 (fast), e5–e7 (default), e8–e9 (quality) |
| Size | 64² (fixed-overhead regime), 256², 1 MP, 4 MP+; multi-group always |
| Threads | 1T and 8T walls — quiet-box zenbench-grade only |

Corpora: imazen-26 strata (content axis), hdr-crops-512 (HDR),
CID22/CLIC (photo continuity), gb82-sc (W44 continuity),
`benchmarks/lossless_bench_set_2026-06-10.tsv` (lossless walls).

## Scoreboard

A nightly/on-demand harness runs the matrix and emits one table:
per cell `WE-DOMINATE / TIE / CJXL-DOMINATES` with axis deltas.
Every CJXL-DOMINATES cell becomes a numbered wedge investigation
(the W44-ledger pattern: 50 open parity cells → 0).

Existing pieces to unify under one runner: `just quality-compare`
(CID22 SDR), `scripts/hdr_quantize_wp_ab.py` (HDR lossy),
`scripts/bench_lossless_ab.py` (lossless walls + bytes),
`just rd-regression` (regression floor).

## Per-cell playbook (in order of cheapness)

1. **Port what cjxl does that we don't.** Free wins sitting in libjxl
   source, found by section-diffing one bad cell (`jxl-oxide info
   --with-offset` both encoders, diff the parts). 2026-06-12 yielded
   three in one day: QuantizeWP DC shaping, prefix-vs-ANS auto choice,
   HLG forward OOTF.
2. **Fixed-overhead hunting** for small-file cells (per-stream ANS
   state flushes, empty-section parity, LfGlobal header deltas). At
   64²–512² the intercept dominates the slope.
3. **Our-side lifts where parity isn't enough** — zenanalyze dispatch,
   patches, learned pickers, content-class gates. Parity ports can only
   tie; these flip TIE → WE-DOMINATE.

## Standings at adoption (2026-06-12)

| Cell family | Status |
|---|---|
| Lossless photos e7 | **WE WIN** (−0.7 % bytes) |
| Lossless screenshots | **WE WIN** (patches, −36.7 % corpus) |
| SDR lossy photos e7 (CID22 369 cells) | quality better, +0.4 % bytes → near-tie |
| JPEG transcode | +0.115 % (200-file, byte-exact recon) |
| HDR lossy e5/e7 | quality at-or-better, **+1.2..+4.6 % bytes median**; smooth-sky residual +17..21 % on 1–4 KB files (DC-stream wedge, LfGroup 1256 vs 1024 B) |
| 16-bit lossless e5/e6 | e6 wins (−3.4 %), e5 +6.8 % |
| 16-bit lossless e2/e4 | **CJXL WINS** (+9..20 %) — open wedge (#72) |
| 16-bit lossless e7+ wall | **CJXL WINS** (18–24×) — worst wall cell (#72) |
| Lossless walls (8-bit, e5–e9) | ~1.16× at 8T post-#41/#64 arcs |
| alpha_distance vs cjxl-default | gap = unported Squeeze-on-extras |

Wall axis is the long pole — hence the ratcheted budget rather than
day-one dominance.
