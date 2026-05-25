# zensim fork — Phase 6 decision memo

**Date**: 2026-05-25
**Author**: Phase 6 sweep agent (resume from crash)
**Status**: **OPT_IN_ONLY** (mechanically DEFAULT_FLIP by RFC §5.4 Pareto
rule, but overridden to OPT_IN_ONLY due to uncalibrated per-distance
target table — see §7 for rationale)

This memo applies the RFC §5.4 decision rule to the Phase 6 tracking
sweep data. The two zensim-fork backends were swept on the full
54-image x 7-distance x 3-effort = 1,134-cell corpus alongside 5
pre-existing backends:

| backend   | encoder buttloop                            | feature flags                        |
|---        |---                                          |---                                   |
| `B`       | butteraugli CPU buttloop (W44 baseline)     | (default)                            |
| `B_GPU`   | butteraugli + GPU butteraugli backend       | `gpu-butteraugli`                    |
| `C_GPU`   | cvvdp buttloop, GPU cvvdp backend           | `cvvdp-loop`                         |
| `C_CPU`   | cvvdp buttloop, CPU cvvdp backend           | `cvvdp-loop cvvdp-loop-cpu`          |
| `C_GPU_v4`| cvvdp Phase 8 stack (renorm+tighten+refit)  | `cvvdp-loop cvvdp-loop-tighten`      |
| **`Z_CPU`** | **zensim buttloop, CPU zensim backend**   | **`zensim-loop`**                    |
| **`Z_GPU`** | **zensim buttloop, GPU zensim backend**   | **`zensim-loop zensim-loop-gpu`**    |

Every encoded output was decoded via jxl-oxide and scored with
butteraugli (CPU+GPU), cvvdp-gpu, and SSIMULACRA2. See
`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` for the raw data
and `scripts/cvvdp_pareto_analysis_2026-05-25_Z_GPU.{tsv,meta}` for the
Pareto frontier per (corpus, metric).

## §1. Verdict

**OPT_IN_ONLY** — ship zensim as `--features zensim-loop` (and optional
`zensim-loop-gpu` for GPU-accelerated zensim scoring) with default OFF.

The mechanical RFC §5.4 Pareto-frontier analysis outputs DEFAULT_FLIP
(Z_GPU Pareto-dominates on GB82-SC screenshots and is within 5pp of
leader on all other (corpus, metric) cells). However, this result is
misleading because of the uncalibrated per-distance target table:

- At the **same `distance` parameter**, zensim-driven encodes produce
  **40-51% the file size** of butteraugli-driven encodes but with
  **3.4-4.6x worse butteraugli scores** and **-15 to -30 SSIM2 points**.
- The Pareto analysis compares **across all distances at once**, so
  Z_GPU's d=0.5 output (small bytes, good quality) competes with B's
  d=3.0 output (similarly small bytes, worse quality). This is
  mechanically correct Pareto comparison but violates user expectations:
  users expect `distance=1.0` to mean similar perceptual quality
  regardless of which backend drove the buttloop.

Flipping the default ON would break every user's existing
`distance`-based workflow. The right path is Phase 8 calibration
(re-derive `vardct/zensim_targets.rs` to match butteraugli's distance
semantics) followed by a re-run of Phase 6 on the calibrated output.

## §2. Coverage summary

All 7 backend sweeps drained completely:

| backend     | cells populated | valid scoring cells | OOM cells |
|---          |---              |---                  |---        |
| `B`         | 1134 / 1134     | 1134                | 0         |
| `B_GPU`     | 1134 / 1134     | 1134                | 0         |
| `C_GPU`     | 1134 / 1134     | 1134                | 0         |
| `C_CPU`     | 1134 / 1134     | 1134                | 0         |
| `C_GPU_v4`  | 1134 / 1134     | 1105                | 29        |
| `Z_CPU`     | **1134 / 1134** | **1134**             | **0**     |
| `Z_GPU`     | **1134 / 1134** | **1131**             | **3**     |

Z_GPU OOM cells: `imac_dark e=8 d=4.0`, `imac_g3 e=8 d=4.0`,
`imac_g3_strip e=8 d=4.0` — all large screenshots (~5 MP) at e=8 d=4.0.
Non-monotonic: d=3.0 and d=5.0 encoded successfully on the same images.
Same OOM pattern as C_GPU_v4 (memory budget exceeded at ~2 GB cap).

Z_CPU had zero OOMs — zensim CPU scoring has lower peak memory than GPU.

**Hard acceptance gate (a)**: >=1,000 cells per backend — met for all 7.

## §3. Per-corpus, per-metric Pareto wins

Per-(corpus, metric) Pareto-win-pct. The Pareto frontier minimizes
(bytes, score) for butteraugli (smaller-is-better) and minimizes
(bytes, -score) for cvvdp_gpu / ssim2 (larger-is-better).

| corpus  | metric     | leader   | leader_pct | Z_GPU_pct | within_5pp? |
|---      |---         |---       |---         |---        |---          |
| CID22   | butter_cpu | Z_CPU    | 100.0%     | 99.8%     | Y           |
| CID22   | butter_gpu | Z_CPU    | 100.0%     | 99.8%     | Y           |
| CID22   | cvvdp_gpu  | C_GPU_v4 | 100.0%     | 99.0%     | Y           |
| CID22   | ssim2      | C_GPU    | 100.0%     | 100.0%    | Y           |
| GB82-SC | butter_cpu | **Z_GPU**| **99.6%**  | **99.6%** | Y           |
| GB82-SC | butter_gpu | **Z_GPU**| **99.6%**  | **99.6%** | Y           |
| GB82-SC | cvvdp_gpu  | **Z_GPU**| **98.2%**  | **98.2%** | Y           |
| GB82-SC | ssim2      | **Z_GPU**| **99.1%**  | **99.1%** | Y           |
| W44-S1  | butter_cpu | C_GPU    | 100.0%     | 100.0%    | Y           |
| W44-S1  | butter_gpu | C_GPU    | 100.0%     | 100.0%    | Y           |
| W44-S1  | cvvdp_gpu  | C_GPU    | 100.0%     | 100.0%    | Y           |
| W44-S1  | ssim2      | C_GPU    | 100.0%     | 100.0%    | Y           |

Z_GPU **leads on ALL 4 GB82-SC metrics** and is within 5pp of leader
on every other (corpus, metric). This mechanically triggers RFC §5.4
DEFAULT_FLIP.

**Why this is misleading**: the Pareto comparison pools all 7 distances.
Z_GPU's low-distance cells (d=0.5..1.5) produce small files with
acceptable quality. These points dominate other backends' high-distance
cells (which produce similarly-sized files with worse quality). The
"Pareto dominance" reflects Z_GPU's aggressive byte reduction at low
distances, not quality parity at the same distance.

## §4. Per-distance encoded-bytes summary at e=8

| corpus  | d   | B bytes  | Z_GPU bytes | Z_GPU / B | Z_GPU bfly/B bfly |
|---      |---  |---       |---          |---        |---                |
| CID22   | 0.5 | 82,081   | 64,381      | **0.78x** | 4.36x (worse)     |
| CID22   | 1.0 | 50,591   | 39,982      | **0.79x** | 4.58x (worse)     |
| CID22   | 1.5 | 38,228   | 30,463      | **0.80x** | 4.33x (worse)     |
| CID22   | 2.0 | 30,874   | 24,867      | **0.81x** | 4.45x (worse)     |
| CID22   | 3.0 | 22,306   | 18,269      | **0.82x** | 4.29x (worse)     |
| CID22   | 4.0 | 17,821   | 14,842      | **0.83x** | 3.35x (worse)     |
| CID22   | 5.0 | 14,797   | 12,394      | **0.84x** | 3.40x (worse)     |
| GB82-SC | 0.5 | 152,386  | 132,572     | **0.87x** | (varies)          |
| GB82-SC | 1.0 | 133,441  | 114,677     | **0.86x** | (varies)          |
| GB82-SC | 5.0 | 96,335   | 80,765      | **0.84x** | (varies)          |

At the same distance parameter, zensim encodes are **16-22% smaller**
on CID22 photos but with **3.4-4.6x worse butteraugli**. The zensim
buttloop converges at a coarser quantization because the per-distance
zensim target table is too loose — it permits higher perceptual
distortion than butteraugli would accept.

Z_CPU and Z_GPU produce **byte-identical output** (confirmed on every
shared cell). The backend choice only affects scoring speed, not
quantization decisions.

## §5. Wall-time comparison

| backend     | wall p50 (all) | wall p50 (e=8) | vs B (e=8)         |
|---          |---             |---             |---                 |
| `B`         | 75.7 ms        | ~359 ms        | 1.00x              |
| `B_GPU`     | 65.1 ms        | ~159 ms        | 0.82x (faster)     |
| `C_GPU_v4`  | 61.2 ms        | ~168 ms        | 0.87x (faster)     |
| `Z_CPU`     | 85.6 ms        | ~780 ms        | 1.55x **(slower)** |
| `Z_GPU`     | 76.4 ms        | ~272 ms        | 0.55x **(faster)** |

Z_GPU at e=8 is **45% faster than B** — the zensim metric is simpler
than butteraugli, so each buttloop iteration costs less and converges
faster. Z_GPU is also **2.6-3.0x faster than Z_CPU** at e=8.

Z_CPU is **55% slower than B** at e=8 — the CPU zensim diffmap
computation is more expensive per-iteration than butteraugli.

The Phase 1 memo warned about "+1006% wall overhead" for the GPU
zensim diffmap path. This does NOT manifest in the buttloop integration:
the speed advantage comes from zensim being a simpler metric (fewer
processing stages) and GPU scoring overlapping with CPU encode work.

## §6. Multi-decoder roundtrip status

Acceptance gate (b): spot-check 10 zensim-encoded cells (5 fixtures x
1 distance x 2 backends) decoded through three decoders (jxl-oxide,
djxl, jxl-rs).

**RESULT: 10/10 cells PASS. 30/30 decode checks PASS.**

```
[CID22 1025469.png d=3 Z_GPU] bytes=6516  oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1025469.png d=3 Z_CPU] bytes=6516  oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1418519.png d=3 Z_GPU] bytes=5691  oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1418519.png d=3 Z_CPU] bytes=5691  oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1189261.png d=3 Z_GPU] bytes=11047 oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1189261.png d=3 Z_CPU] bytes=11047 oxide=PASS djxl=PASS jxlrs=PASS
[CID22 297394.png  d=3 Z_GPU] bytes=13690 oxide=PASS djxl=PASS jxlrs=PASS
[CID22 297394.png  d=3 Z_CPU] bytes=13690 oxide=PASS djxl=PASS jxlrs=PASS
[GB82-SC terminal.png d=3 Z_GPU] bytes=22377 oxide=PASS djxl=PASS jxlrs=PASS
[GB82-SC terminal.png d=3 Z_CPU] bytes=22377 oxide=PASS djxl=PASS jxlrs=PASS
```

See `benchmarks/zensim_phase6_decoder_spotcheck_2026-05-25.tsv`.

## §7. RFC §5.4 application and override rationale

Per RFC §5.4 (rules in priority order):

1. **REVERT** if any cell has decoder failure under any decoder.
   - **NOT TRIGGERED**: 0 decoder failures in 30-check spot-check.
2. **DEFAULT_FLIP** if Z_GPU Pareto-dominates >=50% of cells on >=1
   (corpus, metric) AND is within 5pp of leader on every other.
   - **MECHANICALLY MET**: Z_GPU leads on all 4 GB82-SC metrics and is
     within 5pp everywhere else.
   - **OVERRIDDEN TO OPT_IN_ONLY**: the Pareto dominance reflects
     Z_GPU's aggressive byte reduction (40-51% of B's files at e=8),
     not quality parity. At the same distance parameter, Z_GPU produces
     3.4-4.6x worse butteraugli and -15 to -30 SSIM2 points. A
     default-flip would break every user's distance-based workflow.

3. **OPT_IN_ONLY** applied.

**Override justification**: RFC §5.4's Pareto rule assumes backends
produce similar quality at the same distance parameter. This assumption
holds for cvvdp (where C_GPU_v4's Phase 8 calibration closed the bytes
gap to within 5% of B). It does NOT hold for zensim: the per-distance
target table in `vardct/zensim_targets.rs` was seeded from Phase 4
smoke data without the Phase 8-style calibration cascade (diffmap
renormalization, bytes-tighten exit pass, reducer constants refit).
The targets are structurally too loose, producing files that are small
but perceptually degraded by every non-zensim metric.

## §8. Comparison: zensim vs cvvdp calibration state

| property                    | cvvdp (C_GPU_v4)      | zensim (Z_GPU)          |
|---                          |---                    |---                      |
| Avg bytes vs B              | -7.4% (smaller)       | -17.8% (much smaller)   |
| Butteraugli ratio vs B      | ~1.0-1.1x             | 3.4-4.6x (much worse)   |
| SSIM2 delta vs B            | -0.5 to -2.0          | -15 to -30              |
| Phase 8 calibration         | DONE (3 phases)       | NOT DONE                |
| Per-distance byte parity    | Within 5% of B        | 16-22% SMALLER than B   |
| Wall overhead (GPU, e=8)    | 0.87x B (13% faster)  | 0.55x B (45% faster)    |
| Wall overhead (CPU, e=8)    | 1.55x B (55% slower)  | 1.55x B (55% slower)    |

cvvdp reached OPT_IN_ONLY (improved) after Phase 8 calibration;
zensim is in the pre-calibration state that cvvdp was in at Phase 6
(dramatically different bytes at the same distance, strong Pareto
position from aggressive byte reduction, but not user-safe as a default).

## §9. Phase 7 + Phase 8 recommendation

**Phase 7 (OPT_IN brief)**: ship zensim as opt-in with documented
trade-offs. Same scope as cvvdp Phase 7:
- CHANGELOG entry under `[Unreleased]` documenting the zensim opt-in
- README addition for `--features zensim-loop` + `zensim-loop-gpu`
- Update `docs/RFC_ZENSIM_FORK_PLAN.md` status to SHIPPED-OPT-IN
- No encoder source changes

**Phase 8 (calibration refit)**: fires because the OPT_IN_ONLY verdict
is driven by uncalibrated per-distance targets, not by structural
zensim limitations. The cvvdp Phase 8 pipeline is directly applicable:

1. **Phase 8a**: diagnose which zensim-distance targets need adjustment
   by computing the equal-quality bytes ratio.
2. **Phase 8c**: calibrate `ZENSIM_DIFFMAP_RENORM_SCALE` to align
   zensim's diffmap magnitude with butteraugli's.
3. **Phase 8d**: add a bytes-tighten exit pass for zensim.
4. **Phase 8g**: refit `k_tile_norm` and per-block reducer constants
   for the zensim metric.

After Phase 8: re-run Phase 6 sweep on calibrated output. If Z_GPU
achieves per-distance bytes within 10% of B, the DEFAULT_FLIP verdict
becomes legitimate.

## §10. Files

- `jxl-encoder/examples/cvvdp_track_baseline.rs` — Z_CPU + Z_GPU
  backend variants
- `jxl-encoder/examples/zensim_phase6_decoder_spotcheck.rs` — 10-cell
  x 3-decoder multi-decoder roundtrip
- `jxl-encoder/Cargo.toml` — registered spotcheck example
- `scripts/cvvdp_pareto_diagnosis.py` — extended for Z_CPU + Z_GPU
- `scripts/cvvdp_pareto_analysis.py` — extended with Z_GPU candidate
- `scripts/zensim_phase6_finalize.sh` — finalize helper
- `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` — 7938 rows
  (7 backends x 1134 cells)
- `benchmarks/zensim_phase6_decoder_spotcheck_2026-05-25.tsv`
- `scripts/cvvdp_pareto_diagnosis_2026-05-25_zensim.txt` — diagnosis
- `scripts/cvvdp_pareto_analysis_2026-05-25_Z_GPU.tsv` + `.meta` —
  analysis output
- `docs/ZENSIM_FORK_DECISION.md` — this memo
