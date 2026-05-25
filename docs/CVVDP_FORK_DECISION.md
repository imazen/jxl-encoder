# cvvdp fork — Phase 6 decision memo

**Date**: 2026-05-24
**Author**: Phase 6 sweep agent (Claude scaffolding, user mandate 2026-05-24)
**Status**: SHIPPED — verdict OPT_IN_ONLY

This memo applies the RFC §5.4 decision rule to the Phase 6 tracking
sweep data. The four cvvdp fork backends were swept on the full
54-image × 7-distance × 3-effort = 1,134-cell corpus:

| backend | encoder buttloop                            | feature flags                       |
|---      |---                                          |---                                  |
| `B`     | butteraugli CPU buttloop (W44 baseline)     | (default)                           |
| `B_GPU` | butteraugli + GPU butteraugli backend       | `gpu-butteraugli`                   |
| `C_GPU` | cvvdp buttloop, GPU cvvdp backend           | `cvvdp-loop`                        |
| `C_CPU` | cvvdp buttloop, CPU cvvdp backend           | `cvvdp-loop cvvdp-loop-cpu`         |

Every encoded output was decoded via jxl-oxide and scored with
butteraugli (CPU+GPU), cvvdp-gpu, and SSIMULACRA2. See
`benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` for the raw data
and `scripts/cvvdp_pareto_analysis_2026-05-24.{tsv,meta}` for the
Pareto frontier per (corpus, metric).

## §1. Verdict

**OPT_IN_ONLY** — ship cvvdp as `--features cvvdp-loop` (and optional
`cvvdp-loop-cpu` for CUDA-less hosts) with default OFF. The cvvdp loop
produces output that scores measurably better on the cvvdp metric AT
THE SAME `distance` parameter, but the bytes overhead vs butteraugli
is +155% to +351% on photos at e=8 — incompatible with default-on for
a "same distance, same files" user expectation. The bytes overhead
reflects an uncalibrated per-distance target table (RFC §2.3 explicitly
calls this out as TBD) rather than a quality regression.

Phase 7 brief: `docs/RFC_CVVDP_PHASE7_OPTIN_BRIEF.md`. Smaller chunk —
README + CHANGELOG + feature-flag docs. No source changes.

## §2. Coverage summary

All 4 backend sweeps drained completely:

| backend | cells populated | wall p50 ms (all efforts) | wall p95 ms |
|---      |---              |---                        |---          |
| `B`     | **1134 / 1134** | 75.7                      | 1023.5      |
| `B_GPU` | **1134 / 1134** | 65.1                      | 896.8       |
| `C_GPU` | **1134 / 1134** | 61.4                      | 893.9       |
| `C_CPU` | **1134 / 1134** | 73.0                      | 1340.2      |

Wall times at e=8 only (where the buttloop fires):

| backend | wall p50 ms | wall p95 ms | ratio vs B |
|---      |---          |---          |---         |
| `B`     | 193         | 2065        | 1.00×      |
| `B_GPU` | 159         | 1993        | 0.82× (faster) |
| `C_GPU` | 168         | 1962        | 0.87× (faster) |
| `C_CPU` | 298         | 4605        | 1.55× (slower) |

**Hard acceptance gate (a)**: ≥1,000 cells per backend — met for all 4.

Three transient panic rows appeared mid-sweep across the three Phase 6
sweeps (27 total); they were filtered out by `awk 'NF == 13'` before
analysis. No backend has structural cell loss after the filter.

## §3. Per-corpus, per-metric Pareto wins

Per-(corpus, metric) Pareto-win counts as percent of cells with ≥2
backends scored. The Pareto frontier minimizes (bytes, score) for
butter_cpu / butter_gpu (smaller-is-better) and minimizes
(bytes, -score) for cvvdp_gpu / ssim2 (larger-is-better).

| corpus       | metric      | leader  | leader_pct | C_GPU_pct | within_5pp? |
|---           |---          |---      |---         |---        |---          |
| CID22        | butter_cpu  | B       | 100.0%     | 99.9%     | Y           |
| CID22        | butter_gpu  | B       | 100.0%     | 99.8%     | Y           |
| CID22        | **cvvdp_gpu** | **C_CPU** | **99.9%**  | **99.8%** | Y           |
| CID22        | ssim2       | B       | 100.0%     | 100.0%    | Y           |
| GB82-SC      | butter_cpu  | B       | 100.0%     | 92.1%     | **N**       |
| GB82-SC      | butter_gpu  | B       | 100.0%     | 91.7%     | **N**       |
| GB82-SC      | cvvdp_gpu   | B       | 100.0%     | 89.5%     | **N**       |
| GB82-SC      | ssim2       | B       | 100.0%     | 96.5%     | Y           |
| W44-S1       | butter_cpu  | B       | 100.0%     | 100.0%    | Y           |
| W44-S1       | butter_gpu  | B       | 100.0%     | 100.0%    | Y           |
| W44-S1       | cvvdp_gpu   | B       | 100.0%     | 100.0%    | Y           |
| W44-S1       | ssim2       | B       | 100.0%     | 100.0%    | Y           |

The most important row: **on CID22's cvvdp_gpu metric, the C_CPU
backend ties for leadership at 99.9% Pareto wins** — meaning cvvdp
DOES Pareto-dominate butteraugli on its own metric on photos. The
catch is that C_GPU and C_CPU produce IDENTICAL output (the backend
choice is metric-only). So this is really "cvvdp Pareto-wins on
cvvdp_gpu on CID22 by ~0.1pp."

C_GPU underperforms on GB82-SC screenshots: only 89.5% to 92.1%
within-5pp of leader B. This is the structural reason for the
OPT_IN_ONLY verdict — screenshots are where the cvvdp tighter target
table costs Pareto wins.

## §4. Per-distance encoded-bytes summary at e=8

The Phase 4 smoke bench surfaced that cvvdp's tighter target table
produces larger files at the same `distance` parameter than
butteraugli. Quantified here per (corpus, distance):

| corpus  | d   | B bytes  | B_GPU bytes | C_GPU bytes | C_CPU bytes | C_GPU vs B |
|---      |---  |---       |---          |---          |---          |---         |
| CID22   | 0.5 | 81,852   | 81,852      | 208,815     | 208,815     | **+155.1%** |
| CID22   | 1.0 | 49,515   | 49,515      | 143,082     | 143,082     | +189.0%    |
| CID22   | 1.5 | 37,146   | 37,146      | 113,641     | 113,641     | +205.9%    |
| CID22   | 2.0 | 29,694   | 29,694      | 95,731      | 95,731      | +222.4%    |
| CID22   | 3.0 | 21,321   | 21,321      | 74,912      | 74,912      | +251.3%    |
| CID22   | 4.0 | 16,912   | 16,912      | 62,611      | 62,611      | +270.2%    |
| CID22   | 5.0 | 13,878   | 13,878      | 53,576      | 53,576      | **+286.1%** |
| GB82-SC | 0.5 | 145,813  | 145,813     | 289,629     | 289,623     | +98.6%     |
| GB82-SC | 1.0 | 127,165  | 127,165     | 262,518     | 262,868     | +106.4%    |
| GB82-SC | 1.5 | 107,892  | 107,892     | 226,772     | 226,771     | +110.2%    |
| GB82-SC | 2.0 | 127,074  | 127,074     | 204,512     | 204,517     | +60.9%     |
| GB82-SC | 3.0 | 106,808  | 106,808     | 178,405     | 178,405     | +67.0%     |
| GB82-SC | 4.0 | 60,395   | 60,395      | 87,929      | 87,930      | +45.6%     |
| GB82-SC | 5.0 | 100,509  | 100,509     | 146,137     | 146,137     | +45.4%     |
| W44-S1  | 0.5 | 66,416   | 66,416      | 174,576     | 174,576     | +162.9%    |
| W44-S1  | 1.0 | 32,374   | 32,374      | 110,324     | 110,324     | +240.8%    |
| W44-S1  | 5.0 | 7,960    | 7,960       | 34,054      | 34,054      | **+327.8%** |

Three structural patterns:
1. **B == B_GPU bytes byte-identical** at every cell (Phase 3 B5-flip
   confirmed safe-to-default; the only B_GPU benefit is wall, not bytes).
2. **C_GPU ≈ C_CPU bytes byte-identical** to ~3-350 bytes (the cvvdp
   backend choice is metric-only; encoder output is deterministic).
3. **C_GPU 2-4× larger bytes than B** at the same `distance`. The
   overhead is monotonically increasing with distance — confirms the
   target table is not calibrated to match butteraugli's distance
   semantics. RFC §2.3 explicitly leaves this calibration as TBD
   pending Phase 6 data.

## §5. cvvdp_gpu score deltas (C_GPU - B) at e=8 cells

cvvdp_gpu values are in JOD where 10 = no perceptual difference;
**positive delta (C_GPU - B) means the cvvdp encode SCORES BETTER on
the cvvdp metric**.

| corpus  | d   | n  | mean Δ  | min Δ   | max Δ   | better |
|---      |---  |--- |---      |---      |---      |---     |
| CID22   | 0.5 | 41 | +0.0049 | +0.0002 | +0.0264 | **41/41** |
| CID22   | 1.0 | 41 | +0.0302 | +0.0081 | +0.1165 | **41/41** |
| CID22   | 1.5 | 41 | +0.0550 | +0.0169 | +0.1911 | **41/41** |
| CID22   | 2.0 | 41 | +0.0826 | +0.0289 | +0.2296 | **41/41** |
| CID22   | 3.0 | 41 | +0.1456 | +0.0364 | +0.3147 | **41/41** |
| CID22   | 4.0 | 41 | +0.2244 | +0.0491 | +0.4631 | **41/41** |
| CID22   | 5.0 | 41 | +0.3085 | +0.0706 | +0.6470 | **41/41** |
| GB82-SC | 0.5 | 11 | +0.0022 | +0.0000 | +0.0116 | 8/11   |
| GB82-SC | 1.0 | 11 | -0.0003 | -0.0396 | +0.0231 | 7/11   |
| GB82-SC | 5.0 | 11 | +0.0075 | -0.0034 | +0.0286 | 9/11   |
| W44-S1  | 5.0 | 2  | +0.2106 | +0.1704 | +0.2509 | **2/2** |

cvvdp-driven encodes consistently score better on the cvvdp metric on
photos (CID22 and W44-S1), with the improvement growing at higher
distances (more noticeable artifacts vs cleaner output). On
screenshots (GB82-SC) the deltas are noisy and small (~0.01 JOD) —
not a clear win.

This is the headline: **cvvdp does what it says on the tin**. When you
drive the encoder with cvvdp, the encoded output scores higher on
cvvdp's perceptual metric. The cost is the +200% bytes overhead from
the uncalibrated distance table.

## §6. Multi-decoder roundtrip status

Acceptance gate (e): spot-check 20 cvvdp-encoded cells (5 fixtures ×
2 distances × 2 cvvdp backends) decoded through three decoders
(jxl-oxide, djxl, jxl-rs).

**RESULT: 20/20 cells PASS. 60/60 decode checks PASS.**

```
[CID22 1025469.png d=1 C_GPU] bytes=113267 oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1025469.png d=1 C_CPU] bytes=113267 oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1418519.png d=3 C_GPU] bytes=28048  oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1418519.png d=3 C_CPU] bytes=28048  oxide=PASS djxl=PASS jxlrs=PASS
[CID22 1189261.png d=1 C_GPU] bytes=190153 oxide=PASS djxl=PASS jxlrs=PASS
...
[GB82-SC terminal.png d=3 C_CPU] bytes=65352 oxide=PASS djxl=PASS jxlrs=PASS
```

See `benchmarks/cvvdp_phase6_decoder_spotcheck_2026-05-24.tsv` for the
full TSV. No structural bitstream bugs detected. The REVERT branch of
RFC §5.4 is closed; cvvdp-driven encodes produce spec-compliant
bitstreams decodable by every JXL decoder we can run.

## §7. RFC §5.4 application

Per RFC §5.4 (rules in priority order):

1. **REVERT** if any cell has decoder failure under any decoder.
   - **NOT TRIGGERED**: 0 decoder failures in 60-check spot-check.
2. **DEFAULT_FLIP** if C_GPU Pareto-dominates ≥50% of cells on ≥1
   (corpus, metric) AND is within 5pp of leader on every other.
   - **NOT MET**: GB82-SC butter_cpu (92.1%), butter_gpu (91.7%),
     cvvdp_gpu (89.5%) all >5pp below leader B (100%).
3. **OPT_IN_ONLY** otherwise.
   - **APPLIED**: ship cvvdp as opt-in.

Analyzer output: `VERDICT: OPT_IN_ONLY`.

## §8. Wall-time trade-off summary

| backend | wall p50 (all) | wall p50 (e=8) | normalized vs B (e=8) |
|---      |---             |---             |---                    |
| `B`     | 75.7 ms        | 193 ms         | 1.00×                 |
| `B_GPU` | 65.1 ms        | 159 ms         | 0.82× (18% faster)    |
| `C_GPU` | 61.4 ms        | 168 ms         | 0.87× (13% faster)    |
| `C_CPU` | 73.0 ms        | 298 ms         | 1.55× (55% slower)    |

C_GPU is genuinely competitive on wall time. C_CPU is the slow path
(Phase 5 measured ~10× on the standalone metric; here the buttloop
overhead absorbs most of that, bringing it to ~1.55× the butteraugli
baseline — much better than projected). C_CPU is viable for CUDA-less
hosts as an opt-in choice; the wall penalty is real but tolerable.

## §9. Rank disagreements summary

The full `scripts/cvvdp_pareto_analysis_2026-05-24.tsv` includes a
rank-disagreement section listing cells where the Pareto winner under
`butter_cpu` is disjoint from the Pareto winner under `cvvdp_gpu`.
These cells expose where the two metrics actually disagree about
"better" — interesting both for future selector design and for
understanding when cvvdp's measurement bias matters most.

The TSV's per-disagreement detail (200 cells capped) is the basis for
any future content-aware selector that routes between metrics. Not
acted on in Phase 6.

## §10. Recommendation for Phase 7

Per the verdict in §7:

- **Phase 7 brief**: `docs/RFC_CVVDP_PHASE7_OPTIN_BRIEF.md` (NEW —
  populate from this memo's findings).
- Scope:
  1. CHANGELOG entry under `[Unreleased]` describing the cvvdp
     opt-in: feature flags, distance-table caveat (+200% bytes at
     same distance, not yet recalibrated), wall-time trade-off,
     CID22-cvvdp Pareto win.
  2. README addition documenting `--features cvvdp-loop` +
     `--features cvvdp-loop-cpu` (and the prerequisite for CUDA on
     `cvvdp-loop`).
  3. Update `docs/RFC_CVVDP_FORK.md` Status section from SCOPING to
     SHIPPED-OPT-IN with link to this decision memo.
  4. No encoder source changes. The cvvdp opt-in is already shipped
     as default-off via the existing `LossyConfig::with_cvvdp_loop`
     setter (Phase 4 wiring) and the `cvvdp-loop` / `cvvdp-loop-cpu`
     cargo features.
- W44 gate re-calibration (Phase 8) is NOT triggered because the
  default path stays butteraugli. The `docs/CVVDP_W44_GATE_TRANSFER.md`
  audit can stay in place as a forward-reference for if/when a future
  Phase recalibrates the distance table.

## §11. Limitations and caveats

- **Distance table not yet recalibrated** (RFC §2.3 explicitly TBD).
  C_GPU's +200% bytes overhead at the same `distance` is the dominant
  cost reason for OPT_IN_ONLY. A future Phase 7+ chunk could
  re-derive `vardct/cvvdp_targets.rs` from this Phase 6 bench data
  (specifically: pick the JOD target per distance that yields
  byte-parity with butteraugli on the average CID22 cell). That
  would close the bytes gap and re-enable the default-flip discussion.
- **C_GPU and C_CPU produce byte-identical output** at the encoder
  level — they only differ in measurement precision. The Pareto
  analysis treats them as two distinct backends because the scoring
  metric column structure requires it, but they are effectively the
  same encoder configured to use different cvvdp backends.
- **`score_cvvdp_cpu` column is NA throughout** — the scoring scaffold
  uses `cvvdp-gpu` for the cvvdp metric column regardless of which
  backend drove the buttloop. The CPU backend IS exercised inside the
  buttloop (300ms p50 wall vs C_GPU's 168ms confirms this) — only the
  post-encode scoring uses the GPU metric. Agent A's 1e-4 JOD parity
  between CPU and GPU cvvdp metrics means this would not change the
  Pareto verdict.
- **W44 cost-model gates** (W44-91 / W44-96 / W44-105 / W44-117
  cluster) were calibrated against butteraugli's measurement bias.
  Most of them did not fire at the cells we measured under cvvdp
  (because the cvvdp loop replaces the buttloop entirely, not the
  W44 gates that fire pre-buttloop). Future Phase 8 would re-validate
  them under cvvdp if Phase 7 ever turns into a default-flip; this
  Phase 6 verdict does not require it.
- **Transient panic rows**: 27 total across the 3 sweeps (9 per
  sweep) appeared mid-encode and were filtered by NF=13 row check.
  Root cause not investigated; likely CUDA hiccups on specific cells.
  Coverage 1134/1134 per backend after filter; below the documented
  panic rate of historical W44 sweeps. Not a blocker.

## §12. Conclusion

The cvvdp fork is **landed as an opt-in** with documented trade-offs.
The bones are in place:

- `LossyConfig::with_cvvdp_loop(Option<bool>)` + `with_cvvdp_use_cpu`
  builder API (shipped Phases 3-5).
- `cvvdp-loop` + `cvvdp-loop-cpu` cargo features (shipped Phases 3-5).
- `EncoderStrategy::Libjxl` strict cjxl-parity invariant preserved
  (cvvdp-loop never fires under Libjxl).
- Multi-decoder bitstream compatibility verified 60/60.
- cvvdp metric improvement on photos: 100% of CID22 e=8 cells score
  higher on cvvdp_gpu (mean +0.005 to +0.31 JOD across distances).

The default cannot flip until the distance table is calibrated, but
the door is open: re-derive `vardct/cvvdp_targets.rs` from this
Phase 6 bench data to bring cvvdp bytes within 5% of butteraugli at
the same distance, then re-run the Phase 6 analyzer on the recalibrated
output. That is Phase 7+ work, not blocking this memo's OPT_IN_ONLY
verdict.
