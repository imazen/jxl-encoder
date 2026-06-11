# Expert knobs & configs measured ALWAYS-WORSE (or danger-banded)

Verdicts from committed A/Bs. Each row links the measurement; the same
text lives as a **MEASURED** warning on the field/method rustdoc. Do not
re-spawn tuning chunks on these without new structure — the binding
constraints in CLAUDE.md govern.

| Knob / config | Surface | Verdict | Evidence |
|---|---|---|---|
| `gather_dedup` (Phase 2 inline dedup) | `EffortProfile` / `LosslessInternalParams` / `TreeLearningParams` | ALWAYS worse-or-tie vs packed-key sort at every size (1024² and 4–12 MP) | W44-137 triage; `benchmarks/perf_dedup_8mp_rebench_2026-06-10.meta` |
| `gather_dedup_phase3` (inline-fingerprint table) | same | P2↔P3 noise (±1.3 %); family loses end-to-end | same |
| `with_squeeze(true)` (lossless) | `LosslessConfig` | ALWAYS larger: +14.7 % photos, +62 % screenshots — progressive-decode feature, not compression | CLAUDE.md "Lossless Compression Status" |
| `k_ac_quant = 0.65` global | `EffortProfile` / `LossyInternalParams` | RULED OUT: 29/36 cells fail ±0.30 SSIM2; smooth-photo proxy gate also ruled out | issue #25; CODE-HISTORY 2026-06-10 |
| `Tier2Knobs` with `k1 < 0.5` or `k2 < 1.0` on screen/{very_high,high} | `with_knobs` (`tuning-override`) | SHIP-cell catastrophe −4.9…−5.1 SSIM2 | W44-105 / W44-228c1 |
| Owned-clone tree-learn fallback | issue #42 dispatch infra | Regresses at every measured size; default stays borrowed-view | CLAUDE.md binding constraints (issue #42) |
| CfL screenshot x=0-start, ungated/global | (shipped only as per-image-proxied dispatch) | Strictly worse under the Zenjxl cost model when unproxied | W44-AUDIT-5 P3 |
| `JPEG_GRADIENT_DC` | env (A/B-only) | Never production: kWPFixedDC is the shipped DC predictor for JPEG | EX-J31 |
| `strip-tile-butteraugli` | butteraugli crate feature | Slower at every size; framework-only for a future true-tile refactor | W44-PHASE3-B7d |
| 1-worker rayon pool at `--threads 1` | (not a knob — rejected design) | +70…+195 % wall; '1' has never meant sequential internals | `benchmarks/perf_pool1t_2026-06-10.meta` |

Related but NOT in this table: `--lf-frame` (+1.2–3.8 % by design, buys
progressive DC preview), `--error-diffusion` (libjxl-divergent, default
OFF), `nb_rcts_to_try = 0` (legitimate fast path; GBR_SUBGR fallback is
calibrated — just a weak A/B signal, see jxl-encoder#67).
