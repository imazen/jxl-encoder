# `with_resampling(N)` downsample domain: linear RGB → opsin (#45 T1)

Measured 2026-08-30, macOS aarch64, release, no `target-cpu=native`.
Full grid + provenance: `jxl_resampling_domain_2026-08-30.{tsv,meta}`.
Harness: `jxl-encoder/examples/resampling_domain_ab.rs`.

Metric: in-process Rust butteraugli over a jxl-oxide `srgb_linear` decode
(lower is better). Every stream decoded in-process; 0 undecodable.

## CID22-512 corpus, mean over 4 images × 5 distances (d1/2/4/7/10)

| effort | resampling | bfly BEFORE | bfly AFTER | Δ bfly | bytes BEFORE | bytes AFTER | Δ bytes | cjxl bfly | cjxl bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e1 | 2× | 21.338 | **12.550** | -8.788 (-41.2 %) | 17529 | 16482 | -6.0 % | 12.832 | 13939 |
| e1 | 4× | 25.667 | **23.084** | -2.583 (-10.1 %) | 5069 | 5006 | -1.3 % | 23.065 | 3805 |
| e1 | 8× | 45.103 | **42.225** | -2.878 (-6.4 %) | 2244 | 2217 | -1.2 % | 42.246 | 1239 |
| e3 | 2× | 21.338 | **12.550** | -8.788 (-41.2 %) | 14269 | 13248 | -7.2 % | 12.832 | 13067 |
| e3 | 4× | 25.667 | **23.084** | -2.583 (-10.1 %) | 3790 | 3726 | -1.7 % | 23.065 | 3650 |
| e3 | 8× | 45.103 | **42.225** | -2.878 (-6.4 %) | 1269 | 1234 | -2.7 % | 42.246 | 1213 |
| e5 | 2× | 22.640 | **13.682** | -8.959 (-39.6 %) | 13571 | 12532 | -7.7 % | 13.822 | 12468 |
| e5 | 4× | 28.695 | **26.889** | -1.805 (-6.3 %) | 3511 | 3451 | -1.7 % | 26.453 | 3497 |
| e5 | 8× | 46.784 | **44.852** | -1.932 (-4.1 %) | 1121 | 1096 | -2.2 % | 44.670 | 1079 |
| e7 | 2× | 22.775 | **13.945** | -8.829 (-38.8 %) | 14670 | 12901 | -12.1 % | 14.460 | 12554 |
| e7 | 4× | 28.229 | **26.162** | -2.067 (-7.3 %) | 3526 | 3435 | -2.6 % | 26.233 | 3539 |
| e7 | 8× | 47.206 | **44.928** | -2.278 (-4.8 %) | 1131 | 1101 | -2.7 % | 44.968 | 1099 |
| e9 | 2× | 22.607 | **15.014** | -7.593 (-33.6 %) | 14883 | 11870 | -20.2 % | 13.386 | 12702 |
| e9 | 4× | 28.453 | **27.486** | -0.967 (-3.4 %) | 3175 | 3104 | -2.2 % | 26.233 | 3519 |
| e9 | 8× | 48.853 | **45.242** | -3.611 (-7.4 %) | 1038 | 1014 | -2.3 % | 44.968 | 1103 |
| e10 | 2× | 12.519 | **12.519** | +0.000 (+0.0 %) | 12130 | 12130 | +0.0 % | 12.421 | 12720 |
| e10 | 4× | 28.453 | **27.372** | -1.081 (-3.8 %) | 3175 | 3099 | -2.4 % | 26.022 | 3515 |
| e10 | 8× | 49.012 | **45.256** | -3.756 (-7.7 %) | 1038 | 1010 | -2.7 % | 44.968 | 1104 |

## Hash-lock cells (synthetic `noise_rgb_512x512`, d1.0)

These are the exact streams `tests/hash_lock_expected.txt` pins.

| lock cell | bytes BEFORE | bytes AFTER | Δ bytes | bfly BEFORE | bfly AFTER | Δ bfly |
|---|---:|---:|---:|---:|---:|---:|
| `lossy_mg_rgb_512x512_noise_r2_e3 (NEW)` | 92400 | 75388 | -18.4 % | 55.046 | 10.233 | -44.813 |
| `lossy_mg_rgb_512x512_noise_r2_e7 (NEW)` | 96984 | 83743 | -13.7 % | 55.687 | 10.398 | -45.288 |
| `lossy_mg_rgb_512x512_noise_r2_e9 (NEW)` | 149909 | 79665 | -46.9 % | 55.108 | 10.398 | -44.710 |
| `lossy_mg_rgb_512x512_noise_r4_e7 (NEW)` | 6847 | 7507 | +9.6 % | 11.460 | 12.517 | +1.056 |
| `lossy_mg_rgb_512x512_noise_r8_e7 (NEW)` | 1144 | 1048 | -8.4 % | 13.451 | 14.499 | +1.048 |
| `lossy_mg_rgb_512x512_noise_r2_e10 (existing — UNCHANGED)` | 80215 | 80215 | +0.0 % | 9.830 | 9.830 | +0.000 |

## Same-cell check against the two cells issue #45 originally cited

`effort_ladder_tiers::iterative_downsample_cjxl_differential` (`--ignored`,
needs `cjxl` on PATH), d1.0 r2, CID22-512 validation:

| cell | e9 sharper BEFORE | e9 sharper AFTER | e10 iterative | cjxl e10 |
|---|---:|---:|---:|---:|
| 1025469 | 19.68 | **11.52** (15516 B) | 7.196 (15718 B) | 7.189 (15974 B) |
| 1044329 | 30.20 | **14.51** (40784 B) | 10.660 (42306 B) | 9.899 (39596 B) |

e9 sharper stays worse than e10 iterative, **by design** — the adjoint
refinement is a real kernel improvement, which is why libjxl gates it at
`kGlacier`. This change fixed the *domain*; the remaining gap is the effort
ladder working as intended.

## Note on the r4/r8 noise-fixture cells

The r4/r8 noise cells move the "wrong" way on butteraugli (+1.06 each).
That is a **degenerate fixture**, not a regression: per-pixel uniform random
noise has no structure for either domain to preserve, an 8× box of it is
essentially a constant, and butteraugli on that is not a meaningful signal.
The corpus table above — real photographs, every distance, every effort — is
the evidence, and it is uniformly favourable on both axes. The noise cells are
retained as *hash* locks (their job is byte pinning, not quality).

