# Immutable ANS table reuse — September 24, 2026

The shared tables remove repeated construction without changing normalization,
uint-config candidates, predictor selection, or output bytes. This is an
allocation cleanup, not a demonstrated broad speedup. Peak memory was not
measured. A fixed table set now stays alive for the process lifetime; racing
initializers may allocate temporary duplicates before `OnceBox` selects one.

The source change replaces seven production table constructors with a private
shared accessor: the public histogram wrapper, five entropy-builder/config
sites, and the strict cluster-cost site. The public `AllowedCountsCache::new` remains unchanged.
Both exact and normal normalization tables remain present.

The before binary is source `e4fa0ba0`, SHA256
`483ff1da9a154915281d02f994d81426c6b4cfd182c4a7b1d05d0e107b34c475`.
The after binary was built from the three changed entropy source files at
`95750709`, against the same pinned dependency closure. Both are release builds
with `std,parallel,profile-phases`, default features, and no target-cpu=native.
The host is macOS ARM64. Complete commands, binary/source hashes and artifact
locations are in the companion metadata.

The input is the existing real-photo sectioned bar: 3840×2160,
SHA256 `0e72993af2f43ba978b5f2cbb3e630dc4c537ea615de4fd9c3aa5734c67821c9`.
`SectionedTrees::On` is explicit. This is one performance regression sentinel,
not a content/size calibration grid, and no defaults were fitted from it.
Each process warms its configuration once. Rust timings cover the second
encode only; artifact hashing and persistence occur afterward. Reference
v0.12 timings cover the whole cjxl process. The reference ratio retains that
historical mixed-scope bar; it must not be read as equal-scope encoder latency.

Three repeats alternate binary order and retain every result. Values below are
median (range), milliseconds. Paired ratios divide after by before within each
repeat. The last column is the historical minimum-wall ratio to cjxl.

| Effort | Threads | Before ms | Shared ms | Paired median | Shared min / cjxl min |
|---|---|---|---|---|---|
| 7 | 1 | 10108.5 (10108.4–10173.4) | 10050.4 (10008.9–10050.8) | 0.9901 | 1.438 |
| 7 | 8 | 1992.3 (1962.0–1994.5) | 1985.6 (1964.9–1991.2) | 0.9983 | 1.724 |
| 9 | 1 | 33786.5 (33724.6–33943.6) | 33631.0 (33595.3–33681.5) | 0.9969 | 0.847 |
| 9 | 8 | 6116.8 (5908.2–6138.9) | 6283.6 (6106.4–6306.0) | 1.0273 | 1.030 |

The e9/t8 +2.7% first-pass result triggered seven additional alternating
repetitions. Their paired median is **1.0024**, range **0.9455–1.0265**.
Before median/range: **6152.3 (5863.6–6428.0) ms**; shared:
**6164.7 (5874.9–6344.1) ms**. Neither a stable wall regression nor a speedup
is established by these variable runs. Keep both measurements; do not quote
only the favorable minimum.

The unchanged sectioned e7 target remains unmet: 1.438× at t1 and 1.724× at
t8 against the 1.3× bar in the first comparison. The shared-table change does
not remove the split-search or tree-learning cost. The separate refreshed
k8/default baseline also remains committed, including all repetitions.

Validation: 19/19 measured before/after pairs have identical SHA256s; all 57
TSV rows resolve to artifacts whose hashes and lengths match. The first run's
24 warmup hashes match its timed hashes. All 75 lock/drift tests, 1,618 library
tests (29 existing ignores), both real-image RD regression tests, default
workspace all-target Clippy, and no-default-features checking pass. The latter
emits 30 warnings. The sectioned harness's real 64×64 and 259×133 regression
reconstructs exact source pixels through zenjxl-decoder and djxl. Six driver
tests cover failure propagation, alternating order and selected grid cells.

Source-history boundary: `0784632b` changed normal kBest uint-config scoring
from a Shannon/header estimate to normalized ANS cost. The part-18 legacy
branch preserves the pre-part-18 normalizer, not that older estimator. This
cleanup does not revert it or attribute the historical wall/byte change to it.

Data: [first comparison](sectioned_cache_ab_2026-09-24.tsv),
[first metadata](sectioned_cache_ab_2026-09-24.meta),
[e9/t8 follow-up](sectioned_cache_e9t8_repeat_2026-09-24.tsv),
[follow-up metadata](sectioned_cache_e9t8_repeat_2026-09-24.meta),
[refreshed k8/default baseline](sectioned_bar_refresh_2026-09-24.tsv),
[baseline metadata](sectioned_bar_refresh_2026-09-24.meta).
