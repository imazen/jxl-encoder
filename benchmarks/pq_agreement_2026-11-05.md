# Two retained PQ decoder-agreement findings, November 5

The fresh seed-2966 sweep records two additional FAIL cells. Both reproduce
with unchanged inputs, decoder pins and the 0.0200 clamped RGB agreement
bound. Attribution remains open; this is not a clean-sweep verdict or a
confirmed encoder defect. The newer native precision repair does not close
these two comparisons.

| Index | Input and settings | Maximum pinned Rust/static difference | Samples above bound | Ours source MAE | cjxl source MAE |
|---|---|---:|---:|---:|---:|
| 149042 | windows, 300×260 RGBA PQ f32, d10/e9, progressive DC/LF frame | 0.025484640151262283 | 1 | 0.011809754146257161 | 0.01240237939219034 |
| 179212 | CID22 2190188, 1100×900 RGB PQ f32, d1.5/e3, streaming rows 261 | 0.020733240991830826 | 1 | 0.018990510774835222 | 0.02081343074479673 |

Signal MAE uses clamped decoder RGB and excludes source alpha zero. It is a
separate diagnostic from the maximum agreement gate. Both reference encodes
use the same RGB samples, matching color space and supported options; source
alpha is excluded from the reference coding. These are matched-distance
comparisons, not matched-quality RD claims.

At zero-based red (272,235) in 149042, pinned static djxl returns
-0.009609627537429333; both pinned Rust decoders return
0.025484640151262283. Historical dynamic djxl returns the same negative value.
The newer native decoder returns 0.02546786516904831, with its precision flag
both unset and set. For this stream the two libjxl builds' maximum clamped
RGB difference is 0.00011154636740684509. The paired cjxl stream's
static/new-native maximum is 0.0015381332486867905.

At zero-based blue (402,466) in 179212, static and historical djxl return
0.020733240991830826; upstream Rust returns zero, pinned native returns
-0.0099260238930583, and newer native returns -0.006136397831141949.
The precision flag does not change the newer native result. The two libjxl
builds differ by at most 0.006637332495301962 elsewhere in this stream.
The paired cjxl stream's static/new-native maximum is 0.014362006913870573.

Thus these are additional near-black numerical disagreements for
[decoder issue 73](https://github.com/imazen/zenjxl-decoder/issues/73), with a
different reference-build pattern from the older 2965/28968 stream. The
linear-light cause and applicable precision contract still require
attribution. Neither a majority vote nor passing source MAE settles that.
Do not change encoder quantization or relax the agreement bound to hide them.

## Identities and reproduction

Frozen sweep product base `7b354ee6`, harness `c10e591e`, the existing root
lock and standalone sweep lock, upstream jxl 0.8.0 and native pin `515f2c8a`
remain as recorded in [the full run](conformance_2026-11-05.md).
The replayed input and stream hashes exactly match the live sweep artifacts:

| Index | Encoded SHA256 | Raw input SHA256 |
|---|---|---|
| 149042 | `392c38ef443cc97e6bf4281db2594a95d5dabd30e4c6777a5399d4e887ad2644` | `7fa3cdcc19627ff9fffeb7d9f397ddebe2f2192a602e830e433fccc6d4e4e223` |
| 179212 | `211d1d60bc432ad0edab4f78beb276f53059dafdc81e92a23e0057a03f3f48c4` | `2a58028421693274290010d698fd658b06d9c37c91c32e9532b655d55d658851` |

Static djxl SHA256 is
`67bdc98fdca7f93d7b77f3473e818b251ba107109ee83b3b37ccbd90066f2454`;
historical dynamic djxl is
`ef38c21001892cfd449753a19e257746de7d6d2ae3030616f536547d20c75991`.
The historical binary is diagnostic only; its full build provenance is
unresolved. New native CLI SHA256 is
`ddb7173e57a49deb89cb13304031f48cab4cbec62ea9fb574596b50e9544d56e`,
from the independently retained `310207f4`/`c1d457d9` precision repair.
It is not substituted for the sweep's pinned decoder.

Raw decoder samples, JSON dimensions, comparison reports, all encoded streams,
source inputs and complete logs are retained on r5900xt under
`/mnt/v/output/jxl-encoder/nightly-timeout-20261105/pq-deviation-replay/`.
Use `just conformance-replay` with a TSV selecting these two seed/index pairs
and `--retain`; use `just compare-float-decodes` to compare exact f32 samples.
PFM byte order and bottom-up rows are honored. The diagnostic retention path
changes no verdict or encoder bytes. Tool unit tests and strict tool Clippy
pass; replay preserves both failures. Wrapper rc0, 44 seconds, peak RSS
1.02 GiB, minimum available 46773 MiB, peak load 1.34. Its rc0 means the
controller observed the two expected failing replays, not that they passed.
