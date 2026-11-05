# Four retained PQ decoder-agreement findings, November 5

The fresh seed-2966 sweep records four additional FAIL cells. Both reproduce
with unchanged inputs, decoder pins and the 0.0200 clamped RGB agreement
bound. Attribution remains open; this is not a clean-sweep verdict or a
confirmed encoder defect. The newer native precision repair does not close
these comparisons.

| Index | Input and settings | Maximum pinned Rust/static difference | Samples above bound | Ours source MAE | cjxl source MAE |
|---|---|---:|---:|---:|---:|
| 149042 | windows, 300×260 RGBA PQ f32, d10/e9, progressive DC/LF frame | 0.025484640151262283 | 1 | 0.011809754146257161 | 0.01240237939219034 |
| 179212 | CID22 2190188, 1100×900 RGB PQ f32, d1.5/e3, streaming rows 261 | 0.020733240991830826 | 1 | 0.018990510774835222 | 0.02081343074479673 |
| 194142 | windows95, 900×1100 RGB PQ f32, d4.5/e3, EPF off, faster decoding 2 | 0.024154406040906906 | 1 | 0.008867056717274911 | 0.009076828083822647 |
| 210357 | CID22 225228, 257×255 RGBA PQ f32, d10/e2, progressive DC, resampling 4, EXIF/XMP | 0.02735306601971388 | 1 | 0.029718000192512035 | 0.0305421486190807 |

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

At zero-based red (683,171) in 194142, both libjxl builds return
0.022196322679519653; both pinned Rust decoders return
0.04635072872042656, and newer native returns 0.046546801924705505.
The precision flag does not change this value. The two libjxl builds differ
by at most 0.0002446472644805908 in this stream. Its paired cjxl stream's
static/new-native maximum is 0.002582557499408722. The third replay retains
its expected failing verdict and matches the frozen input and stream hashes.
Its controller passes in five seconds, peak RSS 0.31 GiB, minimum available
47562 MiB, peak load 0.53.

At zero-based blue (59,204) in 210357, static djxl returns
0.007006318308413029, historical djxl 0.011277236044406891,
upstream Rust 0.03435938432812691, pinned native 0.033563245087862015,
and newer native 0.034360725432634354. Its paired cjxl stream's
static/new-native maximum is 0.0005341321229934692. Both pinned Rust
comparisons exceed the bound at one sample; the table reports upstream's
larger maximum. The replay exactly matches the frozen stream and input.

Thus these are additional near-black numerical disagreements for
[decoder issue 73](https://github.com/imazen/zenjxl-decoder/issues/73), with a
different reference-build pattern from the older 2965/28968 stream. The
pre-transfer reconstruction cause and applicable precision contract still
require attribution; the controls below isolate the phase. Neither a majority vote nor passing source MAE settles that.
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
| 194142 | `ce9c26eeb0cab28c6452d24897697744c5aae66c54e1e08aec1e1925abc4d54d` | `d3ec0aa626d6b6e2ae9d14fe61749316269a23cbdc373a220bf8b2617adcdb15` |
| 210357 | `0ee9d3a059c589d0e80aa1ae9e200b1e0ad1ea76d34a8ddab70567e88e327c8e` | `5ac33df32fc36e629aeacb5b508c303ec53f2904329091643aa9b7e204276b07` |

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
`/mnt/v/output/jxl-encoder/nightly-timeout-20261105/pq-deviation-replay/`
and its adjacent `pq-deviation-third/` and `pq-deviation-fourth/` directories.
Use `just conformance-replay` with a TSV selecting these two seed/index pairs
and `--retain`; use `just compare-float-decodes` to compare exact f32 samples.
PFM byte order and bottom-up rows are honored. The diagnostic retention path
changes no verdict or encoder bytes. Tool unit tests and strict tool Clippy
pass; replay preserves both failures. Wrapper rc0, 44 seconds, peak RSS
1.02 GiB, minimum available 46773 MiB, peak load 1.34. Its rc0 means the
controller observed the two expected failing replays, not that they passed.

## The discrepancy precedes PQ transfer

`just pq-linear-control` makes a separately named diagnostic copy and flips
one bit inside the existing six-bit transfer enum, PQ to Linear. Every other
bit, including container EXIF/XMP and frame payloads, remains identical;
independent parsing requires that only the transfer field changes. The
explicit 10000-nit intensity stays unchanged. Both decoders fully render
each copy, exposing linear RGB in units where 1.0 is the intensity target.
The original failed streams are preserved and retain their FAIL verdicts.

| Index | Static linear at failed coordinate | New native linear | Absolute linear difference |
|---|---:|---:|---:|
| 149042 | -2.1466540545134194e-7 | 1.4095021469984204e-6 | 1.6241675524497623e-6 |
| 179212 | 9.313747568739927e-7 | -9.535416722883383e-8 | 1.0267289241028266e-6 |
| 194142 | 1.0672372354747495e-6 | 5.1062779675703496e-6 | 4.0390407320956e-6 |
| 210357 | 1.2087791390058555e-7 | 2.6317015908716712e-6 | 2.5108236769710857e-6 |

Applying one signed double-precision inverse PQ EOTF to each decoder's linear
output reproduces its original clamped PQ output with maximum difference
0.00000373004877474159 over all four streams. This localizes the disagreement
to the reconstruction before transfer; evaluating the final PQ formula more
precisely does not reconcile those linear samples. This control does not
separate IDCT, filtering, upsampling and inverse-color arithmetic, or settle
which implementation meets the required precision contract.

The formula and sign extension are checked against the immutable
[native transfer implementation](https://github.com/imazen/zenjxl-decoder/blob/c1d457d9/zenjxl-decoder/src/color/tf.rs).
The reference's inverse-color accumulation is in
[libjxl v0.12 XybToRgb](https://github.com/libjxl/libjxl/blob/v0.12.0/lib/jxl/dec_xyb-inl.h).
Those source links describe the arithmetic; the table is measured from the
retained CLI outputs, not calculated from source alone.

All four final controls pass, including the metadata container. Complete
original/diagnostic hashes, raw linear outputs and comparison reports are
under each replay's `linear-control-final/`. Final wrapper:
`~/tmp/jxl-pq-linear-final-20261105.log` on r5900xt, rc0, 2 seconds,
peak RSS 0.21 GiB, minimum available 47883 MiB, peak load 0.44.
The earlier combined replay also passes all 69 hash checks (one existing
ignore), preserving the approved single PQ fingerprint and every other
expectation. Its wrapper is rc0, 25 seconds, peak RSS 0.21 GiB, minimum
available 47762 MiB, peak load 2.08. An initial fourth-control attempt
refused the container before writing output; the repaired diagnostic preserves
container bytes and its failing setup log remains retained.
