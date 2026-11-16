# Shifted modular channel grids (issue 144)

Tracking: [issue 144](https://github.com/imazen/jxl-encoder/issues/144).

A lossless request's shifted extra buffer contains `ceil(width / 2^shift)` by
`ceil(height / 2^shift)` native samples. For each group, the encoder must shift
the full group rectangle before clipping it against that native grid. Clipping
in color coordinates first drops a final partial sample. Channels whose minimum
horizontal/vertical shift is at least three use LF groups, whose color-space
extent is eight times the selected AC group dimension.

## Producers and controls

| Route | Evidence / applicability |
|---|---|
| Normal unsqueezed multi-group modular | Shared rectangle extraction; separate LF routing |
| Full palette and ChannelCompact | Global metadata retained; transformed spatial channels use the same routing |
| Explicit lossy palette | Separate producer, same rectangle/LF helpers; palette stays global |
| Squeeze with fixed or learned predictors | Existing separate partitioner; selected interaction controls retained |
| Lossless request packed/padded input | Native depth samples and exact packed/strided byte equality |
| CLI `--ec_resampling` | Alpha split with factors 2/4/8, RGB/Gray source; bulk and streaming flag combinations |
| Lossless streaming API | No extra-channel attachment surface |
| Lossy still/precomputed and optional subsampling | Shifted extras unsupported; no affected modular groups |
| Animation request | Additional request extras unsupported |
| JPEG transcode | No caller shifted-pixel extra buffer |

The offered-policy inventory comes from `tests/support/encoder_policies.rs`:
named strategies, Reference/Experimental and the applicable Custom compact-header
and smoothing combinations. This selected matrix exercises all four group sizes,
8/16-bit sources, both image orientations and palette/tree/Huffman/squeeze
choices. It is not the full Cartesian product of encoder options.

The unmodified libjxl v0.12 `Rect` component oracle supplies the committed
channel-rectangle vectors. The native unit checks dimensions, retained shifts
and every copied sample. Production controls compare fully rendered odd images
with crops of aligned canvases containing the same native grid, independently
within pinned jxl-rs 0.8 and djxl v0.12. The aligned marker must remain visible;
main black/white samples are exact. These full-resolution extra planes include
interpolation, so they are not mislabeled as raw native samples. The earlier
jxl-rs 0.4.3 rejects some nonzero aligned controls accepted by the current
primary and reference; retained diagnostics are a decoder compatibility gap.

## Verified local evidence

The main source-layout matrix covers 18 of the 20 supported lossless layouts.
The caller-selected CMYK control separately passes both CMYK8 and CMYK16 with
a real CMYK/XYZ profile, all offered policies, shifts 1/3, global/local trees
and both small/multi-group dimensions. Packed/strided bytes and every native
8/16-bit depth code agree; both decoders fully render. This checks the extra
channel, not physical CMYK color accuracy (tracked separately in issue 144).
The six nonlinear float layouts are refused by the lossless API. No full
option Cartesian product is claimed.

Run `just extra-edge-cmyk-check <pinned-manifest> <CMYK-profile> <new-output>`.
The successful run uses `ps_cmyk.icc` (5,340 bytes, SHA256
`020236a6458406743848a12c6be280e54bcd0fd8e2d4bea9c4899410cef98cb2`).
Its `completed.tsv` records every successful cell and JXL hash; profile/binary
hashes accompany it. `extra-edge-cmyk-small-profile.log` has rc0, 40 seconds,
peak RSS 0.22 GiB and minimum available 22,393 MiB. The earlier large-profile
run was interrupted and remains partial evidence.


Before logs: `extra-edge-before-current.log` and
`extra-edge-before-shared-context.log`; streams and native buffers are retained
outside git under the test output root. `extra-edge-lf-boundaries3.log` passes
all selected runtime checks including multiple LF groups. It also retains the
new local-Huffman correction's preceding failed runs; those failures were in
session-added code, not relaxed existing assertions.

Reproduce with `just extra-rect-oracle <v012-source> <v012-build> <binary>`,
`just extra-edge-unit <pinned-manifest>`, `just extra-edge-check <pinned-manifest>`
and `just extra-edge-cli-check <pinned-manifest>`, under the resource wrapper
with explicit verified `CJXL_PATH`/`DJXL_PATH`. The final local pixel/resource/CLI command passes in
`extra-edge-accepted-coverage.log` (rc0, 286 seconds, peak RSS 1.85 GiB,
minimum available 20,529 MiB). The broad library/unchanged-lock and minimal-build matrix also passes in
`extra-edge-final-validation.log` (rc0, 621 seconds, peak RSS 1.74 GiB,
minimum available 20,803 MiB). The threaded control asserts that requested
one/two-worker settings actually reach the encoder. A later cancellation audit
is documented separately in [the cancellation contract](CANCELLATION.md).
The current-source repeat also passes in
`extra-edge-current-final-validation.log` (rc0, 599 seconds, peak RSS 1.71 GiB,
minimum available 20,807 MiB), including the unchanged byte locks and minimal
feature matrix. The current commit's CI verdict remains separate.

## Coverage and evidence retention

`extra-edge-accepted-frozen.json` re-exports the accepted geometry snapshot
from only its five retained binaries and a retained frozen profile. Its
`.inputs.json` sidecar points to those surviving inputs, rather than the
removed Cargo cache. The frozen
profile hash is `c4e962d5f3e4e141040c660948166bd49910367b1d568d542ade03211fdaddf9`.
Its JSON hash is `87a1d118a7f8b7a4de3bf2e2dc10fe53e8259da5a80e4e677659221a757cf782`.
Automatic exports can include obsolete code mappings from earlier builds in
the same target. The explicit-object export avoids using those as evidence.
Its three hash-mismatch diagnostics name `LosslessConfig::with_buffering`,
`LosslessConfig::with_container_mode` and `Buffering::from_i8`; none is one of
the affected functions listed below.

| Selected function | Recorded code regions executed | Remaining regions |
|---|---:|---|
| `extract_group_region` | 15/15 | none |
| `split_lf_groups` | 52/53 | allocation/error propagation from extraction |
| `encode_shifted_lf_groups` | 171/179 | eight defensive serializer error returns |
| ANS chunk accumulation | 14/14 | none |
| ANS sequential/parallel accumulation | 58/58 | none |

The serializer gaps cover writes of validated headers, trees and hybrid-coded
values into a growable writer. Invalid internal code generation and allocator
failure were not injected; those defensive error returns remain unexecuted.
Budget refusal and cancellation are exercised separately. ANS debug validation
also retains four unexecuted panic regions for a generated token outside its
alphabet or at zero frequency. These fractions locate remaining code branches;
they do not establish semantic completeness or total encoder coverage.

`just coverage-selected-report <llvm-cov> <profile> <new-json> <objects>...`
records explicit binary/profile hashes. `just archive-test-evidence <source>
<destination> <new-journal>` relocates completed evidence only: each destination
is checksum-verified and its journal record is flushed before source removal.
Retained streams and profiles stay outside git. The stop adapter has since
changed; its independent current-binary evidence lives in CANCELLATION.md. A report file without a
successful command and matching provenance is not an accepted verdict.

## CLI and CMYK entry controls

The CLI now forwards both existing GrayAlpha PNG precisions to their matching
pixel layouts. The ordinary CLI feature build passes the odd-edge resampling
and GrayAlpha sample checks across all offered strategy spellings, Reference/
Experimental and bulk/streaming flags. Packed/strided API byte controls and
full Rust/djxl rendering retain the expected alpha codes.
`extra-edge-cli-current.log` passes with strict CLI lint (rc0, 49 seconds;
peak RSS 0.97 GiB, minimum available 20,472 MiB). Run
`just extra-edge-cli-selected-check <pinned-manifest> <coverage-env>` to repeat
without combining obsolete profile mappings into a report. The CMYK caller
probe and mutation commands are retained alongside these controls.
