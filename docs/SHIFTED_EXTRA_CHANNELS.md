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

## Evidence under validation

The source-layout matrix covers 18 of the 20 supported lossless layouts;
CMYK8/16 with a CMYK ICC profile is being checked separately. This is an
explicit gap until that command succeeds. The six nonlinear float layouts are
refused by the lossless API. No full option Cartesian product is claimed.


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
minimum available 20,529 MiB). Fresh broad/minimal/lint checks and landing
remain pending. The production repair described here is local work until its
implementation commits land.

## Coverage and evidence retention

`extra-edge-accepted-selected.json` is exported from only the five retained
current binaries, with input hashes in its `.inputs.json` sidecar; the frozen
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
| `RequestStop` methods | 23/23 | none |
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
Retained streams and profiles stay outside git. A report file without a
successful command and matching provenance is not an accepted verdict.
