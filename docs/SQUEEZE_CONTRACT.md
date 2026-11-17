# Squeeze descriptor and group contracts

[Issue 144](https://github.com/imazen/jxl-encoder/issues/144) separates explicit
field admission (E2), later-band geometry (E3) and numerical admission (E5).
E2/E3 share repair `78cb89eb`; `07250ad2` keeps its wire-vector test
lint-clean on Rust 1.100. E5 landed in `6579e397`, with boundary controls
in `e7127fae`; see [its separate evidence](SQUEEZE_RCT_ADMISSION.md).

## Explicit fields and derived schedules

The original strict 16-square RGB request with seventeen full-size depth extras
panics at an explicit count of twenty. Part 1 H.6.2.1 represents counts 1–19
explicitly. An implicit descriptor derives its count from the channel set;
it has a different contract. Unmodified libjxl v0.12.0 field/default writers
supply the small frozen vectors in `src/modular/squeeze_*_v012.tsv`.
The native writer accepts 296 operations and begin channel 9287, refusing
297 and 9288. The repair validates every explicit field before writing bits.

The repair uses the reference default schedule above nineteen channels,
then writes an implicit descriptor. Existing explicitly representable schedules
retain their generation and bytes. Tiny images keep their prior no-squeeze
choice. A derived schedule that would create a zero-sized residual uses the
ordinary unsqueezed producer: the Rust channel representation does not support
that band. This matters for one-pixel axes and shifted extras. A shifted first
extra in grayscale can instead skip the chroma operations and retain a valid
full-prefix schedule. Both outcomes need direct controls.

## Remaining group tiles

The original 8x65665 RGB8 stream with 256-pixel groups and its transpose fail
both decoders before repair (`squeeze-four-before.log`). Their retained SHA256s
are `782ee629c4fbeea9cf1163324fbd53130eab01b43295dd6e69a6326f7c1200e5`
and `cd86b00b77157fcc9f6397bb54910c38bd202fa055c877caccc35ecffe72b74c`.
Source buffers/hashes are retained too. The E2 candidate additionally exposed
an invalid implicit 20-channel 8x16449 stream at 128-pixel groups, SHA256
`decdd425061f6e8a94310f2a183f73e2346195f7b04b4e70fb121d0e41942090`.
E2 cannot land independently while accepting that E3 failure.

The global prefix ends at the first oversized channel. Later small bands stay
outside that prefix. Their tile extent must remain positive after their shifts,
using LF group dimensions only when both shifts are at least three (G.4.2).
The repair simulates ordered band metadata before transforming pixels.
Valid full schedules stay unchanged. For an invalid explicit schedule it retains
a valid prefix within v0.12's conservative shift-count cap. An invalid larger
implicit schedule uses the ordinary producer: truncating its derived script
would disagree with v0.12. No strategy gate or public API is added.

The metadata controls and original tall/transposed/implicit production
regressions pass locally (`squeeze-group-unit-controls.log`,
`squeeze-group-regressions.log`). Larger-group controls use 256/512/1024-pixel
groups and both axes through 1,049,089 pixels. Source samples and extra-channel
codes recover exactly through the primary Rust decoder and pinned djxl.

## LF producers

LF frames append zero extra-channel placeholders. Their lossy default schedules
keep explicit descriptors and the established degenerate-axis quantizer shifts.
Above nineteen channels the repair's full-prefix steps address only the
three XYB channels. Native `FwdSqueeze` controls confirm every ordered nonempty
color band's dimensions, shifts, component and coefficient matches the full
schedule; every extra placeholder remains zero. These controls span 19/20/21
channels, 1-square, both 1x17/1x257 axes, 16-square, 17x9 and 257-square.
This establishes transform separability on those inputs, not full byte parity.

The public LF boundary checks require exact per-decoder color identity across
19/20/21 channels, exact U16 extras and packed/strided equality. Additional
controls include integer alpha/grayscale, an active e8 loop and a 2049-square
LF grid. The selected LF checks pass, including the minimal build that has no
optional loop setter. Earlier harness compilation/lint failures remain in the
logs; no source samples or pixel assertions were weakened to repair them.

## Caller inventory and limits

| Producer | Applicable route |
|---|---|
| Simple single-group lossless | Huffman and ANS |
| Learned single-group lossless | ANS plus tree learning |
| Simple multi-group lossless | Huffman and ANS |
| Learned multi-group lossless | Active squeeze for global/local/Hybrid requests |
| Single-group LF lossy modular | LF DC emission and its reconstruction |
| Multi-group LF lossy modular | LF group emission and its reconstruction |

Global/local/Hybrid requests share the learned squeeze writer while squeeze is
active; these are compatibility controls, not three distinct squeeze writers.
The ordinary sectioned/Hybrid producers participate when squeeze falls back.
Multi-group lossy palette has a preceding dispatch branch and bypasses squeeze.
Patch-reference streams disable squeeze. Single-group patched display hosts
share the affected writers; multi-group patched hosts bypass squeeze.
Individual VarDCT extra squeeze starts
from one channel and retains the explicit descriptor.

Request-added channels supply the larger sets. Legacy/streaming/planar APIs
cannot attach those sets, and animation rejects request-added extras. Those
APIs can still reach E3 through their ordinary color/alpha planes and need
separate geometry controls. E3's admission uses dimensions/shifts/order, not
sample values or precision; integer, binary16 and binary32 layouts have direct
route checks. The mode inventory includes both offered modes, all named
strategies and four selected Custom combinations of compact headers/DC
smoothing. It does not enumerate unrelated Custom gate combinations.

The caller-selected CMYK control passes with 21 coded channels, both integer
precisions, selected odd/multi-group shapes, shifts 0/1/3 and all offered
policies. Both decoders recover exact depth codes and packed/strided bytes
match. The hash-bound ICC is caller-selected. Physical CMYK accuracy remains
H10; descriptor/depth checks do not settle that contract.

## Current verification and its limits

Allocation refusal and defensive writer-error returns were not fault-injected.
Native decoder CI for the exact dependency revision has no completed macOS,
WASM or i686 verdict because a later main push canceled that run. These are
explicit gaps; passing pixel matrices do not establish full encoder coverage.

The current source passes `squeeze-current-final-validation.log`: native
schedule/field controls, boundary channel sets, maximum 256 request extras,
integer/float layouts, shifted first extras, LF controls, the complete default
library suite and unchanged normal/strict/divergence locks. No established
expected bytes, tolerance or ignore changed. The wrapper took 1,246 seconds;
run-heavy peak RSS was 1.73 GiB, minimum available 14,196 MiB. This measures the
whole wrapper, not encoder memory for a particular image.

Tall-grid request, legacy, streaming and strided inputs cover all 18 supported
lossless layouts. Integer animation covers both public entries, displayed
frames and ReferenceOnly layers. Planar 29/30/31-bit controls use selected
exactly representable source codes; float decoder output does not prove
arbitrary wide-integer recovery. Actual one/two-worker controls assert reported
worker counts and every source sample. The mode inventory includes all offered
named policies and selected Custom combinations, not every unrelated gate
combination. Global/local/Hybrid requests are distinguished from distinct
active squeeze writers in the inventory above.

The patched-host controls require an actual ReferenceOnly dictionary frame,
then inspect the displayed host's group count. Strict and Experimental modes,
8-bit 384-square and 16-bit 768-square repeated terminal crops, Huffman/ANS/
learned coding, and single/multi-group hosts all recover every color/depth
sample. Multi-group patched bytes equal the explicit squeeze-disabled control.
The smaller 16-bit fixture did not select a dictionary; that failed fixture
attempt is retained and supplies no affected-route evidence.

Fault injection detects descriptor range/coding/default errors, invalid group
admission, incompatible implicit fallback and a wrong last-oversized global
cutoff. The first cutoff mutation escaped a fixture with one oversized band;
that failed sensitivity check is retained. A second oversized band makes the
current independent geometry assertion distinguish first from last.
The retained exact unit executable/profile maps 94/95 regions in
`default_squeeze_params_for_groups` (uncovered allocation-error return), 39/39
in `squeeze_geometry_step`, and 28/28 in `squeeze_group_geometry_fits`, without
LLVM mapping diagnostics. Those fractions cover selected functions only.

Exact executables, profiles, compiled sources, journals and decoder diagnostics
remain archived outside git. `scripts/preserve_test_evidence.py` records copied
hashes; `scripts/run_selected_test_evidence.py` verifies profiler output before
running a selected executable. Reports use the matching preserved object and
profiles, never all historical binaries in a shared Cargo cache.

## Decoder limitation isolated by the layout matrix

The all-policy 3/19/20/21-channel E3 matrix passes (300 seconds, run-heavy
peak RSS 0.25 GiB, minimum available 21,975 MiB). The initial layout/entry
matrix retains a failing verdict: tall grayscale inputs panic in jxl-rs 0.8
while djxl recovers the retained Gray8 source exactly. jxl-rs 0.4.3 shares the
panic; the independent synthetic control is exact in jxl-oxide. Native full
squeeze metadata confirms legal global high shifts and positive remaining
tiles. [Decoder localization and retained stream](DECODER_FINDINGS.md#tall-grayscale-global-squeeze-bands-2026-11-17).

Those cases are explicitly journaled as `DECODER`, rather than counted as
successful external jxl-rs rendering. The owner-authorized native dependency
repair is pinned at `4d2eaf98`; its public raw output recovers every source
sample in the refreshed layout, animation and planar matrices, alongside
pinned djxl. The external 0.8/legacy 0.4 decoders retain the localized panic.
The initial oxide fallback also exposed a separate GrayF16 disagreement;
that failed run is retained and no tolerance was added to accept it.

The final unchecked/minimal wrapper passes (`squeeze-final-feature-check-3.log`,
rc0, 588 seconds; run-heavy peak RSS 1.68 GiB, minimum available 14,038 MiB).
It explicitly disables encoder package debug assertions and overflow checks,
requires the build-only refusal marker, then checks all four production
regressions. Minimal std/expert builds pass the core and LF contracts; the
no-default-features library check passes. Matching Rust 1.100 strict workspace
all-target and focused expert Clippy plus field-vector tests also pass. The
first local matching-compiler attempt failed on incomplete extracted dependency
sources; the isolated Cargo home leaves the shared cache unchanged.
