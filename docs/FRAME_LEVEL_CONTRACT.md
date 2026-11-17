# Frame-dependent codestream levels

Issue 144 H14 concerns level selection after the actual frame sequence is
known. The cumulative zero-duration-area regression is prepared and its strict
encoder run is starting. Pinned native results are below. No encoder repair
or encoder passing verdict is recorded yet.

## Current caller inventory

`container::compute_codestream_level` covers image dimensions, channel model,
profile size and modular-buffer width. It receives no frame sequence.
`api::ingest::compute_required_level` delegates to it at the request still,
streaming lossless and finalized animation wrappers. The JPEG wrapper calls
the classifier directly. The raw/prepared VarDCT surface returns a codestream;
its caller owns container-level signaling.

Both animation writers emit into a shared BitWriter. Lossless automatic-delta
trials use temporary writers and append only the winning result. Lossy patch
references and LF frames write their headers through the shared FrameHeader
writer. Consequently, requested full-canvas buffers alone do not describe the
actual serialized sequence: crops, winning deltas and auxiliary frames matter.

The pinned unmodified v0.12 `VerifyLevelSettings` source leaves cumulative
zero-duration area and frame duration as explicit unimplemented checks.
Part 1 Annex M remains the independent requirement; matching a reference
level declaration does not establish its correctness.

## Prepared baseline

The public area control uses 4096-square RGB8 frames at effort one, in both
writers. Sixteen and seventeen explicit full-canvas zero-duration frames are
followed by a displayed frame; a repeated-source crop control distinguishes
actual frame area from requested canvas area. Every emitted frame extent and
duration must be observed, the 16-bit modular-buffer promise must remain true,
and primary Rust plus pinned djxl must fully render the displayed source.
These are prepared controls, not executed evidence.

## Observed pinned v0.12 baseline

The unmodified public encoder and decoder finish all four controls. Both
lossless and lossy select Level 5 at an observed total zero-duration area of
268,435,456 pixels and also at 285,212,672 pixels. All native layer samples
match their chosen source endpoints. The latter declaration exceeds the
Annex M Level 5 area bound; successful decoding does not repair that metadata.

Exact source/executable hashes, actual per-layer dimensions/durations and full
native logs are retained in `zero-area-native`. The wrapper completes with rc0 in 43 seconds (run-heavy peak RSS 1.10 GiB,
minimum available 20,165 MiB). The executable SHA256 is
`a524f46e8983bac8aa5a85e23087d57f7158ab6d59c6e82c9293397417a94409`.

## Native duration baseline

The unmodified pinned public v0.12 encoder and decoder also accept durations
below 1/120 second while declaring Level 5. Sixteen selected writer/timebase
controls fully render; actual serialized clocks, duration fields and samples
are retained in `duration-level-before/native`. Part 1 Annex M supplies the
level requirement, so reproducing this reference omission is not acceptance.

Source SHA256 `fd93479c19d7fc560566d925b4bf077f9fcd1d60fab612d1bc45b5a644cad53d`;
executable SHA256 `0523ad5d599f07b5d1fc3159569596700922c089a70debd7bd91e3743ee97595`.
Run `just animation-duration-native <source> <build> <oracle> <new-output>`.
This reference control does not repair encoder level signaling.
