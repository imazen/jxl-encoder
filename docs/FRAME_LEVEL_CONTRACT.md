# Frame-dependent codestream levels

Issue 144 H14 remains open for still-image auxiliary-frame propagation and LF
caller verification. Animation now classifies the actual emitted sequence:
consecutive zero-duration frame area above 2^28 pixels or a positive duration
below 1/120 second requires Level 10. Successful header records travel with
selected writer bytes, including winning full-frame deltas; discarded trials
do not contribute. No public API or preset value changes.

The animation-only production source passes the pinned library baseline,
unchanged normal/strict byte locks and divergence checks, strict focused lint,
no-default build and scoped formatting. Public legacy/request controls exercise
both writers across every offered policy, including Experimental variants.
Full rendering checks use primary Rust and pinned djxl v0.12; independently
parsed extents and rational durations establish the classification inputs.

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

## Baseline configuration

The public area control uses 4096-square RGB8 frames at effort one, in both
writers, with explicit fixed-tree/Zero-predictor lossless settings to isolate
frame-area accounting. Sixteen and seventeen explicit full-canvas zero-duration frames are
followed by a displayed frame; a repeated-source crop control distinguishes
actual frame area from requested canvas area. Every emitted frame extent and
duration must be observed, the 16-bit modular-buffer promise must remain true,
and primary Rust plus pinned djxl must fully render the displayed source.
The completed before and local candidate runs are recorded below.

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

The first automatic-learner strict attempt was deliberately interrupted before
completing a cell. A retained live perf sample observes neighbor gathering,
weighted prediction and sample gathering; it is not a level or pixel verdict.
Its exact source, executable, partial journal and profile remain retained under
`zero-area-strict-before`. The controlled replay disables the unrelated learner
and pins the fixed predictor; dimensions, areas and sample assertions remain.

## Observed strict encoder before repair

`zero-area-strict-production-before` fails the real public regression in both
writers: 17 full zero-duration frames produce an observed 285,212,672 pixels
while declaring Level 5. Both primary Rust and pinned djxl fully render the
selected source endpoints; the modular 16-bit promise is explicitly true.
Sixteen full frames observe 268,435,456 pixels and correctly retain Level 5.
The actual crop controls observe 16,777,232 lossless pixels and 16,778,240 lossy
pixels, also correctly retaining Level 5. Requested canvas area times frame
count would misclassify these controls.

The large replay instruments only its test driver; ordinary optimized encoder
code emits the stream. It supplies source/frame/pixel evidence and no production
coverage credit. Separate small emission and branch controls must measure the
changed code. The earlier two globally instrumented attempts were interrupted
without a completed cell; neither is a passing or failing level regression.
Their logs, binaries, profiles and journals remain retained.

## Observed duration before repair

The expanded strict before run retains every selected header, native render
and independent timing parse. Positive durations immediately below the 1/120
second boundary declare Level 5 in both writers; primary jxl-rs 0.8 rejects
all six cases for exceeding the declared frame-rate limit. Zero durations,
ReferenceOnly fields and legal rational-duration boundaries remain controls.
The first run stopped at the primary rejection; the expanded run records that
rejection as a failure and continues the remaining cells, still failing overall.

The new pinned public native controls fully render sixteen selected timebases
and declare Level 5 throughout, including the below-boundary cases. Their
actual tick rates and durations are retained in `duration-level-before/native`;
source SHA256 `fd93479c19d7fc560566d925b4bf077f9fcd1d60fab612d1bc45b5a644cad53d`,
executable SHA256 `0523ad5d599f07b5d1fc3159569596700922c089a70debd7bd91e3743ee97595`.

## Candidate history and separate still-image work

The first tracker records successful FrameHeader writes and carries those
records with the selected BitWriter payload. The animation finalizer applies
the resulting requirement to the actual common-header level. Its unit,
strict-duration and strict-area controls pass (`frame-level-strict-candidate`:
164 seconds, run-heavy peak RSS 1.89 GiB, minimum available 19,314 MiB).
The private still/streaming/VarDCT transport compiles and passes its unit gate
(34 seconds, peak RSS 1.87 GiB, minimum available 19,313 MiB); public raw output
signatures are unchanged. These are candidate checks, not a landed repair.

The all-offered duration and area matrices pass on their recorded candidate
(test runtime 1433.39 seconds). These large runs instrument only the integration
driver, so they receive no production-region coverage credit.
The later animation-only baseline, fault and region gates are recorded below.
The auxiliary audit found a units issue in the first tracker: it counted coded
samples after spatial downsampling. The corrected tracker counts reconstructed
pixels, retaining only the LF frame's own grid reduction. An actual 212×207
patch reference plus a 16384-square reconstructed main frame totals 268,479,340
pixels. The retained before stream declares Level 5; the corrected candidate
selects Level 10. Both decoders fully render, with every clamped color sample
agreeing within the existing 0.001 cross-decoder bound. Packed, strided,
legacy, request, writer and seven-row streaming bytes agree.

The corrected unit/delta/auxiliary command passes in 147 seconds, run-heavy
peak RSS 6.40 GiB, minimum available 14,391 MiB. The strict large-area before
and candidate comparison preserves all six codestreams and every frame payload;
only the two over-limit wrappers change. This selected comparison is recorded
in `zero-area-strict-payloads.json` and does not establish unchanged bytes for
all other callers. The upsampled auxiliary before/after comparison also
preserves both codestreams and every frame payload; only its patch-enabled
wrapper changes (`auxiliary-strict-payloads.json`).

The winning full-delta probe independently exposed a source-slot defect,
described in [the delta contract](ANIMATION_DELTA_CONTRACT.md). Its repair is landed separately. Complete H14 caller verification remains open.

## Winning-delta before control

After the independent source-slot repair, a regional-change 4096-square
sequence selects sixteen full-frame zero-duration deltas. Their observed run
area is 285,212,672 pixels, but the wrapper declares Level 5. The competing
cropped replacements total 218,103,808 pixels and correctly retain Level 5.
Both variants fully render the exact final source codes through primary Rust
and pinned djxl. The corrected run retains its rc101 level failure in
`frame-level-winning-delta-corrected-before`; the first attempt stopped on
an own-fixture count that included the final positive-duration identical Add
frame and receives no completed level/render verdict.

The separate lossless still/streaming auxiliary before probe completes on
an isolated larger host: both variants observe 268,481,032 pixels and declare
Level 5, with every primary and native source sample exact. The candidate
selects Level 10 for both, preserves still/streaming byte equality and passes
the same complete source-sample checks. Before and after use exact retained
ELFs and a checksum-bound copy of the same djxl v0.12.0 `a7a9c78` executable,
loader and dynamic libraries. Source and executable identities stay distinct.

The first admission attempt omitted the request's explicit limit and refused
before encoding. The corrected local attempt completed still rendering but
hit its 16 GiB cgroup cap during streaming encoding; its terminated time
collector supplies no valid process peak-RSS result. The default 8 GiB
encoder admission limit remains unchanged; the trusted probe explicitly raises
it while an independent process cap remains in force.

The larger-host before run fails only the two level declarations: 231 seconds,
run-heavy peak RSS 16.62 GiB, minimum available 28,336 MiB. Its after run passes:
232 seconds, peak RSS 16.63 GiB, minimum available 28,323 MiB. These are resource
observations, not a performance comparison. The retained run directories are
`issue144-large-replay-2026-11-18/{before,after}` and the corresponding `pixels`
subdirectories.


## Large resource probes and ordinary checks

The new 4096-square multi-policy area matrices and 16384-square auxiliary
probes are explicitly selected with `--include-ignored` through the justfile.
Their assertions and retained before/after verdicts are unchanged. The large
lossless streaming probe measured above 16 GiB and runs under a 24 GiB cap;
ordinary workspace discovery must not launch it on a small CI runner. Small
clock/duration controls and metadata/emission units remain ordinary tests.
Missing corpus files always fail a selected probe.

## Verified animation-only boundary

The isolated animation source retains the parent still/VarDCT callers. Its
combined library, unchanged lock/drift, strict lint, minimal-build, strict-area,
all-policy duration and fault command passes: 954 seconds, run-heavy peak RSS
1.84 GiB, minimum available 19,422 MiB. Thirteen intentional tracker faults
are each detected and restored: record omission, compact emission, aligned and
unaligned appends, area boundary/reset, rational duration boundary/denominator,
spatial units, hidden/static duration and LF/crop geometry.

A final fresh all-policy area run passes on that same production source:
1,453 seconds, peak RSS 1.07 GiB, minimum available 19,793 MiB. Its executable
SHA256 is `292f83a66ff894b87ae4c413755a5921a69ea8328872d914fe441f0a86aadd29`.
This large replay instruments only the driver and earns no production-region
credit. Separate instrumented units/public duration controls retain exact source
regions: required-level body 65/65, record 31/32, append 10/11, byte append
17/18, bit append 25/28, writer record 5/5, writer classification 7/7, animation
classification 7/8 and final container maximum 2/2. These are observed region
fractions, not proof of every input or platform. Missing regions are allocation
or storage-write error returns and a classification error closure unreachable
through validated public animation input. Deterministic allocator-failure
injection remains absent.

Reproduction recipes and evidence directories are
`frame-level-animation-isolated-validation` / `frame-level-animation-isolated`,
`frame-level-animation-final-sources`, and
`animation-large-area-production-evidence` / `frame-level-animation-final-policies`.
The real LF/main-grid probe now completes before and after the still-transport
candidate. Its 16384-square main frame plus 2048-square LF frame observes
272,629,760 pixels. Both still and streaming declared Level 5 before and Level
10 afterward; the no-LF 2^28-pixel control remains Level 5. Complete primary
and native rendering and still/streaming byte equality pass. Before: rc101,
149 seconds, peak RSS 14.20 GiB, minimum available 29,308 MiB. After: rc0,
142 seconds, peak RSS 14.20 GiB, minimum available 31,298 MiB. These are resource
observations, not a performance comparison.

The all-offered LF matrix also passes as an individual test. Its enclosing
command fails a separate upsampled-parser invocation and is later interrupted
in lossless testing; it supplies no positive coverage credit. The remaining
all-offered upsampled and lossless replays are still required. Still/streaming
transport remains an unlanded candidate.


## Still transport verification and retained replay tools

The isolated still candidate passes the library baseline, unchanged byte locks
and divergence checks, strict focused lint, no-default build and formatting
(rc0, 556 seconds, run-heavy peak RSS 1.91 GiB, minimum available 19,004 MiB).
The final small controls use 9x3, 129-square and 300x17 RGB8 inputs, efforts
one/three, lossy/lossless and every offered policy. They require Level 5,
packed/strided/writer/streaming byte agreement, complete rendering in both
oracles and exact lossless samples. The existing raw/precomputed/prequantized
alpha regression also passes, including progressive and multi-group outputs.
The continuation with strict lint passes in 187 seconds, peak RSS 1.07 GiB,
minimum available 20,062 MiB. No existing expectation is changed.

The positive strict upsampled and lossless replays retain instrumented production
classification with only the unchanged patch-template search body excluded.
The compiled map confirms the exclusion; unrelated generic parallel-map
instantiations remain mapped. The strict lossless replay passes in 620 seconds,
peak RSS 16.63 GiB, minimum available 26,982 MiB. Its actual still and streaming
area is 268,481,032 pixels, both Level 10, with exact source samples and equal
encoded bytes. The corrected upsampled replay passes in 106 seconds, peak RSS
6.24 GiB, minimum available 39,159 MiB. None of these resource measurements is
a speed comparison.

Exact-source region windows in `frame-level-still-regions` observe both final
container maxima at 4/4, the still and streaming lossless captures at 7/8 each,
the lossy stats assignment at 1/1, the VarDCT classifier at 13/13, two-pass output
transport at 6/6, the static output window at 13/14, two-pass serialization at
8/9, precomputed return at 2/2 and prequantized return at 1/1. The four missing
regions are classification-error propagation after successful validated header
emission. Valid generated static headers have no animation clock and use legal
geometry/scales; invalid inputs fail before these finalizers. This is a
source-invariant reachability explanation, not injected coverage of those error
returns. Existing allocation-failure gaps remain separately documented above.

`stage_archived_runtime.py` checks an exact retained ELF and its required ELF
library names against a checksum-bound runtime closure. `run_archived_test.py`
checks sources, object and libraries before launch, records the actual PID and
exit status, and retains the completed profile. `archived_test_status.py`
checks the live command and selected per-case journals. Positive coverage
export refuses failed or unexecuted identities. The small replay control
passes through the staged loader with matching object/profile hashes. An
initial replay on a non-executable artifact mount fails before launch and is
retained; executable staging uses a caller-selected suitable filesystem.
