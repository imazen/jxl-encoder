# Frame-dependent codestream levels

Issue 144 H14 now propagates actual emitted-frame requirements through the
container-owning still/streaming paths as well as animation. The optional
CLI rate-control bypass is a separate outstanding caller. Consecutive
zero-duration frame area above 2^28 pixels or a positive duration below 1/120
second requires Level 10. Successful header records travel with
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


## Historical November 19 auxiliary source-identity gate

The all-offered upsampled replay passes with frozen binary
69acdde7484b62131af0ed9925e5c43cd7162a08c4a46fb1c7aad229e7a94a5c.
At a 16384×16384 display canvas the actual patch reference plus main requires
Level 10, while the no-reference control stays Level 5. The 16-bit modular promise
is observed independently so it cannot mask cumulative area. Complete frame/TOC
parsing, primary/native reconstructed samples and streaming/legacy/writer/stride
identity pass. Wrapper all-offered-upsampled-2026-11-19 exits 0 / 1259 seconds, run-heavy
peak RSS 6.28 GiB / minimum available 39,369 MiB. This measures that retained run only.

At that stage source comparison established a qualification gap. The frozen
output_frame_level, encode_two_pass and request encode_prepared bodies match
the current candidate, but enclosing lossless/lossy/streaming and encode_inner
bodies predate the landed CMYK, modular-width and spline updates. This is not
whole-current-source qualification. Rebuild current main plus only the private
level repair and replay the large upsampled, lossless-patch and LF families
before final acceptance. Frozen streams, binaries and incomplete predecessors
remain retained separately; no earlier result is relabeled as a current pass.


## November 19: remaining CLI rate-control caller

The CLI rate-control branch constructs VarDctEncoder directly and writes the
returned prepared-stream bytes as its final file. It accepts the LF-frame flag
and can use the prepared patch producer; it does not use the request wrapper
that classifies actual emitted frame records. Raw/prepared library methods leave
container-level signaling to their callers, so the CLI needs its own gate in
this audit. The small actual-file controls below establish independent bypasses; the large
LF caller still has no execution verdict.

A new caller-selected large probe prepares small and 16384-square RGB PNM
controls, LF on/off, records actual frame geometry and the modular-width
promise, then completely renders primary/native samples before reporting a
level mismatch. It has not executed yet. The frozen older auxiliary binary is
not substituted for this caller or for a current-source qualification.


## November 19: observed CLI rate-control bypasses

A current-main source-isolated before run uses 32x17 and 262144/262145x1
RGB8 controls at effort one, zero rate-control iterations, DCT8 and no filters.
The actual modular-buffer promise remains 16-bit. At width 262145, the output
is a bare implicit-Level-5 codestream although Annex M requires Level 10.
Explicit container-on is ignored even for the small control. Container-off at
the Level 10 dimension is accepted and overwrites the caller's existing file.
Both primary Rust and pinned native v0.12 completely render every emitted
stream; decoder acceptance does not establish correct level signaling.

The same CLI route accepts an RGB input with a supplied CMYK ICC, drops that
profile and emits no Black channel. It also replaces the source PNG's gamma
50000 metadata with sRGB. The ordinary request control retains the corresponding
Gamma encoding. Original-profile native and primary float samples agree within
0.001 in these runs; no physical magnitude of the gamma error is inferred from
those original-encoding samples.

The expected failing wrapper retains source identities, commands, destination
sentinels, headers and complete native/primary outputs under
`frame-level-cli-before-gates-v2`; the three groups are
`frame-level-auxiliary-contract/run-8` through `run-10`. Production matches
main `aac8dcce`, without the private H14 transport candidate. The wrapper exits
1 in 43 seconds, run-heavy peak RSS 1.04 GiB, minimum available 18,362 MiB.
The preceding compile failure used the wrong PNG test API and is retained
separately; the corrected fixture uses the existing zenpng dependency.

The repair must carry the selected iteration's actual frame requirement and
combine it with the serialized basic-header constraints. Raw library methods
retain their existing codestream contract. Container-owning CLI code also needs
to consume or explicitly refuse source color/metadata options rather than
silently dropping them. The private transport prototype is unqualified and
adds no public API at this stage.


## November 19: LF and additional CLI input layouts

The separate source-isolated follow-up reuses the exact main-only CLI ELF
`aeafeee5eef02f7d0956672f5cf7f58d444f16f704ca2ba66207e38ed006133a`.
The LF-off 32x17 RGB8 control completely renders. Enabling LF is accepted,
but both primary Rust and pinned djxl v0.12 fail complete rendering: the
primary reports `NoLfFrame(0)`, while native reports a failed image decode.
This is an actual small CLI caller failure, separate from the still-unexecuted
large CLI LF-area probe.

Gray8, Gray16 and RGB16 inputs at 32x17 and 259x17 panic in that rate-control
branch. Gray inputs reach an RGB-triplet remainder index; RGB16 reaches a
precomputed linear-RGB length assertion (3264 versus 1632 for 32x17).
The matching RGB8 controls encode and fully render. The failed input layouts
have no decoded-pixel error magnitude because encoding does not complete.

The follow-up also reproduces the earlier container/dimension, CMYK-ICC and
PNG gamma failures. Full child logs, identities and accepted control outputs
remain in `frame-level-cli-followup-before-gates-v1/before`, with pixel/control
records in `frame-level-auxiliary-contract/run-11` through `run-15`. The wrapper
exits 1 in 24 seconds, run-heavy peak RSS 1.02 GiB, minimum available 18,324 MiB.
These are before failures. No rate-control or prepared-LF repair is qualified
by this run.

## November 19: still/streaming auxiliary finalization

Successful frame records now remain attached to the selected still, prepared,
precomputed and prequantized VarDCT payload internally. Still and streaming
lossless capture the selected writer's frame requirement. Container finalizers
take the maximum of header and emitted-sequence requirements. Public raw byte
return signatures and preset values are unchanged. Raw surfaces retain their
caller-owned container contract; request wrappers consume the private level.

The refreshed archived source binary
`041b0c619b5f5a73333436d234ea16d6c5df05cf16ca5186ed87eca2fb61a89d`
completes all 17 offered bindings for upsampled patch references, lossless
patches and LF frames. Each family retains streams, journals, source identities
and completed primary/native rendering. The LF-off control observes area
268435456 at Level 5; LF-on observes 272629760 at Level 10, with identical
still/streaming bytes. Its completed profile is
`d3960f80b298914b1518e66e33360d39e5ab3b05edf5964d2940917568823bc9`.
The LF wrapper reports peak RSS 14.22 GiB and minimum available 31956 MiB.
Its 8736-second wall includes serialized intermissions and is not latency.

Those large results are not relabeled as whole-current-head executions.
Exact classifier, two-pass and enclosing still/streaming transport bodies
match the selected current source. `encode_inner` differs only by the already
landed LF-plus-upsampling refusal. The LF cells use scale one; upsampled cells
have LF disabled. This establishes why that new precondition accepts these
inputs; it supplies no executed-region credit for the newer guard.

The current-main continuation passes small packed/strided/writer/streaming
controls, the library baseline, unchanged normal/strict locks and divergence
checks, strict focused lint, no-default build and scoped formatting. Its
retained CLI and selective production binaries are build-only evidence.
Wrapper rc0, 761 seconds, peak RSS 1.59 GiB, minimum available 17488 MiB.
A separate current instrumented small-frame replay completes full primary and
native pixels. Direct current raw/prepared LF, attached dictionary/alpha,
DC-width and geometry checks also pass in instrumented default and minimal
builds (wrapper rc0, 698 seconds, peak RSS 0.93 GiB, minimum available 17598 MiB).
Existing ignore annotations, thresholds and expected bytes are unchanged.

The earlier exact region report above remains scoped to its recorded bodies
and windows; it does not cover allocation failures or every input. Current
small/prepared profiles and the large sequence profiles retain separate
identities. The CLI controller's source/metadata/container bypass remains
under the separately approved request-API repair.

The standalone H14 test bundle contains the small finalization and explicitly
selected large resource families. Pending CLI controller tests live with the
CLI repair instead of entering the core bundle. The separated bundle passes
current instrumented small pixels, minimal execution and strict test lint.
Its first packaging command fails only on two trailing blank lines; that
nonzero record remains retained. Removing those lines changes no function
body, assertion or mapped line position, and the separate scoped format check
passes. CI explicitly selects small default/minimal controls after native
reference tools are installed; large families remain caller-selected probes.


## November 19: configured pixel-output container policy

The actual CLI controller follow-up and separate ordinary requests reproduce
ignored container policies: `Always` returned a bare32x17 stream, and `Never`
accepted the262145x1 Level10 output. The config stored the selection but no
output finalizer read it. `container-policy-before-gates-v1` retains both
ordinary failing regressions and successful unmodified v0.12 controls.

Still/prepared requests, the lossless streaming finalizer and animation now
apply the same private policy helper. Lossy streaming already reuses the
request path. `Always` forces framing; `Auto` preserves prior byte choices;
`Never` drops container-only metadata and returns InvalidInput when the
computed header/frame level requires framing. Embedded ICC stays inside the
codestream: complete extracted codestream bytes agree across all three
policies on the frozen sRGB ICC controls.

The native v0.12 CLI's `--container=0` can wrap when required. Our existing
`ContainerMode::Never` contract is stricter and promises a refusal; that
intentional API difference remains. JPEG reconstruction and explicit raw
codestream methods have their separate mandatory-wrapper/raw contracts and
do not pass through these pixel finalizers.

The final minimal-feature matrix passes every selected request, legacy,
streaming, animation, metadata and dimension-level case under all17 offered
policies. The final default ICC control, default library, unchanged normal/
strict byte locks and divergence checks, strict focused Clippy, minimal
compilation and scoped format checks pass. Earlier combined default batches
retain their failing ICC observer-domain verdict; their ordinary/streaming/
animation/level cases pass. The corrected ICC observer explicitly requests
native sRGB and maps the primary's documented linear fallback for the frozen
sRGB profile, preserving the0.01 source and0.001 decoder-agreement bounds.

Accepted source: `container-policy-core-selected-v4`; final qualification:
`container-policy-final-isolated-v4` and `container-policy-final-gates-v4`.
Wrapper rc0,216seconds,peak RSS1.56GiB,minimum available17,499MiB. These are
build/test resources, not encoder latency. Destination-wrapper and mapped-region follow-up is recorded below; test
counts do not establish coverage. The separately approved request-controller/CLI API remains
under qualification and is not part of this shared policy change.


### Current-main destination wrappers and remaining region gaps

The original v5 profile leaves five animation-finalizer regions and eight
colr/hCdR emission regions unobserved. The supplement below extends this union
and records the remaining gap. No allocation-failure injection or whole-code
coverage is claimed.

The current-main wrapper gate passes every offered policy in both encoder
modes, lossless/lossy,32x1 Auto/Always/Never and262145x1 Never. Request stats,
append and Write; legacy append; and streaming stats, append, Write and
seekable Write all match the ordinary encoded bytes exactly. Every negative
case preserves the existing destination. The accompanying native/primary
policy controls also pass. Journal: `run-744/case-748/wrappers.tsv` under the
retained frame-level-auxiliary-contract output root.

The source-bound region export observes policy helper16/16, metadata wrapper
18/18, config accessor7/7, animation finalizer30/35 and box wrapper25/33.
These are mapped regions of the exact retained functions, not a count of
covered inputs or complete encoder coverage. Exact ELF
`0f9dcf4ee37d58f8d4df510140a35e92e93fbb7a38c663d6d00cc59ba7589453`;
profile`dc2b7aed1b574926ca81ed32c03d59c6d8da299935483544af136db109033e6b`.
Runtime archive`container-policy-wrappers-gates-v5`; region export
`container-policy-regions-v5`. Runtime wrapper rc0,1064seconds,peakRSS0.97GiB,
minimum available18,119MiB; this includes repeated encodes/decodes and artifact
I/O, not an encoder latency measurement. Focused strict lint and format pass
in `container-policy-wrapper-lint-isolated-v5`. The initial encompassing region
queue fails on reversed tool arguments, before export; the corrected container
export succeeds. The stale request profile is correctly refused when its full
animation-finalizer body differs from current source; that separate request
profile is being regenerated, without weakening body-identity checks.

### Reproducible evidence operations

`just serialized-qualification-queue <plan> <output>` executes explicit
commands in order and retains each verdict. A failing cell keeps the overall
queue nonzero; independent later controls retain their own results. Optional
before gates require a retained runtime failure and its diagnostic before
applying a candidate. Build-only evidence supplies no runtime verdict.

`just stage-archived-runtime <archive> <template> <output>` checks the retained
ELF, library hashes and required dynamic dependencies before runtime-only
replay. `just archive-inactive-cargo <source> <output>` refuses a live build or
executable, verifies every archived file, then reclaims only the named Cargo
output. Source exports, reference binaries and evidence archives are separate.
These commands have been exercised by the issue-144 selected-source queues;
source-body binding remains mandatory when combining profiles.


### Optional boxes, alpha and supported CMYK animation routes

One animation-finalizer error-propagation region remains unexecuted: the
`compute_required_level` failure return. This supplement does not classify it
as unreachable or inject allocation failure. The successful exact-body union
observes output policy16/16, metadata wrapper18/18, configuration accessor7/7,
box wrapper33/33 and animation finalizer34/35 mapped regions. Region fractions
are execution evidence, not complete input-space coverage.

The supplement renders RGBA8 animation through native v0.12 and the pinned
primary under every inventoried policy, both encoder modes, lossless/lossy and
all three container selections. Alpha137/255 is exact in both decoders; the
frozen sRGB ICC bytes and optional colr/hCdR payloads remain exact. Lossy CMYK
animation with its required profile reaches the Black classifier and refuses
Level10 under Never. Lossless interleaved CMYK animation has no writer route:
the test requires its exact UnsupportedPixelLayout(Cmyk8) error before the
container finalizer. Earlier retained test failures confused these two routes;
the subsequent fixture correction also needed the traced error's inner value.
Neither changes production behavior or an established expectation.

The maintained destination-wrapper probe uses small F32 input for its Level10
refusal, retaining all routes, exact successful bytes and unchanged destinations
on error. The separately named wide request/streaming/animation controls remain;
the historical262145-wide wrapper evidence above is retained. The journal now
records the layout as well as width and policy.

Accepted test source: `container-policy-boxes-selected-v8`. Runtime and strict
lint/format qualification: `container-boxes-q47-isolated-v1`; actual runtime
archive: `container-boxes-q47-runtime-v8`. Combined exact-body regions:
`container-current-regions-q47-v1`. ELF SHA256
`4bbb532590439c7efb1e7e924fd2ca2916a780bfedf8552df2e1712ad602624d`;
profile SHA256
`5455ed60a70b348e0602cc825626d5c3e6d6cc911e7e1fb9f9287c6c9e352e0a`.


The complete maintained container matrix also passes with defaults disabled
and explicit std/convenience/expert support (`container-minimal-q48-isolated-v1`).
It retains all native/primary rendering, exact alpha/ICC/optional payload,
destination and wide-geometry assertions. The native Linux ARM64, macOS ARM64
and Windows ARM jobs now select the default and minimal matrix explicitly.
These workflow commands have local runtime evidence; their fresh platform
verdicts remain separate. No threshold, ignored test or production choice changes.
