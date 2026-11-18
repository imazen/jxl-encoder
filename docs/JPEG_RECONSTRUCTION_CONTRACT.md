# JPEG reconstruction edge contracts

Issue 144 C6 is repaired on main. C2's current source, restored controls and passing broad gates
are described below; remote CI remains a separate verdict. The DHT portion of C3 has separate verified controls below. C3 sampling/scan
items and C4 remain independent.

## Grayscale sampling

A real grayscale 16×16 JPEG retains the same coefficient grid when its SOF
sampling byte changes from 0x11 to 0x22. The public dependency admits it. The
before encoder emits sampling modes zero in all three frame fields, and both
Rust and pinned djxl rebuild 0x11. Complete rendering succeeds and agrees with
the pinned reference; this is lost original-byte identity rather than a claim
that the pixel codestream is invalid.

The selected 64×32 controls distinguish horizontal-only, vertical-only and
both-axis factors. The native v0.12 transcodes reconstruct every selected
original, with independently observed field arrays [2,2,2], [3,3,3] and [1,1,1].
Both ordinary 1×1 controls remain exact. The independent full-envelope/TOC
checker accepts all six native controls and detects the before writer's first
non-1×1 field mismatch. Part 2 A.1 and Part 1 F.2 supply the wire relationship.

Retained before archive: `jpeg-sampling-supported-strict-before`; independent
field controls: `jpeg-header-control`. The earlier grayscale-fixture attempt
actually generated three components and failed its own precondition. A later
expanded attempt stopped at native factor-four refusal. Neither supplies a
completed supported-grid verdict. Factors above two already have an explicit
encoder refusal and remain outside the supported preservation grid.

The shared correction computes grayscale's absolute sampling modes through
its existing channel map. It retains all relative channel shifts. Independent
complete-envelope/TOC parsing checks the three serialized factors; both Rust
and native v0.12 reconstruct every selected original exactly and fully render
against the pinned reference. The restored original writer's retained failing
streams provide the fault control for this one-call correction.

| Caller | Selected execution |
|---|---|
| Parsed codestream/container | Explicit effort and default-effort wrappers |
| Raw codestream/container | Ordinary and cancellable entry points; all offered policies |
| JPEG recompression | Explicit, automatic and planar methods; exact floor and actually selected coarsened candidates |
| Embedded JPEG gain map | Grayscale secondary factors; exact gain codestream and complete original JPEG tail reconstruction |
| CLI | Automatic JPEG input and forced input with a non-JPEG extension; every strategy spelling/alias, experimental and implicit default |
| Synthetic optional subsampling | RGB8-only admission; cannot construct a one-component grayscale coefficient image |

Ordinary controls include 16x16, 64x32 and multigroup 272x144 images,
1x1/2x1/1x2/2x2 factors, and strict efforts 1/3/7/8/9/11 around entropy-policy
boundaries. Factors 3 and 4 retain clean raw/parsed refusals. Recompression
checks the actual winning candidates, native/primary sample agreement and
independent wire factors; it does not claim a recompression quality result.

Checksum-bound LLVM evidence in `c6-last-regions/regions.json` covers 25/25
regions in `compute_jpeg_upsampling` and 4/4 regions on the corrected call.
Both complete production bodies match the earlier retained instrumented
source byte for byte; the source hashes are recorded in that report. This is
coverage of the sampling correction, not the complete JPEG encoder or every
input/option combination. The newer caller tests have separate executable
identities and are not silently attributed to the older coverage profile.

The current pinned-graph library, unchanged normal/strict/divergence locks,
minimal JPEG build and focused strict lint pass in
`c6-grayscale-current-final.log` (144 seconds; build/test peak RSS 1.93 GiB,
minimum available 18,634 MiB). Existing ignores and expected bytes remain
unchanged. Additional caller/lint/region checks pass in
`c6-grayscale-all-callers.log` (23 seconds; peak RSS 0.36 GiB, minimum
available 20,140 MiB). Final fixture reach assertions have their own replay.
Remote CI is a separate verdict.

## Remaining conditional checks

C4's unrelated-APP0 before observation and candidate remain separate from the
DHT repair below. RGB-subsampled JPEG admission and actually populated scan
reset/extra-zero-run metadata still need their own production controls. The
existing restart fixtures report zero for both of those metadata arrays and
therefore do not qualify those conditional branches.

## Pinned native edge observations

`jpeg-native-edge-controls` completes small unmodified v0.12 encoder/decoder
controls. The 96×64 explicit-RGB fixture, both with Adobe transform zero and
with Adobe removed, reconstructs exactly. Adding unrelated APP0 data still
reconstructs every original byte but changes 18,273 native RGB8 samples; the
maximum code difference is 225. These are observed PNG8 samples, not a claim
about 16-bit precision. This confirms shared reference interpretation behavior;
it does not yet establish the current Rust encoder's runtime verdict.

Native prefix-only Exif/XMP and unused DC3/AC3 zero-symbol-table controls all
encode, render fully and reconstruct exactly. The selected RGB 4:2:0 input is
refused by native encoding. The retained inputs, output streams, reconstruction,
logs, tool executable hashes and sample hashes remain under that evidence
label. The Python tool version-checks both explicitly supplied executables for
v0.12.0 a7a9c78 and uses Pillow only to read the explicitly selected PNG8 output.
These observations do not replace the pending current-encoder before probes.

## Prefix-only metadata and parsed classifications (C2)

Current-main replays reproduce prefix-only EXIF/XMP reconstruction failures,
including a prefix preceding genuine metadata. Forged parsed input with fewer
classifications than APP records panics during color access, before JBRD
serialization. Those are separate regression triggers. The shared classifier
now leaves the prefix-only record opaque, matching unmodified v0.12's strict
nonempty-payload classification. It cannot consume the first-record slot.

A private validator checks parallel array counts, the marker/prefix/nonempty
payload for classified APP1, and the single EXIF/XMP-reference limit from
Part 2 clause 9.12. It runs before color access in the shared JPEG entry and
at direct JBRD serialization. This adds no public type, signature or gate.
Raw repeated metadata stays opaque after the first record and reconstructs
exactly. Manually classifying two records as external metadata is rejected,
even when their contents match: that representation violates the reference
limit. Complete TIFF/XML-content validation is not claimed; nonempty raw
content acceptance remains compatible with v0.12.

The duplicate-EXIF before control loses byte reconstruction in native v0.12:
its 22,980-byte source becomes a 28,266-byte pixel fallback, differing first
at byte 3. The original/reconstructed SHA256s and artifacts are retained with
`jpeg-prefix-distinct-content-before`. The first newly authored identical-alias
positive was incorrect; a native failure and the normative reference limit
corrected that fixture. Ordinary repeated APP1 records remain a positive.

Selected execution covers all offered policies, raw/cancellable/parsed and
bare entries, direct JBRD negatives, valid one-byte payload boundaries,
no-APP input, gain-map secondary input, and automatic/forced CLI routes with
all spellings, experimental and implicit defaults. Complete primary/native
rendering and exact JPEG reconstruction accompany the storage controls.

`jpeg-metadata-production` retains exact source, object and profile identities.
Mapped regions are classifier 19/19, first-record tracking 16/16, validator
38/38, direct-JBRD entry 3/3 and shared pre-color entry 3/3. The fresh profile
also renews C6's sampling-helper 25/25 and corrected-call 4/4 evidence against
the enlarged current function body. The older C6 whole-body binding correctly
refuses this changed body; its mixed caller wrapper therefore ends rc1 despite
passing its runtime/lint stages. No source-identity guard was weakened.
These fractions describe the named functions/windows, not all JPEG or metadata
validation. Restored faults, feature checks, broad checks and remote CI retain
their own verdicts.

The complete restored-source sequence passes in
`jpeg-metadata-final-validation.log` (rc0, 329 seconds; build/test peak RSS
1.95 GiB, minimum available 18,552 MiB): fresh production profile, CLI and
strict focused lint, JPEG-enabled library, unchanged locks and reconstruction
conformance/validity checks, default-disabled runtime matrix, optional
subsampling callers, doctests and strict workspace all-target lint. Existing
ignores, expected bytes and thresholds remain unchanged. The six restored
fault checks pass in `jpeg-metadata-regions-and-faults.log` (136 seconds; peak
RSS 1.11 GiB, minimum available 19,232 MiB). A fresh original-writer replay of
428f04ab retains the duplicate-metadata source/object/profile and runtime
artifacts, then verifies restoration of the candidate sources; its wrapper
passes while requiring the regression's nonzero test verdict.

## Empty DHT segments and unused zero-symbol definitions (C3)

An empty DHT segment has four bytes; a zero-symbol table definition has
21 bytes. A grayscale control renames its active DC0 table to DC1 and inserts
one of those two forms with DC0. The former parser produces identical complete
parsed models for the distinct inputs. Unmodified pinned v0.12 admits both
source JPEGs and reconstructs each exactly. The original encoder instead
reconstructs the zero-symbol input as an empty segment through both oracles.
Unused DC3 and AC3 definitions independently reproduce the lost 17 bytes.

The parser now retains a private empty-segment discriminator. Only an actual
empty segment emits seventeen zero counts without a sentinel; real table
definitions retain the sentinel, including a zero-symbol definition. Invalid
public-field edits to a cloned empty descriptor return an error. The table
type lives in a private module without a public re-export or constructor;
its existing inferred cloning and public-field editing remain exercised by an
external integration test. No public name, signature or strategy gate is added.

The selected ordinary/cancellable/parsed/default-effort/bare paths pass across
the offered policies. Primary and pinned native complete samples match the
native reference for ordinary, empty, unused DC3/AC3 and restart fixtures.
The distinct DC0 pair, mixed-table segments, unused class/id boundaries and
strict effort 1/3/7/8/9/11 controls reconstruct exact originals. Automatic and
forced CLI input, all strategy spellings/aliases, experimental and implicit
default input also pass. Embedded gain-map controls render the secondary stream
and reconstruct the whole primary-plus-secondary original exactly.

Initial after controls pass in `jpeg-dht-model-after.log` (26 seconds, peak
RSS 1.08 GiB, minimum available 19,315 MiB). Extended API/CLI and strict lint
pass in `jpeg-dht-callers.log` (41 seconds, peak RSS 1.08 GiB, minimum available
19,279 MiB). A newly written default-wrapper assertion initially compared its
effort-7 result with effort 3; the corrected test checks the documented effort
and decoded samples. The failed attempt remains in `jpeg-dht-expanded.log`.
Coverage, restored faults and broad checks are recorded separately; none of
these controls establishes RGB-subsampling support or rare scan-signal coverage.

The source-bound DHT report observes 104/106 parser regions and 67/74 in the
Huffman writer window. The parser's two missing regions are early truncated
length returns. They are reachable through the public marker scanner; direct
truncated-input cases were added after correcting the initial classification. The writer's seven
missing regions are propagated bit-writer allocation/error returns; successful
field, empty/real/zero table selection, sentinel and inconsistent-empty paths
execute. No allocator-failure injection or full JPEG coverage is claimed.
The existing restart fixtures do not exercise nonempty reset/extra-zero arrays.

The four restored runtime faults independently remove the parser distinction,
replace the writer discriminator with its old count-based inference, and bypass
the empty-descriptor consistency guard. Each regression fails and every source
checksum is restored (`jpeg-dht-regions-and-faults.log`, rc0, 133 seconds;
peak RSS 1.11 GiB, minimum available 19,221 MiB). Final restored profiles and
broader feature/library/byte-lock checks run independently after these faults.

The full library, unchanged normal/strict byte and divergence locks, selected
JBRD validity, default-disabled runtime target, doctests and strict workspace
all-target lint pass in `jpeg-dht-final-validation-v2.log` (349 seconds; peak
RSS 1.91 GiB, minimum available 18,505 MiB). That command's misspelled optional
filter executes no tests and supplies no optional-path evidence. The corrected
reconstruction-conformance and optional subsampling/metadata targets pass in
`jpeg-dht-gaps.log`; its overall rc1 is a single extra blank line at the final
format check (127 seconds; peak RSS 1.11 GiB, minimum available 19,020 MiB).
The formatting repair changes no test or production behavior. Final DHT-only
error-path/profile/lint/format results are recorded independently.

Final restored evidence in `jpeg-dht-complete` observes 106/106 DHT parser
regions and 68/74 in the exact Huffman writer window. The additional cloned
out-of-range count checks the typed U32-selector refusal. The six remaining
writer regions are propagated bit-writer errors for valid fixed-width fields,
zero counts and byte/sentinel values; allocator failure is not injected.
Default and default-disabled DHT execution, focused strict lint and scoped
formatting pass (`jpeg-dht-complete.log`, rc0, 71 seconds; peak RSS 1.11 GiB,
minimum available 19,199 MiB). These named regions do not qualify the complete
JPEG pipeline or the still-open C3 sampling and rare scan metadata items.

## Baseline scan metadata and absolute-factor frame grids

The producer pin `5071e85c` retains baseline trailing ZRL records in scan-block
order; ordinary coefficients remain unchanged. It is a narrow frozen-family
backport retained as an ancestor of producer main `800ec6d1`, which keeps its
newer dependency family and progress reporting. The decoder pin `f8e0f0de`
includes baseline trailing-run ordering and visible-grid reconstruction, plus
the independent native control generator. See the companion repositories'
JPEG reconstruction contracts for their separate coverage and remaining
allocation/error-path gaps. No package publication is required.

Retaining absolute grayscale SOF factors exposed another encoder defect.
At 16x8 with factors 2x2, the header matches native v0.12, but relative channel
shifts are all zero. Computing frame padding from those shifts writes a 2x1
block grid where the declared absolute factors require 2x2. The primary Rust
decoder reports AnsChecksumMismatch and djxl refuses the stream; both render
the unmodified native control. The same distinction applies to YCbCr JPEGs
whose component factors share a non-unit horizontal or vertical factor.
The shared calculation now uses the maximum raw shifts, matching v0.12
FrameDimensions::Set/YCbCrChromaSubsampling::MaxHShift and Part 1 F.2.

The retained before executable is `jpeg-scan-gray-origin-before`;
`jpeg-common-grid-app0-before` expands grayscale/YCbCr, all four 1/2 factor
pairs, one-pixel axes, 16x8/8x16, 17x9 and 257x9 through every offered policy.
The selected repaired outputs match complete primary/native samples and
original JPEG bytes. Baseline/progressive reset/ZRL/padding controls and
thirteen nonzero/long-code/restart/edge source controls separately check
nonempty reconstruction signals. These are controlled regressions, not
complete JPEG syntax or option coverage.

Caller inventory for this increment:

| Entry | Executed control |
|---|---|
| Raw/config, cancellable and parsed-data container | Every offered policy; independent native pixels and exact JPEG bytes. |
| Bare/default-effort and strict entropy efforts | Nonzero AC plus trailing ZRL, both 16x8/8x16 orientations and factors 1x1/2x1/1x2/2x2; exact pixels, and exact original bytes where JBRD exists. |
| Explicit/automatic/planar recompression | Lossless floor and scale two; primary/native samples use the existing two-code U16 agreement bound. |
| Nested JPEG gain map | Both orientations and non-unit factors; exact gain pixels and complete original primary/secondary JPEG tail. |
| CLI automatic, forced and coarsening | Both orientations, efforts 3/11, every offered strategy spelling/alias and Experimental switch; exact API bytes, native pixels and contracted reconstruction. |
| Optional RGB8 coefficient source | Existing optional shape/metadata tests; its unequal chroma factors retain the previous grid calculation. |
| Pixel requests/streaming | These accept pixels rather than compressed JPEG bytes and do not call this JPEG writer. No streaming JPEG API is claimed. |

JPEG transcode passes effort/resources to the shared writer and does not resolve
LosslessConfig's strategy gates. The policy matrix checks accepted request
bindings, not distinct strategy-specific JPEG algorithms. The new grid tests
include one-pixel axes and the 256-pixel group boundary. Progressive reset
signals are exercised separately; common-factor progressive JPEG input remains
a follow-up control. Arbitrary caller-forged scan metadata is not fully
validated by this increment.

The instrumented broad library/unchanged locks/reconstruction/minimal/optional/
doctest/workspace lint chain passes (`jpeg-scan-grid-core-check.log`,
807 seconds; peak RSS 1.95 GiB, minimum available 18,428 MiB). The normal
uninstrumented chain passes those functional stages plus the expanded CLI
controls, then fails one new-test unnecessary-unwrap lint (`jpeg-scan-grid-normal-check.log`,
630 seconds; peak RSS 1.98 GiB, minimum available 18,028 MiB). An equivalent
match preserves every assertion; corrected focused normal tests, encoder/CLI
strict lint and formatting pass (`jpeg-scan-grid-focused-check.log`, 56 seconds;
peak RSS 0.68 GiB, minimum available 19,033 MiB). No existing threshold, byte
expectation or ignore changes.

Fresh bound API regions observe sampling 25/25, grid 8/8, scan import 16/17
and scan-more emission 48/53. The import gap is the defensive parameter-mismatch
return; emission gaps are propagated writer errors. No allocator failure was
injected. Four independent runtime faults detect the old horizontal/vertical
grids and omitted reset/extra-run imports, restoring source by checksum
(`jpeg-scan-grid-faults.log`, 126 seconds; peak RSS 1.11 GiB, minimum available
19,148 MiB). The initial region exporter used the wrong helper name; the
corrected report reuses the same bound object/profile without relaxing hashes.

CLI profiles are unique per child PID and are exported against the retained
CLI executable separately from the parent driver. They observe grid 8/8,
transcode selector 4/4, transcode call 3/3 and coarsen call 8/8. This is selected
window coverage, not full CLI or JPEG coverage. The corrected report/CLI chain
passes (`jpeg-scan-grid-restored-regions-v2.log`, 49 seconds; peak RSS 1.15 GiB,
minimum available 19,119 MiB).

### Progressive absolute-factor follow-up

`common_grid::progressive_common_absolute_sampling_frame_grid_matches_native`
uses progressive SOF2, one DC scan and separate visible-block AC scans. Grayscale
and equal-factor YCbCr use each supported absolute factor at 1x1, 16x8, 8x16,
17x9 and 257x9. Unmodified v0.12 transcodes and reconstructs each source first;
all offered API policy requests then require complete primary/native frame
samples and exact reconstructed JPEG bytes. This adds progressive scan geometry
to the baseline common-factor controls; it does not claim arbitrary progressive
JPEG coverage.

Normal and default-disabled builds pass, as does strict focused lint.
Restoring either old dimension calculation fails both baseline and progressive
controls. Dropped reset and extra-run mutations remain detected by the scan
controls. All four mutations restore exact source checksums before the fresh
passing evidence capture. Logs are `jpeg-progressive-common-grid-first.log`
and `jpeg-progressive-common-grid-qualification.log`; the latter exits zero
in 269 seconds (run-heavy peak RSS 1.11 GiB, minimum available 19,064 MiB).
Exact executable/profile/source evidence is retained as
`jpeg-progressive-common-grid-restored`; its region report retains the same
25/25 sampling, 8/8 grid, 16/17 import and 48/53 scan-writer measurements and
previously documented defensive error-return gaps. No production arithmetic,
expected bytes, thresholds or ignores change in this follow-up.
