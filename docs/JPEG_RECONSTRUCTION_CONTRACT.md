# JPEG reconstruction edge contracts

Issue 144 C6 is repaired on main. C2's current source, restored controls and passing broad gates
are described below; remote CI remains a separate verdict. C3 and C4 retain
independent before evidence and unlanded candidates.

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

## Metadata and conditional checks

The prepared unrelated-APP0 control preserves explicit RGB IDs, with and
without Adobe transform-zero metadata, and compares full rendering to the
unaltered RGB reference. Native mutated-input behavior is recorded separately
because the inspected reference source shares the APP0-presence rule. This
control has not run.

The prepared prefix-only APP1 controls cover isolated Exif/XMP prefixes and a
prefix preceding genuine metadata. They require opaque preservation, unchanged
interpretation and exact reconstruction. A dependency-independent forged
parsed-data control and unusual RGB-sampling/DHT/scan probes remain pending.
These selected cases do not establish full JPEG acceptance or Annex A coverage.

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
