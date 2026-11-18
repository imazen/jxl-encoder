# JPEG reconstruction edge contracts

Issue 144 C6 is reproduced through the public raw, cancellable and parsed
entries. C2, C3 and C4 still need completed runtime attribution; their newly
prepared tests are not evidence of execution.

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

A shared correction now computes grayscale's absolute sampling modes through
the existing channel map. Its after, offered-policy, native reconstruction,
exact-region, fault, unchanged-baseline and lint gates remain unrun. No C6
repair is landed yet.

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
