# Animation alpha precision and visibility

The current lossy animation API accepts floating layouts. Earlier statements
that it supported only integer layouts were incorrect. Alpha is lossless by
default, including when color is lossy. Progressive equality against Single
alone cannot establish that contract: both paths previously coded float alpha
from an eight-bit approximation.

## Independent reference and failing observations

`scripts/animation_alpha_oracle.cc` calls the unmodified libjxl v0.12 public
encoder API. `just animation-alpha-native` builds against the existing pinned
source/build; `just animation-alpha-native-check` renders all displayed frames
through the version-checked djxl. Values 0.0625, 0.9375 and binary32 0.1234567
retain their exact input bits. A second control compares native preprocessing
with keep-invisible off/on, with every alpha nonzero and a 1/1024-alpha red
sample between opaque blue samples; decoded colors agree exactly. This proves
those native observations, without asserting complete encoder bit identity.

The source-precision regression fails before repair in both primary Rust and
native decoders. At 16-square and 259x17, f16/f32 RGB/gray alpha changes, including
0.0625 becoming 0.0627451. Both displayed frames and legacy/request entries are
checked. Retained executable SHA256
`7e15ed65d4efb5830ef800a097e15520e5159a2b5c0d24a6e0fd0da67b3e252c`,
PID 2127868, exit 101. `animation-alpha-source-before.log`: rc101, 20 seconds,
peak RSS 1.06 GiB, minimum available 20,755 MiB. The earlier real typed-frame
observation separately found 22,679 of 34,447 native f16 alpha samples changed;
it is recorded in the progressive contract.

The independent visibility regression compares enabled/disabled simplification
with no zero alpha anywhere. Integer alpha 100/65535 and float alpha 1/1024
must remain visible. Both decoders observe different colors before the mask
repair at 16-square and 259x17, through both animation entries. Retained object
SHA256 `a1cf2bb6478c96eba6fbed4375fbcac4ce857f7cf9824a36eecad1a1991ce6c3`,
PID 2132357, exit 101. `animation-alpha-visibility-before-runtime.log`: rc101,
21 seconds, peak RSS 1.01 GiB, minimum available 20,690 MiB. The first compile
attempt of that new test failed a borrow check; it is not runtime evidence.
The first expanded native harness also failed because its newly extracted
non-void wrapper omitted its return; the corrected harness passes. Both failed
logs remain retained.

## Candidate implementation and caller scope

Animation selects one common alpha coding over all supplied frames before its
file header. It prepares each cropped frame at that precision, sharing the
still-image representation choice. Exact binary16 is preferred when all values
permit it; other in-range values use binary32. Non-half out-of-range f32 values
use the existing clamped U16 policy for every frame, consistently with the common
header. Integer layouts retain their original declared precision. No public
signature, new strategy gate or alpha quantization setting is changed.

The simplification mask comes from exact source samples, using the same helper
as still images. Association and explicit simplification-disable guards remain.
The byte-oriented alpha copy still serves approximate color preprocessing.

The two accepted animation entry points share this implementation. Packed
buffers are required by the current animation API; it has no row-streaming
animation entry point or caller-added-extra support. Still/streaming share the
representation selector, so their existing precision and byte controls must also
pass. The policy matrix includes every offered named/Experimental policy and
selected Custom compact-header/smoothing combinations. Layouts, association,
aligned/odd groups, interleaved reference/display frames and progressive modes
are selected interactions, not a complete Cartesian product. A later displayed
frame requiring binary32 must not inherit a half-only first-frame header.

The completed full policy/source/visibility run also repeats the existing
reference/crop and typed-animation checks. `animation-alpha-policy-first.log`
passes (rc0, 284 seconds; peak RSS 1.04 GiB, minimum available 20,702 MiB).
Retained object SHA256
`a3e2bb2139121385bcea779d55d4544c9e09382bcfc5cd3b7b3b9570f2d38852`,
PID 2134046. The original included PNG is verified inside that object and
attached afterward, explicitly recorded in `include-inputs.json`.

The minimal build repeats those checks and strengthens the common-header
fixture: its first supplied reference frame has only half-representable alpha,
while a later displayed frame needs binary32. Retained object
`a72d7c0f8d2c4d1f09cd35985ae4356897a548f5293f90ddab0f6bae3f36eb12`,
PID 2143077, exit zero. All three selected fault replays are detected: original
eight-bit alpha, first-frame-only representation selection, and rounded
visibility. Each mutated source is restored after its failed regression.
The enclosing `animation-alpha-validation.log` subsequently fails because the
evidence helper assumed `tests/it.rs` instead of Cargo's folder `main.rs`
entry; it is not an overall passing run. The corrected helper honors explicit
registered paths and both automatic test layouts. Still-image precision and the full library baseline pass separately in
`animation-alpha-validation-rest.log`; byte locks, region evidence and strict
lint remain under verification. The existing ignores are unchanged.

Final verification passes. `animation-alpha-validation-rest.log` completes
still precision, the full library baseline, unchanged normal/strict/drift
expectations and expanded feature/CLI/workspace strict Clippy (rc0, 423 seconds;
peak RSS 1.68 GiB, minimum available 20,078 MiB). The final opt-in identical
zero-delta control preserves source alpha through synthesized zero payloads;
it checks alpha, without asserting exact lossy color. Its run and focused lint
pass in `animation-alpha-final-evidence.log` (rc0, 25 seconds; peak RSS 1.05 GiB,
minimum available 20,669 MiB). The CLI's APNG animation adapter supplies RGB8 or
RGBA8; it cannot construct floating animation input. No new CLI API is added.

## Selected region evidence and limits

`just animation-alpha-combined-regions` verifies every captured source hash,
exports each retained successful object/profile separately, and requires the
three affected source files to be byte-identical before combining code-region
observations. Animation uses object
`a72d7c0f8d2c4d1f09cd35985ae4356897a548f5293f90ddab0f6bae3f36eb12`;
the supporting established still-alpha tests use
`4a10bdf19269da0c97910b0382f034e040a47488372c00078d7ed1bed5476f74`.

| Selected function/window | Observed LLVM code regions |
| --- | --- |
| AlphaCoding selection and metadata | 41/41 |
| AlphaPlane construction | 25/25 |
| Borrowed float-alpha reader | 34/35 |
| Animation common-header selection | 44/44 |
| Animation exact visibility preparation/application | 36/36 |
| Animation precision payload preparation | 36/36 |

The reader's one missing region is its defensive `return None` for a layout
without floating alpha. Both actual animation callers are dominated by the
same positive floating-alpha guard; that continuation is unreachable from
those callers and was not fault-injected. These are selected code regions,
not proof of every branch outcome, every input or total encoder coverage.
Platform CI remains a separate remote verdict; no new GPU execution or
physical-color qualification is claimed.

Existing alpha settings not forwarded by animation remain a separate
configuration-wiring item; CMYK/Black interpretation remains H10. No existing
byte expectation, tolerance or ignore was changed.
