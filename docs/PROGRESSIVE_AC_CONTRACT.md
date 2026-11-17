# Progressive AC coefficient controls

Issue 144 V1 has a locally verified rectangular-LLF repair. The wider caller
and feature audit remains open; the remaining gaps are listed below. The native component control links
unmodified libjxl v0.12.0 `a7a9c78` objects and invokes actual
`AcStrategy::ComputeNaturalCoeffOrder` and
`ProgressiveSplitter::SplitACCoefficients` for the nineteen transform codes
this encoder offers. It records each normalized transform shape and natural
order LLF prefix, plus sparse coefficient positions, their observed ranks and
native two-pass values at shifts 1/0. The wrapper contains no replacement
coefficient-order or split algorithm.

For native DCT16x16 (code 4), the LLF prefix is positions 0, 1, 16 and 17.
Positions 2 and 3 have natural ranks 5 and 6, so they are AC. Native input
+3 at position 2 produces coarse +1 and final +1; input -3 produces -1/-1.
The stored control is
[jxl-encoder/src/vardct/progressive_ac_v012.tsv](../jxl-encoder/src/vardct/progressive_ac_v012.tsv).
These observations establish the native order and splitter behavior at the
selected positions. Separate production-token and real-image controls below
demonstrate the encoder failure.

The encoder's existing symmetric nearest precision split differs from native
truncation toward zero. Final coefficient preservation is the intended V1
property; this investigation does not authorize changing that rounding policy.
The three-pass preset also differs from native CLI frequency splitting, as
recorded in [the alpha contract](ALPHA_SQUEEZE_CONTRACT.md).

Reproduce with `python3 scripts/frame_header_oracle.py --component progressive_ac
<V012_SOURCE> <V012_BUILD> <NEW_EXECUTABLE>`. The builder verifies the reference
version and refuses an existing executable. The initial native build log is
`~/tmp/jxl144/native-progressive-ac-elongated.log`; source and reference build
identities remain in the issue's pinned oracle closure. The native control is independent of the encoder token regression and
final-pixel comparisons below.


## Caller inventory before repair

`bitstream.rs::tokenize_ac_group` has one production caller in the two-pass
writer's per-DC-group AC traversal. It receives the actual progressive pass
configuration and optional per-bucket/channel coefficient order. Both ordinary
pixel encoding and existing precomputed/prequantized entries reach that writer.
Request/legacy/writer and retained-row streaming share the pixel path; lossy
animation forwards both progressive mode and forced transform. The CLI forwards
its two progressive spellings into these paths. The static entropy producer is
separate and does not call this tokenizer; low-effort dispatch must be observed
rather than inferred from the option spelling.

## Observed before and after

The original production tokenizer drops native DCT16x16 position 2 value +2
(recovered zero). Its flat-prefix count also counts nonzero LLF positions as
AC. The retained sparse test fails before repair, covering the nineteen native
transform codes with natural/reversed-valid AC orders, all three color channels,
strict/normal context maps, first/offset group origins and all three pass modes.
These are selected combinations, not their full Cartesian product.

Actual frymire inputs at 64x64 and 259x133 expose changed final pixels for
internal raw types 3,4,10,11,16,17,18. The other twelve offered raw types retain
single-pass final pixels on these inputs. For example, 64-square raw 3 two-pass
coding differs in 6,666 primary-decoder samples before repair. Both decoders
fully render the faulty files: syntax acceptance does not detect the omission.

The repair shares the original transform quantizer's rectangular nonzero count
with progressive tokenization. The coefficient split, rounding, presets,
context maps and public API retain their existing policies. All nineteen
transforms now preserve each decoder's own single-pass final f32 bits on both
real-image sizes. All 38 measured single-pass source/transform streams retain
their exact before hashes. This is selected identity evidence, not full encoder
byte parity with native libjxl.

Additional passing controls exercise:

- Every policy in `tests/support/encoder_policies.rs`, including Experimental
  mode and selected Custom compact-header/DC-smoothing combinations, through
  legacy/request/writer/strided/streaming entries at both real-image sizes.
- All 24 non-CMYK pixel layouts, including binary16, HDR float and alpha,
  comparing final color and extra-plane bits in both decoders.
- Legacy and request animation entries, two distinct displayed frames, RGB8,
  RGBA16 and GrayAlpha8, straight and genuinely associated integer samples,
  both encoder modes and both sizes. Every displayed frame is checked.
- The existing precomputed and prequantized methods, with and without extras,
  both modes and sizes. Their prepared data and quantizer field are held fixed
  across passes. These APIs expose the actual default engine policy; they do
  not expose a named-strategy setter.
- All six CLI strategy spellings, both encoder modes, bulk and row-streaming
  input/output, both sizes and both progressive flags. Each decoder's final
  samples equal its own single-pass CLI control.

Before archives are `progressive-ac-unit-before` (ELF SHA256
`204e131f29bbece5bc28e13940774bfaf082ac37f19c3c379b3778f9e4cf3b67`)
and `progressive-ac-real-before` (ELF SHA256
`8f608807a24b244415a99218e6f911e931b4a6b8105b0e82a6d7c4ed88e975ea`).
Both recorded test exits are 101. Actual streams, PFMs and the completed
sample journal remain under `live-tests/progressive-ac-contract/run-0` at
this issue's artifact root. The original included PNG was verified in the
retained ELF and attached after the run; `include-inputs.json` records that
attachment timing rather than claiming a contemporaneous raw-buffer capture.

Passing objects, frozen sources and profiles are retained in
`progressive-ac-unit-first-fix`, `progressive-ac-real-first-fix`,
`progressive-ac-callers-first-fix`, `progressive-ac-animation-corrected`,
`progressive-ac-engines-typed` and `progressive-ac-cli-first`. The first
animation attempt failed in the new test's PFM extra-plane filename reader;
three engine attempts failed during new-test compilation. Those failed
attempts remain distinct from the corrected passing runs.

`progressive-ac-callers-baseline.log` passes the library baseline, unchanged
normal/strict/divergence locks and focused strict Rust 1.100 lint against the
pinned sibling graph. The expanded animation/engine/CLI tests pass strict Rust
1.100 library/test
lint in `progressive-ac-feature-lint.log` (rc0, 96 seconds; peak RSS
0.89 GiB, minimum available 20,865 MiB). Its minimal std/convenience/expert
control runs the real-image, policy/public-entry, typed-layout, animation and
producer-option comparisons with parallel and loop support disabled. The first
minimal attempt failed to compile the new explicit-loop fixture without the
loop feature; that fixture now carries the same feature condition as its setter.

## Open verification

The original flat-prefix mutation is detected by the sparse production test;
its exact mutated object, profile, source and nonzero verdict remain in
`progressive-ac-count-fault`. The mutable export is restored afterward.

`progressive-ac-regions.json` compares exact function bodies across four
passing retained objects before unioning their LLVM code regions. The shared
rectangular counter covers 35/35 regions, the existing quantizer delegate
20/20 and the coefficient splitter 39/39. The tokenizer covers 269/281;
the twelve remaining regions are its optional `JXL_TOKEN_FILL_STATS` stderr
diagnostic loop. A separate enabled-diagnostic run passes and awaits its
combined export. These numbers describe only the selected functions, not
encoder completeness or all possible inputs.

Producer-option controls pass selected e1/e2/e7/e8, explicit loop, LF-frame,
2x resampling, two-worker, buffering and canonicalization cases. Independently
parsed HfGlobal fields observe actual strict custom orders `0x0005` on the
selected odd multi-group cells; normal policy retains `0x0000`. This establishes
both order branches rather than inferring a winner from effort alone.

The header observation also finds a separate dispatch defect: selected Zen
e1/e2 RGB8 files advertise one pass for both requested progressive modes,
while strict controls advertise the requested two/three passes. Their final
pixels match, so the pixel comparisons alone do not assert that the requested
pass schedule reached the file. The separate pass-count regression fails
before the dispatch repair, with
actual PID 2032323 and retained object SHA256
`abbbda5cced07ff432062df51b5541fd552353a4c7c4f3c5943c73b80878e876`.
The predicate now requires `ProgressiveMode::Single` for static coding.
Afterward all selected policies, both modes, efforts 1/2, 64-square/259x133
and five public entries carry the requested counts and preserve each decoder's
own Single final bits. All 340 measured Single hashes remain unchanged
(`dispatch-single-identity.json`). Library/unchanged locks and focused/workspace
strict lint pass in `progressive-dispatch-baseline-lint.log` (rc0, 510 seconds;
peak RSS 1.70 GiB, minimum available 19,755 MiB).

CMYK8/16 with the caller-selected 5,340-byte `ps_cmyk.icc` profile pass all
seventeen selected policies, both sizes and all three modes. Both decoders'
final color and Black bits equal their own Single output. Profile SHA256 is
`020236a6458406743848a12c6be280e54bcd0fd8e2d4bea9c4899410cef98cb2`;
retained binary SHA256 is
`f3239f0bc893b76f5307d17f298027c6624a5141d1a7a7844c4d1df49c8b0c18`.
`progressive-ac-cmyk-first.log` passes (24 seconds, peak RSS 1.01 GiB,
minimum available 20,732 MiB). Physical CMYK interpretation remains H10.
Already-downsampled output,
winning patches and truly sectioned multi-DC-group buffering remain additional
caller checks. V1 remains open while these paths are resolved.


## Reference-only frame schedule interaction

A real-image animation sequence with reference-only frames before and between
two displayed frames, followed by an actual crop, fails both decoders when
progressive coding is requested. Single controls render. Primary Rust reports
`NonZeroPadding`; native failures are retained with each stream.
The before object is SHA256
`cb8c7328ddd3a583eec299e7782de4e8f7626afe14630a7a65ea31a45d3de29e`,
actual PID 2069907, recorded test exit 101.

The unmodified native FrameHeader writer produces identical ReferenceOnly
header bytes for requested pass counts 1/2/3, with and without alpha. Regular
header controls differ. Part 1 F.2 implies one pass for ReferenceOnly.
[Native observations](../jxl-encoder/src/headers/reference_passes_v012.tsv)
come from `scripts/reference_passes_oracle.cc`, not a grammar transcription.
Reproduce with `just reference-progressive-native-oracle <v012-source>
<v012-build> <new-executable>`. The successful component log and executable
identities are retained in `native-reference-and-color-controls`.

The current local candidate selects Single before each reference-only frame's
helpers and restores the request for each regular frame. This also makes
modular stream numbering use the effective count. Its full selected policy,
layout, size and animation-entry matrix passes (first-fix log rc0, 99 seconds;
peak RSS 1.06 GiB, minimum available 20,679 MiB). Separate assertions that
regular frames restore two/three passes pass in `progressive-ac-reference-regular-fields`.
Minimal, mutation and broad gates remain in progress; this section does not
claim the candidate has landed or close the wider V1 audit.
