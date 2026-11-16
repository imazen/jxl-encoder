# Lossless squeeze and RCT admission

[Issue 144](https://github.com/imazen/jxl-encoder/issues/144), E5, concerns
packed binary32 input whose signed integer codes pass the source squeeze
range check, then widen under YCoCg before squeeze. Checked arithmetic panics;
the retained unchecked build emits four wrong samples out of 48, with both
decoders agreeing. The first differing bits are `f8000000` versus `faaaaaab`.
This is observed decoder output, not a numerical model.

The existing transform budget, `RCT_BUDGET_LIMIT = 30`, already excludes
binary32 RCT in the ordinary modular writer and simple multi-group squeeze.
The simple single-group, learned single-group and learned multi-group squeeze
writers omitted that guard. They now share the existing permission. Integer
and binary16 samples retain their prior choice. Source-admitted binary32
samples can still use squeeze without an RCT; this is not full byte parity
with the reference's more conservative responsive-transform policy.

The pinned v0.12 responsive production control recovers the exact finite-float
trigger bits. Its bit-budget policy can suppress responsive transforms on
binary32 input. Part 1 H.6.2 and H.6.3 govern the operations.

The route matrix checks all offered modes and named strategies, with four
selected Custom controls. RGB/RGBA fail before repair in all three missing-guard
writer families through packed/strided requests, the legacy shortcut and
seven-row streaming. The existing guarded multi-group writer and grayscale
controls pass. Both axes and the single/multi-group boundary are exercised.
Every successful stream must recover every color/alpha bit in jxl-rs and pinned
djxl, and the public entry routes must produce identical bytes for each case.

Animation accepts integer layouts, not these binary32 layouts. Binary16 and
unsigned integer source codes cannot create the signed binary32 counterexample.
Additional channels lie outside the RGB RCT prefix. These scope statements do
not establish the absence of unrelated arithmetic bugs in those paths.

The before route journal records 1,088 failures and 2,176 passes. Those are
execution records, not evidence of completeness by count. The selected cases
and their source/stream hashes remain available with the decoder diagnostics.
The exact main baseline plus only the three missing guards passes the checked
route matrix, the full library suite, and unchanged normal/strict byte locks
and divergence checks. The same selected source also passes the complete
binary32 route matrix with package debug assertions and overflow checks off;
a separate executing wraparound marker verifies that build condition. This
is an optimized package-only control, not a full Cargo release build.

Retained runs: `squeeze-rct-isolated`, `squeeze-rct-isolated-resume` and
`squeeze-rct-isolated-unchecked`. Their source identities are recorded before
each build. The library/lock wrapper completes in 438 seconds (peak RSS
1.69 GiB, minimum available 20,907 MiB); the unchecked replay in 141 seconds
(peak RSS 1.70 GiB, minimum available 20,675 MiB), as reported by run-heavy.
The initial library compile stopped on disk exhaustion; its earlier completed
route/native checks remain evidence, but that encompassing run is not green.

The larger-channel and elongated-group repairs remain separate work.

The planar API also reaches these guards. Boundary controls declare 29, 30
and 31 bits on RGB/RGBA/gray/gray-alpha, with single/multi-group shapes and
simple Huffman, simple ANS and learned ANS. Both decoders recover the chosen
exactly representable integer codes. This does not claim arbitrary 31-bit
recovery through their float output APIs. The boundary replay passes in 99
seconds (run-heavy peak RSS 0.90 GiB, minimum available 21,379 MiB).

Selected LLVM evidence uses the exact planar test object and its own completed
profile: all 15 mapped regions of the three guard expressions/conditions run.
Each guard's RCT body and refusal path run. Gray inputs short-circuit the color
prefix; the `< 30` condition receives both outcomes. A non-gray public image
with fewer than three color channels is refused before these writers. The
report excludes allocation and downstream serializer error returns: those are
not newly introduced guards and were not fault-injected by this change.
`squeeze-rct-planar-coverage` retains JSON, annotated frozen sources, object
and profile identities. A broader merged report has mapping mismatches and is
not used for this selected claim; the single-object report has none.

Strict library/regression Clippy passes against the same isolated source.
The local CI target selection passes with JPEG and parallel tree learning
(886 seconds, run-heavy peak RSS 1.11 GiB; minimum-available sampling was
unavailable). It runs the standalone header, ICC, cancellation, shifted-channel,
native squeeze and binary32 targets. Existing integer tests retain their
expected bytes. The new CI step selects these previously omitted targets and
retains failed streams. Actual remote CI results remain a separate verdict.
