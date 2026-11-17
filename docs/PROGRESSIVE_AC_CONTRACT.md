# Progressive AC coefficient controls

Issue 144 V1 remains under investigation. The native component control links
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
selected positions. They do not demonstrate this encoder's runtime failure
or a real-image realization.

The encoder's existing symmetric nearest precision split differs from native
truncation toward zero. Final coefficient preservation is the intended V1
property; this investigation does not authorize changing that rounding policy.
The three-pass preset also differs from native CLI frequency splitting, as
recorded in [the alpha contract](ALPHA_SQUEEZE_CONTRACT.md).

Reproduce with `python3 scripts/frame_header_oracle.py --component progressive_ac
<V012_SOURCE> <V012_BUILD> <NEW_EXECUTABLE>`. The builder verifies the reference
version and refuses an existing executable. The initial native build log is
`~/tmp/jxl144/native-progressive-ac-elongated.log`; source and reference build
identities remain in the issue's pinned oracle closure. Direct production
token tests, real-image controls, affected callers and selected-region
coverage remain pending.
