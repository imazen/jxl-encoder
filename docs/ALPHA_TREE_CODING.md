# Lossy alpha tree integer coding

Issue [144](https://github.com/imazen/jxl-encoder/issues/144) caller testing
exposed a shared tree writer failure independently of animation forwarding.
A 32x17 RGBA16 still with alpha distance 10 and squeeze enabled produces a
40527-symbol Huffman alphabet. The writer declares raw 15-bit integer coding;
both the primary Rust decoder and pinned native v0.12 reject that stream.
The independently encoded native source reference renders successfully.

The local gradient and channel-split tree writers now select the existing
HybridUint configuration `(4,2,0)` when their largest raw token is at least
32767. Below that boundary they retain `(15,0,0)` raw coding. Histogram
construction and token emission use the same selection, including the extra
bits. This also removes the premature u16 narrowing of large multiplier
tokens. The multiplier, channel decisions, predictor and quantized alpha data
remain unchanged. This representation is permitted by Part 1 C.2/C.3; it is
not a claim of complete libjxl entropy-byte parity.

The general MA tree writer already uses `(4,2,0)`. The repair reuses its
existing `HybridUintConfig` implementation for the specialized lossy writers.
No new public API, strategy gate, preset value or alpha quantizer is added.

The original still failure is retained in `alpha-large-tree-q71-runtime-v1`,
ELF `ff4bddd6738343fc722b23e139c6579e49ecc1b31d5cea9589863f95f1e00b8a`,
profile `4ad02e2d11e22877e12dd564b4b5c44a561d32ae1cf6d24d6f7fe3bdf2bff424`,
with an actual runtime failure in both decoders. The earlier animation
forwarding failure and its three-line correction are separate findings.

The independent default matrix in `alpha-large-tree-q73-runtime-v2/matrix`
checks every alpha code against a native encode. Its selected interactions
include all seventeen audit policies, integer and converted-float alpha,
three progressive modes, one-pixel axes, transposed/edge/LF-group shapes,
request/stride/writer/legacy/streaming entries and ANS/Huffman controls.
The literal largest selected shape is 2049x2048. The 4/8/8.1/10/25 alpha
distances exercise multiple multiplier decompositions; these are selected
values, not an exhaustive distance grid.

That matrix's completed ELF/profile are
`06022ada527450594b015b3e26895489a4edb45a7492ff2084af7e6611d238fb` /
`49c34614fdffd883dd125de9b4255303cb11f3f8aeee470d80ba7c498f6f00f3`.
The enclosing v2 run later fails a missing import in the new controller test
helper; its matrix runtime remains a separate zero exit. The corrected
continuation is retained as v3, without relabeling v2 green.

The request controller separately checks exact native alpha at caps zero/two
for RGBA16, GrayAlpha16 and linear-f32 RGBA under the same seventeen policies,
including packed/strided byte equality. A cap of two does not assert that two
adjustments actually occurred. The existing controller termination policy is
unchanged.

Public raw/precomputed/prequantized engine methods accept `ExtraChannel`.
Its public alpha constructor supplies u8 data; its u16/f16/f32 alpha constructors
are crate-private, and the descriptor getter is immutable. Those public raw
constructors therefore do not expose the high-precision alpha trigger. The
request controller does reach private prepared entries with layout-derived
high-precision alpha, covered above. Non-alpha extras retain multiplier one.
No new constructor is introduced to manufacture a public caller.

The first new matrix fixture expanded progressive modes twice. Its retained
run fails the existing output-preservation guard on duplicate cell names;
completed unique cells still retain their own pixel verdicts. The corrected
fixture lets the existing runner expand modes once and checks expanded name
uniqueness before encoding. No pixel assertion or quality bound is weakened.

Full source identities, region reports and remaining serializer/error paths
retain their separate qualification records. None of the matrix names, file
counts or runtime totals proves complete encoder or input-space coverage.

The corrected q73-v3 continuation passes default/minimal controller and
high-precision pixel checks, the full selected-feature library, unchanged
normal/strict locks and divergence checks, strict default/minimal affected
lint and scoped formatting. Earlier v1/v2 failures retain their own verdicts.

`alpha-tree-q74-regions-v1` binds exact production bodies from four passing
objects/profiles, without LLVM mapping diagnostics:

| Function | Executed / mapped regions |
|---|---|
| Representation selection | 5 / 5 |
| Channel maximum | 19 / 19 |
| Shared symbol writer | 31 / 34 |
| Single-leaf histogram | 51 / 61 |
| Single-leaf tokens | 19 / 20 |
| Channel-split histogram | 72 / 82 |
| Channel-split tokens | 72 / 80 |

Missing regions include fallible serialization/Huffman error returns, which
were not injected. Both histogram single-symbol fallback arms are excluded
by their fixed zero and Gradient tokens; symbol depth zero likewise has no
valid caller here. The channel split's negative threshold arm is excluded
by its zero-based increasing channel decisions. Raw/hybrid and extra-bit
choices are observed. These fractions do not prove a full input-space or
whole-encoder qualification. The later q76 restored-fault controls detect both a forced raw representation
and zeroed HybridUint extra-bit payloads through the independent still check.
A separate single-leaf-only raw fault is detected by the high-precision matrix,
including observed 32769/32770-entry invalid alphabets. Each mutated source
is restored byte-for-byte; every control has an actual runtime exit 101,
retained fixed/fault source hashes, executable and profile. Their exact records
are in `animation-alpha-alpha-coding-q76-faults-v1/results.json`.
