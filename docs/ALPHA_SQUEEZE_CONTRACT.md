# Squeezed alpha sections

Issue [144](https://github.com/imazen/jxl-encoder/issues/144), V3. This record
covers the existing lossy alpha-squeeze writer, not general JPEG XL conformance.

The progressive pass placement defect (V2) remains separate. Animation currently
does not forward `alpha_distance` or `alpha_squeeze` to its frame encoder; it
therefore does not exercise this writer. The optional subsampled RGB8 writer
also does not use it. Neither route is counted as tested squeeze production.

## Reproductions and correction

At 1025x1024 and 1026x1024, alternating alpha rows 0/210 at alpha distance 2
decode as 56/154 along the right edge, with a 1/209 neighboring halo. Both the
primary Rust decoder and pinned djxl v0.12 show the error. Native v0.12 cjxl
responsive-alpha controls recover every selected source code, as does the
group-aligned negative control. The retained original JXL SHA256s are:

- 1025x1024: `1b8540c536b8fcd2dfa366b70cdf31711d4459b7cd0fe8ad34210a375d53e7ba`
- 1026x1024: `a55cd4f8dbbb625560dddc87632abcccaaf0067dc9de329d1b9f40a717961ca9`

DecodeGroup removes empty cropped channels before assigning local channel
indices. The writer now constructs that same ordered surviving sequence,
pairing each channel with its original multiplier before building its tree.
Empty per-group sub-images write no modular bits. A direct writer regression
finds 56 unused bits before repair; the 257-square image already rendered
before repair, so this is not a claim that that stream was undecodable.
LfGlobal still writes its transform and header when its local subset is empty.

The wider precision matrix exposed another defect: the non-LZ77 entropy header
used a one-context helper for a tree with several leaves. Both decoders reject
retained 257-square and floating-alpha streams. A private context-count-aware
adapter emits the required all-zero context map when there is more than one
context. Its one-context caller retains the original bits. These are shared
syntax corrections under Part 1 G.4.2, H.4.1 and C.2, without a strategy gate.

## Actual caller controls

`alpha_squeeze_contract` isolates edge compaction, an aligned control, and an
empty corner using DCT8 and strict Libjxl mode. `alpha_squeeze_matrix` checks
every alpha sample against independently encoded pinned native controls. It
uses finite signed sample codes without clamping; quantized alpha can extend
beyond the nominal range. Float input uses exactly representable 0, 0.5 and 1,
with the existing 16-bit lossy-alpha conversion represented explicitly.

| Axis | Observed scope |
|---|---|
| Policy | All offered named strategies in Reference/Experimental, plus the selected Custom compact-header/DC-smoothing combinations in `support/encoder_policies.rs` |
| Public entry | Request, padded stride, writer, legacy, streaming in seven-row chunks; corresponding bytes agree within a configuration |
| Engine entry | Existing precomputed and prequantized paths, both modes, single/global/HF/LF sizes; engine default policy, because this surface has no named-strategy setter |
| Layout | All twelve supported alpha layouts: integer, BGRA, grayscale, binary16, binary32, PQ, HLG and BT.709 |
| Geometry | One-pixel axes, 2-square, 128x64, 256/257 boundaries, transposed 1024/1025 edges, and 2049x2048 LF groups |
| Quantization | Alpha distances 0.125, 1, 2, 10 and 25; associated RGBA8 and GrayAlpha16 controls |
| Workers | Actual requested one-worker and two-worker encodes, with recorded statistics |
| Buffering | All five public enum values produce matching bytes on the selected LF cases. Source inspection shows these pixel-path values are currently scaffolding, not five distinct buffering producers |

The full optional-feature matrix passed with an exact instrumented executable
SHA256 `c98d4def891cc93d91e151e1b1ae38e31bd3252231a1e8b3c0f88162be684ee8`,
retained with its source and completed profile. These are selected interactions,
not the full Cartesian product. They do not establish lossy color quality,
arbitrary floating alpha payload fidelity, or progressive pass correctness.

## Reproduction and retained evidence

Use an isolated source export with the committed sibling pins, a separate
Cargo target, and explicit guarded v0.12 tools. Relevant just recipes are
`alpha-squeeze-edge-regression`, `alpha-squeeze-layout-controls`,
`alpha-squeeze-caller-controls`, `alpha-squeeze-extra-callers`,
`alpha-squeeze-mutations`, `alpha-squeeze-broad-baseline`,
`alpha-squeeze-minimal-controls` and `alpha-squeeze-rust1100-lint`.

Local supporting artifacts are under `~/tmp/jxl144/archive-mount/`:
`alpha-squeeze-matrix-before-context-map`, `alpha-squeeze-callers-evidence`,
`alpha-squeeze-extra-callers-verified` and `alpha-squeeze-mutations-2`.
Logs are `~/tmp/jxl144/alpha-squeeze-*.log`. Mutation controls restore the
original noncompacted placeholders, remove the empty-sub-image return, and
remove the multi-context map; each causes its regression to fail.

An earlier coupled-multiplier mutation escaped the 0/210 edge fixture because
it changed both residual division and the declared multiplier consistently,
and 210 is divisible by both candidate multipliers. That retained result is
not a successful sensitivity check. The restored original placeholder fault
is detected.

The full default library, unchanged normal/strict/divergence locks, minimal
alpha matrices and no-default library check pass. The combined mutation and
baseline wrapper completed with rc0 in 722 seconds; run-heavy peak RSS was
1.71 GiB and minimum available memory was 20,346 MiB. Existing alpha integration
controls and Rust 1.100 workspace all-target/focused strict Clippy pass too
(`alpha-squeeze-final-mapping-lint-2.log`, rc0, 40 seconds, peak 1.01 GiB,
minimum available 20,818 MiB). These are check-resource observations, not
encoder performance or memory-budget measurements. No existing expectation,
tolerance or ignore changed. Remote platform CI is a separate verdict.

## Selected code regions and remaining gaps

The coverage union compares exact function bodies in checksum-preserved sources,
normalizing only line offsets before declarations. Each report comes from its
matching retained object/profile; no LLVM mapping diagnostics were emitted.
The full-feature and minimal matrices plus the direct empty-section unit produce:

| Function scope | Executed / mapped code regions |
|---|---|
| Single-group wrapper | 11 / 11 |
| Global wrapper | 7 / 7 |
| LF-group wrapper | 9 / 9 |
| HF-group wrapper | 9 / 9 |
| Shared alpha section | 352 / 382 |
| Context-aware histogram adapter | 33 / 40 |

The shared section's remaining thirty regions comprise twenty fallible serializer
error returns and ten defensive or unreachable regions: two zero-size-channel
continuations after prior filtering, false branches for positive Huffman depths
of actually used symbols, distance-extra-bit writes for the fixed RLE distance
one (its extra-bit count is always zero), and the non-Raw arm in the no-run
path. The histogram adapter's seven missing regions are fallible writer/error
returns. These error returns were not fault-injected. The global empty-data
header is directly checked; normal generated pipelines have a nonempty coarse
global band. The materialization, empty-group return, one/many-leaf trees,
lossless/lossy multipliers, LZ77/non-LZ77 and one/multiple context choices are
observed in these scopes. Fractions do not imply coverage of the entire encoder.

Exact report: `alpha-squeeze-empty-unit-coverage/scope-union.json` under the
artifact root above; reproduce its body comparison with
`alpha-squeeze-scope-summary`. The original full-feature object and a separately
retained minimal object remain available. The report script initially missed
Rust v0 identifiers; correcting that matcher changed reporting only, not the
profiles or production source. V2, animation configuration forwarding and their
own subsequent affected-path checks remain open.
