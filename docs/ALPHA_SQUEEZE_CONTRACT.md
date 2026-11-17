# Squeezed alpha sections

Issue [144](https://github.com/imazen/jxl-encoder/issues/144), V3. This record
covers the existing lossy alpha-squeeze writer, not general JPEG XL conformance.

The original local-section repairs are recorded at `dc90b122`. The thin-image
global-prefix follow-up and progressive routing are described below; their
final verification is still in progress.

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

## Progressive routing and thin-image follow-up

With DCT8 isolating V1's separate color-coefficient issue, the original
1024-square/1025x1024 two-pass streams and the odd-edge three-pass stream fail
full Rust and djxl rendering. Single-pass and single-group controls preserve
every selected alpha code. The candidate routes HF bands by their actual
header bracket, retaining original channel order before empty-crop compaction.
Raw unsqueezed extras retain their existing last-pass predicate and writer.

The observed header controls distinguish the existing three-pass precision
policy from native v0.12 CLI frequency splitting:

| Mode | Encoder pass shifts / downsampling hints | Native CLI control |
|---|---|---|
| Single | No extra passes or hints | Same |
| QuantizedAcFullAc | 1/0; 2x hint after pass 0 | Same with `--qprogressive_ac` |
| DcVlfLfAc | 2/0/0; 8x/4x hints | `--progressive_ac`: 0/0/0, frequency subsets, 4x/2x hints |

`check_progressive_alpha_headers.py` observes these actual fields and requires
complete section-size accounting in each retained stream. The unmodified native
bracket component separately observes the brackets for this encoder's supplied
header configurations. Neither control asserts full preset or per-pass byte
parity. The existing nearest precision rounding also differs from native
v0.12's truncation toward zero; this repair does not change that policy.

The additional thin controls independently exposed a global-partition defect
even in Single mode. Native FwdSqueeze produces a 257x4 band followed by 256x9,
and the transposed pair, at 1025x17/17x1025. Global coding ends at the first
oversized band. The original partition instead selected indices 0..6 plus 8;
the ordered global prefix is only 0..6. The candidate closes the prefix
permanently at the first oversized channel. The committed metadata fixture's
dimensions/shifts are measured from native FwdSqueeze; its prefix marker is
wrapper-derived from the global-prefix contract, not native decoder output.

The corrected combined source passes the full caller/layout/policy matrix in
all three progressive modes, the thin effort/LF controls, independent actual
gray canonicalization, entropy permission selections and ordinary lossless
alpha. Both full and minimal feature controls recover the selected native
alpha codes. Source-native alpha controls remain separate from lossy color
quality; these checks force DCT8 and do not settle V1.

Public request/strided/writer/legacy/streaming and the two engine entries share
the section producer described above. The CLI has no alpha-squeeze switch and
cannot select this producer. JPEG coefficient encoding has no layout alpha.
Animation forwards progressive mode but still omits alpha-distance/squeeze,
so its configuration repair remains a separate pending caller task. Float
layouts with positive alpha distance are converted to integer alpha before
squeeze; their private binary32-alpha constructor is used for lossless alpha
and does not select this pipeline.

The restored-source library and unchanged normal/strict/drift baseline pass
after the prefix correction, together with the selected minimal matrices and
no-default check (`alpha-squeeze-prefix-final-validation.log`, rc0, 1976 s;
run-heavy peak 1.71 GiB, minimum available 20,251 MiB). Strict Rust 1.100 lint
also passes the corrected prefix source and effort/LF controls
(`thin-alpha-prefix-fixed-controls.log`, rc0, 129 s, peak 1.70 GiB, minimum
available 20,215 MiB). Four V3 fault replays and both V2 pass/boundary faults
are detected. The subsequent declared-precision controls pass for every integer
precision 1–16 in RGBA16 and GrayAlpha16, with request/strided/writer/streaming
entries at 128x64 and 1025x17 in all three modes, full and minimal features.
A separate native lossless-alpha encode first calibrates each source precision
before the lossy-alpha reference is compared; PAM reads honor MAXVAL and its
one/two-byte sample width. No legacy precision override or arbitrary binary32
payload recovery is inferred. Strict Rust 1.100 workspace/focused lint passes
with these controls (`progressive-alpha-precision-final.log`, rc0, 140 s,
peak 1.03 GiB, minimum available 19,815 MiB). Remote CI remains separate.

The final exact-function union combines the retained full/minimal matrix,
full/minimal precision and original core-control objects/profiles. It observes
23/23 regions in `modular_shift_range`, 52/55 in `encode_ac_group_section`,
120/123 in `encode_dc_group`, and 9/9 in the HF alpha wrapper. The AC producer's
three missing regions propagate token/alpha/raw-extra writer errors. The DC
producer's three missing regions propagate LF writer failure, cancellation
inside its group worker, and the worker result error. Those failures were not
injected in this matrix; these fractions do not establish whole-pipeline
cancellation coverage. The separate prefix union adds its direct regression
and observes 36/37 partition regions. Its missing `continue` handles a
zero-sized sub-channel; the one-alpha default producer starts nonempty and
only squeezes an axis while it has more than eight samples, so its generated
bands are nonempty. No fault injection of a synthetic empty band is claimed.

Reports are `progressive-alpha-prefix-final-matrix-coverage-final/`
`progressive-scope.json` and `partition-scope.json` under the artifact root.
The extraction/union command completes without mapping diagnostics
(`progressive-alpha-final-coverage.log`, rc0, 5 s; peak 0.28 GiB,
minimum available 21,431 MiB). The tools compare function bodies exactly
and normalize only offsets before declarations. Coverage does not qualify
V1, animation forwarding, early-preview color quality or every option product.
