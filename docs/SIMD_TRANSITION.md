# SIMD API transition and remaining work

Observed 2026-11-05 against main `93f4991d` and the review below. This is a
source/work inventory, not release qualification.

[PR 141](https://github.com/imazen/jxl-encoder/pull/141) adds the array/slice
`target_features` API and deprecates retained token-taking calls. It is open
and assigned for API review. These additions are not a published release;
check that a consumer's resolved source contains them before migrating it.
No token removal, merge or publication is authorized by this transition.

## Calling the API

Ordinary code keeps using runtime-dispatched functions such as
`jxl_encoder_simd::dct_8x8`. An operation already compiled for a CPU tier can
call `target_features::dct_8x8_v3` with arrays and no token parameter:

```rust
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe")]
fn transform(input: &[f32; 64], output: &mut [f32; 64]) {
    jxl_encoder_simd::target_features::dct_8x8_v3(input, output);
}
```

On native targets, a runtime CPU check alone is not compiler proof. E0133
means move the call into a matching feature function, or use dispatch; do not
add an unsafe block. WASM permits safe baseline calls because the engine
validates module instructions on load. SIMD128 still needs engine support.
Scalar functions need no SIMD context.

The module documents complete feature sets. V4 also requires `aes` and
`pclmulqdq`, beyond V3 and the five AVX-512 features. Proof tokens and vector
representations stay inside new consumer operation boundaries. Internal
kernels may continue using them. Existing token signatures remain compatible
and deprecated; the hidden cube-root benchmark helpers retain token-free
runtime-dispatched alternatives.

Array boundaries do not prove zero copies, inlining or a speedup. Preserve
length, layout, stride and rounding contracts when composing operations.

## Validation and dependency boundary

All 110/110 facade entry points were compared bitwise with their retained
same-tier kernels: 49 V3, 6 V4, 30 NEON and 25 WASM128, each with three
input-size cases. Validation explicitly requires the named hardware tier.
Native call rejection, safe composition, and passing WASM baseline calls are
covered. This does not assert cross-tier identity for the encoder's existing
[registered divergences](SIMD_PARITY_KNOWN_DIVERGENCES.md).

Strict lint, native default/no-default tests, ARM/WASM typechecks, i686
no-default library checks, docs and complete x86/ARM snapshots were checked.
The historical NEON benchmark intentionally compares retained token entry
points with its existing scalar arms; only that function allows deprecation
warnings. Its arithmetic and arms are unchanged.

The full workspace's pinned dependency graph resolves archmage/magetypes
0.9.30. Forcing beta.2 is rejected by pinned Butteraugli's stable
`magetypes ^0.9.14` requirement. Isolated SIMD V4 validation used published
0.9.31-beta.2, which supplies narrow-vector traits absent from 0.9.30. Updating
the complete graph is separate dependency work; do not replace the root lock
with a partial three-package edit. The current root lock was regenerated and
verified with the exact [.github/sibling-revisions.tsv](../.github/sibling-revisions.tsv)
source closure. CI status must be checked on the current PR head.

Reproduce primitive checks with `just simd-api-stage <new-directory>` followed
by `just simd-api-check <stage>/Cargo.toml`. `just simd-api-doc-check` verifies
this architecture's inventory; snapshots live under
`docs/public-api/{x86_64,aarch64}/`. `just stage-ci-workspace <new-directory>
<commit>` exports the pinned CI closure without changing sibling checkouts.
Run builds and source staging through the project's resource wrapper.

## Branch and issue inventory

The live inventory contains 41 remote branches, including main and PR 141,
and two local branch heads. Seven remote tips are ancestors of main; 34 are
not. Ancestry does not establish patch equivalence after squashing or rebasing.
Full branch names, SHAs, ahead/behind counts, linked PR states and local refs
are retained privately by `just review-inventory <private-directory>`.
Nothing was deleted or closed.

Every remote tip is accounted for by these mutually exclusive dispositions:

| Disposition | Branches | Action |
|---|---:|---|
| Ancestor of main, including main | 7 | Already reachable; retain refs. |
| Open PR | 3 | Review 140, 132 and 141 independently. |
| Rejected or abandoned | 10 | Preserve evidence; no approval to replay. |
| Superseded | 6 | Compare residual changes before any future adoption. |
| Preserved experiment/research/release | 15 | Requalify against current contracts; no merge decision implied. |

The two local heads are retained and included in the private evidence, along
with all local remote-tracking refs and jj bookmarks.

The other open PRs remain separate:

- [140](https://github.com/imazen/jxl-encoder/pull/140): padded-plane strides
  in perceptual loops; retain its stride/pixel qualification.
- [132](https://github.com/imazen/jxl-encoder/pull/132): good-block exponent
  and overshoot work; retain measured quality/size gates before adoption.

Preserved experiment/research/rejection branches are evidence, not an approved
merge queue. The issue-138 nearest-PQ head is already in main; other CfL,
gradient, PQ and gamut experiments are not thereby approved. The crate-split,
coefficient-gap, high-bit-depth preservation and older release branch require
individual contract/evidence review. Do not replay their whole patches from
a branch name or divergent tip.

All 12 open issues were inspected:

| Issue | Concrete remaining work |
|---|---|
| [136](https://github.com/imazen/jxl-encoder/issues/136) | Keep the current correctness/PR/evidence tracker synchronized with reviewed changes. |
| [114](https://github.com/imazen/jxl-encoder/issues/114) | Resolve delivered-quality inversions with the existing thresholds and retained streams. |
| [113](https://github.com/imazen/jxl-encoder/issues/113) | Qualify RD/time candidates with complete grids and paired measurements. |
| [112](https://github.com/imazen/jxl-encoder/issues/112) | Separate implementation agreement from specification and sample-fidelity qualification. |
| [111](https://github.com/imazen/jxl-encoder/issues/111) | Validate the identified inference-dependent behavior against the official standard text. |
| [110](https://github.com/imazen/jxl-encoder/issues/110) | Qualify remaining chain-length/keep-best tradeoffs beyond the landed production selector. |
| [108](https://github.com/imazen/jxl-encoder/issues/108) | Identify and validate the actual application workload and deployment limits. |
| [105](https://github.com/imazen/jxl-encoder/issues/105) | Evaluate existing smart policies before introducing another policy or trained selector. |
| [103](https://github.com/imazen/jxl-encoder/issues/103) | Establish current matched-quality screenshot benefit and optional resampling cost on held-out inputs. |
| [99](https://github.com/imazen/jxl-encoder/issues/99) | Meet the remaining sectioned-lossless wall target; do not reopen already-landed admission repairs. |
| [95](https://github.com/imazen/jxl-encoder/issues/95) | Finish exact 17–31-bit integer/input coverage rather than inferring it from narrower tests. |
| [24](https://github.com/imazen/jxl-encoder/issues/24) | Complete lossless picker/oracle qualification through the existing training pipeline. |

## Source markers and guidance

The initial tracked-text scan found 28 TODO/FIXME matches, including historical
records and quoted upstream notes. These are not 28 missing product features.
The private capture retains every match and location. Current follow-ons are:

- `vardct/zensim_loop.rs`: strategy refinement remains disabled pending RD
  evidence; do not re-enable it by changing thresholds in this API migration.
- `vardct/dc_tree_learn.rs`: `bits_saved` still uses a zero diagnostic placeholder.
- `examples/debug/parse_jxl_header.rs`: floating-point exponent parsing remains
  an example/parser follow-on.
- Registered WASM cross-tier arithmetic differences remain separate work. No
  existing ignore, tolerance or quality gate was relaxed here.
- Dot detection's old staged-port checklist was stale: connected components,
  fast ellipse fitting, loss filtering, patch conversion and gated integration
  are present. Its module documentation now names the existing implementation.

[NEON_PORTING_GUIDE.md](NEON_PORTING_GUIDE.md) preserves historical 0.5 kernel
notes; use this document and the checked current module for consumer API work.
Older release/API audits describe their dated revisions, not current release
permission or current publication status.
