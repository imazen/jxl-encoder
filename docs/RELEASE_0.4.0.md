# jxl-encoder 0.4.0 — release prep (NOT published)

Status 2026-08-29: version bumped to 0.4.0 on `main`, public surface
narrowed per #76, semver + consumer verification done, publish gates
below **not** run. Nothing is tagged and nothing is on crates.io — the
owner publishes. Supersedes the per-item claims in
`RELEASE_DEP_AUDIT.md` (2026-06) where they conflict; the numbers here
are from executed probes on 2026-08-29 (see "Evidence").

## September 24 pre-publication audit

**The tested source passes the local correctness gates below. Publication is
still blocked by registry resolution and API inventory review.** Nothing was
published or tagged. Production source is `cfd6ef0a`; audit tests and retained
performance evidence landed through `7a9a5a8c` on `libjxl-exact`. Builds use the
CI-pinned sibling export, `--locked`, four workers and `nice -n 19` on this
ARM64 Mac. A source export is not evidence that registry consumers can build.

### Correctness and settings coverage

- **4,320 selected RGBA cells pass full Rust and djxl v0.12 decoding**:
  2,160 with default features plus parallel, and the same matrix with
  `--no-default-features --features std,__expert,corpus-tests`. Each run
  includes 1,080 lossless and 1,080 lossy cells. Nine dimensions include
  one-pixel axes, 255/256/257 boundaries, 259x133 and 2049x9. Eight
  pathological patterns cover transparent/opaque extremes, saturated
  checkerboards, isolated impulses, ramps, noise, periodic stripes and
  alpha spikes; two sources supply natural photograph and real graphics.
  Twelve configurations per path exercise effort boundaries, Zen/Libjxl,
  ANS/Huffman, sectioned/squeeze, forced WP 0–4 and progressive modes.
  Lossless pixels and lossy alpha are exact; packed/strided encodes agree,
  as do lossless streaming encodes. All 1,080 lossless cells are also
  byte-identical across the two builds. This is a selected interaction
  matrix, not exhaustive configuration coverage. Auto-resampling is off
  in this matrix to preserve its exact-alpha contract.
- **40 float-extreme cells pass bit-exact roundtrips in both decoders**:
  f16/f32, 31x17 and 259x17, Zen/Libjxl, forced WP 0–4 at e7. Signed zero,
  subnormals and finite extrema are included; NaN/infinity are excluded.
- Workspace all-target tests pass: 1,638 encoder unit tests, 508 default
  integration tests, 194 SIMD tests, CLI/macros/tuning-runner targets and
  examples. Existing ignored tests remain ignored. The separately enabled
  expert/internal/parallel/JPEG/HDR integration suite passes 580 tests,
  including 63 normal locks, five strict byte locks and seven drift checks.
- Workspace doctests and all-target Clippy with `-D warnings` pass. Nine
  selected compile configurations pass, covering no_std, std, parallel,
  expert, JPEG, HDR gain-map, Butteraugli/SSIM2 and Zensim. Compile success
  does not qualify GPU execution or all feature combinations.
- Both real-image RD regression tests pass unchanged. All five resource
  tests pass, covering cancellation, concurrent limits/canonicalization
  and streaming layout/color/dispatch. The explicitly selected ignored
  djxl odd-size streaming resampling regression also passes. Initial
  resource invocations with a missing/wrong corpus root failed loudly;
  the successful run uses `~/Library/Caches/codec-corpus/v1`.

Reproduction: `just prepublish-matrix`, `prepublish-float`,
`prepublish-feature-check`, `production-workspace-tests`,
`production-clippy` and `production-resources` accept the pinned manifest.
Full logs, streams and decoder outputs are retained under
`~/tmp/jxl-prepublish-2026-09-24/`; the machine-readable
[check record](../benchmarks/prepublish_checks_2026-09-24.json) names them.
These checks add no production encoding changes or relaxed expectations.

### Performance and baseline validity

The [lossless](../benchmarks/prepublish_lossless_2026-09-24.tsv),
[Zen d1](../benchmarks/prepublish_lossy-zen-d1_2026-09-24.tsv) and
[strict d4](../benchmarks/prepublish_lossy-exact-d4_2026-09-24.tsv)
A/Bs compare `f4bfa242` with `cfd6ef0a`, both against the same current pinned
sibling closure. Each has 48 cells: two natural sources, sizes
64/256/1024/2048, efforts 3/7/9, threads 1/4, five interleaved repetitions.
Process wall includes CLI image IO; tiny cells contain millisecond fixed
cost. Companion `.meta` files record commands and source/binary hashes.

| Mode | Median of cell deltas | Range of cell deltas |
|---|---:|---:|
| Lossless | -0.07% | -7.11% to +2.30% |
| Zen d1 | -1.215% | -10.81% to +1.78% |
| Strict d4 | -6.215% | -38.86% to +2.19% |

The largest observed slowdown is 2.30% on this grid; this is not a
fleetwide regression-free claim. No tuning constants were derived from it.
All 96 lossless/Zen cells retain exact bytes. Ten strict e7 cells change
bytes, and the identity harness correctly fails that arm. All five distinct
changed **baseline** streams fail djxl decoding; one also reproduces
`AnsChecksumMismatch` in Rust. The current counterparts decode successfully,
consistent with the intervening strict EPF metadata-context correction.
The [baseline failure record](../benchmarks/prepublish_baseline_decode_2026-09-24.tsv)
is retained rather than treating a byte change as an unexplained regression.
All 76 distinct current source/stream pairs (covering all 144 cells) fully
render through both decoders, with exact lossless pixels.

### API findings and remaining release gates

Fresh rustdoc comparisons against `f4bfa242`, with expert/JPEG/HDR features,
run 223 semver checks: 221 pass, two fail, 30 skip. One failure is the
owner-approved addition of `CustomEncoderImprovements` fields
`lossless_tree_self_repair` and `lossless_large_tree_bucket_reduction`:
exhaustive downstream struct literals must add them. The other flags shifted
`ValidationError` discriminants after the approved forced-WP variant. A
compiler probe rejects the lint's advertised `as isize` use with E0605
because this enum has data-carrying variants; that particular numeric-cast
break mechanism does not apply. This is not a blanket clean-semver verdict.
Both source revisions are unreleased 0.4.0; no additional version bump was
made. Against reconstructed published 0.3.1, explicitly forcing the minor
check repeats the existing nine breaking-check categories (187 pass,
nine fail, 57 skip), supporting the already planned 0.4 release.

`ZEN_API_DOC=check` still fails stale committed inventories. Native ARM
regeneration changes the encoder supported inventory from 1,256 to 1,302
item lines; these include older additions as well as this batch. The SIMD
inventory is architecture-sensitive, so replacing x86 entries with ARM
entries is not an API-removal finding. A cross-target regeneration also
produced an implausible public/internal split and was not accepted. Proposed
native changes are retained in `api-snapshots.patch`, not applied to the
committed expectations. Review and a valid x86 SIMD refresh remain required.
Existing rustdoc link warnings also need review before publication.

Direct consumer-style semver validation still fails registry resolution on
`butteraugli ^0.9.4`; live registry search returns 0.9.3. The macros crate is
still unpublished, SIMD is at 0.3.0 and zenanalyze at 0.1.0. No package
verification passed. The dependency ordering and optional-dependency
publication decisions below remain release gates. README approval and a
fully green release-commit CI run are also required. This audit does not
substitute for the licensed normative standards audit (#111); deployment
rollout (#108) and fleetwide picker tuning remain deferred by the owner.

Dev was fetched and advanced with jj to the pushed audit head `7a9a5a8c`.
Twelve named bookmarks were preserved, with no named local-only bookmark.
Its divergent `main` remains separate; this is a `libjxl-exact` landing,
not a main merge. Later documentation commits must be synchronized too.

## September 8 validation in progress

CI sibling sources are pinned in
[.github/sibling-revisions.tsv](../.github/sibling-revisions.tsv). The clone
step fetches those exact commits and rejects an existing checkout at another
revision. This pins the source closure; it does not replace a dependency
lockfile or prove a registry consumer can resolve the package.

A clean export of `db4b441c` with all sixteen pinned sibling revisions and a
fresh 133,272-byte lockfile passes the five production tests under `--locked`.
The lockfile and source provenance are in the verified R2/Tower
[validation archive](../benchmarks/production_validation_2026-09-08.pointer.md).
The owner approved committing the lockfile: `05a3976f` tracks it and enforces
`--locked` in CI. `e2d2a49f` adds the pinned Butteraugli accounting API.
Source-build success does not remove the registry publication blockers below.

The crates.io API was checked again on 2026-09-08: the published versions in
section 3 remain unchanged. The four GPU/CVVDP dependencies remain unpublished,
and `jxl-encoder-macros` still has no registry release. No tag, GitHub release,
or crate publication was made during this validation. The remaining consumer
and release gates below still apply.

The September 8 packaging attempt first failed on the missing `cvvdp`
version requirement. Adding `0.1.0` (the local package version) gets packaging
to registry resolution, where it fails on unpublished `butteraugli ^0.9.4`.
The other unpublished requirements below remain blockers. Optional dependencies
must resolve even when disabled. No package verification or publication passed.

Fresh rustdoc was generated for `std,parallel,butteraugli-loop` against the
workspace source closure. Comparing it with the previously reconstructed,
magetypes-pinned 0.3.1 rustdoc with `--release-type minor` executes 196 checks:
187 pass, 9 fail, 57 skip. The nine failing checks cover removed/narrowed API,
new animation fields and changed validation discriminants; they require the
already-planned 0.4 version. The automatic 0.3→0.4 check skips all 253 checks,
so its success is not used as API-validation evidence. Direct source-consumer
semver invocation fails dependency resolution; using explicit rustdoc inputs
isolates the API inventory from that separate packaging blocker.

The concrete [#106](https://github.com/imazen/jxl-encoder/issues/106)
accounting defect is fixed by `e2d2a49f`, `33155ec0` and `9157db37`,
using Butteraugli `d2466a4e`.
The 1.6 GB counterexample rejects before encoding (27.5 MB measured process
peak); the default-budget result is byte-identical and renders in both decoders.
Its measured 2,665,889,792-byte RSS is below the 3,064,194,048-byte
maximum estimate. All twelve repeated resource cells retain their exact
bitstreams and render through both decoders. The unchanged 2 GiB screenshot
regression passes. The API
accounts image buffers, not allocator-retained pages or a hard process ceiling.
Application concurrency and fleet admission still depend on the deployment
workload and host limits; no worldwide rollout was performed.

Local validation of `9157db37` passes workspace all-targets, workspace doctests,
all-target clippy, the explicit production-resource tests, 60 unchanged hash
locks, five Libjxl byte locks, divergence checks and both RD regressions.
Exact CI and nightly results are recorded on the tracking issues; a local
pass is not a substitute for the full platform matrix.

The dependency follow-up [#107](https://github.com/imazen/jxl-encoder/issues/107)
updates the tuning runner to Arrow/Parquet 59.3.0 (`6510695d`), removing Thrift
from the lock. It also pins all thirteen secondary-decoder crates to the
Imazen fork's upstream 0.12.6 merge (`08395e61`), retaining the empty-section
fix. Decoder compatibility CI passes on i686, Windows ARM64, and macOS ARM64.
The encoder's updated dependency closure passes local workspace all-target
tests and both multigroup empty-section regressions through all three decoders.
The per-push encoder matrix and nightly corpus checks remain separate gates.

## 1. What 0.4.0 is

The deliberate public-surface narrowing tracked in #76, folding in the
four accidental 0.3.2-dev breaks from `RELEASE_SEMVER_0.3.1_to_0.3.2.md`
rather than shipping them under a patch version:

- Supported surface (default features): **2,397 → 1,256 item lines,
  46 → 7 pub modules, 177 → 100 types** (`docs/public-api/jxl-encoder.txt`).
  What remains: `api` + the crate-root re-exports, `entropy_coding`
  (`Lz77Method` + `ANSHistogramStrategy` only, both also at the root),
  `modular` (`RctType` only), `validation`, feature-gated
  `jpeg`/`hdr`/`sweep`/`convenience`, and `vardct`'s feature-gated
  remainder (`hdr_metrics`, `chroma_subsampling`, `rate_control`).
- Hidden/internal: `headers`, `vardct` engine internals (`VarDctEncoder`,
  `dct`, `transform`), `modular` internals, `effort`, `bit_writer`,
  `color`, `container`, `error`, `heuristics`, `tuning`, `trace`,
  `debug_rect`, `profile_phases`, `zq_seed`; the legacy `image` module
  was dead and is deleted; all 17 `#[macro_export]` macros are
  doc-hidden.
- Escape hatches (doc-hidden, unsupported, may change freely):
  `__pre_quantized` / `__internals` (unchanged, jxl-gpu's seams),
  new `__gpu` (DCT parity primitives for jxl-gpu), `__test_exports`
  (this repo's it-suite/examples only), root compat re-exports for
  zenjxl (`is_container`, `is_bare_codestream`, `append_gain_map_box`,
  `estimate_encode`, `estimate_encode_threaded`, `encode_threading_info`,
  `EncodeEstimate`, `ThreadingInfo`), and `tuning_runtime` behind
  `tuning-override`.
- Byte-invariance held throughout: hash-locks 53/53 + Libjxl byte-locks
  green on every narrowing commit
  (22e16b3e → fdcf7a6d → 32d76354 → 2df8c275 → bfb880f9 → 4363a3d5).

## 2. semver vs published 0.3.1 — method + verdict

`cargo semver-checks 0.49.0`, run **before** the 0.4.0 bump (manifest at
0.3.2, so no self-baseline trap), features `std,parallel,butteraugli-loop`.

**The published 0.3.1 baseline no longer builds under fresh resolution.**
magetypes 0.9.28 removed APIs (`from_float32x4_t` et al.) that published
jxl-encoder-simd 0.3.0 (req `^0.9.15`) uses → 44 compile errors. Anyone
depending on jxl-encoder 0.3.1 with a fresh `Cargo.lock` is broken
today, independent of anything in this release. Workaround for users:
`cargo update -p magetypes --precise 0.9.23`. This makes publishing
jxl-encoder-simd 0.4.0 (req `magetypes 0.9.27`, verified building
against 0.9.28) a fix in itself.

The comparison ran against a reconstructed baseline: the published
`jxl-encoder-0.3.1.crate` with `magetypes = "=0.9.23"` (the version its
packaged lockfile shipped) added as a direct pin
(`--baseline-root ~/tmp/jxl-encoder-0.3.1-baseline`).

**Verdict: 11 major checks failed, 0 minor — all intentional or
documented** (`~/tmp/jxl-api-narrow-040-semver2.log`):

| check | classification |
|---|---|
| module/struct/enum/function/const `_missing`, `inherent_method_now_doc_hidden`, `macro_now_doc_hidden`, `declarative_macro_missing` | the #76 narrowing (intentional) |
| `feature_missing`: `unsafe-performance` | intentional removal (issue #76 item 7) |
| `constructible_struct_adds_field`: `AnimationParams.premultiplied_alpha`, `AnimationFrame.{blend_mode, blend_source, save_as_reference}` | pre-existing 0.3.2-dev drift, ships under this major (CHANGELOG'd) |
| `enum_no_repr_variant_discriminant_changed`: `ValidationError` (`IterCountOutOfRange` 3→4 …) | pre-existing 0.3.2-dev drift, ships under this major (CHANGELOG'd) |

No keep-list item was removed. Caveat honoured: a green/clean
semver-checks run is a lower bound (it cannot see return-type or
behavioural changes), so real consumers were compiled too:

- **jxl-gpu** (imazen/jxl-gpu @ 8a7010c, shallow clone, path dep pointed
  at this tree, `--no-default-features --features encoder`): reproduces
  its pre-narrowing baseline **exactly** — one pre-existing error
  (`encode_from_pre_quantized_ac` now takes `&[Vec<Vec<i32>>; 3]`
  quant_dc; jxl-gpu still passes i16 — drift from before this work),
  nothing new introduced by the narrowing. Its runtime uses only
  `__pre_quantized` + `api` keep-list items; its parity tests will
  migrate `jxl_encoder::vardct::dct::*` → `jxl_encoder::__gpu::*` and
  `jxl_encoder::effort::EntropyMulTable` → root `EntropyMulTable`.
- **zenjxl** (`cargo check --locked`, scratch target dir): exactly one
  default-surface break — `src/lib.rs:97`
  `pub use jxl_encoder::container::{append_gain_map_box,
  is_bare_codestream, is_container};` → the same three names are now
  doc-hidden **root** re-exports (one-line migration). Feature-gated
  paths to migrate in the same pass: `headers::color_encoding::…` (root
  re-exports), `entropy_coding::ans::ANSHistogramStrategy` (root),
  `heuristics::…` (root).

## 3. Publish blockers — measured, not inherited

Probe: `[patch.crates-io]` `butteraugli` and `zenanalyze` entries
removed from a scratch copy of the root manifest, builds against
crates.io (2026-08-29; manifest restored afterwards). Registry
requirements additionally checked mechanically against the sparse
index for all four members.

**Resolution note:** optional deps must exist on crates.io at the
declared version for `cargo` to resolve AT ALL — every entry below
blocks the unpatched build/publish even when its feature is off.

| dep (req) | crates.io state | class |
|---|---|---|
| `jxl-encoder-macros 0.4.0` | never published | first-publish, free (wave 1) |
| `jxl-encoder-simd 0.4.0` | 0.3.0 latest | publish with this release (wave 1); also fixes the magetypes-0.9.28 breakage |
| `zenanalyze 0.2.0` (optional, default via `learned-admission`) | 0.1.0 latest | wave 2 — **behind `zenanalyze-api 0.1.1` (must publish first)**; release prep DONE (`zenanalyze/docs/RELEASE_0.2.0.md`) |
| `butteraugli 0.9.4` (optional, default via `butteraugli-loop`) | 0.9.3 latest | wave 1 — **0.9.4 DOES gate jxl-encoder** (see below); release prep DONE (`butteraugli/docs/RELEASE_0.9.4.md`, CI 25/25 green); one open owner decision (the `internals`-feature `consts::XYB_*` removals — audit found zero external consumers; ship-as-0.9.4 recommended) |
| `zensim 0.3.0` (optional) | 0.2.7 latest | wave 2 |
| `zenjpeg 0.9.0` (optional + dev-dep) | 0.8.4 latest | wave 3 (its own manifest needs 3 versionless-dep edits first, per the workspace map) |
| `zensim-gpu 0.0.1` (optional) | never published, `publish = false` | **OWNER DECISION** |
| `butteraugli-gpu 0.0.1` (optional) | never published, `publish = false` | **OWNER DECISION** |
| `cvvdp-gpu 0.0.1` (optional) | never published, `publish = false` | **OWNER DECISION** |
| `cvvdp 0.1.0` (path + version, optional) | never published | **OWNER DECISION** |

The four OWNER DECISION rows are the hard stop the workspace map
(`~/work/zen-workspace/PUBLISH_ORDER_2026-08-29.md`, §6 class B3) calls
"the hardest blocker in the wave": a registry dep must exist, so either
the GPU metric crates get published from zenmetrics, or the
`gpu-butteraugli`/`zensim-loop-gpu`/`cvvdp-loop`* features (and their
deps) are stripped from the **published** manifest.

**Corrected fact — butteraugli.** The workspace map (§2 claim 4) and the
2026-08-29 job brief both said butteraugli 0.9.4 does **not** gate
jxl-encoder ("0.9.3 satisfies; none of the 0.9.4-only API is used").
**Measured false**: with the patch removed, butteraugli 0.9.3 fails to
compile `vardct/perceptual_loop.rs:1066` —
`ButteraugliReference::estimated_reference_bytes` (the #93 buttloop
budget guard, added to the fork 2026-06-23) is 0.9.4-only. The earlier
audits grepped only for the `linear_planes`/`ScorerBuilder` surface and
missed this symbol. The dep req is now truthfully `0.9.4` (was a
truncated `"0.9"`; dev-dep likewise). So the default-features publish
chain is: **butteraugli 0.9.4 (wave 1) AND zenanalyze-api 0.1.1 →
zenanalyze 0.2.0 (waves 1→2) both precede jxl-encoder**.

Baseline-availability finding recorded above in §2: published 0.3.1 is
already broken by magetypes 0.9.28 regardless of this release.

## 4. Publish order

Within this repo (unchanged from the issue):

```
jxl-encoder-macros 0.4.0  →  jxl-encoder-simd 0.4.0  →  jxl-encoder 0.4.0  →  jxl-encoder-cli 0.4.0
```

Within the workspace-wide map (stay consistent with
`~/work/zen-workspace/PUBLISH_ORDER_2026-08-29.md` — note it lists the
pre-bump "0.3.2" strings; this repo is now 0.4.0):

- wave 1: `butteraugli 0.9.4`, `zenanalyze-api 0.1.1`,
  `jxl-encoder-macros`, `jxl-encoder-simd`
- wave 2: `zenanalyze 0.2.0`, `zensim 0.3.0`
- wave 3: `zenjpeg 0.9.0`
- wave 4: **`jxl-encoder 0.4.0`** (after the GPU-crate owner decision)
- wave 5: `jxl-encoder-cli 0.4.0`, `zenjxl` (needs its 1-line container
  migration + the feature-gated path fixes from §2)

## 5. Gates before tagging/publishing (the owner runs these)

Per the global release rules — in order, stop on any failure:

1. `cargo test --all-targets` + `cargo test --doc` locally green.
2. Push; CI green on all enabled platforms (including windows-11-arm,
   macOS ARM64, and i686 via cross). The owner disabled macOS Intel CI
   on 2026-09-08.
3. Owner decisions resolved: GPU-crate deps (strip vs publish), `cvvdp`
   path dep, and whether any of the doc-hidden compat re-exports
   (zenjxl's container/heuristics set) get promoted to supported.
4. Upstream waves published through zenjpeg 0.9.0 (or the optional-dep
   reqs temporarily point at published versions — NOT recommended; the
   code needs the newer APIs).
5. `git tag v0.4.0` + push tag + `gh release create v0.4.0` — only after
   CI is green.
6. `cargo publish` in the §4 order (macros → simd → jxl-encoder → cli),
   verifying each lands before the next.
7. README review by the owner before any `cargo publish` (standing
   rule).

Also at release time: regenerate `docs/public-api/jxl-encoder-simd.txt`
on x86_64 — the snapshot is arch-sensitive (aarch64 regen flips the
arch-gated fn list; this prep deliberately left the committed x86
version in place).

## Evidence

- semver run: `~/tmp/jxl-api-narrow-040-semver2.log` (reconstructed
  baseline at `~/tmp/jxl-encoder-0.3.1-baseline`)
- patch-removal probes: `~/tmp/jxl-api-narrow-040-step7-build{,2,3}.log`
  (zenanalyze resolution failure; butteraugli 0.9.3 E0599)
- consumer checks: `~/tmp/jxl-gpu-check-baseline-step1.log` vs
  `~/tmp/jxl-api-narrow-040-step4-gpucheck.log` (identical error
  signature), `~/tmp/jxl-api-narrow-040-zenjxl-check.log`
- registry enumeration: sparse-index check in the step-7 session log
  (results inlined in §3)
