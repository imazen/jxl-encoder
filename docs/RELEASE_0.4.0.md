# jxl-encoder 0.4.0 — release prep (NOT published)

Status 2026-08-29: version bumped to 0.4.0 on `main`, public surface
narrowed per #76, semver + consumer verification done, publish gates
below **not** run. Nothing is tagged and nothing is on crates.io — the
owner publishes. Supersedes the per-item claims in
`RELEASE_DEP_AUDIT.md` (2026-06) where they conflict; the numbers here
are from executed probes on 2026-08-29 (see "Evidence").

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
| `cvvdp` (path, no version, optional) | never published | **OWNER DECISION** |

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
2. Push; CI green on ALL platforms (incl. windows-11-arm, macOS Intel,
   i686 via cross).
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
