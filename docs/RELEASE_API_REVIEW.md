# `jxl-encoder` Public API / Release-Readiness Review (0.3.2)

**Date:** 2026-06-13
**Scope:** Public API surface of `jxl-encoder/` for a crates.io publish at `0.3.2`.
**Method:** READ + grep only (no builds/tests beyond a single `cargo doc --no-deps`). File:line citations throughout.

> **Premise correction (important):** The task brief said "no prior published
> version exists, no semver baseline." That is **wrong**. crates.io serves
> `jxl-encoder` up to **`0.3.1`** (versions `0.3.1, 0.3.0, 0.2.0, 0.1.4, 0.1.3,
> 0.1.2` — confirmed via the crates.io API). `0.3.2` is therefore a **patch
> release over a published `0.3.1` baseline**, NOT a first publish. A
> `cargo semver-checks` run against `0.3.1` is warranted before tagging.
> (The crate's own source acknowledges the baseline: `lib.rs:21,95` reference
> "backwards-compatibility with 0.3.0 which re-exported `EffortProfile`".)

---

## 1. Public surface summary

### Crate-root re-exports (the intended public API)

The intended user-facing API is re-exported at the crate root from `api.rs`
(`lib.rs:77-122`):

- **Configs (layer 1):** `LossyConfig`, `LosslessConfig` (`api.rs:4449`, `api.rs:2136`)
- **Request (layer 2):** `EncodeRequest<'a>` (`api.rs:7681`)
- **Streaming encoders (layer 3):** `LossyEncoder` (`api.rs:10230`), `LosslessEncoder` (`api.rs:11617`) with `push_rows`/`finish`/`finish_into`/`finish_to`/`finish_to_seekable`
- **Pixel/quality:** `PixelLayout` (`api.rs:396`), `Quality` (`api.rs:653`), `quality_to_distance`/`calibrated_jxl_quality` (`api.rs:703,722`)
- **Errors:** `EncodeError` (`api.rs:34`), `At`/`ResultAtExt`/`at` (re-exported from `whereat`), `ValidationError` (`lib.rs:115`)
- **Metadata/limits:** `ImageMetadata<'a>` (`api.rs:1183`), `Limits` (`api.rs:1435`)
- **Modes/dispatch enums:** `EncoderMode`, `EncodeMode`, `ContainerMode`, `Buffering`, `ProgressiveMode`, `ChromaSubsampling`, `Lz77Method`, `BlendMode`, `EpfDispatch`, `PatchesDispatch`, `PixelLossDispatch`, `SinglePassEntropyDispatch`, `PremultipliedAlphaMode`, `NonFiniteAction`, `EncoderStrategy`, `StrategyOverrides`
- **Animation:** `AnimationParams`, `AnimationFrame<'a>`
- **Cancellation:** `Stop`, `Unstoppable` (from `enough`)
- **Color:** `CIExy`, `ColorEncoding`, `ColorSpace`, `CustomPrimaries`, `Primaries`, `RenderingIntent`, `TransferFunction`, `WhitePoint` (`lib.rs:110-113`)
- **Splines/RCT:** `Spline`, `SplinePoint`, `RctType`
- **Consts:** `GROUP_DIM`, `BLOCK_DIM`, `BLOCK_SIZE`, `JXL_SIGNATURE`, `MAX_FASTER_DECODING`, `MAX_PROGRESSIVE_DC`, `MAX_QUANT_LOOP_ITERS`, `DEFAULT_MAX_MEMORY_BYTES`
- **Feature-gated re-exports:** `WritableSeek` (`std`), `HdrLoss` (`butteraugli-loop`), `LossyInternalParams`/`LosslessInternalParams` (`__expert`)

### Wide internal surface exposed as `pub mod` (concern, see §3)

`lib.rs:14-71` marks many internal modules `pub` rather than `pub(crate)`.
Approximate public-item counts (grep of `pub fn|struct|enum|trait|type|const|mod`):

| module | pub items | nature |
|---|---|---|
| `vardct` | **517** | VarDCT internals (DCT, CfL, AC strategy, quant, patches, splines) |
| `modular` | **221** | lossless internals (RCT, tree-learn, ANS plumbing) |
| `entropy_coding` | **168** | ANS / Huffman / LZ77 / tokens |
| `headers` | 70 | bitstream header structs |
| `tuning` | 70 | const-access layer (W44-211) |
| `effort` | 35 | effort-derived knobs |
| `image` | 25 | buffer types |
| `bit_writer` | 24 | bitstream writer |
| `container` | 22 | box/container layout |
| `debug_rect` | 13 | **debug instrumentation** (feature `debug-rect`) |
| `profile_phases` | 11 | **profiling instrumentation** (feature `profile-phases`) |
| `color`, `validation`, `error` | 11 / 5 / 2 | small |

Plus `trace` (bitstream tracing macros). `error.rs` exposes a **second, older**
`Error` enum + `Result` alias (`error.rs:13,18`) alongside the new
`api::EncodeError` — a duplicated error surface.

### Correctly-hidden / gated internals (good)

- `#[doc(hidden)]` modules: `test_helpers`, `__bench_internals` (feat `__bench_internals`), `__internals` (feat `__internals`), `__pre_quantized` (feat `__pre_quantized`) — `lib.rs:140-587`. All gated behind underscore-prefixed non-default features and clearly labelled "Not part of the stable API."
- `#[doc(hidden)] pub use effort::EffortProfile` (`lib.rs:104-105`) — kept `pub` for 0.3.0/0.3.1 back-compat, hidden to discourage new use. This is the **right** call for semver (removing it would break `0.3.x` consumers).
- `pub(crate) mod` correctly used for: `budget`, `f16`, `icc`, `parallel`, `strategy_def_prototype`, `gate_registry`.
- Debug/diagnostic env hooks are compiled out by default (feature `__env_var_diagnostics`, `Cargo.toml:274`); `debug_rect`/`profile_phases`/`trace` expand to no-ops when their features are off.

---

## 2. API convergence status (from CLAUDE.md "API Convergence TODOs")

Verified against code. The three-layer pattern (Config → `EncodeRequest<'a>` →
streaming `Encoder`) is **implemented and consistent**.

### DONE (`[x]` in CLAUDE.md — verified present)

`LossyConfig`/`LosslessConfig` split; `EncodeRequest<'a>` (`api.rs:7681`);
one-shot `encode`/`encode_into`/`encode_to` (`api.rs:8228-8253`); `PixelLayout`
(`api.rs:396`); `EncodeError` (`api.rs:34`); u32 dims; `Limits` (`api.rs:1435`);
`ImageMetadata` type **and wired** (ICC/EXIF/XMP); `Quality` enum (`api.rs:653`);
`&dyn Stop` cancellation; `with_`/bare-name builder convention; fluent
`encode()`/`encode_into()` on configs; free functions removed (the only
free-fn-style helpers, `encode_rgb8` etc., are gated behind the opt-in
`convenience` feature — `lib.rs:118-122`); streaming
`LossyEncoder`/`LosslessEncoder` with `push_rows`/`finish*`; `encode_to`/`finish_to`
correctly `#[cfg(feature = "std")]`-gated (`api.rs:8251`, `1802`); `At<>` error
location; `EncodeStats` (`api.rs:320`); `Rgba8`/`Bgra8`/`Bgr8`/lossy+alpha
layouts (`api.rs:396-598`).

### PENDING (`[ ]` in CLAUDE.md)

- **`estimate_memory()` / `estimate_memory_ceiling()`** — listed PENDING, but a
  **differently-named** equivalent ships: `LossyConfig::estimate_peak_memory_bytes`
  / `LosslessConfig::estimate_peak_memory_bytes` (`api.rs:7524`, `3216`).
  Functionally DONE under a different name; reconcile the TODO/naming before
  freezing the API.
- **Probing: `ImageInfo::from_bytes(&[u8])` + `PROBE_BYTES`** — genuinely
  **NOT IMPLEMENTED** (no symbols found). PENDING.
- **Two-phase decoder / `Bgra8` decode** — decoder-side TODOs, **out of scope**
  for this encoder crate.

---

## 3. Release-readiness issues (ranked)

### 🔴 BLOCKER 1 — `jxl-encoder-macros` is NOT published on crates.io

`Cargo.toml:26`:
```toml
jxl_encoder_macros = { package = "jxl-encoder-macros", version = "0.3.2", path = "../jxl-encoder-macros" }
```
This is an **always-on, non-optional** dependency. The crates.io API returns
`"crate jxl-encoder-macros does not exist"` and `cargo search` finds nothing.
`cargo publish` strips the `path =` and depends on `jxl-encoder-macros = "0.3.2"`
from the registry, which does not exist. **`jxl-encoder 0.3.2` cannot be
published until `jxl-encoder-macros 0.3.2` is published first.** (The macro was
introduced in W44-192/193; `jxl-encoder 0.3.1` predates it, which is why a 0.3.1
publish succeeded without it.)

*Sibling crate status:* `jxl-encoder-simd = "0.3.0"` **is** published and matches
the dep version (`Cargo.toml:30`) — not a blocker.

### 🟠 HIGH 2 — default build is developed/tested against a path-patched `butteraugli`, not the registry crate

`default = ["std", "butteraugli-loop"]` (`Cargo.toml:327`), and `butteraugli-loop`
pulls `butteraugli` (`Cargo.toml:140`). The workspace `[patch.crates-io]`
redirects `butteraugli` to `../../butteraugli/butteraugli` (`Cargo.toml:93`) with
a note that this carries "latest unpublished fixes" vs crates.io. The published
crate depends on `butteraugli = "0.9"` (registry has `0.9.3`). The **patch is not
applied for downstream consumers or for `cargo publish`** — so the API the code
is compiled/tested against locally may differ from the published `butteraugli
0.9.3`. *Mitigation:* `butteraugli` types do **not** leak into public signatures
(grep clean), so this is a build-against-vs-ship-against risk, not an API leak.
**Action:** verify the crate builds + tests pass with the `[patch.crates-io]`
removed (i.e. against published `butteraugli 0.9.3`) before publishing.
No path-only crate is in the **default** dependency closure otherwise.

### 🟡 MEDIUM 3 — unpublished path-dep types in the public API are correctly feature-gated (NOT a leak in default build)

`DisplayConfig::display_model()` / `display_geometry()` return
`cvvdp_gpu::params::DisplayModel` / `DisplayGeometry` (`api.rs:5059`, `5093`).
`cvvdp_gpu` is `publish = false` / path-pinned. **Both methods are
`#[cfg(feature = "cvvdp-loop")]`-gated** and `cvvdp-loop` is **not** a default
feature, so these unpublished types do **not** appear in the default-publish
surface. *However*, anyone enabling `cvvdp-loop` (or `cvvdp-loop-cpu`,
`zensim-loop`, `zensim-loop-gpu`, `gpu-butteraugli`) pulls path-only/`publish=false`
crates (`cvvdp-gpu`, `cvvdp-cpu`/`cvvdp`, `zensim-gpu`, `zensim`, `butteraugli-gpu`,
plus `zensim = path` and `zenjpeg = path` for `jpeg-reencoding`) — **those feature
combinations are unpublishable** until their deps are on crates.io.
**Verdict:** publishable default + the published-registry optional features
(`parallel`, `convenience`, `hdr-gainmap` via `ultrahdr-core 0.5`, `chroma-subsampling`
via `zenyuv 0.1.3`, `brotli-metadata`, `rate-control`, `ssim2-loop`) are fine; the
GPU/metric-fork features are dev-only.
> One public **type** to note: `DisplayConfig` (the enum) is unconditionally
> `pub`, so its docs link to `cvvdp_gpu::params::*` and produce broken-doc-link
> warnings on docs.rs even in the default build (see issue 5).

### 🟡 MEDIUM 4 — very large internal surface exposed as `pub mod`

`vardct` (517), `modular` (221), `entropy_coding` (168), `headers` (70),
`bit_writer` (24), `container` (22) and the debug/profiling modules
(`debug_rect`, `profile_phases`, `trace`) are all `pub` (`lib.rs:14-71`). This
exposes ~1,200 internal items (DCT kernels, ANS plumbing, bitstream layout,
debug CSV loggers) as stable public API. Because **`0.3.1` already shipped this
surface**, narrowing it now is itself a breaking change — but every additional
patch release further entrenches it. **Recommendation:** before any `0.4`,
demote these to `pub(crate)` (or move the genuinely-needed parity items behind
the existing `__internals` feature) and re-export only the curated root set.
Not a 0.3.2 blocker, but the single biggest long-term API-debt item.

### 🟡 MEDIUM 5 — ~30 broken intra-doc links → broken links on docs.rs

`cargo doc --no-deps` (default features) emits ~30
`rustdoc::broken_intra_doc_links` warnings. Public docs link to **private items**
(`crate::validation::ITER_MAX`, `crate::gate_registry`,
`crate::vardct::patches::find_and_build_with_per_patch_gate`,
`StrategyOverrides::apply_to`, `EncoderStrategy::resolve`, `ResolvedImprovements`,
`crate::modular::palette::MAX_PALETTE_COLORS`, …) and to **nonexistent items**
(`ExtraChannelType::Black`, `LosslessConfig::encode_jpeg_transcode`,
`AnimationRequest::encode`, `cvvdp_gpu::params::DisplayModel::compute_y_refl`,
`Self::with_content_aware_entropy_mul`). These render as dead links on docs.rs.
Cheap to fix; should be cleaned before the doc build that users see.

### 🟢 LOW 6 — root re-export inconsistency (ergonomics)

Re-export hygiene is mixed. `EncoderStrategy`, `StrategyOverrides`,
`ButtloopQfSeedPolicy` are at the crate root, but related types that callers need
in the same code are **only** under `api::`: `ScreenshotEntropyMulPolicy`,
`HighDPhotoEntropyMulPolicy`, `Dct64SearchPolicy`, `Dct32SearchPolicy`,
`SmoothPhotoDct64Policy`, `AdaptiveQuantQfSeedPolicy`, `EpfSharpnessSeed`,
`EffortGate`, `DisplayConfig`, `PerceptualMetric`, `PerceptualDevice`, plus
`ExtraChannel`/`ExtraChannelBuf` (used by the public extra-channel request API at
`api.rs:7757-7851`). Callers must mix `jxl_encoder::X` and `jxl_encoder::api::Y`
imports. Decide one home per type. Not a blocker.

### 🟢 LOW 7 — `no_std` claim vs reality

`#![forbid(unsafe_code)]` **holds** (`lib.rs:10`; grep finds zero `unsafe` blocks
in `src/`) — good. `extern crate alloc` is present (`lib.rs:12`) and `std`-only
items are feature-gated behind the `std` feature (default-on). But there is **no
`#![no_std]` attribute**, so the crate always links `std`. CLAUDE.md's project
standard says "no_std+alloc (minimum: wasm32)"; the crate does not currently meet
that literally (it's std-by-default with std-gated extras). Confirm whether a
`--no-default-features` (`std` off) build actually compiles for `no_std` targets,
or update the claim. MSRV is declared `1.89` (`Cargo.toml:52`, README badge);
edition `2024`.

### Non-issues confirmed (good)

- `#![forbid(unsafe_code)]` enforced; no `unsafe` in `src/`.
- No public item references `W44`/`dump`/`probe`/`diagnostic` by name (grep clean) — internal naming did not leak into stable symbols.
- README present with badges + dual-license (AGPL-3.0 / commercial) text (`README.md`).
- CHANGELOG present at workspace root with `[Unreleased]` + `[0.3.2] - 2026-05-06` + `[0.3.1]` sections (verify `[Unreleased]` items get folded into the final `0.3.2` notes before tag).
- `EffortProfile` retained `pub` (doc-hidden) for `0.3.x` back-compat — correct semver hygiene.

---

## 4. Verdict

### ❌ NOT READY for the `0.3.2` publish (as configured)

**Hard blocker:** `jxl-encoder-macros` (an always-on dependency at `version =
"0.3.2"`) is **not on crates.io** — `cargo publish` will fail to resolve it
(BLOCKER 1). This must be published first.

**Must do before tagging `0.3.2`:**
1. **Publish `jxl-encoder-macros 0.3.2`** to crates.io (it is currently
   path-only/unpublished). `jxl-encoder-simd 0.3.0` is already published.
2. **Build + test with `[patch.crates-io]` removed** to prove the default build
   compiles/tests against published `butteraugli 0.9.3` (HIGH 2), not the local
   path-patched fork.
3. **Run `cargo semver-checks` against the published `0.3.1`** — there IS a
   baseline; confirm `0.3.2` is patch-compatible (additive only) and not an
   accidental break.
4. **Fix the ~30 broken intra-doc links** (MEDIUM 5) so docs.rs renders cleanly.

**Not blocking, but track for `0.4`:** narrow the ~1,200-item `pub mod` internal
surface to `pub(crate)`/`__internals` (MEDIUM 4); unify root re-exports (LOW 6);
reconcile `estimate_peak_memory_bytes` naming + implement probing
(`ImageInfo::from_bytes`) per the convergence TODOs; resolve the `no_std` claim
(LOW 7).

**Default + registry-published optional features are otherwise publish-clean**
(no `butteraugli`/path-dep types leak into the default public signature; the
GPU/metric-fork features that pull `publish=false` deps are correctly opt-in and
should be documented as dev-only / unpublishable).
