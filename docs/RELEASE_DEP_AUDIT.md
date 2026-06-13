# Release Dependency Publish-Readiness Audit — jxl-encoder 0.3.2

**Date:** 2026-06-13
**Scope:** Publishing `jxl-encoder` (+ `jxl-encoder-simd`, `jxl-encoder-macros`,
optionally `jxl-encoder-cli`) to crates.io at workspace version **0.3.2**.
**Method:** `cargo metadata --no-deps`, `cargo tree -e no-dev` (default *and*
`--no-default-features`), and the crates.io HTTP API
(`GET /api/v1/crates/<name>`). Every publish-status / version claim below is
from live command output — nothing inferred.

> **CORRECTION to the task premise.** The brief stated "none of jxl-encoder,
> jxl-encoder-simd, jxl-encoder-macros are currently on crates.io." That is
> **wrong**: `jxl-encoder` is already published (latest **0.3.1**) and
> `jxl-encoder-simd` is already published (latest **0.3.0**). Only
> `jxl-encoder-macros` has never been published. So 0.3.2 is a *version bump*
> of two existing crates plus a *first publish* of the macros crate — not a
> first publish of the whole set.

---

## 1. Workspace crates (the things we are publishing)

| Crate | Version (workspace) | On crates.io? | `publish` field | License |
|---|---|---|---|---|
| `jxl-encoder` | 0.3.2 | **Yes — 0.3.1** (need 0.3.2) | (publishable) | `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` |
| `jxl-encoder-simd` | 0.3.2 | **Yes — 0.3.0** (need 0.3.2) | (publishable) | same |
| `jxl-encoder-macros` | 0.3.2 | **NO — never published** | (publishable) | same |
| `jxl-encoder-cli` | 0.3.2 | not checked (optional to ship) | (publishable) | same |
| `zenjxl-tuning-runner` | 0.1.0 | n/a | **`publish = false`** | same |

All four shippable crates carry `license = "AGPL-3.0-only OR LicenseRef-Imazen-Commercial"`
via `workspace.package`. `zenjxl-tuning-runner` is `publish = false` and is out
of scope (an internal sweep-worker binary).

---

## 2. Path / git dependencies (the blockers and near-blockers)

`cargo metadata` reports each dependency's `source` **after** applying
`[patch.crates-io]`. Because the workspace root patches several deps to local
sibling paths, only deps whose version requirement is *unsatisfiable* from
crates.io (or that are declared as a literal `path = ` dep with no registry
coordinate) actually behave as path deps for a **published** build. The table
distinguishes the two cases.

### 2a. Declared as literal `path = ` (no registry coordinate) — true path deps

| Dep (name / package) | Kind | Required/Optional (feature) | Workspace-internal? | On crates.io? | Blocks default publish? |
|---|---|---|---|---|---|
| `jxl-encoder-macros` (`jxl_encoder_macros`) | normal | **REQUIRED** (always, default) | **internal** | **NO** | **YES — hard blocker** |
| `jxl-encoder-simd` (`jxl_simd`) | normal | **REQUIRED** (always, default) | **internal** | yes (0.3.0; need 0.3.2) | YES until 0.3.2 published |
| `jxl-encoder` (`jxl_encoder`, from CLI) | normal | REQUIRED for CLI | **internal** | yes (0.3.1; need 0.3.2) | only blocks CLI |
| `cvvdp` (dep name `cvvdp-cpu`) | normal | OPTIONAL (`cvvdp-loop-cpu`) | external (zenmetrics) | **NO** (`publish=false`) | no (non-default opt-in) |

> `jxl-encoder-macros` and `jxl-encoder-simd` carry **both** a `path` AND a
> registry `version`, so a published `jxl-encoder` resolves them from crates.io.
> `jxl-encoder-simd` 0.3.0 exists but the manifest dep string says
> `version = "0.3.0"` while the crate is now 0.3.2 — see §5 (staleness). The
> macros crate has no published version at all.
>
> `cvvdp` is the **only** dependency declared with a bare `path = ` and **no**
> registry version (`req = *`), because a `[patch.crates-io]` cannot rename a
> package. It is optional (`cvvdp-loop-cpu` feature) and never in the default
> build.

### 2b. Path-PATCHED but with a registry version (patch is dev-only)

These appear in `[patch.crates-io]` pointing at local siblings, but each
`jxl-encoder` dep declares a real crates.io version. For a **published** build
the patch is ignored and the crates.io version is used (if it exists).

| Dep | Required ver | Required/Optional (feature) | External? | On crates.io? | Default-publish impact |
|---|---|---|---|---|---|
| `butteraugli` | `^0.9` | OPTIONAL — but in **default** via `butteraugli-loop` | external | **yes** (0.9.3) | OK; published build uses 0.9.3 (see §4 behaviour flag) |
| `zenjpeg` | `^0.8.4` | OPTIONAL (`jpeg-reencoding`) | external | **yes** (0.8.4) | none (non-default) |
| `zenjxl-decoder` | `^0.3.8` | OPTIONAL (`rate-control`) + DEV | external | **yes** (0.3.8–0.3.10) | none (non-default) |
| `zensim` | `^0.3.0` | OPTIONAL (`zensim-loop`) | external | **version MISSING** (latest 0.2.7) | none for default; **`zensim-loop` unpublishable** |
| `butteraugli-gpu` | `^0.0.1` | OPTIONAL (`gpu-butteraugli`) | external | **NO** (`publish=false`) | none (non-default) |
| `cvvdp-gpu` | `^0.0.1` | OPTIONAL (`cvvdp-loop`) + DEV | external | **NO** (`publish=false`) | none (non-default) |
| `zensim-gpu` | `^0.0.1` | OPTIONAL (`zensim-loop-gpu`) | external | **NO** (`publish=false`) | none (non-default) |

> **`zensim-loop` is broken for crates.io even though it's not a blocker.**
> The req is `zensim = "0.3.0"` but crates.io only serves up to **0.2.7**.
> A published `jxl-encoder` with `--features zensim-loop` would fail to resolve.
> The `[patch.crates-io]` (local `../zensim/zensim` = unpublished 0.3.x) masks
> this locally. Either bump zensim to 0.3.x on crates.io or relax the req
> before advertising `zensim-loop`.

### 2c. `[patch.crates-io]` git entries (jxl-oxide fork — dev-only)

The root manifest patches **14 jxl-oxide-family crates** (`jxl-oxide`,
`jxl-color`, `jxl-image`, `jxl-bitstream`, `jxl-coding`, `jxl-frame`,
`jxl-grid`, `jxl-modular`, `jxl-render`, `jxl-vardct`, `jxl-threadpool`,
`jxl-jbr`, `jxl-oxide-common`) to `github.com/imazen/jxl-oxide@fd4e2c3`.

- These are **only ever DEV-dependencies** of `jxl-encoder` (roundtrip-test
  decoders; jxl-rs/`jxl` is primary). They are **not** in the no-dev tree.
- `[patch.crates-io]` is **stripped from published manifests** by cargo, and
  patches never affect downstream consumers. So they **do not block** publish.
- **Behaviour flag:** anyone building the *test suite* of a published crate
  would resolve the crates.io jxl-oxide (0.12.5) instead of the fork. Per
  CLAUDE.md the fork only changes which empty-modular-section bitstreams
  jxl-oxide *accepts*; encoder output is unaffected. No runtime behaviour
  change for library users.

---

## 3. Default-features dependency closure (the minimal publishable surface)

`default = ["std", "butteraugli-loop"]`. `cargo tree -p jxl-encoder -e no-dev`
(default features) resolves **only two non-registry deps**, both
workspace-internal path crates:

- `jxl-encoder-macros` (path) — **REQUIRED, unpublished → BLOCKER**
- `jxl-encoder-simd` (path) — REQUIRED, published 0.3.0 (need 0.3.2)

Every other default-tree dep resolves from crates.io and exists at the required
version:

```
archmage 0.9.26, archmage-macros, bytemuck 1.25 + bytemuck_derive,
butteraugli 0.9.x (path-patched locally; crates.io 0.9.3 for published build),
crossbeam-{deque,epoch,utils}, either, enough 0.4.4, foldhash,
hashbrown 0.16, imgref 1.12, libm 0.2.16, magetypes 0.9.26, once_cell 1.21,
proc-macro2, quote, rayon 1.12 + rayon-core, rgb 0.8, rustversion,
safe_unaligned_simd, syn 2.0, thiserror 2.0 + thiserror-impl,
unicode-ident, whereat 0.1.5
```

`--no-default-features` drops `butteraugli`, `imgref`, `rgb`, `rayon` and the
crossbeam stack; the only path deps remain `jxl-encoder-macros` +
`jxl-encoder-simd`. **Conclusion: the default surface needs exactly the two
internal crates published; no external path/git dep is in the default closure.**

`jxl-encoder-simd` (default) pulls only `archmage`, `magetypes`, `libm` (all
crates.io). `jxl-encoder-macros` pulls only `proc-macro2`, `quote`, `syn` (all
crates.io). Both are self-contained.

---

## 4. License check

- All workspace crates: `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`. The
  AGPL arm is OSI/SPDX-valid; the commercial arm is a `LicenseRef-` custom
  expression (valid SPDX form). crates.io accepts this.
- No required (default) dependency has a license incompatible with shipping an
  AGPL-3.0 crate. The default closure is the usual permissive Rust stack
  (MIT/Apache-2.0/Zlib/BSD-ish) plus the in-org `butteraugli` (which is itself
  AGPL/commercial dual, compatible). No GPL-incompatible or missing-license dep
  was found in the default no-dev tree.
- *Not exhaustively SPDX-audited per-crate here* (out of read-only scope to run
  `cargo deny`); flag if a formal license-compatibility gate is required before
  publish. Nothing in the default tree is a known red flag.

---

## 5. Version-string staleness (fix before publish)

These compile today only because path deps override the version string. For a
correct published manifest they must be bumped to match the 0.3.2 workspace:

| File | Dep | Declared | Should be |
|---|---|---|---|
| `jxl-encoder/Cargo.toml` | `jxl_simd` (= `jxl-encoder-simd`) | `version = "0.3.0"` | `0.3.2` |
| `jxl-encoder-cli/Cargo.toml` | `jxl_encoder` (= `jxl-encoder`) | `version = "0.3.0"` | `0.3.2` |
| `jxl-encoder/Cargo.toml` | `jxl_encoder_macros` | `version = "0.3.2"` | already correct ✓ |

(Leaving these at 0.3.0 means a downstream consumer of published `jxl-encoder`
0.3.2 could resolve `jxl-encoder-simd` 0.3.0/0.3.1 instead of 0.3.2 — a silent
version skew. Lockstep `=0.3.2` or `0.3.2` is the safe choice for internal
crates.)

---

## 6. Topological publish order

Internal crates have **no inter-dependency cycles**. Order, bottom-up:

1. **`jxl-encoder-macros` 0.3.2** — FIRST PUBLISH. Deps: only crates.io
   (`proc-macro2`, `quote`, `syn`). Nothing blocks it.
2. **`jxl-encoder-simd` 0.3.2** — version bump. Deps: only crates.io
   (`archmage`, `magetypes`, `libm`). Independent of the macros crate; can be
   published in parallel with step 1.
3. **`jxl-encoder` 0.3.2** — version bump. Requires steps 1 + 2 to be live on
   crates.io first (both are non-optional path deps). After they land, the
   **default-features** build resolves entirely from crates.io.
4. **`jxl-encoder-cli` 0.3.2** *(optional to ship)* — requires step 3 live.

**Deferred indefinitely (do NOT block 0.3.2):** every GPU/metric feature crate
(`butteraugli-gpu`, `cvvdp-gpu`, `zensim-gpu`, `cvvdp`) is `publish = false`
and behind a non-default opt-in feature. The `zensim-loop` feature's `zensim
0.3.0` requirement is also unsatisfiable from crates.io but is non-default.
These can ship later (or stay path-only) without affecting the default crate.

**Count of crates that MUST be published before `jxl-encoder` 0.3.2: 2**
(`jxl-encoder-macros`, then/with `jxl-encoder-simd`).

---

## 7. Blockers for first (0.3.2) publish

**Hard blockers (default `cargo publish -p jxl-encoder` fails today):**

1. **`jxl-encoder-macros` is not on crates.io.** It is a non-optional,
   default-pulled dependency. `cargo publish -p jxl-encoder` will reject the
   path dep until the macros crate is published at 0.3.2. → Publish it first.
2. **`jxl-encoder-simd` 0.3.2 is not yet published** (0.3.0 is). The path dep
   resolves locally but a published `jxl-encoder` 0.3.2 needs simd 0.3.2 on
   crates.io (or to keep `version = "0.3.0"`, which then skews — see §5).
   → Publish simd 0.3.2.

**Not blockers, but fix/flag before/at publish:**

3. **Stale version strings** (§5): `jxl_simd` `0.3.0` and CLI's `jxl_encoder`
   `0.3.0` should be `0.3.2`.
4. **`zensim-loop` feature is unpublishable** (req `zensim = "0.3.0"`, crates.io
   max 0.2.7). Non-default, so it won't block, but advertising the feature on a
   published crate would give consumers an unresolvable build. Either bump
   zensim on crates.io to 0.3.x, relax the req to `0.2.7`, or document the
   feature as path-only.
5. **GPU/metric opt-in features** (`gpu-butteraugli`, `cvvdp-loop`,
   `cvvdp-loop-cpu`, `zensim-loop-gpu`) depend on `publish = false` crates
   (`butteraugli-gpu`, `cvvdp-gpu`, `cvvdp`, `zensim-gpu`). They are non-default
   and will simply be unbuildable for crates.io consumers — acceptable as long
   as the README/feature docs note these as internal/path-only.
6. **`[patch.crates-io]` jxl-oxide git fork + local butteraugli/path patches**
   are dev-only and stripped on publish — **not blockers**. Behaviour note: a
   published-crate *test build* resolves crates.io `jxl-oxide 0.12.5` and
   `butteraugli 0.9.3` instead of the local forks; per CLAUDE.md neither changes
   encoder *output* (jxl-oxide fork only widens decode acceptance; local
   butteraugli carries buttloop fixes that affect the dev regression-gate, not
   library users on the default path).

**Bottom line:** Once `jxl-encoder-macros` 0.3.2 and `jxl-encoder-simd` 0.3.2
are published (and the two stale version strings bumped), a
**default-features** `cargo publish -p jxl-encoder` 0.3.2 resolves entirely from
crates.io and is clean. No external path or git dependency sits in the default
publishable surface.
