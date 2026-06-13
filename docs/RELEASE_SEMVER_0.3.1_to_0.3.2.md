# Semver gate: 0.3.1 → (proposed 0.3.2) — BREAKING, needs version decision

`cargo semver-checks check-release -p jxl-encoder --baseline-version 0.3.1 --default-features`
(cargo-semver-checks 0.47.0, run 2026-06-13): **4 major checks failed, 0 minor.**
"Summary: semver requires new major version."

> Note: the *all-features* run aborts earlier because `butteraugli-gpu`
> (publish=false) isn't on crates.io — the publish surface is
> default-features-only regardless. Verdict above is the default-features
> surface.

## The 4 breaking changes vs published 0.3.1

1. **`constructible_struct_adds_field`** — pub struct gained pub field(s),
   breaking downstream struct literals:
   - `FileHeader.upsampling_mode` (file_header.rs:241)
   - `FileHeader.upsampling_factor` (file_header.rs:246)
   - `ImageMetadata.relative_to_max_display` (file_header.rs:183)
   - `ImageMetadata.linear_below` (file_header.rs:191)
2. **`constructible_struct_adds_private_field`** — pub struct gained private
   field → no longer struct-literal constructible downstream:
   - `VarDctEncoder.budget` (vardct/encoder.rs:2393)
   - `VarDctEncoder.zenanalyze_proxies` (:2427)
   - `VarDctEncoder.resolved_improvements` (:2463)
3. **`enum_struct_variant_field_added`** — pub enum struct-variant gained a field:
   - `predictor_id` on `GlobalModularState::Huffman` (modular/section.rs:102)
   - `predictor_id` on `GlobalModularState::Ans` (:110)
   - `lz77` on `GlobalModularState::AnsWithTree` (:127)
4. **`feature_missing`** — cargo feature `unsafe-performance` removed/renamed.

## Why this matters

- These are real semver breaks. For a 0.x crate the **minor** is the
  leading-non-zero digit, so a break forces **0.3.x → 0.4.0**, NOT a
  patch bump. Shipping these under 0.3.2 would silently break downstream.
- Per CLAUDE.md: a leading-digit bump requires (a) confirming the break
  is genuine [done: semver-checks], (b) confirming it's unavoidable —
  *actively redesign to dodge it* — and (c) **explicit user approval**.

## Root cause / dodge-ability

Almost all of #1–#3 are **incidentally-public internal types** leaked by
`pub mod headers` / `pub mod vardct` / `pub mod modular` (the
RELEASE_API_REVIEW "~1,200-item pub surface" finding). They are not the
intended public API (`LossyConfig`/`LosslessConfig`/`EncodeRequest`).
But they were already public in 0.3.1, so:

- **Making them additive is hard retroactively**: `#[non_exhaustive]`
  added now is itself a break; narrowing `pub`→`pub(crate)` is also a
  break. The leak already shipped.
- **Feature #4 (`unsafe-performance`)** IS cheaply dodgeable: re-add it as
  a no-op (empty) feature alias to restore the 0.3.1 feature set (the
  crate is `#![forbid(unsafe_code)]`, so a no-op `unsafe-performance = []`
  is harmless and removes that break).

## Options (USER DECISION)

- **A — Ship 0.4.0.** Accept the breaks (they're on internal-ish types
  few/no downstreams construct), document them in CHANGELOG QUEUED
  BREAKING CHANGES, bump workspace 0.3.2 → 0.4.0. Cleanest; honest.
- **B — Keep 0.3.x, dodge what's dodgeable.** Re-add `unsafe-performance`
  no-op (kills #4); for #1–#3 the only true fix is reverting the field
  additions (not viable — they're real features) OR accepting that these
  internal types should never have been pub and doing a deliberate
  pub-surface narrowing as part of a 0.4.0 anyway. So B collapses toward
  A for #1–#3.
- **C — Defer the release**, narrow the public surface properly first
  (move headers/vardct/modular internals to `pub(crate)`), then 0.4.0
  with a small, intentional public API.

**Recommendation: A (0.4.0)** now if a release is wanted soon, with a
follow-up tracking issue for C (surface narrowing) — OR C if there's
appetite to fix the leak before more downstreams pin to internals.

## Other publish blockers (see RELEASE_DEP_AUDIT.md)

- `jxl-encoder-macros` unpublished (non-optional default dep) — hard blocker.
- default build is patch-tested against a `butteraugli` fork; must
  build+test with `[patch.crates-io]` removed vs published `butteraugli 0.9.3`.
- `jxl-encoder-simd` 0.3.0 → needs the coordinated bump.
