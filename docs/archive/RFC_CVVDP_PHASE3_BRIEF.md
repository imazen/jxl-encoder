# Phase 3 agent brief: CvvdpBackend impl

**Status**: BLOCKED on Agent B (cvvdp-gpu diffmap API) + Agent C (PerceptualBackend rename).

Fire this brief as a new agent the moment both B and C report shipped commits. The brief is self-contained; the launcher (human or orchestrator) just paste-and-go.

---

You are adding `CvvdpBackend` — the cvvdp impl of jxl-encoder's `PerceptualBackend` trait. This is Phase 3 of the cvvdp fork. See `docs/RFC_CVVDP_FORK.md` §2.1 and §4 Phase 3.

## Pre-flight (verify before starting)

These two MUST be visible on `cvvdp-fork-rfc` branch before you start:

1. **Agent B's diffmap API** — `cvvdp_gpu::CvvdpOpaque::score_from_linear_planes_with_diffmap(ref_r, ref_g, ref_b, dist_r, dist_g, dist_b, diffmap_out) -> Result<f32>` exists and is published. Verify by reading `~/work/zen/zenmetrics/crates/cvvdp-gpu/src/opaque.rs`.
2. **Agent C's rename** — `pub(crate) trait PerceptualBackend` lives in `jxl-encoder/src/vardct/perceptual_backend.rs`. Verify with `grep -r 'trait PerceptualBackend' ~/work/zen/jxl-encoder/jxl-encoder/src/`.

If either is missing: HONEST-STOP. Don't try to work around it.

## Read FIRST

1. `/home/lilith/.claude/CLAUDE.md` — every default.
2. `~/work/zen/jxl-encoder/CLAUDE.md` — encoder rules, divergence-table rule, hash-lock + drift mandate.
3. `~/work/claudehints/topics/multi-agent-excellence.md`.
4. `~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` §2 (architecture), §3 (diffmap contract), §6 (acceptance gates).
5. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/perceptual_backend.rs` (post-rename version — Agent C's output). This is the structural reference for your new file.
6. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/butteraugli_loop.rs` — call sites that consume the backend. You'll be wiring CvvdpBackend into the construct_backend dispatch.
7. `~/work/zen/zenmetrics/crates/cvvdp-gpu/src/opaque.rs` — the CvvdpOpaque API surface, especially the new diffmap methods Agent B added.

## Workspace claim

Create your own sibling workspace:
```bash
cd ~/work/zen/jxl-encoder
jj workspace add ~/work/zen/jxl-encoder--cvvdp-backend-impl --revision cvvdp-fork-rfc@origin
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && printf '%s agent-cvvdp-backend-impl bootstrapping\n' "$(cat /tmp/ts)" > ~/work/zen/jxl-encoder--cvvdp-backend-impl/.workongoing
cd ~/work/zen/jxl-encoder--cvvdp-backend-impl
```

Refresh marker every 2 min.

## File ownership

**YOURS:**
- NEW file `jxl-encoder/src/vardct/cvvdp_backend.rs`
- `jxl-encoder/src/vardct/mod.rs` (add `mod cvvdp_backend;`)
- `jxl-encoder/src/vardct/perceptual_backend.rs` — extend `construct_backend()` to route to cvvdp when requested. Don't refactor anything else.
- `jxl-encoder/Cargo.toml` — add `cvvdp-loop` feature, optional `cvvdp-gpu` dep
- `jxl-encoder/src/api.rs` — add `LossyConfig::with_cvvdp_loop(bool)` + field. Field default = `false`. Plumbs through to `VarDctEncoder.cvvdp_loop` field for buttloop's `construct_backend` dispatch (full plumbing is Phase 4; this chunk only exposes the field).
- New unit tests at `jxl-encoder/src/vardct/cvvdp_backend.rs` bottom (5-10 tests)
- Integration test `jxl-encoder/tests/cvvdp_backend_smoke.rs`
- `docs/LIBJXL_DIVERGENCES.md` Section E (opt-in API) — add row for `with_cvvdp_loop`

**DO NOT TOUCH:**
- `vardct/butteraugli_loop.rs` core logic (Phase 4 — separate chunk). You may touch the `construct_backend` import / call site but the buttloop's body is off-limits for this chunk.
- `vardct/perceptual_backend.rs::CpuButteraugliBackend` impl
- `vardct/perceptual_backend.rs::gpu` module
- Any encoder-strategy / gate-registry source (this is just adding an opt-in API)

## Deliverable

### 1. New file `vardct/cvvdp_backend.rs`

Mirrors the GPU butteraugli backend's shape (in `perceptual_backend.rs::gpu`). Sketch:

```rust
// SPDX-License-Identifier: AGPL-3.0-or-later

//! cvvdp-based PerceptualBackend impls (W44-CVVDP Phase 3).
//!
//! Provides:
//! - `GpuCvvdpBackend` (feature `cvvdp-loop`) — wraps cvvdp_gpu::CvvdpOpaque,
//!   uploads linear-f32 planes directly via score_from_linear_planes_with_diffmap.
//! - `CpuCvvdpBackend` (feature `cvvdp-loop-cpu`, Phase 5) — wraps the
//!   forthcoming cvvdp-cpu crate. Stubbed `unimplemented!()` in this chunk;
//!   wiring is a Phase 5 follow-on.
//!
//! Both produce a `BackendCompareResult.score` in butteraugli-direction
//! semantics (smaller = better): `score = (10.0 - jod).clamp(0.0, 10.0)`.

use alloc::format;

use crate::error::Result;
use super::perceptual_backend::{PerceptualBackend, BackendCompareResult};

#[cfg(feature = "cvvdp-loop")]
pub(crate) struct GpuCvvdpBackend {
    inner: cvvdp_gpu::CvvdpOpaque,
    width: u32,
    height: u32,
    reference_warmed: bool,
}

#[cfg(feature = "cvvdp-loop")]
impl GpuCvvdpBackend {
    pub(crate) fn try_new(width: u32, height: u32, /* params */) -> Option<Self> {
        // catch_unwind around cubecl init — silent fallback to butteraugli on
        // missing CUDA. Mirror GpuButteraugliBackend::try_new pattern.
        // Pick backend via cvvdp_gpu::Backend::auto or similar.
        // ...
    }
}

#[cfg(feature = "cvvdp-loop")]
impl core::fmt::Debug for GpuCvvdpBackend { /* ... */ }

#[cfg(feature = "cvvdp-loop")]
impl PerceptualBackend for GpuCvvdpBackend {
    fn name(&self) -> &'static str { "cvvdp-gpu" }

    fn set_reference(
        &mut self,
        ref_r: &[f32], ref_g: &[f32], ref_b: &[f32],
        width: usize, height: usize,
    ) -> Result<()> {
        // dim check
        // call cvvdp_gpu::CvvdpOpaque::warm_reference_from_linear_planes
        self.reference_warmed = true;
        Ok(())
    }

    fn compare_with_reference(
        &mut self,
        dist_r: &[f32], dist_g: &[f32], dist_b: &[f32],
        padded_width: usize, width: usize, height: usize,
        diffmap_out: &mut alloc::vec::Vec<f32>,
    ) -> Result<BackendCompareResult> {
        // 1. Strip padding (mirror GpuButteraugliBackend::copy_strided_row_into_scratch).
        //    cvvdp expects tight stride on the distorted side.
        // 2. Call cvvdp_gpu::CvvdpOpaque::score_from_linear_planes_with_warm_ref_diffmap.
        // 3. Convert JOD to butteraugli-direction: score = (10.0 - jod).clamp(0.0, 10.0).
        // 4. diffmap_out.len() == width * height (cvvdp diffmap is already tight).
        Ok(BackendCompareResult { score })
    }
}
```

### 2. Update `perceptual_backend.rs::construct_backend`

Add a `cvvdp_requested: bool` parameter (or use a `BackendKind` enum). When `cvvdp_requested == true` AND `cfg!(feature = "cvvdp-loop")` AND cubecl init succeeds → return `GpuCvvdpBackend`. Fall back to existing butteraugli dispatch otherwise. Mirror W44-PHASE3-B5-flip's `resolve_*` resolver pattern.

### 3. `LossyConfig::with_cvvdp_loop(Option<bool>) → Self`

Optional override: `None` = use default (butteraugli), `Some(true)` = use cvvdp, `Some(false)` = explicitly use butteraugli even if cvvdp is the eventual default.

Wire field through to `VarDctEncoder` at all 3 sites (still / streaming / animation) — but DON'T touch the buttloop body in this chunk; the field just propagates to `construct_backend`'s caller in `butteraugli_loop.rs::run_buttloop`.

### 4. `EncoderStrategy::Libjxl` invariant

Even with `with_cvvdp_loop(Some(true))`, `EncoderStrategy::Libjxl` MUST use butteraugli (strict cjxl-parity). Add the resolver:

```rust
impl LossyConfig {
    pub(crate) fn resolve_cvvdp_loop(&self) -> bool {
        if matches!(self.strategy, EncoderStrategy::Libjxl) { return false; }
        self.cvvdp_loop.unwrap_or(false)
    }
}
```

This MUST be respected by every site that constructs a backend. Add a unit test `test_libjxl_strategy_forces_butteraugli_even_with_cvvdp_loop`.

## Acceptance gates

- (a) `cargo build -p jxl-encoder` PASS on default features.
- (b) `cargo build -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel"` PASS.
- (c) `cargo build -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop gpu-butteraugli parallel"` PASS (mixed feature combo).
- (d) `cargo test -p jxl-encoder --lib` PASS.
- (e) `cargo test -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop parallel" --lib` PASS.
- (f) **Hash-locks 36/36 BYTE-IDENTICAL at default features** — cvvdp opt-in must not regress the default path.
- (g) **`strategy_libjxl_byte_lock` 4/4 PASS even with cvvdp-loop feature ENABLED at compile time** — Libjxl strategy must structurally bypass cvvdp.
- (h) `divergence_table_drift` PASS — add the new opt-in row before running.
- (i) New integration test `tests/cvvdp_backend_smoke.rs`: encode 64×64 d=1.0 with `cvvdp_loop=Some(true)` if `cvvdp-loop` feature enabled, decode via jxl-rs + djxl, assert pixel sanity. Skip cleanly when feature off.
- (j) `cargo clippy --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" -- -D warnings` PASS.
- (k) `cargo fmt --check` PASS.
- (l) `docs/LIBJXL_DIVERGENCES.md` Section E gets a new row for `with_cvvdp_loop`.

## Chunk N+1 plan inline

Phase 4 brief (full buttloop wiring) lands as a separate agent right after this. Document in commit:

- **Phase 4 (next chunk)**: rename `butteraugli_loop.rs` → `perceptual_loop.rs`. Plumb `cvvdp_loop` field through `run_buttloop` body. Add per-distance JOD target table at `vardct/cvvdp_targets.rs`. Multi-decoder smoke on 5 cells with `cvvdp_loop=Some(true)`.

## DO NOT

- DO NOT cite "FMA precision" for ANY byte drift. cvvdp opt-in OFF must be byte-identical with HEAD.
- DO NOT wire cvvdp into the buttloop body — that's Phase 4.
- DO NOT add `EffortProfile`-level dispatch — Phase 4 follow-on.
- DO NOT use unsafe.
- DO NOT push to a new branch — stack on `cvvdp-fork-rfc`.

## Honest-stop

If `cvvdp_gpu::CvvdpOpaque::warm_reference_from_linear_planes` doesn't exist (Agent B chose a different method name): adapt to whatever the actual API surface is. If the API is structurally incompatible (e.g. cvvdp-gpu can't produce a tight `Vec<f32>` diffmap): SURFACE the divergence as a memo + ship a stubbed `unimplemented!()` placeholder, file as next chunk. Don't paper over.

## Progress + final report

Memo at `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase3_backend_impl_<status>_2026-MM-DD.md`. Update MEMORY.md. Return summary ≤300 words.
