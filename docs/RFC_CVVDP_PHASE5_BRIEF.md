# Phase 5 agent brief: cvvdp-cpu integration into encoder

**Status**: BLOCKED on Phase 4 (cvvdp-loop body wiring). Fire once
Phase 4's commit lands on `cvvdp-fork-rfc` with
`run_buttloop` (or its renamed `run_perceptual_loop`) calling
`construct_backend()` with the resolved `cvvdp_loop` flag AND the
buttloop body successfully terminates the encode with cvvdp-loop=ON
on a 5-cell smoke (tests/cvvdp_loop_smoke.rs).

Brief is self-contained; paste-and-go.

---

You are integrating Agent A's `cvvdp-cpu` crate (already shipped to
`zenmetrics/master`) as a new `CpuCvvdpBackend` impl of jxl-encoder's
`PerceptualBackend` trait. Mirror the existing
`CpuButteraugliBackend` / `GpuButteraugliBackend` split for
butteraugli, but for cvvdp.

This is Phase 5 of the cvvdp fork. See
`docs/RFC_CVVDP_FORK.md` §2, §3 (diffmap invariants), §4 Phase 5.

## Pre-flight

1. **Phase 4 landed**: `grep "run_perceptual_loop\|run_buttloop" ~/work/zen/jxl-encoder/jxl-encoder/src/vardct/*.rs` shows the
   buttloop entry point and the cvvdp dispatch.
2. **cvvdp-cpu v0.0.1 on origin/master**: `ls ~/work/zen/zenmetrics/crates/cvvdp-cpu/Cargo.toml`. If missing: `jj git fetch` in
   the zenmetrics workspace.
3. **Tracking TSV has `C_GPU` rows from Phase 4**: `awk -F'\t' '$5=="C_GPU"' ~/work/zen/jxl-encoder/benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv | wc -l` ≥ 20.

If any pre-flight fails: STOP.

## Read FIRST

1. `/home/lilith/.claude/CLAUDE.md`.
2. `~/work/zen/jxl-encoder/CLAUDE.md`.
3. `~/work/claudehints/topics/multi-agent-excellence.md`.
4. `~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` §2, §3, §4 (Phase 5
   subsection, ~50 lines).
5. **Agent A's cvvdp-cpu memo**: `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_cpu_port_shipped_2026-05-24.md` — covers
   the public API surface, the 5-PRACTICAL-invariant set, the 4.4×-
   off-floor honest-stop, and the queued SIMD chunks.
6. **cvvdp-cpu source**: `~/work/zen/zenmetrics/crates/cvvdp-cpu/src/{lib.rs, pipeline.rs, diffmap.rs}`. Read the public types you'll
   wrap.
7. **Phase 3's `cvvdp_backend.rs`** (the GPU impl + `construct_backend`
   routing). You're adding to this file.

## Workspace claim

```bash
cd ~/work/zen/jxl-encoder
jj workspace add ~/work/zen/jxl-encoder--cvvdp-cpu-integration --revision cvvdp-fork-rfc@origin
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && printf '%s agent-cvvdp-cpu-integration bootstrapping\n' "$(cat /tmp/ts)" > ~/work/zen/jxl-encoder--cvvdp-cpu-integration/.workongoing
cd ~/work/zen/jxl-encoder--cvvdp-cpu-integration
```

Refresh marker every 2 min. **Commit first, validate after** (Agent C
lesson — uncommitted edits don't survive `jj workspace update-stale`).

## File ownership

**YOURS:**
- `jxl-encoder/src/vardct/cvvdp_backend.rs` — add `CpuCvvdpBackend`
  struct + impl alongside the existing `GpuCvvdpBackend`. Add a `cpu`
  sub-module if file size becomes unwieldy (>1500 lines).
- `jxl-encoder/src/vardct/perceptual_backend.rs` — extend
  `construct_backend()` dispatch to select CPU vs GPU cvvdp.
- `jxl-encoder/Cargo.toml` — new feature `cvvdp-loop-cpu` (depends on
  `cvvdp-loop`); optional dep `cvvdp-cpu = { path = "..." }` (use
  `[patch.crates-io]` mirroring how `cvvdp-gpu` is wired by D's harness)
  OR pull via the published zenmetrics version once Agent A publishes.
- `Cargo.toml` (workspace) — `[patch.crates-io]` entry for cvvdp-cpu
  if needed.
- `jxl-encoder/src/api.rs` — extend `LossyConfig::resolve_cvvdp_loop`
  (already exists from Phase 3) with a sub-resolver:
  `fn cvvdp_use_cpu(&self) -> bool` returning whether to prefer CPU
  over GPU when cvvdp-loop is active.
- `jxl-encoder/tests/cvvdp_cpu_backend_smoke.rs` — integration smoke
  for the CPU backend.
- `benchmarks/cvvdp_cpu_vs_gpu_buttloop_<YYYY-MM-DD>.{tsv,meta}` —
  10-cell A/B bench: cvvdp-loop=ON with cvvdp_use_cpu=true vs false.
- `docs/LIBJXL_DIVERGENCES.md` Section E — add row for the new
  `cvvdp-loop-cpu` feature.

**DO NOT TOUCH:**
- `vardct/butteraugli_loop.rs` / `vardct/perceptual_loop.rs` body
  (Phase 4 ships that; this chunk just adds a new backend impl).
- Other agents' workspace files.
- The `EncoderStrategy::Libjxl` invariant.
- `crates/cvvdp-cpu/*` (Agent A's deliverable; READ ONLY).

## Deliverable

### Step 1 — Backend selection policy

When `cvvdp-loop` is opted in, the encoder needs to decide CPU vs GPU.
Policy (mirrors the butteraugli pattern):

| `cvvdp-loop` | `cvvdp-loop-cpu` | `cvvdp_use_cpu` field | Backend selected |
|---           |---               |---                    |---               |
| OFF          | any              | any                   | butteraugli (no change) |
| ON           | NOT compiled     | any                   | GPU cvvdp (Phase 3 impl) |
| ON           | compiled         | None (default)        | **GPU cvvdp** (perf primacy — Agent A's CPU is 4.4× off floor)|
| ON           | compiled         | Some(false)           | GPU cvvdp |
| ON           | compiled         | Some(true)            | CPU cvvdp |

**Rationale for "GPU default when both compiled"**: Agent A measured
cvvdp-cpu at 222 ms / 1024² (8 threads), vs cvvdp-gpu at ~17-22 ns/px
warm-ref ≈ 18-23 ms / 1024² — ~10× faster. The CPU backend exists for
hosts without CUDA AND for callers who explicitly opt in for
reproducibility (CPU is deterministic to 1e-4 JOD vs pycvvdp v0.5.4
goldens; GPU has the ~1e-7 reduction-order variance documented in
W44-RECON-DEEP/A7). Default to GPU; respect explicit opt-in.

If `cvvdp-loop-cpu` is compiled but the host lacks CUDA, fall back to
CPU cvvdp (not butteraugli) — that's the whole point of the feature.

### Step 2 — `CpuCvvdpBackend` impl

```rust
#[cfg(feature = "cvvdp-loop-cpu")]
pub(crate) struct CpuCvvdpBackend {
    inner: cvvdp_cpu::Cvvdp,
    width: u32,
    height: u32,
    reference_warmed: bool,
    // No "shadow" mechanism (B5b detector was GPU-only).
}

#[cfg(feature = "cvvdp-loop-cpu")]
impl CpuCvvdpBackend {
    pub(crate) fn try_new(width: u32, height: u32, /* params */) -> Option<Self> {
        // catch_unwind is unnecessary for CPU (no driver init).
        // Just construct.
        let inner = cvvdp_cpu::Cvvdp::new(width, height, params).ok()?;
        Some(Self { inner, width, height, reference_warmed: false })
    }
}

#[cfg(feature = "cvvdp-loop-cpu")]
impl PerceptualBackend for CpuCvvdpBackend {
    fn name(&self) -> &'static str { "cvvdp-cpu" }

    fn set_reference(&mut self, /* args */) -> Result<()> {
        self.inner.warm_reference_from_linear_planes(ref_r, ref_g, ref_b)?;
        self.reference_warmed = true;
        Ok(())
    }

    fn compare_with_reference(
        &mut self,
        dist_r: &[f32], dist_g: &[f32], dist_b: &[f32],
        padded_width: usize, width: usize, height: usize,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<BackendCompareResult> {
        // Strip padding if padded_width != width (mirror GpuButteraugliBackend).
        let jod = self.inner.score_from_linear_planes_with_warm_ref_diffmap(
            tight_dist_r, tight_dist_g, tight_dist_b, diffmap_out,
        )?;
        let score = (10.0 - jod as f64).clamp(0.0, 10.0);
        Ok(BackendCompareResult { score })
    }
}
```

Mirror the existing `GpuCvvdpBackend` for stride handling, error
mapping, etc. **The JOD → score mapping (10 - jod) MUST be identical
between CPU and GPU cvvdp backends** — the buttloop's termination
target table treats them as the same metric.

### Step 3 — Update `construct_backend`

Extend Phase 3's dispatch tree. Given resolved `(use_cvvdp, use_cpu)`:

```rust
match (use_cvvdp, use_cpu, cfg!(feature = "cvvdp-loop"), cfg!(feature = "cvvdp-loop-cpu")) {
    (true, true,  _, true)  => CpuCvvdpBackend::try_new(...).map(box_it)
                                  .or_else(|| GpuCvvdpBackend::try_new(...).map(box_it))
                                  .or_else(|| CpuButteraugliBackend::new(cpu_params).into()),
    (true, false, true, _)  => GpuCvvdpBackend::try_new(...).map(box_it)
                                  .or_else(|| { try cvvdp-cpu if feature compiled }),
    (true, _,    false, _)  => /* cvvdp requested but feature off; warn + fall back to butteraugli */,
    (false, _, _, _)        => /* existing butteraugli dispatch */,
}
```

Fallback chain when GPU cvvdp init fails: try CPU cvvdp if compiled,
else butteraugli. ONE-shot `eprintln!` warning per fallback.

### Step 4 — Smoke test

`tests/cvvdp_cpu_backend_smoke.rs`:
- 5 cells, `LossyConfig::with_cvvdp_loop(Some(true)).with_cvvdp_use_cpu(Some(true))`.
- Encode → jxl-rs roundtrip → pixel sanity.
- One cell at `EncoderStrategy::Libjxl` → assert byte-identical to
  default Libjxl encode regardless of cvvdp flags.
- `#[ignore]` (needs cvvdp-loop-cpu feature compiled in).

### Step 5 — A/B bench

`benchmarks/cvvdp_cpu_vs_gpu_buttloop_<YYYY-MM-DD>.tsv`:
- 10 cells across 3 corpora (CID22 + GB82-SC + tiny synthetic).
- Encode each with cvvdp-loop=ON + cvvdp_use_cpu=true vs false.
- Columns: image, distance, effort, backend, bytes, wall_ms, score_butter_cpu,
  score_cvvdp_gpu, ssim2.
- Goal: confirm bytes/score similar (within 1% bytes, within ±0.05
  JOD); wall_ms shows the expected ~10× GPU advantage.

## Acceptance gates

- (a) `cargo build -p jxl-encoder` default — PASS.
- (b) `cargo build -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel"` — PASS.
- (c) `cargo test -p jxl-encoder --lib` PASS.
- (d) `cargo test -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel" --lib` PASS.
- (e) Hash-locks 36/36 BYTE-IDENTICAL at default features. ZERO regen.
- (f) `strategy_libjxl_byte_lock` 4/4 BYTE-IDENTICAL with all cvvdp
  features compiled.
- (g) `divergence_table_drift` PASS — Section E row added.
- (h) `cargo clippy --features "__expert butteraugli-loop cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel" -- -D warnings` PASS.
- (i) `cargo fmt --check` PASS.
- (j) Multi-decoder smoke PASS.
- (k) Bench TSV committed; meta documents commit + hostname + feature
  flags + cvvdp-cpu commit SHA.
- (l) Push to `cvvdp-fork-rfc`. Stack on Phase 4's commit.

## Chunk N+1 plan inline

- **Phase 6**: full tracking sweep + Pareto decision. Run the
  cvvdp_track_baseline harness with each of (B, B_GPU, C_GPU, C_CPU)
  backends across the 1,134-cell corpus. Apply RFC §5.4 decision rule.
  Update CHANGELOG. If `C_GPU` strictly Pareto-dominates `B`: propose
  flipping default. Otherwise: ship opt-in only.
- **Phase 7** (only if Phase 6 says ship default): default-flip commit,
  documentation cascade, CHANGELOG ENTRY.

## DO NOT

- DO NOT default-flip to CPU when GPU is available — Agent A's perf
  honest-stop measured 10× GPU advantage.
- DO NOT cite "FMA precision" for any byte drift at default. The opt-
  in OFF path must remain byte-identical.
- DO NOT skip the Libjxl invariant test.
- DO NOT use unsafe.
- DO NOT modify cvvdp-cpu source (Agent A's deliverable). If the API
  surface needs extension, file as cvvdp-cpu v0.1.0 follow-on.

## Honest-stop discipline

- If `cvvdp-cpu` is missing the `warm_reference_from_linear_planes` /
  `score_from_linear_planes_with_warm_ref_diffmap` methods Agent B
  added to `cvvdp-gpu`: file a follow-on for cvvdp-cpu (Phase 5b) and
  ship a partial impl using whatever Agent A's API surface IS. Don't
  paper over a missing API; adapt to it.
- If `cvvdp-cpu`'s 5 PRACTICAL invariants don't match Phase 1's GPU
  impl in shape (e.g. the GPU emits a `[height][width]` diffmap and
  CPU emits `[width][height]` — silly but possible): align them. The
  test `cvvdp_cpu_vs_gpu_buttloop_<YYYY-MM-DD>.tsv` is the gate.

## Progress + final report

Memo at `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase5_cpu_integration_<status>_2026-MM-DD.md`.
Update MEMORY.md. Return ≤300-word summary.
Cleanup-on-merge: `jj workspace forget` + `rm -rf`. **MANDATORY** —
don't leave orphan sibling.
