# Phase 4 agent brief: cvvdp-loop body wiring

**Status**: BLOCKED on Phase 3 (CvvdpBackend impl). Fire once
Phase 3's commit lands on `cvvdp-fork-rfc` with `CvvdpBackend`
constructed by `construct_backend()` at compile time AND
`LossyConfig::with_cvvdp_loop(Option<bool>)` field plumbed.

Brief is self-contained; the launcher pastes-and-goes.

---

You are wiring cvvdp into the actual buttloop body. This is Phase 4 of
the cvvdp fork — the FINAL chunk before the tracking benchmark (Phase
6) can produce real `C_GPU` rows. See `docs/RFC_CVVDP_FORK.md` §2,
§3 (Five PRACTICAL diffmap invariants — Agent A 2026-05-24 honest-
stop), and §4 Phase 4.

## Pre-flight (verify before starting)

1. **Phase 3 landed**: `grep "GpuCvvdpBackend" ~/work/zen/jxl-encoder/jxl-encoder/src/vardct/cvvdp_backend.rs`
   returns the impl block.
2. `LossyConfig::with_cvvdp_loop` builder + `cvvdp_loop: Option<bool>`
   field exist in `api.rs`.
3. `LossyConfig::resolve_cvvdp_loop(&self) -> bool` resolver exists +
   short-circuits to `false` for `EncoderStrategy::Libjxl`.
4. `Cargo.toml` has the `cvvdp-loop` feature with proper optional dep
   guards.

If any pre-flight fails: STOP. Don't paper over.

## Read FIRST

1. `/home/lilith/.claude/CLAUDE.md` — JJ marker, divergence-table rule.
2. `~/work/zen/jxl-encoder/CLAUDE.md` — encoder rules, hash-lock +
   drift mandate, multi-decoder roundtrip rule.
3. `~/work/claudehints/topics/multi-agent-excellence.md`.
4. `~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` (re-read whole doc;
   §3 was substantively updated 2026-05-24 to reflect Agent A's
   honest-stopped invariant set).
5. `~/work/zen/jxl-encoder/docs/CVVDP_W44_GATE_TRANSFER.md` — your
   gate-transfer scoping reference.
6. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/butteraugli_loop.rs`
   (3800+ lines — the file you're renaming + threading cvvdp through).
7. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/perceptual_backend.rs`
   (Phase 2 + Phase 3 result — your construction site).
8. `~/work/zen/jxl-encoder/jxl-encoder/src/vardct/cvvdp_backend.rs`
   (Phase 3 result).
9. `~/work/zen/jxl-encoder/benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
   — Agent D's baseline rows. Use the bytes/wall/score columns to seed
   the JOD calibration table (Phase 4 step 3 below).

## Workspace claim

```bash
cd ~/work/zen/jxl-encoder
jj workspace add ~/work/zen/jxl-encoder--cvvdp-loop-wiring --revision cvvdp-fork-rfc@origin
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && printf '%s agent-cvvdp-loop-wiring bootstrapping\n' "$(cat /tmp/ts)" > ~/work/zen/jxl-encoder--cvvdp-loop-wiring/.workongoing
cd ~/work/zen/jxl-encoder--cvvdp-loop-wiring
```

Refresh marker every 2 min.

**Lesson from Agent C**: `jj describe` locks the snapshot. If you
encounter sibling-workspace races (other agents push to
cvvdp-fork-rfc while you're working), commit FIRST with `jj describe`
before validating. Uncommitted edits don't survive `jj workspace
update-stale` cleanly; described commits do (you may need to rebase,
but the content is preserved).

## File ownership

**YOURS:**
- `jxl-encoder/src/vardct/butteraugli_loop.rs` — rename to
  `perceptual_loop.rs`. Update content where it touches the backend
  dispatch (you do NOT need to rename `CpuButteraugliBackend` /
  `GpuButteraugliBackend` impls — those are butteraugli-specific).
- `jxl-encoder/src/vardct/mod.rs` — update module decl.
- `jxl-encoder/src/vardct/cvvdp_targets.rs` — NEW. JOD-direction
  target table per distance (see step 3 below).
- `jxl-encoder/src/api.rs` — extend `LossyConfig::resolve_cvvdp_loop`
  plumbing into the 3 encode entry sites (still / streaming /
  animation) so the resolver value reaches the buttloop body.
- `jxl-encoder/Cargo.toml` — if any feature wiring needs adjusting.
- `jxl-encoder/tests/cvvdp_loop_smoke.rs` — integration test.
- `docs/LIBJXL_DIVERGENCES.md` Section E — add a row for the new opt-in
  behaviour. NOT Section A (this isn't a gate change at default).
- `benchmarks/cvvdp_loop_smoke_<YYYY-MM-DD>.{tsv,meta}` — 20-cell
  bench of `cvvdp_loop=Some(true)` vs default.

**DO NOT TOUCH:**
- `vardct/perceptual_backend.rs` non-trivially. You may add a one-line
  helper that takes the resolved cvvdp flag, but don't refactor the
  backend dispatch.
- `vardct/cvvdp_backend.rs` (Phase 3's deliverable).
- Other agents' workspace files.
- The `EncoderStrategy::Libjxl` invariant — `resolve_cvvdp_loop()` must
  return `false` for Libjxl regardless of the field. Test:
  `cvvdp_loop = Some(true) + strategy = Libjxl → butteraugli, byte-
  identical to default Libjxl encode`.

## Deliverable

### Step 1 — Rename `butteraugli_loop.rs` → `perceptual_loop.rs`

`jj mv` (or `git mv` with jj watching). Update `vardct/mod.rs`.
Update every import site (grep for `butteraugli_loop`). The file's
internal content stays butteraugli-heavy in this chunk — only the file
name + module name change.

Function names like `run_buttloop` stay the same — they're now
"perceptual loop" but the historical name is load-bearing in commit
messages and W44 docs. Optionally add an alias `run_perceptual_loop` →
`run_buttloop` as a comment-only marker; don't add a second symbol.

### Step 2 — Plumb `cvvdp_loop` flag through the 3 propagation sites

Mirror the existing `gpu_butteraugli` plumbing in `api.rs`. Three
sites:

1. Still-image `encode_inner` → `VarDctEncoder.cvvdp_loop` field.
2. Streaming `LossyEncoder::new` → same.
3. Animation `encode_animation_lossy` per-frame setup → same.

The buttloop body's `construct_backend()` call already takes a
`gpu_requested` bool (Phase 3 added a second `cvvdp_requested` slot —
verify this in Phase 3's output). Pass the resolved value.

### Step 3 — `vardct/cvvdp_targets.rs` — JOD calibration table

Seed from Agent D's tracking TSV. Method:

1. For each distance in `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0}`, read
   the median `score_cvvdp_gpu` across cells at that distance from
   the TSV. That's the "what JOD does the existing butteraugli buttloop
   converge to at this distance?"
2. Add a small safety margin (1.05× the median in score-direction =
   slightly stricter target, since cvvdp-driven buttloop should at
   minimum match butteraugli-driven on the same metric).
3. Encode the table as a `static [(f32, f32); 7] = [(distance, target)]`
   in `vardct/cvvdp_targets.rs` with linear interpolation between
   points and clamp-outside-band.
4. Conversion: `target_jod = 10.0 - (target value)` so that
   `BackendCompareResult::score = 10 - jod` (butteraugli direction)
   compares against the lookup-table value directly.

Document the seed methodology in the file's top docstring with a
pointer at the TSV + Agent D's commit SHA.

### Step 4 — Pre-buttloop dispatch

In `run_buttloop` (or whatever the entry point is named after the
rename), at the existing `construct_backend()` call: pass both
`gpu_butteraugli` AND `cvvdp_loop` flags. The dispatch logic in
`perceptual_backend.rs::construct_backend` is Phase 3's deliverable;
you're just feeding it the resolved flag.

### Step 5 — Multi-decoder smoke

`jxl-encoder/tests/cvvdp_loop_smoke.rs`:
- 5 cells (mix of CID22 + GB82-SC + a tiny synthetic): with
  `cvvdp_loop=Some(true)`, encode → decode via jxl-rs → assert pixel
  sanity (decoded width × height matches, no NaN/Inf, byte length > 0).
- Same cells with `cvvdp_loop=Some(false)` → assert byte-identical to
  pre-Phase-4 main on the same commit (sanity that the flag's "off"
  path is a no-op).
- `EncoderStrategy::Libjxl` with `cvvdp_loop=Some(true)` → assert
  byte-identical to default Libjxl encode (the invariant test).
- `#[ignore]` so this test doesn't run on every CI build (needs CUDA
  for the cvvdp-gpu backend). Document the run command in a comment.

## Acceptance gates

- (a) `cargo build -p jxl-encoder` default — PASS.
- (b) `cargo build -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel"` — PASS.
- (c) `cargo test -p jxl-encoder --lib` — PASS.
- (d) `cargo test -p jxl-encoder --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" --lib` — PASS.
- (e) Hash-locks 36/36 BYTE-IDENTICAL at default. ZERO regen.
- (f) `strategy_libjxl_byte_lock` 4/4 BYTE-IDENTICAL with
  `--features "__expert cvvdp-loop"` — Libjxl invariant.
- (g) `divergence_table_drift` PASS — Section E row added.
- (h) `cargo clippy --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" -- -D warnings` PASS.
- (i) `cargo fmt --check` PASS.
- (j) Multi-decoder smoke `tests/cvvdp_loop_smoke.rs` — PASS when run
  with `cargo test --features "__expert butteraugli-loop cvvdp-loop ssim2-loop parallel" -- --ignored cvvdp_loop_smoke`.
- (k) Tracking bench TSV `benchmarks/cvvdp_loop_smoke_<YYYY-MM-DD>.tsv`
  — 20 cells, append to the master tracking TSV with `backend=C_GPU`
  rows (per RFC §5.3 schema).
- (l) `docs/LIBJXL_DIVERGENCES.md` Section E updated.
- (m) Push to `cvvdp-fork-rfc` bookmark. Stack on Phase 3's commit.

## Chunk N+1 plan inline

In your final commit + memo:
- **Phase 5**: cvvdp-cpu integration. Add `CpuCvvdpBackend` impl
  wrapping `cvvdp_cpu::Cvvdp`. Feature gate `cvvdp-loop-cpu`. Default
  cvvdp backend stays GPU when feature combo `cvvdp-loop + cvvdp-loop-cpu`
  with both compiled — prefer CPU. Performance budget per RFC §4
  Phase 5.
- **Phase 6**: tracking sweep + decision. Run full 1,134-cell sweep
  with each backend (B / B_GPU / C_GPU / C_CPU). Apply RFC §5.4
  decision rule. Update CHANGELOG.

## DO NOT

- DO NOT cite "FMA precision" for any byte drift at default (cvvdp opt-
  in OFF MUST be byte-identical).
- DO NOT skip the Libjxl invariant test — the strategy must structurally
  bypass cvvdp.
- DO NOT recalibrate any W44 cost-model gates in this chunk (per
  CVVDP_W44_GATE_TRANSFER.md, gate recalibration is Phase 6+ work
  driven by tracking benchmark data).
- DO NOT use unsafe.
- DO NOT change `target_distance` semantics — the JOD calibration table
  reads butteraugli-direction distance as input.

## Honest-stop discipline

- If the cvvdp diffmap shape produces structurally different per-block
  median/MAD adjustments such that the buttloop diverges instead of
  converging: STOP. Document the convergence behaviour. File as
  "Phase 4b: tune buttloop convergence rule for cvvdp signal range".
  Ship a `cvvdp_loop_smoke.rs` that exercises the OFF path only +
  documents the ON-path issue.
- If Agent D's tracking TSV has < 100 complete cells when you start:
  STOP. Phase 4 needs the calibration data. Wait for D, or shrink the
  calibration to a hand-picked 7-cell anchor set + document.

## Progress + final report

Memo at `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase4_loop_wiring_<status>_2026-MM-DD.md`.
Update MEMORY.md. Return ≤300-word summary. Cleanup-on-merge:
`jj workspace forget` + `rm -rf` your sibling.
