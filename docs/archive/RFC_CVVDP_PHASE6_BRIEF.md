# Phase 6 agent brief: tracking sweep + Pareto decision

**Status**: BLOCKED on Phase 5 (cvvdp-cpu integration). Fire once
Phase 5 lands with `CpuCvvdpBackend` filled in + `cvvdp-loop-cpu`
feature working + smoke roundtrip 5/5 PASS.

This is the FINAL phase of the cvvdp fork. Output: a decision +
CHANGELOG entry. If cvvdp Pareto-dominates butteraugli on enough
corpora, propose default-flip (Phase 7). Otherwise: ship as opt-in
only + document trade-offs.

---

You are running the full tracking sweep + applying the decision rule
from `docs/RFC_CVVDP_FORK.md` §5.4. See also §5.1/§5.2/§5.3 for
methodology details.

## Read FIRST

1. `/home/lilith/.claude/CLAUDE.md` — sweep discipline, NEVER
   fabricate / estimate numbers, bench-data-in-repo rule.
2. `~/work/zen/jxl-encoder/CLAUDE.md` — encoder rules.
3. `~/work/claudehints/topics/multi-agent-excellence.md`.
4. `~/work/zen/jxl-encoder/docs/RFC_CVVDP_FORK.md` (whole doc; §5
   in particular).
5. `~/work/zen/jxl-encoder/docs/CVVDP_W44_GATE_TRANSFER.md` — relevant
   when interpreting cvvdp-vs-butteraugli per-cell deltas.
6. **Agent D's harness** at `jxl-encoder/examples/cvvdp_track_baseline.rs`
   — the foundation you'll extend.
7. **Existing tracking TSV** at `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`
   — Agent D's 1,131 `backend=B` rows. APPEND-ONLY; never edit.
8. Phase 4 + 5 commits on `cvvdp-fork-rfc` — confirm cvvdp-loop +
   cvvdp-loop-cpu features fully working.

## Workspace claim

```bash
cd ~/work/zen/jxl-encoder
jj workspace add ~/work/zen/jxl-encoder--cvvdp-tracking-sweep --revision cvvdp-fork-rfc@origin
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && printf '%s agent-cvvdp-tracking-sweep bootstrapping\n' "$(cat /tmp/ts)" > ~/work/zen/jxl-encoder--cvvdp-tracking-sweep/.workongoing
cd ~/work/zen/jxl-encoder--cvvdp-tracking-sweep
```

Refresh marker every 2 min (more frequent — the encode loop runs for
hours).

## File ownership

**YOURS:**
- `jxl-encoder/examples/cvvdp_track_baseline.rs` — extend Agent D's
  harness with `--backend {B|B_GPU|C_GPU|C_CPU}` CLI flag. Default
  stays `B` (so re-running for B doesn't double-write). Document
  what features need to be compiled in for each backend.
- `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` — APPEND
  rows ONLY. New backend values are `B_GPU`, `C_GPU`, `C_CPU`.
- `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.meta` — update
  with the new backend coverage + commit SHA notes.
- NEW: `scripts/cvvdp_pareto_analysis.py` — Pareto frontier analysis
  + decision script.
- NEW: `scripts/cvvdp_pareto_analysis_<YYYY-MM-DD>.tsv` and `.meta` —
  the analysis output (Pareto fronts per corpus, per-cell deltas,
  decision verdict).
- NEW: `docs/CVVDP_FORK_DECISION.md` — the final decision memo:
  conclusion, supporting data, recommended next step (Phase 7 default-
  flip OR opt-in-only OR revert).

**DO NOT TOUCH:**
- Any encoder source (`jxl-encoder/src/`). You're sweeping the
  shipped encoder.
- Existing TSV rows from Agent D — append only.
- Other agents' workspace files.
- `crates/cvvdp-cpu` / `crates/cvvdp-gpu` source.

## Methodology (RFC §5.2, §5.3, §5.4)

### Step 1 — Run 3 backend sweeps (each ~1,134 cells, ~20 min on a
CUDA host)

For each `backend ∈ {B_GPU, C_GPU, C_CPU}`:

```bash
# B_GPU: butteraugli-driven encoder, but use GPU butteraugli inside the buttloop
cargo run --release --features '__expert butteraugli-loop gpu-butteraugli ssim2-loop parallel' \
  --example cvvdp_track_baseline -- \
  --output benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv \
  --backend B_GPU \
  --commit <current-sha>

# C_GPU: cvvdp-driven encoder, GPU cvvdp backend in the buttloop
cargo run --release --features '__expert butteraugli-loop cvvdp-loop ssim2-loop parallel' \
  --example cvvdp_track_baseline -- \
  --output benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv \
  --backend C_GPU \
  --commit <current-sha>

# C_CPU: cvvdp-driven encoder, CPU cvvdp backend in the buttloop
cargo run --release --features '__expert butteraugli-loop cvvdp-loop cvvdp-loop-cpu ssim2-loop parallel' \
  --example cvvdp_track_baseline -- \
  --output benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv \
  --backend C_CPU \
  --commit <current-sha>
```

Each cell scores 4 metrics on the decoded output:
`score_butter_cpu`, `score_butter_gpu`, `score_cvvdp_gpu`, `score_ssim2`.

**Resumability**: the harness skips populated cells. If a sweep crashes
midway, just re-run. Each backend sweep is ~20 min on the rig
(`lilith`, RTX 5070). C_CPU may take longer due to Agent A's perf
honest-stop (4.4× off floor at 1024² ≈ 222 ms/iter); budget 60-90 min
for it.

**Total wall**: ~2-3 hours of compute. Commit-push every 200 cells.

### Step 2 — Pareto frontier analysis (`scripts/cvvdp_pareto_analysis.py`)

For each (corpus, metric) tuple ∈ {CID22, GB82-SC, W44-S1-extras} ×
{butter_cpu, cvvdp_gpu, ssim2}:

- Compute the Pareto frontier on (bytes, metric) per cell across the 4
  backends.
- Count Pareto wins per backend (a cell is a "Pareto win" for backend X
  if X is on the frontier AND no other backend strictly dominates X at
  that bytes-or-metric).
- Surface cells where rankings ACROSS metrics disagree (cvvdp says A
  wins, butteraugli says B wins) — these are the "where the metric
  matters" cells.
- Report wall_ms p50 and p95 per backend (Phase 5 already measured
  cvvdp-cpu's 10× wall overhead — confirm at scale).

### Step 3 — Decision rule (RFC §5.4)

Per the original RFC:

> Ship cvvdp default-ON only if it strictly Pareto-dominates
> butteraugli on at least one corpus + comes within 5% on the others.
>
> Otherwise: ship as opt-in flag; document trade-offs in CHANGELOG.

Apply the rule. Report the conclusion in `docs/CVVDP_FORK_DECISION.md`
with supporting data:

- Per-corpus win-loss table (4 backends × 3 metrics)
- Per-distance summary (cvvdp's 1.05×-tighter target table generates
  larger files at the same distance per Phase 4's bench finding —
  quantify how much vs butteraugli)
- Wall-time comparison (cvvdp-cpu is the slow path)
- Multi-decoder roundtrip status (any cells that produced cvvdp output
  that any decoder couldn't decode → IMMEDIATE STOP, find structural
  cause)
- Recommended next step:
  - **Default-flip** (Phase 7): if cvvdp Pareto-dominates on ≥1 corpus
    + within 5% on others. Phase 7 = flip default + CHANGELOG entry +
    docs cascade.
  - **Opt-in only**: ship `--features cvvdp-loop` as a documented
    opt-in. Default stays butteraugli. CHANGELOG entry explains the
    trade-off.
  - **Revert** (unlikely): if cvvdp produces broken bitstreams or
    catastrophically worse output on any corpus, document + revert
    the encoder-side wiring (Phase 3-5 commits). Keep cvvdp-cpu +
    cvvdp-gpu diffmap APIs as standalone deliverables.

## Acceptance gates

- (a) ≥1,000 of 1,134 cells per backend populated for B_GPU + C_GPU +
  C_CPU. (Same OOM cells as Agent D may fail; that's OK if documented.)
- (b) Pareto analysis script committed + runnable + produces TSV + meta.
- (c) Decision memo committed with explicit recommendation tied to
  supporting bench numbers.
- (d) `cargo fmt --check` PASS on any new Rust code.
- (e) Multi-decoder roundtrip: spot-check 10 cvvdp-encoded outputs via
  jxl-rs + djxl. If any cell fails to decode under any decoder: STOP +
  surface as honest-stop.
- (f) Bench data + decision memo all in `benchmarks/` and `docs/` —
  committed to repo per multi-agent excellence rule #7.
- (g) Push to `cvvdp-fork-rfc`. Stack on Phase 5's commit.
- (h) Memo at `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase6_tracking_sweep_<status>_2026-MM-DD.md`.
- (i) Cleanup-on-merge: `jj workspace forget cvvdp-tracking-sweep` +
  `rm -rf`.

## Chunk N+1 plan inline

Per the decision (in your final memo + commit):

- **Phase 7 (only if decision = default-flip)**: spawn Phase 7 agent
  with a brief that includes (1) the decision memo, (2) the specific
  feature flag flip + default plumbing, (3) CHANGELOG entry, (4)
  hash-lock regen (this WILL regen 36/36 since the default path
  changes), (5) docs cascade. Default-flip Phase 7 brief: file as
  `docs/RFC_CVVDP_PHASE7_DEFAULTFLIP_BRIEF.md`.
- **Phase 7-opt-in**: smaller chunk — update README + CHANGELOG +
  feature-flag docs. No source changes. File as
  `docs/RFC_CVVDP_PHASE7_OPTIN_BRIEF.md`.
- **W44 gate re-calibration**: per `CVVDP_W44_GATE_TRANSFER.md`,
  several W44 gates (W44-105/W44-109/W44-117 cluster) were calibrated
  against butteraugli's measurement bias. If cvvdp is the new default,
  those gates need re-validation under cvvdp. File as
  `docs/RFC_CVVDP_PHASE8_W44_RECAL.md`.

## DO NOT

- DO NOT fabricate / estimate any cell values. If a cell can't be
  measured (OOM, decoder failure), leave `NA` with `notes` explanation.
- DO NOT cite "FMA precision" for any cross-backend score divergence.
  Butteraugli CPU vs GPU is documented at ~3% drift; cvvdp CPU vs GPU
  is documented at 1e-6 pointwise diffmap parity (Agent A + B
  cross-aligned).
- DO NOT skip the multi-decoder roundtrip check. ANY decoder failure
  is a structural bug, not a measurement noise floor.
- DO NOT make the default-flip decision without ≥1,000 cells of
  C_GPU data. Sparse coverage = sparse confidence.
- DO NOT touch encoder source. The encoder is shipped; you're measuring
  it.
- DO NOT push to a new branch — stack on `cvvdp-fork-rfc`.

## Honest-stop discipline

- If C_CPU sweep times out (some cells too slow at 222 ms × buttloop
  iters × 1,134 cells = significant wall): ship partial data + memo
  the timeout. Phase 7 can be informed by the partial picture.
- If C_GPU produces output that DECODE fails on (jxl-oxide, djxl, or
  jxl-rs): STOP IMMEDIATELY. This means the cvvdp-driven encoder
  produces broken bitstreams. File as a separate bug ticket + revert
  Phase 4 wiring until fixed.
- If the per-cell butter_cpu scores under C_GPU are wildly worse than
  under B (e.g. cvvdp encodes get butter_cpu ≥ 2.0 at distance ≤ 1.5):
  surface as a memo + recommend opt-in-only. cvvdp ≠ butteraugli; the
  fact that they prioritize different perceptual axes is the entire
  point of the fork, but ≥2× worse butter_cpu at low distance suggests
  the JOD calibration table needs adjusting before considering
  default-flip.

## Progress + final report

Memo path above. Update MEMORY.md. Final summary ≤500 words covering:
acceptance gates, cell coverage per backend, Pareto wins per backend,
wall_ms comparison, decision verdict (default-flip / opt-in / revert),
chunk N+1 brief reference.

This chunk's wall time is bounded by encoder speed × 3 backends ×
~1,134 cells. ~3-6 hours of compute. Pace yourself. Commit incrementally.