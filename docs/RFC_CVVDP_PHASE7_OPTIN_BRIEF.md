# Phase 7 agent brief — cvvdp fork opt-in landing (docs + CHANGELOG only)

**Status**: READY — Phase 6 verdict OPT_IN_ONLY in
`docs/CVVDP_FORK_DECISION.md`.

This is the smaller path of the two Phase 7 candidates. The cvvdp
opt-in is ALREADY shipped in source (Phases 3-5 landed all encoder
wiring + builder API + feature flags + Phase 5b CPU backend). Phase 7
just makes it discoverable: CHANGELOG, README, and an RFC status
flip to SHIPPED-OPT-IN.

If you instead want default-flip, see
`docs/RFC_CVVDP_PHASE7_DEFAULTFLIP_BRIEF.md` (NOT created — Phase 6
verdict was OPT_IN_ONLY, so the default-flip brief is unreachable
without first calibrating the distance table; see Phase 6 §11 for the
prerequisite).

## Read FIRST

1. `/home/lilith/.claude/CLAUDE.md` — discipline.
2. `~/work/zen/jxl-encoder/CLAUDE.md` — encoder rules + the CHANGELOG
   formatting rule (§"CHANGELOG.md" — every entry has a short commit
   hash reference).
3. `docs/CVVDP_FORK_DECISION.md` — the Phase 6 verdict + per-corpus
   data. The CHANGELOG entry text should pull from §1 / §3 / §4 / §8.
4. `docs/RFC_CVVDP_FORK.md` — RFC scope. Status line at §9 says
   "TBD: Phase 1 ship target, ...". Update to "SHIPPED-OPT-IN
   (Phase 6 verdict 2026-05-24, see CVVDP_FORK_DECISION.md)".
5. `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` — the full
   sweep data. Cite specific numbers from it where useful.

## Workspace claim

```bash
cd ~/work/zen/jxl-encoder
jj workspace add ~/work/zen/jxl-encoder--cvvdp-phase7-optin --revision cvvdp-fork-rfc@origin
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && printf '%s agent-cvvdp-phase7-optin docs+changelog\n' "$(cat /tmp/ts)" > ~/work/zen/jxl-encoder--cvvdp-phase7-optin/.workongoing
cd ~/work/zen/jxl-encoder--cvvdp-phase7-optin
```

## File ownership

**YOURS:**
- `CHANGELOG.md` — append entries under `[Unreleased]`.
- `README.md` — add a section documenting the cvvdp opt-in.
- `docs/RFC_CVVDP_FORK.md` §9 — flip status SCOPING →
  SHIPPED-OPT-IN.
- `docs/CVVDP_FORK_DECISION.md` — DO NOT MODIFY (it's the source of
  truth). You can reference it from the README and CHANGELOG.

**DO NOT TOUCH:**
- Any encoder source (`jxl-encoder/src/`). The opt-in is shipped.
- `docs/LIBJXL_DIVERGENCES.md` — cvvdp opt-in does not change
  libjxl-parity behaviour (Libjxl strategy never uses cvvdp per the
  `resolve_cvvdp_loop` invariant).
- `docs/HYPOTHESIS_LEDGER.md` — Phase 6 verdict didn't shift any
  hypothesis registered there.
- Other agents' work in flight on main (W44-AUDIT-5 / W44-AUDIT-6 etc).

## Scope (3 deliverables)

### 1. CHANGELOG entry

Append under `## [Unreleased]` → `### Added`:

```markdown
- **CVVDP opt-in for the buttloop** (`--features cvvdp-loop`): drives
  the quantization loop with ColorVideoVDP (Mantiuk et al. 2024)
  instead of butteraugli. Default OFF. Opt in via
  `LossyConfig::with_cvvdp_loop(Some(true))` + the `cvvdp-loop`
  cargo feature (requires CUDA). On photos at e=8 cvvdp encodes
  score better on the cvvdp metric vs butteraugli encodes (100% of
  CID22 cells improve at every distance). Trade-off: cvvdp ships
  with an uncalibrated distance target table, producing +155% to
  +286% larger files than butteraugli at the same `distance`
  parameter; the default cannot flip until the table is calibrated.
  See `docs/CVVDP_FORK_DECISION.md` for the full Phase 6 sweep data.
- **CPU CVVDP backend opt-in** (`--features cvvdp-loop-cpu`): pin
  the cvvdp buttloop to the CPU backend (no CUDA required). Opt
  in via `LossyConfig::with_cvvdp_use_cpu(Some(true))` + the
  `cvvdp-loop-cpu` cargo feature. ~1.55× wall vs butteraugli CPU
  buttloop on e=8 cells; output byte-identical to the GPU backend.
- **Tracking benchmark + Pareto analyzer** for cvvdp vs butteraugli
  at `benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv` and
  `scripts/cvvdp_pareto_analysis.py`.
```

Reference the commit SHA where each of Phases 3-5 + Phase 6 landed
(see `git log --oneline -- jxl-encoder/src/vardct/cvvdp_backend.rs
jxl-encoder/src/api.rs benchmarks/cvvdp_vs_buttloop_tracking_2026-05-24.tsv`).

### 2. README addition

Add a `## CVVDP-driven quantization loop (experimental opt-in)`
section under the existing `## Features` or `## Usage` section
(grep for the right anchor). Cover:

- What it is (cvvdp = ColorVideoVDP, Mantiuk et al. 2024)
- How to opt in (`--features cvvdp-loop` for GPU,
  `--features cvvdp-loop-cpu` for CPU)
- Builder API: `with_cvvdp_loop(Some(true))` /
  `with_cvvdp_use_cpu(Some(true))`
- Why it's opt-in (per-distance bytes overhead — point at Phase 6
  decision memo for numbers)
- Wall time (C_GPU is 13% FASTER than butteraugli CPU at e=8;
  C_CPU is 55% slower)
- When cvvdp wins (photos at e=8 d≥1.0)
- When butteraugli still wins (screenshots; default path)

Keep it under 400 words. Link to `docs/CVVDP_FORK_DECISION.md` and
`docs/RFC_CVVDP_FORK.md` for the full story.

### 3. RFC status flip

Edit `docs/RFC_CVVDP_FORK.md` §9 status notes block. Change the line
"**Status**: SCOPING" near the top of the doc to:
"**Status**: SHIPPED-OPT-IN (Phase 6 verdict 2026-05-24)"

Append to §9 a final 2026-05-24 status line:
"Phase 6 sweep landed. Verdict OPT_IN_ONLY per
`docs/CVVDP_FORK_DECISION.md`. cvvdp + cvvdp-loop-cpu features
shipped behind opt-in API. Distance-table calibration left as
future work."

## Acceptance gates

- (a) `cargo fmt --check` PASS (no Rust source changes expected).
- (b) `cargo build` PASS on default features (no opt-in must regress
  the default build).
- (c) CHANGELOG entries reference commit SHAs.
- (d) README section ≤400 words.
- (e) RFC status flipped.
- (f) Single commit pushed to `cvvdp-fork-rfc`. Stack on Phase 6.
- (g) Memo at
  `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_fork_phase7_optin_shipped_<date>.md`.
- (h) `.workongoing` removed; workspace forgotten and removed per
  CLAUDE.md cleanup-on-merge rule.

## Out of scope (do NOT do)

- Distance-table calibration. That's a separate chunk — would need a
  new sweep that bisects the JOD target per distance bucket to
  achieve byte-parity with butteraugli, then re-runs the Phase 6
  analyzer.
- W44 gate re-validation under cvvdp. Phase 8 territory; only
  triggered by future default-flip.
- Wiring cvvdp into the CLI tool (`jxl-encoder-cli`). The opt-in API
  exists; the CLI doesn't expose it. Can be a follow-on chunk.
- Refactoring Phase 4 encoder source. The wiring is done.

## DO NOT

- DO NOT flip the default. Phase 6 verdict explicitly ruled this out
  per RFC §5.4.
- DO NOT cite "FMA precision" anywhere (per W44-66 user correction).
- DO NOT modify the Phase 6 decision memo
  (`docs/CVVDP_FORK_DECISION.md`) — reference it instead.
- DO NOT add new feature flags. The 3 existing (`butteraugli-loop`,
  `cvvdp-loop`, `cvvdp-loop-cpu` + `gpu-butteraugli`) are correctly
  scoped per RFC §2.2.

## Final report shape

≤300 words covering:
- Acceptance gates PASS/FAIL
- The CHANGELOG entries added
- README anchor and approximate word count
- RFC status line updated
- Commit SHA pushed
- `.workongoing` removal + workspace cleanup
