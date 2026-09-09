# RFC — RD monotonicity: contract, gate, and tuning designs

Status: **Design D SHIPPED as a gate** (`examples/rd_monotonicity_gate.rs`);
A/B/C proposed. Owner directives that shape this doc, 2026-09-09:

> "our mimic-libjxl mode should be faithful but zen mode should fix the cliffs
> and require monotonicity with distance+resultant iqa scores"

> "D is the most important as we train zensim. ssim2 can be our oracle for the
> moment. Also, avoid increasing time per effort over our baseline and track it
> relative to libjxl as well. keep-best should be measured and considered in the
> context of the effort level"

## 1. Two contracts, not one

"Monotonic" is ambiguous, and conflating the two is how the previous round
produced a lateral RD move. They need different mechanisms:

- **Effort monotonicity** — at fixed distance, more effort must never cost more
  bytes at equal quality. Today's failures are *dominance* failures rather than
  inversions. Measured: lossless float e3 and e5 are **byte-identical on 26/26
  (image, size) pairs** from 64² to 4096² (e5 spends e5 time for e3 bytes), and
  e9 at 4096² costs 198 s for 1.4 % fewer bytes than e7.
- **Distance monotonicity** — at fixed effort, coarsening the requested distance
  must not increase bytes, and delivered IQA must track the request. This is
  #103.

## 2. What is promised, and what provably cannot be

**Promised (zen mode):** within a reference-filter regime, bytes and delivered
SSIM2 are non-increasing as distance coarsens.

**Not promised, with evidence:** monotonicity ACROSS a filter boundary. libjxl
v0.12 is not byte-monotone there either — measured on imazen-26 7026, d = 0.5 →
0.6 goes **71,368 → 94,002 B** — because Gaborish is gated at `d > 0.5` and the
EPF thresholds step at 0.7 / 1.5 / 4.0. The gate therefore treats
`FILTER_BOUNDARIES = [0.5, 0.7, 1.5, 4.0]` crossings as **declared
discontinuities**: recorded, never silently tolerated, never failed. Demanding
byte monotonicity across them is a deliberate divergence from libjxl and needs
to be requested as one.

**Libjxl mode stays faithful.** Every knob below is zen-only.

## 3. Design D — the staircase gate (shipped)

`examples/rd_monotonicity_gate.rs`. Per (image, effort) it sweeps a
low-end-dense distance ladder (19 points, 0.4 … 15) and records bytes, delivered
SSIM2, and wall, alongside cjxl v0.12 at the same effort and distance. It then
asserts the within-regime staircase and exits non-zero on violation.

Oracle: SSIMULACRA2 via `fast-ssim2`, **pinned at 0.7.1** — 0.8.2 moves every
score (0 of 84 cells bit-identical), so a bump would silently rebase this gate.
Both sides are fed **sRGB u8**, because `compute_ssimulacra2` linearises
internally; requesting linear output here would double-linearise, which is the
documented way to get garbage scores out of this crate.

**Time is a first-class assertion.** Per effort the gate reports total wall
against a committed baseline (`benchmarks/rd_monotonicity_baseline_2026-09-09.tsv`)
and the ratio against cjxl. An RD win bought with unbounded time is not a win.

### Validation — it catches the known bug

The gate is only worth having if it fails on the documented cliffs, so that was
tested directly rather than assumed:

- **4 photographic images × e{3,5,7,9} × 19 distances = 304 cells: clean**, 64
  boundary crossings skipped (`benchmarks/rd_monotonicity_2026-09-09.tsv`).
- **The #103 cliff images at e8: 4 within-regime violations**
  (`benchmarks/rd_monotonicity_knowncliffs_2026-09-09.tsv`), and they surface as
  **quality** inversions, matching #103's mis-targeting framing rather than a
  byte story:

| cell | inversion |
|---|---|
| 5058 e8 | SSIM2 **86.655 → 88.168** as d went 3 → 3.5 |
| 5058 e8 | SSIM2 92.589 → 92.924 as d went 0.4 → 0.5 |
| 5058 e8 | SSIM2 68.139 → 69.311 as d went 12 → 15 |
| 9291 e8 | SSIM2 88.544 → 89.758 as d went 0.4 → 0.5 |

Both branches are mutation-verified: tightening the byte tolerance to 0.5 makes
the byte branch fire on 14 cells, and the SSIM2 branch fires unaided above.

Time datum from that run: e8 **ours/cjxl = 1.043**.

## 4. Designs A–C (proposed)

**A — keep-best instead of a threshold.** Generalise the shipped
`cfl_keep_best` pattern (per-tile: compute both candidates, keep the cheaper
under a coded-cost proxy; +17.7 % on aliased line art, 0/24 regressions).
Applied to LZ77: accept iff it reduces the **clustered ANS** cost, not the
estimator's guess — CLAUDE.md records that only the ≤96-histogram-clustered cost
reproduces the real gap, while the ideal estimator saw 1.7 % of a 27 % effect.
Structurally monotone, and it deletes a constant instead of tuning it.

**Binding owner constraint:** keep-best must be **measured and decided per
effort level**, not adopted globally. Its cost is a second candidate evaluation,
which is affordable at high effort and may not be at low effort; the shipped
`cfl_keep_best` is already `effort >= 7`, and any LZ77 analogue needs its own
per-effort measurement before it defaults on anywhere.

**B — monotone envelope over the ladder.** Effort *N* may not lose to *N−1*;
where features are nested this holds by construction, otherwise *N* falls back
to *N−1*'s decision. This is what removes the measured e3≡e5 dominated point.

**C — distance as a target, not a seed.** The real #103 fix. Feasibility is
already measured: with the seed lift off, the same perceptual loop lands at
1.02–1.07 delivered/requested. IQA monotonicity then holds by construction.
Needs the contract question in §2 settled first.

## 5. The rate/time dial (the question that started this)

The LZ77 acceptance threshold `total_symbols * 0.2 + 16` is a genuine rate/time
dial, measured (interleaved, 3 reps, min per arm):

| scale | wall | bytes |
|---|---|---|
| 0.5 | **1.377** | **0.971** |
| 1.0 | 1.000 | 1.000 |
| 2.0 | 0.949 | 1.0029 |
| 10.0 | 0.927 | 1.0029 |

It survives with the early-out disabled, so the cost is the downstream work an
**accepted** stream causes, not the walk length. Past 2× nothing changes.

**Keying: mode × effort, and mode first.** `0.2` is libjxl parity
(`enc_lz77.cc:165`), so keying on effort alone would move Libjxl-mode bytes.
Libjxl pins 0.2; zen makes it effort-dependent, because it is a rate/time dial
and effort *is* the rate/time dial.

**Structural blocker:** the threshold lives on the modular path, and
`LosslessConfig` has **no `EncoderStrategy` axis** (it threads
`EncoderMode{Reference, Experimental}`, a different enum). CLAUDE.md already
names this as the gate on the whole of `LIBJXL_PARITY_TRACKING.md`. Adding that
axis is the prerequisite for every lossless divergence, this dial included.

Note the early-out shipped in `06019e25` needs **no** mode gate: it is
result-identical, so it is faithful in both modes. That is the shape to prefer —
wins that cost no fidelity.

## 6. Order of work

1. **D** — shipped. Without it A–C are unfalsifiable.
2. **B** — mechanical; catches the dominated points already measured.
3. **A** — per-effort measured, per the owner constraint above.
4. **C** — highest value, blocked on the §2 contract decision.
