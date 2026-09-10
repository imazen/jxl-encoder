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

## 2. What is promised — owner decision, 2026-09-09

> "byte monotonicity is a great-to-have, iqa monotonicity is the key"

So the two halves are graded differently, and the gate enforces exactly that:

- **HARD — delivered IQA must be non-increasing as the requested distance
  coarsens, EVERYWHERE, filter boundaries included.** This is the only thing
  that fails the gate. A boundary crossing is not an excuse: a crossing that
  raises delivered quality is precisely the case where the caller asked for
  coarser output and got finer. Holding this line across boundaries is a
  deliberate divergence from libjxl, which does not hold it.
- **ADVISORY — byte monotonicity is reported, never failed.** It provably cannot
  hold across a reference-filter transition without diverging further: libjxl
  v0.12 measured at d = 0.5 → 0.6 goes **71,368 → 94,002 B** on imazen-26 7026,
  because Gaborish is gated at `d > 0.5` and the EPF thresholds step at
  0.7 / 1.5 / 4.0.

**Libjxl mode stays faithful.** Every knob in §4/§5 is zen-only.

### The measured split is cleaner than expected, and it simplifies the roadmap

An earlier draft of this RFC predicted that filter boundaries would be where IQA
monotonicity breaks. **Measured, that is wrong**, and the correction is useful:

| | within-regime | at filter boundary |
|---|---|---|
| **IQA inversions** (hard) | **4 of 4** | **0** |
| byte inversions (advisory) | 0 | all of them |

On the #103 cliff images at e8, every IQA violation is a within-regime
**targeting** bug, and the boundary crossing produces only a byte inversion —
i.e. it lands entirely in the half just deprioritised.

Consequence: **design C (distance as a target, not a seed) addresses the whole
of the hard contract**, and the filter-boundary question is confined to the
advisory half. There is no need to redesign the filter gating to satisfy the key
contract.

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
tested directly rather than assumed.

**Photographic grid** — 4 images × e{3,5,7,9} × 19 distances = 304 cells
(`benchmarks/rd_monotonicity_2026-09-09.tsv`): **IQA monotonicity clean**, even
with boundaries graded. 2 byte inversions, both at boundaries (advisory).

**#103 cliff images at e8** (`benchmarks/rd_monotonicity_knowncliffs_2026-09-09.tsv`):
exit 1, **4 IQA violations, all within-regime**:

| cell | inversion |
|---|---|
| 5058 e8 | SSIM2 **86.655 → 88.168** as d went 3 → 3.5 |
| 5058 e8 | SSIM2 92.589 → 92.924 as d went 0.4 → 0.5 |
| 5058 e8 | SSIM2 68.139 → 69.311 as d went 12 → 15 |
| 9291 e8 | SSIM2 88.544 → 89.758 as d went 0.4 → 0.5 |

plus 1 advisory byte inversion at a boundary (56,670 → 66,266 B, d 0.5 → 0.6) —
the same crossing libjxl is non-monotone at.

Both branches are mutation-verified: tightening the byte tolerance to 0.5 fires
the byte branch on 14 cells; the SSIM2 branch fires unaided above.

**Time, per effort, against the committed baseline and against cjxl:**

| effort | ours | cjxl | ours/cjxl | vs baseline |
|---|---|---|---|---|
| e3 | 672 ms | 942 ms | **0.714** | 1.006 |
| e5 | 2,347 ms | 1,285 ms | **1.827** | 0.996 |
| e7 | 4,862 ms | 1,914 ms | **2.541** | 1.000 |
| e9 | 18,791 ms | 17,695 ms | 1.062 | 1.001 |

e7 at 2.54× cjxl is the standout; e3 is faster than cjxl and e9 is near parity,
so the gap is specifically the e5–e7 band. Wall is stable against baseline
(0.996–1.006), which is the check the owner asked for.

## 4. Designs A–C (proposed)

**A — keep-best instead of a threshold — IMPLEMENTED opt-in, MEASURED, and the
recommendation is narrow.** `JXL_LZ77_KEEP_BEST=1` replaces the estimator's
`bit_decrease > total_symbols * 0.2 + 16` with an actual coded-size comparison:
both candidate streams are built for real — clustered ANS histogram plus tokens,
through the writers production uses — and the smaller wins. The motivation was
that CLAUDE.md records the proxy as weak in exactly this regime (only the
≤96-histogram-clustered cost reproduces the real gap; the ideal per-context
entropy saw 1.7 % of a measured 27 % effect).

Measured (`benchmarks/lz77_keep_best_2026-09-10.{tsv,meta}`, 3 reps, min per
arm, interleaved):

| content | path | bytes on/off | best cell | wall |
|---|---|---|---|---|
| **real graphics** | **lossless e8/e9** | **0.9934–0.9946** | **0.928** | +10 % |
| real graphics | lossy e9 | **1.0037** (worse) | 1.0000 | +13 % |
| synthetic line-art | lossy e9 | 0.944 | — | +172 % |
| photos | either | 1.0000 | — | +8 % |

**The lossy win was synthetic-only and inverts on real content** — a textbook
instance of the no-synthetic-only rule, and the reason the real-graphics arm was
run before drawing any conclusion.

**Recommendation:** worth considering for **lossless at high effort only**
(0.5–0.7 % mean, up to 7.2 % on the best cell, for ~10 % wall, on content where
time is already being spent). **Do not enable on the lossy path** at any effort.
Not flipped by default here: that is a byte-moving change on 8 images, which is
not a corpus.

**Two caveats that must travel with it.** (1) Keep-best **disables the sound
early-out**, whose proof is derived from the very threshold keep-best removes —
under keep-best there is no threshold for the bound to prove unreachable, so the
~73 % of streams the early-out discards nearly free are all evaluated in full.
That is most of the +10 % wall. (2) The comparison is made in a *standalone*
context (`total_pixel_hint: None`, per-stream clustering) while production
builds with the real hint and may group differently, so keep-best can
mis-decide: the measured +0.22 % on one photo-lossless cell and the +0.37 % on
real-graphics lossy are that gap showing up, not noise.

**Binding owner constraint:** keep-best must be **measured and decided per
effort level**, not adopted globally. Its cost is a second candidate evaluation,
which is affordable at high effort and may not be at low effort; the shipped
`cfl_keep_best` is already `effort >= 7`, and any LZ77 analogue needs its own
per-effort measurement before it defaults on anywhere.

**B — monotone envelope over the ladder — GATE SHIPPED, finding is inherited.**
`examples/effort_monotonicity_gate.rs` grades the effort axis on strict **Pareto
domination**: effort *N* "loses" only when worse on BOTH axes. (An earlier draft
flagged any byte rise where quality merely failed to fall; that mislabels
genuine trades and produced 5 false positives before being tightened.) It also
reports DOMINATED (byte-identical for measurably more wall — hard on lossless,
which has no quality axis to trade against) and THIN (>1.5× wall for <0.5 %
bytes) as advisories.

**Result: 2 violations, both `e4 → e5` at d=1, and libjxl has them too.** e4 is
the strict optimum of the low ladder (car 31,664 B @ 88.954 vs e3's 32,128 @
88.954). The mechanism splits: **gaborish costs ~1–1.3 SSIM2** (disabling it at
e5 lifts the food crop 90.872 → 92.201) and **the AC-strategy search spends the
bytes** (36,472 vs 31,664 even with gaborish off). `adaptive_gaborish` does not
recover it.

cjxl v0.12 on the same crops shows the same domination — food e4 16,256/90.769 →
e5 19,180/90.198, car e4 30,938/88.308 → e5 34,046/87.237 — so **Libjxl mode
must keep it** and this is a zen-mode divergence candidate, not a porting bug.

Two constraints on any fix. **Gaborish is tuned against butteraugli, not
SSIM2**, so the quality half is metric-dependent and must be re-checked against
butteraugli before a gate moves. And **do not re-gate the AC search from effort
5 to 6** — that merely renames e5 to e4; the search does not pay at d=1 but
clearly does at d=4 (e5 → e7 trades of +2.47 and +3.32 SSIM2 for ~2 % bytes), so
the fix is distance-dependent, i.e. a keep-best evaluated per effort — design A.

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
