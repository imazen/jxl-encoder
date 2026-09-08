# High-bit-depth wall-clock: F1 primitives, and what the wide/float paths cost

Harness: `jxl-encoder/examples/bit_depth_perf.rs`. Machine: macOS aarch64 (M-series),
release build, `nice -n 19`. Repeats **interleaved** across arms inside every rep
(block-ordering once inverted the sign of a sectioned-K result), reporting **min of N**.

Run 2026-09-08 against `main` at the #95 input-surface landing.

## 1. Did F1's arithmetic cost anything?

F1 (`5f5789c0`) replaced three per-pixel modular primitives with libjxl's
overflow-correct forms. Byte-identity was proven at the time; this is the
wall-clock A/B that was promised and not run.

| primitive | pre-F1 ns/op | post-F1 ns/op | delta |
|---|--:|--:|--:|
| `clamped_gradient` | 0.349 | 0.382 | **+9.5 %** |
| `average2` | 0.208 | 0.245 | **+17.5 %** |
| `pack_signed` | 0.174 | 0.174 | −0.1 % |

Real but tiny in absolute terms: +0.033 and +0.037 ns/op, on operations that
cost a third of a nanosecond. Read these as ordering information, not as an
encode-time budget — a tight loop over `windows(3)` with a `match` per element
is not the shape these run in inside the tree learner, and the end-to-end
numbers below show no corresponding movement. The point of measuring was to
find out whether the correctness fix bought a *structural* slowdown. It did not.

## 2. Does the DECLARED bit depth cost anything? No.

Identical samples (values < 2^17), varying only the depth they are declared at.
512x512, min of 3.

| declared bits | ms | bytes |
|---|--:|--:|
| 17 | 30338.8 | 2017 |
| 24 | 29303.8 | 2017 |
| 31 | 28727.0 | 2017 |

**Flat, with byte-identical output.** Wide declared depths are free. (The
absolute number here is the subject of §4.)

## 3. Is the NEW planar path slower than the shipped layout path? No.

Identical samples (values < 2^16, ~57k distinct, smooth ramp) through
`encode_planar_int(.., 17, ..)` and through the long-shipped
`PixelLayout::Gray16`. 512x512, min of 3.

| path | ms | bytes |
|---|--:|--:|
| planar 17-bit (new) | 23.6 | 1624 |
| Gray16 layout (shipped) | 23.3 | 1624 |

**1.01x wall, identical bytes.** The new input surface costs nothing. This is
the control that separates "the wide path is slow" from "this content is slow",
and it lands firmly on the second.

## 4. What DOES drive wall time: content, by a factor of thousands

Fixed 31-bit declared depth, 256x256, varying only sample entropy:

| content | ms | bytes |
|---|--:|--:|
| smooth | 10.4 | 351 |
| noisy | 33.8 | 254059 |

And in §2, a *smooth* 512x512 ramp with ~75k distinct values took **~30 s** —
while §3's smooth 512x512 ramp with ~57k distinct values took **23 ms** on the
same code path at the same declared depth. Same shape of content, same path,
~1300x apart.

So there is a **content-dependent wall-clock cliff in lossless** somewhere
between those two fixtures. It is:

- **not** caused by the declared bit depth (§2 is flat),
- **not** caused by the new planar surface (§3 is 1.01x),
- reachable by 16-bit content too in principle — the wide-input work merely made
  it easy to construct, because a 17-bit ramp naturally has more distinct values
  than a 16-bit one.

The obvious suspect is distinct-value count driving the tree learner or a
palette/property path, but that is a **hypothesis, not a measurement** — no
profile was taken. Tracked separately rather than guessed at here.

## 5. Consequence for "should float get its own optimised path?"

No — not for the packing. f16 through the layout path measured **18.8 ms at
512x512, identical to `Gray16`'s 18.8 ms**, and f32 packing is a bitcast. The
per-sample conversion is not where the time is. The cases where float looked
catastrophically slow were maximally-noisy fixtures, and §4 shows integer
content of the same entropy behaves identically. Optimising the float packing
would be optimising a measured non-bottleneck; the cliff in §4 is the thing
worth fixing, and fixing it helps integer and float alike.

---

# ADDENDUM, same day: §4's cliff had a single root cause, and it is fixed

Investigated on request. Everything in §4 — and the "float is catastrophically
slow" reading of §5's headline table — turned out to be **one bug**, in
`DistinctPropertyValues` (`modular/tree_learn.rs`), the streaming collector that
feeds property threshold derivation.

## How it was found

Bisecting the value range at 512x512, declared 17-bit, isolated a transition
between range 70000 (24 ms) and 76000 (8456 ms). Two hypotheses were wrong and
are recorded so they are not re-tried: it is **not** the 2^16 boundary (65535
and 65537 are both fast), and it is **not** a sharp cliff in distinct count (the
profile shows a threshold followed by roughly linear growth). At 256x256 it does
not reproduce at all.

`--features profile-phases` then named it outright:

| phase | range 70000 | range 76000 |
|---|--:|--:|
| `modular/compute_best_tree` | 12.2 ms | 8438.8 ms |
| **`tree/pre_quantize`** | **8.4 ms** | **8435.2 ms** |
| everything else | ~18 ms | ~18 ms |

## The bug

`DistinctPropertyValues::push` compacted (full `sort_unstable` + `dedup`)
whenever the buffer reached a FIXED `COMPACT_AT = 65_536`. Once a property's
distinct set approaches that number, each compaction reclaims almost nothing, so
the next one fires a handful of pushes later — a full 65k-element sort per push.
Quadratic in the sample count, with a threshold exactly where the distinct set
meets the constant.

## The fix

Double the threshold past whatever survived compaction
(`compact_at = max(COMPACT_AT, 2 * surviving)`). Standard amortization; total
work returns to O(n log n). It changes only WHEN compaction happens, never what
the collector yields — both exit paths sort and dedup again — so thresholds, the
tree, and the bytes are independent of the schedule.

One trap worth recording: the struct had `#[derive(Default)]`, which would have
initialised the new field to 0 and compacted on EVERY push — worse than the bug.
It now has an explicit `Default`, and a test asserts that.

## Result (all byte-identical to before)

| cell | before | after | factor |
|---|--:|--:|--:|
| declared 17-bit, 512x512 | 30338.8 ms | **22.3 ms** | 1360x |
| declared 24-bit, 512x512 | 29303.8 ms | **22.4 ms** | 1308x |
| declared 31-bit, 512x512 | 28727.0 ms | **22.3 ms** | 1288x |
| end-to-end 24-bit | 23964.8 ms | **25.4 ms** | 943x |
| end-to-end 31-bit | 42813.3 ms | **27.6 ms** | 1551x |
| end-to-end f32 | 42758.1 ms | **55.1 ms** | 776x |

Range sweep at 512x512 after the fix is **flat**: 70000 / 71000 / 76000 /
100000 / 131000 all land at 22–24 ms. The cliff is gone rather than moved.

## What the corrected end-to-end table says

| path | ms | bytes | vs Gray16 |
|---|--:|--:|--:|
| Gray8 (layout) | 16.7 | 181 | 0.87x |
| Gray16 (layout) | 19.2 | 208 | 1.00x |
| 17-bit (planar) | 20.3 | 181 | 1.06x |
| 24-bit (planar) | 25.4 | 4784 | 1.32x |
| 31-bit (planar) | 27.6 | 18595 | 1.44x |
| f16 (layout) | 19.0 | 182 | 0.99x |
| f32 (layout) | 55.1 | 205387 | 2.87x |

The wide and float paths are now ordinary. f16 is indistinguishable from
`Gray16`. f32's 2.87x is not a path cost — its fixture produces 205 KB against
`Gray16`'s 208 bytes, i.e. it is doing ~1000x more entropy coding.

**§5's conclusion is unchanged and now better supported: no float-specific
optimised path is warranted.** The thing that looked like a float problem was
this collector, and fixing it helped integer and float alike.

## Scope: who was affected

The trigger is a *sampled property's* distinct set approaching 65536. Sampled
property columns carry neighbour values and their differences, so:

- reachable from the new >16-bit and float paths, whose samples span more than
  16 bits — confirmed here;
- **plausibly reachable from existing multi-channel 16-bit content**, where a
  property that is a difference of two 16-bit values spans 17 bits — NOT
  verified, and worth checking, since it decides whether this was a latent bug
  or a shipping one. A checker can revert the one-line doubling and re-run.

---

# ADDENDUM 2: the scope question is answered — #115 was a SHIPPING bug

The first addendum left one thing open, and said it decided severity: whether
the pathology reached the long-shipped `PixelLayout::Rgb16` surface, or only
the new >16-bit and float paths.

**It reached the shipped surface.** Measured on REAL content — the imazen-26
16-bit HDR renders (`png-v3/**/*.hdr.png`, 16-bit RGB, gain-map sources), not
synthetic — through `PixelLayout::Rgb16` with no wide input anywhere:

| image (1024x1024 centre crop) | before fix | after fix | factor | bytes |
|---|--:|--:|--:|--:|
| `1064_general_castle-bridge-moat_montjuic-castle-barcelona` | **42605.3 ms** | **2823.5 ms** | **15.1x** | 3 594 946 (identical) |

Harness `examples/scope_probe_16bit.rs`; "before" is the same binary with the
one-line `compact()` doubling reverted.

Two further after-fix cells on the same corpus, for context (1024x1024 crops):
`1065_..._sagrada-familia` 2315.0 ms / 2 486 310 B, `1066_..._park-guell`
3038.5 ms / 3 928 225 B.

## Why this was reachable without wide input

The trigger is a *sampled property column* whose distinct set approaches
65536. A single 16-bit channel already carries up to 65536 distinct values,
and the learner's properties include neighbour and reference-channel
DIFFERENCES, which span 17 bits (-65535..65535). Real 16-bit photographic
content fills that space; the synthetic ramps used earlier in this document
did not, which is why the size effect looked wide-input-specific at first.

## Consequence

Every 16-bit RGB lossless encode of detailed content has been paying this,
at roughly an order of magnitude, for as long as the fixed threshold has been
in place. It is not a regression from the September 2026 high-bit-depth work —
that work only made it easy to construct and therefore to find.

## Note on corpus

The 512x512 crop of the same image shows only 625.4 ms before the fix: the
sample count at that size does not push the distinct set far enough into the
degenerate region. **A performance claim about 16-bit content needs a real
16-bit image at a realistic size**; the local `codec-corpus` checkout contains
no 16-bit PNGs at all, and the imazen-26 `png-v3` HDR renders are the only
16-bit source on this machine.
