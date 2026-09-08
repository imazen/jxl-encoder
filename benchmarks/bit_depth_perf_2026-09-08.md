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
