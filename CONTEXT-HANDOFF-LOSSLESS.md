# Handoff: Matching libjxl Lossless Compression

## The Gap

Our lossless encoder at effort 7 produces files **28.5% larger** than cjxl e7 on 1024x1024 photos and **100-700% larger** on screenshots. We're at parity with cjxl e1.

**The entire gap comes from one missing feature: tree learning at effort 7.**

libjxl e7 (SpeedTier::kSquirrel) uses full MA (Meta-Adaptive) tree learning for lossless. Our encoder gates tree learning to effort 8+. Everything else is in place.

## What We Have (Working, Verified)

- **RCT (YCoCg)** on all lossless paths — single-group, multi-group, squeeze
- **Palette** with heuristic (skips when colors ≥ 50% of pixels)
- **ANS entropy coding** with histogram clustering
- **Gradient predictor** (single fixed predictor, single context)
- **Tree learning** (`tree_learn.rs`) — fully working, gated to effort 8+
  - 14 predictors (Zero, Left, Top, Average0-4, Select, Gradient, Weighted, TopRight, TopLeft, LeftLeft)
  - 15 of 16 properties (property 15 disabled — see below)
  - Multi-group support (global tree from all groups)
  - 13 tests, all passing, verified with jxl-rs and djxl
- **Weighted Predictor** — bit-exact with libjxl (golden-number verified)
- **Squeeze** — working but disabled by default (hurts without tree learning)

## What To Do

### Step 1: Enable tree learning at effort 7

**File:** `jxl_encoder/src/api.rs:500`

Change:
```rust
tree_learning: effort >= 8,
```
To:
```rust
tree_learning: effort >= 7,
```

This alone should close most of the 28.5% gap. The tree learning implementation is complete and tested.

### Step 2: Match libjxl e7 tree learning parameters

Our tree learning hardcodes some parameters. libjxl tunes them per effort level.

**libjxl effort 7 (kSquirrel) parameters:**
- `max_property_values = 48` (how many split thresholds to try per property)
- `num_properties = 7` base splitting properties (channels 0-2 × spatial properties)
- `splitting_heuristics_node_threshold = 117` (stop splitting when too few samples in a node)
- `Predictor::Best` — evaluates all spatial + weighted predictors (this maps to our approach already — we evaluate all 14)

**Our current parameters** (in `tree_learn.rs`):
- `max_nodes = 256`
- `split_threshold = 1.0` (minimum entropy reduction to split)
- All properties except 15 available for splitting

Check if these already match or need tuning. The libjxl parameters come from `lib/jxl/modular/encoding/enc_ma.cc` around the `CompressParams` struct.

### Step 3: Fix property 15 (wp_max_error) — Optional, ~1-2% impact

**The bug:** When tree learning splits on property 15 (weighted predictor's max error across sub-predictors), 128×128+ gradient images get off-by-1 roundtrip errors.

**What we know:**
- The weighted predictor itself is bit-exact (verified with golden numbers)
- Property 15 is computed correctly
- The bug manifests only when tree *splits* on property 15
- Root cause: encoder and decoder traverse the tree differently when property 15 is a split point, causing error buffer state divergence
- The error buffer maintains running WP sub-predictor errors. If encoder and decoder compute different properties at any pixel, subsequent error buffers diverge

**Investigation path:**
1. Enable property 15, encode a 128×128 gray gradient
2. Decode with jxl-rs, find the first pixel that diverges
3. At that pixel, compare encoder vs decoder: which tree node does each visit? What property values does each compute?
4. The divergence is likely in `WeightedPredictorState::predict_and_property()` error buffer update vs what the decoder does — focus on the `errors` array layout and update order
5. Check if `(xsize+2)*2` position-major error layout matches the decoder exactly

**Files:**
- `jxl_encoder/src/modular/tree_learn.rs:68` — where property 15 is disabled
- `jxl_encoder/src/modular/predictor.rs:384-559` — WP implementation + property 15 computation
- `jxl_encoder/src/api_tests.rs` → `test_tree_learning_gray_gradient_128x128` — repro case

### Step 4: Tune squeeze + tree learning together — Optional

Once tree learning is enabled at e7, squeeze becomes potentially beneficial again (it was only bad *because* tree learning was disabled). Test:

1. Enable tree learning + squeeze together at e7
2. Measure compression on CLIC photos and screenshots
3. If squeeze + tree learning beats tree learning alone, consider making squeeze the default again at e7

libjxl does NOT use squeeze at e7 for lossless, so this is speculative. But squeeze + good predictors *can* help on some content.

### Step 5: Fix palette + ANS checksum bug — Low priority

**The bug:** Images with many unique colors (approaching or at palette capacity) cause AnsChecksumMismatch when encoded through the palette path. Currently masked by the heuristic that skips palette when colors ≥ 50% of pixels.

**Repro:** 16×16 RGB gradient (256 unique colors, 256 pixels) → palette encoding → ANS checksum mismatch on decode.

This is low priority because:
- The palette heuristic correctly avoids this case
- Palette is only beneficial for few-color images (screenshots, graphics)
- Tree learning will help screenshots far more than palette tuning

## Expected Impact

| Change | Photo Impact | Screenshot Impact | Risk |
|--------|-------------|-------------------|------|
| Enable tree learning at e7 | -15 to -25% file size | -50 to -80% file size | Low — already tested at e8 |
| Match libjxl e7 parameters | -1 to -5% additional | -1 to -5% additional | Low — parameter tuning |
| Fix property 15 | -1 to -2% on gradients | Minimal | Medium — debugging required |
| Squeeze + tree learning | Unknown | Unknown | Medium — needs measurement |

## Key Files

| File | What's There |
|------|-------------|
| `jxl_encoder/src/api.rs:500` | Effort-level gating for tree_learning |
| `jxl_encoder/src/modular/tree_learn.rs` | Full tree learning algorithm (263-640 lines) |
| `jxl_encoder/src/modular/encode.rs:1970-2150` | Tree-based encoding path |
| `jxl_encoder/src/modular/frame.rs:145-152` | Single-group tree dispatch |
| `jxl_encoder/src/modular/frame.rs:241-249` | Multi-group tree dispatch |
| `jxl_encoder/src/modular/predictor.rs:384-559` | Weighted predictor + property 15 |
| `jxl_encoder/src/modular/rct.rs` | RCT (YCoCg) forward/inverse |
| `jxl_encoder/src/modular/palette.rs` | Palette analysis + heuristic |
| `jxl_encoder/src/entropy_coding/ans.rs` | ANS encoder (histogram normalization fix here) |

## Validation Checklist

After each change:
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test
```

Then verify:
1. Encode a 1024×1024 CLIC photo losslessly, decode with djxl and jxl-rs — 0 diff pixels
2. Encode a 1024×1024 screenshot — 0 diff pixels
3. Encode 13×17, 4×4, grayscale, RGBA — all roundtrip clean
4. Compare file sizes vs cjxl e7 (target: within 5-10%)
5. Run `just rd-regression` for lossy non-regression (lossless changes shouldn't affect VarDCT)

## Recent Commits (This Session)

```
4248a86 docs: update CLAUDE.md with lossless compression status (Feb 16)
418c728 fix: ANS histogram normalization for equal-count symbols, improve palette heuristic
0b0c258 fix: disable squeeze default, fix with_effort clobbering, disable palette in RCT path
edc4343 fix: swap 16-bit PNG byte order from big-endian to native
e78e48d feat: add RCT (YCoCg) color decorrelation to all lossless encoding paths
```

## libjxl Reference

The lossless encoding in libjxl is at:
- `~/work/jxl-efforts/libjxl/lib/jxl/modular/encoding/enc_ma.cc` — tree learning
- `~/work/jxl-efforts/libjxl/lib/jxl/modular/encoding/enc_encoding.cc` — modular encoding
- `~/work/jxl-efforts/libjxl/lib/jxl/modular/modular_image.cc` — squeeze, RCT
- `~/work/jxl-efforts/libjxl/lib/jxl/enc_modular.cc` — high-level lossless dispatch

At effort 7 for lossless, libjxl calls `LearnTree` with `Predictor::Best` (all predictors), all 16 properties enabled, then encodes with the learned tree providing per-pixel adaptive predictor/context selection.
