# Plan: Integrate Tree Learning into Squeeze Path

## Problem

Our lossless squeeze output is 48.5% larger than cjxl e7 (1568KB vs 1056KB on 1024x1024 CLIC photo).

**Section-by-section breakdown:**
- LfGlobal: 149KB vs 4KB (34.7x, 28% of gap) — tree/histogram metadata + global channel data
- Groups (16): 1387KB vs 1027KB (1.35x, 72% of gap) — per-group entropy coding

**Root cause:** The squeeze path uses a fixed gradient predictor with a single entropy context.
cjxl uses a learned MA tree with per-channel/per-region context adaptation (multi-context ANS).
The dispatch at `frame.rs:134-183` makes squeeze and tree learning **mutually exclusive**.

## Approach

Integrate the existing tree learning infrastructure (`tree_learn.rs`) into both the
single-group and multi-group squeeze paths. The pattern already exists for non-squeeze
multi-group in `section.rs:write_global_modular_section_with_tree`.

Pipeline: **RCT → squeeze → gather samples → learn tree → collect residuals → multi-context ANS**

## Changes

### 1. Add `channel_offset` parameter to `collect_residuals_with_tree` (tree_learn.rs)

The tree learns channel indices 0..42 from the full squeezed image. When collecting residuals
from a sub-image (e.g., PassGroup channels 36..42 cropped to a grid cell), the sub-image has
channel indices 0..6 but the tree expects 36..42. Add `channel_offset: u32` to offset property[0].

```rust
pub fn collect_residuals_with_tree(
    image: &ModularImage,
    tree: &Tree,
    group_id: u32,
    channel_offset: u32,  // NEW: added to ch_idx for property[0]
) -> Vec<Token>
```

All existing callers pass `channel_offset: 0` (no behavior change).

### 2. Update dispatch in `frame.rs` to combine squeeze + tree learning

**Single-group (line 134-161):** When both `use_squeeze` and `use_tree_learning` are set,
use a new combined path that applies squeeze first, then tree learning on the squeezed data.

**Multi-group (line 175-179):** When `use_tree_learning && use_ans` is set alongside squeeze,
use tree learning in `encode_modular_multi_group_squeeze`.

### 3. Add tree learning to `encode_modular_multi_group_squeeze` (frame.rs)

Replace the fixed gradient predictor + single-histogram approach with:

1. **gather_samples** from the FULL squeezed image (all 43 channels, correct indices)
2. **compute_best_tree** — learns per-channel/per-region context splits
3. **collect_residuals_with_tree** per section:
   - Global: channels[0..global_cutoff], channel_offset=0
   - PassGroup: for each group (gx,gy), crop each PassGroup channel, channel_offset=global_cutoff
4. **build_entropy_code_ans** from ALL tokens (all sections combined) — multi-context
5. **Write LfGlobal:** learned tree + multi-context ANS header + global channel tokens
6. **Write PassGroup sections:** tokens for each group using shared ANS code

Key detail: LfGroup is always empty for squeeze (all high-shift channels fit in global).

### 4. Add tree learning to single-group squeeze (encode.rs)

Create `write_modular_stream_with_squeeze_and_tree` that:
1. Apply RCT (YCoCg)
2. Apply squeeze
3. Gather samples from squeezed image
4. Learn tree
5. Collect residuals with tree
6. Build multi-context ANS
7. Write: tree + ANS header + GroupHeader (RCT + squeeze transforms) + tokens

### 5. Update all callers of `collect_residuals_with_tree`

Add `channel_offset: 0` to:
- `encode.rs:write_modular_stream_with_tree` (~line 2024)
- `section.rs:write_global_modular_section_with_tree` (~line 294)
- `section.rs:write_group_modular_section` (~line 435)

### 6. Update hash lock tests

Expected hashes will change for all lossless tests that use effort ≥ 8 (tree learning default-on).
Tests at effort 7 use squeeze without tree learning (no change).

Actually — at effort 7, squeeze is on and tree learning is off. At effort 8+, both are on.
The hash lock tests use default effort (7), so hashes should NOT change unless we also lower
the tree learning threshold. We should check what effort the hash tests use.

## Non-goals

- Not changing tree learning algorithm itself (max_nodes, split_threshold, property set)
- Not changing squeeze parameters or channel classification
- Not adding LZ77 to the squeeze path (separate feature)

## Testing

1. All existing roundtrip tests must pass (djxl, jxl-rs, jxl-oxide)
2. `just rd-regression` — zero lossy regression
3. New test: compare lossless squeeze sizes with and without tree learning
4. Compare against cjxl e7 on the 1024x1024 CLIC photo — should close a significant portion of the gap

## Risk

Low — tree learning infrastructure is battle-tested. The integration is additive
(old path works at lower efforts). Multi-context ANS is already proven in the non-squeeze path.
