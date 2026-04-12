> **Resolved.** The mutual exclusion gate between palette and tree learning has been removed. Tree learning now handles palette internally when beneficial (see `frame.rs` comment "Tree learning: handles palette internally when beneficial" and `encode.rs:1294` where `should_use_palette` is called inside the tree-learning path). This document is retained as historical design reference.

# Modular Palette + Tree Learning Investigation

## Problem

Lossless screenshots have wildly varying compression ratios. The root cause is a mutual exclusion gate in `frame.rs:306-319`: **palette and tree learning are mutually exclusive**.

Screenshots with ≤256 unique RGB colors hit the palette path and lose all advanced encoding (no tree learning, no LZ77, no multi-context ANS, no predictor search). Screenshots with >256 colors get tree learning with 14 predictors, LZ77, and multi-context ANS. This creates a binary quality cliff.

## Decision Cascade (single-group, effort 7)

```
frame.rs:273-328

1. lossy_palette?                        → no (lossless)
2. squeeze + tree_learning + ANS?        → no (squeeze hardcoded false)
3. squeeze alone?                        → no (squeeze hardcoded false)
4. tree_learning + ANS + !palette?       → DEPENDS on should_use_palette()
5. fallback: write_modular_stream_with_rct  → gradient predictor, 1 context, no LZ77
```

Step 4 is the gate. `should_use_palette()` returning `Some` skips tree learning entirely.

## Palette Path Weaknesses

`write_modular_stream_with_palette` (encode.rs:364) applies palette transform but then encodes with:
- Gradient predictor only (1 of 14 available)
- Single ANS histogram for all channels (no context modeling)
- No LZ77 (huge loss for screenshots — palette indices have long runs)
- Hardcoded YCoCg RCT (no search among 42 variants)
- No per-channel context separation

## Why Palette + Tree Learning Are Mutually Exclusive

Comment at frame.rs:310-311:
> "palette + tree learning has a meta-channel encoding mismatch that causes decoder failures"

## The Meta-Channel Encoding Mismatch — RESOLVED

**The mismatch is a Rust encoder bug, not a spec limitation.** libjxl handles palette + tree learning together with no issues.

### How libjxl does it (the correct approach)

In libjxl, palette and tree learning are NOT mutually exclusive. The sequence is:

1. Palette transform is applied: creates meta-channel (nb_colors × num_c) + index channel (width × height), replaces the original color channels
2. `image.nb_meta_channels` is incremented to mark the palette as a meta-channel
3. Tree learning runs on ALL channels, including the palette meta-channel
4. The tree splits on property 0 (channel index) to use different contexts/predictors for the meta-channel vs index channels

**Key design elements:**
- The palette meta-channel has different dimensions (e.g., 200×3) from image channels (1920×1080). This is handled by `PrecomputeReferences()` which only uses reference channels with matching (w, h, hshift, vshift) for neighbor prediction.
- Channel index (property 0) lets the tree distinguish meta-channels from pixel channels — different leaves handle each.
- Meta-channels are ALWAYS included in tree data gathering (the `c >= nb_meta_channels` check in libjxl only gates *skipping* large non-meta channels).

### What the Rust encoder gets wrong

The Rust encoder's `write_modular_stream_with_tree_dc_quant` (encode.rs:1156) writes the GroupHeader transform section at line 1355-1366. It only handles RCT and Squeeze transforms:

```rust
let num_transforms = has_rct as u32 + has_squeeze as u32;
writer.write(2, num_transforms as u64)?;
if let Some(rct_type) = rct_type { write_rct_transform(...)?; }
if let Some(ref params) = squeeze_params { write_squeeze_transform(...)?; }
```

To support palette + tree, this needs to also write the palette transform descriptor. And the channel list fed to `gather_samples_strided` / `collect_residuals_with_tree` must be the post-palette channel list (meta-channel + index channel), not the original RGB channels.

The "decoder failure" was likely: palette was applied to the pixel data, but the GroupHeader didn't declare the palette transform, so the decoder didn't know to invert it — producing garbage.

### What needs to change in the Rust encoder

1. **`write_modular_stream_with_tree_dc_quant`** needs a palette option:
   - Accept an optional `PaletteAnalysis`
   - If palette: apply palette transform to get `[meta_channel, index_channel, maybe_alpha]`
   - Feed that transformed image to tree learning (gather + collect)
   - Write GroupHeader with palette transform + optional RCT (palette replaces RCT for the color channels)

2. **`frame.rs` decision cascade** should become:
   ```
   if tree_learning + ANS {
       // Try palette on the image first
       let palette_info = should_use_palette(image);
       // Apply palette if beneficial, then feed to tree learning
       write_modular_stream_with_tree(image, palette_info, ...)?;
   }
   ```

3. **The `should_use_palette` gate** at frame.rs:308 should be removed. Instead, palette becomes an input to the tree-learning path, not a reason to skip it.

4. **Palette + RCT interaction**: In libjxl, palette replaces RCT for the palettized channels (you wouldn't RCT-then-palette or vice versa on the same channels). The Rust encoder should skip RCT when palette is active for those channels.

## Multi-Group Path

For images >256×256 (all real screenshots), multi-group path (`frame.rs:355-358`) never considers palette. Always uses RCT + tree learning. So the variance may also be between:
- Small screenshots → single-group → palette gate → weak encoding
- Large screenshots → multi-group → tree learning → strong encoding

The multi-group path should also support palette. In libjxl, palette is a global transform applied before group splitting.

## Fix Plan

### Phase 1: Integrate palette into tree-learning path (single-group)

1. Modify `write_modular_stream_with_tree_dc_quant` to accept optional palette params
2. When palette is requested: apply `apply_palette()`, then proceed with tree learning on the transformed image
3. Write palette transform in GroupHeader (alongside or instead of RCT)
4. Remove the `should_use_palette().is_none()` gate from frame.rs:308
5. In frame.rs decision cascade: always use tree-learning path when `use_tree_learning && use_ans`, with palette as an optional pre-transform

### Phase 2: Integrate palette into multi-group path

1. Apply palette globally (before group splitting)
2. Feed palettized image to existing multi-group tree-learning path
3. Write palette transform in LfGlobal section

### Phase 3: Quick wins for non-tree paths (effort < 7)

1. Add LZ77 to `write_modular_stream_with_palette` for effort levels without tree learning
2. Use multi-context ANS even without tree learning (separate contexts for meta-channel vs index)

## Key File Locations

- `jxl_encoder/src/modular/frame.rs:273-328` — Single-group decision cascade
- `jxl_encoder/src/modular/frame.rs:306-319` — Tree learning vs palette gate (TO REMOVE)
- `jxl_encoder/src/modular/frame.rs:342-358` — Multi-group decision cascade
- `jxl_encoder/src/modular/encode.rs:364-459` — `write_modular_stream_with_palette` (weak path)
- `jxl_encoder/src/modular/encode.rs:1110-1393` — `write_modular_stream_with_tree` (good path, needs palette support)
- `jxl_encoder/src/modular/encode.rs:1355-1366` — GroupHeader transform writing (needs palette)
- `jxl_encoder/src/modular/palette.rs:269-310` — `analyze_palette` threshold logic
- `jxl_encoder/src/modular/palette.rs:349-397` — `apply_palette` (channel replacement)
- `jxl_encoder/src/modular/palette.rs:1037-1067` — `should_use_palette` gate
- `jxl_encoder/src/modular/encode_transforms.rs:81` — `write_palette_transform` (already exists)
- `jxl_encoder/src/api.rs:551` — squeeze hardcoded false
- `jxl_encoder/src/effort.rs:258-343` — lossless effort profiles

## libjxl Reference Locations

- `lib/jxl/modular/transform/enc_palette.cc:238-241, 276-277, 638-641` — Palette application + nb_meta_channels++
- `lib/jxl/enc_modular.cc:328-427` — try_palettes() before tree learning
- `lib/jxl/enc_modular.cc:884-894` — Global palette applied before tree learning
- `lib/jxl/modular/encoding/enc_encoding.cc:555-634` — LearnTree with full channel loop
- `lib/jxl/modular/encoding/enc_encoding.cc:614-626` — GatherTreeData on all channels including meta
- `lib/jxl/modular/encoding/context_predict.h:411-443` — PrecomputeReferences dimension filtering
