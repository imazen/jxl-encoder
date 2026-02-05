# Context Handoff: AFV Inverse Transform Implementation

## Current State (Feb 4, 2026)

AFV corner DCT is implemented and works correctly for **forced encoding**, but **auto-selection is disabled in pixel-domain mode** (the default) because we lack the proper inverse AFV transform.

### What Works
- AFV0-3 transforms (forward): `afv_transform_from_pixels()` in `afv.rs`
- AFV quantization weights: `generate_afv_weights()` in `quant.rs`
- AFV DC extraction: `dc_from_afv()` returns `coefficients[0]`
- Forced AFV encoding: `AcStrategyMap::force_strategy(RAW_STRATEGY_AFVx)`
- Decoder validation: jxl-oxide and djxl both decode AFV files correctly

### What's Broken
- **AFV auto-selection in pixel-domain mode**: Disabled because `apply_idct_for_strategy()` uses `idct_8x8()` for AFV, which is wrong
- Root cause: AFV has a hybrid inverse transform (inverse AFV4x4 + inverse DCT4x4 + inverse DCT4x8), not a standard 8x8 IDCT
- Effect: Wrong IDCT produces meaningless pixel-domain loss values, causing 35-43% cost underestimation

### Commits Made This Session
1. `fix: disable AFV auto-selection in pixel-domain loss mode` (02e8946)
2. `docs: update CLAUDE.md with AFV auto-selection root cause` (b4586f0)
3. `style: fix clippy warnings and formatting` (82274b9)

## Next Step: Implement Inverse AFV Transform

### Where to Look in jxl-rs

The inverse AFV transform should be in the decoder. Check:
```bash
cd ~/work/jxl-rs
grep -rn "AFV\|afv" src/
```

Key files to examine:
- `src/render/transform.rs` or similar - inverse transforms
- Look for `IAFV` or inverse AFV4x4 basis matrix

### What the Inverse AFV Transform Needs

Based on forward transform structure in our `afv.rs`:

1. **Inverse AFV4x4** at (even_row, even_col) positions
   - Uses transposed AFV4x4 basis matrix (we have `AFV4X4_BASIS_TRANSPOSE`)
   - May need `AFV4X4_BASIS` (non-transposed) for inverse

2. **Inverse DCT4x4** at (even_row, odd_col) positions
   - Standard 4x4 IDCT

3. **Inverse DCT4x8** at (odd_row, any_col) positions
   - Standard 4x8 IDCT

4. **Pixel reassembly** based on AFV kind (0-3) with mirroring

### Implementation Location

Add to `jxl_enc/src/tiny/afv.rs`:
```rust
/// Inverse AFV transform: coefficients → pixels
pub fn inverse_afv_transform(coefficients: &[f32; 64], afv_kind: usize, pixels: &mut [f32; 64]) {
    // TODO: Implement based on jxl-rs decoder
}
```

Then update `apply_idct_for_strategy()` in `ac_strategy.rs` to call it for AFV strategies.

### Testing

After implementing inverse AFV:
1. Test roundtrip: `afv_transform_from_pixels` → `inverse_afv_transform` should recover original pixels
2. Enable AFV auto-selection in pixel-domain mode (remove the `if !use_pixel_domain` guard)
3. Run `just rd-regression` to verify no quality regression

## Files Modified This Session
- `jxl_enc/src/tiny/ac_strategy.rs` - AFV auto-selection guard
- `jxl_enc/src/tiny/afv.rs` - clippy allows
- `jxl_enc/src/tiny/dc_coding.rs` - dead_code allows  
- `jxl_enc/src/tiny/dc_tree_learn.rs` - module-level allows
- `CLAUDE.md` - documentation update

## Known Pre-existing Issue
- `test_lz77_backref_roundtrip` fails with jxl-oxide decoder error - unrelated to AFV work

## jxl-rs Reference Found!

The inverse AFV transform is in `~/work/jxl-rs/jxl_transforms/src/transform.rs`:

```rust
fn afv_transform_to_pixels<D: SimdDescriptor>(
    d: D,
    afv_kind: usize,
    coefficients: &[f32],
    pixels: &mut [f32],
)
```

Key components:
1. **DC extraction**: `dcs[0,1,2]` from coefficients at positions 0, 1, 8
2. **avfidct4x4()**: Inverse AFV4x4 using `AFV4X4BASIS` matrix (256 floats)
3. **idct2d_4_4()**: Standard IDCT4x4
4. **idct2d_4_8()**: Standard IDCT4x8
5. **Pixel mirroring**: Based on `afv_kind` (0-3) for different corners

Also need `AFV4X4BASIS` constant (line 32 in same file) - 256 floats for the basis matrix.

This is a straightforward port - no SIMD needed for correctness, just the scalar version.
