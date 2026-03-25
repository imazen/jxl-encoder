# Side-by-Side Decoder Comparison: VarDCT (Lossy) Flow

Comparing jxl-rs, jxl-oxide, and libjxl parsing logic for VarDCT frames.

## Decoder Priority for Testing

**ALWAYS test with jxl-rs first** (PRIMARY), then djxl/libjxl, then jxl-oxide.

| Priority | Decoder | Location | Crate | GitHub |
|----------|---------|----------|-------|--------|
| 1 (PRIMARY) | **jxl-rs** | `~/work/jxl-rs` | `jxl` | https://github.com/lilith/jxl-rs |
| 2 | djxl (libjxl) | `~/work/jxl-efforts/libjxl` | CLI tool | https://github.com/libjxl/libjxl |
| 3 | jxl-oxide | `~/work/jxl-efforts/jxl-oxide` | `jxl-oxide` | https://github.com/tirr-c/jxl-oxide |

## Frame Data Order (VarDCT)

```
Frame:
├── FrameHeader
├── TOC (table of contents)
├── Section 0: LfGlobal
│   ├── Patches (optional)
│   ├── Splines (optional)
│   ├── Noise (optional)
│   ├── LfChannelDequantization
│   ├── LfGlobalVarDct (quantizer, HfBlockContext, LfChannelCorrelation)
│   └── GlobalModular (MA config, global modular subimage)
├── Section 1: HfGlobal
│   ├── DequantMatrices
│   ├── num_histograms (if multi-group)
│   ├── CoeffOrder
│   └── HistogramSet (for AC coefficients)
├── Sections 2..(1+num_lf_groups): LfGroup[i]
│   ├── extra_precision (2 bits)
│   ├── LfCoeff (VarDCT DC coefficients as modular substream)
│   ├── ModularLF (extra channels, optional)
│   └── HfMetadata (modular substream with 4 channels)
├── Sections after LF groups: PassGroup[pass][group]
│   └── AC coefficients
```

## ModularHeader / GroupHeader

Both decoders parse identical structure for each modular substream:

| Field | libjxl (encoding.h:32-55) | jxl-oxide (lib.rs:126-134) |
|-------|---------------------------|---------------------------|
| use_global_tree | `Bool(false, &use_global_tree)` | `ty(Bool)` |
| wp_header | `VisitNested(&wp_header)` | `ty(Bundle(WpHeader))` |
| nb_transforms | `U32(Val(0), Val(1), BitsOffset(4,2), BitsOffset(8,18))` | `ty(U32(0, 1, 2+u(4), 18+u(8)))` |
| transforms | Loop: `VisitNested(&transforms[i])` | `ty(Vec[Bundle(TransformInfo)])` |

### U32 Encoding for nb_transforms
- Selector 0 (2 bits = 00): value 0
- Selector 1 (2 bits = 01): value 1
- Selector 2 (2 bits = 10): read 4 more bits, add 2 → values 2-17
- Selector 3 (2 bits = 11): read 8 more bits, add 18 → values 18-273

## WpHeader (Weighted Predictor Parameters)

| Field | libjxl | jxl-oxide |
|-------|--------|-----------|
| all_default | `Bool(true, &all_default)` | `default_wp: Bool` |
| (if !all_default) | Read 5 parameters | Read 5 parameters |

When `all_default=true`, no additional fields are read.

## TransformInfo (TransformId)

| Value | Transform | libjxl | jxl-oxide (transform.rs:97-112) |
|-------|-----------|--------|--------------------------------|
| 0 | RCT | `kRCT` | `TransformInfo::Rct` |
| 1 | Palette | `kPalette` | `TransformInfo::Palette` |
| 2 | Squeeze | `kSqueeze` | `TransformInfo::Squeeze` |
| 3 | **ERROR** | Invalid | `InvalidEnum { name: "TransformId", value: 3 }` |

The TransformId is read as 2 bits. Value 3 is invalid.

## LfGroup Parsing (VarDCT)

### jxl-oxide (lf_group.rs:32-130)

```
1. LfCoeff::parse() if VarDCT && !using_lf_frame
   - extra_precision: 2 bits
   - Modular image (3 channels: Y, X, B at LF resolution)
   - subimage_idx = 1 + lf_group_idx

2. mlf_group decode (modular LF group for extra channels)
   - subimage_idx = 1 + num_lf_groups + lf_group_idx

3. HfMetadata::parse() if VarDCT && mlf_group complete
   - Modular image (4 channels)
   - subimage_idx = 1 + 2*num_lf_groups + lf_group_idx
```

### libjxl (dec_group.cc / enc_group.cc)

```
1. extra_precision (2 bits)
2. VarDCTLF modular stream (DC coefficients)
3. ModularLF stream (extra channels)
4. HF metadata modular stream
```

## HfMetadata Structure

4 modular channels encoded as single substream:

| Channel | Name | Dimensions | Content |
|---------|------|------------|---------|
| 0 | x_from_y / ytox_map | (tiles_x, tiles_y) | Chroma-from-luma X correlation |
| 1 | b_from_y / ytob_map | (tiles_x, tiles_y) | Chroma-from-luma B correlation |
| 2 | block_info / transform_image | (count, 2) | (dct_select, raw_quant-1) pairs |
| 3 | sharpness / epf_map | (blocks_x, blocks_y) | EPF sigma values |

Where:
- tiles_x = blocks_x / 8 (rounded up)
- tiles_y = blocks_y / 8 (rounded up)
- count = number of varblocks (for DCT8-only: = num_blocks)

## Entropy Coding (Histogram Set)

Both decoders parse histograms identically:

```
1. lz77_enabled (1 bit)
2. If num_contexts > 1: context_map
3. use_prefix_code (1 bit)
4. If use_prefix_code:
   - IntegerConfig (log_alphabet_size=15 implicit)
   - alphabet_size (VarLenUint16)
   - Huffman table
5. Else:
   - ANS distribution tables
```

## Our Encoder Issue: TransformId=3 Error

> **Resolved (January 4, 2026).** This bug was fixed; all VarDCT sizes decode correctly.

The error `InvalidEnum { name: "TransformId", value: 3 }` means:
- jxl-oxide is reading 2 bits and getting value 3
- This happens during HfMetadata modular substream parsing
- The GroupHeader's nb_transforms field is being misread

### Likely Cause

Our `write_group_header()` writes:
```
use_global_tree = 0 (1 bit)
wp_header.all_default = 1 (1 bit)
nb_transforms = 0 (2 bits, selector 0)
```

But if bit alignment is wrong, the decoder could read:
- use_global_tree from wrong position
- Or nb_transforms bits contain garbage

### Debug Checklist

1. Verify bit position before GroupHeader write matches decoder expectation
2. Check that LfCoeff modular stream ends at correct bit boundary
3. Verify HfMetadata "count" field is written correctly (uses ceil_log2 bits)
4. Compare byte-by-byte with libjxl reference output

## Key File Locations

### jxl-rs (PRIMARY - github.com/lilith/jxl-rs)
| Component | File |
|-----------|------|
| Decoder API | `jxl/src/api/decoder.rs` |
| Modular decode | `jxl/src/frame/modular/` |
| VarDCT decode | `jxl/src/frame/vardct/` |
| Entropy coding | `jxl/src/entropy_coding/` |
| Headers | `jxl/src/headers/` |
| BitReader | `jxl/src/bit_reader.rs` |

### jxl-oxide
| Component | File | Lines |
|-----------|------|-------|
| LfGroup parsing | `crates/jxl-frame/src/data/lf_group.rs` | 32-130 |
| LfCoeff parsing | `crates/jxl-vardct/src/lf.rs` | 138-182 |
| HfMetadata parsing | `crates/jxl-vardct/src/hf_metadata.rs` | 55-229 |
| ModularHeader | `crates/jxl-modular/src/lib.rs` | 126-134 |
| TransformInfo | `crates/jxl-modular/src/transform.rs` | 97-112 |
| WpHeader | `crates/jxl-modular/src/predictor.rs` | bundle def |

### libjxl
| Component | File |
|-----------|------|
| GroupHeader | `lib/jxl/modular/encoding/encoding.h:32-55` |
| Transform | `lib/jxl/modular/transform/transform.h` |
| WpHeader | `lib/jxl/modular/encoding/weighted.h` |
| LfGroup encode | `lib/jxl/enc_group.cc` |
| Modular encode | `lib/jxl/modular/encoding/enc_encoding.cc` |

### Our Encoder
| Component | File |
|-----------|------|
| write_group_header | `jxl_encoder/src/vardct/encoder.rs:940-960` |
| write_hf_metadata | `jxl_encoder/src/vardct/encoder.rs:856-938` |
| write_vardct_modular_substream | `jxl_encoder/src/modular/improved.rs:491-611` |
