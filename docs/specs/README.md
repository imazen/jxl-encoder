# JPEG XL Specification Resources

## Official Standards

### ISO/IEC 18181 (JPEG XL Standard)
The official ISO standard in four parts:
- **18181-1**: Core codestream (bitstream syntax and semantics)
- **18181-2**: File format (ISOBMFF container)
- **18181-3**: Conformance testing
- **18181-4**: Reference implementation

Available from ISO (paid): https://www.iso.org/standard/77977.html

### Committee Draft (Free, Detailed)
**arXiv:1908.03565** - "JPEG XL Image Coding System"
- PDF: https://arxiv.org/pdf/1908.03565
- Abstract page: https://arxiv.org/abs/1908.03565
- Authors: Rhatushnyak, Wassenberg, et al.
- License: CC BY 4.0

This is the most detailed freely available specification.

## Quick Reference

### File Signatures
- **Naked codestream**: `0xFF 0x0A` (JXL marker)
- **ISOBMFF container**: `0x00000000C 4A584C20 0D0A870A`

### Two Encoding Modes

1. **VarDCT** (lossy): Variable-sized DCT transforms (2x2 to 32x32), XYB color space
2. **Modular** (lossless/lossy): Integer arithmetic, prediction, optional Squeeze transform

### Group Structure
- DC groups: 256x256 DCs (from 2048x2048 pixels)
- AC groups: 256x256 pixels
- Modular groups: 128x128, 256x256, 512x512, or 1024x1024

### VarDCT Pipeline (Encoder)
1. RGB → XYB color space conversion
2. Optional gaborish sharpening
3. DCT with per-block size selection
4. Chroma from luma prediction
5. Adaptive quantization
6. DC prediction
7. Entropy coding (ANS + context modeling)

### Modular Pipeline
1. Optional RCT (Reversible Color Transform)
2. Optional Palette transform
3. Optional Squeeze (Haar-like for progressive)
4. Adaptive prediction (weighted predictor)
5. Entropy coding (ANS)

### Modular Transforms (TransformId)
- 0: RCT (Reversible Color Transform)
- 1: Palette
- 2: Squeeze

## libjxl Documentation

Local copies from `/home/lilith/work/jxl-efforts/libjxl/doc/`:
- `format_overview.md` - High-level format overview
- `xl_overview.md` - Technical architecture overview

## Key libjxl Encoder Files

For porting reference, these are the main encoder source files:

| File | Purpose |
|------|---------|
| `enc_bit_writer.cc/h` | Bitstream writing |
| `enc_ans.cc/h` | ANS entropy encoder |
| `enc_huffman.cc/h` | Huffman encoder |
| `enc_modular.cc/h` | Modular (lossless) encoder |
| `enc_frame.cc/h` | Frame assembly |
| `enc_group.cc/h` | Group encoding |
| `enc_transforms.cc/h` | Color transforms |
| `enc_ac_strategy.cc/h` | AC strategy for VarDCT |
| `enc_xyb.cc/h` | XYB color space conversion |

## VarDCT Frame Structure

```
Frame:
├── FrameHeader
├── TOC (table of contents with group offsets)
├── GlobalModular (optional: shared MA tree, etc.)
├── LFGlobal
│   ├── LF quant factors
│   ├── HF block context map
│   └── LfCoefOrder
├── LF groups (DC + LF AC coefficients, 1:8 resolution)
│   └── Each contains: ModularTree, HfMetadata, DCGlobalInfo
├── HF groups (AC coefficients)
│   └── Each contains: AC coefficients per pass
└── PassGroup data
```

## HfMetadata (VarDCT)

HfMetadata is encoded as a modular substream with 4 channels:
1. `ytox_map` - Chroma from luma X multipliers (8x downsampled)
2. `ytob_map` - Chroma from luma B multipliers (8x downsampled)
3. `transform_image` - DCT block types (8x8 block grid)
4. `epf_map` - Edge-preserving filter strength map (EPF sharpness)

## Entropy Coding

### ANS (Asymmetric Numeral Systems)
- Table-based rANS variant
- Contexts for coefficient magnitudes
- Cluster-based context modeling

### Hybrid Integers
Format: `<token, extra_bits>`
- token encodes split_exponent and MSBs
- extra_bits are raw bits for LSBs

### Huffman
Used for simpler streams (ICC profiles, metadata)

## Color Spaces

### XYB
Perceptual color space used for lossy encoding:
- Derived from LMS cone responses
- Gamma curve with bias for better dark tone encoding
- Y: luminance-like, X/B: chrominance-like

### sRGB/Display-P3/Rec.2100
Common output color spaces - decoder converts from XYB.

## Encode Effort Levels

The `cjxl` encoder offers effort levels 1-9 (plus expert e10) balancing speed vs compression:

| Effort | Modular (Lossless) | VarDCT (Lossy) |
|--------|-------------------|----------------|
| e1-e3 | Huffman→ANS, fixed predictors | Basic tools |
| e4-e6 | Adaptive techniques, variable blocks | Error diffusion, patches |
| e7-e9 | Exhaustive parameter testing | Butteraugli iterations |
| e10 | Expert-only, exhaustive search | Requires `--allow_expert_options` |

**Lossy**: Higher effort = better visual quality at same size, more consistency
**Lossless**: Higher effort = generally smaller files (not guaranteed)

Source: https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md

## WG1 GitLab (Standards Development)

The official JPEG committee working group repository:
- https://gitlab.com/wg1/jpeg-xl

Raw documentation (commit f8874549):
- Format overview: https://gitlab.com/wg1/jpeg-xl/-/raw/f88745497118727f861cb00887cadcb395d10f1c/doc/format_overview.md
- Technical overview: https://gitlab.com/wg1/jpeg-xl/-/raw/f88745497118727f861cb00887cadcb395d10f1c/doc/xl_overview.md

## References

- **jxl-rs (PRIMARY Rust decoder)**: https://github.com/lilith/jxl-rs - Use for roundtrip tests
- libjxl repo: https://github.com/libjxl/libjxl
- jxl-oxide (Rust decoder): https://github.com/tirr-c/jxl-oxide
- JPEG XL website: https://jpegxl.info/
- WG1 GitLab: https://gitlab.com/wg1/jpeg-xl
