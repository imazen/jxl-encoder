# Side-by-Side Comparison: JPEG XL Lossy Decoders

This document compares the lossy (VarDCT) decoding logic across three implementations:

| Implementation | Language | Repository |
|----------------|----------|------------|
| **libjxl** | C++ | Reference implementation |
| **jxl-rs** | Rust | `/home/lilith/work/jxl-rs` |
| **jxl-oxide** | Rust | `/home/lilith/work/jxl-efforts/jxl-oxide` |

---

## High-Level Pipeline Overview

All three implementations follow the same general VarDCT decoding pipeline:

```
Bitstream → Entropy Decode → Dequantize → Chroma from Luma → IDCT → XYB→RGB
```

---

## Step-by-Step Comparison

### 1. Frame Structure Parsing

| Step | libjxl | jxl-rs | jxl-oxide |
|------|--------|--------|-----------|
| **Entry Point** | `dec_frame.cc` | `frame/decode.rs` | `jxl-frame/src/lib.rs` |
| **Frame Header** | `FrameHeader` struct | `FrameHeader` in `headers/frame_header.rs` | `FrameHeader` in `jxl-frame` |
| **TOC Parsing** | `dec_frame.cc::ReadToc()` | `Toc` struct | `FrameData` |
| **VarDCT Check** | `frame_header.encoding == kVarDCT` | `Encoding::VarDCT` enum | `Encoding::VarDct` enum |

### 2. LF Global Data (Low-Frequency Global)

| Component | libjxl | jxl-rs | jxl-oxide |
|-----------|--------|--------|-----------|
| **File** | `dec_cache.h`, `dec_modular.cc` | `frame/modular/mod.rs:699+` | `jxl-frame/src/data/lf_global.rs` |
| **Quant Params** | `Quantizer` class | `QuantizerParams` | `Quantizer` in `jxl-vardct` |
| **Color Correlation** | `ColorCorrelation` | `ColorCorrelationParams` | `LfChannelCorrelation` |
| **Dequant Matrices** | `DequantMatrices` | `DequantMatrices` | `DequantMatrices` in `jxl-vardct` |
| **Block Context Map** | `BlockContextMap` | `BlockContextMap` | Part of HF global |

### 3. HF Global Data (High-Frequency Global)

| Component | libjxl | jxl-rs | jxl-oxide |
|-----------|--------|--------|-----------|
| **File** | `dec_group.cc` | `frame/decode.rs` | `jxl-frame/src/data/hf_global.rs` |
| **Coeff Order** | `coeff_order.cc` | `coeff_order.rs` | `jxl-vardct/src/hf_coeff.rs` |
| **Histograms** | `ANSCode` in `dec_ans.h` | `Histograms` | `jxl-coding` crate |

### 4. Entropy Decoding (ANS + Huffman)

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **ANS Decoder** | `dec_ans.cc`, `ANSSymbolReader` | `entropy_coding/decode.rs` | `jxl-coding` crate |
| **Huffman** | Embedded in ANS code | Separate module | Separate module |
| **Hybrid Uint** | `ReadHybridUint()` | `HybridUint` decoder | Part of coding |
| **Context Modeling** | `ACContext` in `ac_context.h` | Per-block context | Block context in `jxl-vardct` |

**Key files:**
- libjxl: `lib/jxl/dec_ans.h`, `lib/jxl/dec_ans.cc`
- jxl-rs: `jxl/src/entropy_coding/decode.rs`
- jxl-oxide: `jxl-coding/src/lib.rs`

### 5. AC Coefficient Decoding (Per Group)

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **Main Function** | `DecodeGroupImpl()` | `decode_vardct_group()` | `render_vardct()` calls group parsing |
| **File** | `dec_group.cc:182` | `frame/group.rs:310` | `jxl-render/src/vardct/mod.rs` |
| **Block Iterator** | `GetBlock` interface | Direct iteration | Via `PassGroupParams` |
| **Strategy Check** | `AcStrategy::IsFirstBlock()` | `is_first_block` flag (bit 7) | `BlockInfo` struct |
| **Coefficient Storage** | `ACImage` | `hf_coefficients` | `HfCoeff` in `jxl-vardct` |

**Code pattern for skipping non-first blocks:**
```cpp
// libjxl
if (!acs.IsFirstBlock()) { bx += llf_x; continue; }

// jxl-rs
if !is_first_block { continue; }

// jxl-oxide
// Handled via BlockInfo iteration
```

### 6. Dequantization

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **Function** | `DequantBlock<>()` template | Inline in `decode_vardct_group` | `dequant_hf_varblock_grouped()` |
| **File** | `dec_group.cc`, `quantizer-inl.h` | `frame/group.rs` | `jxl-render/src/vardct/mod.rs:442` |
| **LF Dequant** | From DC image | From `lf_image` | `copy_lf_dequant()` |
| **Quant Matrices** | `DequantMatrices::InvMatrix()` | `dequant_matrices.get()` | `dequant_matrices.get()` |
| **QM Scale** | `x_dm_multiplier`, `b_dm_multiplier` | Same names | `qm_scale` array |

**Dequantization formula (same across all):**
```
coeff_dequant = coeff_quant * dequant_matrix[pos] * quant_scale * qm_scale
```

### 7. Chroma from Luma (CfL)

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **File** | `chroma_from_luma.h` | `frame/group.rs` | `jxl-render/src/vardct/mod.rs` |
| **Y→X Ratio** | `YtoXRatio()` | `color_correlation_params.y_to_x()` | `chroma_from_luma_hf_grouped()` |
| **Y→B Ratio** | `YtoBRatio()` | `color_correlation_params.y_to_b()` | `x_from_y`, `b_from_y` maps |
| **Tile Size** | 64 blocks | `COLOR_TILE_DIM_IN_BLOCKS` = 64 | 64 blocks |

### 8. Inverse DCT (IDCT)

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **Main Function** | `TransformToPixels()` | `jxl_transforms::transform` | `transform_with_lf_grouped()` |
| **File** | `dec_transforms-inl.h` | `jxl_transforms` crate | `jxl-render/src/vardct/transform_common.rs` |
| **Strategy Dispatch** | Via `AcStrategy::Strategy()` | Via `TransformType` | Via `TransformType` |
| **SIMD** | Highway (hwy) library | Architecture-specific modules | `x86_64`, `aarch64`, `wasm32` modules |
| **Supported Sizes** | 2x2 to 256x256 | Same | Same |

**Transform types (same across all):**
- DCT 8x8, 16x16, 32x32 (square)
- DCT 16x8, 8x16, 32x8, etc. (rectangular)
- DCT 256x256, 256x128 (large blocks)
- Hornuss, DCT2, DCT4, AFV variants

### 9. XYB to Linear RGB Transform

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **Function** | `XybToRgb()` / `OpsinToLinear()` | Render pipeline stage | `jxl-color/src/xyb.rs::run()` |
| **File** | `dec_xyb-inl.h`, `dec_xyb.cc` | `render/stages/xyb.rs` | `jxl-color/src/xyb.rs` |
| **Pipeline Stage** | `stage_xyb.cc` | `RenderPipelineInPlaceStage` | Inline in render |
| **Inverse Matrix** | From `OpsinParams` | `OpsinInverseMatrix` | `opsin_inverse_matrix` |

**XYB to Linear RGB formula (same across all):**
```
gamma_l = Y + X
gamma_m = Y - X
gamma_s = B

// Unbias
gamma_l -= cbrt(opsin_bias[0])
gamma_m -= cbrt(opsin_bias[1])
gamma_s -= cbrt(opsin_bias[2])

// Inverse gamma (cube)
linear_l = gamma_l^3 + opsin_bias[0]
linear_m = gamma_m^3 + opsin_bias[1]
linear_s = gamma_s^3 + opsin_bias[2]

// Apply inverse matrix
[R, G, B] = inverse_matrix * [linear_l, linear_m, linear_s]
```

### 10. Render Pipeline Integration

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **Architecture** | `RenderPipeline` with stages | `RenderPipeline` trait | Inline rendering |
| **File** | `render_pipeline/` directory | `render/` directory | `jxl-render/src/` |
| **Threading** | `ThreadPool` | `rayon` integration | `JxlThreadPool` |
| **Group Processing** | `RenderPipelineInput` | `BufferSplitter` | Region-based |

---

## Key Differences

### Architecture Patterns

| Aspect | libjxl | jxl-rs | jxl-oxide |
|--------|--------|--------|-----------|
| **Error Handling** | `Status`, `JXL_RETURN_IF_ERROR` | `Result<T, Error>` | `Result<T>` |
| **Memory** | `JxlMemoryManager` | Standard Rust allocator | `AlignedGrid` with tracker |
| **SIMD** | Highway (compile-time dispatch) | Feature detection + modules | Feature detection + modules |
| **Parallelism** | Custom ThreadPool | Built-in async + rayon | `JxlThreadPool` |

### Crate/Module Organization

| Concern | libjxl | jxl-rs | jxl-oxide |
|---------|--------|--------|-----------|
| **Entropy Coding** | `dec_ans.*`, inline | `entropy_coding/` | `jxl-coding` crate |
| **VarDCT Logic** | `dec_group.cc` | `frame/group.rs` | `jxl-vardct` + `jxl-render` |
| **Transforms** | `dec_transforms-inl.h` | `jxl_transforms` crate | `jxl-render/src/vardct/` |
| **Color** | `dec_xyb.*` | `render/stages/` | `jxl-color` crate |
| **Headers** | `frame_header.*` | `headers/` | `jxl-frame` crate |

---

## File Reference Quick Lookup

### Entropy Decoding
- libjxl: `lib/jxl/dec_ans.cc`, `lib/jxl/dec_ans.h`
- jxl-rs: `jxl/src/entropy_coding/decode.rs`
- jxl-oxide: `jxl-coding/src/lib.rs`

### Group Decoding (VarDCT)
- libjxl: `lib/jxl/dec_group.cc:182` (`DecodeGroupImpl`)
- jxl-rs: `jxl/src/frame/group.rs:310` (`decode_vardct_group`)
- jxl-oxide: `jxl-render/src/vardct/mod.rs:48` (`render_vardct`)

### Dequantization
- libjxl: `lib/jxl/quantizer-inl.h`, `lib/jxl/dec_group.cc`
- jxl-rs: `jxl/src/frame/group.rs`, `jxl/src/frame/quant_weights.rs`
- jxl-oxide: `jxl-render/src/vardct/mod.rs:442` (`dequant_hf_varblock_grouped`)

### Inverse DCT
- libjxl: `lib/jxl/dec_transforms-inl.h` (`TransformToPixels`)
- jxl-rs: `jxl_transforms/src/`
- jxl-oxide: `jxl-render/src/vardct/transform_common.rs`

### XYB Color Transform
- libjxl: `lib/jxl/dec_xyb-inl.h` (`XybToRgb`)
- jxl-rs: `jxl/src/render/stages/xyb.rs`
- jxl-oxide: `jxl-color/src/xyb.rs`

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BITSTREAM                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FRAME HEADER + TOC                                                      │
│  ├─ libjxl:    dec_frame.cc                                             │
│  ├─ jxl-rs:    frame/decode.rs                                          │
│  └─ jxl-oxide: jxl-frame/src/lib.rs                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  LF GLOBAL       │  │  HF GLOBAL           │  │  LF GROUPS           │
│  ├─ Quantizer    │  │  ├─ Coeff Order      │  │  ├─ LF Coeffs        │
│  ├─ CfL params   │  │  ├─ Histograms (ANS) │  │  ├─ AC Strategy      │
│  └─ Dequant mtx  │  │  └─ Context map      │  │  └─ HF Metadata      │
└──────────────────┘  └──────────────────────┘  └──────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PASS GROUPS (HF Coefficients)                                           │
│  Per-group entropy decode → Quantized AC coefficients                   │
│  ├─ libjxl:    dec_group.cc::DecodeGroupImpl                            │
│  ├─ jxl-rs:    frame/group.rs::decode_vardct_group                      │
│  └─ jxl-oxide: jxl-render/src/vardct/mod.rs::render_vardct              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DEQUANTIZATION                                                          │
│  coeff_dequant = coeff_quant * matrix[pos] * scale                      │
│  ├─ libjxl:    quantizer-inl.h::DequantBlock                            │
│  ├─ jxl-rs:    (inline in decode_vardct_group)                          │
│  └─ jxl-oxide: vardct/mod.rs::dequant_hf_varblock_grouped               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CHROMA FROM LUMA (CfL)                                                  │
│  X += Y * x_factor,  B += Y * b_factor                                  │
│  ├─ libjxl:    chroma_from_luma.h                                       │
│  ├─ jxl-rs:    (inline in decode_vardct_group)                          │
│  └─ jxl-oxide: vardct/mod.rs::chroma_from_luma_hf_grouped               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  INVERSE DCT (IDCT)                                                      │
│  Frequency domain → Spatial domain                                      │
│  ├─ libjxl:    dec_transforms-inl.h::TransformToPixels                  │
│  ├─ jxl-rs:    jxl_transforms crate                                     │
│  └─ jxl-oxide: vardct/transform_common.rs                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  XYB → LINEAR RGB                                                        │
│  [X,Y,B] → unbias → cube → matrix multiply → [R,G,B]                    │
│  ├─ libjxl:    dec_xyb-inl.h::XybToRgb                                  │
│  ├─ jxl-rs:    render/stages/xyb.rs                                     │
│  └─ jxl-oxide: jxl-color/src/xyb.rs                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT (Linear RGB / sRGB / other color space)                          │
└─────────────────────────────────────────────────────────────────────────┘
```
