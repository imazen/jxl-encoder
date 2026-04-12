# Implementation Differences: jxl-encoder vs libjxl

Last updated: 2026-04-12 (source-verified against libjxl C++ code)

Systematic audit against libjxl source. Each item verified against actual C++ code.

---

## REMAINING DIFFERENCES

### Coefficient-domain mode constants (legacy, non-default path)

Only affects `--no-pixel-domain-loss`. The default pixel-domain mode matches libjxl.

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| k8x8mul1 | -0.55 * 0.75 = -0.4125 | -0.4 | **DIFFERS** |
| k8x8mul2 | 1.0736 * 0.75 = 0.8052 | 1.0 | **DIFFERS** |
| k8x8base | 1.4 | 1.4 | **MATCH** |

Extra 0.75 scale factor and different base values. Legacy from initial tuning.
Per-strategy multipliers (k8x16, k16x16, k4x8, k4x4) are a custom extension
not present in libjxl.

---

## MISSING FEATURES

### Input/Format

| Feature | libjxl | Our status | Priority |
|---------|--------|------------|----------|
| FLOAT16 pixel input | `JXL_TYPE_FLOAT16` | Not implemented | Low |
| Configurable bits_per_sample | 12-bit in u16 container | Always full-range | Low |
| Row stride/padding | `JxlPixelFormat.align` | Tightly packed only | Low |
| Endianness parameter | `JxlEndianness` | Native-endian only | Low |
| Premultiplied alpha | `alpha_associated` | Not tracked — **wrong output if set** | Medium |
| Extra channel types | Depth, SpotColor, CFA, etc. | Alpha only; 4 latent serialization bugs | Medium |
| PQ/HLG/BT.709 pixel conversion | Full TF implementations | Header signaling only (f32 workaround) | Low |
| Wide-gamut input (P3, Rec2020) | Linear sRGB via CMS | XYB assumes sRGB primaries — **silent wrong colors for P3** | High |

### Encoding Pipeline

| Feature | libjxl | Our status | Impact |
|---------|--------|------------|--------|
| Streaming frame encoding | `EncodeFrameStreaming` (>2048x2048) | One-shot only | Large images |
| TectonicPlate brute-force | ~20 param combinations for lossless | Not implemented | Lossless |
| 2x/4x/8x resampling | Auto at distance >= 10 | Not implemented | High distance |
| Center-first group permutation | Progressive rendering order | Raster order only | Progressive UX |
| `decoding_speed_tier` | Simplified output for fast decoders | Not implemented | Decoder perf |
| Invisible pixel simplification | Smooth alpha=0 regions | Not implemented | 5-20% on sprites/UI |
| Recursive LfFrame (progressive_dc > 1) | Multi-level DC pyramid | 1 level only | Progressive |
| Dot detection | Star fields, specular highlights | Not implemented (d>=3.0, niche) | Very low |

### Container Format

| Feature | libjxl | Our status | Impact |
|---------|--------|------------|--------|
| `jxll` level box | Codestream level > 5 | Not implemented | Huge images |
| `jxli` frame index box | Random-access animations | Not implemented | Animation |
| `brob` Brotli-compressed metadata | Compressed EXIF/XMP | Not implemented | Metadata size |

### JPEG Reencoding

| Feature | libjxl | Our status | Impact |
|---------|--------|------------|--------|
| CfL for JPEG recompression | `force_cfl_jpeg_recompression` | Not implemented | 1-3% size |
| 4-component JPEG | kMaxComponents = 4 | 1 or 3 only | CMYK JPEG |
| Effort-scaled Brotli | `11 - speed_tier` | Fixed level | Minor |

---

## VERIFIED MATCHES (all constants confirmed identical)

### VarDCT Core Constants

| Constant | Value | Our file | libjxl file |
|----------|-------|----------|-------------|
| K_AC_QUANT | 0.765 | `vardct/adaptive_quant.rs` | `enc_adaptive_quantization.cc:837` |
| DC_QUANT | 1.095924 | `vardct/frame.rs` | `enc_adaptive_quantization.cc:835` |
| DC_QUANT_POW | 0.83 | `vardct/frame.rs` | `enc_adaptive_quantization.cc:836` |
| kFavor2X2 | -0.4 | `vardct/ac_strategy.rs` | `enc_ac_strategy.cc:588` |
| kAvoidEntropyOfTransforms | 0.5 | `vardct/ac_strategy.rs` | `enc_ac_strategy.cc:595` |
| GLOBAL_SCALE_DENOM | 65536 | `vardct/frame.rs` | `quantizer.cc` |
| kLimit (AdjustQuantField) | 1.54138 | `vardct/ac_strategy.rs` | `enc_adaptive_quantization.cc:1209` |
| kMul (AdjustQuantField) | 0.56391 | `vardct/ac_strategy.rs` | `enc_adaptive_quantization.cc:1210` |

### AdjustQuantBias

| Channel | Value | Status |
|---------|-------|--------|
| X (+-1) | 0.94535 | **MATCH** |
| Y (+-1) | 0.92995 | **MATCH** |
| B (+-1) | 0.95006 | **MATCH** |
| Reciprocal | 0.145 | **MATCH** |

### Dead-Zone Thresholds (QuantizeBlockAC)

| | Y | X/B | Status |
|---|---|---|---|
| Thresholds | {0.56, 0.62, 0.62, 0.62} | {0.58, 0.62, 0.62, 0.62} | **MATCH** |
| Multi-block adj | -0.00744 * xsize*ysize | same | **MATCH** |

### Entropy Multipliers (all 12 strategies)

| Strategy | Value | Status |
|----------|-------|--------|
| DCT8 | 0.8 | **MATCH** |
| DCT4X4 | 1.08 | **MATCH** |
| DCT4X8/8X4 | 0.8593 | **MATCH** |
| IDENTITY | 1.0428 | **MATCH** |
| DCT2X2 | 0.95 | **MATCH** |
| AFV0-3 | 0.8178 | **MATCH** |
| DCT16X8 | 1.21 | **MATCH** |
| DCT16X16 | 1.34 | **MATCH** |
| DCT16X32 | 1.49 | **MATCH** |
| DCT32X32 | 1.48 | **MATCH** |
| DCT64X32 | 2.25 | **MATCH** |
| DCT64X64 | 2.25 | **MATCH** |

### Distance-Scaled Cost Model

| Constant | Value | Status |
|----------|-------|--------|
| info_loss_mul base | 1.2 | **MATCH** |
| zeros_mul base | 9.30891 | **MATCH** |
| cost_delta base | 10.83327 | **MATCH** |
| kBias | 0.13732 | **MATCH** |
| kPow (info_loss) | 0.33678 | **MATCH** |
| kPow (zeros_mul) | 0.50991 | **MATCH** |
| kPow (cost_delta) | 0.36703 | **MATCH** |

### Gaborish Inverse Kernel

| Weight | Value | Status |
|--------|-------|--------|
| Orthogonal | -0.09496 | **MATCH** |
| Diagonal | -0.04103 | **MATCH** |
| Ortho dist 2 | 0.01371 | **MATCH** |
| Knight's move | 0.00651 | **MATCH** |
| Corner dist 2 | -0.00148 | **MATCH** |

### XYB Color Space

| Constant | Value | Status |
|----------|-------|--------|
| Opsin matrix row 0 | [0.300, 0.622, 0.078] | **MATCH** |
| Opsin matrix row 1 | [0.230, 0.692, 0.078] | **MATCH** |
| Opsin matrix row 2 | [0.243, 0.205, 0.552] | **MATCH** |
| Opsin bias | 0.003793073 | **MATCH** |
| NEG_BIAS_CBRT | -0.15595 | **MATCH** |

### Noise Synthesis

| Constant | Value | Status |
|----------|-------|--------|
| kNoisePrecision | 1024.0 | **MATCH** |
| kNumNoisePoints | 8 | **MATCH** |
| Laplacian filter | [-0.25,-1,-0.25;-1,5,-1;-0.25,-1,-0.25] | **MATCH** |
| kReg | 0.005 | **MATCH** |
| kAsym | 1.1 | **MATCH** |

### CfL

| Constant | Value | Status |
|----------|-------|--------|
| kDefaultColorFactor | 84 | **MATCH** |
| K_INV_COLOR_FACTOR | 1.0/84.0 | **MATCH** |
| dc_cfl_factor (B) | 0.5 | **MATCH** |
| K_DISTANCE_MULTIPLIER_AC | 1e-9 | **MATCH** |

### Patches

| Constant | Value | Status |
|----------|-------|--------|
| CHANNEL_DEQUANT_XYB | [0.01615, 0.08875, 0.1922] | **MATCH** |
| CHANNEL_WEIGHTS_XYB | [30.0, 3.0, 1.0] | **MATCH** |
| MAX_PATCH_SIZE | 32 | **MATCH** |
| SIMILAR_THRESHOLD | 0.8 | **MATCH** |
| DISTANCE_LIMIT | 50 | **MATCH** |
| MIN_PATCH_OCCURRENCES | 2 | **MATCH** |
| MIN_MAX_PATCH_SIZE | 20 | **MATCH** |

### Splines

| Constant | Value | Status |
|----------|-------|--------|
| DESIRED_RENDERING_DISTANCE | 1.0 | **MATCH** |
| NUM_SPLINE_CONTEXTS | 6 | **MATCH** |
| CHANNEL_WEIGHT | [0.0042, 0.075, 0.07, 0.3333] | **MATCH** |

### ANS Entropy Coding

| Constant | Value | Status |
|----------|-------|--------|
| ANS_LOG_TAB_SIZE | 12 | **MATCH** |
| ANS_TAB_SIZE | 4096 | **MATCH** |
| ANS_MAX_ALPHABET_SIZE | 256 | **MATCH** |
| ANS_SIGNATURE | 0x13 | **MATCH** |
| RECIPROCAL_PRECISION | 44 | **MATCH** |
| Initial state | 0x130000 | **MATCH** |
| Fast shift values | [0, 6, 12] | **MATCH** |

### LZ77

| Constant | Value | Status |
|----------|-------|--------|
| WINDOW_SIZE | 1 << 20 | **MATCH** |
| NUM_SPECIAL_DISTANCES | 120 | **MATCH** |
| Hash chain max steps | 256 | **MATCH** |
| Hash mask | 32768 entries | **MATCH** |
| Hash shift | 5 | **MATCH** |
| min_symbol (ANS) | 224 | **MATCH** |
| min_symbol (Huffman) | 512 | **MATCH** |
| LEN_COST_TABLE | 17 entries | **MATCH** |

### Coefficient Order

| Constant | Value | Status |
|----------|-------|--------|
| NUM_ORDER_BUCKETS | 13 | **MATCH** |
| NUM_PERMUTATION_CONTEXTS | 8 | **MATCH** |
| STRATEGY_TO_BUCKET | 27 entries | **MATCH** |
| Natural order (LLF-first zig-zag) | Algorithm | **MATCH** |
| Lehmer code (Fenwick tree) | Algorithm | **MATCH** |

### ICC Encoding

| Constant | Value | Status |
|----------|-------|--------|
| NUM_ICC_CONTEXTS | 41 | **MATCH** |
| 17 recognized tags | cprt..lumi | **MATCH** |
| 8 type strings | XYZ..gbd | **MATCH** |
| Header prediction template | 128-byte | **MATCH** |
| Varint (7-bit groups) | Algorithm | **MATCH** |
| Prediction orders 0/1/2 | Algorithm | **MATCH** |
| Unshuffle | Algorithm | **MATCH** |

### Quantization Matrices (Parametric Band Parameters)

| Transform | Seed X | Seed Y | Seed B | Bands | Status |
|-----------|--------|--------|--------|-------|--------|
| DCT8 | 3150 | 560 | 512 | 6 | **MATCH** |
| DCT16x16 | 8997 | 3191 | 1158 | 7 | **MATCH** |
| DCT32x32 | 15718 | 7306 | 3804 | 8 | **MATCH** |
| DCT64x64 | 23966 | 8380 | 4493 | 8 | **MATCH** |

Band interpolation: `a * (b/a)^frac` (geometric). Band multiplier: `1+v` if v>0,
`1/(1-v)` if v<=0. **MATCH**.

### DC Quantization Constants

| Constant | Value | Status |
|----------|-------|--------|
| INV_DC_QUANT | [4096.0, 512.0, 256.0] | **MATCH** |

### Adaptive Quantization Pipeline

All constants verified against `enc_adaptive_quantization.cc`:
ComputeMask (kBase, kMul0-4, kOffset2-4), SimpleGamma (kSGmul, kSGmul2, kSGRetMul,
kSGVOffset), GammaModulation (kBias=0.16, kGamma=0.1006), HfModulation (all),
FuzzyErosion (kMulBase, kMulAdd, kTotal=0.2996), MaskingSqrt (kLogOffset, kMul),
Mask1x1 (kScaler=1.0, kMul=1.0, kOffset=0.01), Symmetric5 blur (all 5 weights).
**MATCH**.

### Butteraugli Loop

| Constant | Value | Status |
|----------|-------|--------|
| kPow | [0.2, 0.2, 0, ...] | **MATCH** |
| kOriginalComparisonRound | 1 | **MATCH** |
| Iterations e8 | 2 | **MATCH** |
| Iterations e9+ | 4 | **MATCH** |
| Gate | effort >= 8 (speed_tier <= kKitten) | **MATCH** |

### Squeeze Transform

| Algorithm | Status |
|-----------|--------|
| `smooth_tendency()` cubic formula | **MATCH** |
| `AVERAGE(A,B)` ceiling-biased | **MATCH** |
| MAX_FIRST_PREVIEW_SIZE = 8 | **MATCH** |

### RCT

All 42 RCT variants implemented. YCoCg (type 6) integer arithmetic matches exactly:
`Co = R - B`, `tmp = B + (Co >> 1)`, `Cg = G - tmp`, `Y = tmp + (Cg >> 1)`. **MATCH**.

### Palette

| Constant | Value | Status |
|----------|-------|--------|
| LARGE_CUBE | 5 (125 entries) | **MATCH** |
| SMALL_CUBE | 4 (64 entries) | **MATCH** |
| IMPLICIT_PALETTE_SIZE | 189 | **MATCH** |
| MIN_IMPLICIT_PALETTE_INDEX | -143 | **MATCH** |
| 72 delta palette entries | All values | **MATCH** |
| MAX_PALETTE_COLORS | 1024 | **MATCH** |
| CHANNEL_COLORS_PERCENT | 95.0 | **MATCH** |

### BitWriter

| Feature | Status |
|---------|--------|
| LSB-first, little-endian | **MATCH** |
| MAX_BITS_PER_CALL = 56 | **MATCH** |
| No-accumulator design (read-modify-write) | **MATCH** |
| +8 byte padding | **MATCH** |
| ZeroPadToByte | **MATCH** |
| AppendByteAligned | **MATCH** |
| AppendUnaligned (byte-at-a-time) | **MATCH** |

### Container Format

| Feature | Status |
|---------|--------|
| 32-byte header (12 sig + 20 ftyp) | **MATCH** |
| jxlc box | **MATCH** |
| jxlp boxes (4-byte counter, bit 31 = last) | **MATCH** |
| Exif box (4-byte Tiff offset prefix) | **MATCH** |
| xml box (raw XMP) | **MATCH** |
| jbrd box (JPEG reconstruction) | **MATCH** |

### sRGB Transfer Function

Two-piece formula: `v / 12.92` for `<= 0.04045`, else `((v + 0.055) / 1.055)^2.4`.
**MATCH**.

---

## FIX LOG (resolved items, newest first)

### Entropy Coding

| ID | Issue | Fix |
|----|-------|-----|
| OPT-8 | Max histogram clusters capped at 96 vs 128 | d03697f — changed to 128 matching `kClustersLimit` |
| OPT-7 | No LZ77 for ICC profile encoding | e51a035 — optimal/greedy LZ77 before Huffman |
| OPT-6 | LZ77 distance cost table 11 entries short | a42664b — extended to 139 entries |
| OPT-5 | No RLE in ANS logcount encoding | 7c25011 — RLE for runs of 4+ identical counts |
| OPT-4 | Missing flat distribution cost baseline | dc321e4 — flat-first approach |
| OPT-3 | Missing kBest 28-config HybridUint search | 6900b43 — exact configs from libjxl |
| OPT-2 | ANS population cost approximate in clustering | 71943ac — precise ANS cost function |
| OPT-1 | Missing greedy optimization in RebalanceHistogram | 795a077 — full port of greedy loop |

### Behavioral

| ID | Issue | Fix |
|----|-------|-----|
| DIFF-8 | Custom coefficient orders for DCT64+ | 37e34d5 — skip buckets 7+ |
| DIFF-7 | Reference channel properties missing in tree learning | Added indices 16+, effort-gated |
| DIFF-5 | XYB missing `intensity_target` scaling | 399b697 — `linear_rgb_to_xyb_scaled()` |
| DIFF-4 | XYB missing `ZeroIfNegative` clamp | f534bb4 — `.max(0.0)` before bias |
| DIFF-3 | F16 Inf/NaN not rejected on encode | cf75fe8 — merged with DIFF-1 |
| DIFF-1 | F16 overflow handling: clamp vs reject | cf75fe8 — returns `Result<u16>` |

### Bugs

| ID | Issue | Fix |
|----|-------|-----|
| BUG-2 | Container box size overflow for >4GB | 25a813e — extended 64-bit box headers |
| BUG-1 | `write_u64_coder` broken for values >= 273 | dd2a29a — correct varint loop |

### False Alarms

| ID | Issue | Finding |
|----|-------|---------|
| DIFF-9 | CfL pass 1 missing full weighting | Both use `q=1` for pass 1 — identical |
| DIFF-6 | Modular tree properties differ | `Property` enum unused in production; `compute_spec_properties()` matches |
| DIFF-2 | F16 underflow boundary off-by-one | Different code paths, same output |
