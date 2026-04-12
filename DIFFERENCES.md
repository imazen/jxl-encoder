# Implementation Differences: jxl-encoder-rs vs libjxl

Last updated: 2026-04-12 (source-verified against libjxl C++ code, fixes applied)

Systematic audit against `/home/lilith/work/jxl-efforts/libjxl/docs/src/` (55 doc files).
Each item verified against actual libjxl source code (not just docs).
Organized by severity: bugs first, then behavioral differences, then optimizations,
then missing features, then verified matches.

**Score: 15 fixed, 3 false alarms, 1 remaining (1 DIFF)**

---

## BUGS (will produce invalid bitstreams under certain conditions)

### ~~BUG-1~~: `write_u64_coder` broken for values >= 273 — FIXED (dd2a29a)

**File**: `jxl_encoder/src/bit_writer.rs:399-430`
**Status**: FIXED. Replaced incorrect fixed-width encoding with correct varint loop
matching libjxl `enc_fields.cc:129-166`. Added roundtrip tests for all values from
libjxl's `TestU64Coder` (0, 1, 15, 16, 17, 271, 272, 273, 4096, 1<<16, 1<<28,
(1<<32)-1, 1<<32, 1<<63). Was latent — all current callers pass values 0-179.

### ~~BUG-2~~: Container box size overflow for >4GB payloads — FIXED (25a813e)

**File**: `jxl_encoder/src/container.rs`
**Status**: FIXED. All three box-writing functions (`write_box`, `write_jxlp_box`,
`write_exif_box`) now use extended 64-bit box headers (size=1 + 8-byte extended size)
when total_size > u32::MAX. Also extracted Exif box into `write_exif_box()` helper.
Was latent — no current images this large.

**Not yet done**: `jxll` level box for codestream level > 5 (separate feature).

---

## BEHAVIORAL DIFFERENCES (valid output, but not matching libjxl exactly)

### ~~DIFF-1~~: F16 overflow handling — clamp vs reject — FIXED (cf75fe8)

**Status**: FIXED. `f32_to_f16_bits()` now returns `Result<u16>`, rejecting Inf, NaN,
and overflow (|value| > 65504). Matches libjxl `F16Coder::CanEncode` behavior.

### ~~DIFF-2~~: ~~F16 underflow boundary off-by-one~~ FALSE ALARM

**Verified against source**: Both implementations produce identical output for all inputs.
Our subnormal path for biased_exp32=102 computes `half_mantissa = (m >> (13+shift)) = 0`
(zero), same result as libjxl's early-out `if (exp < -24) return 0`. Different code paths,
same output. No fix needed.

### ~~DIFF-3~~: F16 Inf/NaN not rejected on encode — FIXED (cf75fe8)

**Status**: FIXED. Merged with DIFF-1 fix. `f32_to_f16_bits()` returns error for
`exp == 0xFF` (Inf/NaN), matching libjxl `F16Coder::CanEncode`.

### ~~DIFF-4~~: XYB missing `ZeroIfNegative` clamp — FIXED (f534bb4)

**Status**: FIXED. Added `.max(0.0)` clamp to mixed values before bias addition,
matching libjxl `enc_xyb.cc:91-95`. No-op for sRGB, prevents `cbrt(negative)` for
wide-gamut inputs.

### ~~DIFF-5~~: XYB missing `intensity_target` scaling — FIXED (399b697)

**Status**: FIXED. Added `linear_rgb_to_xyb_scaled()` with explicit intensity_target
parameter. Existing `linear_rgb_to_xyb()` uses default 255.0 (no-op for SDR).
Matches libjxl `ComputePremulAbsorb` (enc_xyb.cc:214-228).

### ~~DIFF-6~~: ~~Modular tree properties differ from libjxl~~ FALSE ALARM (dead code)

**Verified against source**: The `Property` enum in `tree.rs:54-87` is unused in production.
The production tree learning in `tree_learn.rs` uses `compute_spec_properties()` with
inline index constants matching libjxl's `context_predict.h:534-554` exactly.
`Property`/`PixelProperties` are only used by `traverse_tree` (test helper). Re-exported
in public API but has no external consumers.

### ~~DIFF-7~~: Reference channel properties (16+) in tree learning — FIXED

**Status**: FIXED. Added dynamic reference channel properties (indices 16+) to
modular tree learning. For each channel, preceding channels with matching dimensions
contribute 4 properties: |ref_value|, ref_value, |ref_value - clamped_gradient|,
ref_value - clamped_gradient. Effort-gated: e<9 uses only gradient residual (offset 3)
per ref channel, e9+ uses all 4. Squeeze and lossy paths excluded (channels have
different dimensions). 11.4% compression improvement on cross-channel correlated data.
Pixel-exact roundtrip verified with jxl-rs.

### ~~DIFF-8~~: Custom coefficient orders for DCT64+ — FIXED (37e34d5)

**Status**: FIXED. Added `if bucket > 6 { continue; }` in `compute_custom_orders()`,
matching libjxl's `ComputeUsedOrders` (`enc_coeff_order.cc:53-58`). Buckets 7+
(DCT64x64, DCT64x32/DCT32x64) now use natural (default) order.

### ~~DIFF-9~~: ~~CfL pass 1 missing full weighting~~ FALSE ALARM

**Verified against source**: libjxl's `enc_chroma_from_luma.cc:326-331` shows pass 1
uses `q = use_dct8 ? 1 : quantizer->Scale() * 128 * qq`. When `use_dct8=true` (pass 1),
`q=1` — no multiplier applied. Our pass 1 also uses `q=1`. Our pass 2 correctly uses
`quant_scale * 128.0 * qq`. Both passes match libjxl exactly.

---

## ENTROPY CODING OPTIMIZATIONS (valid output, suboptimal compression)

### ~~OPT-1~~: Missing greedy entropy optimization in RebalanceHistogram — FIXED (795a077)

**File**: `jxl_encoder/src/entropy_coding/ans.rs`
**Status**: FIXED. Ported libjxl's `RebalanceHistogram` greedy optimization
(enc_ans.cc:416-559). Key components:
- `build_allowed_counts(shift)`: builds sorted table of all representable count values
- `find_allowed_leq()`: snaps counts DOWN to nearest allowed value (prevents rest < 0)
- Greedy loop navigates allowed counts by index, computing entropy deltas with f64 log2.
  Each iteration picks the best bin to increment or decrement, with the balancing bin
  (highest-frequency symbol) absorbing changes. Tractor logic handles out-of-range rest.
- Omit enforcement loop ensures decoder's omit_pos detection matches encoder.

### ~~OPT-2~~: ANS population cost is approximate in clustering — FIXED (71943ac)

**File**: `jxl_encoder/src/entropy_coding/ans.rs`
**Status**: FIXED. Replaced Shannon cross-entropy approximation with precise ANS
cost function matching libjxl's `EstimateDataBits` (enc_ans.cc:362-370).
New `estimate_data_bits_normalized()` computes `cost = total * ANS_LOG_TAB_SIZE -
sum(actual_count * log2(norm_count))` using actual symbol counts against normalized
ANS table counts. Uses f64 for precision. Affects shift selection in `from_histogram`.

### ~~OPT-3~~: Missing kBest 28-config HybridUint search — FIXED (6900b43)

**Status**: FIXED. Added `optimize_uint_configs_best()` with exact 28 curated configs
from libjxl `enc_ans.cc:747-783`. Activated when enhanced_clustering is enabled.
Per-histogram Shannon entropy + extra bits + signaling cost evaluation.

### ~~OPT-4~~: Missing flat distribution cost baseline in ANS strategy selection — FIXED (dc321e4)

**Status**: FIXED. `from_histogram()` now initializes with flat distribution cost
(header + `total_count * log2(alphabet_size)`) before trying shift-based encodings.
Matches libjxl's flat-first approach (`enc_ans.cc:97-102`).

### ~~OPT-5~~: No RLE in ANS logcount encoding — FIXED (7c25011)

**Status**: FIXED. Added RLE compression for runs of 4+ consecutive symbols with
identical normalized counts. Emits RLE marker (symbol 13) + VarLenUint8(run-4).
Skips precision bits for RLE-covered positions (matching decoder). Runs check
actual counts and break at omit_pos boundaries.

### ~~OPT-6~~: LZ77 distance cost table 11 entries short — FIXED (a42664b)

**Status**: FIXED. Extended `DIST_COST_TABLE` from 128 to 139 entries, adding 11
special distance code costs (2.4-9.7 range) from libjxl `enc_lz77.cc:442-446`.
Previously clamped to entry 127 (17.2), dramatically undervaluing vertical matches.

### ~~OPT-7~~: No LZ77 for ICC profile encoding — FIXED (e51a035)

**Status**: FIXED. ICC profiles now use LZ77 backward references before Huffman
entropy coding. Optimal LZ77 for small profiles (<16KB), greedy for larger.
Matches libjxl `enc_icc_codec.cc:455-482`.

### ~~OPT-8~~: Max histogram clusters capped at 96 vs 128 — FIXED (d03697f)

**Status**: FIXED. Changed caller-side cap from 96 to 128, matching libjxl
`kClustersLimit = 128` (`enc_context_map.h:24`).

---

## MISSING FEATURES (not implemented)

### Input/Format

| Feature | libjxl | Our status | Priority |
|---------|--------|------------|----------|
| FLOAT16 pixel input | `JXL_TYPE_FLOAT16` | Not implemented | Low |
| Configurable bits_per_sample | 12-bit in u16 container | Always full-range | Low |
| Row stride/padding | `JxlPixelFormat.align` | Tightly packed only | Low |
| Endianness parameter | `JxlEndianness` | Native-endian only | Low |
| Premultiplied alpha | `alpha_associated` | Not tracked | Low |
| Extra channel types | Depth, SpotColor, CFA, etc. | Alpha only | Medium |
| PQ/HLG/BT.709 pixel conversion | Full TF implementations | Header signaling only | Low |
| Wide-gamut input (P3, Rec2020) | CMS integration | sRGB only | Low |

### Encoding Pipeline

| Feature | libjxl | Our status | Impact |
|---------|--------|------------|--------|
| Streaming frame encoding | `EncodeFrameStreaming` (>2048x2048) | One-shot only | Large images |
| TectonicPlate brute-force | ~20 param combinations for lossless | Not implemented | Lossless |
| 2x/4x/8x resampling | Auto at distance >= 10 | Not implemented | High distance |
| Center-first group permutation | Progressive rendering order | Raster order only | Progressive UX |
| `decoding_speed_tier` | Simplified output for fast decoders | Not implemented | Decoder perf |
| Invisible pixel simplification | Smooth alpha=0 regions | Not implemented | ~0.5% size |
| Multi-threading | Group-level parallelism | Group-level parallelism (rayon, opt-in `parallel` feature) | Encode speed |
| `FindBestQuantizationHQ` | 5-iter max-error at Tortoise | Standard 2-4 iter only | Quality e9+ |
| `FindBestQuantizationMaxError` | For LfFrame DC quality | Not implemented | LfFrame quality |
| Recursive LfFrame (progressive_dc > 1) | Multi-level DC pyramid | 1 level only | Progressive |
| Dot detection | Star fields, specular highlights | Not implemented | Niche |
| Phase 3 non-aligned merging | Effort 8-9 feature | Not implemented | Quality e8+ |

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

## COEFFICIENT-DOMAIN MODE (legacy, not default path)

These differences only affect the non-default coefficient-domain mode
(`--no-pixel-domain-loss`). The default pixel-domain mode matches libjxl.

| Constant | Rust | libjxl | Status |
|----------|------|--------|--------|
| k8x8mul1 | -0.55 * 0.75 = -0.4125 | -0.4 | **DIFFERS** |
| k8x8mul2 | 1.0736 * 0.75 = 0.8052 | 1.0 | **DIFFERS** |
| k8x8base | 1.4 | 1.4 | **MATCH** |

Extra 0.75 scale factor and different base values. Legacy from initial tuning.
Per-strategy multipliers (k8x16, k16x16, k4x8, k4x4) are a custom extension
not present in libjxl.
