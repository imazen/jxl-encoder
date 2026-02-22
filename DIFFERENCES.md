# Implementation Differences: jxl-encoder-rs vs libjxl

Last updated: 2026-02-22 (source-verified against libjxl C++ code)

Systematic audit against `/home/lilith/work/jxl-efforts/libjxl/docs/src/` (55 doc files).
Each item verified against actual libjxl source code (not just docs).
Organized by severity: bugs first, then behavioral differences, then optimizations,
then missing features, then verified matches.

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

**Verified against source**: The `Property` enum in `tree.rs:54-87` is **dead code**.
The production tree learning in `tree_learn.rs:416-454` uses `compute_spec_properties()`
which matches libjxl's `context_predict.h:534-554` exactly: |N|, |W|, N signed, W signed,
W-prev_gradient, W+N-NW, W-NW, NW-N, N-NE, N-NN, W-WW.

**Cleanup**: The stale `Property` enum in `tree.rs` should be deleted to avoid confusion.

### DIFF-7: Missing reference channel properties (16+) in tree learning

**File**: `jxl_encoder/src/modular/tree.rs:94`
**Impact**: Cannot use cross-channel prediction in MA tree. libjxl uses 4 properties
per reference channel (ref value, abs diff, ref row above, abs diff row above) for
multi-channel correlation.

**Our code**: `NUM_PROPERTIES = 16` (no reference channel properties)
**libjxl**: `kNumNonrefProperties = 16` + 4 per reference channel

**Fix**: Add properties 16+ for each already-decoded channel. The decoder already
handles these (property indices >= 16 reference channels `(prop - 16) / 4`). Would
improve RGB compression where channels are correlated.

### DIFF-8: Custom coefficient orders for DCT64+ are unnecessary overhead

**File**: `jxl_encoder/src/vardct/coeff_order.rs`
**Verified**: libjxl's `ComputeUsedOrders` (`enc_coeff_order.cc:53-58`) has
`if (ord > 6) continue;` — it explicitly skips customization for buckets 7+.
**Impact**: We encode and signal custom coefficient orders for buckets 7-12 (DCT64x64,
DCT64x32/DCT32x64, etc.). This adds permutation encoding overhead for orders with
4096 and 2048 elements that libjxl never customizes.

**Fix**: Gate custom order encoding to buckets 0-6 only, matching libjxl. Use natural
(default) order for buckets 7+. Saves a few bytes per frame.

### ~~DIFF-9~~: ~~CfL pass 1 missing full weighting~~ FALSE ALARM

**Verified against source**: libjxl's `enc_chroma_from_luma.cc:326-331` shows pass 1
uses `q = use_dct8 ? 1 : quantizer->Scale() * 128 * qq`. When `use_dct8=true` (pass 1),
`q=1` — no multiplier applied. Our pass 1 also uses `q=1`. Our pass 2 correctly uses
`quant_scale * 128.0 * qq`. Both passes match libjxl exactly.

---

## ENTROPY CODING OPTIMIZATIONS (valid output, suboptimal compression)

### OPT-1: Missing greedy entropy optimization in RebalanceHistogram

**File**: `jxl_encoder/src/entropy_coding/ans.rs:749-877`
**Impact**: Slightly suboptimal ANS frequency normalization. Estimated <0.1% file size.

**Our code**: Rounds raw counts to normalized precision, absorbs remainder into omit_pos.
**libjxl**: Uses a sophisticated `AllowedCounts` table-driven greedy optimization
(`enc_ans.cc:416-559`) that is more complex than simple +-1 adjustments.

**Fix**: Port libjxl's `RebalanceHistogram` from `enc_ans.cc:416-559`. Key components:
1. **AllowedCounts table** (`enc_ans.cc:318-400`): Pre-computes for each `(count, shift)`
   pair the set of valid normalized counts, along with log2 deltas for each transition.
   This constrains the search space to ensure each symbol's count maps to a valid ANS
   table entry (power-of-2 or specific bit patterns depending on ANS_LOG_TAB_SIZE).
2. **Greedy loop**: For each symbol, evaluates the cost delta of moving its normalized
   count up or down within the AllowedCounts set. Picks the best (count, direction) pair
   that reduces total cost, compensating another symbol. Repeats until no improvement.
3. **Fixed-point log2 cost**: Uses the same 4097-entry lookup table as `ANSPopulationCost`
   (see OPT-2) for precise cost evaluation during the greedy search.
4. **Precision constraint**: All normalized counts must be valid for the ANS table size
   (sum to `1 << ANS_LOG_TAB_SIZE = 4096`).

### OPT-2: ANS population cost is approximate in clustering

**File**: `jxl_encoder/src/entropy_coding/ans.rs:912-924`
**Impact**: Histogram merge decisions in Best clustering mode use imprecise cost model.
Estimated 0.1-0.3% file size impact.

**Our code**: `cost = -sum(count * log2(count/total))` (Shannon cross-entropy)
**libjxl**: `ANSPopulationCost()` normalizes histogram to ANS table and computes the
real coding cost using fixed-point log2 with normalized counts. Includes the actual
rounding loss from normalization.

**Fix**: Implement `ans_population_cost()` that normalizes the histogram to
`ANS_TAB_SIZE=4096` and computes `cost = total * ANS_LOG_TAB_SIZE - sum(count * log2(norm_count))`.
Use this in `histogram_distance()` when `ClusteringType::Best`.

### OPT-3: Missing kBest 28-config HybridUint search

**File**: `jxl_encoder/src/entropy_coding/hybrid_uint.rs`
**Impact**: Per-histogram HybridUint config selection tries only 4 configs (kFast)
instead of 28 (kBest). Estimated <0.1% file size.

**Our code**: Tries `[(4,2,0), (4,1,2), (0,0,0), (2,0,1)]`
**libjxl kBest** (`enc_ans.cc:747-783`): 28 curated `(split, msb, lsb)` configs, NOT
all valid triples. The list includes configs with split_exponents 0-12 and specific
msb/lsb combinations chosen for good coverage:
```
(0,0,0), (1,0,0), (2,0,0), (2,0,1), (3,0,0), (3,1,0), (3,0,1), (3,1,1),
(4,0,0), (4,2,0), (4,1,0), (4,0,1), (4,2,1), (4,1,1), (5,0,0), (5,2,0),
(5,1,0), (5,0,1), (5,2,1), (6,0,0), (6,2,0), (6,1,0), (7,0,0), (7,2,0),
(8,0,0), (8,2,0), (10,0,0), (12,0,0)
```

**Fix**: Add a `optimize_uint_configs_best()` function with the exact 28 configs.
Use for `ClusteringType::Best` or effort >= 9.

### OPT-4: Missing flat distribution cost baseline in ANS strategy selection

**File**: `jxl_encoder/src/entropy_coding/ans.rs`
**Impact**: May miss cases where flat distribution (all symbols equiprobable) is cheaper
than the computed distribution. Rare in practice.

**libjxl**: Always computes `flat_cost = total_count * ANS_LOG_TAB_SIZE` as a baseline
and picks the cheaper of flat vs computed. Our code only uses flat when detected by
`is_flat()` in the standalone distribution path.

**Fix**: In `ANSEncodingHistogram::from_histogram()`, compute flat cost and compare
with best computed cost. If flat is cheaper, use method=0 (flat encoding).

### OPT-5: No RLE in ANS logcount encoding

**File**: `jxl_encoder/src/entropy_coding/ans.rs` (histogram serialization)
**Impact**: Slightly larger histogram headers when there are long runs of zero-frequency
symbols. Estimated <0.05% file size.

**libjxl**: When 5+ consecutive symbols have the same logcount, emits RLE marker
(symbol 13 in the logcount Huffman code) + run length. Our code writes every
logcount individually.

**Fix**: During logcount serialization, detect runs of length >= 5 with the same
logcount value. Emit `[logcount, RLE_MARKER, run_length - 5]` instead of repeating
the logcount. The RLE marker is symbol index 13 in the logcount prefix code.

### OPT-6: LZ77 distance cost table 11 entries short

**File**: `jxl_encoder/src/entropy_coding/lz77.rs:72`
**Impact**: Distance cost lookup clamps to entry 127 for distances that would index
128-138. The missing 11 entries represent **special distance codes** with costs in the
2.4-9.7 range, dramatically lower than our clamped value of 17.2. This significantly
undervalues special distance codes in LZ77 cost estimation, potentially causing the
optimizer to reject beneficial matches that use special distances.

**Our code**: `DIST_COST_TABLE: [f32; 128]`
**libjxl** (`enc_lz77.cc:399-447`): 139-entry table. Entries 128-138 are the costs
for special distance codes (codes that encode distances as multiples of the image width,
useful for vertical matches in image data).

**Fix**: Extend `DIST_COST_TABLE` to 139 entries. Copy the 11 missing values from
libjxl's `kDistCost` table (`enc_lz77.cc:399-447`). These entries have dramatically
lower costs than our current clamp value.

### OPT-7: No LZ77 for ICC profile encoding

**File**: `jxl_encoder/src/icc.rs:764`
**Impact**: ICC profiles are Huffman-only without LZ77. For large ICC profiles (>16KB),
libjxl uses optimal LZ77; for smaller profiles, greedy LZ77. Typical ICC profiles
are 0.5-4KB, so impact is modest (estimated 5-15% larger ICC encoding).

**Our code**: `writer.write(1, 0)` — LZ77 disabled
**libjxl**: `use_lz77 = true` with optimal method for size < 16KB, greedy otherwise

**Fix**: Enable LZ77 for ICC streams. Use the existing `apply_lz77_optimal()` or
`apply_lz77_backref()` on the ICC token stream before entropy coding. The LZ77
header should use the same parameters as other streams (min_symbol=224 for ANS,
512 for Huffman).

### OPT-8: Max histogram clusters capped at 96 vs 128

**File**: `jxl_encoder/src/entropy_coding/cluster.rs:19`
**Impact**: At most 96 histogram clusters for ANS. libjxl allows up to 128 in
Fast/Best modes. More clusters can sometimes capture distinct distribution patterns.
Estimated <0.1% impact on typical images.

**Our code**: `CLUSTERS_LIMIT = 256` but caller caps at 96
**libjxl**: kFast/kBest allow 128

**Fix**: Increase the caller-side cap from 96 to 128 for the Best clustering mode.

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
| Multi-threading | Group-level parallelism | Single-threaded | Encode speed |
| `FindBestQuantizationHQ` | 5-iter max-error at Tortoise | Standard 2-4 iter only | Quality e9+ |
| `FindBestQuantizationMaxError` | For LfFrame DC quality | Not implemented | LfFrame quality |
| Recursive LfFrame (progressive_dc > 1) | Multi-level DC pyramid | 1 level only | Progressive |
| Dot detection | Star fields, specular highlights | Not implemented | Niche |
| Phase 3 non-aligned merging | Effort 8-9 feature | Not implemented | Quality e8+ |

### Container Format

| Feature | libjxl | Our status | Impact |
|---------|--------|------------|--------|
| Extended 64-bit box headers | For >4GB payloads | Not implemented | See BUG-2 |
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
