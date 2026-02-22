# Implementation Differences: jxl-encoder-rs vs libjxl

Last updated: 2026-02-22

Systematic audit against `/home/lilith/work/jxl-efforts/libjxl/docs/src/` (55 doc files).
Organized by severity: bugs first, then behavioral differences, then optimizations,
then missing features, then verified matches.

---

## BUGS (will produce invalid bitstreams under certain conditions)

### BUG-1: `write_u64_coder` broken for values >= 273

**File**: `jxl_encoder/src/bit_writer.rs:399-420`
**Severity**: Latent (currently dormant — all U64 fields we write are 0 or small)
**Impact**: Any code path that writes a U64 value >= 273 will produce an invalid bitstream

**What's wrong**: Our selector 3 encoding does not implement the varint continuation
scheme. We write a fixed 12+32 bit format instead of the spec's variable-length
continuation with stop bits.

**Our code** (wrong):
```rust
// For 273..=4368: writes 12 bits directly (MISSING stop bit)
self.write(2, 3)?;
self.write(12, value - 273)?;
// For > 4368: writes 12 bits + flag + 32 raw bits (WRONG format entirely)
self.write(2, 3)?;
self.write(12, low | 0x1000)?; // Set high bit to indicate more data
self.write(32, high)?;
```

**libjxl** (correct, from `enc_fields.cc:129-166`):
```cpp
// Selector 3: varint starting with 12-bit group, then 8-bit groups
writer->Write(2, 3);
writer->Write(12, value & 4095);
value >>= 12;
int shift = 12;
while (value > 0 && shift < 60) {
    writer->Write(1, 1);  // continuation bit
    writer->Write(8, value & 255);
    value >>= 8;
    shift += 8;
}
if (value > 0) {
    writer->Write(1, 1);     // continuation bit
    writer->Write(4, value & 15);  // final 4-bit group (implicitly closed)
} else {
    writer->Write(1, 0);  // stop bit
}
```

**Fix**: Replace our selector-3 branch with the varint loop. Max encoded bits =
2 + 12 + 6*(8+1) + (4+1) = 73 bits. The loop emits continuation bit (1) + 8 data
bits per group, with a final stop bit (0) when value is exhausted or a 4-bit final
group at shift=60.

**Test**: `TestU64Coder` in libjxl's `fields_test.cc` tests values
{0, 1, 15, 16, 17, 271, 272, 273, 4096, 1<<16, 1<<28, (1<<32)-1, 1<<32, 1<<63}.

### BUG-2: Container box size overflow for >4GB payloads

**File**: `jxl_encoder/src/container.rs:170` and `:156`
**Severity**: Latent (no current images this large)
**Impact**: Silent data corruption for files with payloads > 4,294,967,287 bytes

**What's wrong**: `write_box()` casts `(8 + payload.len()) as u32` — silently truncates.
`write_jxlp_box()` casts `(8 + 4 + data.len()) as u32` — same issue.

**Fix**: Check if `box_size > u32::MAX`. If so, write extended box header:
```
[0x00, 0x00, 0x00, 0x01]  // box_size = 1 signals 64-bit extended size
[box_type; 4]
[extended_size as u64; 8]  // 8-byte big-endian total size
[payload]
```
The extended size includes the 16-byte header itself, so `extended_size = 16 + payload.len()`.

Also add the `jxll` box type (4 bytes: `0x00 0x00 0x00 0x0D, "jxll", level_byte`)
for codestream level > 5 support.

---

## BEHAVIORAL DIFFERENCES (valid output, but not matching libjxl exactly)

### DIFF-1: F16 overflow handling — clamp vs reject

**File**: `jxl_encoder/src/f16.rs:27-37`
**Impact**: Values too large for f16 silently clamp to 65504.0 instead of erroring

**Our code**: `if exp == 0xFF → clamp to 0x7BFF`, `if new_exp >= 31 → clamp to 0x7BFF`
**libjxl**: `F16Coder::CanEncode` returns false for Inf/NaN and values with exp > 15.
`F16Coder::Write` returns `JXL_FAILURE` (error status).

**Fix**: Return `Err(Error::InvalidValue)` instead of clamping. Callers should validate
before encoding. Affects: custom DC quant, custom gaborish weights, custom EPF sigma.

### DIFF-2: F16 underflow boundary off-by-one

**File**: `jxl_encoder/src/f16.rs:42`
**Impact**: Values at f32 exponent -25 (biased_exp32=102, ~3e-8) are subnormal in our
code but zero in libjxl

**Our code**: `if new_exp < -10` → zeros at biased_exp32 < 102 (exponent < -25)
**libjxl**: `if exp < -24` → zeros at biased_exp32 < 103 (exponent < -24)

**Fix**: Change `if new_exp < -10` to `if new_exp < -9` (or equivalently,
`if new_exp <= -10`). This makes biased_exp32=102 (exponent -25) map to zero,
matching libjxl.

### DIFF-3: F16 Inf/NaN not rejected on encode

**File**: `jxl_encoder/src/f16.rs:27-29`
**Impact**: Inf → 65504.0, NaN → some value with exponent 0x1F

**Our code**: Clamps Inf to max finite. NaN would encode with mantissa bits intact.
**libjxl**: `F16Coder::CanEncode` rejects both. Spec says NaN/Inf are invalid in headers.

**Fix**: Return error for `exp == 0xFF` instead of clamping. (Merge with DIFF-1 fix.)

### DIFF-4: XYB missing `ZeroIfNegative` clamp for wide-gamut

**File**: `jxl_encoder/src/color/xyb.rs:112`
**Impact**: None for sRGB. Would produce wrong XYB for wide-gamut (P3, Rec2020) inputs
where matrix multiply can produce negative values before bias addition.

**Our code**: `(mixed0 + OPSIN_ABSORBANCE_BIAS[0]).cbrt()` — no clamp before cbrt
**libjxl**: `ZeroIfNegative(mixed0)` before adding bias

For in-gamut sRGB, bias (0.00379) ensures positivity. For wide-gamut inputs with
large negative mixed values, `cbrt(negative)` returns a negative result vs libjxl's
`cbrt(0 + bias)`.

**Fix**: Add `let mixed0 = mixed0.max(0.0)` (and mixed1, mixed2) before bias addition.
Only matters when we add wide-gamut/HDR input support.

### DIFF-5: XYB missing `intensity_target` scaling for HDR

**File**: `jxl_encoder/src/color/xyb.rs:100-108`
**Impact**: None for SDR (intensity_target=255, factor=1.0). Wrong for HDR.

**Our code**: `mixed[c] = sum(matrix[c][i] * rgb[i]) + bias`
**libjxl**: `mixed[c] = sum(matrix[c][i] * rgb[i] * intensity_target/255.0) + bias`

**Fix**: Multiply each linear RGB component by `intensity_target / 255.0` before the
matrix multiply. The `intensity_target` comes from `ImageMetadata` (default 255 for SDR).

### DIFF-6: Modular tree properties differ from libjxl

**File**: `jxl_encoder/src/modular/tree.rs:54-87`
**Impact**: Learned trees make different split decisions. Compression may be slightly
better or worse on different content. Not a correctness bug (both are valid property
sets for the decoder).

**Our properties 4-14**:
| Index | Our property | libjxl property |
|-------|-------------|-----------------|
| 4 | `\|N - NW\|` | `\|N\|` (abs north) |
| 5 | `\|N - W\|` | `\|W\|` (abs west) |
| 6 | `FloorLog2(W)` | `N` (signed north) |
| 7 | `FloorLog2(N)` | `W` (signed west) |
| 8 | `FloorLog2(NW)` | `W - prev_row_gradient` |
| 9 | `\|N - NN\|` | `W + N - NW` (gradient) |
| 10 | `\|W - WW\|` | `W - NW` |
| 11 | `\|NW - NWW\|` | `NW - N` |
| 12 | `\|NE - N\|` | `N - NE` |
| 13 | `\|NW - W\|` | `N - NN` |
| 14 | `\|W\| + \|N\| + \|NW\|` | `W - WW` |

Properties 0-3 (Channel, GroupId, Y, X) and 15 (WpMaxError) match.

**Fix**: Consider switching to libjxl's property set for tree learning parity, or
keep ours if benchmarks show it's equivalent/better. Would require updating
`PropertyValues::compute()` and the `Property` enum.

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

### DIFF-8: Custom coefficient orders for DCT64+ may be unnecessary

**File**: `jxl_encoder/src/vardct/coeff_order.rs`
**Impact**: We encode and signal custom coefficient orders for buckets 7-8 (DCT64x64,
DCT64x32/DCT32x64). libjxl's docs say "orders with bucket > 6 never customized."

**Fix**: Verify whether libjxl actually sends custom orders for these buckets. If not,
gate our custom order encoding to buckets 0-6 only. This would save a few bytes per
frame (the permutation encoding overhead for 4096-element and 2048-element orders).

### DIFF-9: CfL pass 1 missing full weighting

**File**: `jxl_encoder/src/vardct/chroma_from_luma.rs`
**Impact**: CfL pass 1 uses simpler weighting (just inverse quant matrix) vs libjxl's
full weighting (`quantizer_scale * 128 * raw_quant * inv_dequant_matrix`). Pass 2
with full weighting compensates, so impact is minor — pass 1 is just the initial
estimate that pass 2 refines.

**Fix**: Multiply AC coefficient weights by `128.0 * raw_quant_value` in pass 1.
The `128` is `kStrangeMultiplier` from `chroma_from_luma.h`.

---

## ENTROPY CODING OPTIMIZATIONS (valid output, suboptimal compression)

### OPT-1: Missing greedy entropy optimization in RebalanceHistogram

**File**: `jxl_encoder/src/entropy_coding/ans.rs:749-877`
**Impact**: Slightly suboptimal ANS frequency normalization. Estimated <0.1% file size.

**Our code**: Rounds raw counts to normalized precision, absorbs remainder into omit_pos.
**libjxl**: Additionally runs a greedy optimization loop that iteratively adjusts
normalized counts to maximize entropy (minimize cross-entropy loss from rounding).

**Fix**: After initial rounding, iterate over symbols and try +-1 adjustments to each
normalized count, keeping changes that reduce the total cross-entropy cost:
```
loop {
    best_improvement = 0
    for each symbol with freq > 1:
        try freq[sym] -= 1, freq[other] += 1
        if cross_entropy improves: record best
    if no improvement: break
    apply best
}
```

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

### OPT-3: Missing kBest 27-config HybridUint search

**File**: `jxl_encoder/src/entropy_coding/hybrid_uint.rs`
**Impact**: Per-histogram HybridUint config selection tries only 4 configs (kFast)
instead of 27 (kBest). Estimated <0.1% file size.

**Our code**: Tries `[(4,2,0), (4,1,2), (0,0,0), (2,0,1)]`
**libjxl kBest**: Tries all combinations of split_exponent 0-12, msb 0-2, lsb 0-5
where `split >= msb + lsb`. The 27 configs are the valid subset.

**Fix**: Add a `optimize_uint_configs_best()` function that enumerates all valid
`(split, msb, lsb)` triples. Use for `ClusteringType::Best` or effort >= 9.

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

### OPT-6: LZ77 distance cost table 3 entries short

**File**: `jxl_encoder/src/entropy_coding/lz77.rs:72`
**Impact**: Distance cost lookup clamps to entry 127 for distances that would index
128-130. Minimal practical impact since these represent very large distances.

**Our code**: `DIST_COST_TABLE: [f32; 128]`
**libjxl**: 131-entry table

**Fix**: Add 3 more entries to match libjxl's table. The missing values can be
extrapolated from the pattern or copied from libjxl source.

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
