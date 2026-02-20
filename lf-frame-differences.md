# LfFrame (DC Frame) Implementation Differences

## Overview

The LfFrame (frame_type=1, dc_level=1) encodes DC coefficients as a separate
modular frame at 1/8 resolution. The decoder processes it first, making DC
data available before the main VarDCT frame's AC coefficients arrive.

**Rust**: `jxl_encoder/src/vardct/lf_frame.rs` (470 lines)
**libjxl**: `lib/jxl/enc_cache.cc:117-223`, `lib/jxl/enc_modular.cc:754-825`,
`lib/jxl/enc_quant_weights.cc:166-180`

---

## Behavioral Differences

### 1. kMinButteraugliDistance

| | Rust | libjxl |
|---|---|---|
| **Value** | 0.01 | 0.05 |
| **Source** | `lf_frame.rs:22` | `enc_params.h:201` |
| **Effect** | Lower floor on DC distance | Higher floor on DC distance |

```rust
// Rust (lf_frame.rs:22)
const K_MIN_BUTTERAUGLI_DISTANCE: f32 = 0.01;

// Used at lf_frame.rs:104:
let dc_distance = (main_distance * 0.02).max(K_MIN_BUTTERAUGLI_DISTANCE * 0.02);
// = (main_distance * 0.02).max(0.0002)
```

```cpp
// libjxl (enc_params.h:201)
static constexpr float kMinButteraugliDistance = 0.05f;

// enc_cache.cc:137-139:
cparams.butteraugli_distance =
    std::max(kMinButteraugliDistance * 0.02f,         // = 0.001
             enc_state->cparams.butteraugli_distance * 0.02f);
```

**Impact**: Negligible in practice. The floor value is `0.0002` (Rust) vs `0.001`
(libjxl) for DC distance. At any practical main distance (d >= 0.5), the
`main_distance * 0.02` term dominates (= 0.01 at d=0.5). The floor only matters
at extremely low distances where both are effectively lossless anyway.

### 2. Single-Level Only (No Recursive DC Pyramid)

| | Rust | libjxl |
|---|---|---|
| **Levels** | 1 (dc_level=1 only) | 1–4 (recursive) |
| **progressive_dc** | Always 1 | 1–4 |

libjxl supports a multi-level DC pyramid where each LfFrame can recursively
encode its own DC as another LfFrame:

```cpp
// enc_cache.cc:128-129
JXL_ENSURE(cparams.progressive_dc > 0);
cparams.progressive_dc--;
```

```
progressive_dc = 2 produces:
  Frame 1: LfFrame dc_level=2 (modular, 64× downsampled)
  Frame 2: LfFrame dc_level=1 (VarDCT, 8× downsampled)
  Frame 3: Main frame (full AC, references dc_level=1)
```

Rust always produces exactly one LfFrame at dc_level=1.

**Impact**: Only affects progressive rendering at very low resolutions. For
single-image encoding (not streaming/progressive), one level is sufficient.
libjxl's default `progressive_dc` is 0 (disabled); cjxl uses 1 when progressive
is requested.

### 3. No Intermediate VarDCT Levels

For `progressive_dc > 1`, libjxl encodes intermediate LfFrames as VarDCT
(not modular) with special parameters:

```cpp
// enc_cache.cc:140-148
if (cparams.progressive_dc == 0) {
    // FINAL level: modular encoding
    cparams.modular_mode = true;
    cparams.speed_tier = max(kTortoise, speed_tier - 1);
    cparams.butteraugli_distance = max(kMinButteraugliDistance * 0.02f, distance * 0.02f);
} else {
    // INTERMEDIATE level: VarDCT with max_error_mode
    cparams.max_error_mode = true;
    for (size_t c = 0; c < 3; c++)
        cparams.max_error[c] = shared.quantizer.MulDC()[c];
    cparams.butteraugli_distance = max(kMinButteraugliDistance, distance * 0.1f);
}
```

Rust only supports the final level (modular encoding). It has no
`FindBestQuantizationMaxError` and no VarDCT LfFrame path.

**Impact**: None for progressive_dc=1 (the common case). Only matters for
progressive_dc >= 2 which is rarely used in practice.

### 4. DC Extraction: Direct Average vs Forward DCT

| | Rust | libjxl |
|---|---|---|
| **Method** | `sum(64 pixels) / 64` | Forward DCT → `DCFromLowestFrequencies` |
| **Source** | `lf_frame.rs:46-73` | `enc_group.cc:478-511` |

```rust
// Rust (lf_frame.rs:59-68)
for dy in 0..BLOCK_DIM {
    let row_offset = (base_y + dy) * padded_width + base_x;
    for dx in 0..BLOCK_DIM {
        sum += ch[row_offset + dx];
    }
}
dc[c][by * xsize_blocks + bx] = sum / 64.0;
```

libjxl runs the full forward DCT and extracts the DC coefficient:
```cpp
// enc_group.cc:478-483 (Y channel)
TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                    opsin_stride, coeffs_in + c * size, scratch_space);
DCFromLowestFrequencies(acs.Strategy(), coeffs_in + size,
                        dc_rows[1] + bx, dc_stride, scratch_space);
```

For DCT8, the forward DCT DC coefficient = `sum(64 pixels) * (1/8) * (1/8)` =
`sum / 64`, which is mathematically identical to Rust's direct average. The
values are the same to floating-point precision.

**However**, libjxl's DC extraction happens inside `ComputeCoefficients`, which
also runs CfL unapply and Y roundtrip quantization. This means:

- **Y DC**: extracted from original forward DCT (pre-roundtrip), same as Rust
- **X DC**: extracted AFTER CfL unapply: `X_dc = original_X_dc - x_factor * Y_dc_roundtripped`
- **B DC**: extracted AFTER CfL unapply: `B_dc = original_B_dc - b_factor * Y_dc_roundtripped`

Rust computes all three channels from raw XYB pixels without CfL decorrelation.

**Impact**: The X and B channel DC values differ by the CfL contribution:
`x_factor * Y_dc` for X, `b_factor * Y_dc` for B. When the main VarDCT frame
decoder applies CfL restoration to the LfFrame DC, libjxl's CfL-decorrelated
values get correctly restored, while Rust's raw values receive an extra CfL
contribution. This is a **potential quality issue** on images with strong
chroma-from-luma correlation at the DC level. The impact depends on the CfL
tile factors and DC magnitudes. Testing shows pixel-exact decode (verified with
djxl and jxl-rs), suggesting the quantization noise absorbs the CfL difference
in practice, but this warrants investigation for images with high CfL at DC.

### 5. No Decode-Back Step

libjxl encodes the LfFrame, then **immediately decodes it** to extract the
reconstructed DC image for the main frame's reference:

```cpp
// enc_cache.cc:195-222
auto decoded = make_unique<ImageBundle>(...);
auto dec_state = make_unique<PassesDecoderState>(...);
dec_state->output_encoding_info.SetFromMetadata(*shared.metadata);

for (int i = 0; i <= cparams.progressive_dc; ++i) {
    DecodeFrame(dec_state.get(), pool, frame_start, encoded_size,
                nullptr, decoded.get(), *shared.metadata);
    frame_start += decoded->decoded_bytes();
    encoded_size -= decoded->decoded_bytes();
}

const Image3F& dc_frame = dec_state->shared->dc_frames[frame_header.dc_level];
CopyImageTo(dc_frame, &shared.dc_storage);
ZeroFillImage(&shared.quant_dc);
shared.dc = &shared.dc_storage;
```

This decode-back ensures the encoder's DC reference matches what the decoder
will produce. The quantization roundtrip through modular encoding introduces
small errors (integer rounding), and the decode-back captures these exactly.

Rust does **not** decode back. The main frame proceeds with the original float
DC values, while the decoder will have the LfFrame's quantized/dequantized DC.

**Impact**: The main VarDCT frame's encoder-side reconstruction uses slightly
different DC values than the decoder will see. This affects:
- Butteraugli loop accuracy (encoder sees pre-quantization DC, decoder sees
  post-quantization DC)
- EPF sharpness selection (minor, since EPF operates on pixels not DC)

The practical effect is small because:
1. DC quantization at 0.02× distance is very fine-grained
2. The F16 roundtrip ensures enc_factors match decoder expectations
3. The integer rounding error is typically ±1 in the modular integer domain

However, this likely contributes to the **overhead gap**: Rust +10.8% avg vs
libjxl +4.6% avg. Without decode-back, the main frame may sub-optimally
quantize AC coefficients around DC boundaries where the DC approximation is
slightly off.

### 6. LfFrame Size Overhead

| | Rust | libjxl |
|---|---|---|
| **Avg overhead** | +10.8% | +4.6% |
| **Source** | CLAUDE.md measurements | CLAUDE.md measurements |

The Rust LfFrame adds roughly 6 percentage points more overhead than libjxl's.
Likely contributors:

1. **No decode-back** (see above): main frame may encode redundant information
   that the decoder already has from the LfFrame
2. **Modular encoding efficiency**: Rust's modular encoder may produce slightly
   larger output for the small DC images (e.g., 32×32 at 256×256 input)
3. **CfL decorrelation at DC**: Without CfL decorrelation, the X and B channels
   carry more entropy (correlated with Y), requiring more bits
4. **No ZeroFillImage(&quant_dc)**: libjxl zeroes the inline quant_dc when using
   LfFrame, meaning the main frame's DC section is truly empty. The Rust encoder
   may still encode non-trivial DC data

### 7. Speed Tier / Effort Mapping

| | Rust | libjxl |
|---|---|---|
| **Formula** | `effort.max(2) - 1` | `max(kTortoise, speed_tier - 1)` |
| **Minimum** | 1 | kTortoise (1) |
| **Source** | `lf_frame.rs:241` | `enc_cache.cc:134-136` |

```rust
// Rust (lf_frame.rs:241)
let lf_effort = effort.max(2) - 1; // One level slower, minimum 1
```

```cpp
// libjxl (enc_cache.cc:134-136)
cparams.speed_tier = static_cast<SpeedTier>(
    std::max(static_cast<int>(SpeedTier::kTortoise),
             static_cast<int>(cparams.speed_tier) - 1));
```

**Impact**: Equivalent. Both produce `max(1, effort - 1)`. kTortoise = 1 in
libjxl's speed tier enum.

### 8. Disabled Features in LfFrame

libjxl explicitly disables several features for the LfFrame:

```cpp
// enc_cache.cc:119-127
cparams.dots = Override::kOff;
cparams.noise = Override::kOff;
cparams.patches = Override::kOff;
cparams.gaborish = Override::kOff;
cparams.epf = 0;
cparams.resampling = 1;
cparams.ec_resampling = 1;
cparams.keep_invisible = Override::kOn;
```

Rust achieves equivalent behavior through the frame header:

```rust
// lf_frame.rs:265-272 (FrameHeader::lf_frame)
frame_type: FrameType::LfFrame,
encoding: Encoding::Modular,      // No VarDCT features
flags: SKIP_ADAPTIVE_LF_SMOOTHING,
gaborish: false,
epf_iters: 0,
```

**Impact**: Equivalent. Modular encoding inherently skips dots, noise, patches,
and gaborish. EPF is disabled via `epf_iters: 0`. The
`SKIP_ADAPTIVE_LF_SMOOTHING` flag prevents the decoder from applying DC
smoothing to the LfFrame output.

---

## Confirmed Matches

### enc_factors Formula

Both use identical quantization factors:

```
enc_factors[0] = 65536 / (1 + 23 * dc_distance)    // X channel
enc_factors[1] = 4096  / (1 + 14 * dc_distance)    // Y channel
enc_factors[2] = 4096  / (1 + 14 * dc_distance)    // B channel

where dc_distance = main_distance * 0.02
```

Rust: `lf_frame.rs:106-109`
libjxl: `enc_modular.cc:757-762`

### F16 Roundtrip

Both ensure dc_quant values match the decoder by performing an F16 roundtrip:

```rust
// Rust (lf_frame.rs:115-118)
let dc_quant: [f32; 3] = [
    f16_roundtrip(128.0 / enc_factors[0]) / 128.0,
    f16_roundtrip(128.0 / enc_factors[1]) / 128.0,
    f16_roundtrip(128.0 / enc_factors[2]) / 128.0,
];
```

```cpp
// libjxl (enc_quant_weights.cc:166-180)
Status DequantMatricesSetCustomDC(..., const float* dc) {
    matrices->SetDCQuant(dc);
    // Roundtrip encode/decode DC to ensure same values as decoder.
    BitWriter writer{memory_manager};
    DequantMatricesEncodeDC(*matrices, &writer, ...);
    writer.ZeroPadToByte();
    BitReader br(writer.GetSpan());
    matrices->DecodeDC(&br);
    br.Close();
}
```

libjxl roundtrips through the full encode/decode path (write F16 to bitstream,
read back). Rust uses `f16_roundtrip()` directly. Both produce the same result
because the bitstream F16 encoding IS the rounding bottleneck.

### Channel Order: YX(B-Y)

Both store modular channels in [Y, X, B-Y] order:

```rust
// Rust (lf_frame.rs:170-194)
// XYB input: [0=X, 1=Y, 2=B]
// Modular output: [0=Y, 1=X, 2=B-Y]
let y_int = round_hafz(dc_y * factors.inv_dc_quant[1]);
let x_int = round_hafz(dc_x * factors.inv_dc_quant[0]);
let b_int = (dc_b * factors.inv_dc_quant[2] + 0.5) as i32 - y_int;
```

```cpp
// libjxl (enc_modular.cc:784-800)
// XYB is encoded as YX(B-Y)
if (cparams_.color_transform == ColorTransform::kXYB && c < 2)
    c_out = 1 - c_out;     // Swaps X↔Y: c=0→c_out=1, c=1→c_out=0
// B channel (c=2):
row_out[x] = row_in[x] * factor + 0.5f;
row_out[x] -= row_Y[x];   // B_int - Y_int
```

### Rounding Behavior

Both use round-half-away-from-zero for Y and X channels:

```rust
// Rust (lf_frame.rs:133-135)
fn round_hafz(val: f32) -> i32 {
    (val + if val < 0.0 { -0.5 } else { 0.5 }) as i32
}
```

```cpp
// libjxl (enc_modular.cc:165-172)
// float_to_int, non-fp path:
row_out[x] = row_in[x] * factor + (row_in[x] < 0 ? -0.5f : 0.5f);
```

For B channel, both use always-positive 0.5 bias (then subtract Y_int):

```rust
// Rust (lf_frame.rs:190)
let b_int = (dc_b * factors.inv_dc_quant[2] + 0.5) as i32 - y_int;
```

```cpp
// libjxl (enc_modular.cc:798-799)
row_out[x] = row_in[x] * factor + 0.5f;
row_out[x] -= row_Y[x];
```

### Frame Header

Both produce equivalent frame headers for the LfFrame:

| Field | Rust | libjxl |
|---|---|---|
| frame_type | LfFrame (1) | kDCFrame (1) |
| encoding | Modular | modular_mode = true |
| xyb_encoded | true | SetFromImage(LinearSRGB) but ib_needs_color_transform=false |
| dc_level | 1 | frame_header.dc_level + 1 = 1 |
| flags | SKIP_ADAPTIVE_LF_SMOOTHING (0x80) | Implicit via disabled features |
| gaborish | false | Override::kOff |
| epf_iters | 0 | epf = 0 |
| is_last | false | dc_frame_info defaults |
| save_before_ct | false | save_before_color_transform = true |

Note: libjxl sets `save_before_color_transform = true` and
`ib_needs_color_transform = false`. These interact to skip the RGB→XYB
conversion in EncodeFrame (since the DC image is already in XYB). Rust handles
this by directly encoding the XYB integers without any color transform.

### USE_LF_FRAME Flag on Main Frame

Both set the same flag on the main VarDCT frame:

```rust
// Rust (frame_header.rs:96, bitstream.rs:1810)
pub const USE_LF_FRAME: u64 = 0x20;
fh.flags |= USE_LF_FRAME;
```

```cpp
// libjxl (frame_header.h, enc_frame.cc:265)
kUseDcFrame = 0x20
flags |= FrameHeader::kUseDcFrame;
```

### Group Layout

Both support single-group and multi-group LfFrame encoding:

- **Single group**: DC image fits in one 256×256 group (original ≤ 2048×2048)
- **Multi-group**: DC image requires multiple groups, with standard JXL section
  layout (LfGlobal | LfGroup×N | HfGlobal | PassGroup×N)

```rust
// Rust (lf_frame.rs:249-286)
if num_groups == 1 {
    // Single group: combined dc_quant + tree + data
} else {
    encode_lf_frame_multi_group(...)
}
```

### Custom dc_quant in LfGlobal

Both write custom DC quantization factors as part of the LfGlobal section:

```rust
// Rust (lf_frame.rs:261)
Some(factors.dc_quant)  // Passed to modular section writer
```

```cpp
// libjxl: DequantMatricesSetCustomDC → DequantMatricesEncodeDC writes to LfGlobal
```

The dc_quant values are encoded as F16 in the bitstream: `round_to_f16(dc_quant[c] * 128)`.
The decoder reads these and computes `inv_dc_quant[c] = 1.0 / (f16_value / 128)`.

---

## Summary of Impact

| Difference | Severity | Notes |
|---|---|---|
| kMinButteraugliDistance 0.01 vs 0.05 | Negligible | Only affects extreme low distances |
| Single-level only | Low | progressive_dc=1 is the common case |
| No intermediate VarDCT | Low | Only for progressive_dc >= 2 |
| DC without CfL decorrelation | **Medium** | May cause quality issues with strong CfL at DC |
| No decode-back step | **Medium** | Likely contributor to +6pp overhead gap |
| Size overhead +10.8% vs +4.6% | Medium | Multiple contributing factors |
| Speed tier mapping | None | Equivalent behavior |

**Overall**: The Rust LfFrame is functionally correct for progressive_dc=1
(verified pixel-exact with djxl and jxl-rs). The main gaps are the missing
decode-back step (causes sub-optimal AC quantization near DC boundaries) and
the CfL decorrelation at DC level (may cause small quality differences on
images with strong chroma correlation). Both likely contribute to the +6pp
overhead gap.
