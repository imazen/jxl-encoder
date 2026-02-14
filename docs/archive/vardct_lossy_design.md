# Rust Design Specification: VarDCT Lossy Encoding
## From RGB Pixels to Compressed Bitstream (Matching libjxl Exactly)

---

## Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Constants & Parameters](#2-constants--parameters)
3. [Module 1: Color Transform (XYB)](#3-module-1-color-transform-xyb)
4. [Module 2: AC Strategy Selection](#4-module-2-ac-strategy-selection)
5. [Module 3: DCT Transforms](#5-module-3-dct-transforms)
6. [Module 4: Chroma From Luma](#6-module-4-chroma-from-luma)
7. [Module 5: Quantization](#7-module-5-quantization)
8. [Module 6: Adaptive Quantization](#8-module-6-adaptive-quantization)
9. [Module 7: Coefficient Ordering](#9-module-7-coefficient-ordering)
10. [Module 8: Tokenization & Context](#10-module-8-tokenization--context)
11. [Module 9: Frame Orchestration](#11-module-9-frame-orchestration)
12. [Data Structures Summary](#12-data-structures-summary)
13. [Implementation Order](#13-implementation-order)
14. [Test Strategy](#14-test-strategy)

---

## 1. Overview & Architecture

### The VarDCT Pipeline
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VARDCT LOSSY ENCODING PIPELINE                         │
└──────────────────────────────────────────────────────────────────────────────┘

 RGB Pixels                                              Compressed Bitstream
     │                                                          ▲
     ▼                                                          │
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  Color  │   │   AC    │   │   DCT   │   │  Chroma │   │Quantize │
│Transform│──▶│Strategy │──▶│Transform│──▶│  From   │──▶│  Coeffs │
│  (XYB)  │   │Selection│   │         │   │  Luma   │   │         │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
                                                              │
     ┌────────────────────────────────────────────────────────┘
     ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Coeff   │   │Tokenize │   │ Entropy │   │  Write  │──▶ .jxl file
│ Order   │──▶│& Context│──▶│ Coding  │──▶│Bitstream│
│         │   │         │   │  (ANS)  │   │         │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### Key Concepts

**VarDCT** = **Var**iable **DCT** - the encoder chooses different DCT sizes for different regions:
- 8×8 (standard JPEG-like)
- 16×16, 32×32, 64×64 (smooth areas)
- 16×8, 8×16, 32×16, etc. (directional)
- 4×4, 2×2, identity (high-detail areas)
- AFV (corner transforms)

### Crate Structure
```
jxl-vardct/
├── src/
│   ├── lib.rs                 // Public API
│   ├── color/
│   │   ├── mod.rs
│   │   ├── xyb.rs             // RGB → XYB transform
│   │   └── opsin_params.rs    // Transform matrices
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── ac_strategy.rs     // AcStrategy types & image
│   │   └── selection.rs       // Heuristics for choosing strategy
│   ├── transform/
│   │   ├── mod.rs
│   │   ├── dct.rs             // Forward DCT implementations
│   │   ├── dct_scales.rs      // Scaling factors
│   │   └── afv.rs             // Corner transforms
│   ├── cfl/
│   │   ├── mod.rs
│   │   └── chroma_from_luma.rs
│   ├── quant/
│   │   ├── mod.rs
│   │   ├── quantizer.rs       // Main quantizer
│   │   ├── quant_weights.rs   // Dequant matrices
│   │   └── adaptive.rs        // Adaptive quantization
│   ├── coeff/
│   │   ├── mod.rs
│   │   ├── order.rs           // Coefficient ordering
│   │   └── context.rs         // Block context mapping
│   ├── encode/
│   │   ├── mod.rs
│   │   ├── frame.rs           // Frame-level orchestration
│   │   ├── group.rs           // Group-level encoding
│   │   └── tokens.rs          // Tokenization
│   └── common/
│       ├── mod.rs
│       ├── image.rs           // Image types
│       └── frame_dim.rs       // Frame dimensions
└── tests/
    ├── parity_xyb.rs
    ├── parity_dct.rs
    ├── parity_quant.rs
    └── integration.rs
```

---

## 2. Constants & Parameters

```rust
// src/common/constants.rs

/// Block dimension (8×8 is the fundamental unit)
pub const BLOCK_DIM: usize = 8;
pub const DCT_BLOCK_SIZE: usize = BLOCK_DIM * BLOCK_DIM;  // 64

/// Maximum block dimensions for largest transforms
pub const MAX_COEFF_BLOCKS: usize = 32;  // 256×256 DCT = 32×32 blocks
pub const MAX_BLOCK_DIM: usize = BLOCK_DIM * MAX_COEFF_BLOCKS;  // 256
pub const MAX_COEFF_AREA: usize = MAX_BLOCK_DIM * MAX_BLOCK_DIM;  // 65536

/// Group dimensions
pub const GROUP_DIM: usize = 256;
pub const GROUP_DIM_IN_BLOCKS: usize = GROUP_DIM / BLOCK_DIM;  // 32

/// Color tile dimensions (for CfL parameters)
pub const COLOR_TILE_DIM: usize = 64;
pub const COLOR_TILE_DIM_IN_BLOCKS: usize = COLOR_TILE_DIM / BLOCK_DIM;  // 8

/// Quantization constants
pub const GLOBAL_SCALE_DENOM: i32 = 1 << 16;
pub const GLOBAL_SCALE_NUMERATOR: i32 = 4096;
pub const QUANT_MAX: i32 = 256;

/// DC quantization base values (inverse, per channel)
pub const INV_DC_QUANT: [f32; 3] = [4096.0, 512.0, 256.0];
pub const DC_QUANT: [f32; 3] = [
    1.0 / INV_DC_QUANT[0],
    1.0 / INV_DC_QUANT[1],
    1.0 / INV_DC_QUANT[2],
];

/// Default CfL color factor
pub const DEFAULT_COLOR_FACTOR: u8 = 84;
pub const CFL_FIXED_POINT_PRECISION: u8 = 11;

/// Zero-bias defaults for quantization
pub const ZERO_BIAS_DEFAULT: [f32; 3] = [0.5, 0.5, 0.5];

/// Quantization bias for dequantization adjustment
pub const DEFAULT_QUANT_BIAS: [f32; 4] = [
    1.0 - 0.05465007330715401,
    1.0 - 0.07005449891748593,
    1.0 - 0.049935103337343655,
    0.145,
];

/// Context constants
pub const NON_ZERO_BUCKETS: usize = 37;
pub const ZERO_DENSITY_CONTEXT_COUNT: usize = 458;
pub const NUM_ORDERS: usize = 13;  // Number of distinct coefficient orders
```

---

## 3. Module 1: Color Transform (XYB)

The XYB color space is perceptually uniform and decorrelates luminance from chrominance.

### 3.1 Opsin Absorption Matrix

```rust
// src/color/opsin_params.rs

/// Opsin absorbance matrix coefficients
/// Transforms linear RGB to "cone responses"
pub const OPSIN_MATRIX: [[f32; 3]; 3] = [
    [0.30, 0.622, 0.078],           // Row 0: R contribution
    [0.23, 0.692, 0.078],           // Row 1: G contribution
    [0.24342268924547819, 0.20476744424496821, 0.5518096357545336],  // Row 2: B
];

/// Inverse opsin matrix (for decoding)
pub const INV_OPSIN_MATRIX: [[f32; 3]; 3] = [
    [11.031566901960783, -9.866943921568629, -0.16462299647058826],
    [-3.254147380392157, 4.418770392156863, -0.16462299647058826],
    [-3.6588512862745097, 2.7129230470588235, 1.9459282392156863],
];

/// Opsin absorbance bias (added before cube root)
pub const OPSIN_BIAS: f32 = 0.0037930732552754493;

/// Scaling for final XYB values
pub const SCALED_XYB_OFFSET: [f32; 3] = [0.015386134, 0.0, 0.27770459];
pub const SCALED_XYB_SCALE: [f32; 3] = [22.995788804, 1.183000077, 1.502141333];

/// Y-to-B ratio for chroma correlation
pub const Y_TO_B_RATIO: f32 = 1.0;
```

### 3.2 XYB Transform Implementation

```rust
// src/color/xyb.rs

use crate::opsin_params::*;

/// Convert linear RGB to XYB color space.
///
/// The transform is:
/// 1. Apply opsin absorbance matrix (weighted sum of RGB)
/// 2. Add bias and take signed cube root
/// 3. Convert to X, Y, B channels:
///    - Y = (L + M) / 2      (luminance)
///    - X = (L - M) / 2      (red-green)
///    - B = S                (blue)
///
/// Implements libjxl's LinearRGBRowToXYB
#[inline]
pub fn linear_rgb_to_xyb(
    r: f32, g: f32, b: f32,
    premul_absorb: &[f32; 12],
) -> (f32, f32, f32) {
    // Apply opsin absorbance matrix (premultiplied by intensity target)
    let mixed0 = premul_absorb[0] * r + premul_absorb[1] * g + premul_absorb[2] * b;
    let mixed1 = premul_absorb[4] * r + premul_absorb[5] * g + premul_absorb[6] * b;
    let mixed2 = premul_absorb[8] * r + premul_absorb[9] * g + premul_absorb[10] * b;

    // Add bias and apply signed cube root
    let mixed0 = mixed0 + premul_absorb[3];
    let mixed1 = mixed1 + premul_absorb[7];
    let mixed2 = mixed2 + premul_absorb[11];

    let gamma0 = signed_cbrt(mixed0);
    let gamma1 = signed_cbrt(mixed1);
    let gamma2 = signed_cbrt(mixed2);

    // Convert to XYB
    let y = (gamma0 + gamma1) * 0.5;
    let x = (gamma0 - gamma1) * 0.5;
    let b_out = gamma2;

    (x, y, b_out)
}

/// Signed cube root: sign(x) * |x|^(1/3)
#[inline]
fn signed_cbrt(x: f32) -> f32 {
    let abs_x = x.abs();
    let cbrt = abs_x.cbrt();
    if x < 0.0 { -cbrt } else { cbrt }
}

/// Compute premultiplied absorption matrix for given intensity target.
///
/// Implements libjxl's ComputePremulAbsorb
pub fn compute_premul_absorb(intensity_target: f32) -> [f32; 12] {
    let mut premul = [0.0f32; 12];
    let scale = intensity_target / 255.0;

    for row in 0..3 {
        for col in 0..3 {
            premul[row * 4 + col] = OPSIN_MATRIX[row][col] * scale;
        }
        premul[row * 4 + 3] = OPSIN_BIAS;
    }

    premul
}

/// Scale XYB values to [0, 1] range for storage.
///
/// Implements libjxl's ScaleXYB
#[inline]
pub fn scale_xyb(x: f32, y: f32, b: f32) -> (f32, f32, f32) {
    (
        (x - SCALED_XYB_OFFSET[0]) * SCALED_XYB_SCALE[0],
        (y - SCALED_XYB_OFFSET[1]) * SCALED_XYB_SCALE[1],
        (b - SCALED_XYB_OFFSET[2]) * SCALED_XYB_SCALE[2],
    )
}

/// Convert entire image from linear sRGB to XYB.
///
/// Implements libjxl's ToXYB
pub fn image_to_xyb(
    image: &mut Image3F,
    intensity_target: f32,
    pool: &ThreadPool,
) -> Result<(), Error> {
    let premul_absorb = compute_premul_absorb(intensity_target);

    pool.run(image.ysize(), |y| {
        let row0 = image.plane_row_mut(0, y);
        let row1 = image.plane_row_mut(1, y);
        let row2 = image.plane_row_mut(2, y);

        for x in 0..image.xsize() {
            let (out_x, out_y, out_b) = linear_rgb_to_xyb(
                row0[x], row1[x], row2[x],
                &premul_absorb
            );
            row0[x] = out_x;
            row1[x] = out_y;
            row2[x] = out_b;
        }
    })?;

    // Scale to [0, 1] range
    scale_xyb_image(image);

    Ok(())
}
```

---

## 4. Module 2: AC Strategy Selection

### 4.1 AC Strategy Types

```rust
// src/strategy/ac_strategy.rs

/// All supported AC (transform) strategy types.
/// Each strategy defines a different DCT size/shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AcStrategyType {
    // Standard 8×8 DCT
    DCT = 0,

    // Pixel-domain strategies (no transform)
    IDENTITY = 1,      // Pass-through
    DCT2X2 = 2,        // 2×2 Hadamard
    DCT4X4 = 3,        // 4×4 DCT (4 per 8×8 block)

    // Large square transforms
    DCT16X16 = 4,      // 16×16 DCT (covers 2×2 blocks)
    DCT32X32 = 5,      // 32×32 DCT (covers 4×4 blocks)
    DCT64X64 = 18,     // 64×64 DCT (covers 8×8 blocks)
    DCT128X128 = 21,   // 128×128 DCT
    DCT256X256 = 24,   // 256×256 DCT

    // Rectangular transforms
    DCT16X8 = 6,
    DCT8X16 = 7,
    DCT32X8 = 8,
    DCT8X32 = 9,
    DCT32X16 = 10,
    DCT16X32 = 11,
    DCT4X8 = 12,
    DCT8X4 = 13,
    DCT64X32 = 19,
    DCT32X64 = 20,
    DCT128X64 = 22,
    DCT64X128 = 23,
    DCT256X128 = 25,
    DCT128X256 = 26,

    // AFV (Asymmetric Frequency Variation) corner transforms
    AFV0 = 14,  // Top-left corner
    AFV1 = 15,  // Top-right corner
    AFV2 = 16,  // Bottom-left corner
    AFV3 = 17,  // Bottom-right corner
}

impl AcStrategyType {
    pub const NUM_VALID: usize = 27;

    /// Number of 8×8 blocks covered horizontally
    pub fn covered_blocks_x(&self) -> usize {
        const LUT: [u8; 27] = [
            1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1,
            8, 4, 8, 16, 8, 16, 32, 16, 32
        ];
        LUT[*self as usize] as usize
    }

    /// Number of 8×8 blocks covered vertically
    pub fn covered_blocks_y(&self) -> usize {
        const LUT: [u8; 27] = [
            1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1,
            8, 8, 4, 16, 16, 8, 32, 32, 16
        ];
        LUT[*self as usize] as usize
    }

    /// Log2 of total blocks covered
    pub fn log2_covered_blocks(&self) -> usize {
        const LUT: [u8; 27] = [
            0, 0, 0, 0, 2, 4, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0,
            6, 5, 5, 8, 7, 7, 10, 9, 9
        ];
        LUT[*self as usize] as usize
    }

    /// Is this a multi-block strategy?
    pub fn is_multiblock(&self) -> bool {
        self.covered_blocks_x() > 1 || self.covered_blocks_y() > 1
    }
}

/// Represents a strategy at a particular position in the image.
#[derive(Clone, Copy, Debug)]
pub struct AcStrategy {
    strategy: AcStrategyType,
    /// True if this is the top-left (first) block of a multi-block transform
    is_first: bool,
}

impl AcStrategy {
    pub fn new(strategy: AcStrategyType, is_first: bool) -> Self {
        Self { strategy, is_first }
    }

    pub fn from_raw(strategy: AcStrategyType) -> Self {
        Self { strategy, is_first: true }
    }

    pub fn strategy(&self) -> AcStrategyType { self.strategy }
    pub fn is_first_block(&self) -> bool { self.is_first }
    pub fn raw_strategy(&self) -> u8 { self.strategy as u8 }
}

/// Image storing AC strategy for each 8×8 block.
///
/// Each byte encodes:
/// - Bits 7-1: Strategy type
/// - Bit 0: Is first block flag
pub struct AcStrategyImage {
    data: ImageU8,
}

impl AcStrategyImage {
    pub fn new(xsize_blocks: usize, ysize_blocks: usize) -> Result<Self, Error> {
        Ok(Self {
            data: ImageU8::new(xsize_blocks, ysize_blocks)?,
        })
    }

    /// Fill entire image with DCT8×8
    pub fn fill_dct8(&mut self) {
        let value = ((AcStrategyType::DCT as u8) << 1) | 1;
        self.data.fill(value);
    }

    /// Set strategy for a block (handles multi-block setup)
    pub fn set(&mut self, x: usize, y: usize, strategy: AcStrategyType) -> Result<(), Error> {
        let acs = AcStrategy::from_raw(strategy);
        let blocks_x = acs.strategy().covered_blocks_x();
        let blocks_y = acs.strategy().covered_blocks_y();

        for iy in 0..blocks_y {
            for ix in 0..blocks_x {
                let is_first = (iy | ix) == 0;
                let value = ((strategy as u8) << 1) | (is_first as u8);
                *self.data.pixel_mut(x + ix, y + iy) = value;
            }
        }

        Ok(())
    }

    /// Get strategy at block position
    pub fn get(&self, x: usize, y: usize) -> AcStrategy {
        let value = self.data.pixel(x, y);
        let strategy = AcStrategyType::try_from(value >> 1).unwrap();
        let is_first = (value & 1) != 0;
        AcStrategy { strategy, is_first }
    }

    pub fn xsize(&self) -> usize { self.data.xsize() }
    pub fn ysize(&self) -> usize { self.data.ysize() }
}
```

### 4.2 Strategy Selection Heuristics

```rust
// src/strategy/selection.rs

/// Configuration for AC strategy selection
pub struct ACSConfig {
    pub dequant: DequantMatrices,
    pub quant_field: ImageF,
    pub masking_field: ImageF,
    pub src_image: Image3F,  // XYB image
    pub info_loss_multiplier: f32,
    pub cost_delta: f32,
    pub zeros_mul: f32,
}

/// Find best AC strategy for the image.
///
/// This is a complex heuristic that evaluates each potential strategy
/// by estimating rate-distortion cost.
///
/// Implements libjxl's FindBestAcStrategy
pub fn find_best_ac_strategy(
    config: &ACSConfig,
    cparams: &CompressParams,
    ac_strategy: &mut AcStrategyImage,
    pool: &ThreadPool,
) -> Result<(), Error> {
    let speed_tier = cparams.speed_tier;

    // For fastest modes, just use DCT8×8 everywhere
    if speed_tier >= SpeedTier::Cheetah {
        ac_strategy.fill_dct8();
        return Ok(());
    }

    // Process in blocks
    pool.run_rect(ac_strategy.xsize(), ac_strategy.ysize(), |rect| {
        process_ac_strategy_rect(config, cparams, ac_strategy, rect)
    })?;

    Ok(())
}

/// Process a rectangle for AC strategy selection.
///
/// For each block position, evaluates candidate strategies and picks
/// the one with lowest rate-distortion cost.
fn process_ac_strategy_rect(
    config: &ACSConfig,
    cparams: &CompressParams,
    ac_strategy: &mut AcStrategyImage,
    rect: Rect,
) -> Result<(), Error> {
    // List of strategies to try (varies by speed tier)
    let candidates = get_candidate_strategies(cparams.speed_tier);

    for by in rect.y0()..rect.y1() {
        for bx in rect.x0()..rect.x1() {
            // Skip if already covered by a larger transform
            if !ac_strategy.get(bx, by).is_first_block() {
                continue;
            }

            let mut best_strategy = AcStrategyType::DCT;
            let mut best_cost = f32::MAX;

            for &strategy in &candidates {
                // Check if strategy fits
                let blocks_x = strategy.covered_blocks_x();
                let blocks_y = strategy.covered_blocks_y();

                if bx + blocks_x > ac_strategy.xsize() ||
                   by + blocks_y > ac_strategy.ysize() {
                    continue;
                }

                // Estimate cost of this strategy
                let cost = estimate_strategy_cost(
                    config, bx, by, strategy
                );

                if cost < best_cost {
                    best_cost = cost;
                    best_strategy = strategy;
                }
            }

            ac_strategy.set(bx, by, best_strategy)?;
        }
    }

    Ok(())
}

/// Get candidate strategies based on speed tier
fn get_candidate_strategies(speed: SpeedTier) -> Vec<AcStrategyType> {
    use AcStrategyType::*;

    match speed {
        SpeedTier::Lightning | SpeedTier::Thunder | SpeedTier::Cheetah => {
            vec![DCT]
        }
        SpeedTier::Hare | SpeedTier::Wombat => {
            vec![DCT, DCT16X16, DCT32X32, IDENTITY]
        }
        SpeedTier::Squirrel | SpeedTier::Kitten => {
            vec![
                DCT, DCT4X4, DCT16X16, DCT32X32,
                DCT16X8, DCT8X16, DCT32X16, DCT16X32,
                IDENTITY, DCT2X2
            ]
        }
        SpeedTier::Tortoise => {
            // All strategies
            (0..AcStrategyType::NUM_VALID as u8)
                .filter_map(|i| AcStrategyType::try_from(i).ok())
                .collect()
        }
    }
}

/// Estimate the rate-distortion cost of using a particular strategy.
///
/// This is a simplified version - full implementation requires:
/// - Forward DCT
/// - Quantization simulation
/// - Entropy estimation
/// - Distortion computation (ideally using Butteraugli)
fn estimate_strategy_cost(
    config: &ACSConfig,
    bx: usize, by: usize,
    strategy: AcStrategyType,
) -> f32 {
    // Extract pixel block
    // Apply DCT
    // Quantize
    // Estimate bits
    // Estimate distortion
    // Return rate + lambda * distortion
    todo!("Implement cost estimation")
}
```

---

## 5. Module 3: DCT Transforms

### 5.1 DCT Scaling Factors

```rust
// src/transform/dct_scales.rs

/// DCT scaling factors for each transform size.
/// These normalize the DCT so that DC coefficient represents the mean.
///
/// For N-point DCT: scale[0] = 1/sqrt(N), scale[k>0] = sqrt(2/N)

pub fn dct_scales(n: usize) -> Vec<f32> {
    let mut scales = vec![0.0f32; n];
    scales[0] = 1.0 / (n as f32).sqrt();
    let other = (2.0 / n as f32).sqrt();
    for i in 1..n {
        scales[i] = other;
    }
    scales
}

/// 2D DCT scales: product of 1D scales
pub fn dct_scales_2d(nx: usize, ny: usize) -> Vec<f32> {
    let sx = dct_scales(nx);
    let sy = dct_scales(ny);

    let mut scales = vec![0.0f32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            scales[y * nx + x] = sx[x] * sy[y];
        }
    }
    scales
}
```

### 5.2 Forward DCT Implementation

```rust
// src/transform/dct.rs

use std::f32::consts::PI;

/// Perform forward DCT-II on a block of samples.
///
/// X[k] = sum_{n=0}^{N-1} x[n] * cos(pi/N * (n + 0.5) * k)
///
/// Implements the separable 2D DCT by applying 1D DCT to rows then columns.
pub fn forward_dct_2d(
    input: &[f32],        // Input pixels, row-major
    output: &mut [f32],   // Output coefficients
    nx: usize,            // Width
    ny: usize,            // Height
    input_stride: usize,  // Input row stride
    scratch: &mut [f32],  // Temporary buffer
) {
    // DCT on rows
    for y in 0..ny {
        forward_dct_1d(
            &input[y * input_stride..],
            &mut scratch[y * nx..],
            nx,
        );
    }

    // DCT on columns (transpose, DCT, transpose)
    for x in 0..nx {
        // Extract column
        let mut col = vec![0.0f32; ny];
        for y in 0..ny {
            col[y] = scratch[y * nx + x];
        }

        // DCT
        let mut col_out = vec![0.0f32; ny];
        forward_dct_1d(&col, &mut col_out, ny);

        // Store back
        for y in 0..ny {
            output[y * nx + x] = col_out[y];
        }
    }
}

/// 1D forward DCT-II
///
/// For efficiency, use specialized implementations for common sizes
/// (8, 16, 32, 64) using factored algorithms.
fn forward_dct_1d(input: &[f32], output: &mut [f32], n: usize) {
    match n {
        8 => forward_dct8(input, output),
        16 => forward_dct16(input, output),
        32 => forward_dct32(input, output),
        _ => forward_dct_generic(input, output, n),
    }
}

/// Generic DCT implementation (O(N^2), used for unusual sizes)
fn forward_dct_generic(input: &[f32], output: &mut [f32], n: usize) {
    let scale = PI / n as f32;

    for k in 0..n {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += input[i] * (scale * (i as f32 + 0.5) * k as f32).cos();
        }
        output[k] = sum;
    }
}

/// Optimized 8-point DCT using Chen's factorization.
///
/// This is the most common case and should be highly optimized,
/// ideally using SIMD.
fn forward_dct8(input: &[f32], output: &mut [f32]) {
    // Constants for 8-point DCT
    const C1: f32 = 0.9807852804032304;  // cos(1*pi/16)
    const S1: f32 = 0.19509032201612825; // sin(1*pi/16)
    const C2: f32 = 0.9238795325112867;  // cos(2*pi/16)
    const S2: f32 = 0.3826834323650898;  // sin(2*pi/16)
    const C3: f32 = 0.8314696123025452;  // cos(3*pi/16)
    const S3: f32 = 0.5555702330196022;  // sin(3*pi/16)
    const SQRT2: f32 = 1.4142135623730951;
    const SQRT1_2: f32 = 0.7071067811865476;

    // Stage 1: butterflies
    let t0 = input[0] + input[7];
    let t7 = input[0] - input[7];
    let t1 = input[1] + input[6];
    let t6 = input[1] - input[6];
    let t2 = input[2] + input[5];
    let t5 = input[2] - input[5];
    let t3 = input[3] + input[4];
    let t4 = input[3] - input[4];

    // Stage 2
    let u0 = t0 + t3;
    let u3 = t0 - t3;
    let u1 = t1 + t2;
    let u2 = t1 - t2;

    // Stage 3
    output[0] = (u0 + u1) * 0.5;
    output[4] = (u0 - u1) * 0.5;
    output[2] = u2 * C2 * SQRT1_2 + u3 * S2 * SQRT1_2;
    output[6] = u3 * C2 * SQRT1_2 - u2 * S2 * SQRT1_2;

    // Odd terms (more complex)
    let v4 = t4 * C3 + t7 * S3;
    let v7 = t7 * C3 - t4 * S3;
    let v5 = t5 * C1 + t6 * S1;
    let v6 = t6 * C1 - t5 * S1;

    output[1] = (v4 + v5) * SQRT1_2;
    output[7] = (v4 - v5) * SQRT1_2;
    output[3] = (v7 - v6) * SQRT1_2;
    output[5] = (v7 + v6) * SQRT1_2;
}

/// 16-point DCT
fn forward_dct16(input: &[f32], output: &mut [f32]) {
    // Can be implemented as two 8-point DCTs plus rotations
    // Or use dedicated factorization
    todo!("Implement optimized 16-point DCT")
}

/// 32-point DCT
fn forward_dct32(input: &[f32], output: &mut [f32]) {
    todo!("Implement optimized 32-point DCT")
}
```

### 5.3 Transform From Pixels (Per Strategy)

```rust
// src/transform/mod.rs

/// Apply the forward transform based on AC strategy.
///
/// Input: pixel values from the image
/// Output: DCT coefficients in coefficient layout
///
/// Implements libjxl's TransformFromPixels
pub fn transform_from_pixels(
    strategy: AcStrategyType,
    pixels: &[f32],
    pixel_stride: usize,
    coeffs: &mut [f32],
    scratch: &mut [f32],
) {
    use AcStrategyType::*;

    match strategy {
        DCT => {
            forward_dct_2d(pixels, coeffs, 8, 8, pixel_stride, scratch);
        }
        DCT16X16 => {
            forward_dct_2d(pixels, coeffs, 16, 16, pixel_stride, scratch);
        }
        DCT32X32 => {
            forward_dct_2d(pixels, coeffs, 32, 32, pixel_stride, scratch);
        }
        DCT16X8 => {
            forward_dct_2d(pixels, coeffs, 16, 8, pixel_stride, scratch);
        }
        DCT8X16 => {
            forward_dct_2d(pixels, coeffs, 8, 16, pixel_stride, scratch);
        }
        IDENTITY => {
            // No transform - copy pixels directly
            for y in 0..8 {
                for x in 0..8 {
                    coeffs[y * 8 + x] = pixels[y * pixel_stride + x];
                }
            }
        }
        DCT4X4 => {
            // Four 4×4 DCTs within the 8×8 block
            for qy in 0..2 {
                for qx in 0..2 {
                    let pix_off = qy * 4 * pixel_stride + qx * 4;
                    let coef_off = qy * 32 + qx * 4;
                    forward_dct_2d(
                        &pixels[pix_off..],
                        &mut coeffs[coef_off..],
                        4, 4, pixel_stride, scratch
                    );
                }
            }
        }
        DCT2X2 => {
            // 2×2 Hadamard transforms
            for qy in 0..4 {
                for qx in 0..4 {
                    let pix_off = qy * 2 * pixel_stride + qx * 2;
                    let a = pixels[pix_off];
                    let b = pixels[pix_off + 1];
                    let c = pixels[pix_off + pixel_stride];
                    let d = pixels[pix_off + pixel_stride + 1];

                    let coef_off = qy * 16 + qx * 4;
                    coeffs[coef_off + 0] = (a + b + c + d) * 0.25;
                    coeffs[coef_off + 1] = (a - b + c - d) * 0.25;
                    coeffs[coef_off + 2] = (a + b - c - d) * 0.25;
                    coeffs[coef_off + 3] = (a - b - c + d) * 0.25;
                }
            }
        }
        AFV0 | AFV1 | AFV2 | AFV3 => {
            // AFV corner transforms
            afv_transform(strategy, pixels, pixel_stride, coeffs, scratch);
        }
        _ => {
            // Other large transforms
            let nx = strategy.covered_blocks_x() * 8;
            let ny = strategy.covered_blocks_y() * 8;
            forward_dct_2d(pixels, coeffs, nx, ny, pixel_stride, scratch);
        }
    }
}

/// Extract DC from the lowest-frequency coefficients.
///
/// For multi-block transforms, the DC is derived from the LLF
/// (lowest-low-frequency) coefficients.
///
/// Implements libjxl's DCFromLowestFrequencies
pub fn dc_from_lowest_frequencies(
    strategy: AcStrategyType,
    coeffs: &[f32],
    dc: &mut f32,
    dc_stride: usize,
    scratch: &mut [f32],
) {
    let blocks_x = strategy.covered_blocks_x();
    let blocks_y = strategy.covered_blocks_y();

    if blocks_x == 1 && blocks_y == 1 {
        // Single block: DC is just the first coefficient
        *dc = coeffs[0];
    } else {
        // Multi-block: need to extract LLF and inverse DCT
        // to get per-block DCs
        let llf_size = blocks_x * blocks_y;
        // Extract LLF coefficients and apply inverse DCT
        // to distribute DC across covered blocks
        todo!("Implement LLF to DC conversion")
    }
}
```

---

## 6. Module 4: Chroma From Luma

### 6.1 Color Correlation Map

```rust
// src/cfl/chroma_from_luma.rs

/// Color correlation parameters.
///
/// Predicts X and B channels from Y channel:
///   X_residual = X - Y * x_factor
///   B_residual = B - Y * b_factor
#[derive(Clone, Debug)]
pub struct ColorCorrelation {
    /// Base correlation for X channel
    base_correlation_x: f32,
    /// Base correlation for B channel
    base_correlation_b: f32,
    /// Scaling factor for the factor values
    color_factor: u32,
    color_scale: f32,
    /// DC correlation factors
    ytox_dc: i32,
    ytob_dc: i32,
    /// Precomputed DC factors
    dc_factors: [f32; 4],
}

impl Default for ColorCorrelation {
    fn default() -> Self {
        let mut cc = Self {
            base_correlation_x: 0.0,
            base_correlation_b: Y_TO_B_RATIO,  // 1.0
            color_factor: DEFAULT_COLOR_FACTOR as u32,
            color_scale: 1.0 / DEFAULT_COLOR_FACTOR as f32,
            ytox_dc: 0,
            ytob_dc: 0,
            dc_factors: [0.0; 4],
        };
        cc.recompute_dc_factors();
        cc
    }
}

impl ColorCorrelation {
    /// Get Y-to-X ratio for a given factor value
    pub fn ytox_ratio(&self, x_factor: i32) -> f32 {
        self.base_correlation_x + x_factor as f32 * self.color_scale
    }

    /// Get Y-to-B ratio for a given factor value
    pub fn ytob_ratio(&self, b_factor: i32) -> f32 {
        self.base_correlation_b + b_factor as f32 * self.color_scale
    }

    pub fn dc_factors(&self) -> &[f32; 4] {
        &self.dc_factors
    }

    fn recompute_dc_factors(&mut self) {
        self.dc_factors[0] = self.ytox_ratio(self.ytox_dc);
        self.dc_factors[2] = self.ytob_ratio(self.ytob_dc);
    }
}

/// Per-tile CfL parameters.
///
/// Each 64×64 tile has its own X and B factor.
pub struct ColorCorrelationMap {
    base: ColorCorrelation,
    /// Y-to-X factor for each tile (-127 to 127)
    pub ytox_map: ImageI8,
    /// Y-to-B factor for each tile (-127 to 127)
    pub ytob_map: ImageI8,
}

impl ColorCorrelationMap {
    pub fn new(xsize: usize, ysize: usize) -> Result<Self, Error> {
        let tiles_x = (xsize + COLOR_TILE_DIM - 1) / COLOR_TILE_DIM;
        let tiles_y = (ysize + COLOR_TILE_DIM - 1) / COLOR_TILE_DIM;

        Ok(Self {
            base: ColorCorrelation::default(),
            ytox_map: ImageI8::new(tiles_x, tiles_y)?,
            ytob_map: ImageI8::new(tiles_x, tiles_y)?,
        })
    }

    pub fn base(&self) -> &ColorCorrelation {
        &self.base
    }
}

/// Compute optimal CfL parameters for each tile.
///
/// Finds factors that minimize the energy of the residual.
///
/// Implements libjxl's FindBestColorCorrelationMap
pub fn find_best_color_correlation_map(
    opsin: &Image3F,  // XYB image
    cmap: &mut ColorCorrelationMap,
    pool: &ThreadPool,
) -> Result<(), Error> {
    let tile_size = COLOR_TILE_DIM;

    pool.run_2d(cmap.ytox_map.xsize(), cmap.ytox_map.ysize(), |tx, ty| {
        let x0 = tx * tile_size;
        let y0 = ty * tile_size;
        let x1 = (x0 + tile_size).min(opsin.xsize());
        let y1 = (y0 + tile_size).min(opsin.ysize());

        // Find optimal X factor
        let x_factor = find_optimal_cfl_factor(
            opsin, x0, y0, x1, y1,
            0,  // X channel
            1,  // Y channel
            &cmap.base,
        );
        *cmap.ytox_map.pixel_mut(tx, ty) = x_factor;

        // Find optimal B factor
        let b_factor = find_optimal_cfl_factor(
            opsin, x0, y0, x1, y1,
            2,  // B channel
            1,  // Y channel
            &cmap.base,
        );
        *cmap.ytob_map.pixel_mut(tx, ty) = b_factor;
    })?;

    Ok(())
}

/// Find optimal CfL factor for a tile.
///
/// Minimizes: sum((target[i] - y[i] * factor)^2)
/// Solution: factor = sum(target[i] * y[i]) / sum(y[i]^2)
fn find_optimal_cfl_factor(
    opsin: &Image3F,
    x0: usize, y0: usize, x1: usize, y1: usize,
    target_channel: usize,
    y_channel: usize,
    base: &ColorCorrelation,
) -> i8 {
    let mut sum_ty = 0.0f64;  // sum(target * y)
    let mut sum_yy = 0.0f64;  // sum(y * y)

    for y in y0..y1 {
        for x in x0..x1 {
            let target = opsin.pixel(target_channel, x, y) as f64;
            let luma = opsin.pixel(y_channel, x, y) as f64;
            sum_ty += target * luma;
            sum_yy += luma * luma;
        }
    }

    if sum_yy < 1e-10 {
        return 0;
    }

    let optimal = sum_ty / sum_yy;

    // Convert to quantized factor
    let base_corr = if target_channel == 0 {
        base.base_correlation_x
    } else {
        base.base_correlation_b
    } as f64;

    let factor = ((optimal - base_corr) / base.color_scale as f64).round();
    factor.clamp(-127.0, 127.0) as i8
}
```

---

## 7. Module 5: Quantization

### 7.1 Quantizer

```rust
// src/quant/quantizer.rs

/// Main quantizer for VarDCT encoding.
///
/// Converts floating-point DCT coefficients to integers.
pub struct Quantizer {
    dequant: DequantMatrices,

    // Serialized parameters
    global_scale: i32,
    quant_dc: i32,

    // Derived values
    inv_global_scale: f32,
    global_scale_float: f32,
    inv_quant_dc: f32,

    // Per-channel DC multipliers
    mul_dc: [f32; 4],
    inv_mul_dc: [f32; 4],

    // Quantization biases
    zero_bias: [f32; 3],
}

impl Quantizer {
    pub fn new(dequant: DequantMatrices) -> Self {
        let mut q = Self {
            dequant,
            global_scale: GLOBAL_SCALE_NUMERATOR,
            quant_dc: 1,
            inv_global_scale: 0.0,
            global_scale_float: 0.0,
            inv_quant_dc: 0.0,
            mul_dc: [1.0; 4],
            inv_mul_dc: [1.0; 4],
            zero_bias: ZERO_BIAS_DEFAULT,
        };
        q.recompute_from_global_scale();
        q
    }

    /// Recompute derived values after changing global_scale or quant_dc
    pub fn recompute_from_global_scale(&mut self) {
        self.global_scale_float = self.global_scale as f32 / GLOBAL_SCALE_DENOM as f32;
        self.inv_global_scale = GLOBAL_SCALE_DENOM as f32 / self.global_scale as f32;
        self.inv_quant_dc = self.inv_global_scale / self.quant_dc as f32;

        for c in 0..3 {
            self.mul_dc[c] = self.get_dc_step(c);
            self.inv_mul_dc[c] = self.get_inv_dc_step(c);
        }
    }

    /// Get DC quantization step for channel
    pub fn get_dc_step(&self, c: usize) -> f32 {
        self.inv_quant_dc * self.dequant.dc_quant(c)
    }

    pub fn get_inv_dc_step(&self, c: usize) -> f32 {
        self.dequant.inv_dc_quant(c) * (self.global_scale_float * self.quant_dc as f32)
    }

    /// Scale factor for combining with raw quant field
    pub fn scale(&self) -> f32 {
        self.global_scale_float
    }

    pub fn inv_global_scale(&self) -> f32 {
        self.inv_global_scale
    }

    /// Inverse quant for AC coefficients at given quant level
    pub fn inv_quant_ac(&self, quant: i32) -> f32 {
        self.inv_global_scale / quant as f32
    }

    /// Get dequantization matrix for a strategy/channel
    pub fn dequant_matrix(&self, strategy: AcStrategyType, c: usize) -> &[f32] {
        self.dequant.matrix(strategy, c)
    }

    /// Get inverse dequantization matrix (for encoding)
    pub fn inv_dequant_matrix(&self, strategy: AcStrategyType, c: usize) -> &[f32] {
        self.dequant.inv_matrix(strategy, c)
    }

    /// Set quantization field from floating-point values
    pub fn set_quant_field(
        &mut self,
        quant_dc: f32,
        qf: &ImageF,
        raw_quant_field: &mut ImageI,
    ) -> Result<(), Error> {
        // Compute global scale and DC quant from the field
        self.compute_global_scale_and_quant(quant_dc, qf);

        // Convert float field to integer field
        for y in 0..qf.ysize() {
            for x in 0..qf.xsize() {
                let val = qf.pixel(x, y) * self.scale();
                let clamped = val.round().clamp(1.0, QUANT_MAX as f32) as i32;
                *raw_quant_field.pixel_mut(x, y) = clamped;
            }
        }

        Ok(())
    }

    fn compute_global_scale_and_quant(&mut self, quant_dc: f32, qf: &ImageF) {
        // Find median and median absolute deviation
        let mut values: Vec<f32> = qf.pixels().iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = values[values.len() / 2];
        let median_absd = {
            let mut devs: Vec<f32> = values.iter()
                .map(|&v| (v - median).abs())
                .collect();
            devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            devs[devs.len() / 2]
        };

        // Compute global scale
        let upper = median + 3.0 * median_absd;
        self.global_scale = ((QUANT_MAX as f32 * GLOBAL_SCALE_DENOM as f32) / upper)
            .round()
            .clamp(1.0, GLOBAL_SCALE_DENOM as f32 * 4.0) as i32;

        // Compute DC quant
        self.quant_dc = (quant_dc * self.global_scale as f32 / GLOBAL_SCALE_DENOM as f32)
            .round()
            .clamp(1.0, QUANT_MAX as f32) as i32;

        self.recompute_from_global_scale();
    }
}
```

### 7.2 Quantization Weights (Dequant Matrices)

```rust
// src/quant/quant_weights.rs

/// Quantization/dequantization matrices for all transform types.
///
/// These define how aggressively each frequency is quantized.
/// Lower values = more aggressive quantization = more loss.
pub struct DequantMatrices {
    /// Forward table: multipliers for dequantization
    table: AlignedVec<f32>,
    /// Inverse table: multipliers for quantization
    inv_table: AlignedVec<f32>,
    /// Offsets into tables for each (strategy, channel)
    table_offsets: [usize; AcStrategyType::NUM_VALID * 3],
    /// DC quantization values per channel
    dc_quant: [f32; 3],
    inv_dc_quant: [f32; 3],
    /// Which strategies have been computed
    computed_mask: u32,
    /// Encoding parameters for each quant table
    encodings: Vec<QuantEncoding>,
}

impl DequantMatrices {
    pub fn new() -> Self {
        Self {
            table: AlignedVec::new(),
            inv_table: AlignedVec::new(),
            table_offsets: [0; AcStrategyType::NUM_VALID * 3],
            dc_quant: DC_QUANT,
            inv_dc_quant: INV_DC_QUANT,
            computed_mask: 0,
            encodings: Vec::new(),
        }
    }

    /// Get dequantization matrix for a strategy/channel
    pub fn matrix(&self, strategy: AcStrategyType, c: usize) -> &[f32] {
        let offset = self.table_offsets[strategy as usize * 3 + c];
        let size = strategy.covered_blocks_x() * strategy.covered_blocks_y() * DCT_BLOCK_SIZE;
        &self.table[offset..offset + size]
    }

    /// Get inverse (quantization) matrix
    pub fn inv_matrix(&self, strategy: AcStrategyType, c: usize) -> &[f32] {
        let offset = self.table_offsets[strategy as usize * 3 + c];
        let size = strategy.covered_blocks_x() * strategy.covered_blocks_y() * DCT_BLOCK_SIZE;
        &self.inv_table[offset..offset + size]
    }

    pub fn dc_quant(&self, c: usize) -> f32 { self.dc_quant[c] }
    pub fn inv_dc_quant(&self, c: usize) -> f32 { self.inv_dc_quant[c] }

    /// Ensure matrices are computed for the given strategies
    pub fn ensure_computed(&mut self, acs_mask: u32) -> Result<(), Error> {
        let needed = acs_mask & !self.computed_mask;
        if needed == 0 {
            return Ok(());
        }

        // Compute missing matrices
        for i in 0..AcStrategyType::NUM_VALID {
            if (needed & (1 << i)) != 0 {
                self.compute_quant_table(AcStrategyType::try_from(i as u8)?)?;
            }
        }

        self.computed_mask |= needed;
        Ok(())
    }

    fn compute_quant_table(&mut self, strategy: AcStrategyType) -> Result<(), Error> {
        // Get the encoding for this strategy
        let quant_table = strategy_to_quant_table(strategy);
        let encoding = self.encodings.get(quant_table as usize)
            .cloned()
            .unwrap_or_else(|| QuantEncoding::library(0));

        let blocks_x = strategy.covered_blocks_x();
        let blocks_y = strategy.covered_blocks_y();
        let size = blocks_x * blocks_y * DCT_BLOCK_SIZE;

        for c in 0..3 {
            let offset = self.table.len();
            self.table_offsets[strategy as usize * 3 + c] = offset;

            // Compute weights based on encoding
            let weights = compute_quant_weights(&encoding, strategy, c)?;

            // Store forward and inverse
            self.table.extend_from_slice(&weights);
            let inv_weights: Vec<f32> = weights.iter().map(|&w| 1.0 / w).collect();
            self.inv_table.extend_from_slice(&inv_weights);
        }

        Ok(())
    }
}

/// Map AC strategy to its quant table
fn strategy_to_quant_table(strategy: AcStrategyType) -> QuantTable {
    const MAP: [QuantTable; 27] = [
        QuantTable::DCT, QuantTable::IDENTITY, QuantTable::DCT2X2,
        QuantTable::DCT4X4, QuantTable::DCT16X16, QuantTable::DCT32X32,
        QuantTable::DCT8X16, QuantTable::DCT8X16, QuantTable::DCT8X32,
        QuantTable::DCT8X32, QuantTable::DCT16X32, QuantTable::DCT16X32,
        QuantTable::DCT4X8, QuantTable::DCT4X8, QuantTable::AFV0,
        QuantTable::AFV0, QuantTable::AFV0, QuantTable::AFV0,
        QuantTable::DCT64X64, QuantTable::DCT32X64, QuantTable::DCT32X64,
        QuantTable::DCT128X128, QuantTable::DCT64X128, QuantTable::DCT64X128,
        QuantTable::DCT256X256, QuantTable::DCT128X256, QuantTable::DCT128X256,
    ];
    MAP[strategy as usize]
}

/// Quant table enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantTable {
    DCT = 0,
    IDENTITY,
    DCT2X2,
    DCT4X4,
    DCT16X16,
    DCT32X32,
    DCT8X16,
    DCT8X32,
    DCT16X32,
    DCT4X8,
    AFV0,
    DCT64X64,
    DCT32X64,
    DCT128X128,
    DCT64X128,
    DCT256X256,
    DCT128X256,
}

/// Quantization encoding modes
#[derive(Clone, Debug)]
pub enum QuantEncoding {
    Library(u8),          // Use predefined library table
    Identity(IdWeights),  // Identity transform weights
    DCT2(DCT2Weights),    // 2×2 DCT weights
    DCT4(DctQuantWeightParams, DCT4Multipliers),
    DCT4X8(DctQuantWeightParams, [f32; 3]),
    AFV(DctQuantWeightParams, DctQuantWeightParams, AFVWeights),
    DCT(DctQuantWeightParams),
    RAW(Vec<i32>, f32),   // JPEG-style explicit table
}

impl QuantEncoding {
    pub fn library(index: u8) -> Self {
        QuantEncoding::Library(index)
    }
}

/// Compute actual quant weights from encoding
fn compute_quant_weights(
    encoding: &QuantEncoding,
    strategy: AcStrategyType,
    channel: usize,
) -> Result<Vec<f32>, Error> {
    // Implementation depends on encoding mode
    // Most common is Library(0) which uses predefined weights
    todo!("Implement quant weight computation")
}
```

### 7.3 Block Quantization

```rust
// src/quant/block.rs

/// Quantize a block of AC coefficients.
///
/// Applies:
/// 1. Multiply by inverse dequant matrix
/// 2. Multiply by quant scale
/// 3. Apply thresholding (dead zone)
/// 4. Round to integer
///
/// Implements libjxl's QuantizeBlockAC
pub fn quantize_block_ac(
    quantizer: &Quantizer,
    channel: usize,
    qm_multiplier: f32,  // Extra multiplier (for X/B channels)
    strategy: AcStrategyType,
    xsize: usize,  // In 8×8 blocks
    ysize: usize,
    thresholds: &mut [f32; 4],  // Dead zone thresholds per quadrant
    block_in: &[f32],     // Input coefficients
    quant: &i32,          // Quantization level
    block_out: &mut [i32], // Output quantized values
) {
    let inv_dequant = quantizer.inv_dequant_matrix(strategy, channel);
    let qac = quantizer.scale() * (*quant as f32);
    let size = xsize * ysize * DCT_BLOCK_SIZE;

    // Adjust thresholds for larger blocks (reduces blocking artifacts)
    if channel != 1 && xsize * ysize >= 4 {
        for t in thresholds.iter_mut() {
            *t -= 0.00744 * (xsize * ysize) as f32;
            *t = t.max(0.5);
        }
    }

    for y in 0..(ysize * BLOCK_DIM) {
        // Determine which quadrant (for threshold selection)
        let yfix = if y >= ysize * BLOCK_DIM / 2 { 2 } else { 0 };

        for x in 0..(xsize * BLOCK_DIM) {
            let xfix = if x >= xsize * BLOCK_DIM / 2 { 1 } else { 0 };
            let threshold = thresholds[yfix + xfix];

            let pos = y * xsize * BLOCK_DIM + x;
            let q = inv_dequant[pos] * qac * qm_multiplier;
            let val = block_in[pos] * q;

            // Apply dead zone thresholding
            let quantized = if val.abs() < threshold {
                0
            } else {
                val.round() as i32
            };

            block_out[pos] = quantized;
        }
    }
}

/// Adjust quantization level based on block content.
///
/// Increases quant for flat blocks to reduce blocking artifacts.
/// Decreases quant for highly detailed blocks.
///
/// Implements libjxl's AdjustQuantBlockAC
pub fn adjust_quant_block_ac(
    quantizer: &Quantizer,
    channel: usize,
    qm_multiplier: f32,
    strategy: AcStrategyType,
    xsize: usize,
    ysize: usize,
    thresholds: &mut [f32; 4],
    block_in: &[f32],
    quant: &mut i32,
) {
    // Skip small block strategies
    let partial_block_kinds = [
        AcStrategyType::IDENTITY,
        AcStrategyType::DCT2X2,
        AcStrategyType::DCT4X4,
        AcStrategyType::DCT4X8,
        AcStrategyType::DCT8X4,
        AcStrategyType::AFV0,
        AcStrategyType::AFV1,
        AcStrategyType::AFV2,
        AcStrategyType::AFV3,
    ];

    if partial_block_kinds.contains(&strategy) {
        return;
    }

    // Analyze block characteristics and adjust quant
    // ... (complex heuristics)
    todo!("Implement quant adjustment heuristics")
}

/// Apply quantization bias adjustment for dequantization.
///
/// Adjusts quantized values to be closer to the expected value
/// of the original (accounting for non-uniform distribution of residuals).
#[inline]
pub fn adjust_quant_bias(
    quantized: i32,
    channel: usize,
    biases: &[f32; 4],
) -> f32 {
    if quantized == 0 {
        0.0
    } else if quantized.abs() == 1 {
        quantized as f32 * biases[channel]
    } else {
        quantized as f32 - biases[3] / quantized as f32
    }
}
```

---

## 8. Module 6: Adaptive Quantization

```rust
// src/quant/adaptive.rs

/// Compute initial quantization field based on image content.
///
/// Uses perceptual masking to determine where more quantization
/// is acceptable (textured areas) vs. where it should be conservative
/// (smooth gradients, edges).
///
/// Returns an image where each pixel corresponds to one 8×8 block,
/// with values indicating relative quantization strength.
///
/// Implements libjxl's InitialQuantField
pub fn initial_quant_field(
    butteraugli_target: f32,
    opsin: &Image3F,  // XYB image
    rect: &Rect,
    pool: &ThreadPool,
    rescale: f32,
) -> Result<(ImageF, ImageF, ImageF), Error> {
    let xsize_blocks = rect.xsize() / BLOCK_DIM;
    let ysize_blocks = rect.ysize() / BLOCK_DIM;

    // Main quant field
    let mut quant_field = ImageF::new(xsize_blocks, ysize_blocks)?;
    // Masking field (for AC strategy selection)
    let mut mask = ImageF::new(xsize_blocks, ysize_blocks)?;
    // 1×1 masking field (per-pixel detail)
    let mut mask_1x1 = ImageF::new(rect.xsize(), rect.ysize())?;

    // Base quantization from target quality
    let base_quant = butteraugli_target_to_quant(butteraugli_target);

    pool.run(ysize_blocks, |by| {
        for bx in 0..xsize_blocks {
            let x = rect.x0() + bx * BLOCK_DIM;
            let y = rect.y0() + by * BLOCK_DIM;

            // Compute local masking based on variance and edge strength
            let (block_mask, detail) = compute_block_masking(opsin, x, y);

            // Quant value: base adjusted by masking
            // High masking = can use higher quant (more compression)
            let quant = base_quant * block_mask * rescale;

            *quant_field.pixel_mut(bx, by) = quant;
            *mask.pixel_mut(bx, by) = block_mask;

            // Fill 1×1 masking
            for py in 0..BLOCK_DIM {
                for px in 0..BLOCK_DIM {
                    *mask_1x1.pixel_mut(bx * BLOCK_DIM + px, by * BLOCK_DIM + py) = detail;
                }
            }
        }
    })?;

    Ok((quant_field, mask, mask_1x1))
}

/// Convert Butteraugli target distance to quantization value
fn butteraugli_target_to_quant(target: f32) -> f32 {
    // Empirically derived mapping
    // Lower target = higher quality = lower quant
    if target <= 0.0 {
        return 1.0;
    }

    // Approximate mapping
    let quant = 0.64 * target.powf(0.5);
    quant.clamp(0.01, 100.0)
}

/// Compute masking for a single block
fn compute_block_masking(opsin: &Image3F, x: usize, y: usize) -> (f32, f32) {
    // Simplified masking computation
    // Full implementation uses Butteraugli-based masking

    let mut variance = 0.0f32;
    let mut mean = 0.0f32;

    // Compute variance of Y channel
    for dy in 0..BLOCK_DIM {
        for dx in 0..BLOCK_DIM {
            let pixel = opsin.pixel(1, x + dx, y + dy);
            mean += pixel;
        }
    }
    mean /= (BLOCK_DIM * BLOCK_DIM) as f32;

    for dy in 0..BLOCK_DIM {
        for dx in 0..BLOCK_DIM {
            let pixel = opsin.pixel(1, x + dx, y + dy);
            let diff = pixel - mean;
            variance += diff * diff;
        }
    }
    variance /= (BLOCK_DIM * BLOCK_DIM) as f32;

    // Higher variance = more masking = can tolerate more quantization
    let masking = 1.0 + variance.sqrt() * 2.0;

    (masking, masking)
}

/// Adjust quantization field after AC strategy is selected.
///
/// Larger transforms need adjusted quantization.
///
/// Implements libjxl's AdjustQuantField
pub fn adjust_quant_field(
    ac_strategy: &AcStrategyImage,
    rect: &Rect,
    butteraugli_target: f32,
    quant_field: &mut ImageF,
) -> Result<(), Error> {
    for by in 0..rect.ysize() {
        for bx in 0..rect.xsize() {
            let acs = ac_strategy.get(rect.x0() + bx, rect.y0() + by);

            if !acs.is_first_block() {
                continue;
            }

            let strategy = acs.strategy();
            let blocks_x = strategy.covered_blocks_x();
            let blocks_y = strategy.covered_blocks_y();

            // Compute average quant for covered blocks
            let mut avg = 0.0f32;
            for iy in 0..blocks_y {
                for ix in 0..blocks_x {
                    avg += quant_field.pixel(bx + ix, by + iy);
                }
            }
            avg /= (blocks_x * blocks_y) as f32;

            // Apply adjustment factor for large transforms
            let factor = match strategy {
                AcStrategyType::DCT16X16 => 0.95,
                AcStrategyType::DCT32X32 => 0.90,
                AcStrategyType::DCT64X64 => 0.85,
                _ => 1.0,
            };

            let adjusted = avg * factor;

            // Set all covered blocks to adjusted value
            for iy in 0..blocks_y {
                for ix in 0..blocks_x {
                    *quant_field.pixel_mut(bx + ix, by + iy) = adjusted;
                }
            }
        }
    }

    Ok(())
}
```

---

## 9. Module 7: Coefficient Ordering

```rust
// src/coeff/order.rs

/// Coefficient order types for different strategies
pub const NUM_ORDERS: usize = 13;

/// Map from AC strategy to order bucket
pub const STRATEGY_ORDER: [u8; 27] = [
    0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1,
    1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
];

/// Compute natural coefficient order for a strategy.
///
/// "Natural order" means increasing anisotropic frequency
/// (diagonal wavefront scanning).
///
/// Implements libjxl's ComputeNaturalCoeffOrder
pub fn compute_natural_coeff_order(strategy: AcStrategyType) -> Vec<u16> {
    let blocks_x = strategy.covered_blocks_x();
    let blocks_y = strategy.covered_blocks_y();
    let size = blocks_x * blocks_y * DCT_BLOCK_SIZE;

    // Generate positions with their "frequency" metric
    let mut positions: Vec<(f32, u16)> = Vec::with_capacity(size);

    for y in 0..(blocks_y * BLOCK_DIM) {
        for x in 0..(blocks_x * BLOCK_DIM) {
            // Skip DC coefficient
            if x == 0 && y == 0 {
                continue;
            }

            // Frequency metric: distance from origin with slight diagonal bias
            let fx = x as f32 / (blocks_x * BLOCK_DIM) as f32;
            let fy = y as f32 / (blocks_y * BLOCK_DIM) as f32;
            let freq = fx * fx + fy * fy + 0.001 * (fx + fy);

            let pos = (y * blocks_x * BLOCK_DIM + x) as u16;
            positions.push((freq, pos));
        }
    }

    // Sort by frequency
    positions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Extract order (DC is always first)
    let mut order = vec![0u16];
    order.extend(positions.iter().map(|(_, pos)| *pos));

    order
}

/// Custom coefficient order (can be optimized per-image).
pub struct CoeffOrder {
    /// Order for each (order_bucket, channel)
    orders: Vec<Vec<u16>>,
}

impl CoeffOrder {
    pub fn new_natural() -> Self {
        let mut orders = Vec::with_capacity(NUM_ORDERS * 3);

        for order_idx in 0..NUM_ORDERS {
            // Find a strategy that uses this order
            let strategy = (0..AcStrategyType::NUM_VALID)
                .map(|i| AcStrategyType::try_from(i as u8).unwrap())
                .find(|&s| STRATEGY_ORDER[s as usize] as usize == order_idx)
                .unwrap_or(AcStrategyType::DCT);

            let order = compute_natural_coeff_order(strategy);

            // Same order for all channels
            for _c in 0..3 {
                orders.push(order.clone());
            }
        }

        Self { orders }
    }

    pub fn get(&self, order_idx: usize, channel: usize) -> &[u16] {
        &self.orders[order_idx * 3 + channel]
    }
}
```

---

## 10. Module 8: Tokenization & Context

### 10.1 Block Context

```rust
// src/coeff/context.rs

/// Block context map configuration.
///
/// Maps (channel, strategy, quant level, DC value) → context ID
pub struct BlockCtxMap {
    /// DC value thresholds per channel
    pub dc_thresholds: [Vec<i32>; 3],
    /// Quantization field thresholds
    pub qf_thresholds: Vec<u32>,
    /// Context map: index → context ID
    pub ctx_map: Vec<u8>,
    /// Number of distinct contexts
    pub num_ctxs: usize,
    /// Number of DC contexts
    pub num_dc_ctxs: usize,
}

impl Default for BlockCtxMap {
    fn default() -> Self {
        // Default context map clusters large transforms together
        const DEFAULT_CTX_MAP: [u8; 39] = [
            0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6,   // channel 0
            7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,  // channel 1
            7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,  // channel 2
        ];

        Self {
            dc_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![],
            ctx_map: DEFAULT_CTX_MAP.to_vec(),
            num_ctxs: 15,  // max(DEFAULT_CTX_MAP) + 1
            num_dc_ctxs: 1,
        }
    }
}

impl BlockCtxMap {
    /// Get context ID for a block
    pub fn context(
        &self,
        dc_idx: usize,
        qf: u32,
        order: usize,
        channel: usize,
    ) -> usize {
        // Compute QF index
        let qf_idx = self.qf_thresholds.iter()
            .filter(|&&t| qf > t)
            .count();

        // Reorder channels: [Y, X, B] → [X, Y, B] for better correlation
        let c = if channel < 2 { channel ^ 1 } else { 2 };

        let idx = c * NUM_ORDERS + order;
        let idx = idx * (self.qf_thresholds.len() + 1) + qf_idx;
        let idx = idx * self.num_dc_ctxs + dc_idx;

        self.ctx_map[idx] as usize
    }

    /// Context for number of non-zeros
    pub fn non_zero_context(&self, non_zeros: u32, block_ctx: u32) -> u32 {
        let ctx = if non_zeros >= 64 {
            36  // Saturate at 64
        } else if non_zeros < 8 {
            non_zeros
        } else {
            4 + non_zeros / 2
        };
        ctx * self.num_ctxs as u32 + block_ctx
    }

    /// Context for AC coefficient based on position and remaining non-zeros
    pub fn zero_density_context(
        &self,
        nonzeros_left: usize,
        k: usize,
        covered_blocks: usize,
        log2_covered_blocks: usize,
        prev: usize,
    ) -> usize {
        // Normalize by block size
        let nzl = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
        let k = k >> log2_covered_blocks;

        debug_assert!(k > 0 && k < 64);
        debug_assert!(nzl > 0 && nzl < 64);

        // Lookup tables for context clustering
        const COEFF_FREQ_CTX: [u16; 64] = [
            0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
            23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
            27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
        ];

        const COEFF_NUM_NZ_CTX: [u16; 64] = [
            0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123,
            152, 152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
            180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        ];

        (COEFF_NUM_NZ_CTX[nzl] + COEFF_FREQ_CTX[k]) as usize * 2 + prev
    }

    /// Number of AC contexts
    pub fn num_ac_contexts(&self) -> usize {
        self.num_ctxs * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT)
    }
}
```

### 10.2 Tokenization

```rust
// src/encode/tokens.rs

use crate::{Token, HybridUintConfig};

/// Tokenize quantized AC coefficients for entropy coding.
///
/// Converts coefficient values to tokens with context information.
pub fn tokenize_ac_block(
    quantized: &[i32],          // Quantized coefficients
    strategy: AcStrategyType,
    order: &[u16],              // Coefficient scan order
    block_ctx: u32,             // Block context
    ctx_map: &BlockCtxMap,
    tokens: &mut Vec<Token>,
) {
    let size = strategy.covered_blocks_x() * strategy.covered_blocks_y() * DCT_BLOCK_SIZE;
    let covered_blocks = strategy.covered_blocks_x() * strategy.covered_blocks_y();
    let log2_covered = strategy.log2_covered_blocks();

    // Count non-zeros (excluding DC)
    let num_nonzeros: usize = quantized[1..size].iter()
        .filter(|&&v| v != 0)
        .count();

    // Token for number of non-zeros
    let nz_ctx = ctx_map.non_zero_context(num_nonzeros as u32, block_ctx);
    tokens.push(Token::new(nz_ctx, num_nonzeros as u32));

    if num_nonzeros == 0 {
        return;
    }

    // Tokenize coefficients in scan order
    let mut nonzeros_left = num_nonzeros;
    let mut prev = 0;  // Previous coefficient sign (for context)

    for (k, &scan_pos) in order.iter().enumerate().skip(1) {
        if nonzeros_left == 0 {
            break;
        }

        let coeff = quantized[scan_pos as usize];

        // Compute context
        let zd_ctx = ctx_map.zero_density_context(
            nonzeros_left, k, covered_blocks, log2_covered, prev
        );
        let ctx = ctx_map.num_ctxs * NON_ZERO_BUCKETS +
                  ZERO_DENSITY_CONTEXT_COUNT * block_ctx as usize +
                  zd_ctx;

        if coeff == 0 {
            tokens.push(Token::new(ctx as u32, 0));
        } else {
            // Encode (abs(coeff) - 1) * 2 + sign
            let abs_coeff = coeff.unsigned_abs();
            let value = (abs_coeff - 1) * 2 + if coeff < 0 { 1 } else { 0 };
            tokens.push(Token::new(ctx as u32, value));

            nonzeros_left -= 1;
            prev = if coeff < 0 { 1 } else { 0 };
        }
    }
}

/// Tokenize DC coefficients for a group.
pub fn tokenize_dc(
    dc: &Image3F,
    rect: &Rect,
    uint_config: &HybridUintConfig,
    tokens: &mut Vec<Token>,
) {
    // DC coefficients are encoded using prediction
    // (difference from predicted value based on neighbors)

    for c in 0..3 {
        for y in rect.y0()..rect.y1() {
            for x in rect.x0()..rect.x1() {
                let dc_val = dc.pixel(c, x, y);

                // Predict from left and above
                let left = if x > 0 { dc.pixel(c, x - 1, y) } else { 0.0 };
                let above = if y > 0 { dc.pixel(c, x, y - 1) } else { 0.0 };
                let predicted = (left + above) / 2.0;

                // Residual
                let residual = dc_val - predicted;
                let quantized = residual.round() as i32;

                // Pack sign and magnitude
                let value = pack_signed(quantized) as u32;

                // Context based on channel and neighbor values
                let ctx = c as u32;  // Simplified - real impl uses neighbor-based context

                tokens.push(Token::new(ctx, value));
            }
        }
    }
}

/// Pack signed integer using zigzag encoding
#[inline]
fn pack_signed(x: i32) -> u32 {
    if x >= 0 {
        (x * 2) as u32
    } else {
        (-x * 2 - 1) as u32
    }
}
```

---

## 11. Module 9: Frame Orchestration

```rust
// src/encode/frame.rs

/// Main VarDCT frame encoding entry point.
///
/// Implements libjxl's EncodeFrame for VarDCT mode.
pub fn encode_vardct_frame(
    image: &Image3F,         // Linear sRGB input
    params: &CompressParams,
    frame_header: &FrameHeader,
    pool: &ThreadPool,
) -> Result<EncodedFrame, Error> {
    let frame_dim = FrameDimensions::from_image(image);

    // Stage 1: Convert to XYB color space
    let mut opsin = image.clone();
    image_to_xyb(&mut opsin, params.intensity_target, pool)?;

    // Stage 2: Initialize encoder state
    let mut state = PassesEncoderState::new(&frame_dim)?;

    // Stage 3: Initialize quantization matrices
    state.shared.matrices.ensure_computed(0xFFFFFFFF)?;

    // Stage 4: Compute initial quant field (adaptive quantization)
    let (quant_field, mask, mask_1x1) = initial_quant_field(
        params.butteraugli_distance,
        &opsin,
        &Rect::from_size(opsin.xsize(), opsin.ysize()),
        pool,
        1.0,
    )?;

    // Stage 5: Find best AC strategy
    let mut ac_strategy = AcStrategyImage::new(
        frame_dim.xsize_blocks, frame_dim.ysize_blocks
    )?;

    let acs_config = ACSConfig {
        dequant: state.shared.matrices.clone(),
        quant_field: quant_field.clone(),
        masking_field: mask,
        src_image: opsin.clone(),
        // ... other fields
        info_loss_multiplier: 1.0,
        cost_delta: 1.0,
        zeros_mul: 1.0,
    };

    find_best_ac_strategy(&acs_config, params, &mut ac_strategy, pool)?;

    // Stage 6: Compute chroma-from-luma parameters
    let mut cmap = ColorCorrelationMap::new(opsin.xsize(), opsin.ysize())?;
    find_best_color_correlation_map(&opsin, &mut cmap, pool)?;

    // Stage 7: Adjust quant field for chosen strategies
    adjust_quant_field(&ac_strategy, &Rect::from_size(
        frame_dim.xsize_blocks, frame_dim.ysize_blocks
    ), params.butteraugli_distance, &mut state.shared.quant_field)?;

    // Stage 8: Set up quantizer
    state.shared.quantizer.set_quant_field(
        initial_quant_dc(params.butteraugli_distance),
        &quant_field,
        &mut state.shared.raw_quant_field,
    )?;

    // Stage 9: Process groups (compute coefficients)
    let dc = process_groups_compute_coefficients(
        &opsin, &ac_strategy, &cmap, &mut state, pool
    )?;

    // Stage 10: Tokenize coefficients
    let tokens = tokenize_all_groups(&state, &ac_strategy, pool)?;

    // Stage 11: Entropy coding (histogram clustering + ANS)
    let encoded_data = build_and_encode_histograms(
        &params.histogram_params(),
        state.shared.block_ctx_map.num_ac_contexts(),
        &tokens,
    )?;

    // Stage 12: Assemble bitstream
    let bitstream = assemble_frame_bitstream(
        frame_header,
        &state,
        &encoded_data,
        &dc,
    )?;

    Ok(EncodedFrame { bitstream })
}

/// Process all groups to compute DCT coefficients.
fn process_groups_compute_coefficients(
    opsin: &Image3F,
    ac_strategy: &AcStrategyImage,
    cmap: &ColorCorrelationMap,
    state: &mut PassesEncoderState,
    pool: &ThreadPool,
) -> Result<Image3F, Error> {
    let frame_dim = &state.shared.frame_dim;
    let num_groups = frame_dim.num_groups();

    // DC image (one sample per 8×8 block)
    let mut dc = Image3F::new(frame_dim.xsize_blocks, frame_dim.ysize_blocks)?;

    pool.run(num_groups, |group_idx| {
        let group_rect = frame_dim.group_rect(group_idx);
        let block_rect = frame_dim.block_group_rect(group_idx);

        compute_coefficients_group(
            group_idx,
            opsin,
            &group_rect,
            ac_strategy,
            cmap,
            state,
            &mut dc,
            &block_rect,
        )
    })?;

    Ok(dc)
}

/// Compute coefficients for a single group.
///
/// This is the core of VarDCT encoding:
/// 1. For each block, apply forward DCT based on AC strategy
/// 2. Apply chroma-from-luma correlation
/// 3. Quantize coefficients
/// 4. Store for later entropy coding
///
/// Implements libjxl's ComputeCoefficients
fn compute_coefficients_group(
    group_idx: usize,
    opsin: &Image3F,
    rect: &Rect,
    ac_strategy: &AcStrategyImage,
    cmap: &ColorCorrelationMap,
    state: &mut PassesEncoderState,
    dc: &mut Image3F,
    block_rect: &Rect,
) -> Result<(), Error> {
    let quantizer = &state.shared.quantizer;
    let mut scratch = vec![0.0f32; MAX_COEFF_AREA * 5];
    let mut coeffs_in = vec![0.0f32; MAX_COEFF_AREA * 3];
    let mut quantized = vec![0i32; MAX_COEFF_AREA * 3];

    for by in 0..block_rect.ysize() {
        let cmap_ty = by / COLOR_TILE_DIM_IN_BLOCKS;

        for bx in 0..block_rect.xsize() {
            let acs = ac_strategy.get(block_rect.x0() + bx, block_rect.y0() + by);

            if !acs.is_first_block() {
                continue;
            }

            let strategy = acs.strategy();
            let xblocks = strategy.covered_blocks_x();
            let yblocks = strategy.covered_blocks_y();
            let size = xblocks * yblocks * DCT_BLOCK_SIZE;

            // Get quant level for this block
            let quant = state.shared.raw_quant_field.pixel(
                block_rect.x0() + bx,
                block_rect.y0() + by,
            );
            let mut quant_ac = *quant;

            // Get CfL parameters for this tile
            let cmap_tx = bx / COLOR_TILE_DIM_IN_BLOCKS;
            let x_factor = cmap.ytox_map.pixel(cmap_tx, cmap_ty);
            let b_factor = cmap.ytob_map.pixel(cmap_tx, cmap_ty);
            let x_ratio = cmap.base().ytox_ratio(*x_factor as i32);
            let b_ratio = cmap.base().ytob_ratio(*b_factor as i32);

            // Transform all 3 channels
            for c in 0..3 {
                let px = rect.x0() + bx * BLOCK_DIM;
                let py = rect.y0() + by * BLOCK_DIM;

                transform_from_pixels(
                    strategy,
                    &opsin.plane_pixels(c, px, py),
                    opsin.pixels_per_row(),
                    &mut coeffs_in[c * size..],
                    &mut scratch,
                );
            }

            // Quantize Y channel with roundtrip
            quantize_roundtrip_y_block(
                state, size, quantizer, strategy,
                xblocks, yblocks,
                &mut quant_ac,
                &mut coeffs_in,
                &mut quantized,
            );

            // Unapply color correlation
            for k in 0..size {
                let y = coeffs_in[size + k];
                coeffs_in[k] -= x_ratio * y;          // X residual
                coeffs_in[2 * size + k] -= b_ratio * y; // B residual
            }

            // Quantize X and B channels
            for c in [0, 2] {
                let qm_mul = if c == 0 {
                    state.x_qm_multiplier
                } else {
                    state.b_qm_multiplier
                };

                let mut thres = [0.58f32, 0.62, 0.62, 0.62];
                quantize_block_ac(
                    quantizer, c, qm_mul, strategy,
                    xblocks, yblocks,
                    &mut thres,
                    &coeffs_in[c * size..],
                    &quant_ac,
                    &mut quantized[c * size..],
                );

                // Extract DC
                dc_from_lowest_frequencies(
                    strategy,
                    &coeffs_in[c * size..],
                    dc.pixel_mut(c, bx, by),
                    dc.pixels_per_row(),
                    &mut scratch,
                );
            }

            // Store quantized coefficients for entropy coding
            state.store_coefficients(group_idx, bx, by, &quantized[..3 * size]);
        }
    }

    Ok(())
}
```

---

## 12. Data Structures Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KEY DATA STRUCTURES                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐
│ Image Types                     │
├─────────────────────────────────┤
│ Image3F: 3-channel float image  │
│ ImageF:  1-channel float image  │
│ ImageI:  1-channel i32 image    │
│ ImageU8: 1-channel u8 image     │
│ ImageI8: 1-channel i8 image     │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ AC Strategy                     │
├─────────────────────────────────┤
│ AcStrategyType: enum (27 types) │
│ AcStrategy: type + is_first     │
│ AcStrategyImage: per-block map  │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Quantization                    │
├─────────────────────────────────┤
│ Quantizer: main quantizer       │
│ DequantMatrices: weight tables  │
│ QuantEncoding: table definition │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Color Correlation               │
├─────────────────────────────────┤
│ ColorCorrelation: base params   │
│ ColorCorrelationMap: per-tile   │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Context & Tokenization          │
├─────────────────────────────────┤
│ BlockCtxMap: context mapping    │
│ Token: (context, value) pair    │
│ CoeffOrder: scan order          │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ Encoder State                   │
├─────────────────────────────────┤
│ PassesEncoderState: full state  │
│ SharedState: shared across grps │
│ FrameDimensions: size info      │
└─────────────────────────────────┘
```

---

## 13. Implementation Order

```
Phase 1: Core Types & Infrastructure
├── common/image.rs (Image types)
├── common/frame_dim.rs (Dimensions)
├── common/constants.rs
└── Tests: basic image operations

Phase 2: Color Transform
├── color/opsin_params.rs (Constants)
├── color/xyb.rs (RGB→XYB)
└── Tests: parity with libjxl ToXYB

Phase 3: Transforms
├── transform/dct_scales.rs
├── transform/dct.rs (Forward DCT)
├── transform/mod.rs (Per-strategy dispatch)
└── Tests: DCT parity (exact coefficients)

Phase 4: AC Strategy
├── strategy/ac_strategy.rs (Types)
├── strategy/selection.rs (Heuristics)
└── Tests: strategy image operations

Phase 5: Quantization
├── quant/quant_weights.rs (Dequant matrices)
├── quant/quantizer.rs (Main quantizer)
├── quant/block.rs (Block quantization)
├── quant/adaptive.rs (Adaptive quant)
└── Tests: quantization parity

Phase 6: Chroma From Luma
├── cfl/chroma_from_luma.rs
└── Tests: CfL parameter selection

Phase 7: Coefficient Ordering & Context
├── coeff/order.rs
├── coeff/context.rs (BlockCtxMap)
└── Tests: context computation

Phase 8: Tokenization
├── encode/tokens.rs
└── Tests: token generation parity

Phase 9: Frame Orchestration
├── encode/group.rs (Group processing)
├── encode/frame.rs (Main entry)
└── Integration tests

Phase 10: Full Pipeline Integration
├── End-to-end encoding
├── Decode verification (using libjxl decoder)
└── Quality verification (SSIMULACRA2)
```

---

## 14. Test Strategy

### 14.1 Stage-by-Stage Parity Tests

```rust
// Each stage should produce identical output to libjxl

#[test]
fn test_xyb_transform_parity() {
    let rgb = load_test_image();
    let rust_xyb = image_to_xyb(&rgb);
    let cpp_xyb = load_reference_xyb();  // From libjxl
    assert_images_equal(&rust_xyb, &cpp_xyb, 1e-6);
}

#[test]
fn test_dct_parity() {
    for strategy in ALL_STRATEGIES {
        let pixels = random_block(strategy);
        let rust_coeffs = forward_dct(strategy, &pixels);
        let cpp_coeffs = reference_dct(strategy, &pixels);
        assert_coeffs_equal(&rust_coeffs, &cpp_coeffs, 1e-5);
    }
}

#[test]
fn test_quantization_parity() {
    let coeffs = random_coeffs();
    let rust_quant = quantize_block(&coeffs);
    let cpp_quant = reference_quantize(&coeffs);
    assert_eq!(rust_quant, cpp_quant);  // Exact match for integers
}
```

### 14.2 Round-Trip Tests

```rust
#[test]
fn test_encode_decode_roundtrip() {
    let original = load_test_image();
    let encoded = encode_vardct(&original, quality);
    let decoded = libjxl_decode(&encoded);  // Use libjxl to decode

    let ssimulacra2 = compute_ssimulacra2(&original, &decoded);
    assert!(ssimulacra2 > expected_quality_floor);
}
```

### 14.3 Bitstream Compatibility

```rust
#[test]
fn test_bitstream_decodable() {
    let image = load_test_image();
    let encoded = encode_vardct(&image, params);

    // Must be decodable by reference decoder
    let result = libjxl_decode(&encoded);
    assert!(result.is_ok());
}
```

---

## Reference: libjxl Source Files

| Component | Primary File(s) |
|-----------|-----------------|
| XYB Transform | `enc_xyb.cc`, `cms/opsin_params.h` |
| AC Strategy | `ac_strategy.h`, `enc_ac_strategy.cc` |
| DCT | `dct-inl.h`, `enc_transforms-inl.h` |
| Chroma-from-Luma | `chroma_from_luma.h`, `enc_chroma_from_luma.cc` |
| Quantization | `quantizer.h`, `quant_weights.h` |
| Adaptive Quant | `enc_adaptive_quantization.cc` |
| Coefficient Order | `coeff_order.h`, `enc_coeff_order.cc` |
| Context | `ac_context.h` |
| Frame Encoding | `enc_frame.cc`, `enc_group.cc` |
| Tokenization | `enc_ans.cc` (WriteTokens) |
