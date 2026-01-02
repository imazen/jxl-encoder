//! Coefficient tokenization for VarDCT encoding.
//!
//! Converts quantized DCT coefficients into tokens for entropy coding.

use super::context::{BlockContextMap, zero_density_context};
use super::enc_coeff::pack_signed;

/// A token representing a coefficient or non-zero count.
#[derive(Clone, Copy, Debug)]
pub struct Token {
    /// Context for this token.
    pub context: u32,
    /// Token value (packed using pack_signed for coefficients).
    pub value: u32,
}

impl Token {
    /// Create a new token.
    pub fn new(context: u32, value: u32) -> Self {
        Self { context, value }
    }
}

/// Tokenize a block of quantized AC coefficients.
///
/// # Arguments
/// * `coeffs` - Quantized coefficients in natural order (DC at index 0)
/// * `order` - Coefficient scan order (permutation of indices)
/// * `block_context` - Block context from BlockContextMap
/// * `bcm` - Block context map for context computation
/// * `log2_covered_blocks` - Log2 of number of 8x8 blocks in transform
/// * `tokens` - Output token vector
pub fn tokenize_block(
    coeffs: &[i32],
    order: &[usize],
    block_context: usize,
    bcm: &BlockContextMap,
    log2_covered_blocks: usize,
    tokens: &mut Vec<Token>,
) {
    let size = coeffs.len();
    let covered_blocks = 1usize << log2_covered_blocks;

    // Count non-zeros (excluding DC which is at position 0)
    let nzeros: i32 = coeffs[covered_blocks..].iter().filter(|&&c| c != 0).count() as i32;

    // Emit non-zero count token
    let nzero_ctx = bcm.nonzero_context(nzeros as usize, block_context) as u32;
    tokens.push(Token::new(nzero_ctx, nzeros as u32));

    // Skip if no non-zeros
    if nzeros == 0 {
        return;
    }

    // Get zero-density context offset
    let histo_offset = bcm.zero_density_context_offset(block_context);

    // Process coefficients in scan order, skipping LLF (low-low frequency = DC area)
    let mut nzeros_left = nzeros as usize;
    let mut prev = if nzeros_left > size / 16 { 0 } else { 1 };

    for k in covered_blocks..size {
        if nzeros_left == 0 {
            break;
        }

        let coeff = coeffs[order[k]];
        let ctx = histo_offset + zero_density_context(nzeros_left, k, log2_covered_blocks, prev);

        let u_coeff = pack_signed(coeff);
        tokens.push(Token::new(ctx as u32, u_coeff));

        if coeff != 0 {
            prev = 1;
            nzeros_left -= 1;
        } else {
            prev = 0;
        }
    }
}

/// Tokenize an 8x8 block using default (natural) coefficient order.
pub fn tokenize_block_8x8(
    coeffs: &[i32; 64],
    block_context: usize,
    bcm: &BlockContextMap,
    tokens: &mut Vec<Token>,
) {
    // Natural order for 8x8 DCT
    let natural_order: Vec<usize> = (0..64).collect();
    tokenize_block(coeffs, &natural_order, block_context, bcm, 0, tokens);
}

/// Natural (raster) scan order for 8x8 block.
pub const NATURAL_ORDER_8X8: [usize; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
];

/// Zigzag scan order for 8x8 block (JPEG-style).
pub const ZIGZAG_ORDER_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Generate natural coefficient order for a block.
///
/// The natural order places LLF (DC-equivalent) coefficients first,
/// then AC coefficients in a zigzag-like pattern.
///
/// # Arguments
/// * `cx` - Number of 8x8 blocks in x direction (covered_blocks_x)
/// * `cy` - Number of 8x8 blocks in y direction (covered_blocks_y)
pub fn generate_natural_order(cx: usize, cy: usize) -> Vec<usize> {
    let block_dim = 8;
    let width = cx * block_dim;
    let height = cy * block_dim;
    let size = width * height;

    let mut order = vec![0usize; size];
    let covered_blocks = cx * cy;

    // Generate zigzag order for the full block
    // The first covered_blocks positions are the LLF (one per 8x8 block)
    // The rest follow zigzag pattern

    // For square blocks, use standard zigzag
    if cx == cy {
        let mut cur = covered_blocks;

        // First half of zigzag (upper-left triangle)
        for i in 0..width {
            for j in 0..=i {
                let (x, y) = if i % 2 == 0 { (i - j, j) } else { (j, i - j) };

                if x < width && y < height {
                    // Check if this is an LLF position
                    let block_x = x / block_dim;
                    let block_y = y / block_dim;
                    let local_x = x % block_dim;
                    let local_y = y % block_dim;

                    if local_x == 0 && local_y == 0 {
                        // LLF position - DC of this 8x8 block
                        let llf_idx = block_y * cx + block_x;
                        order[llf_idx] = y * width + x;
                    } else {
                        // AC position
                        if cur < size {
                            order[cur] = y * width + x;
                            cur += 1;
                        }
                    }
                }
            }
        }

        // Second half of zigzag (lower-right triangle)
        for ip in (1..width).rev() {
            let i = ip;
            for j in 0..i {
                let x_base = width - i + j;
                let y_base = width - 1 - j;
                let (x, y) = if (width - i).is_multiple_of(2) {
                    (x_base, y_base)
                } else {
                    (y_base, x_base)
                };

                if x < width && y < height {
                    let block_x = x / block_dim;
                    let block_y = y / block_dim;
                    let local_x = x % block_dim;
                    let local_y = y % block_dim;

                    if local_x == 0 && local_y == 0 {
                        let llf_idx = block_y * cx + block_x;
                        order[llf_idx] = y * width + x;
                    } else if cur < size {
                        order[cur] = y * width + x;
                        cur += 1;
                    }
                }
            }
        }
    } else {
        // For non-square, use simple raster order with LLF first
        let mut cur = covered_blocks;
        for by in 0..cy {
            for bx in 0..cx {
                let llf_idx = by * cx + bx;
                order[llf_idx] = (by * block_dim) * width + (bx * block_dim);
            }
        }
        for y in 0..height {
            for x in 0..width {
                let local_x = x % block_dim;
                let local_y = y % block_dim;
                if local_x != 0 || local_y != 0 {
                    order[cur] = y * width + x;
                    cur += 1;
                }
            }
        }
    }

    order
}

/// Get log2 of covered blocks for a given AC strategy.
#[inline]
pub fn log2_covered_blocks_for_strategy(cx: usize, cy: usize) -> usize {
    let covered = cx * cy;
    match covered {
        1 => 0,  // DCT8
        4 => 2,  // DCT16
        16 => 4, // DCT32
        _ => (covered as f32).log2().ceil() as usize,
    }
}

/// Collect all tokens from multiple blocks.
pub struct TokenCollector {
    pub tokens: Vec<Token>,
}

impl Default for TokenCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCollector {
    /// Create a new token collector.
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Create with reserved capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
        }
    }

    /// Add tokens for a block.
    pub fn add_block(
        &mut self,
        coeffs: &[i32],
        order: &[usize],
        block_context: usize,
        bcm: &BlockContextMap,
        log2_covered_blocks: usize,
    ) {
        tokenize_block(
            coeffs,
            order,
            block_context,
            bcm,
            log2_covered_blocks,
            &mut self.tokens,
        );
    }

    /// Get collected tokens.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Take collected tokens.
    pub fn take_tokens(self) -> Vec<Token> {
        self.tokens
    }

    /// Clear collected tokens.
    pub fn clear(&mut self) {
        self.tokens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let t = Token::new(42, 100);
        assert_eq!(t.context, 42);
        assert_eq!(t.value, 100);
    }

    #[test]
    fn test_tokenize_all_zeros() {
        let coeffs = [0i32; 64];
        let bcm = BlockContextMap::new_default();
        let mut tokens = Vec::new();

        tokenize_block_8x8(&coeffs, 0, &bcm, &mut tokens);

        // Should have exactly 1 token (the nzeros=0 count)
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, 0); // 0 non-zeros
    }

    #[test]
    fn test_tokenize_one_nonzero() {
        let mut coeffs = [0i32; 64];
        coeffs[1] = 5; // One non-zero at position 1

        let bcm = BlockContextMap::new_default();
        let mut tokens = Vec::new();

        tokenize_block_8x8(&coeffs, 0, &bcm, &mut tokens);

        // Should have 2 tokens: nzeros=1, then the coefficient
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, 1); // 1 non-zero
        assert_eq!(tokens[1].value, pack_signed(5)); // coefficient 5
    }

    #[test]
    fn test_tokenize_multiple_nonzeros() {
        let mut coeffs = [0i32; 64];
        coeffs[1] = 10;
        coeffs[5] = -3;
        coeffs[10] = 1;

        let bcm = BlockContextMap::new_default();
        let mut tokens = Vec::new();

        tokenize_block_8x8(&coeffs, 0, &bcm, &mut tokens);

        // Should have 4 tokens: nzeros=3, then 3 coefficients
        // But we emit tokens for all positions from 1 to the last non-zero
        assert!(tokens.len() >= 4);
        assert_eq!(tokens[0].value, 3); // 3 non-zeros
    }

    #[test]
    fn test_token_collector() {
        let mut collector = TokenCollector::new();
        let coeffs = [0i32; 64];
        let order: Vec<usize> = (0..64).collect();
        let bcm = BlockContextMap::new_default();

        collector.add_block(&coeffs, &order, 0, &bcm, 0);
        collector.add_block(&coeffs, &order, 0, &bcm, 0);

        assert_eq!(collector.tokens().len(), 2); // 2 nzeros tokens
    }

    #[test]
    fn test_zigzag_order() {
        // First few positions in zigzag order
        assert_eq!(ZIGZAG_ORDER_8X8[0], 0); // DC
        assert_eq!(ZIGZAG_ORDER_8X8[1], 1); // (0,1)
        assert_eq!(ZIGZAG_ORDER_8X8[2], 8); // (1,0)
        assert_eq!(ZIGZAG_ORDER_8X8[3], 16); // (2,0)
        assert_eq!(ZIGZAG_ORDER_8X8[4], 9); // (1,1)
        assert_eq!(ZIGZAG_ORDER_8X8[5], 2); // (0,2)
    }

    #[test]
    fn test_generate_natural_order_8x8() {
        let order = generate_natural_order(1, 1);
        assert_eq!(order.len(), 64);
        // First position should be DC (position 0)
        assert_eq!(order[0], 0);
        // All positions should be unique
        let mut sorted = order.clone();
        sorted.sort();
        for i in 0..64 {
            assert_eq!(sorted[i], i);
        }
    }

    #[test]
    fn test_generate_natural_order_16x16() {
        let order = generate_natural_order(2, 2);
        assert_eq!(order.len(), 256);
        // First 4 positions are LLF (DC of each 8x8 block)
        // LLF positions are at (0,0), (8,0), (0,8), (8,8) in the 16x16 grid
        let llf_positions = [0, 8, 16 * 8, 16 * 8 + 8]; // 0, 8, 128, 136
        for i in 0..4 {
            assert!(
                llf_positions.contains(&order[i]),
                "order[{}] = {} should be an LLF position",
                i,
                order[i]
            );
        }
        // All positions should be unique
        let mut sorted = order.clone();
        sorted.sort();
        for i in 0..256 {
            assert_eq!(sorted[i], i, "Position {} missing from order", i);
        }
    }

    #[test]
    fn test_generate_natural_order_32x32() {
        let order = generate_natural_order(4, 4);
        assert_eq!(order.len(), 1024);
        // First 16 positions are LLF (DC of each 8x8 block)
        // All positions should be unique
        let mut sorted = order.clone();
        sorted.sort();
        for i in 0..1024 {
            assert_eq!(sorted[i], i, "Position {} missing from order", i);
        }
    }

    #[test]
    fn test_log2_covered_blocks() {
        assert_eq!(log2_covered_blocks_for_strategy(1, 1), 0); // DCT8
        assert_eq!(log2_covered_blocks_for_strategy(2, 2), 2); // DCT16
        assert_eq!(log2_covered_blocks_for_strategy(4, 4), 4); // DCT32
    }
}
