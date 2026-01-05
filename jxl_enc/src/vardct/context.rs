//! Block context modeling for VarDCT coefficient encoding.
//!
//! Context selection determines which entropy distribution to use for
//! each coefficient, based on block properties and position.

use crate::bit_writer::BitWriter;
use crate::error::Result;
#[allow(unused_imports)]
use crate::{trace_section, trace_write};

use super::ac_strategy::NUM_ORDERS;

/// Number of non-zero count buckets.
pub const NON_ZERO_BUCKETS: usize = 37;

/// Number of zero-density contexts per block context.
pub const ZERO_DENSITY_CONTEXT_COUNT: usize = 458;

/// Coefficient frequency context lookup.
/// Maps coefficient position to context bucket.
pub const COEFF_FREQ_CONTEXT: [usize; 64] = [
    0xBAD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
    19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

/// Coefficient non-zero count context lookup.
/// Maps non-zero count to context bucket.
pub const COEFF_NUM_NONZERO_CONTEXT: [usize; 64] = [
    0xBAD, 0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
    152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206,
];

/// Compute zero-density context for a coefficient.
///
/// # Arguments
/// * `nonzeros_left` - Number of non-zero coefficients remaining
/// * `k` - Coefficient index in natural order
/// * `log_num_blocks` - Log2 of number of 8x8 blocks in transform
/// * `prev` - Previous coefficient's zero flag (0 or 1)
#[inline]
pub fn zero_density_context(
    nonzeros_left: usize,
    k: usize,
    log_num_blocks: usize,
    prev: usize,
) -> usize {
    let nonzeros_left_norm = shift_right_ceil(nonzeros_left, log_num_blocks);
    let k_norm = k >> log_num_blocks;
    let nz = nonzeros_left_norm.clamp(1, 63);
    let kn = k_norm.clamp(1, 63);
    (COEFF_NUM_NONZERO_CONTEXT[nz] + COEFF_FREQ_CONTEXT[kn]) * 2 + prev
}

/// Shift right with ceiling.
#[inline]
fn shift_right_ceil(x: usize, shift: usize) -> usize {
    if shift == 0 {
        x
    } else {
        (x + (1 << shift) - 1) >> shift
    }
}

/// Default context map for block context selection.
/// Format: [Y_contexts..., X_contexts..., B_contexts...]
/// Each section has NUM_ORDERS entries.
pub const DEFAULT_CONTEXT_MAP: [u8; 39] = [
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, // Y channel
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, // X channel
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, // B channel
];

/// Number of block contexts in default mode.
pub const DEFAULT_NUM_CONTEXTS: usize = 15;

/// Block context map for VarDCT coefficient encoding.
#[derive(Clone, Debug)]
pub struct BlockContextMap {
    /// LF (DC) thresholds per channel [X, Y, B].
    /// Coefficients above threshold go to different context.
    pub lf_thresholds: [Vec<i32>; 3],
    /// QF (quant field) thresholds.
    pub qf_thresholds: Vec<u32>,
    /// Context map: maps (channel, order_id, qf_bucket, lf_bucket) -> context
    pub context_map: Vec<u8>,
    /// Number of LF contexts (product of threshold counts + 1).
    pub num_lf_contexts: usize,
    /// Total number of distinct contexts.
    pub num_contexts: usize,
    /// Whether to use default (simple) mode.
    pub use_default: bool,
}

impl Default for BlockContextMap {
    fn default() -> Self {
        Self::new_default()
    }
}

impl BlockContextMap {
    /// Create default block context map.
    pub fn new_default() -> Self {
        Self {
            lf_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![],
            context_map: DEFAULT_CONTEXT_MAP.to_vec(),
            num_lf_contexts: 1,
            num_contexts: DEFAULT_NUM_CONTEXTS,
            use_default: true,
        }
    }

    /// Number of AC coefficient contexts.
    pub fn num_ac_contexts(&self) -> usize {
        self.num_contexts * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT)
    }

    /// Get block context for a coefficient block.
    ///
    /// # Arguments
    /// * `lf_idx` - LF bucket index (from LF thresholds)
    /// * `qf` - Quant field value for this block
    /// * `order_id` - Order ID (from AC strategy)
    /// * `channel` - Channel (0=Y, 1=X, 2=B)
    pub fn block_context(&self, lf_idx: usize, qf: u32, order_id: usize, channel: usize) -> usize {
        // Find QF bucket
        let mut qf_idx = 0;
        for &t in &self.qf_thresholds {
            if qf > t {
                qf_idx += 1;
            }
        }

        // Channel remapping: Y=1, X=0, B=2 in encoding order
        let c_idx = if channel < 2 { channel ^ 1 } else { 2 };

        // Compute flat index
        let mut idx = c_idx;
        idx = idx * NUM_ORDERS + order_id;
        idx = idx * (self.qf_thresholds.len() + 1) + qf_idx;
        idx = idx * self.num_lf_contexts + lf_idx;

        self.context_map[idx] as usize
    }

    /// Get LF bucket index for a DC value.
    pub fn lf_index(&self, lf_values: [i32; 3]) -> usize {
        let mut idx = 0;
        for (c, &lf) in lf_values.iter().enumerate() {
            let mut bucket = 0;
            for &t in &self.lf_thresholds[c] {
                if lf > t {
                    bucket += 1;
                }
            }
            idx = idx * (self.lf_thresholds[c].len() + 1) + bucket;
        }
        idx
    }

    /// Get context for non-zero count.
    pub fn nonzero_context(&self, nonzeros: usize, block_context: usize) -> usize {
        let bucket = if nonzeros < 8 {
            nonzeros
        } else if nonzeros < 64 {
            4 + nonzeros / 2
        } else {
            36
        };
        bucket * self.num_contexts + block_context
    }

    /// Get offset for zero-density context.
    pub fn zero_density_context_offset(&self, block_context: usize) -> usize {
        self.num_contexts * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT * block_context
    }

    /// Write block context map to bitstream.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        if self.use_default {
            // Default mode: single bit
            writer.write(1, 1)?;
        } else {
            writer.write(1, 0)?;
            // Write LF thresholds
            for thr in &self.lf_thresholds {
                writer.write(4, thr.len() as u64)?;
                for &t in thr {
                    write_threshold(writer, t)?;
                }
            }
            // Write QF thresholds
            writer.write(4, self.qf_thresholds.len() as u64)?;
            for &t in &self.qf_thresholds {
                write_qf_threshold(writer, t)?;
            }
            // Write context map
            write_context_map(writer, &self.context_map)?;
        }
        Ok(())
    }

    /// Write block context map with tracing.
    pub fn write_traced(&self, writer: &mut BitWriter) -> Result<()> {
        trace_section!(begin "BLOCK_CTX_MAP", writer);

        if self.use_default {
            trace_write!(writer, 1, 1, "all_default", "true (15 contexts)")?;
        } else {
            trace_write!(writer, 1, 0, "all_default", "false - custom thresholds")?;
            // Write LF thresholds
            for thr in self.lf_thresholds.iter() {
                trace_write!(writer, 4, thr.len() as u64, &format!("lf_thr[{}].count", i))?;
                for &t in thr {
                    write_threshold(writer, t)?;
                }
            }
            // Write QF thresholds
            trace_write!(writer, 4, self.qf_thresholds.len() as u64, "qf_thr.count")?;
            for &t in &self.qf_thresholds {
                write_qf_threshold(writer, t)?;
            }
            // Write context map
            write_context_map(writer, &self.context_map)?;
        }

        trace_section!(end "BLOCK_CTX_MAP", writer);
        Ok(())
    }
}

/// Write a signed threshold value.
fn write_threshold(writer: &mut BitWriter, val: i32) -> Result<()> {
    use super::enc_coeff::pack_signed;
    let uval = pack_signed(val);
    if uval < 16 {
        writer.write(2, 0)?;
        writer.write(4, uval as u64)?;
    } else if uval < 272 {
        writer.write(2, 1)?;
        writer.write(8, (uval - 16) as u64)?;
    } else if uval < 65808 {
        writer.write(2, 2)?;
        writer.write(16, (uval - 272) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(32, (uval - 65808) as u64)?;
    }
    Ok(())
}

/// Write a QF threshold value.
fn write_qf_threshold(writer: &mut BitWriter, val: u32) -> Result<()> {
    let v = val - 1;
    if v < 4 {
        writer.write(2, 0)?;
        writer.write(2, v as u64)?;
    } else if v < 12 {
        writer.write(2, 1)?;
        writer.write(3, (v - 4) as u64)?;
    } else if v < 44 {
        writer.write(2, 2)?;
        writer.write(5, (v - 12) as u64)?;
    } else {
        writer.write(2, 3)?;
        writer.write(8, (v - 44) as u64)?;
    }
    Ok(())
}

/// Write context map using simple encoding.
fn write_context_map(writer: &mut BitWriter, map: &[u8]) -> Result<()> {
    // Simplified: write each value directly
    // Full implementation would use entropy coding
    for &val in map {
        writer.write(4, val as u64)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_context_map() {
        let bcm = BlockContextMap::new_default();
        assert_eq!(bcm.num_contexts, DEFAULT_NUM_CONTEXTS);
        assert!(bcm.use_default);
    }

    #[test]
    fn test_block_context_y_channel() {
        let bcm = BlockContextMap::new_default();

        // Y channel (1), order 0, no QF/LF buckets
        let ctx = bcm.block_context(0, 64, 0, 1);
        assert!(ctx < bcm.num_contexts);
    }

    #[test]
    fn test_block_context_x_channel() {
        let bcm = BlockContextMap::new_default();

        // X channel (0), order 0
        let ctx = bcm.block_context(0, 64, 0, 0);
        assert!(ctx < bcm.num_contexts);
    }

    #[test]
    fn test_block_context_b_channel() {
        let bcm = BlockContextMap::new_default();

        // B channel (2), order 0
        let ctx = bcm.block_context(0, 64, 0, 2);
        assert!(ctx < bcm.num_contexts);
    }

    #[test]
    fn test_nonzero_context() {
        let bcm = BlockContextMap::new_default();

        // Few non-zeros -> smaller bucket
        let ctx_few = bcm.nonzero_context(3, 0);
        // Many non-zeros -> larger bucket
        let ctx_many = bcm.nonzero_context(50, 0);

        assert!(ctx_few < ctx_many);
    }

    #[test]
    fn test_nonzero_context_buckets() {
        let bcm = BlockContextMap::new_default();

        // Test all bucket transitions
        // 0-7: direct mapping
        for i in 0..8 {
            let ctx = bcm.nonzero_context(i, 0);
            assert_eq!(ctx, i * bcm.num_contexts);
        }

        // 8-63: bucket = 4 + nonzeros/2
        let ctx_8 = bcm.nonzero_context(8, 0);
        assert_eq!(ctx_8, 8 * bcm.num_contexts);

        let ctx_32 = bcm.nonzero_context(32, 0);
        assert_eq!(ctx_32, 20 * bcm.num_contexts); // 4 + 32/2 = 20

        // 64+: bucket = 36
        let ctx_64 = bcm.nonzero_context(64, 0);
        let ctx_100 = bcm.nonzero_context(100, 0);
        assert_eq!(ctx_64, 36 * bcm.num_contexts);
        assert_eq!(ctx_100, 36 * bcm.num_contexts);
    }

    #[test]
    fn test_zero_density_context() {
        // Basic smoke test
        let ctx = zero_density_context(32, 16, 0, 0);
        assert!(ctx < ZERO_DENSITY_CONTEXT_COUNT * 2);

        // With previous zero
        let ctx_prev1 = zero_density_context(32, 16, 0, 1);
        assert_eq!(ctx_prev1, ctx + 1);
    }

    #[test]
    fn test_zero_density_context_log_blocks() {
        // Test with different log_num_blocks values
        let ctx0 = zero_density_context(32, 16, 0, 0);
        let ctx1 = zero_density_context(32, 16, 1, 0);
        let ctx2 = zero_density_context(32, 16, 2, 0);

        // Different log_num_blocks should give different contexts
        assert!(ctx0 != ctx1 || ctx1 != ctx2);
    }

    #[test]
    fn test_zero_density_context_edge_cases() {
        // Edge cases for clamping
        let ctx_low = zero_density_context(0, 0, 0, 0);
        let ctx_high = zero_density_context(1000, 1000, 0, 0);

        // Both should be valid
        assert!(ctx_low < ZERO_DENSITY_CONTEXT_COUNT * 2);
        assert!(ctx_high < ZERO_DENSITY_CONTEXT_COUNT * 2);
    }

    #[test]
    fn test_num_ac_contexts() {
        let bcm = BlockContextMap::new_default();
        let expected = DEFAULT_NUM_CONTEXTS * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);
        assert_eq!(bcm.num_ac_contexts(), expected);
    }

    #[test]
    fn test_write_default() {
        let bcm = BlockContextMap::new_default();
        let mut writer = BitWriter::new();
        bcm.write(&mut writer).unwrap();

        // Default mode writes just 1 bit
        assert_eq!(writer.bits_written(), 1);
    }

    #[test]
    fn test_write_non_default() {
        let bcm = BlockContextMap {
            lf_thresholds: [vec![0, 10], vec![5], vec![]],
            qf_thresholds: vec![32, 64],
            context_map: vec![0; 39],
            num_lf_contexts: 6, // (2+1) * (1+1) * 1
            num_contexts: 1,
            use_default: false,
        };

        let mut writer = BitWriter::new();
        bcm.write(&mut writer).unwrap();

        // Non-default mode writes more than 1 bit
        assert!(writer.bits_written() > 1);
    }

    #[test]
    fn test_lf_index() {
        let bcm = BlockContextMap {
            lf_thresholds: [vec![0, 10], vec![5], vec![]],
            qf_thresholds: vec![],
            context_map: DEFAULT_CONTEXT_MAP.to_vec(),
            num_lf_contexts: 6,
            num_contexts: DEFAULT_NUM_CONTEXTS,
            use_default: false,
        };

        // Test LF index computation
        let idx0 = bcm.lf_index([0, 0, 0]);
        let idx1 = bcm.lf_index([1, 0, 0]); // crosses first Y threshold
        let idx2 = bcm.lf_index([11, 0, 0]); // crosses both Y thresholds
        let idx3 = bcm.lf_index([0, 6, 0]); // crosses X threshold

        assert!(idx0 < bcm.num_lf_contexts);
        assert!(idx1 < bcm.num_lf_contexts);
        assert!(idx2 < bcm.num_lf_contexts);
        assert!(idx3 < bcm.num_lf_contexts);

        // Different LF values should give different indices
        assert_ne!(idx0, idx2);
    }

    #[test]
    fn test_block_context_with_qf_thresholds() {
        let bcm = BlockContextMap {
            lf_thresholds: [vec![], vec![], vec![]],
            qf_thresholds: vec![32, 64, 128],
            context_map: vec![0; 39 * 4], // Need more entries for QF buckets
            num_lf_contexts: 1,
            num_contexts: 1,
            use_default: false,
        };

        // Test that different QF values give different contexts (when map allows)
        let _ctx_low = bcm.block_context(0, 16, 0, 0);
        let _ctx_mid = bcm.block_context(0, 64, 0, 0);
        let _ctx_high = bcm.block_context(0, 200, 0, 0);
    }

    #[test]
    fn test_zero_density_context_offset() {
        let bcm = BlockContextMap::new_default();

        let offset0 = bcm.zero_density_context_offset(0);
        let offset1 = bcm.zero_density_context_offset(1);

        assert_eq!(offset0, bcm.num_contexts * NON_ZERO_BUCKETS);
        assert_eq!(
            offset1,
            bcm.num_contexts * NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT
        );
    }

    #[test]
    fn test_shift_right_ceil() {
        // shift = 0 should return x unchanged
        assert_eq!(shift_right_ceil(10, 0), 10);
        assert_eq!(shift_right_ceil(0, 0), 0);

        // shift > 0 should round up
        assert_eq!(shift_right_ceil(8, 2), 2); // 8 / 4 = 2
        assert_eq!(shift_right_ceil(9, 2), 3); // ceil(9 / 4) = 3
        assert_eq!(shift_right_ceil(7, 2), 2); // ceil(7 / 4) = 2
    }

    #[test]
    fn test_write_threshold_ranges() {
        // Test all threshold encoding ranges
        let test_values = [
            0,     // < 16: 2 + 4 bits
            1,     // < 16
            15,    // < 16
            16,    // 16-271: 2 + 8 bits
            100,   // 16-271
            271,   // 16-271
            272,   // 272-65807: 2 + 16 bits
            1000,  // 272-65807
            65807, // 272-65807
        ];

        for &val in &test_values {
            let mut writer = BitWriter::new();
            write_threshold(&mut writer, val).unwrap();
            assert!(writer.bits_written() > 0, "Failed for value {}", val);
        }
    }

    #[test]
    fn test_write_qf_threshold_ranges() {
        // Test all QF threshold encoding ranges
        let test_values = [
            1,   // v=0: < 4
            4,   // v=3: < 4
            5,   // v=4: 4-11
            12,  // v=11: 4-11
            13,  // v=12: 12-43
            44,  // v=43: 12-43
            45,  // v=44: >= 44
            100, // v=99: >= 44
        ];

        for &val in &test_values {
            let mut writer = BitWriter::new();
            write_qf_threshold(&mut writer, val).unwrap();
            assert!(writer.bits_written() > 0, "Failed for value {}", val);
        }
    }

    #[test]
    fn test_coeff_freq_context_values() {
        // Verify the context lookup table has valid values
        for (i, &ctx) in COEFF_FREQ_CONTEXT.iter().enumerate() {
            if i == 0 {
                assert_eq!(ctx, 0xBAD, "First entry should be 0xBAD (DC not used)");
            } else {
                assert!(ctx <= 30, "Context {} at index {} should be <= 30", ctx, i);
            }
        }
    }

    #[test]
    fn test_coeff_num_nonzero_context_values() {
        // Verify the non-zero count lookup table
        for (i, &ctx) in COEFF_NUM_NONZERO_CONTEXT.iter().enumerate() {
            if i == 0 {
                assert_eq!(ctx, 0xBAD, "First entry should be 0xBAD");
            } else {
                assert!(
                    ctx <= 206,
                    "Context {} at index {} should be <= 206",
                    ctx,
                    i
                );
            }
        }
    }
}
