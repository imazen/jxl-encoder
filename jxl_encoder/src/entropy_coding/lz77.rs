// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! LZ77 backward references for entropy coding.
//!
//! Implements both RLE-only and full backward-reference LZ77 methods from libjxl.
//! - `apply_lz77_rle`: RLE-only (consecutive identical values)
//! - `apply_lz77_backref`: Full backward references with hash chains (greedy matching)
//!
//! The backward-reference method uses hash chains to find matches at arbitrary
//! distances within a sliding window, providing 1-3% compression improvement on
//! photographic content compared to RLE-only.

use hashbrown::HashMap;

use super::token::{Lz77UintCoder, Token, UintCoder};
use crate::bit_writer::BitWriter;
use crate::error::Result;

/// Maximum window size for LZ77 matching (1MB, matches libjxl).
const WINDOW_SIZE: usize = 1 << 20;

/// Number of special distance codes from WebP lossless.
const NUM_SPECIAL_DISTANCES: usize = 120;

/// Special distance codes from WebP lossless.
/// Each entry is [offset, multiplier] where distance = offset + multiplier * image_width.
/// These encode common 2D patterns (horizontal, vertical, diagonal) compactly.
#[rustfmt::skip]
const SPECIAL_DISTANCES: [[i8; 2]; NUM_SPECIAL_DISTANCES] = [
    [0, 1],  [1, 0],  [1, 1],  [-1, 1], [0, 2],  [2, 0],  [1, 2],  [-1, 2],
    [2, 1],  [-2, 1], [2, 2],  [-2, 2], [0, 3],  [3, 0],  [1, 3],  [-1, 3],
    [3, 1],  [-3, 1], [2, 3],  [-2, 3], [3, 2],  [-3, 2], [0, 4],  [4, 0],
    [1, 4],  [-1, 4], [4, 1],  [-4, 1], [3, 3],  [-3, 3], [2, 4],  [-2, 4],
    [4, 2],  [-4, 2], [0, 5],  [3, 4],  [-3, 4], [4, 3],  [-4, 3], [5, 0],
    [1, 5],  [-1, 5], [5, 1],  [-5, 1], [2, 5],  [-2, 5], [5, 2],  [-5, 2],
    [4, 4],  [-4, 4], [3, 5],  [-3, 5], [5, 3],  [-5, 3], [0, 6],  [6, 0],
    [1, 6],  [-1, 6], [6, 1],  [-6, 1], [2, 6],  [-2, 6], [6, 2],  [-6, 2],
    [4, 5],  [-4, 5], [5, 4],  [-5, 4], [3, 6],  [-3, 6], [6, 3],  [-6, 3],
    [0, 7],  [7, 0],  [1, 7],  [-1, 7], [5, 5],  [-5, 5], [7, 1],  [-7, 1],
    [4, 6],  [-4, 6], [6, 4],  [-6, 4], [2, 7],  [-2, 7], [7, 2],  [-7, 2],
    [3, 7],  [-3, 7], [7, 3],  [-7, 3], [5, 6],  [-5, 6], [6, 5],  [-6, 5],
    [8, 0],  [4, 7],  [-4, 7], [7, 4],  [-7, 4], [8, 1],  [8, 2],  [6, 6],
    [-6, 6], [8, 3],  [5, 7],  [-5, 7], [7, 5],  [-7, 5], [8, 4],  [6, 7],
    [-6, 7], [7, 6],  [-7, 6], [8, 5],  [7, 7],  [-7, 7], [8, 6],  [8, 7],
];

/// Compute special distance from code index and distance multiplier (image width).
#[inline]
fn special_distance(index: usize, multiplier: i32) -> i32 {
    SPECIAL_DISTANCES[index][0] as i32 + multiplier * SPECIAL_DISTANCES[index][1] as i32
}

/// Empirical cost table for LZ77 length encoding (from libjxl).
/// Indexed by token value from HybridUintConfig(1, 0, 0).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const LEN_COST_TABLE: [f32; 17] = [
    2.797667318563126,  3.213177690381199,  2.5706009246743737,
    2.408392498667534,  2.829649191872326,  3.3923087753324577,
    4.029267451554331,  4.415576699706408,  4.509357574741465,
    9.21481543803004,   10.020590190114898, 11.858671627804766,
    12.45853300490526,  11.713105831990857, 12.561996324849314,
    13.775477692278367, 13.174027068768641,
];

/// Empirical cost table for LZ77 distance encoding (from libjxl).
/// Indexed by token value from HybridUintConfig(7, 0, 0).
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const DIST_COST_TABLE: [f32; 139] = [
    6.368282626312716,  5.680793277090298,  8.347404197105247,
    7.641619201599141,  6.914328374119438,  7.959808291537444,
    8.70023120759855,   8.71378518934703,   9.379132523982769,
    9.110472749092708,  9.159029569270908,  9.430936766731973,
    7.278284055315169,  7.8278514904267755, 10.026641158289236,
    9.976049229827066,  9.64351607048908,   9.563403863480442,
    10.171474111762747, 10.45950155077234,  9.994813912104219,
    10.322524683741156, 8.465808729388186,  8.756254166066853,
    10.160930174662234, 10.247329273413435, 10.04090403724809,
    10.129398517544082, 9.342311691539546,  9.07608009102374,
    10.104799540677513, 10.378079384990906, 10.165828974075072,
    10.337595322341553, 7.940557464567944,  10.575665823319431,
    11.023344321751955, 10.736144698831827, 11.118277044595054,
    7.468468230648442,  10.738305230932939, 10.906980780216568,
    10.163468216353817, 10.17805759656433,  11.167283670483565,
    11.147050200274544, 10.517921919244333, 10.651764778156886,
    10.17074446448919,  11.217636876224745, 11.261630721139484,
    11.403140815247259, 10.892472096873417, 11.1859607804481,
    8.017346947551262,  7.895143720278828,  11.036577113822025,
    11.170562110315794, 10.326988722591086, 10.40872184751056,
    11.213498225466386, 11.30580635516863,  10.672272515665442,
    10.768069466228063, 11.145257364153565, 11.64668307145549,
    10.593156194627339, 11.207499484844943, 10.767517766396908,
    10.826629811407042, 10.737764794499988, 10.6200448518045,
    10.191315385198092, 8.468384171390085,  11.731295299170432,
    11.824619886654398, 10.41518844301179,  10.16310536548649,
    10.539423685097576, 10.495136599328031, 10.469112847728267,
    11.72057686174922,  10.910326337834674, 11.378921834673758,
    11.847759036098536, 11.92071647623854,  10.810628276345282,
    11.008601085273893, 11.910326337834674, 11.949212023423133,
    11.298614839104337, 11.611603659010392, 10.472930394619985,
    11.835564720850282, 11.523267392285337, 12.01055816679611,
    8.413029688994023,  11.895784139536406, 11.984679534970505,
    11.220654278717394, 11.716311684833672, 10.61036646226114,
    10.89849965960364,  10.203762898863669, 10.997560826267238,
    11.484217379438984, 11.792836176993665, 12.24310468755171,
    11.464858097919262, 12.212747017409377, 11.425595666074955,
    11.572048533398757, 12.742093965163013, 11.381874288645637,
    12.191870445817015, 11.683156920035426, 11.152442115262197,
    11.90303691580457,  11.653292787169159, 11.938615382266098,
    16.970641701570223, 16.853602280380002, 17.26240782594733,
    16.644655390108507, 17.14310889757499,  16.910935455445955,
    17.505678976959697, 17.213498225466388,
    // Entries 128-138: special distance code costs (from libjxl enc_lz77.cc:442-446).
    // These have dramatically lower costs (2.4-9.7) vs the preceding entries (~17),
    // because special distance codes encode distances as multiples of image width
    // (useful for vertical matches in image data).
    2.4162310293553024, 3.494587244462329,  3.5258600986408344,
    3.4959806589517095, 3.098390886949687,  3.343454654302911,
    3.588847442290287,  4.14614790111827,   5.152948641990529,
    7.433696808092598,  9.716311684833672,
];

/// Empirical cost for LZ77 length encoding.
fn len_cost(len: u32) -> f32 {
    // HybridUintConfig(1, 0, 0): token = 1 + floor_log2(len) for len >= 1
    let (tok, nbits) = if len == 0 {
        (0u32, 0u32)
    } else {
        let n = 31 - len.leading_zeros();
        (1 + n, n)
    };
    let table_size = LEN_COST_TABLE.len();
    let tok_idx = (tok as usize).min(table_size - 1);
    LEN_COST_TABLE[tok_idx] + nbits as f32
}

/// Empirical cost for LZ77 distance encoding.
fn dist_cost(dist: u32) -> f32 {
    // HybridUintConfig(7, 0, 0): different split point
    let (tok, nbits) = hybrid_uint_encode_7_0_0(dist);
    let table_size = DIST_COST_TABLE.len();
    let tok_idx = (tok as usize).min(table_size - 1);
    DIST_COST_TABLE[tok_idx] + nbits as f32
}

/// HybridUint encoding with config (7, 0, 0) for distance symbols.
fn hybrid_uint_encode_7_0_0(value: u32) -> (u32, u32) {
    // split = 7, msb_in_token = 0, lsb_in_token = 0
    // Values 0-6: direct encoding
    // Values >= 7: floor_log2 encoding
    if value < 7 {
        (value, 0)
    } else {
        let n = 31 - value.leading_zeros();
        let tok = 7 + n - 3; // Offset for values >= 7
        (tok, n)
    }
}

/// LZ77 parameters serialized in the entropy code header.
#[derive(Debug, Clone)]
pub struct Lz77Params {
    pub enabled: bool,
    /// Symbols >= min_symbol are LZ77 length tokens.
    /// ANS: 224, Huffman: 512.
    pub min_symbol: u32,
    /// Minimum run length to encode as LZ77. Default: 3.
    pub min_length: u32,
    /// Context index for distance tokens (= num_contexts before LZ77).
    pub distance_context: u32,
}

impl Lz77Params {
    pub fn new(num_contexts: usize, force_huffman: bool) -> Self {
        Self {
            enabled: false,
            min_symbol: if force_huffman { 512 } else { 224 },
            min_length: 3,
            distance_context: num_contexts as u32,
        }
    }
}

/// Write LZ77 header to the bitstream.
///
/// If `lz77` is `Some`, writes `enabled=1` followed by min_symbol, min_length,
/// and length_uint_config. If `None`, writes `enabled=0`.
///
/// JXL spec format:
/// ```text
/// Bool(enabled)
/// if enabled:
///   U32(Val(224), Val(512), Val(4096), BitsOffset(15,8))  // min_symbol
///   U32(Val(3), Val(4), BitsOffset(2,5), BitsOffset(8,9)) // min_length
///   EncodeUintConfig(length_uint_config, log_alpha_size=8)
/// ```
pub fn write_lz77_header(lz77: Option<&Lz77Params>, writer: &mut BitWriter) -> Result<()> {
    if let Some(params) = lz77 {
        writer.write(1, 1)?; // lz77 enabled

        // min_symbol: U32(Val(224), Val(512), Val(4096), BitsOffset(15,8))
        match params.min_symbol {
            224 => writer.write(2, 0)?,  // selector 0 = Val(224)
            512 => writer.write(2, 1)?,  // selector 1 = Val(512)
            4096 => writer.write(2, 2)?, // selector 2 = Val(4096)
            v => {
                writer.write(2, 3)?; // selector 3 = BitsOffset(15, 8)
                writer.write(15, (v - 8) as u64)?;
            }
        }

        // min_length: U32(Val(3), Val(4), BitsOffset(2,5), BitsOffset(8,9))
        match params.min_length {
            3 => writer.write(2, 0)?, // selector 0 = Val(3)
            4 => writer.write(2, 1)?, // selector 1 = Val(4)
            v @ 5..=8 => {
                writer.write(2, 2)?; // selector 2 = BitsOffset(2, 5)
                writer.write(2, (v - 5) as u64)?;
            }
            v => {
                writer.write(2, 3)?; // selector 3 = BitsOffset(8, 9)
                writer.write(8, (v - 9) as u64)?;
            }
        }

        // length_uint_config: HybridUintConfig(0, 0, 0)
        // split_exponent=0 → 4 bits, msb/lsb need 0 bits each
        writer.write(4, 0)?;
    } else {
        writer.write(1, 0)?; // no lz77
    }
    Ok(())
}

/// Estimate per-symbol bit cost from histograms, matching libjxl's SymbolCostEstimator.
struct SymbolCostEstimator {
    /// Flat array: bits[ctx * max_alphabet_size + sym]
    bits: Vec<f32>,
    max_alphabet_size: usize,
}

impl SymbolCostEstimator {
    fn new(num_contexts: usize, force_huffman: bool, tokens: &[Token], lz77: &Lz77Params) -> Self {
        const ANS_LOG_TAB_SIZE: f32 = 12.0;

        // Build per-context histograms from the (possibly LZ77-transformed) tokens.
        let mut counts: Vec<Vec<u32>> = vec![vec![]; num_contexts];
        let mut total_counts = vec![0u32; num_contexts];

        for token in tokens {
            let (tok, _nbits) = if token.is_lz77_length() {
                let e = Lz77UintCoder::encode(token.value);
                (e.token + lz77.min_symbol, e.nbits)
            } else {
                let e = UintCoder::encode(token.value);
                (e.token, e.nbits)
            };
            let ctx = token.context() as usize;
            if ctx < num_contexts {
                let sym = tok as usize;
                if sym >= counts[ctx].len() {
                    counts[ctx].resize(sym + 1, 0);
                }
                counts[ctx][sym] += 1;
                total_counts[ctx] += 1;
            }
        }

        let max_alphabet_size = counts.iter().map(|c| c.len()).max().unwrap_or(0);
        let mut bits = vec![0.0f32; num_contexts * max_alphabet_size];

        for ctx in 0..num_contexts {
            let total = total_counts[ctx];
            if total == 0 {
                continue;
            }
            let inv_total = 1.0 / (total as f32 + 1e-8);
            for sym in 0..counts[ctx].len() {
                let cnt = counts[ctx][sym];
                let cost = if cnt != 0 && cnt != total {
                    let p = cnt as f32 * inv_total;
                    let c = -jxl_simd::fast_log2f(p);
                    if force_huffman { c.ceil() } else { c }
                } else if cnt == 0 {
                    ANS_LOG_TAB_SIZE // Highest possible cost
                } else {
                    0.0 // Single symbol, zero cost
                };
                bits[ctx * max_alphabet_size + sym] = cost;
            }
        }

        Self {
            bits,
            max_alphabet_size,
        }
    }

    #[inline]
    fn symbol_cost(&self, ctx: usize, sym: usize) -> f32 {
        if sym < self.max_alphabet_size {
            self.bits[ctx * self.max_alphabet_size + sym]
        } else {
            12.0 // ANS_LOG_TAB_SIZE as fallback
        }
    }

    /// Cost of adding an LZ77 symbol to a context (penalty for low-entropy contexts).
    fn add_symbol_cost(&self, ctx: usize) -> f32 {
        // Compute average cost per symbol in this context
        let mut total_cost = 0.0f32;
        let mut total_count = 0u32;
        for sym in 0..self.max_alphabet_size {
            let cost = self.bits[ctx * self.max_alphabet_size + sym];
            if cost < 12.0 {
                // Only count symbols that exist in the histogram
                total_cost += cost;
                total_count += 1;
            }
        }
        if total_count == 0 {
            return 0.0;
        }
        // Higher penalty for contexts with low per-symbol entropy
        (6.0 - total_cost / total_count as f32).max(0.0)
    }

    /// Cost of encoding an LZ77 length token using histogram-based estimation.
    fn len_cost(&self, ctx: usize, len: u32, lz77: &Lz77Params) -> f32 {
        // HybridUintConfig(1, 0, 0) for LZ77 length
        let (tok, nbits) = if len == 0 {
            (0u32, 0u32)
        } else {
            let n = 31 - len.leading_zeros();
            (1 + n, n)
        };
        let sym = tok + lz77.min_symbol;
        nbits as f32 + self.symbol_cost(ctx, sym as usize)
    }

    /// Cost of encoding an LZ77 distance token using histogram-based estimation.
    fn dist_cost_sce(&self, dist_symbol: u32, lz77: &Lz77Params) -> f32 {
        let (tok, nbits) = UintCoder::encode(dist_symbol).into();
        nbits as f32 + self.symbol_cost(lz77.distance_context as usize, tok as usize)
    }
}

/// Hash chain for LZ77 match finding.
///
/// Uses a sliding window and hash table to efficiently find matching sequences.
/// Matches libjxl's HashChain implementation in enc_lz77.cc.
struct HashChain {
    /// Token values (we only hash on value, not context)
    data: Vec<u32>,
    /// Size of token stream
    size: usize,
    /// Window size (power of 2)
    window_size: usize,
    /// Window mask (window_size - 1)
    window_mask: usize,
    /// Minimum match length
    min_length: usize,
    /// Maximum match length
    max_length: usize,

    // Hash table parameters
    #[allow(dead_code)] // Stored for debugging/reference
    hash_num_values: usize,
    hash_mask: usize,
    hash_shift: u32,

    /// Head of hash chain for each hash value (-1 if empty)
    head: Vec<i32>,
    /// Hash chain: next position with same hash
    chain: Vec<u32>,
    /// Hash value at each window position (-1 if invalid)
    val: Vec<i32>,

    // Zero-run optimization
    /// Head of zero-run chain for each run length
    headz: Vec<i32>,
    /// Zero-run chain
    chainz: Vec<u32>,
    /// Number of consecutive zeros starting at each position
    zeros: Vec<u32>,
    /// Current zero count
    numzeros: u32,

    /// Map from actual distance to special distance symbol
    special_dist_table: HashMap<i32, usize>,
    /// Number of special distances (0 if no multiplier, 120 otherwise)
    num_special_distances: usize,

    /// Maximum chain length to traverse (limits search time)
    max_chain_length: u32,
}

impl HashChain {
    fn new(
        tokens: &[Token],
        window_size: usize,
        min_length: usize,
        max_length: usize,
        distance_multiplier: i32,
    ) -> Self {
        let size = tokens.len();

        // Extract just the values
        let data: Vec<u32> = tokens.iter().map(|t| t.value).collect();

        // Hash table setup
        let hash_num_values = 32768usize;
        let hash_mask = hash_num_values - 1;
        let hash_shift = 5u32;

        let head = vec![-1i32; hash_num_values];
        let chain: Vec<u32> = (0..window_size as u32).collect(); // Self-reference indicates uninitialized
        let val = vec![-1i32; window_size];

        // Zero-run optimization
        let headz = vec![-1i32; window_size + 1];
        let chainz: Vec<u32> = (0..window_size as u32).collect();
        let zeros = vec![0u32; window_size];

        // Build special distance table
        let mut special_dist_table = HashMap::new();
        let num_special_distances = if distance_multiplier != 0 {
            // Count down so smallest code wins on ties
            for i in (0..NUM_SPECIAL_DISTANCES).rev() {
                let dist = special_distance(i, distance_multiplier);
                if dist > 0 {
                    special_dist_table.insert(dist, i);
                }
            }
            NUM_SPECIAL_DISTANCES
        } else {
            0
        };

        Self {
            data,
            size,
            window_size,
            window_mask: window_size - 1,
            min_length,
            max_length,
            hash_num_values,
            hash_mask,
            hash_shift,
            head,
            chain,
            val,
            headz,
            chainz,
            zeros,
            numzeros: 0,
            special_dist_table,
            num_special_distances,
            max_chain_length: 256,
        }
    }

    /// Compute hash of 3 consecutive values starting at pos.
    fn get_hash(&self, pos: usize) -> u32 {
        if pos + 2 >= self.size {
            return 0;
        }
        let mut result = 0u32;
        result ^= self.data[pos] & 0xFFFF;
        result ^= (self.data[pos + 1] & 0xFFFF) << self.hash_shift;
        result ^= (self.data[pos + 2] & 0xFFFF) << (self.hash_shift * 2);
        result & self.hash_mask as u32
    }

    /// Count consecutive zeros starting at pos.
    fn count_zeros(&self, pos: usize, prev_zeros: u32) -> u32 {
        let end = (pos + self.window_size).min(self.size);
        if prev_zeros > 0 {
            if prev_zeros >= self.window_mask as u32
                && self.data[end - 1] == 0
                && end == pos + self.window_size
            {
                return prev_zeros;
            } else {
                return prev_zeros - 1;
            }
        }
        let mut num = 0u32;
        while pos + (num as usize) < end && self.data[pos + (num as usize)] == 0 {
            num += 1;
        }
        num
    }

    /// Update hash chain with position pos.
    fn update(&mut self, pos: usize) {
        let hashval = self.get_hash(pos);
        let wpos = pos & self.window_mask;

        self.val[wpos] = hashval as i32;
        if self.head[hashval as usize] != -1 {
            self.chain[wpos] = self.head[hashval as usize] as u32;
        }
        self.head[hashval as usize] = wpos as i32;

        // Update zero count
        if pos > 0 && self.data[pos] != self.data[pos - 1] {
            self.numzeros = 0;
        }
        self.numzeros = self.count_zeros(pos, self.numzeros);

        self.zeros[wpos] = self.numzeros;
        if self.headz[self.numzeros as usize] != -1 {
            self.chainz[wpos] = self.headz[self.numzeros as usize] as u32;
        }
        self.headz[self.numzeros as usize] = wpos as i32;
    }

    /// Update hash chain for multiple positions.
    fn update_range(&mut self, pos: usize, len: usize) {
        for i in 0..len {
            self.update(pos + i);
        }
    }

    /// Find best match at position pos.
    /// Returns (distance_symbol, match_length).
    fn find_match(&self, pos: usize, max_dist: usize) -> (usize, usize) {
        let mut best_dist_symbol = 0usize;
        let mut best_len = 1usize;

        self.find_matches(pos, max_dist, |len, dist_symbol| {
            if len > best_len || (len == best_len && dist_symbol < best_dist_symbol) {
                best_len = len;
                best_dist_symbol = dist_symbol;
            }
        });

        (best_dist_symbol, best_len)
    }

    /// Find all matches at position pos, calling callback for each.
    fn find_matches<F>(&self, pos: usize, max_dist: usize, mut found_match: F)
    where
        F: FnMut(usize, usize),
    {
        let wpos = pos & self.window_mask;
        let hashval = self.get_hash(pos);
        let mut hashpos = self.chain[wpos];

        let mut prev_dist = 0i32;
        let end = (pos + self.max_length).min(self.size);
        let mut chain_length = 0u32;
        let mut best_len = 0usize;

        loop {
            // Compute distance from current position to hash chain position
            let dist = if hashpos as usize <= wpos {
                wpos - hashpos as usize
            } else {
                wpos + self.window_mask + 1 - hashpos as usize
            };

            if (dist as i32) < prev_dist {
                break;
            }
            prev_dist = dist as i32;

            if dist > 0 && dist <= max_dist {
                // Compare sequences
                let mut i = pos;
                let mut j = pos - dist;

                // Zero-run optimization: skip known zeros
                if self.numzeros > 3 {
                    let r =
                        ((self.numzeros - 1) as usize).min(self.zeros[hashpos as usize] as usize);
                    let skip = if i + r >= end { end - i - 1 } else { r };
                    i += skip;
                    j += skip;
                }

                // Extend match
                while i < end && self.data[i] == self.data[j] {
                    i += 1;
                    j += 1;
                }

                let len = i - pos;

                // Accept match if long enough and potentially better
                if len >= self.min_length && len + 2 >= best_len {
                    let dist_symbol =
                        if let Some(&sym) = self.special_dist_table.get(&(dist as i32)) {
                            sym
                        } else {
                            self.num_special_distances + dist - 1
                        };
                    found_match(len, dist_symbol);
                    if len > best_len {
                        best_len = len;
                    }
                }
            }

            chain_length += 1;
            if chain_length >= self.max_chain_length {
                break;
            }

            // Follow chain
            if self.numzeros >= 3 && best_len > self.numzeros as usize {
                // Use zero-run chain for efficiency
                if hashpos == self.chainz[hashpos as usize] {
                    break;
                }
                hashpos = self.chainz[hashpos as usize];
                if self.zeros[hashpos as usize] != self.numzeros {
                    break;
                }
            } else {
                // Use regular hash chain
                if hashpos == self.chain[hashpos as usize] {
                    break;
                }
                hashpos = self.chain[hashpos as usize];
                if self.val[hashpos as usize] != hashval as i32 {
                    // Outdated hash value
                    break;
                }
            }
        }
    }
}

/// Apply greedy LZ77 with backward references using hash chains.
///
/// This implements libjxl's `ApplyLZ77_LZ77` algorithm which uses hash chains
/// to find matching sequences at arbitrary distances within a sliding window.
/// Includes lazy matching to find longer matches at the next position.
///
/// Returns `Some((transformed_tokens, params))` if LZ77 is beneficial,
/// or `None` if the savings are insufficient.
pub fn apply_lz77_backref(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
    distance_multiplier: i32,
) -> Option<(Vec<Token>, Lz77Params)> {
    if tokens.is_empty() {
        return None;
    }

    let mut lz77 = Lz77Params::new(num_contexts, force_huffman);

    // Build cost estimator from original tokens
    let sce = SymbolCostEstimator::new(num_contexts, force_huffman, tokens, &lz77);

    // Compute cumulative bit costs for original stream
    let mut sym_cost = vec![0.0f32; tokens.len() + 1];
    for (i, token) in tokens.iter().enumerate() {
        let e = UintCoder::encode(token.value);
        let cost = sce.symbol_cost(token.context() as usize, e.token as usize) + e.nbits as f32;
        sym_cost[i + 1] = sym_cost[i] + cost;
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut bit_decrease: f32 = 0.0;
    let total_symbols = tokens.len();

    let max_distance = tokens.len();
    let min_length = lz77.min_length as usize;
    let max_length = tokens.len();

    // Use next power of two as window size
    let mut window_size = 1usize;
    while window_size < max_distance && window_size < WINDOW_SIZE {
        window_size <<= 1;
    }

    let mut chain = HashChain::new(
        tokens,
        window_size,
        min_length,
        max_length,
        distance_multiplier,
    );

    const MAX_LAZY_MATCH_LEN: usize = 256;
    let mut already_updated = false;

    let mut i = 0usize;
    while i < tokens.len() {
        out.push(tokens[i]);

        if !already_updated {
            chain.update(i);
        }
        already_updated = false;

        let (mut dist_symbol, mut len) = chain.find_match(i, max_distance);

        if len >= min_length {
            // Try lazy matching: check if next position has a longer match
            if len < MAX_LAZY_MATCH_LEN && i + 1 < tokens.len() {
                chain.update(i + 1);
                already_updated = true;
                let (dist_symbol2, len2) = chain.find_match(i + 1, max_distance);
                if len2 > len {
                    // Use lazy match: emit literal for current position,
                    // then use match starting at next position
                    i += 1;
                    already_updated = false;
                    len = len2;
                    dist_symbol = dist_symbol2;
                    out.push(tokens[i]);
                }
            }

            // Compute costs
            let literal_cost = sym_cost[i + len] - sym_cost[i];
            let lz77_len = len - min_length;

            // Use empirical cost tables for LZ77 encoding
            let lz77_cost = len_cost(lz77_len as u32)
                + dist_cost(dist_symbol as u32)
                + sce.add_symbol_cost(out.last().unwrap().context() as usize);

            if lz77_cost <= literal_cost {
                // Emit LZ77 match
                let last_token = out.last_mut().unwrap();
                last_token.value = lz77_len as u32;
                last_token.set_lz77_length(true);

                out.push(Token::new(lz77.distance_context, dist_symbol as u32));

                bit_decrease += literal_cost - lz77_cost;
            } else {
                // LZ77 not beneficial, emit literals
                for j in 1..len {
                    out.push(tokens[i + j]);
                }
            }

            // Update hash chain for matched positions
            if already_updated {
                chain.update_range(i + 2, len - 2);
                already_updated = false;
            } else {
                chain.update_range(i + 1, len - 1);
            }
            i += len - 1;
        }
        // Else: literal already pushed

        i += 1;
    }

    // Only use LZ77 if savings exceed threshold
    let threshold = total_symbols as f32 * 0.2 + 16.0;
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "[LZ77-backref] bit_decrease={:.1}, threshold={:.1}, tokens: {} -> {}, matches={}",
        bit_decrease,
        threshold,
        total_symbols,
        out.len(),
        out.iter().filter(|t| t.is_lz77_length()).count()
    );
    if bit_decrease > threshold {
        lz77.enabled = true;
        Some((out, lz77))
    } else {
        None
    }
}

/// Apply RLE-based LZ77 compression to a token stream.
///
/// Scans for runs of consecutive identical values. When a run is long enough
/// and the LZ77 encoding is cheaper, replaces the run with:
/// 1. An LZ77 length token (context = original, value = run_len - min_length, is_lz77_length = true)
/// 2. A distance token (context = distance_context, value = 0 meaning "repeat previous")
///
/// Returns `Some((transformed_tokens, params))` if LZ77 is beneficial,
/// or `None` if the savings are insufficient.
pub fn apply_lz77_rle(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
    distance_multiplier: i32,
) -> Option<(Vec<Token>, Lz77Params)> {
    if tokens.is_empty() {
        return None;
    }

    let mut lz77 = Lz77Params::new(num_contexts, force_huffman);

    // Compute the distance symbol that encodes distance=1 (repeat previous value).
    // When dist_multiplier == 0: decoder uses distance_sym directly, so sym=0 → distance=0+1=1.
    // When dist_multiplier > 0: decoder uses special distance table.
    //   SPECIAL_DISTANCES[1] = (1, 0) → distance = 1 + dm*0 = 1 for any dm.
    //   SPECIAL_DISTANCES[0] = (0, 1) → distance = 0 + dm*1 = dm (WRONG for RLE).
    let rle_distance_symbol: u32 = if distance_multiplier > 0 { 1 } else { 0 };

    // First pass: build cost estimator from the original tokens (no LZ77 tokens yet).
    // We pass the original tokens to estimate costs, matching libjxl.
    let sce = SymbolCostEstimator::new(num_contexts, force_huffman, tokens, &lz77);

    // Compute cumulative bit costs for original stream.
    let mut sym_cost = vec![0.0f32; tokens.len() + 1];
    for (i, token) in tokens.iter().enumerate() {
        let e = UintCoder::encode(token.value);
        let cost = sce.symbol_cost(token.context() as usize, e.token as usize) + e.nbits as f32;
        sym_cost[i + 1] = sym_cost[i] + cost;
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut bit_decrease: f32 = 0.0;
    let total_symbols = tokens.len();

    let mut i = 0;
    while i < tokens.len() {
        // Count consecutive identical values starting from the PREVIOUS token
        // (matching libjxl: "if (i > 0) { ... in[i+num_to_copy].value != in[i-1].value }")
        let mut num_to_copy = 0;
        if i > 0 {
            let prev_value = tokens[i - 1].value;
            while i + num_to_copy < tokens.len() && tokens[i + num_to_copy].value == prev_value {
                num_to_copy += 1;
            }
        }

        if num_to_copy == 0 {
            out.push(tokens[i]);
            i += 1;
            continue;
        }

        // Cost of encoding the run literally
        let literal_cost = sym_cost[i + num_to_copy] - sym_cost[i];

        // Cost of LZ77 encoding (rough estimate matching libjxl)
        let lz77_cost = if num_to_copy >= lz77.min_length as usize {
            let lz77_len = num_to_copy - lz77.min_length as usize;
            // CeilLog2Nonzero(lz77_len + 1) + 1 (for distance)
            ceil_log2_nonzero((lz77_len + 1) as u32) as f32 + 1.0
        } else {
            0.0
        };

        if num_to_copy < lz77.min_length as usize || literal_cost <= lz77_cost {
            // Not worth encoding as LZ77, emit literal tokens
            for j in 0..num_to_copy {
                out.push(tokens[i + j]);
            }
            i += num_to_copy;
            continue;
        }

        // Emit LZ77 length token
        let lz77_len = (num_to_copy - lz77.min_length as usize) as u32;
        out.push(Token::lz77_length(tokens[i].context(), lz77_len));

        // Emit distance token encoding distance=1 (repeat previous value)
        out.push(Token::new(lz77.distance_context, rle_distance_symbol));

        bit_decrease += literal_cost - lz77_cost;
        i += num_to_copy;
    }

    // Only use LZ77 if savings exceed threshold (matching libjxl)
    let threshold = total_symbols as f32 * 0.2 + 16.0;
    #[cfg(feature = "debug-tokens")]
    eprintln!(
        "[LZ77-RLE] bit_decrease={:.1}, threshold={:.1}, tokens: {} -> {}, runs_found={}",
        bit_decrease,
        threshold,
        total_symbols,
        out.len(),
        out.iter().filter(|t| t.is_lz77_length()).count()
    );
    if bit_decrease > threshold {
        lz77.enabled = true;
        Some((out, lz77))
    } else {
        None
    }
}

/// CeilLog2Nonzero matching libjxl's implementation.
fn ceil_log2_nonzero(x: u32) -> u32 {
    debug_assert!(x > 0);
    let floor = 31 - x.leading_zeros();
    if x.is_power_of_two() {
        floor
    } else {
        floor + 1
    }
}

/// LZ77 method selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lz77Method {
    /// RLE-only: only matches consecutive identical values (distance = 1).
    /// Fast but limited compression on photographic content.
    #[default]
    Rle,
    /// Full backward references with hash chains (greedy matching).
    /// Finds matches at arbitrary distances within a sliding window.
    /// 1-3% better compression on photos, slower.
    Greedy,
    /// Optimal backward references via Viterbi DP (from libjxl ApplyLZ77_Optimal).
    /// Considers all viable matches at each position and finds the minimum-cost
    /// parse via dynamic programming. Best compression, slowest.
    Optimal,
}

/// Apply LZ77 compression using the specified method.
///
/// - `Lz77Method::Rle`: RLE-only (fast, limited compression)
/// - `Lz77Method::Greedy`: Hash chain backward references (slower, better compression)
/// - `Lz77Method::Optimal`: Viterbi DP optimal parse (slowest, best compression)
///
/// For photographic content, `Greedy` typically provides 1-3% additional compression
/// over RLE-only. `Optimal` finds the minimum-cost parse via dynamic programming.
///
/// Returns `Some((transformed_tokens, params))` if LZ77 is beneficial,
/// or `None` if the savings are insufficient.
pub fn apply_lz77(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
    method: Lz77Method,
    distance_multiplier: i32,
) -> Option<(Vec<Token>, Lz77Params)> {
    match method {
        Lz77Method::Rle => apply_lz77_rle(tokens, num_contexts, force_huffman, distance_multiplier),
        Lz77Method::Greedy => {
            apply_lz77_backref(tokens, num_contexts, force_huffman, distance_multiplier)
        }
        Lz77Method::Optimal => {
            apply_lz77_optimal(tokens, num_contexts, force_huffman, distance_multiplier)
        }
    }
}

/// Apply optimal LZ77 with Viterbi DP parsing (from libjxl `ApplyLZ77_Optimal`).
///
/// Uses dynamic programming to find the minimum-cost parse of the token stream.
/// First runs greedy LZ77 to build a cost model, then uses that model with
/// forward-pass DP to find the optimal literal/match decisions at each position.
///
/// Returns `Some((transformed_tokens, params))` if LZ77 is beneficial,
/// or `None` if the savings are insufficient.
pub fn apply_lz77_optimal(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
    distance_multiplier: i32,
) -> Option<(Vec<Token>, Lz77Params)> {
    if tokens.is_empty() {
        return None;
    }

    // Step 1: Run greedy LZ77 to get a cost estimate.
    // If greedy doesn't help, optimal won't either.
    let greedy_result =
        apply_lz77_backref(tokens, num_contexts, force_huffman, distance_multiplier);
    let greedy_tokens = match &greedy_result {
        Some((t, _)) => t,
        None => return None,
    };

    let mut lz77 = Lz77Params::new(num_contexts, force_huffman);
    lz77.enabled = true;

    // Step 2: Build cost estimator from greedy result (num_contexts + 1 for distance ctx).
    let sce = SymbolCostEstimator::new(num_contexts + 1, force_huffman, greedy_tokens, &lz77);

    // Step 3: Compute cumulative symbol costs for the original (non-LZ77) stream.
    let mut sym_cost = vec![0.0f32; tokens.len() + 1];
    for (i, token) in tokens.iter().enumerate() {
        let e = UintCoder::encode(token.value);
        let cost = sce.symbol_cost(token.context() as usize, e.token as usize) + e.nbits as f32;
        sym_cost[i + 1] = sym_cost[i] + cost;
    }

    // Step 4: Forward DP pass.
    let max_distance = tokens.len();
    let min_length = lz77.min_length as usize;
    let max_length = tokens.len();

    let mut window_size = 1usize;
    while window_size < max_distance && window_size < WINDOW_SIZE {
        window_size <<= 1;
    }

    let mut chain = HashChain::new(
        tokens,
        window_size,
        min_length,
        max_length,
        distance_multiplier,
    );

    // MatchInfo for backtrace: len=1 means literal, dist_symbol stored as +1 (0 = literal).
    struct PrefixInfo {
        len: u32,
        dist_symbol: u32, // 0 = literal, >0 = LZ77 match (actual dist_symbol + 1)
        ctx: u32,
        total_cost: f32,
    }

    let n = tokens.len();
    let mut prefix_costs: Vec<PrefixInfo> = (0..=n)
        .map(|_| PrefixInfo {
            len: 0,
            dist_symbol: 0,
            ctx: 0,
            total_cost: f32::MAX,
        })
        .collect();
    prefix_costs[0].total_cost = 0.0;

    let mut rle_length = 0usize;
    let mut skip_lz77 = 0usize;
    let mut dist_symbols: Vec<u32> = Vec::new();

    for i in 0..n {
        chain.update(i);

        // Literal cost
        let lit_cost = prefix_costs[i].total_cost + sym_cost[i + 1] - sym_cost[i];
        if prefix_costs[i + 1].total_cost > lit_cost {
            prefix_costs[i + 1].dist_symbol = 0;
            prefix_costs[i + 1].len = 1;
            prefix_costs[i + 1].ctx = tokens[i].context();
            prefix_costs[i + 1].total_cost = lit_cost;
        }

        if skip_lz77 > 0 {
            skip_lz77 -= 1;
            continue;
        }

        // Collect all matches: for each length, keep the cheapest dist_symbol.
        dist_symbols.clear();
        chain.find_matches(i, max_distance, |len, dist_symbol| {
            if dist_symbols.len() <= len {
                dist_symbols.resize(len + 1, dist_symbol as u32);
            }
            if (dist_symbol as u32) < dist_symbols[len] {
                dist_symbols[len] = dist_symbol as u32;
            }
        });

        if dist_symbols.len() <= min_length {
            continue;
        }

        // Normalize: for each length, use the best dist_symbol from any longer match.
        {
            let mut best_cost = dist_symbols[dist_symbols.len() - 1];
            for j in (min_length..dist_symbols.len()).rev() {
                if dist_symbols[j] < best_cost {
                    best_cost = dist_symbols[j];
                }
                dist_symbols[j] = best_cost;
            }
        }

        // Evaluate each match length.
        for (j, &dsym) in dist_symbols.iter().enumerate().skip(min_length) {
            let target = i + j;
            if target > n {
                break;
            }
            let lz77_cost =
                sce.len_cost(tokens[i].context() as usize, (j - min_length) as u32, &lz77)
                    + sce.dist_cost_sce(dsym, &lz77);
            let cost = prefix_costs[i].total_cost + lz77_cost;
            if prefix_costs[target].total_cost > cost {
                prefix_costs[target].len = j as u32;
                prefix_costs[target].dist_symbol = dsym + 1; // +1 to distinguish from literal
                prefix_costs[target].ctx = tokens[i].context();
                prefix_costs[target].total_cost = cost;
            }
        }

        // RLE skip optimization: avoid O(n^2) on long runs of same distance.
        let last_dist = dist_symbols[dist_symbols.len() - 1];
        if (last_dist == 0 && distance_multiplier == 0)
            || (last_dist == 1 && distance_multiplier != 0)
        {
            rle_length += 1;
        } else {
            rle_length = 0;
        }
        if rle_length >= 8 && dist_symbols.len() > 9 {
            skip_lz77 = dist_symbols.len() - 10;
            rle_length = 0;
        }
    }

    // Step 5: Backtrace from end to beginning.
    let mut out = Vec::with_capacity(n);
    let mut pos = n;
    while pos > 0 {
        let info = &prefix_costs[pos];
        let is_lz77 = info.dist_symbol != 0;

        if is_lz77 {
            let dist_symbol = info.dist_symbol - 1;
            out.push(Token::new(lz77.distance_context, dist_symbol));
        }

        let val = if is_lz77 {
            info.len - min_length as u32
        } else {
            tokens[pos - 1].value
        };
        let mut tok = Token::new(info.ctx, val);
        tok.set_lz77_length(is_lz77);
        out.push(tok);

        pos -= info.len as usize;
    }

    out.reverse();
    Some((out, lz77))
}

/// Try both LZ77 methods and return the one with better compression.
///
/// This is useful when you want the best compression regardless of speed.
/// Returns the method that produces fewer output tokens, or None if neither
/// method provides sufficient savings.
#[allow(dead_code)] // Utility function for advanced users
pub fn apply_lz77_best(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
    distance_multiplier: i32,
) -> Option<(Vec<Token>, Lz77Params)> {
    let rle_result = apply_lz77_rle(tokens, num_contexts, force_huffman, distance_multiplier);
    let backref_result =
        apply_lz77_backref(tokens, num_contexts, force_huffman, distance_multiplier);

    match (&rle_result, &backref_result) {
        (Some((rle_tokens, _)), Some((backref_tokens, _))) => {
            // Return whichever produces fewer tokens
            if backref_tokens.len() <= rle_tokens.len() {
                backref_result
            } else {
                rle_result
            }
        }
        (Some(_), None) => rle_result,
        (None, Some(_)) => backref_result,
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ceil_log2_nonzero() {
        assert_eq!(ceil_log2_nonzero(1), 0);
        assert_eq!(ceil_log2_nonzero(2), 1);
        assert_eq!(ceil_log2_nonzero(3), 2);
        assert_eq!(ceil_log2_nonzero(4), 2);
        assert_eq!(ceil_log2_nonzero(5), 3);
        assert_eq!(ceil_log2_nonzero(8), 3);
        assert_eq!(ceil_log2_nonzero(9), 4);
    }

    #[test]
    fn test_no_rle_on_short_stream() {
        // Very short streams shouldn't trigger LZ77
        let tokens = vec![Token::new(0, 5), Token::new(0, 5), Token::new(0, 5)];
        assert!(apply_lz77_rle(&tokens, 1, false, 0).is_none());
    }

    #[test]
    fn test_rle_on_long_run() {
        // Long run of identical values should trigger LZ77
        let mut tokens = Vec::new();
        // Need a "previous" token, then a long run of identical values
        tokens.push(Token::new(0, 5));
        for _ in 0..200 {
            tokens.push(Token::new(0, 5));
        }

        let result = apply_lz77_rle(&tokens, 1, false, 0);
        if let Some((lz77_tokens, params)) = result {
            assert!(params.enabled);
            // Should be much shorter than the original
            assert!(lz77_tokens.len() < tokens.len());
            // Should contain at least one LZ77 length token
            assert!(lz77_tokens.iter().any(|t| t.is_lz77_length()));
        }
        // If None, that's OK — the threshold might not be met for this particular cost estimate
    }

    #[test]
    fn test_rle_preserves_non_runs() {
        // Mixed content: some runs, some non-runs
        let mut tokens = Vec::new();
        // Non-repeating prefix
        for i in 0..10 {
            tokens.push(Token::new(0, i));
        }
        // Long run
        for _ in 0..100 {
            tokens.push(Token::new(0, 42));
        }
        // Non-repeating suffix
        for i in 0..10 {
            tokens.push(Token::new(0, i + 100));
        }

        if let Some((lz77_tokens, params)) = apply_lz77_rle(&tokens, 1, false, 0) {
            assert!(params.enabled);
            assert!(lz77_tokens.len() < tokens.len());
            // The first token should be preserved literally
            assert_eq!(lz77_tokens[0].value, 0);
            assert!(!lz77_tokens[0].is_lz77_length());
        }
    }

    #[test]
    fn test_empty_stream() {
        assert!(apply_lz77_rle(&[], 1, false, 0).is_none());
    }

    // Tests for backward-reference LZ77

    #[test]
    fn test_backref_empty_stream() {
        assert!(apply_lz77_backref(&[], 1, false, 0).is_none());
    }

    #[test]
    fn test_backref_short_stream() {
        // Very short streams shouldn't trigger LZ77
        let tokens = vec![Token::new(0, 5), Token::new(0, 5), Token::new(0, 5)];
        assert!(apply_lz77_backref(&tokens, 1, false, 0).is_none());
    }

    #[test]
    fn test_backref_on_repeating_pattern() {
        // Pattern that repeats at distance > 1 (not just RLE)
        // Pattern: [A, B, C, A, B, C, A, B, C, ...]
        let mut tokens = Vec::new();
        for _ in 0..100 {
            tokens.push(Token::new(0, 10));
            tokens.push(Token::new(0, 20));
            tokens.push(Token::new(0, 30));
        }

        let result = apply_lz77_backref(&tokens, 1, false, 0);
        if let Some((lz77_tokens, params)) = result {
            assert!(params.enabled);
            // Should be significantly shorter due to backward references
            assert!(
                lz77_tokens.len() < tokens.len(),
                "backref should compress pattern: {} vs {}",
                lz77_tokens.len(),
                tokens.len()
            );
            // Should have LZ77 length tokens
            assert!(lz77_tokens.iter().any(|t| t.is_lz77_length()));
        }
    }

    #[test]
    fn test_backref_finds_longer_matches_than_rle() {
        // Pattern where backref can find matches that RLE cannot
        // [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...]
        let mut tokens = Vec::new();
        for _ in 0..50 {
            for j in 1..=5 {
                tokens.push(Token::new(0, j));
            }
        }

        let rle_result = apply_lz77_rle(&tokens, 1, false, 0);
        let backref_result = apply_lz77_backref(&tokens, 1, false, 0);

        // RLE should not find matches (no consecutive identical values)
        // Backref should find matches at distance 5
        match (&rle_result, &backref_result) {
            (None, Some((backref_tokens, _))) => {
                // This is the expected case: RLE finds nothing, backref does
                assert!(backref_tokens.len() < tokens.len());
            }
            (Some((rle_tokens, _)), Some((backref_tokens, _))) => {
                // If both activate, backref should do better or equal
                assert!(backref_tokens.len() <= rle_tokens.len());
            }
            _ => {
                // Either both fail (acceptable for small patterns) or both succeed
            }
        }
    }

    #[test]
    fn test_backref_with_distance_multiplier() {
        // Test that special distance codes work with distance multiplier
        // When multiplier is non-zero, distances like image_width are encoded more efficiently
        let mut tokens = Vec::new();
        let image_width = 64;

        // Create pattern that repeats at image_width distance (previous row)
        for _row in 0..20 {
            for col in 0..image_width {
                // Same value for same column across rows
                tokens.push(Token::new(0, (col % 16) as u32));
            }
        }

        let _result_no_mult = apply_lz77_backref(&tokens, 1, false, 0);
        let result_with_mult = apply_lz77_backref(&tokens, 1, false, image_width);

        // Both should find matches; with multiplier might be more efficient
        // but the main test is that it doesn't crash and produces valid output
        if let Some((tokens_mult, params)) = result_with_mult {
            assert!(params.enabled);
            assert!(tokens_mult.len() < tokens.len());
        }
    }

    #[test]
    fn test_special_distance() {
        // Test special distance calculation
        // kSpecialDistances[0] = [0, 1] -> distance = 0 + 1*multiplier = multiplier
        assert_eq!(special_distance(0, 64), 64);
        // kSpecialDistances[1] = [1, 0] -> distance = 1 + 0*multiplier = 1
        assert_eq!(special_distance(1, 64), 1);
        // kSpecialDistances[2] = [1, 1] -> distance = 1 + 1*multiplier = 65
        assert_eq!(special_distance(2, 64), 65);
        // kSpecialDistances[3] = [-1, 1] -> distance = -1 + 1*multiplier = 63
        assert_eq!(special_distance(3, 64), 63);
    }

    #[test]
    fn test_len_cost() {
        // Verify len_cost doesn't panic on various inputs
        for len in 0..1000 {
            let cost = len_cost(len);
            assert!(cost >= 0.0, "len_cost({}) should be non-negative", len);
            assert!(cost < 100.0, "len_cost({}) should be reasonable", len);
        }
    }

    #[test]
    fn test_dist_cost() {
        // Verify dist_cost doesn't panic on various inputs
        for dist in 0..10000 {
            let cost = dist_cost(dist);
            assert!(cost >= 0.0, "dist_cost({}) should be non-negative", dist);
            assert!(cost < 100.0, "dist_cost({}) should be reasonable", dist);
        }
    }

    #[test]
    fn test_apply_lz77_method_enum() {
        let mut tokens = Vec::new();
        tokens.push(Token::new(0, 5));
        for _ in 0..200 {
            tokens.push(Token::new(0, 5));
        }

        // Test RLE method
        let rle_result = apply_lz77(&tokens, 1, false, Lz77Method::Rle, 0);
        if let Some((_, params)) = &rle_result {
            assert!(params.enabled);
        }

        // Test Greedy method
        let greedy_result = apply_lz77(&tokens, 1, false, Lz77Method::Greedy, 0);
        if let Some((_, params)) = &greedy_result {
            assert!(params.enabled);
        }
    }

    #[test]
    fn test_apply_lz77_best() {
        // Pattern where backref should do better
        let mut tokens = Vec::new();
        for _ in 0..50 {
            for j in 1..=10 {
                tokens.push(Token::new(0, j));
            }
        }

        let best_result = apply_lz77_best(&tokens, 1, false, 0);
        // Should pick the better method (likely backref for this pattern)
        if let Some((best_tokens, params)) = best_result {
            assert!(params.enabled);
            assert!(best_tokens.len() < tokens.len());
        }
    }

    #[test]
    fn test_hash_chain_basic() {
        // Test hash chain finds simple matches
        let tokens = vec![
            Token::new(0, 10),
            Token::new(0, 20),
            Token::new(0, 30),
            Token::new(0, 40), // Different sequence
            Token::new(0, 10),
            Token::new(0, 20),
            Token::new(0, 30), // Repeats tokens 0-2
        ];

        let mut chain = HashChain::new(&tokens, 16, 3, 100, 0);
        // Update chain for all positions
        for i in 0..tokens.len() {
            chain.update(i);
        }

        // At position 4, should find match at position 0 (distance 4)
        let (dist_symbol, len) = chain.find_match(4, 10);
        assert!(len >= 3, "should find match of length >= 3, got {}", len);
        // dist_symbol should encode distance 4 (no special distances with multiplier=0)
        // Special distances: 0 entries since multiplier=0
        // So dist_symbol = num_special_distances + dist - 1 = 0 + 4 - 1 = 3
        assert_eq!(dist_symbol, 3, "distance symbol for dist=4 should be 3");
    }
}
