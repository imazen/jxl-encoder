// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Tree histogram writing and tree serialization for modular encoding.
//!
//! Contains tree histogram construction (for zero, gradient, and weighted predictors),
//! tree token writing, weighted predictor header, and general MA tree serialization.

use super::encode_primitives::{pack_signed, write_integer_config, write_varlen_u16};
use crate::bit_writer::BitWriter;
use crate::entropy_coding::huffman_tree::{
    build_and_store_huffman_tree, convert_bit_depths_to_symbols, create_huffman_tree,
};
use crate::error::Result;

/// Write a complete single-leaf Zero predictor tree (histogram + tokens).
/// All tokens are 0, so alphabet size = 1 and no Huffman coding is needed.
pub(super) fn write_zero_tree_complete(writer: &mut BitWriter) -> Result<()> {
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: Starting tree histogram (Zero)",
        writer.bits_written()
    );

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // For raw symbol encoding with only symbol 0, use split_exponent = 15.
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;

    // alphabet_size - 1 = 0 using VarLenUint16 encoding
    write_varlen_u16(writer, 0)?;

    // No Huffman table needed for alphabet_size = 1
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: After tree histogram (Zero)",
        writer.bits_written()
    );

    Ok(())
}

/// Write a tree histogram that can encode tokens 0-5 (for Gradient predictor).
/// Returns (depths, codes) for use in encoding tree tokens.
pub(crate) fn write_tree_histogram_for_gradient(
    writer: &mut BitWriter,
) -> Result<(Vec<u8>, Vec<u16>)> {
    // NOTE: This is for LfGlobal trees which use allow_lz77=true in the decoder
    // Tree tokens are raw symbols (0-5), not hybrid uints.
    // Use split_exponent = log_alphabet_size for raw symbol encoding.
    write_tree_histogram_for_predictor_impl(writer, true, 5)
}

/// Write a tree histogram for a single-leaf tree using the given fixed
/// `predictor_id` (libjxl `cjxl -P N` / `--modular_predictor`).
///
/// `predictor_id` must be in `0..=13` (see [`super::predictor::Predictor`]).
/// Returns `(depths, codes)` for encoding the matching tree tokens via
/// [`write_predictor_tree_tokens`].
///
/// This is the generic counterpart of [`write_tree_histogram_for_gradient`]
/// (`predictor_id == 5`) and [`write_tree_histogram_for_zero`]
/// (`predictor_id == 0`) — those wrappers stay for byte-identical
/// hash-lock parity at the default settings.
pub(crate) fn write_tree_histogram_for_predictor(
    writer: &mut BitWriter,
    predictor_id: u8,
) -> Result<(Vec<u8>, Vec<u16>)> {
    write_tree_histogram_for_predictor_impl(writer, true, predictor_id as u32)
}

/// Write a tree histogram for a single-leaf Gradient-predictor tree that
/// also carries a non-unit multiplier `(mul_log, mul_bits)` such that
/// the decoder reconstructs `pixel = val * multiplier + prediction`
/// with `multiplier = (mul_bits + 1) << mul_log`.
///
/// This is the lossy alpha-extras path (libjxl `QuantizeChannel` +
/// `ModularMultiplierInfo`). Histogram alphabet is widened to cover the
/// extra symbols produced by `mul_log` / `mul_bits` so the Huffman codes
/// emitted here match the tokens written by
/// [`write_gradient_tree_tokens_lossy`].
pub(crate) fn write_tree_histogram_for_gradient_lossy(
    writer: &mut BitWriter,
    mul_log: u32,
    mul_bits: u32,
) -> Result<(Vec<u8>, Vec<u16>)> {
    // Tree tokens are [property=0, predictor=5, offset=0, mul_log, mul_bits].
    // Counts per symbol: symbol 0 appears once for property and once for
    // offset (=2), symbol 5 appears once for predictor, plus one each for
    // mul_log and mul_bits (which may coincide with each other or with
    // symbol 0 / 5). At extreme alpha distances `mul_bits` can be much
    // larger than 5 (e.g. q=39 → mul_log=0, mul_bits=38), so size the
    // histogram by max(mul_log, mul_bits, 5).
    let max_symbol = mul_log.max(mul_bits).max(5) as u16;
    let used = (max_symbol as usize) + 1;
    let mut tree_histogram = vec![0u32; used];
    tree_histogram[0] += 2; // property + offset
    tree_histogram[5] += 1; // predictor = 5 (Gradient)
    tree_histogram[mul_log as usize] += 1;
    tree_histogram[mul_bits as usize] += 1;
    let tree_histogram = &tree_histogram[..];

    // lz77.enabled = 0
    writer.write(1, 0)?;
    // Context map: is_simple=1, bits_per_entry=0
    writer.write(1, 1)?;
    writer.write(2, 0)?;
    // use_prefix_code = 1
    writer.write(1, 1)?;

    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;
    write_varlen_u16(writer, max_symbol)?;

    let (depths, codes) = if used > 1 {
        let table = build_and_store_huffman_tree(tree_histogram, writer)?;
        (table.depths, table.codes)
    } else {
        (vec![0u8; used], vec![0u16; used])
    };

    Ok((depths, codes))
}

/// Write a single-leaf Gradient-predictor tree's tokens with a
/// non-unit multiplier. Pairs with
/// [`write_tree_histogram_for_gradient_lossy`].
pub(crate) fn write_gradient_tree_tokens_lossy(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
    mul_log: u32,
    mul_bits: u32,
) -> Result<()> {
    // Tokens: property=0, predictor=5, offset=0, mul_log, mul_bits
    let tokens = [0u32, 5, 0, mul_log, mul_bits];
    for &token in &tokens {
        let depth = depths.get(token as usize).copied().unwrap_or(0);
        let code = codes.get(token as usize).copied().unwrap_or(0);
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }
    Ok(())
}

/// Write a multi-leaf Gradient-predictor tree that routes each modular
/// channel to its own leaf, with per-leaf `(mul_log, mul_bits)`
/// multipliers. Pairs with the same `pack_signed`-residual emission as
/// the single-leaf paths but lets each channel carry an independent
/// pixel quantizer `q` (libjxl `enc_modular.cc:973-1027` per-extra
/// `quants_[ch_id]`; `dec_ma.cc:DecodeTree` parses the BFS-encoded
/// tree).
///
/// The tree splits on property 0 (`static_props[0] = chan` — the
/// modular-subimage channel index, `encoding.cc:160`). Layout for
/// `quantizers.len() == N`:
///
/// - Decision node 0: `property=0, splitval=0` → LEFT (`chan > 0`)
///   continues into the subtree for channels 1..N-1; RIGHT (`chan <= 0`)
///   is the leaf for channel 0.
/// - Decision node 1: `property=0, splitval=1` → LEFT (`chan > 1`)
///   continues into the subtree for channels 2..N-1; RIGHT is the leaf
///   for channel 1.
/// - ... and so on until the last leaf for channel N-1 (no decision).
///
/// This left-spine shape uses `2*N - 1` total nodes (N-1 decisions
/// plus N leaves) and is decoded in libjxl's BFS order
/// (`dec_ma.cc:75-125`): each internal node pushes its two children
/// onto the to-decode queue, so we write
/// `[D0, L0, D1, L1, ..., D_{N-2}, L_{N-2}, L_{N-1}]`.
///
/// Histogram alphabet covers all tokens emitted by the tree: per
/// decision `prop1 = property + 1` (`prop1 = 1` for property 0) and
/// `splitval` packed-signed; per leaf `predictor = 5`, `offset = 0`,
/// `mul_log`, `mul_bits`. Returns `(depths, codes)` for the caller to
/// write the matching tokens via [`write_channel_split_tree_tokens`].
///
/// Requires `quantizers.len() >= 2` (single-channel callers use the
/// existing [`write_tree_histogram_for_gradient_lossy`] / lossless
/// paths). `quantizers[i] == 0` is treated as `1` (lossless leaf).
pub(crate) fn write_tree_histogram_for_channel_split_lossy(
    writer: &mut BitWriter,
    quantizers: &[u32],
) -> Result<(Vec<u8>, Vec<u16>)> {
    debug_assert!(
        quantizers.len() >= 2,
        "channel-split tree requires at least 2 leaves"
    );

    // Tokens emitted by [`write_channel_split_tree_tokens`]:
    //
    // - For each of N-1 decisions: `prop1 = 1` (property 0) and
    //   `splitval` packed-signed. `splitval = 0` → packed = 0;
    //   `splitval = k > 0` → packed = 2*k - 1 (odd). All decisions
    //   use `prop1 = 1`.
    // - For each of N leaves: `predictor = 5`, `offset = 0`,
    //   `mul_log`, `mul_bits`.
    //
    // Count symbol frequencies so the Huffman code matches the
    // tokens we emit. Symbol values come from the tree contexts the
    // decoder reads (`dec_ma.cc:89-118`) — for raw symbol encoding
    // the value IS the histogram index.
    let n = quantizers.len();
    let mut max_symbol: u32 = 0;
    let mut count = |s: u32| {
        if s > max_symbol {
            max_symbol = s;
        }
    };
    // Decisions: N-1 entries. Each emits prop1=1 and splitval=k
    // packed-signed (k >= 0 → packed = 2k).
    count(1); // prop1 (first decision)
    count(0); // splitval = 0 packed (= 2*0)
    for k in 1..(n - 1) {
        count(1); // prop1 for every decision
        count((2 * k) as u32); // splitval = k packed (= 2k)
    }
    // Leaves: N entries. Each emits prop1=0, predictor=5, offset=0
    // (packed_signed=0), mul_log, mul_bits.
    for &q in quantizers.iter() {
        let q = q.max(1);
        let (mul_log, mul_bits) = decompose_multiplier_pub(q);
        count(0); // prop1 = 0 (leaf marker)
        count(5); // predictor = 5 (Gradient)
        count(0); // offset = 0 (packed)
        count(mul_log);
        count(mul_bits);
    }

    // Rebuild the histogram now that we know the alphabet size. Use
    // the same counting helper so a refactor that adds a token also
    // bumps the histogram.
    let used = (max_symbol as usize) + 1;
    let mut tree_histogram = vec![0u32; used];
    let mut bump = |s: u32| {
        tree_histogram[s as usize] += 1;
    };
    bump(1);
    bump(0);
    for k in 1..(n - 1) {
        bump(1);
        bump((2 * k) as u32);
    }
    for &q in quantizers.iter() {
        let q = q.max(1);
        let (mul_log, mul_bits) = decompose_multiplier_pub(q);
        bump(0);
        bump(5);
        bump(0);
        bump(mul_log);
        bump(mul_bits);
    }

    // Sub-bitstream tree header (matches the single-leaf paths above).
    writer.write(1, 0)?; // lz77.enabled = 0
    writer.write(1, 1)?; // context_map is_simple = 1
    writer.write(2, 0)?; // context_map bits_per_entry = 0 (one shared histogram)
    writer.write(1, 1)?; // use_prefix_code = 1

    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;
    write_varlen_u16(writer, max_symbol as u16)?;

    let (depths, codes) = if used > 1 {
        let table = build_and_store_huffman_tree(&tree_histogram, writer)?;
        (table.depths, table.codes)
    } else {
        (vec![0u8; used], vec![0u16; used])
    };
    Ok((depths, codes))
}

/// Write the tokens for a multi-leaf channel-split Gradient tree.
/// Pairs with [`write_tree_histogram_for_channel_split_lossy`] — the
/// caller passes the same `quantizers` slice and the `(depths, codes)`
/// returned by the histogram writer.
///
/// Token emission follows libjxl's BFS tree-decode order
/// (`dec_ma.cc:75-125`): each internal node pushes its two children
/// onto the to-decode queue, so we write `[D0, L0, D1, L1, ...,
/// D_{N-2}, L_{N-2}, L_{N-1}]`.
pub(crate) fn write_channel_split_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
    quantizers: &[u32],
) -> Result<()> {
    debug_assert!(quantizers.len() >= 2);
    let n = quantizers.len();

    // Helper to write one raw symbol via the prefix-code table.
    let write_sym = |writer: &mut BitWriter, sym: u32| -> Result<()> {
        let d = depths.get(sym as usize).copied().unwrap_or(0);
        let c = codes.get(sym as usize).copied().unwrap_or(0);
        if d > 0 {
            writer.write(d as usize, c as u64)?;
        }
        Ok(())
    };

    // Decoder reads one node, then queues 2 children for internals.
    // BFS order with left-spine internals: [D0, L0, D1, L1, ..., L_{N-1}].
    // For each k in 0..N-1: decision Dk (property=0, splitval=k),
    // followed by leaf L_k (for channel k). After the last decision
    // D_{N-2}, the LEFT-child slot is filled by the final leaf
    // L_{N-1}.
    //
    // BUT — internal nodes' children populate in queue order, which
    // for our left-spine means D_k's RIGHT child is L_k and LEFT
    // child is D_{k+1} (or L_{N-1} for the final decision). Decoder
    // emplaces children at indices `tree.size() + to_decode + 1`
    // (left) and `+ 2` (right). The decoder's BFS reads in queued
    // order, so we emit in queued order too: emit each node, queue
    // its children, repeat. Left-spine means left-child = next
    // decision (or final leaf), right-child = current channel leaf.
    //
    // Order: D0, D1, D2, ..., D_{N-2}, L_{N-1}, L_{N-2}, ..., L_1, L_0
    // is WRONG because the decoder pops in FIFO order and our left
    // children are decisions. Instead the to-decode queue evolves:
    //   start: [root]            pop root=D0, push [L_left_of_D0, L_right_of_D0]
    //   queue: [D1, L_0]         pop D1, push  [L_left_of_D1, L_right_of_D1]
    //   queue: [L_0, D2, L_1]    pop L_0 (leaf, no children)
    //   queue: [D2, L_1]         pop D2 ...
    //   ...
    // Output order: D0, D1, L_0, D2, L_1, D3, L_2, ..., D_{N-2},
    // L_{N-3}, L_{N-1}, L_{N-2}.
    //
    // That's awkward to author. Use the symmetric shape instead: a
    // RIGHT-spine where each decision's LEFT child is the current
    // channel leaf and RIGHT child is the next decision. Then:
    //   pop D0 → push [L_0_leaf, D1]; queue: [L_0_leaf, D1]
    //   pop L_0 leaf; queue: [D1]
    //   pop D1 → push [L_1_leaf, D2]; queue: [L_1_leaf, D2]
    //   ...
    // Output order: D0, L_0, D1, L_1, D2, L_2, ..., D_{N-2}, L_{N-2},
    // L_{N-1}. Much cleaner. But the property direction inverts: with
    // the LEFT branch being the per-channel leaf, the decoder takes
    // LEFT when `properties[0] > splitval`, so `chan > k` → leaf for
    // channel ??? We need the LEFT child taken when `chan == k` so
    // each decision peels off one channel.
    //
    // Concretely: at decision D_k we want `chan == k` → its own leaf,
    // `chan > k` → continue down the spine to find a leaf for a
    // higher channel. With splitval = k - 1:
    //   chan > k - 1  → LEFT  (i.e., chan >= k)
    //   chan <= k - 1 → RIGHT
    // That doesn't peel off cleanly either.
    //
    // Easier: split LEFT-spine where LEFT child = next decision, RIGHT
    // child = leaf for the smallest remaining channel. At D_k peel
    // off channel `k` on the RIGHT branch:
    //   D_k: splitval = k → LEFT (chan > k) goes to D_{k+1}; RIGHT
    //   (chan <= k) goes to leaf for channel k. Combined with prior
    //   decisions D_0..D_{k-1} having already peeled channels 0..k-1
    //   via their RIGHT branches, RIGHT at D_k captures exactly chan
    //   == k. The final decision D_{N-2}'s LEFT child is the leaf for
    //   channel N-1.
    //
    // BFS emission order for LEFT-spine where LEFT child is next
    // decision (or final leaf) and RIGHT child is current leaf:
    //   queue: [D0]; pop D0, push [D1, L_0]; queue: [D1, L_0]
    //   pop D1, push [D2, L_1]; queue: [L_0, D2, L_1]
    //   pop L_0; queue: [D2, L_1]
    //   pop D2, push [D3, L_2]; queue: [L_1, D3, L_2]
    //   pop L_1; queue: [D3, L_2]
    //   ...
    // Final pattern: emit D0, D1, L_0, D2, L_1, D3, L_2, ..., D_{N-2},
    // L_{N-3}, L_{N-1}, L_{N-2}.
    //
    // We'll just author the simple recursive form: emit decision +
    // 2 children pre-order, but using BFS via explicit queue.
    use alloc::collections::VecDeque;
    enum NodeKind {
        Decision { splitval: i32 },
        Leaf { chan: usize },
    }
    let mut queue: VecDeque<NodeKind> = VecDeque::new();
    queue.push_back(NodeKind::Decision { splitval: 0 });
    let mut next_split: i32 = 1;
    let mut leaves_emitted: usize = 0;

    while let Some(node) = queue.pop_front() {
        match node {
            NodeKind::Decision { splitval } => {
                // prop1 = property + 1 = 1 (property = 0).
                write_sym(writer, 1)?;
                // splitval packed-signed.
                let packed = if splitval >= 0 {
                    (splitval as u32) << 1
                } else {
                    (((-splitval) as u32) << 1) - 1
                };
                write_sym(writer, packed)?;

                // Queue children: LEFT (chan > splitval) first, then
                // RIGHT (chan <= splitval). LEFT is the next decision
                // unless we are at the last decision.
                let next_splitval = next_split;
                next_split += 1;
                let is_last_decision = (splitval as usize) == n - 2;
                if is_last_decision {
                    // LEFT-most leaf is for channel N-1 (all chan > N-2).
                    queue.push_back(NodeKind::Leaf { chan: n - 1 });
                } else {
                    queue.push_back(NodeKind::Decision {
                        splitval: next_splitval,
                    });
                }
                // RIGHT child: leaf for the current splitval (channel splitval).
                queue.push_back(NodeKind::Leaf {
                    chan: splitval as usize,
                });
            }
            NodeKind::Leaf { chan } => {
                // prop1 = 0 (leaf marker).
                write_sym(writer, 0)?;
                // predictor = 5 (Gradient).
                write_sym(writer, 5)?;
                // offset = 0 (packed_signed(0) == 0).
                write_sym(writer, 0)?;
                let q = quantizers[chan].max(1);
                let (mul_log, mul_bits) = decompose_multiplier_pub(q);
                write_sym(writer, mul_log)?;
                write_sym(writer, mul_bits)?;
                leaves_emitted += 1;
            }
        }
    }
    debug_assert_eq!(leaves_emitted, n, "channel-split tree must emit all leaves");
    Ok(())
}

/// Decompose `q >= 1` into `(mul_log, mul_bits)` such that
/// `q == (mul_bits + 1) << mul_log`. Mirrors
/// [`super::tree::decompose_multiplier`] but is exposed for the lossy
/// alpha encoder.
pub(crate) fn decompose_multiplier_pub(q: u32) -> (u32, u32) {
    let q = q.max(1);
    let trailing = q.trailing_zeros();
    let mul_log = trailing;
    let mul_bits = (q >> trailing) - 1;
    (mul_log, mul_bits)
}

/// Write tree histogram and return (depths, codes) for encoding tree tokens.
fn write_tree_histogram_for_predictor_impl(
    writer: &mut BitWriter,
    write_lz77: bool,
    predictor_id: u32,
) -> Result<(Vec<u8>, Vec<u16>)> {
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: Starting tree histogram (lz77={})",
        writer.bits_written(),
        write_lz77
    );

    // For LfGlobal trees, allow_lz77=true so we write lz77.enabled
    // For modular substream trees (VarDCT), allow_lz77=false so we skip lz77.enabled
    if write_lz77 {
        writer.write(1, 0)?; // lz77.enabled = 0
        crate::trace::debug_eprintln!(
            "  TREE_HIST [bit {}]: lz77.enabled = 0",
            writer.bits_written()
        );
    }

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    // (all contexts share same histogram)
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: context_map (is_simple=1, bits=0)",
        writer.bits_written()
    );

    // use_prefix_code = 1
    writer.write(1, 1)?;
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: use_prefix_code = 1",
        writer.bits_written()
    );

    // Build full Huffman table
    // Tree tokens for a LEAF node: property(ctx0), predictor(ctx1), offset(ctx2), mul_log(ctx3), mul_bits(ctx4)
    //
    // NOTE: For a leaf node, property should be < 0 after unpack_signed().
    // pack_signed(-1) = 1, so we should encode property=1.
    // HOWEVER, jxl-oxide seems to interpret property=0 specially, treating it as a leaf.
    // Using property=0 works with current decoders; property=1 causes decode failures.
    //
    // For TREE_PREDICTOR=p: tokens [0, p, 0, 0, 0]
    //   - property = 0 (works with jxl-oxide as leaf marker)
    //   - predictor = p
    //   - offset = 0
    //   - mul_log = 0
    //   - mul_bits = 0
    // Histogram: symbol 0 appears 4 times, symbol p appears 1 time
    // (for predictor_id == 0 they collide → symbol 0 appears 5 times).
    let max_symbol = predictor_id as u16;
    let _hist_storage_zero: [u32; 1] = [5];
    let _hist_storage_general: Vec<u32> = if predictor_id != 0 {
        let mut v = vec![0u32; (predictor_id as usize) + 1];
        v[0] = 4;
        v[predictor_id as usize] = 1;
        v
    } else {
        Vec::new()
    };
    let tree_histogram: &[u32] = if predictor_id == 0 {
        &_hist_storage_zero
    } else {
        &_hist_storage_general
    };

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // For raw symbols (no hybrid uint), use split_exponent = 15 (max for prefix codes).
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: IntegerConfig (log_alpha={}, split_exp={}, raw symbols)",
        writer.bits_written(),
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX
    );

    // alphabet_size - 1 using VarLenUint16 encoding (matches libjxl)
    // For prefix codes, this is written AFTER IntegerConfigs, BEFORE Huffman tables
    let _alphabet_size = (max_symbol + 1) as u32;
    write_varlen_u16(writer, max_symbol)?;
    crate::trace::debug_eprintln!(
        "  TREE_HIST [bit {}]: alphabet_size-1 = {} (alphabet_size={})",
        writer.bits_written(),
        max_symbol,
        _alphabet_size
    );

    // Huffman table: skip if alphabet_size == 1 (only one possible symbol)
    // IMPORTANT: Return the codes from build_and_store_huffman_tree to ensure
    // tree tokens are encoded with the same codes that were written to the bitstream.
    let al_size = (max_symbol + 1) as usize;
    let (depths, codes) = if al_size > 1 {
        let table = build_and_store_huffman_tree(tree_histogram, writer)?;
        crate::trace::debug_eprintln!(
            "  TREE_HIST [bit {}]: After Huffman table",
            writer.bits_written()
        );
        (table.depths, table.codes)
    } else {
        crate::trace::debug_eprintln!(
            "  TREE_HIST [bit {}]: No Huffman table (al_size=1)",
            writer.bits_written()
        );
        (vec![0u8; al_size], vec![0u16; al_size])
    };

    Ok((depths, codes))
}

/// Write tree tokens for a single leaf with Gradient predictor.
/// Uses the provided (depths, codes) from write_tree_histogram_for_gradient.
pub(crate) fn write_gradient_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
) -> Result<()> {
    write_single_leaf_tree_tokens(writer, depths, codes, 5)
}

/// Write tree tokens for a single leaf using the given fixed `predictor_id`.
///
/// Pair with [`write_tree_histogram_for_predictor`] (must use the same
/// `predictor_id`). Encodes the libjxl `[property=0, predictor, offset=0,
/// mul_log=0, mul_bits=0]` token sequence.
pub(crate) fn write_predictor_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
    predictor_id: u8,
) -> Result<()> {
    write_single_leaf_tree_tokens(writer, depths, codes, predictor_id as u32)
}

/// Write tree tokens for a single leaf with the given predictor ID.
fn write_single_leaf_tree_tokens(
    writer: &mut BitWriter,
    depths: &[u8],
    codes: &[u16],
    predictor_id: u32,
) -> Result<()> {
    crate::trace::debug_eprintln!(
        "  TREE_TOKENS [bit {}]: Starting tree tokens",
        writer.bits_written()
    );

    // Tree tokens for a single LEAF node:
    // - property = 0 (works with jxl-oxide as leaf marker, though spec says should be pack_signed(-1)=1)
    // - predictor (context 1)
    // - offset = 0 (context 2)
    // - mul_log = 0 (context 3)
    // - mul_bits = 0 (context 4) → multiplier = (0+1) << 0 = 1

    let tree_predictor = predictor_id;

    crate::trace::debug_eprintln!("  TREE_TOKENS: depths = {:?}", depths);
    crate::trace::debug_eprintln!("  TREE_TOKENS: codes = {:?}", codes);

    // Encode: property=0, predictor, offset=0, mul_log=0, mul_bits=0
    let tokens = [0u32, tree_predictor, 0, 0, 0];
    let _token_names = ["property", "predictor", "offset", "mul_log", "mul_bits"];

    #[allow(clippy::unused_enumerate_index)]
    for (_i, &token) in tokens.iter().enumerate() {
        let depth = depths.get(token as usize).copied().unwrap_or(0);
        let code = codes.get(token as usize).copied().unwrap_or(0);
        crate::trace::debug_eprintln!(
            "  TREE_TOKENS [bit {}]: {} = {} (depth={}, code={:0width$b})",
            writer.bits_written(),
            _token_names[_i],
            token,
            depth,
            code,
            width = depth.max(1) as usize
        );
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }

    crate::trace::debug_eprintln!("  TREE_TOKENS [bit {}]: Done", writer.bits_written());
    Ok(())
}

/// Write a tree histogram that can encode tokens 0-6 (for Weighted predictor).
pub(super) fn write_tree_histogram_for_weighted(writer: &mut BitWriter) -> Result<()> {
    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map for 6 contexts: is_simple=1, bits_per_entry=0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // Tree tokens for predictor=6: [0,6,0,0,0] → histogram [4,0,0,0,0,0,1]
    const TREE_PREDICTOR: u32 = 6;
    let max_symbol = TREE_PREDICTOR as u16;
    let tree_histogram: &[u32] = &[4u32, 0, 0, 0, 0, 0, 1]; // 5 tokens: 4x symbol 0, 1x symbol 6

    // IntegerConfig: When use_prefix_code=1, the decoder uses log_alphabet_size=15
    // for parsing IntegerConfig, regardless of actual alphabet size.
    // Tree tokens are raw symbols - set split_exponent = 15.
    const LOG_ALPHABET_SIZE_PREFIX: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE_PREFIX,
        LOG_ALPHABET_SIZE_PREFIX,
        0,
        0,
    )?;

    // alphabet_size-1 using VarLenUint16 encoding
    write_varlen_u16(writer, max_symbol)?;

    // Huffman table
    let al_size = (max_symbol + 1) as usize;
    if al_size > 1 {
        build_and_store_huffman_tree(tree_histogram, writer)?;
    }

    Ok(())
}

/// Write tree tokens for a single leaf with Weighted predictor.
pub(super) fn write_weighted_tree_tokens(writer: &mut BitWriter) -> Result<()> {
    const TREE_PREDICTOR: u32 = 6;

    // Compute Huffman codes for the tree histogram
    let tree_histogram = &[4u32, 0, 0, 0, 0, 0, 1]; // 5 tokens: 4x symbol 0, 1x symbol 6
    let depths = create_huffman_tree(tree_histogram, 15);
    let codes = convert_bit_depths_to_symbols(&depths);

    // Encode: property=0 (leaf), predictor=6, offset=0, mul_log=0, mul_bits=0
    let tokens = [0u32, TREE_PREDICTOR, 0, 0, 0];

    for &token in &tokens {
        let depth = depths.get(token as usize).copied().unwrap_or(0);
        let code = codes.get(token as usize).copied().unwrap_or(0);
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
    }

    Ok(())
}

/// Write weighted predictor header parameters.
pub(super) fn write_wp_header(
    writer: &mut BitWriter,
    params: &super::predictor::WeightedPredictorParams,
) -> Result<()> {
    if params.is_default() {
        // all_default = 1 (no additional fields)
        writer.write(1, 1)?;
    } else {
        // all_default = 0, write all parameters
        writer.write(1, 0)?;
        writer.write(5, params.p1c as u64)?;
        writer.write(5, params.p2c as u64)?;
        writer.write(5, params.p3ca as u64)?;
        writer.write(5, params.p3cb as u64)?;
        writer.write(5, params.p3cc as u64)?;
        writer.write(5, params.p3cd as u64)?;
        writer.write(5, params.p3ce as u64)?;
        writer.write(4, params.w0 as u64)?;
        writer.write(4, params.w1 as u64)?;
        writer.write(4, params.w2 as u64)?;
        writer.write(4, params.w3 as u64)?;
    }
    Ok(())
}

/// Write an arbitrary MA tree to the bitstream.
///
/// The tree is serialized in BFS order using 6 token contexts:
/// - SPLIT_VAL_CONTEXT (0): splitval (signed via pack_signed)
/// - PROPERTY_CONTEXT (1): property+1 for split nodes, 0 for leaf nodes
/// - PREDICTOR_CONTEXT (2): predictor index
/// - OFFSET_CONTEXT (3): predictor offset (signed via pack_signed)
/// - MULTIPLIER_LOG_CONTEXT (4): multiplier log component
/// - MULTIPLIER_BITS_CONTEXT (5): multiplier bits component
///
/// Layout: lz77.enabled=0 + context_map (6 contexts → 1 histogram) +
///         use_prefix_code=1 + IntegerConfig + alphabet_size + Huffman table + tokens
pub fn write_tree(writer: &mut BitWriter, tree: &super::tree::Tree) -> Result<()> {
    use super::tree::collect_tree_tokens;
    use crate::entropy_coding::hybrid_uint::HybridUintConfig;

    let tokens = collect_tree_tokens(tree);

    // Encode tree token values through HybridUint to keep the Huffman alphabet
    // within the 32768 symbol limit. Config {4,2,0} maps values up to ~40000
    // into tokens 0..63, with extra bits for the remaining value.
    //
    // Previously used raw symbol encoding (split_exponent=15, msb=15), which
    // worked for small splitvals but exceeded the Huffman alphabet limit when
    // tree learning produced large splitval thresholds (e.g., LfFrame DC integers
    // scaled by inv_dc_quant can reach ~40000 for the X channel).
    let hybrid_config = HybridUintConfig::new(4, 2, 0);

    // Encode all values through HybridUint and collect (token, extra_bits, num_extra)
    struct EncodedTreeToken {
        token: u32,
        extra_bits: u32,
        num_extra: u32,
    }

    let mut encoded: Vec<EncodedTreeToken> = Vec::with_capacity(tokens.len());
    let mut max_token: u32 = 0;
    for t in &tokens {
        let val = if t.is_signed {
            pack_signed(t.value)
        } else {
            t.value as u32
        };
        let (token, extra_bits, num_extra) = hybrid_config.encode(val);
        max_token = max_token.max(token);
        encoded.push(EncodedTreeToken {
            token,
            extra_bits,
            num_extra,
        });
    }

    let histogram_size = (max_token + 1) as usize;
    let mut histogram = vec![0u32; histogram_size];
    for e in &encoded {
        histogram[e.token as usize] += 1;
    }

    // lz77.enabled = 0
    writer.write(1, 0)?;

    // Context map: 6 contexts all map to histogram 0
    writer.write(1, 1)?; // is_simple = 1
    writer.write(2, 0)?; // bits_per_entry = 0

    // use_prefix_code = 1
    writer.write(1, 1)?;

    // IntegerConfig with HybridUint {4,2,0}
    // For Huffman (use_prefix_code=1), the decoder hardcodes log_alpha_size=15
    // (HUFFMAN_MAX_BITS). We must match this when writing the config header,
    // since the number of bits for split_exponent = ceil_log2(log_alpha_size+1).
    const LOG_ALPHABET_SIZE: u32 = 15;
    write_integer_config(
        writer,
        LOG_ALPHABET_SIZE,
        hybrid_config.split_exponent,
        hybrid_config.msb_in_token,
        hybrid_config.lsb_in_token,
    )?;

    // alphabet_size - 1
    write_varlen_u16(writer, max_token as u16)?;

    // Huffman table
    let (depths, codes) = if histogram_size > 1 {
        let table = build_and_store_huffman_tree(&histogram[..histogram_size], writer)?;
        (table.depths, table.codes)
    } else {
        (vec![0u8; histogram_size], vec![0u16; histogram_size])
    };

    // Write tokens + extra bits
    for e in &encoded {
        let depth = depths.get(e.token as usize).copied().unwrap_or(0);
        let code = codes.get(e.token as usize).copied().unwrap_or(0);
        if depth > 0 {
            writer.write(depth as usize, code as u64)?;
        }
        if e.num_extra > 0 {
            writer.write(e.num_extra as usize, e.extra_bits as u64)?;
        }
    }

    Ok(())
}
