// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Study instrumentation for the IMIJ coefficient-gap study (feature
//! `coefgap`, 2026-10-09; imazen/zqi `benchmarks/coefgap_2026-10-09`).
//!
//! Thread-local switches that turn the JPEG transcode's coefficient-coding
//! mechanisms off one at a time, and a byte report per section and token
//! class. With default switches the transcode is byte-identical to a build
//! without the feature. Switches that change what a decoder derives by itself
//! (the nonzero-count prediction, the previous-coefficient bit, DC prediction
//! restricted to IMIJ's stripes and lanes) produce streams no JPEG XL decoder
//! reads: their byte counts are what that context model would cost.

use alloc::vec;
use alloc::vec::Vec;
use std::cell::{Cell, RefCell};

use crate::entropy_coding::token::Token;
use crate::modular::predictor::pack_signed;
/// Re-exported for offline models of the DC planes.
pub use crate::modular::predictor::{Neighbors, WeightedPredictorState};
/// Re-exported for offline models of the AC tokens (8x8 blocks:
/// `zero_density_context(nonzeros_left, k, 1, 0, prev)`).
pub use crate::vardct::ac_context::zero_density_context;
pub use crate::vardct::dc_tree_learn::{
    DcTree, DcTreeNode, build_gradient_fixed_dc_tree, build_wp_fixed_dc_tree, get_wp_dc_context,
};

/// How the nonzero-count token's context predicts the count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NzPred {
    /// JPEG XL: rounded mean of the block above and the block to the left.
    #[default]
    TopLeft,
    /// A constant (32): no neighbour information.
    Const,
    /// The block above only.
    Top,
    /// The block to the left only.
    Left,
    /// IMIJ's run-level lanes: the block 16 to the left in the same row (the
    /// previous block of IMIJ lane `bx % 16`), else the default.
    Lane16,
}

/// DC predictor and context tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DcMode {
    /// JPEG XL's transcode: weighted predictor, fixed tree on its max error.
    #[default]
    WpTree,
    /// Weighted predictor, one context.
    WpSingle,
    /// Clamped gradient predictor, fixed tree on the gradient property
    /// (libjxl's `kGradientFixedDC`).
    GradTree,
    /// Clamped gradient predictor, one context.
    GradSingle,
    /// Clamped gradient predictor, contexts from the weighted predictor's max
    /// error (the WP tree), i.e. JPEG XL's contexts with a cheap predictor.
    GradWpCtx,
}

/// The study's switches (defaults = the transcode as shipped).
#[derive(Clone, Copy, Debug, Default)]
pub struct Switches {
    /// No chroma-from-luma (zero CfL map).
    pub no_cfl: bool,
    /// One block context per channel (no luma-DC quantile buckets).
    pub no_dc_ctx: bool,
    /// Nonzero-count prediction.
    pub nz_pred: NzPred,
    /// Zero-density contexts without the previous-coefficient bit.
    pub no_prev: bool,
    /// Natural (zigzag) coefficient order.
    pub no_orders: bool,
    /// DC predictor and contexts.
    pub dc_mode: DcMode,
    /// DC prediction inside IMIJ-like segments: stripes of this many DC rows
    /// (0 = the DC group).
    pub dc_stripe_rows: usize,
    /// DC prediction inside this many lane segments of columns per stripe,
    /// IMIJ's lossless lanes (0 or 1 = none).
    pub dc_lanes: usize,
    /// Most clustered AC histograms (0 = the encoder's own limit).
    pub max_ac_histograms: usize,
    /// Most clustered DC histograms (0 = the encoder's own limit).
    pub max_dc_histograms: usize,
    /// The default HybridUint config (4, 2, 0) for every DC and AC histogram
    /// instead of the per-histogram optimized ones.
    pub fixed_uint: bool,
}

/// Bytes and estimated bits of one transcode.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// File header and frame header bytes (the frame header in bits rounded up).
    pub headers: usize,
    /// DC global section (dequant, block context map, tree, DC entropy code).
    pub dc_global: usize,
    /// DC group sections (DC tokens and AC metadata tokens).
    pub dc_groups: usize,
    /// AC global section (quant tables, orders, AC entropy code).
    pub ac_global: usize,
    /// AC group sections.
    pub ac_groups: usize,
    /// Bits of the DC entropy code (context map and histograms).
    pub dc_code_bits: usize,
    /// Bits of the AC entropy code (context map and histograms).
    pub ac_code_bits: usize,
    /// Bits of the coefficient orders.
    pub order_bits: usize,
    /// Estimated bits of the DC tokens per channel (Y, X, B in JPEG XL order:
    /// index 0 = X, 1 = Y, 2 = B).
    pub dc_bits: [f64; 3],
    /// Estimated bits of the AC metadata tokens.
    pub meta_bits: f64,
    /// Estimated bits of AC tokens per channel (0 = X, 1 = Y, 2 = B): nonzero
    /// count tokens, coefficient tokens.
    pub ac_bits: [[f64; 2]; 3],
    /// Token counts: DC, AC metadata, AC.
    pub tokens: [usize; 3],
    /// Clustered histograms: DC code, AC code.
    pub histograms: [usize; 2],
    /// Raw contexts: DC code, AC code.
    pub contexts: [usize; 2],
    /// Custom order mask and luma-DC block contexts.
    pub used_orders: u32,
    pub num_dc_ctxs: usize,
}

thread_local! {
    static SWITCHES: Cell<Switches> = Cell::new(Switches::default());
    static REPORT: RefCell<Option<Report>> = const { RefCell::new(None) };
    static HIST_CAP: Cell<usize> = const { Cell::new(0) };
}

/// Sets this thread's switches for the next transcodes.
pub fn set_switches(s: Switches) {
    SWITCHES.with(|c| c.set(s));
}

/// This thread's switches.
pub fn switches() -> Switches {
    SWITCHES.with(|c| c.get())
}

/// The report of this thread's last transcode.
pub fn take_report() -> Option<Report> {
    REPORT.with(|r| r.borrow_mut().take())
}

pub(crate) fn put_report(r: Report) {
    REPORT.with(|c| *c.borrow_mut() = Some(r));
}

/// Histogram cap for the entropy code being built (0 = none).
pub(crate) fn hist_cap() -> usize {
    HIST_CAP.with(|c| c.get())
}

pub(crate) fn set_hist_cap(n: usize) {
    HIST_CAP.with(|c| c.set(n));
}

/// A one-leaf tree with `predictor` (5 gradient, 6 weighted).
pub(crate) fn single_leaf_tree(predictor: u32) -> (DcTree, u32) {
    (
        vec![DcTreeNode {
            property: -1,
            context_id: 0,
            predictor,
            ..Default::default()
        }],
        1,
    )
}

/// Nonzero-count prediction for block (`bx`, `by`) of a channel under
/// `pred` (`nz` the channel's count rows; `y0` the first row of the group,
/// `x0` its first column).
pub(crate) fn predict_nz(
    pred: NzPred,
    nz: &[Vec<u8>],
    bx: usize,
    by: usize,
    x0: usize,
    y0: usize,
) -> i32 {
    let top = (by > y0).then(|| nz[by - 1][bx] as i32);
    let left = (bx > x0).then(|| nz[by][bx - 1] as i32);
    match pred {
        NzPred::TopLeft => match (top, left) {
            (Some(t), Some(l)) => (t + l + 1) / 2,
            (Some(t), None) => t,
            (None, Some(l)) => l,
            (None, None) => 32,
        },
        NzPred::Const => 32,
        NzPred::Top => top.unwrap_or(32),
        NzPred::Left => left.unwrap_or(32),
        NzPred::Lane16 => {
            if bx >= x0 + 16 {
                nz[by][bx - 16] as i32
            } else {
                32
            }
        }
    }
}

/// Clamped gradient (libjxl `Predictor::Gradient`).
fn clamped_gradient(n: i32, w: i32, nw: i32) -> i32 {
    let (lo, hi) = (n.min(w), n.max(w));
    (n + w - nw).clamp(lo, hi)
}

/// DC tokens of one channel's rectangle under `mode`, with neighbours only
/// inside the rectangle (modular's edge rules).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dc_tokens_rect(
    channel: &[Vec<i32>],
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    mode: DcMode,
    tree: &DcTree,
    wp_tree: &DcTree,
    out: &mut Vec<Token>,
) {
    let width = x1 - x0;
    if width == 0 || y1 <= y0 {
        return;
    }
    let mut wp = WeightedPredictorState::with_defaults(width);
    for y in y0..y1 {
        for x in x0..x1 {
            let at = |yy: usize, xx: usize| channel[yy][xx];
            let w = if x > x0 {
                at(y, x - 1)
            } else if y > y0 {
                at(y - 1, x)
            } else {
                0
            };
            let n = if y > y0 { at(y - 1, x) } else { w };
            let nw = if x > x0 && y > y0 {
                at(y - 1, x - 1)
            } else {
                w
            };
            let ne = if x + 1 < x1 && y > y0 {
                at(y - 1, x + 1)
            } else {
                n
            };
            let ww = if x > x0 + 1 { at(y, x - 2) } else { w };
            let nn = if y > y0 + 1 { at(y - 2, x) } else { n };
            let nee = if x + 2 < x1 && y > y0 {
                at(y - 1, x + 2)
            } else {
                ne
            };
            let nb = Neighbors {
                n,
                w,
                nw,
                ne,
                nn,
                ww,
                nee,
            };
            let (lx, ly) = (x - x0, y - y0);
            let actual = at(y, x);
            let (wp_pred, wp_err) = wp.predict_and_property(lx, ly, width, &nb);
            let (pred, ctx) = match mode {
                DcMode::WpTree | DcMode::WpSingle => {
                    (wp_pred as i32, get_wp_dc_context(tree, wp_err))
                }
                DcMode::GradTree => (
                    clamped_gradient(n, w, nw),
                    get_wp_dc_context(tree, w + n - nw),
                ),
                DcMode::GradSingle => (clamped_gradient(n, w, nw), 0),
                DcMode::GradWpCtx => (
                    clamped_gradient(n, w, nw),
                    get_wp_dc_context(wp_tree, wp_err),
                ),
            };
            out.push(Token::new(ctx, pack_signed(actual - pred)));
            wp.update_errors(actual, lx, ly, width);
        }
    }
}

/// DC tokens of one channel's DC-group rectangle split into IMIJ-like
/// segments (stripes of `stripe_rows`, `lanes` lane segments).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dc_tokens_segmented(
    channel: &[Vec<i32>],
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    sw: &Switches,
    tree: &DcTree,
    wp_tree: &DcTree,
    out: &mut Vec<Token>,
) {
    let rows = if sw.dc_stripe_rows == 0 {
        y1 - y0
    } else {
        sw.dc_stripe_rows
    };
    let lanes = sw.dc_lanes.max(1);
    let lw = (x1 - x0).div_ceil(lanes);
    let mut ys = y0;
    while ys < y1 {
        let ye = (ys + rows).min(y1);
        for l in 0..lanes {
            let xs = (x0 + l * lw).min(x1);
            let xe = (xs + lw).min(x1);
            dc_tokens_rect(channel, xs, xe, ys, ye, sw.dc_mode, tree, wp_tree, out);
        }
        ys = ye;
    }
}

/// Bytes of a DC image coded as the JPEG transcode codes its DC (the
/// modular tree for `sw.dc_mode`, one DC group per 256 x 256 blocks, the
/// contexts clustered with the DC code's settings, one ANS stream per DC
/// group), with any DC values (no JPEG range limit). `planes` in JPEG XL
/// channel order (X, Y, B), `w` x `h` blocks. AC metadata is not coded.
#[derive(Clone, Copy, Debug, Default)]
pub struct DcBytes {
    /// The context tree (with the transcode's AC-metadata subtree).
    pub tree: usize,
    /// The DC entropy code (context map, histograms).
    pub code: usize,
    /// The DC groups' token streams.
    pub groups: usize,
    /// Clustered histograms.
    pub histograms: usize,
}

/// See [`DcBytes`].
pub fn dc_coded_bytes(
    planes: &[Vec<i32>; 3],
    w: usize,
    h: usize,
    sw: &Switches,
) -> crate::error::Result<DcBytes> {
    use crate::bit_writer::BitWriter;
    use crate::entropy_coding::encode::{
        build_entropy_code_ans_with_options, write_entropy_code_ans, write_tokens_ans,
    };
    use crate::vardct::dc_tree_learn::{AcMetaTreeKind, tree_tokens_with_ac_metadata_prefix};
    let total = w * h * 3;
    let (wp_fixed, wp_n) = build_wp_fixed_dc_tree(total, 8);
    let (tree, n) = match sw.dc_mode {
        DcMode::WpTree => (wp_fixed.clone(), wp_n),
        DcMode::WpSingle => single_leaf_tree(6),
        DcMode::GradSingle => single_leaf_tree(5),
        DcMode::GradTree => build_gradient_fixed_dc_tree(total, 8),
        DcMode::GradWpCtx => {
            let mut t = wp_fixed.clone();
            for nd in t.iter_mut().filter(|nd| nd.property < 0) {
                nd.predictor = 5;
            }
            (t, wp_n)
        }
    };
    const G: usize = 256;
    let (gx, gy) = (w.div_ceil(G), h.div_ceil(G));
    let (tree_tokens, total_ctx, remap, _) =
        tree_tokens_with_ac_metadata_prefix(&tree, n, gx * gy, AcMetaTreeKind::Ours, false);
    let rows: [Vec<Vec<i32>>; 3] =
        core::array::from_fn(|c| planes[c].chunks(w).map(|r| r.to_vec()).collect());
    let mut groups: Vec<Vec<Token>> = Vec::new();
    for g in 0..gx * gy {
        let (x0, y0) = ((g % gx) * G, (g / gx) * G);
        let (x1, y1) = ((x0 + G).min(w), (y0 + G).min(h));
        let mut t = Vec::new();
        for &c in &[1usize, 0, 2] {
            dc_tokens_segmented(&rows[c], x0, x1, y0, y1, sw, &tree, &wp_fixed, &mut t);
        }
        for tok in t.iter_mut() {
            tok.set_context(remap[tok.context() as usize]);
        }
        groups.push(t);
    }
    let all: Vec<Token> = groups.iter().flatten().copied().collect();
    set_hist_cap(sw.max_dc_histograms);
    let code = build_entropy_code_ans_with_options(
        &all,
        total_ctx as usize,
        true,
        !sw.fixed_uint,
        None,
        Some(w * h * 64),
    );
    set_hist_cap(0);
    let mut out = DcBytes {
        histograms: code.histograms.len(),
        ..Default::default()
    };
    let mut tw = BitWriter::new();
    crate::vardct::context_tree::write_learned_context_tree(&tree_tokens, gx * gy, &mut tw, None)?;
    out.tree = tw.bits_written().div_ceil(8);
    let mut cw = BitWriter::new();
    write_entropy_code_ans(&code, &mut cw)?;
    out.code = cw.bits_written().div_ceil(8);
    for t in &groups {
        let mut gw = BitWriter::new();
        write_tokens_ans(t, &code, None, &mut gw)?;
        out.groups += gw.bits_written().div_ceil(8);
    }
    Ok(out)
}
