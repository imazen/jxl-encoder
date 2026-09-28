// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! API-side config bits needed by the modular layer (`SectionedTrees`,
//! `cast_pixel_lanes`), extracted so `jxl-modular` does not depend on
//! `jxl-encoder`'s `api` module.

/// lossless e <= 7 therefore depends on the thread configuration by design;
/// pin `On` / `Off` for thread-invariant bytes. Scope: tree-learning ANS
/// encodes, including palette / ChannelCompact content (the meta channels
/// are coded in the global stream with their own tiny tree) and the
/// lossless patches dictionary; only custom-DC-quant (lossy-modular) and
/// the non-tree / non-ANS modes keep the whole-image tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SectionedTrees {
    /// Engage when the memory budget requires it (default).
    #[default]
    Auto,
    /// Never sectioned: always the whole-image global tree.
    Off,
    /// Always sectioned (tree-learning ANS encodes; see the type docs).
    On,
    /// Learn BOTH the global tree and per-group trees, and write each
    /// group with whichever is smaller (per-group `use_global_tree`
    /// choice — measured −2.25% (e7) / −0.25% (e9) vs the global tree on
    /// the 4K photo cell, ≥ global on every content class by
    /// construction). Uses global-mode memory; the per-group learns ride
    /// the gather waves in parallel.
    Hybrid,
}

pub fn cast_pixel_lanes<T: bytemuck::AnyBitPattern>(pixels: &[u8]) -> alloc::borrow::Cow<'_, [T]> {
    debug_assert_eq!(pixels.len() % core::mem::size_of::<T>(), 0);
    match bytemuck::try_cast_slice::<u8, T>(pixels) {
        Ok(lanes) => alloc::borrow::Cow::Borrowed(lanes),
        Err(_) => alloc::borrow::Cow::Owned(
            // `as_chunks::<{size_of::<T>()}>()` needs unstable
            // generic_const_exprs; the lint's suggestion can't apply here.
            #[allow(clippy::chunks_exact_to_as_chunks)]
            pixels
                .chunks_exact(core::mem::size_of::<T>())
                .map(bytemuck::pod_read_unaligned::<T>)
                .collect(),
        ),
    }
}
