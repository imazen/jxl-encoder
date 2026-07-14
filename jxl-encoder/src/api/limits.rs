// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Resource limits (\`Limits\`) — pixel/memory/iteration caps for untrusted input.

/// Resource limits for encoding.
///
/// Every field is `Option<…>`; `None` means "unlimited" (or "use the
/// validator-side default", for fields that have one). Callers wire a
/// [`Limits`] onto an [`EncodeRequest`] via
/// [`EncodeRequest::with_limits`]; the encoder consults it before any
/// dimension-driven allocation and before each per-encode CPU budget
/// check.
///
/// Two policy fields are intentionally NOT bare `pub const`s:
///
/// - [`Self::max_quant_loop_iters`] — the cap on
///   butteraugli/ssim2/zensim quantization-loop iterations. The
///   validator's hard upper bound is [`Self::DEFAULT_MAX_QUANT_LOOP_ITERS`]
///   (= [`crate::validation::ITER_MAX`]); a caller may set a lower limit
///   here, but never a higher one. The encoder saturates at the lower of
///   `Limits.max_quant_loop_iters` (or its default) and the validator
///   max.
/// - [`Self::max_memory_bytes`] — when `None`, the encoder applies a
///   path-aware soft cap so that an image proxy without explicit `Limits`
///   configuration still has a working-set ceiling:
///   [`Self::DEFAULT_MAX_MEMORY_BYTES`] (≈ 4 GiB) for lossy,
///   [`Self::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS`] (8 GiB) for lossless
///   (lossless tree-learning is a heavier memory regime). Both are fixed
///   ceilings — deliberately not scaled with image dimensions. Set to
///   `Some(u64::MAX)` to opt out of the soft cap explicitly (this surfaces
///   in logs as "user explicitly disabled memory limit").
#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub(crate) max_width: Option<u64>,
    pub(crate) max_height: Option<u64>,
    pub(crate) max_pixels: Option<u64>,
    pub(crate) max_memory_bytes: Option<u64>,
    pub(crate) max_quant_loop_iters: Option<u32>,
    pub(crate) fallible_alloc: bool,
}

impl Limits {
    /// Hard upper bound for quantization-loop iterations. Mirrors
    /// [`crate::validation::ITER_MAX`] so the validator and the encoder
    /// agree on what counts as "too many iters".
    pub const DEFAULT_MAX_QUANT_LOOP_ITERS: u32 = crate::validation::ITER_MAX;

    /// Default soft cap on encoder working-set memory when no explicit
    /// [`Self::with_max_memory_bytes`] is set.
    ///
    /// Set to 4 GiB. The measured VarDCT peak working set is ~180
    /// bytes/pixel — e.g. ~2.2 GB at 12 MP, in line with libjxl `cjxl`
    /// (1.2 GB at effort 7 → 3.4 GB at effort 9 on the same image). The
    /// previous 2 GB default rejected ordinary ≥ 11 MP encodes that the
    /// encoder (and libjxl) handle fine. 4 GiB covers up to ~22 MP at the
    /// measured rate while still bounding an oversized upload so it can't
    /// OOM the process. For hostile-input image proxies, set a tighter
    /// cap explicitly via [`Self::with_max_memory_bytes`]; for trusted
    /// batch work, raise it (or pass `Some(u64::MAX)` to opt out).
    pub const DEFAULT_MAX_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;

    /// Default soft cap for the **lossless** path when no explicit
    /// [`Self::with_max_memory_bytes`] is set. Lossless is a heavier
    /// memory regime than lossy at the same dimensions — MA tree-learning
    /// at effort ≥ 7 measures ~440 B/px (≈ 5 GB at 12 MP) vs lossy's
    /// ~85–300 B/px (see [`crate::heuristics`]). A single 4 GiB default
    /// rejected ordinary ≥ 9 MP lossless encodes, so the lossless default
    /// is 8 GiB. Still a fixed DoS ceiling (not scaled with dimensions);
    /// truly oversized untrusted input is still rejected, and trusted
    /// batch raises it explicitly.
    pub const DEFAULT_MAX_MEMORY_BYTES_LOSSLESS: u64 = 8 * 1024 * 1024 * 1024;

    /// Default pre-flight pixel cap (`width × height`) applied to the
    /// untrusted JPEG-transcode path
    /// ([`LosslessConfig::encode_jpeg_transcode`] and friends) when the
    /// caller sets no explicit [`Self::with_max_pixels`].
    ///
    /// Set to 120 MP — admits 108 MP phone photos while bounding a crafted
    /// SOF (JPEG dims are `u16`, so up to ~4.3 Gpx) from forcing a
    /// multi-gigabyte `i16` coefficient allocation in the transcode parser,
    /// which builds no `MemoryBudget`. The cap is checked immediately after
    /// the SOF dimensions are read, before any allocation.
    ///
    /// Callers override it through [`Self::with_max_pixels`]: a higher value
    /// for trusted batch input (or [`u64::MAX`] to opt out entirely), a lower
    /// value to tighten the bound on a hostile-input proxy.
    pub const DEFAULT_MAX_JPEG_TRANSCODE_PIXELS: u64 = 120_000_000;

    /// Path-aware default memory cap: [`Self::DEFAULT_MAX_MEMORY_BYTES`]
    /// for lossy, [`Self::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS`] for
    /// lossless. Used when the caller set no explicit
    /// [`Self::with_max_memory_bytes`].
    pub const fn default_max_memory_bytes(is_lossless: bool) -> u64 {
        if is_lossless {
            Self::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS
        } else {
            Self::DEFAULT_MAX_MEMORY_BYTES
        }
    }

    /// Create limits with no restrictions (all `None`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum image width.
    pub fn with_max_width(mut self, w: u64) -> Self {
        self.max_width = Some(w);
        self
    }

    /// Set maximum image height.
    pub fn with_max_height(mut self, h: u64) -> Self {
        self.max_height = Some(h);
        self
    }

    /// Set maximum total pixels (width × height).
    pub fn with_max_pixels(mut self, p: u64) -> Self {
        self.max_pixels = Some(p);
        self
    }

    /// Set maximum memory bytes the encoder may allocate.
    ///
    /// When unset, the encoder applies a path-aware soft cap:
    /// [`Self::DEFAULT_MAX_MEMORY_BYTES`] (~4 GiB) for lossy,
    /// [`Self::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS`] (8 GiB) for lossless
    /// (lossless tree-learning is a heavier memory regime). Pass
    /// `u64::MAX` explicitly to disable the cap (with the understanding
    /// that an unbounded working set on a hostile input can OOM the
    /// process).
    pub fn with_max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Set maximum quantization-loop iterations (butteraugli / ssim2 /
    /// zensim). Saturated at [`Self::DEFAULT_MAX_QUANT_LOOP_ITERS`] —
    /// passing a higher value silently lowers it to the validator-side
    /// hard limit. Use a lower value to bound CPU on untrusted callers.
    pub fn with_max_quant_loop_iters(mut self, n: u32) -> Self {
        self.max_quant_loop_iters = Some(n.min(Self::DEFAULT_MAX_QUANT_LOOP_ITERS));
        self
    }

    /// Get maximum width, if set.
    pub fn max_width(&self) -> Option<u64> {
        self.max_width
    }

    /// Get maximum height, if set.
    pub fn max_height(&self) -> Option<u64> {
        self.max_height
    }

    /// Get maximum pixels, if set.
    pub fn max_pixels(&self) -> Option<u64> {
        self.max_pixels
    }

    /// Get maximum memory bytes, if set. When `None`, the encoder applies
    /// the path-aware soft default — see [`Self::effective_max_memory_bytes`]
    /// for the caveat about which value that is.
    pub fn max_memory_bytes(&self) -> Option<u64> {
        self.max_memory_bytes
    }

    /// The explicit `max_memory_bytes` if set, else the **lossy** soft
    /// default [`Self::DEFAULT_MAX_MEMORY_BYTES`] (≈ 4 GiB).
    ///
    /// NOTE: this is not path-aware. When `max_memory_bytes` is unset, the
    /// encoder enforces the path-aware default from
    /// [`Self::default_max_memory_bytes`] — which is
    /// [`Self::DEFAULT_MAX_MEMORY_BYTES_LOSSLESS`] (8 GiB) on the lossless
    /// path (lossless tree-learning is a heavier memory regime). This getter
    /// always reports the lossy value, so for an unset limit it understates
    /// the cap a lossless encode actually runs under. Pass `is_lossless` to
    /// [`Self::default_max_memory_bytes`] for the value the encoder uses.
    pub fn effective_max_memory_bytes(&self) -> u64 {
        self.max_memory_bytes
            .unwrap_or(Self::DEFAULT_MAX_MEMORY_BYTES)
    }

    /// Get maximum quantization-loop iterations.
    pub fn max_quant_loop_iters(&self) -> Option<u32> {
        self.max_quant_loop_iters
    }

    /// The cap the encoder will actually apply: explicit
    /// `max_quant_loop_iters` if set, else
    /// [`Self::DEFAULT_MAX_QUANT_LOOP_ITERS`].
    pub fn effective_max_quant_loop_iters(&self) -> u32 {
        self.max_quant_loop_iters
            .unwrap_or(Self::DEFAULT_MAX_QUANT_LOOP_ITERS)
    }

    /// Choose **fallible** allocation for the large dimension-driven buffers
    /// sized from untrusted input (the JPEG-transcode coefficient / working
    /// buffers).
    ///
    /// - `false` (default): `vec![v; n]` — LLVM lowers the zeroed case to a
    ///   single `calloc`, the faster path. An allocation that the OS cannot
    ///   satisfy aborts the process.
    /// - `true`: `try_reserve` — a failed allocation returns
    ///   [`EncodeError`] / [`crate::error::Error::OutOfMemory`] instead of
    ///   aborting. Slightly slower, but a hostile-input image proxy stays up.
    ///
    /// Orthogonal to [`Self::with_max_memory_bytes`]: the budget *rejects*
    /// oversized requests up front; this controls how an *accepted* request is
    /// allocated. A server handling untrusted uploads typically wants both.
    pub fn with_fallible_alloc(mut self, fallible: bool) -> Self {
        self.fallible_alloc = fallible;
        self
    }

    /// Whether fallible allocation is enabled (see [`Self::with_fallible_alloc`]).
    pub fn fallible_alloc(&self) -> bool {
        self.fallible_alloc
    }
}
