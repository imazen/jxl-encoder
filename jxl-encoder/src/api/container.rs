// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Output-container policy and shared knob enums: `ContainerMode`,
//! `WritableSeek`, `Buffering`, `PremultipliedAlphaMode`, and the `MAX_*`
//! limit constants.

/// Container-wrap policy for the encoded JXL output.
///
/// Mirrors libjxl `cjxl --container 0|1`. The default ([`Auto`]) wraps
/// the codestream in a JXL container (`JXL ` signature box +
/// `jxlc`/`jxlp` data boxes + any metadata boxes) **only** when
/// required — i.e., the codestream uses a level that requires the
/// container box (libjxl `MustUseContainer`), or the caller attached
/// EXIF / XMP / JUMBF / colr / hCdR metadata.
///
/// [`Always`] forces a container wrapper even when the bare codestream
/// would have been spec-valid on its own — useful for downstream tools
/// that always expect the ISOBMFF framing. [`Never`] skips the
/// container even when metadata is present (the metadata is silently
/// dropped); this fails the encode (returns
/// [`crate::EncodeError::InvalidInput`]) if the codestream level requires a
/// container, since the result would be unreadable.
///
/// [`Auto`]: ContainerMode::Auto
/// [`Always`]: ContainerMode::Always
/// [`Never`]: ContainerMode::Never
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContainerMode {
    /// **Default.** Wrap in a container box only when required
    /// (metadata present, or `codestream_level != 5`). Matches libjxl's
    /// `MustUseContainer` semantics.
    #[default]
    Auto,
    /// Always emit the container wrapper, even for bare-codestream-OK
    /// encodes. Equivalent to libjxl `--container 1`.
    Always,
    /// Never wrap; emit the bare codestream. Drops attached EXIF / XMP
    /// / JUMBF / colr / hCdR silently (they have nowhere to go without
    /// the container). Returns [`crate::EncodeError::InvalidInput`] when the
    /// codestream level requires a container (e.g. level 10).
    /// Equivalent to libjxl `--container 0`.
    Never,
}

/// Trait bundle for output destinations that support both writing and
/// random-access seeking. Required by the streaming-refactor
/// [`crate::LossyEncoder::finish_to_seekable`] / [`crate::LosslessEncoder::finish_to_seekable`]
/// entry points (jxl-encoder#11 chunk 6).
///
/// The blanket impl covers every type that implements [`std::io::Write`]
/// + [`std::io::Seek`], so concrete callers don't need to opt in:
///
/// ```ignore
/// use std::io::Cursor;
/// use jxl_encoder::{LossyConfig, Quality, PixelLayout};
///
/// let mut buf = Cursor::new(Vec::<u8>::new());
/// LossyConfig::new(1.0)
///     .encoder(1024, 1024, PixelLayout::Rgb8)?
///     .finish_to_seekable(&mut buf)?;
/// let encoded: Vec<u8> = buf.into_inner();
/// # Ok::<(), jxl_encoder::EncodeError>(())
/// ```
///
/// **Chunk 6 (this commit)** plumbs the trait through both encoder
/// builders but uses it only as a [`std::io::Write`]: the buffered-
/// output bytes are produced in memory then written in one pass. The
/// seek capability becomes load-bearing in chunk 7 when the level-3
/// streaming-output path lands (permuted TOC + DC-global placeholder +
/// post-frame seek-back, mirroring libjxl `acc28c0` /
/// `OutputProcessor::Seek`).
///
/// Mirrors libjxl's "streaming_output" assumption in
/// `EncodeFrameStreaming` (`enc_frame.cc:2042-2200`) that the output
/// sink can rewind to the DC-global slot once all per-DC-group section
/// bytes are known.
#[cfg(feature = "std")]
pub trait WritableSeek: std::io::Write + std::io::Seek {}

#[cfg(feature = "std")]
impl<T: std::io::Write + std::io::Seek> WritableSeek for T {}

/// Input/output buffering policy for the encode pipeline. Mirrors
/// libjxl `cjxl --buffering -1..3`
/// ([`JXL_ENC_FRAME_SETTING_BUFFERING`][libjxl-encode-h]).
///
/// This is the scaffolding API for the streaming refactor tracked in
/// jxl-encoder#11 / libjxl PRs #4634 + #4635 + #4637 + #4642 + #4728
/// (commits `acc28c0` + `032d39a` + `b3510d1` + `1389871` + `6553831`).
/// **Chunk 1 (this commit)** introduces the enum, the builder methods
/// on [`crate::LossyConfig`] / [`crate::LosslessConfig`], and the CLI flag. **No
/// dispatch is wired yet** — every variant currently routes through
/// the existing one-shot full-buffer path, so output bytes are
/// identical regardless of which `Buffering` value is selected.
/// Chunks 2-7 land the per-DC-group split, the buffered-output path
/// (libjxl level 2), the permuted-TOC seek-back path (libjxl level
/// 3), and the lossless mirror. See
/// [`libjxl_streaming_refactor_porting_plan_2026-05-18`][plan] for
/// the full chunk schedule.
///
/// libjxl semantics (post-`acc28c0`):
///
/// | libjxl int | This enum                       | Meaning                                                                                          |
/// |-----------:|---------------------------------|--------------------------------------------------------------------------------------------------|
/// |       `-1` | [`Auto`](Self::Auto)            | Encoder picks. Currently resolves to libjxl level **2** for `num_dc_groups > 8`, else level **0**. |
/// |        `0` | [`FullBuffered`](Self::FullBuffered) | Buffer everything — semantically equivalent to today's one-shot encode path.                |
/// |        `1` | [`Threshold2048`](Self::Threshold2048) | Buffer for ≤ 2048×2048; stream input + buffer output for larger images.                   |
/// |        `2` | [`BufferedOutput`](Self::BufferedOutput) | Stream input + buffer output whenever `num_dc_groups > 8`. **libjxl default since `032d39a`.** |
/// |        `3` | [`FullStreaming`](Self::FullStreaming) | Stream input AND stream output. Requires seek-back support on the output sink; the produced bitstream is not progressively decodable. |
///
/// **Critical distinction** (per the libjxl-spec doc-comment in
/// `lib/include/jxl/encode.h` post-`acc28c0`):
///
/// - **Levels 0-2** all produce *progressive-decodability-friendly*
///   bitstreams with a non-permuted TOC and natural-order section
///   layout. Level 2 simply trades the input-side full-buffer for a
///   streaming pixel source while still buffering the encoded
///   per-DC-group sections in `global_group_codes[]` until the loop
///   ends.
/// - **Level 3** is the original "streaming encode" path: permuted
///   TOC with a DC-global placeholder, sections emitted to the sink
///   as soon as each DC group finishes, then a seek-back at the end
///   to fill in the real DC-global + TOC. Smaller peak RAM but the
///   output is *not* progressively decodable.
///
/// **Default**: [`Auto`](Self::Auto) (matches libjxl post-`032d39a`,
/// which changed `JXL_ENC_FRAME_SETTING_BUFFERING` default from `1`
/// to `2`).
///
/// [libjxl-encode-h]: https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/encode.h
/// [plan]: https://github.com/imazen/jxl-encoder/issues/11
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Buffering {
    /// **Default.** Encoder picks per-image based on dimensions and
    /// `num_dc_groups`. Mirrors libjxl `--buffering -1`.
    ///
    /// Current heuristic (chunk 1; will refine in chunks 5/7):
    /// - `num_dc_groups <= 8` → resolves to [`FullBuffered`](Self::FullBuffered).
    /// - Otherwise → resolves to [`BufferedOutput`](Self::BufferedOutput).
    ///
    /// The 2048² threshold matches libjxl's
    /// `CanDoStreamingEncoding` gate (`enc_frame.cc:1779-1820`): a
    /// `2048×2048` image fits in exactly one DC group (so streaming
    /// gives no win), while larger images split into multiple DC
    /// groups where the buffered-output path can drop per-region
    /// pixel buffers as soon as the corresponding sections are
    /// emitted.
    #[default]
    Auto,
    /// Buffer everything. Semantically equivalent to today's one-shot
    /// encode path. Mirrors libjxl `--buffering 0`.
    ///
    /// Peak memory ≈ pixel buffer + full XYB / quant / mask / CfL /
    /// AC-strategy plane buffers + accumulated section bytes. Smallest
    /// code path; same output as the pre-streaming-refactor encoder.
    FullBuffered,
    /// Buffer everything for inputs ≤ 2048×2048; otherwise stream
    /// input + buffer output. Mirrors libjxl `--buffering 1`.
    ///
    /// Chunk 1: no behavioural difference — routes through the
    /// one-shot path. Chunks 3+5 land the per-DC-group split and the
    /// large-image streaming path.
    Threshold2048,
    /// Always stream input + buffer output when `num_dc_groups > 8`
    /// (i.e. images larger than ~ a single 2048×2048 DC group).
    /// Mirrors libjxl `--buffering 2`. **This is libjxl's default
    /// since `032d39a`.**
    ///
    /// Buffered-output path: the encoder still accumulates every
    /// DC-group's bitstream section in `global_group_codes[]` until
    /// the per-group loop finishes, then writes a non-permuted TOC +
    /// sections in natural order. No seek-back required on the
    /// output sink. Lets the encoder drop each DC group's plane
    /// slice as soon as its sections are emitted — the load-bearing
    /// memory win is the absence of the whole-image XYB / quant /
    /// CfL / AC-strategy plane buffers, not the section buffers
    /// themselves.
    ///
    /// Chunk 1: no behavioural difference — routes through the
    /// one-shot path. Chunk 5 lands the buffered-output streaming
    /// path.
    BufferedOutput,
    /// Stream input AND stream output. Mirrors libjxl `--buffering 3`.
    ///
    /// Requires seek-back support on the output sink (the encoder
    /// reserves the DC-global slot, emits per-DC-group sections as
    /// they finish, then seeks back to write the real DC-global +
    /// permuted TOC at end-of-frame). The produced bitstream is *not*
    /// progressively decodable — the TOC permutation reorders the
    /// sections so DC-global sits at the end.
    ///
    /// Chunk 1: no behavioural difference — routes through the
    /// one-shot path. Chunks 6-7 land the `WritableSeek` trait and
    /// the level-3 streaming-output path.
    FullStreaming,
}

impl Buffering {
    /// Convert from the libjxl `--buffering` integer encoding
    /// (`-1..=3`). Values outside the documented range fold to
    /// [`Auto`](Self::Auto) (matches libjxl's
    /// `JXL_ENC_FRAME_SETTING_BUFFERING` defaulting behaviour for
    /// out-of-range input on the C API).
    pub const fn from_i8(v: i8) -> Self {
        match v {
            0 => Self::FullBuffered,
            1 => Self::Threshold2048,
            2 => Self::BufferedOutput,
            3 => Self::FullStreaming,
            _ => Self::Auto,
        }
    }

    /// Inverse of [`Self::from_i8`]: convert to the libjxl `cjxl
    /// --buffering` integer encoding. [`Auto`](Self::Auto) maps to
    /// `-1`.
    pub const fn to_i8(self) -> i8 {
        match self {
            Self::Auto => -1,
            Self::FullBuffered => 0,
            Self::Threshold2048 => 1,
            Self::BufferedOutput => 2,
            Self::FullStreaming => 3,
        }
    }

    /// Resolve [`Auto`](Self::Auto) to a concrete variant for an
    /// image with the given pixel dimensions. Non-`Auto` variants
    /// are returned unchanged.
    ///
    /// Chunk 1 heuristic (mirrors libjxl `CanDoStreamingEncoding`
    /// in `enc_frame.cc:1779-1820`): images with `width * height
    /// <= 2048 * 2048` (i.e. one DC group) resolve to
    /// [`FullBuffered`](Self::FullBuffered); larger images resolve
    /// to [`BufferedOutput`](Self::BufferedOutput) (libjxl level 2,
    /// matching the post-`032d39a` default).
    ///
    /// This is a no-op for chunk 1 — every concrete variant
    /// currently routes through the same one-shot encode path. The
    /// helper exists so chunks 3-7 can dispatch on the resolved
    /// value without re-implementing the threshold rule.
    pub const fn resolve_for(self, width: u32, height: u32) -> Self {
        match self {
            Self::Auto => {
                // libjxl threshold: a single 2048×2048 DC group fits
                // any image ≤ 2048² total pixels. Use `u64` to avoid
                // overflow on the 4G × 4G upper bound.
                let pixels = (width as u64).saturating_mul(height as u64);
                if pixels <= 2048u64 * 2048u64 {
                    Self::FullBuffered
                } else {
                    Self::BufferedOutput
                }
            }
            other => other,
        }
    }

    /// Returns `true` if this buffering policy is compatible with
    /// streaming encoding (i.e. the encoder may drop per-DC-group
    /// XYB / quant / mask storage as soon as the corresponding
    /// section is emitted).
    ///
    /// Mirrors the streaming-side of libjxl's
    /// [`CanDoStreamingEncoding`](https://github.com/libjxl/libjxl/blob/main/lib/jxl/enc_frame.cc)
    /// gate: only [`BufferedOutput`](Self::BufferedOutput) and
    /// [`FullStreaming`](Self::FullStreaming) request the per-region
    /// release path. [`Auto`](Self::Auto) is resolved first via
    /// [`Self::resolve_for`] before this check is meaningful.
    pub const fn is_streaming(self) -> bool {
        matches!(self, Self::BufferedOutput | Self::FullStreaming)
    }

    /// Chunk-8c (#11) streaming gate. Returns the buffering policy
    /// to actually use given a caller-requested mode and whether the
    /// butteraugli quantization loop will run on this encode.
    ///
    /// Mirrors libjxl `CanDoStreamingEncoding` in `enc_frame.cc`:
    /// the butteraugli loop reconstructs the whole image multiple
    /// times to evaluate per-block quality and cannot run from a
    /// sliding-window XYB source. When a caller asks for streaming
    /// (`BufferedOutput` / `FullStreaming` / `Auto` resolved to one
    /// of those) **and** the butteraugli loop is active, this
    /// helper returns [`FullBuffered`](Self::FullBuffered) instead
    /// — the encoder runs the loop on a whole-image XYB then
    /// emits the final pass through the buffered-output path.
    ///
    /// Today the butteraugli loop is feature-gated and effort-gated
    /// (off at default effort 7); the typical request path
    /// (`Auto` + default effort) is unaffected. The `Auto`
    /// resolution happens first so the returned variant is always a
    /// concrete level (never `Auto`).
    pub const fn resolve_for_streaming(
        self,
        width: u32,
        height: u32,
        butteraugli_iters: u32,
    ) -> Self {
        let resolved = self.resolve_for(width, height);
        if butteraugli_iters > 0 && resolved.is_streaming() {
            // Downgrade to FullBuffered so the buttloop sees the
            // whole-image XYB it requires. Mirrors libjxl's
            // CanDoStreamingEncoding which returns false on
            // `use_butteraugli_loop`.
            Self::FullBuffered
        } else {
            resolved
        }
    }
}

/// Premultiplied (associated) alpha policy for inputs with an alpha
/// channel.
///
/// Mirrors libjxl `cjxl --premultiply -1|0|1`.
///
/// - [`Off`]: input alpha is straight (unassociated). Color samples
///   were captured without alpha pre-multiplication. **Default.**
/// - [`On`]: input alpha is premultiplied (associated). Color samples
///   were already multiplied by alpha. Standard for GPU pipelines
///   (Skia, Cairo, Metal, Vulkan, Direct2D, Wayland, CompositorAPI).
/// - [`Auto`]: detect from pixels at encode time. The encoder scans
///   the buffer once: if every color sample is ≤ its alpha sample,
///   the data is treated as premultiplied; otherwise straight. The
///   scan is O(N) and runs before the encode loop; for trusted inputs
///   prefer the explicit [`Off`]/[`On`] forms.
///
/// On the [`crate::LossyConfig`] path the encoder requires the
/// unpremultiplication pre-pass (#13) — calling `finish()` on a lossy
/// encode with [`On`] (or [`Auto`] that resolves to premultiplied)
/// returns [`crate::EncodeError::InvalidInput`]. On the [`crate::LosslessConfig`]
/// path the pixels are preserved bit-exactly and the
/// `alpha_associated` header bit is set so the decoder interprets the
/// stored values correctly.
///
/// [`Auto`]: PremultipliedAlphaMode::Auto
/// [`Off`]: PremultipliedAlphaMode::Off
/// [`On`]: PremultipliedAlphaMode::On
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PremultipliedAlphaMode {
    /// **Default.** Straight (unassociated) alpha. Equivalent to
    /// libjxl `--premultiply 0`.
    #[default]
    Off,
    /// Premultiplied (associated) alpha. Equivalent to libjxl
    /// `--premultiply 1`.
    On,
    /// Detect from pixels at encode time. Equivalent to libjxl
    /// `--premultiply -1`. Adds a single O(N) scan over the input
    /// before encoding.
    Auto,
}

impl PremultipliedAlphaMode {
    /// Convert from the libjxl `--premultiply` integer encoding.
    /// `< 0` = [`Auto`](Self::Auto), `0` = [`Off`](Self::Off), `> 0` =
    /// [`On`](Self::On).
    pub const fn from_i8(v: i8) -> Self {
        if v < 0 {
            Self::Auto
        } else if v == 0 {
            Self::Off
        } else {
            Self::On
        }
    }
}

/// Maximum value for [`crate::LossyConfig::with_faster_decoding`] /
/// [`crate::LosslessConfig::with_faster_decoding`]. Matches libjxl
/// `cjxl --faster_decoding 0..4`.
pub const MAX_FASTER_DECODING: u8 = 4;

/// Maximum value for [`crate::LossyConfig::with_progressive_dc`]. Matches
/// libjxl `cjxl --progressive_dc 0..2`.
///
/// 0 = no progressive DC.
/// 1 = one [`LfFrame`](crate::LossyConfig::with_lf_frame) ahead of the
/// main VarDCT frame.
/// 2 = two nested LfFrames (libjxl path; our encoder currently emits a
/// single LfFrame and warns).
pub const MAX_PROGRESSIVE_DC: u8 = 2;
