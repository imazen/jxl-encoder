// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// This module wraps the battle-tested `ultrahdr-core` gain-map math behind
// a JXL-shaped public API. The `ultrahdr-core` crate authors retain copyright
// on the underlying gain-map computation; this module just adapts.

//! Ultra HDR gain-map encoding for JPEG XL containers.
//!
//! This module is feature-gated behind `hdr-gainmap`. It provides two pieces:
//!
//! 1. **Chunk 3 — typed `GainMapBundle`**: a Rust struct mirroring libjxl's
//!    [`JxlGainMapBundle`] C struct, with a [`GainMapBundle::serialize`]
//!    method that emits bytes matching `JxlGainMapWriteBundle` exactly
//!    (`libjxl/lib/extras/gain_map.cc:83`). The output is the `jhgm` box
//!    payload — wrap it with [`append_gain_map_bundle`] (or the lower-level
//!    [`crate::container::append_gain_map_box`]) to produce a full container.
//!
//! 2. **Chunk 4 — [`HdrFromSdrRequest`]**: an end-to-end "give me HDR + SDR
//!    pixel buffers, get one JXL file" entry point that calls
//!    `ultrahdr_core::gainmap::compute_gainmap` to derive a gain map from
//!    the SDR/HDR pair, encodes the SDR base image as a JXL codestream
//!    via [`crate::api::LossyConfig`], encodes the gain-map plane as a
//!    second lossless JXL codestream, serializes the ISO 21496-1 metadata
//!    via `ultrahdr_core::serialize_iso21496_fmt`, and muxes everything
//!    into a single container.
//!
//! The shipped file renders as the SDR base image on standard JXL decoders
//! and as the reconstructed HDR image on decoders that understand the
//! `jhgm` Ultra HDR box.
//!
//! [`JxlGainMapBundle`]: https://github.com/libjxl/libjxl/blob/main/lib/include/jxl/gain_map.h

mod bundle;
mod from_sdr;

pub use bundle::{GainMapBundle, append_gain_map_bundle};
pub use from_sdr::{
    GainMapEncoding, HdrColorEncoding, HdrFromSdrRequest, HdrImage, HdrPixelLayout,
};

// Re-export the ultrahdr-core types callers need to construct a request /
// a bundle. Re-exporting (rather than asking callers to add their own
// `ultrahdr-core` dependency) keeps the public surface mono-rooted and
// avoids the version-pin trap of two crates picking different
// `ultrahdr-core` versions.
pub use ultrahdr_core::GainMapParams as Iso21496Metadata;
pub use ultrahdr_core::Iso21496Format;
pub use ultrahdr_core::gainmap::GainMapConfig;
pub use ultrahdr_core::{ColorPrimaries as UhdrPrimaries, TransferFunction as UhdrTransfer};
