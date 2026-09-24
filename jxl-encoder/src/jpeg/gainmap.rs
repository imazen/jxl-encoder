// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Expose ISO 21496-1 secondary JPEGs while retaining the original JBRD tail.

use alloc::sync::Arc;
use alloc::vec::Vec;
use enough::Stop;
use zenjpeg::container::marker::{self, MarkerKind};

use super::{JpegData, JpegError, encode_jpeg_to_jxl_with_effort_stop, read_jpeg_with_stop};
use crate::budget::{MemoryBudget, vec_with_capacity_fallible};
use crate::error::{Error, Result};
use crate::hdr::GainMapBundle;

// ISO 21496-1 JPEG APP2 namespace, including its terminating NUL. The bytes
// after it have exactly the JxlGainMapBundle gain_map_metadata wire format.
const ISO_NAMESPACE: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";

fn invalid(message: impl Into<alloc::string::String>) -> Error {
    Error::InvalidInput(message.into())
}

fn poll(stop: Option<&dyn Stop>) -> Result<()> {
    if let Some(stop) = stop {
        stop.check().map_err(|_| Error::Cancelled)?;
    }
    Ok(())
}

/// The parser already separates the primary EOI from its tail. Identify a
/// secondary by its own ISO metadata, not by position alone: MPF files can
/// also contain thumbnails, stereo views, and vendor trailers. Walk complete
/// SOI..EOI ranges so an embedded thumbnail is not mistaken for another image.
struct IsoGainMap<'a> {
    jpeg: &'a [u8],
    metadata: &'a [u8],
}

fn find_iso_gain_map<'a>(
    tail: &'a [u8],
    stop: Option<&dyn Stop>,
) -> Result<Option<IsoGainMap<'a>>> {
    let mut remaining = tail;
    let mut found = None;
    while let Some(start) = remaining.windows(2).position(|bytes| bytes == [0xff, 0xd8]) {
        poll(stop)?;
        remaining = &remaining[start..];
        let mut metadata = None;
        let mut end = None;
        for span in marker::iter(remaining) {
            poll(stop)?;
            if span.kind == MarkerKind::Eoi {
                end = Some(span.offset + span.length);
                break;
            }
            if span.kind != MarkerKind::App(2) {
                continue;
            }
            let Some(payload) = span.payload.strip_prefix(ISO_NAMESPACE) else {
                continue;
            };
            // A version-only primary signal is not gain-map metadata.
            if payload.len() == 4 {
                continue;
            }
            ultrahdr_core::parse_iso21496_fmt(payload, ultrahdr_core::Iso21496Format::JxlJhgm)
                .map_err(|e| invalid(format!("invalid gain-map ISO metadata: {e}")))?;
            if metadata.replace(payload).is_some() {
                return Err(invalid("duplicate ISO metadata in gain-map JPEG"));
            }
        }
        if let Some(metadata) = metadata {
            let end = end.ok_or_else(|| invalid("truncated ISO gain-map JPEG"))?;
            if found.is_some() {
                return Err(invalid("multiple ISO gain maps in JPEG tail"));
            }
            found = Some(IsoGainMap {
                jpeg: &remaining[..end],
                metadata,
            });
        }
        // Arbitrary non-JPEG trailer bytes remain opaque JBRD data, including
        // accidental SOI byte pairs that have no complete image after them.
        remaining = &remaining[end.unwrap_or(2)..];
    }
    Ok(found)
}

pub(super) fn append(
    jpeg: &JpegData,
    container: Vec<u8>,
    effort: u8,
    stop: Option<&dyn Stop>,
    budget: Option<&Arc<MemoryBudget>>,
    max_pixels: Option<u64>,
) -> Result<Vec<u8>> {
    let Some(IsoGainMap {
        jpeg: secondary,
        metadata,
    }) = find_iso_gain_map(&jpeg.tail_data, stop)?
    else {
        return Ok(container);
    };
    poll(stop)?;
    // Keep the primary output accounted for while the secondary encoder runs.
    let _container_guard = MemoryBudget::reserve_opt(budget, container.len() as u64)?;
    let gain_map =
        read_jpeg_with_stop(secondary, max_pixels, stop, budget).map_err(|e| match e {
            JpegError::Cancelled => Error::Cancelled,
            JpegError::ResourceLimit(_) if budget.is_some() => {
                let budget = budget.unwrap();
                Error::AllocationLimit {
                    requested: 0,
                    used: budget.used(),
                    cap: budget.cap(),
                }
            }
            other => invalid(format!("gain-map JPEG: {other}")),
        })?;
    let codestream = encode_jpeg_to_jxl_with_effort_stop(&gain_map, effort, stop, budget)?;
    drop(gain_map);
    poll(stop)?;
    // Copy the ISO rational representation verbatim. Parsing and serializing
    // through floating-point parameters would change the metadata's precision.
    let payload_size = metadata
        .len()
        .checked_add(codestream.len())
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| invalid("gain-map bundle size overflow"))?;
    let output_size = container
        .len()
        .checked_add(payload_size)
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| invalid("gain-map container size overflow"))?;
    let box_size = u32::try_from(payload_size + 8)
        .map_err(|_| invalid("gain-map box exceeds 32-bit box size"))?;
    let _bundle_guard =
        MemoryBudget::reserve_opt(budget, metadata.len() as u64 + output_size as u64)?;
    let fallible = budget.is_some_and(|b| b.is_fallible());
    let mut owned_metadata = vec_with_capacity_fallible(fallible, metadata.len())?;
    owned_metadata.extend_from_slice(metadata);
    let bundle = GainMapBundle::new(owned_metadata, codestream);
    let mut output = vec_with_capacity_fallible(fallible, output_size)?;
    output.extend_from_slice(&container);
    output.extend_from_slice(&box_size.to_be_bytes());
    output.extend_from_slice(b"jhgm");
    bundle
        .serialize_into_reserved(&mut output)
        .map_err(|e| invalid(e.to_string()))?;
    poll(stop)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app2(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0xff, 0xe2];
        out.extend_from_slice(&((ISO_NAMESPACE.len() + payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(ISO_NAMESPACE);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn iso_gain_map_parser_polls_inside_marker_walk() {
        struct CancelAfter(core::sync::atomic::AtomicUsize);
        impl Stop for CancelAfter {
            fn check(&self) -> core::result::Result<(), enough::StopReason> {
                if self.0.fetch_sub(1, core::sync::atomic::Ordering::Relaxed) == 0 {
                    Err(enough::StopReason::Cancelled)
                } else {
                    Ok(())
                }
            }
        }
        let mut jpeg = vec![0xff, 0xd8];
        for _ in 0..100 {
            jpeg.extend(app2(&[0; 4]));
        }
        jpeg.extend([0xff, 0xd9]);
        assert!(matches!(
            find_iso_gain_map(
                &jpeg,
                Some(&CancelAfter(core::sync::atomic::AtomicUsize::new(3)))
            ),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn iso_gain_map_parser_rejects_ambiguous_and_truncated_metadata() {
        let metadata = ultrahdr_core::serialize_iso21496_fmt(
            &ultrahdr_core::GainMapParams::default(),
            ultrahdr_core::Iso21496Format::JxlJhgm,
        );
        let mut jpeg = vec![0xff, 0xd8];
        jpeg.extend(app2(&metadata));
        assert!(
            find_iso_gain_map(&jpeg, None)
                .err()
                .unwrap()
                .to_string()
                .contains("truncated")
        );
        let mut duplicate = jpeg.clone();
        duplicate.extend(app2(&metadata));
        duplicate.extend([0xff, 0xd9]);
        assert!(
            find_iso_gain_map(&duplicate, None)
                .err()
                .unwrap()
                .to_string()
                .contains("duplicate")
        );
        jpeg.extend([0xff, 0xd9]);
        let found = find_iso_gain_map(&jpeg, None).unwrap().unwrap();
        assert_eq!(found.jpeg, jpeg);
        assert_eq!(found.metadata, metadata);
        let mut multiple = jpeg.clone();
        multiple.extend(&jpeg);
        assert!(
            find_iso_gain_map(&multiple, None)
                .err()
                .unwrap()
                .to_string()
                .contains("multiple")
        );
        let mut malformed = vec![0xff, 0xd8];
        malformed.extend(app2(&[0; 5]));
        malformed.extend([0xff, 0xd9]);
        assert!(
            find_iso_gain_map(&malformed, None)
                .err()
                .unwrap()
                .to_string()
                .contains("invalid gain-map ISO metadata")
        );
        let mut signal = vec![0xff, 0xd8];
        signal.extend(app2(&[0; 4]));
        signal.extend([0xff, 0xd9]);
        assert!(find_iso_gain_map(&signal, None).unwrap().is_none());
    }
}
