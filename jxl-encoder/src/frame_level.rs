// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Frame-dependent Annex M limits, observed at successful header emission.
use crate::error::{Error, Result};
use crate::headers::frame_header::{FrameHeader, FrameType, USE_LF_FRAME};

#[derive(Clone, Copy, Debug)]
struct Frame {
    crop: Option<(u32, u32)>,
    upsampling: u32,
    lf_level: u32,
    duration: u32,
}

/// Kept with encoded bytes so discarded trial frames cannot affect the level.
/// Group/entropy writers contain no frame headers and allocate no records.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameSequence {
    frames: Vec<Frame>,
}

impl FrameSequence {
    pub(crate) fn record(&mut self, header: &FrameHeader) -> Result<()> {
        let normal = matches!(
            header.frame_type,
            FrameType::Regular | FrameType::SkipProgressive
        );
        let lf_frame = header.frame_type == FrameType::LfFrame;
        let crop = !lf_frame
            && (header.x0 != 0 || header.y0 != 0 || header.width != 0 || header.height != 0);
        self.frames.try_reserve(1)?;
        self.frames.push(Frame {
            crop: crop.then_some((header.width, header.height)),
            upsampling: if header.flags & USE_LF_FRAME == 0 {
                header.upsampling
            } else {
                1
            },
            lf_level: if lf_frame { header.lf_level } else { 0 },
            duration: if normal && header.have_animation {
                header.duration
            } else {
                0
            },
        });
        Ok(())
    }

    pub(crate) fn append(&mut self, other: &Self) -> Result<()> {
        self.frames.try_reserve(other.frames.len())?;
        self.frames.extend_from_slice(&other.frames);
        Ok(())
    }

    /// Reconstructed pixel dimensions on the frame's LF grid. Spatial
    /// upsampling changes the coded sample count, not this pixel budget.
    /// ReferenceOnly and LF frames have implicit zero duration.
    pub(crate) fn required_level(
        &self,
        width: u32,
        height: u32,
        clock: Option<(u32, u32)>,
    ) -> Result<u8> {
        if clock.is_some_and(|(numerator, denominator)| numerator == 0 || denominator == 0) {
            return Err(Error::InvalidInput(
                "frame sequence has a zero animation clock".into(),
            ));
        }
        let mut level = 5;
        let mut zero_area = 0_u64;
        for frame in &self.frames {
            if !matches!(frame.upsampling, 1 | 2 | 4 | 8) || frame.lf_level > 4 {
                return Err(Error::InvalidInput("invalid emitted frame scale".into()));
            }
            let (w, h) = frame.crop.unwrap_or((width, height));
            let lf_scale = 1_u32 << (3 * frame.lf_level);
            let w = w.div_ceil(lf_scale);
            let h = h.div_ceil(lf_scale);
            crate::headers::file_header::FileHeader::validate_display_dimensions(w, h)?;
            let area = u64::from(w) * u64::from(h);
            if w > 1 << 18 || h > 1 << 18 || area > 1 << 28 {
                level = 10;
            }
            if frame.duration == 0 {
                // Saturation retains the correct > Level-5 verdict; Level 10
                // has no cumulative zero-duration-area cap.
                zero_area = zero_area.saturating_add(area);
                if zero_area > 1 << 28 {
                    level = 10;
                }
            } else {
                let Some((numerator, denominator)) = clock else {
                    return Err(Error::InvalidInput(
                        "frame timing has no animation clock".into(),
                    ));
                };
                if u128::from(frame.duration) * u128::from(denominator) * 120
                    < u128::from(numerator)
                {
                    level = 10;
                }
                zero_area = 0;
            }
        }
        Ok(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(duration: u32) -> FrameHeader {
        let mut frame = FrameHeader::lossless();
        frame.have_animation = true;
        frame.duration = duration;
        frame
    }
    #[test]
    fn consecutive_area_and_duration_boundaries() {
        let mut sequence = FrameSequence::default();
        for _ in 0..16 {
            sequence.record(&frame(0)).unwrap();
        }
        assert_eq!(
            sequence.required_level(4096, 4096, Some((100, 1))).unwrap(),
            5
        );
        sequence.record(&frame(0)).unwrap();
        assert_eq!(
            sequence.required_level(4096, 4096, Some((100, 1))).unwrap(),
            10
        );
        let mut skip = frame(1);
        skip.frame_type = FrameType::SkipProgressive;
        let mut skip_sequence = FrameSequence::default();
        skip_sequence.record(&skip).unwrap();
        assert_eq!(
            skip_sequence.required_level(8, 8, Some((121, 1))).unwrap(),
            10
        );
        let mut reset = FrameSequence::default();
        for _ in 0..2 {
            for _ in 0..16 {
                reset.record(&frame(0)).unwrap();
            }
            reset.record(&frame(1)).unwrap();
        }
        assert_eq!(reset.required_level(4096, 4096, Some((120, 1))).unwrap(), 5);
        assert_eq!(
            reset.required_level(4096, 4096, Some((121, 1))).unwrap(),
            10
        );
        assert_eq!(reset.required_level(4096, 4096, Some((240, 2))).unwrap(), 5);
        assert_eq!(
            reset.required_level(4096, 4096, Some((241, 2))).unwrap(),
            10
        );
    }
    #[test]
    fn crop_scale_hidden_and_trial_composition() {
        let mut full = FrameSequence::default();
        full.record(&frame(0)).unwrap();
        let mut cropped = frame(0);
        cropped.width = 1;
        cropped.height = 1;
        let mut cropped_sequence = FrameSequence::default();
        cropped_sequence.record(&cropped).unwrap();
        for _ in 0..32 {
            full.append(&cropped_sequence).unwrap();
        }
        assert_eq!(full.required_level(4096, 4096, Some((100, 1))).unwrap(), 5);
        let mut hidden = frame(1);
        hidden.frame_type = FrameType::ReferenceOnly;
        let mut hidden_sequence = FrameSequence::default();
        for _ in 0..17 {
            hidden_sequence.record(&hidden).unwrap();
        }
        assert_eq!(
            hidden_sequence
                .required_level(4096, 4096, Some((100, 1)))
                .unwrap(),
            10
        );
        let mut lf = FrameHeader::lf_frame(512, 512, 1);
        let mut auxiliary = FrameSequence::default();
        auxiliary.record(&lf).unwrap();
        assert_eq!(auxiliary.required_level(16384, 16384, None).unwrap(), 5);
        // The LF grid, rather than the full output canvas, sets this area.
        for _ in 1..64 {
            auxiliary.record(&lf).unwrap();
        }
        assert_eq!(auxiliary.required_level(16384, 16384, None).unwrap(), 5);
        auxiliary.record(&lf).unwrap();
        assert_eq!(auxiliary.required_level(16384, 16384, None).unwrap(), 10);
        lf.width = 0;
        lf.height = 0; // LF width hints do not serialize as a crop.
        let mut same = FrameSequence::default();
        same.record(&lf).unwrap();
        assert_eq!(same.required_level(16384, 16384, None).unwrap(), 5);
        let mut scaled = frame(0);
        scaled.upsampling = 8;
        let mut reduced = FrameSequence::default();
        for _ in 0..17 {
            reduced.record(&scaled).unwrap();
        }
        assert_eq!(
            reduced.required_level(4096, 4096, Some((100, 1))).unwrap(),
            10
        );
        scaled.flags |= USE_LF_FRAME;
        let mut implicit = FrameSequence::default();
        for _ in 0..17 {
            implicit.record(&scaled).unwrap();
        }
        assert_eq!(
            implicit.required_level(4096, 4096, Some((100, 1))).unwrap(),
            10
        );
    }
    #[test]
    fn emitted_headers_and_selected_writer_appends_keep_their_sequence() {
        use crate::bit_writer::BitWriter;
        let mut long = BitWriter::new();
        let mut header = frame(0);
        header.all_default_fast_path = false;
        header.is_last = false;
        header.write(&mut long).unwrap();
        long.zero_pad_to_byte();
        let mut compact = BitWriter::new();
        let mut default_header = FrameHeader::lossy();
        default_header.flags = 0;
        default_header.x_qm_scale = 3;
        default_header.gaborish = true;
        default_header.epf_iters = 2;
        default_header.all_default_fast_path = true;
        default_header.write(&mut compact).unwrap();
        assert_eq!(
            compact.bits_written(),
            1,
            "the compact serialization path must actually run"
        );
        let unaligned_header = compact.clone();
        compact.zero_pad_to_byte();
        let mut selected = BitWriter::new();
        for _ in 0..16 {
            selected.append_byte_aligned(&long).unwrap();
        }
        assert_eq!(selected.frame_level(4096, 4096, Some((100, 1))).unwrap(), 5);
        // A rejected seventeen-frame candidate must not affect selected bytes.
        let mut rejected = selected.clone();
        rejected.append_byte_aligned(&compact).unwrap();
        assert_eq!(
            rejected.frame_level(4096, 4096, Some((100, 1))).unwrap(),
            10
        );
        assert_eq!(selected.frame_level(4096, 4096, Some((100, 1))).unwrap(), 5);
        selected.append_unaligned(&unaligned_header).unwrap();
        assert_eq!(
            selected.frame_level(4096, 4096, Some((100, 1))).unwrap(),
            10
        );
        let mut static_header = FrameHeader::lossless();
        static_header.duration = 99; // Not serialized without animation.
        assert!(!static_header.have_animation);
        let mut static_writer = BitWriter::new();
        static_header.write(&mut static_writer).unwrap();
        assert_eq!(static_writer.frame_level(8, 8, None).unwrap(), 5);
        let mut invalid_header = FrameHeader::lossless();
        invalid_header.width = 1;
        invalid_header.height = 1;
        assert!(invalid_header.canvas_size.is_none());
        let mut failed = BitWriter::new();
        for _ in 0..17 {
            assert!(invalid_header.write(&mut failed).is_err());
        }
        assert_eq!(
            failed.frame_level(4096, 4096, None).unwrap(),
            5,
            "failed header writes must not record frames"
        );
        let mut unaligned = BitWriter::new();
        unaligned.write_bit(true).unwrap();
        assert!(unaligned.append_byte_aligned(&long).is_err());
        assert_eq!(unaligned.frame_level(4096, 4096, None).unwrap(), 5);
        assert!(long.append_byte_aligned(&unaligned).is_err());
        assert_eq!(long.frame_level(4096, 4096, Some((100, 1))).unwrap(), 5);
    }

    #[test]
    fn invalid_clock_scale_and_geometry_are_errors() {
        let mut sequence = FrameSequence::default();
        sequence.record(&frame(1)).unwrap();
        assert!(sequence.required_level(1, 1, None).is_err());
        assert!(sequence.required_level(1, 1, Some((0, 1))).is_err());
        assert!(sequence.required_level(1, 1, Some((1, 0))).is_err());
        assert!(sequence.required_level(0, 1, Some((1, 1))).is_err());
        assert!(
            sequence
                .required_level((1 << 30) + 1, 1, Some((1, 1)))
                .is_err()
        );
        for (width, height) in [
            ((1 << 18) + 1, 1),
            (1, (1 << 18) + 1),
            ((1 << 14) + 1, 1 << 14),
        ] {
            let mut spatial = FrameSequence::default();
            spatial.record(&frame(0)).unwrap();
            assert_eq!(spatial.required_level(width, height, None).unwrap(), 10);
        }
        let mut invalid = frame(0);
        invalid.upsampling = 3;
        let mut sequence = FrameSequence::default();
        sequence.record(&invalid).unwrap();
        assert!(sequence.required_level(1, 1, None).is_err());
        invalid.flags |= USE_LF_FRAME;
        let mut implicit = FrameSequence::default();
        implicit.record(&invalid).unwrap();
        assert_eq!(implicit.required_level(1, 1, None).unwrap(), 5);
        invalid.flags = 0;
        invalid.upsampling = 1;
        invalid.frame_type = FrameType::LfFrame;
        invalid.lf_level = 5;
        let mut sequence = FrameSequence::default();
        sequence.record(&invalid).unwrap();
        assert!(sequence.required_level(1, 1, None).is_err());
    }
}
