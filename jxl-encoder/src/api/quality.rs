// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossy quality specification (\`Quality\`) and quality<->distance conversion.

use super::*;

/// Quality specification for lossy encoding.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Quality {
    /// Butteraugli distance (1.0 = high quality, lower = better).
    Distance(f32),
    /// Percentage scale (0–100, 100 = mathematically lossless, invalid for lossy).
    Percent(u32),
}

impl Quality {
    /// Convert to butteraugli distance.
    pub(crate) fn to_distance(self) -> core::result::Result<f32, EncodeError> {
        match self {
            Self::Distance(d) => {
                if d <= 0.0 {
                    return Err(EncodeError::InvalidConfig {
                        message: format!("lossy distance must be > 0.0, got {d}"),
                    });
                }
                Ok(d)
            }
            Self::Percent(q) => {
                if q >= 100 {
                    return Err(EncodeError::InvalidConfig {
                        message: "quality 100 is lossless; use LosslessConfig instead".into(),
                    });
                }
                Ok(percent_to_distance(q))
            }
        }
    }
}

fn percent_to_distance(quality: u32) -> f32 {
    if quality >= 100 {
        0.0
    } else if quality >= 90 {
        (100 - quality) as f32 / 10.0
    } else if quality >= 70 {
        1.0 + (90 - quality) as f32 / 20.0
    } else {
        2.0 + (70 - quality) as f32 / 10.0
    }
}

/// Convert quality on 0–100 scale to JXL butteraugli distance.
///
/// Matches the jxl-encoder's own `percent_to_distance` piecewise mapping:
/// - 90–100 → distance 0.0–1.0  (perceptually lossless zone)
/// - 70–90  → distance 1.0–2.0  (high quality)
/// - 0–70   → distance 2.0–9.0  (lower quality)
#[must_use]
pub fn quality_to_distance(quality: f32) -> f32 {
    let q = quality.clamp(0.0, 100.0);
    if q >= 100.0 {
        0.0
    } else if q >= 90.0 {
        (100.0 - q) / 10.0
    } else if q >= 70.0 {
        1.0 + (90.0 - q) / 20.0
    } else {
        2.0 + (70.0 - q) / 10.0
    }
}

/// Map generic quality (libjpeg-turbo scale) to JXL native quality.
///
/// Calibrated on CID22-512 corpus (209 images) to produce the same median
/// SSIMULACRA2 as libjpeg-turbo at each quality level. The native quality
/// is then mapped to Butteraugli distance by [`quality_to_distance`].
#[must_use]
pub fn calibrated_jxl_quality(generic_q: f32) -> f32 {
    let clamped = generic_q.clamp(0.0, 100.0);
    const TABLE: &[(f32, f32)] = &[
        (5.0, 5.0),
        (10.0, 5.0),
        (15.0, 5.0),
        (20.0, 5.0),
        (25.0, 9.3),
        (30.0, 22.7),
        (35.0, 33.0),
        (40.0, 38.8),
        (45.0, 43.8),
        (50.0, 48.5),
        (55.0, 51.9),
        (60.0, 55.1),
        (65.0, 58.0),
        (70.0, 61.3),
        (72.0, 63.2),
        (75.0, 65.5),
        (78.0, 67.9),
        (80.0, 69.1),
        (82.0, 71.8),
        (85.0, 76.1),
        (87.0, 79.3),
        (90.0, 84.2),
        (92.0, 86.9),
        (95.0, 91.2),
        (97.0, 92.8),
        (99.0, 93.8),
    ];
    interp_quality(TABLE, clamped)
}

/// Piecewise linear interpolation with clamping at table bounds.
pub(crate) fn interp_quality(table: &[(f32, f32)], x: f32) -> f32 {
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 1..table.len() {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}
