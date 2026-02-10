//! zencodec-types trait implementations for jxl-encoder.
//!
//! Provides [`JxlEncoding`] and [`JxlEncodeJob`] types that implement the
//! [`Encoding`] / [`EncodingJob`] traits from zencodec-types, wrapping the
//! native jxl-encoder API.
//!
//! This is an encode-only crate — no [`Decoding`] implementation.

use imgref::ImgRef;
use rgb::{Gray, Rgb, Rgba};
use zencodec_types::{
    EncodeOutput, Encoding, EncodingJob, ImageFormat, ImageMetadata, Stop,
};

use crate::api::{
    EncodeError, LosslessConfig, LossyConfig, PixelLayout,
};

// ── Config enum ─────────────────────────────────────────────────────────────

/// Internal: lossy or lossless JXL config.
#[derive(Clone, Debug)]
enum JxlConfig {
    Lossy(LossyConfig),
    Lossless(LosslessConfig),
}

// ── Encoding ────────────────────────────────────────────────────────────────

/// JPEG XL encoder configuration implementing [`Encoding`].
///
/// Wraps [`LossyConfig`] or [`LosslessConfig`] with limit fields for the
/// trait interface. Defaults to lossy at distance 1.0.
///
/// # Examples
///
/// ```rust,ignore
/// use zencodec_types::Encoding;
/// use jxl_encoder::JxlEncoding;
///
/// let enc = JxlEncoding::lossy(1.0)
///     .with_effort(7);
/// ```
#[derive(Clone, Debug)]
pub struct JxlEncoding {
    config: JxlConfig,
    limit_pixels: Option<u64>,
    limit_memory: Option<u64>,
    limit_output: Option<u64>,
}

impl JxlEncoding {
    /// Create a lossy encoder config with the given butteraugli distance.
    ///
    /// Distance 1.0 is high quality, lower is better.
    #[must_use]
    pub fn lossy(distance: f32) -> Self {
        Self {
            config: JxlConfig::Lossy(LossyConfig::new(distance)),
            limit_pixels: None,
            limit_memory: None,
            limit_output: None,
        }
    }

    /// Create a lossless encoder config.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            config: JxlConfig::Lossless(LosslessConfig::new()),
            limit_pixels: None,
            limit_memory: None,
            limit_output: None,
        }
    }

    /// Access the underlying lossy config (if lossy mode).
    #[must_use]
    pub fn lossy_config(&self) -> Option<&LossyConfig> {
        match &self.config {
            JxlConfig::Lossy(c) => Some(c),
            JxlConfig::Lossless(_) => None,
        }
    }

    /// Access the underlying lossless config (if lossless mode).
    #[must_use]
    pub fn lossless_config(&self) -> Option<&LosslessConfig> {
        match &self.config {
            JxlConfig::Lossy(_) => None,
            JxlConfig::Lossless(c) => Some(c),
        }
    }
}

impl Default for JxlEncoding {
    fn default() -> Self {
        Self::lossy(1.0)
    }
}

impl Encoding for JxlEncoding {
    type Error = EncodeError;
    type Job<'a> = JxlEncodeJob<'a>;

    fn with_quality(mut self, quality: f32) -> Self {
        // Map 0-100 quality to butteraugli distance.
        // 100 → lossless, 0 → very lossy (high distance).
        if quality >= 100.0 {
            self.config = JxlConfig::Lossless(LosslessConfig::new());
        } else {
            let distance = percent_to_distance(quality);
            self.config = match self.config {
                JxlConfig::Lossy(c) => JxlConfig::Lossy(LossyConfig::new(distance).with_effort(c.effort())),
                JxlConfig::Lossless(_) => JxlConfig::Lossy(LossyConfig::new(distance)),
            };
        }
        self
    }

    fn with_effort(mut self, effort: u32) -> Self {
        let effort_u8 = (effort.min(10)) as u8;
        self.config = match self.config {
            JxlConfig::Lossy(c) => JxlConfig::Lossy(c.with_effort(effort_u8)),
            JxlConfig::Lossless(c) => JxlConfig::Lossless(c.with_effort(effort_u8)),
        };
        self
    }

    fn with_lossless(mut self, lossless: bool) -> Self {
        if lossless {
            let effort = match &self.config {
                JxlConfig::Lossy(c) => c.effort(),
                JxlConfig::Lossless(c) => c.effort(),
            };
            self.config = JxlConfig::Lossless(LosslessConfig::new().with_effort(effort));
        } else {
            let effort = match &self.config {
                JxlConfig::Lossy(c) => c.effort(),
                JxlConfig::Lossless(c) => c.effort(),
            };
            self.config = JxlConfig::Lossy(LossyConfig::new(1.0).with_effort(effort));
        }
        self
    }

    fn with_alpha_quality(self, _quality: f32) -> Self {
        // JXL handles alpha uniformly; no separate quality control.
        self
    }

    fn with_limit_pixels(mut self, max: u64) -> Self {
        self.limit_pixels = Some(max);
        self
    }

    fn with_limit_memory(mut self, bytes: u64) -> Self {
        self.limit_memory = Some(bytes);
        self
    }

    fn with_limit_output(mut self, bytes: u64) -> Self {
        self.limit_output = Some(bytes);
        self
    }

    fn job(&self) -> JxlEncodeJob<'_> {
        JxlEncodeJob {
            config: self,
            stop: None,
            icc: None,
            exif: None,
            xmp: None,
            limit_pixels: None,
            limit_memory: None,
        }
    }
}

// ── Encode job ──────────────────────────────────────────────────────────────

/// Per-operation JXL encode job.
///
/// Created by [`JxlEncoding::job()`]. Borrows temporary data (stop token,
/// metadata) and is consumed by terminal encode methods.
pub struct JxlEncodeJob<'a> {
    config: &'a JxlEncoding,
    stop: Option<&'a dyn Stop>,
    icc: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
    limit_pixels: Option<u64>,
    limit_memory: Option<u64>,
}

impl<'a> JxlEncodeJob<'a> {
    /// Common encode path for all pixel types.
    fn do_encode(
        self,
        pixels: &[u8],
        layout: PixelLayout,
        w: u32,
        h: u32,
    ) -> Result<EncodeOutput, EncodeError> {
        // Build metadata
        let meta;
        let has_meta = self.icc.is_some() || self.exif.is_some() || self.xmp.is_some();
        if has_meta {
            let mut m = crate::api::ImageMetadata::new();
            if let Some(icc) = self.icc {
                m = m.with_icc_profile(icc);
            }
            if let Some(exif) = self.exif {
                m = m.with_exif(exif);
            }
            if let Some(xmp) = self.xmp {
                m = m.with_xmp(xmp);
            }
            meta = Some(m);
        } else {
            meta = None;
        }

        // Build limits
        let limits;
        let has_limits = self.limit_pixels.is_some()
            || self.limit_memory.is_some()
            || self.config.limit_pixels.is_some()
            || self.config.limit_memory.is_some();
        if has_limits {
            let mut l = crate::api::Limits::new();
            if let Some(p) = self.limit_pixels.or(self.config.limit_pixels) {
                l = l.with_max_pixels(p);
            }
            if let Some(m) = self.limit_memory.or(self.config.limit_memory) {
                l = l.with_max_memory_bytes(m);
            }
            limits = Some(l);
        } else {
            limits = None;
        }

        // Build request and encode
        let data = match &self.config.config {
            JxlConfig::Lossy(cfg) => {
                let mut req = cfg.encode_request(w, h, layout);
                if let Some(ref m) = meta {
                    req = req.with_metadata(m);
                }
                if let Some(ref l) = limits {
                    req = req.with_limits(l);
                }
                if let Some(stop) = self.stop {
                    req = req.with_stop(stop);
                }
                req.encode(pixels).map_err(|e| e.into_inner())?
            }
            JxlConfig::Lossless(cfg) => {
                let mut req = cfg.encode_request(w, h, layout);
                if let Some(ref m) = meta {
                    req = req.with_metadata(m);
                }
                if let Some(ref l) = limits {
                    req = req.with_limits(l);
                }
                if let Some(stop) = self.stop {
                    req = req.with_stop(stop);
                }
                req.encode(pixels).map_err(|e| e.into_inner())?
            }
        };

        Ok(EncodeOutput::new(data, ImageFormat::Jxl))
    }
}

impl<'a> EncodingJob<'a> for JxlEncodeJob<'a> {
    type Error = EncodeError;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: &'a ImageMetadata<'a>) -> Self {
        if let Some(icc) = meta.icc_profile {
            self.icc = Some(icc);
        }
        if let Some(exif) = meta.exif {
            self.exif = Some(exif);
        }
        if let Some(xmp) = meta.xmp {
            self.xmp = Some(xmp);
        }
        self
    }

    fn with_icc(mut self, icc: &'a [u8]) -> Self {
        self.icc = Some(icc);
        self
    }

    fn with_exif(mut self, exif: &'a [u8]) -> Self {
        self.exif = Some(exif);
        self
    }

    fn with_xmp(mut self, xmp: &'a [u8]) -> Self {
        self.xmp = Some(xmp);
        self
    }

    fn with_limit_pixels(mut self, max: u64) -> Self {
        self.limit_pixels = Some(max);
        self
    }

    fn with_limit_memory(mut self, bytes: u64) -> Self {
        self.limit_memory = Some(bytes);
        self
    }

    fn encode_rgb8(self, img: ImgRef<'_, Rgb<u8>>) -> Result<EncodeOutput, Self::Error> {
        let (buf, w, h) = img.to_contiguous_buf();
        let bytes = rgb_to_bytes(&buf);
        self.do_encode(&bytes, PixelLayout::Rgb8, w as u32, h as u32)
    }

    fn encode_rgba8(self, img: ImgRef<'_, Rgba<u8>>) -> Result<EncodeOutput, Self::Error> {
        let (buf, w, h) = img.to_contiguous_buf();
        let bytes = rgba_to_bytes(&buf);
        self.do_encode(&bytes, PixelLayout::Rgba8, w as u32, h as u32)
    }

    fn encode_gray8(self, img: ImgRef<'_, Gray<u8>>) -> Result<EncodeOutput, Self::Error> {
        let (buf, w, h) = img.to_contiguous_buf();

        // Lossless (modular) supports Gray8 natively.
        // Lossy (VarDCT) does not — expand to RGB.
        match &self.config.config {
            JxlConfig::Lossless(_) => {
                let bytes: Vec<u8> = buf.iter().map(|g| g.value()).collect();
                self.do_encode(&bytes, PixelLayout::Gray8, w as u32, h as u32)
            }
            JxlConfig::Lossy(_) => {
                let bytes = gray_to_rgb_bytes(&buf);
                self.do_encode(&bytes, PixelLayout::Rgb8, w as u32, h as u32)
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Map 0-100 quality percentage to butteraugli distance.
///
/// Uses the same mapping as the jxl-encoder API's `Quality::Percent`.
fn percent_to_distance(quality: f32) -> f32 {
    let q = quality.clamp(0.0, 99.9) as u32;
    if q >= 90 {
        (100 - q) as f32 / 10.0
    } else if q >= 70 {
        1.0 + (90 - q) as f32 / 20.0
    } else {
        2.0 + (70 - q) as f32 / 10.0
    }
}

/// Convert Rgb<u8> slice to flat bytes.
fn rgb_to_bytes(pixels: &[Rgb<u8>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pixels.len() * 3);
    for p in pixels {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
    }
    bytes
}

/// Expand Gray<u8> to flat RGB bytes (R=G=B=gray).
fn gray_to_rgb_bytes(pixels: &[Gray<u8>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pixels.len() * 3);
    for g in pixels {
        let v = g.value();
        bytes.push(v);
        bytes.push(v);
        bytes.push(v);
    }
    bytes
}

/// Convert Rgba<u8> slice to flat bytes.
fn rgba_to_bytes(pixels: &[Rgba<u8>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pixels.len() * 4);
    for p in pixels {
        bytes.push(p.r);
        bytes.push(p.g);
        bytes.push(p.b);
        bytes.push(p.a);
    }
    bytes
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use imgref::Img;
    use zencodec_types::Encoding;

    #[test]
    fn encoding_lossy_default() {
        let enc = JxlEncoding::lossy(1.0);
        let pixels = vec![Rgb { r: 128, g: 64, b: 32 }; 64];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
        assert_eq!(output.format(), ImageFormat::Jxl);
        // Verify JXL signature
        assert_eq!(&output.bytes()[0..2], &[0xFF, 0x0A]);
    }

    #[test]
    fn encoding_lossless() {
        let enc = JxlEncoding::lossless();
        let pixels = vec![Rgb { r: 100, g: 200, b: 50 }; 16];
        let img = Img::new(pixels, 4, 4);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn encoding_with_quality() {
        let enc = JxlEncoding::default().with_quality(80.0);
        let pixels = vec![Rgb { r: 0, g: 0, b: 0 }; 64];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn encoding_quality_100_becomes_lossless() {
        let enc = JxlEncoding::default().with_quality(100.0);
        assert!(enc.lossless_config().is_some());
    }

    #[test]
    fn encoding_gray8() {
        let enc = JxlEncoding::lossy(2.0);
        let pixels = vec![Gray::new(128u8); 64];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_gray8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn encoding_rgba8() {
        let enc = JxlEncoding::lossy(1.0);
        let pixels = vec![
            Rgba {
                r: 100,
                g: 150,
                b: 200,
                a: 128,
            };
            64
        ];
        let img = Img::new(pixels, 8, 8);
        let output = enc.encode_rgba8(img.as_ref()).unwrap();
        assert!(!output.bytes().is_empty());
    }

    #[test]
    fn encoding_with_lossless_toggle() {
        let enc = JxlEncoding::lossy(1.0)
            .with_effort(5)
            .with_lossless(true);
        assert!(enc.lossless_config().is_some());

        let enc2 = enc.with_lossless(false);
        assert!(enc2.lossy_config().is_some());
    }
}
