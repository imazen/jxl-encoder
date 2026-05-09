// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Modular channel and image types.
//!
//! In modular mode, each channel stores signed 32-bit integers representing
//! either raw pixel values or prediction residuals.

use crate::error::{Error, Result};

/// A single channel in a modular image.
///
/// Channels store i32 values, which can represent:
/// - Raw pixel values (0-255 for 8-bit, 0-65535 for 16-bit)
/// - Prediction residuals (can be negative)
/// - Transformed values (after RCT, Squeeze, etc.)
#[derive(Debug, Clone)]
pub struct Channel {
    /// Pixel data stored in row-major order.
    data: Vec<i32>,
    /// Channel width.
    width: usize,
    /// Channel height.
    height: usize,
    /// Horizontal subsampling shift (0 = no subsampling).
    pub hshift: u32,
    /// Vertical subsampling shift (0 = no subsampling).
    pub vshift: u32,
    /// Original color component index (-1 = unset).
    /// Used to look up per-component quantization tables in lossy modular encoding.
    /// Set by LfFrame: Y=0, X=1, B-Y=2. Propagated through Squeeze transforms.
    pub component: i32,
}

impl Channel {
    /// Creates a new channel filled with zeros.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        Self::new_with_budget(width, height, None)
    }

    /// Creates a new channel filled with zeros, charging the
    /// allocation against an optional [`MemoryBudget`].
    ///
    /// `budget = None` is equivalent to [`Channel::new`] and pays
    /// essentially nothing at runtime — a single cold predicted-not-
    /// taken atomic load.
    pub(crate) fn new_with_budget(
        width: usize,
        height: usize,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let size = width
            .checked_mul(height)
            .ok_or(Error::InvalidImageDimensions(width, height))?;

        // Channel data is `Vec<i32>` — 4 bytes per entry.
        let bytes = (size as u64).saturating_mul(core::mem::size_of::<i32>() as u64);
        crate::budget::MemoryBudget::reserve_permanent_opt(budget, bytes)?;

        let mut data = Vec::new();
        data.try_reserve_exact(size)?;
        data.resize(size, 0);

        Ok(Self {
            data,
            width,
            height,
            hshift: 0,
            vshift: 0,
            component: -1,
        })
    }

    /// Creates a channel from existing data.
    pub fn from_vec(data: Vec<i32>, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        if data.len() != width * height {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        Ok(Self {
            data,
            width,
            height,
            hshift: 0,
            vshift: 0,
            component: -1,
        })
    }

    /// Returns the width of the channel.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the channel.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the total number of pixels.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the channel has no pixels.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the pixel at (x, y).
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> i32 {
        debug_assert!(x < self.width && y < self.height);
        self.data[y * self.width + x]
    }

    /// Sets the pixel at (x, y).
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: i32) {
        debug_assert!(x < self.width && y < self.height);
        self.data[y * self.width + x] = value;
    }

    /// Returns a reference to a row.
    #[inline]
    pub fn row(&self, y: usize) -> &[i32] {
        debug_assert!(y < self.height);
        let start = y * self.width;
        &self.data[start..start + self.width]
    }

    /// Returns a mutable reference to a row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [i32] {
        debug_assert!(y < self.height);
        let start = y * self.width;
        &mut self.data[start..start + self.width]
    }

    /// Returns a reference to the underlying data.
    #[inline]
    pub fn data(&self) -> &[i32] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [i32] {
        &mut self.data
    }

    /// Gets a pixel with boundary handling (returns 0 outside bounds).
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> i32 {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            0
        } else {
            self.data[y as usize * self.width + x as usize]
        }
    }

    /// Extracts a region from this channel, accounting for hshift/vshift.
    ///
    /// The rect is specified in full-resolution image coordinates. It is
    /// downshifted by `hshift`/`vshift` and clamped to channel bounds.
    /// Returns `None` if the shifted region has zero area.
    pub fn extract_shifted_region(
        &self,
        rect_x0: usize,
        rect_y0: usize,
        rect_xsize: usize,
        rect_ysize: usize,
    ) -> Option<Channel> {
        let x0 = rect_x0 >> self.hshift;
        let y0 = rect_y0 >> self.vshift;
        let xsize = (rect_xsize >> self.hshift).min(self.width.saturating_sub(x0));
        let ysize = (rect_ysize >> self.vshift).min(self.height.saturating_sub(y0));

        if xsize == 0 || ysize == 0 {
            return None;
        }

        let mut data = Vec::with_capacity(xsize * ysize);
        for y in 0..ysize {
            for x in 0..xsize {
                data.push(self.get(x0 + x, y0 + y));
            }
        }

        let mut ch = Channel::from_vec(data, xsize, ysize).ok()?;
        ch.hshift = self.hshift;
        ch.vshift = self.vshift;
        ch.component = self.component;
        Some(ch)
    }

    /// Extracts a grid cell region matching the decoder's get_grid_rect logic.
    ///
    /// Given a group position (gx, gy) and group_dim (the image-level group size,
    /// typically 256), computes the channel-space sub-region for this group.
    ///
    /// This matches jxl-rs `ModularChannel::get_grid_rect`:
    ///   grid_dim = (group_dim >> hshift, group_dim >> vshift)
    ///   bx = gx * grid_dim.0, by = gy * grid_dim.1
    ///   size = (min(chan_w - bx, grid_dim.0), min(chan_h - by, grid_dim.1))
    ///
    /// Returns None if the sub-region has zero area.
    pub fn extract_grid_cell(&self, gx: usize, gy: usize, group_dim: usize) -> Option<Channel> {
        let grid_w = group_dim >> self.hshift;
        let grid_h = group_dim >> self.vshift;

        if grid_w == 0 || grid_h == 0 {
            return None;
        }

        let bx = gx * grid_w;
        let by = gy * grid_h;

        if bx >= self.width || by >= self.height {
            return None;
        }

        let xsize = (self.width - bx).min(grid_w);
        let ysize = (self.height - by).min(grid_h);

        if xsize == 0 || ysize == 0 {
            return None;
        }

        let mut data = Vec::with_capacity(xsize * ysize);
        for y in 0..ysize {
            for x in 0..xsize {
                data.push(self.get(bx + x, by + y));
            }
        }

        let mut ch = Channel::from_vec(data, xsize, ysize).ok()?;
        ch.hshift = self.hshift;
        ch.vshift = self.vshift;
        ch.component = self.component;
        Some(ch)
    }

    /// Gets a pixel, clamping coordinates to valid range.
    #[inline]
    pub fn get_clamped_to_edge(&self, x: isize, y: isize) -> i32 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }
}

/// A modular image consisting of multiple channels.
///
/// For RGB images, this typically contains 3 channels.
/// For RGBA, 4 channels (with alpha stored last).
#[derive(Debug, Clone)]
pub struct ModularImage {
    /// The channels in this image.
    pub channels: Vec<Channel>,
    /// Bit depth of the original image.
    pub bit_depth: u32,
    /// Whether the image is originally grayscale.
    pub is_grayscale: bool,
    /// Whether the image has alpha.
    pub has_alpha: bool,
}

impl ModularImage {
    /// Creates a new modular image from 8-bit RGB data.
    pub fn from_rgb8(data: &[u8], width: usize, height: usize) -> Result<Self> {
        Self::from_rgb8_with_budget(data, width, height, None)
    }

    /// Build a 3-channel modular image from interleaved RGB8, charging
    /// the resulting channel allocations against an optional budget.
    pub(crate) fn from_rgb8_with_budget(
        data: &[u8],
        width: usize,
        height: usize,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 3,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let mut channels = Vec::with_capacity(3);
        for c in 0..3 {
            let mut channel = Channel::new_with_budget(width, height, budget)?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3 + c;
                    channel.set(x, y, data[idx] as i32);
                }
            }
            channels.push(channel);
        }

        Ok(Self {
            channels,
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: false,
        })
    }

    /// Creates a new modular image from 8-bit RGBA data.
    pub fn from_rgba8(data: &[u8], width: usize, height: usize) -> Result<Self> {
        Self::from_rgba8_with_budget(data, width, height, None)
    }

    /// Build a 4-channel modular image from interleaved RGBA8, charging
    /// the resulting channel allocations against an optional budget.
    pub(crate) fn from_rgba8_with_budget(
        data: &[u8],
        width: usize,
        height: usize,
        budget: Option<&alloc::sync::Arc<crate::budget::MemoryBudget>>,
    ) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 4,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let mut channels = Vec::with_capacity(4);
        for c in 0..4 {
            let mut channel = Channel::new_with_budget(width, height, budget)?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 4 + c;
                    channel.set(x, y, data[idx] as i32);
                }
            }
            channels.push(channel);
        }

        Ok(Self {
            channels,
            bit_depth: 8,
            is_grayscale: false,
            has_alpha: true,
        })
    }

    /// Creates a new modular image from 8-bit grayscale data.
    pub fn from_gray8(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width.checked_mul(height).ok_or(Error::DimensionOverflow {
            width,
            height,
            channels: 1,
        })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let mut channel = Channel::new(width, height)?;
        for (i, &val) in data.iter().enumerate() {
            let x = i % width;
            let y = i / width;
            channel.set(x, y, val as i32);
        }

        Ok(Self {
            channels: vec![channel],
            bit_depth: 8,
            is_grayscale: true,
            has_alpha: false,
        })
    }

    /// Creates a new modular image from 16-bit RGB data (big-endian).
    pub fn from_rgb16(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(6))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 3,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let mut channels = Vec::with_capacity(3);
        for c in 0..3 {
            let mut channel = Channel::new(width, height)?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 6 + c * 2;
                    let val = u16::from_be_bytes([data[idx], data[idx + 1]]);
                    channel.set(x, y, val as i32);
                }
            }
            channels.push(channel);
        }

        Ok(Self {
            channels,
            bit_depth: 16,
            is_grayscale: false,
            has_alpha: false,
        })
    }

    /// Creates a new modular image from native-endian 16-bit RGB data.
    ///
    /// Input is a byte slice interpreted as `&[u16]` in native endian order
    /// (6 bytes per pixel: R_lo, R_hi, G_lo, G_hi, B_lo, B_hi on little-endian).
    pub fn from_rgb16_native(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(6))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 3,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        let pixels: &[u16] = bytemuck::cast_slice(data);
        let mut channels = Vec::with_capacity(3);
        for c in 0..3 {
            let mut channel = Channel::new(width, height)?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3 + c;
                    channel.set(x, y, pixels[idx] as i32);
                }
            }
            channels.push(channel);
        }
        Ok(Self {
            channels,
            bit_depth: 16,
            is_grayscale: false,
            has_alpha: false,
        })
    }

    /// Creates a new modular image from native-endian 16-bit RGBA data.
    ///
    /// Input is 8 bytes per pixel (R, G, B, A as native-endian u16).
    pub fn from_rgba16_native(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(8))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 4,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        let pixels: &[u16] = bytemuck::cast_slice(data);
        let mut channels = Vec::with_capacity(4);
        for c in 0..4 {
            let mut channel = Channel::new(width, height)?;
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 4 + c;
                    channel.set(x, y, pixels[idx] as i32);
                }
            }
            channels.push(channel);
        }
        Ok(Self {
            channels,
            bit_depth: 16,
            is_grayscale: false,
            has_alpha: true,
        })
    }

    /// Creates a new modular image from 8-bit grayscale + alpha data (2 bytes per pixel).
    pub fn from_grayalpha8(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 2,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        let mut gray = Channel::new(width, height)?;
        let mut alpha = Channel::new(width, height)?;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 2;
                gray.set(x, y, data[idx] as i32);
                alpha.set(x, y, data[idx + 1] as i32);
            }
        }
        Ok(Self {
            channels: vec![gray, alpha],
            bit_depth: 8,
            is_grayscale: true,
            has_alpha: true,
        })
    }

    /// Creates a new modular image from native-endian 16-bit grayscale data.
    ///
    /// Input is 2 bytes per pixel (native-endian u16).
    pub fn from_gray16_native(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 1,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        let pixels: &[u16] = bytemuck::cast_slice(data);
        let mut channel = Channel::new(width, height)?;
        for (i, &val) in pixels.iter().enumerate() {
            let x = i % width;
            let y = i / width;
            channel.set(x, y, val as i32);
        }
        Ok(Self {
            channels: vec![channel],
            bit_depth: 16,
            is_grayscale: true,
            has_alpha: false,
        })
    }

    /// Creates a new modular image from native-endian 16-bit grayscale + alpha data.
    ///
    /// Input is 4 bytes per pixel (native-endian u16 gray, u16 alpha).
    pub fn from_grayalpha16_native(data: &[u8], width: usize, height: usize) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::DimensionOverflow {
                width,
                height,
                channels: 2,
            })?;
        if data.len() != expected {
            return Err(Error::InvalidImageDimensions(width, height));
        }
        let pixels: &[u16] = bytemuck::cast_slice(data);
        let mut gray = Channel::new(width, height)?;
        let mut alpha = Channel::new(width, height)?;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 2;
                gray.set(x, y, pixels[idx] as i32);
                alpha.set(x, y, pixels[idx + 1] as i32);
            }
        }
        Ok(Self {
            channels: vec![gray, alpha],
            bit_depth: 16,
            is_grayscale: true,
            has_alpha: true,
        })
    }

    /// Returns the width of the image.
    pub fn width(&self) -> usize {
        self.channels.first().map_or(0, |c| c.width())
    }

    /// Returns the height of the image.
    pub fn height(&self) -> usize {
        self.channels.first().map_or(0, |c| c.height())
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Returns a reference to a channel.
    pub fn channel(&self, idx: usize) -> &Channel {
        &self.channels[idx]
    }

    /// Returns a mutable reference to a channel.
    pub fn channel_mut(&mut self, idx: usize) -> &mut Channel {
        &mut self.channels[idx]
    }

    /// Extracts a rectangular region from the image.
    ///
    /// Creates a new ModularImage containing only the pixels within the
    /// specified bounds. Used for multi-group encoding.
    pub fn extract_region(
        &self,
        x_start: usize,
        y_start: usize,
        x_end: usize,
        y_end: usize,
    ) -> Result<Self> {
        let region_width = x_end.saturating_sub(x_start);
        let region_height = y_end.saturating_sub(y_start);

        if region_width == 0 || region_height == 0 {
            return Err(Error::InvalidImageDimensions(region_width, region_height));
        }

        let mut channels = Vec::with_capacity(self.channels.len());
        for src_channel in &self.channels {
            let mut dst_channel = Channel::new(region_width, region_height)?;

            for dy in 0..region_height {
                let sy = y_start + dy;
                if sy >= src_channel.height() {
                    continue;
                }

                for dx in 0..region_width {
                    let sx = x_start + dx;
                    if sx >= src_channel.width() {
                        continue;
                    }

                    dst_channel.set(dx, dy, src_channel.get(sx, sy));
                }
            }

            channels.push(dst_channel);
        }

        Ok(Self {
            channels,
            bit_depth: self.bit_depth,
            is_grayscale: self.is_grayscale,
            has_alpha: self.has_alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_creation() {
        let channel = Channel::new(100, 100).unwrap();
        assert_eq!(channel.width(), 100);
        assert_eq!(channel.height(), 100);
        assert_eq!(channel.len(), 10000);
    }

    #[test]
    fn test_channel_access() {
        let mut channel = Channel::new(10, 10).unwrap();
        channel.set(5, 5, 42);
        assert_eq!(channel.get(5, 5), 42);
    }

    #[test]
    fn test_channel_clamped() {
        let mut channel = Channel::new(10, 10).unwrap();
        channel.set(0, 0, 100);
        assert_eq!(channel.get_clamped(-1, -1), 0);
        assert_eq!(channel.get_clamped(0, 0), 100);
        assert_eq!(channel.get_clamped(100, 100), 0);
    }

    #[test]
    fn test_modular_image_rgb8() {
        let data = vec![
            255, 0, 0, // Red pixel
            0, 255, 0, // Green pixel
            0, 0, 255, // Blue pixel
            255, 255, 0, // Yellow pixel
        ];
        let img = ModularImage::from_rgb8(&data, 2, 2).unwrap();

        assert_eq!(img.num_channels(), 3);
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);

        // Check R channel
        assert_eq!(img.channel(0).get(0, 0), 255);
        assert_eq!(img.channel(0).get(1, 0), 0);

        // Check G channel
        assert_eq!(img.channel(1).get(0, 0), 0);
        assert_eq!(img.channel(1).get(1, 0), 255);

        // Check B channel
        assert_eq!(img.channel(2).get(0, 0), 0);
        assert_eq!(img.channel(2).get(0, 1), 255);
    }
}
