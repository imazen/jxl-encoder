// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
}

impl Channel {
    /// Creates a new channel filled with zeros.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let size = width
            .checked_mul(height)
            .ok_or(Error::InvalidImageDimensions(width, height))?;

        let mut data = Vec::new();
        data.try_reserve_exact(size)?;
        data.resize(size, 0);

        Ok(Self {
            data,
            width,
            height,
            hshift: 0,
            vshift: 0,
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
        if data.len() != width * height * 3 {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let mut channels = Vec::with_capacity(3);
        for c in 0..3 {
            let mut channel = Channel::new(width, height)?;
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
        if data.len() != width * height * 4 {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let mut channels = Vec::with_capacity(4);
        for c in 0..4 {
            let mut channel = Channel::new(width, height)?;
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
        if data.len() != width * height {
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
        if data.len() != width * height * 6 {
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
