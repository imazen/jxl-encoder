// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Image buffer types for the JPEG XL encoder.

use crate::error::{Error, Result};

/// A 2D image buffer with a single channel.
#[derive(Debug, Clone)]
pub struct Image<T> {
    data: Vec<T>,
    width: usize,
    height: usize,
}

impl<T: Clone + Default> Image<T> {
    /// Creates a new image filled with the default value.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidImageDimensions(width, height));
        }

        let size = width
            .checked_mul(height)
            .ok_or(Error::InvalidImageDimensions(width, height))?;

        let mut data = Vec::new();
        data.try_reserve_exact(size)?;
        data.resize(size, T::default());

        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Creates a new image from existing data.
    pub fn from_vec(data: Vec<T>, width: usize, height: usize) -> Result<Self> {
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
        })
    }
}

impl<T> Image<T> {
    /// Returns the width of the image.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height of the image.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the total number of pixels.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the image has no pixels.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the pixel at (x, y).
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> &T {
        debug_assert!(x < self.width && y < self.height);
        &self.data[y * self.width + x]
    }

    /// Returns a mutable reference to the pixel at (x, y).
    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        debug_assert!(x < self.width && y < self.height);
        &mut self.data[y * self.width + x]
    }

    /// Returns a reference to a row.
    #[inline]
    pub fn row(&self, y: usize) -> &[T] {
        debug_assert!(y < self.height);
        let start = y * self.width;
        &self.data[start..start + self.width]
    }

    /// Returns a mutable reference to a row.
    #[inline]
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        debug_assert!(y < self.height);
        let start = y * self.width;
        &mut self.data[start..start + self.width]
    }

    /// Returns a reference to the underlying data.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the image and returns the underlying data.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

/// A multi-channel image (e.g., RGB, RGBA).
#[derive(Debug, Clone)]
pub struct ImageBundle<T> {
    /// Individual channel images.
    pub channels: Vec<Image<T>>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

impl<T: Clone + Default> ImageBundle<T> {
    /// Creates a new image bundle with the specified number of channels.
    pub fn new(width: usize, height: usize, num_channels: usize) -> Result<Self> {
        let mut channels = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            channels.push(Image::new(width, height)?);
        }
        Ok(Self {
            channels,
            width,
            height,
        })
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Returns a reference to a specific channel.
    pub fn channel(&self, idx: usize) -> &Image<T> {
        &self.channels[idx]
    }

    /// Returns a mutable reference to a specific channel.
    pub fn channel_mut(&mut self, idx: usize) -> &mut Image<T> {
        &mut self.channels[idx]
    }
}

/// Pixel format for input images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Grayscale, 8-bit.
    Gray8,
    /// Grayscale with alpha, 8-bit.
    GrayA8,
    /// RGB, 8-bit per channel.
    Rgb8,
    /// RGBA, 8-bit per channel.
    Rgba8,
    /// Grayscale, 16-bit.
    Gray16,
    /// Grayscale with alpha, 16-bit.
    GrayA16,
    /// RGB, 16-bit per channel.
    Rgb16,
    /// RGBA, 16-bit per channel.
    Rgba16,
    /// RGB, 32-bit float per channel.
    RgbF32,
    /// RGBA, 32-bit float per channel.
    RgbaF32,
}

impl PixelFormat {
    /// Returns the number of channels.
    pub fn num_channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::GrayA8 | Self::GrayA16 => 2,
            Self::Rgb8 | Self::Rgb16 | Self::RgbF32 => 3,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 => 4,
        }
    }

    /// Returns the bytes per sample.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Gray8 | Self::GrayA8 | Self::Rgb8 | Self::Rgba8 => 1,
            Self::Gray16 | Self::GrayA16 | Self::Rgb16 | Self::Rgba16 => 2,
            Self::RgbF32 | Self::RgbaF32 => 4,
        }
    }

    /// Returns the total bytes per pixel.
    pub fn bytes_per_pixel(self) -> usize {
        self.num_channels() * self.bytes_per_sample()
    }

    /// Returns true if this format has an alpha channel.
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            Self::GrayA8 | Self::GrayA16 | Self::Rgba8 | Self::Rgba16 | Self::RgbaF32
        )
    }

    /// Returns true if this is a grayscale format.
    pub fn is_grayscale(self) -> bool {
        matches!(
            self,
            Self::Gray8 | Self::Gray16 | Self::GrayA8 | Self::GrayA16
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let img: Image<f32> = Image::new(100, 100).unwrap();
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
        assert_eq!(img.len(), 10000);
    }

    #[test]
    fn test_image_access() {
        let mut img: Image<u8> = Image::new(10, 10).unwrap();
        *img.get_mut(5, 5) = 42;
        assert_eq!(*img.get(5, 5), 42);
    }

    #[test]
    fn test_image_bundle() {
        let bundle: ImageBundle<f32> = ImageBundle::new(100, 100, 3).unwrap();
        assert_eq!(bundle.num_channels(), 3);
        assert_eq!(bundle.width, 100);
        assert_eq!(bundle.height, 100);
    }

    #[test]
    fn test_pixel_format() {
        assert_eq!(PixelFormat::Rgba8.num_channels(), 4);
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
    }
}
