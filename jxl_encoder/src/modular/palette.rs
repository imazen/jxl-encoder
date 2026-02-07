// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Palette transform for modular encoding.
//!
//! Replaces multi-channel pixel data with single-channel palette indices,
//! plus a palette meta-channel. Huge win for graphics/screenshots (19-57%).

use super::channel::{Channel, ModularImage};
use crate::error::Result;
use std::collections::BTreeMap;

/// Result of palette analysis.
pub struct PaletteAnalysis {
    /// Whether palette transform is beneficial.
    pub use_palette: bool,
    /// Number of unique colors found.
    pub num_colors: usize,
    /// The palette (color vectors, sorted for better compression).
    pub palette: Vec<Vec<i32>>,
    /// Map from color vector to palette index.
    pub color_to_index: BTreeMap<Vec<i32>, i32>,
}

/// Analyze an image to determine if palette transform is beneficial.
///
/// For lossless: palette is beneficial if num_unique_colors <= max_colors.
/// `begin_c` and `num_c` specify the channel range to palettize (e.g., 0..3 for RGB).
pub fn analyze_palette(
    image: &ModularImage,
    begin_c: usize,
    num_c: usize,
    max_colors: usize,
) -> PaletteAnalysis {
    let width = image.width();
    let height = image.height();

    // Collect unique colors
    let mut color_counts: BTreeMap<Vec<i32>, u32> = BTreeMap::new();
    let mut color_buf: Vec<i32> = vec![0; num_c];

    for y in 0..height {
        for x in 0..width {
            for (i, c) in (begin_c..begin_c + num_c).enumerate() {
                color_buf[i] = image.channels[c].get(x, y);
            }
            *color_counts.entry(color_buf.clone()).or_insert(0) += 1;
        }
    }

    let num_colors = color_counts.len();

    if num_colors > max_colors || num_colors <= 1 {
        return PaletteAnalysis {
            use_palette: false,
            num_colors,
            palette: Vec::new(),
            color_to_index: BTreeMap::new(),
        };
    }

    // Sort palette by luminance-like ordering for better prediction/compression.
    // For RGB: weight by approximate luminance (0.299R + 0.587G + 0.114B).
    // For single-channel: just sort by value.
    let mut palette: Vec<Vec<i32>> = color_counts.keys().cloned().collect();

    if num_c >= 3 {
        // Sort by approximate luminance
        palette.sort_by(|a, b| {
            let lum_a = a[0] * 299 + a[1] * 587 + a[2] * 114;
            let lum_b = b[0] * 299 + b[1] * 587 + b[2] * 114;
            lum_a.cmp(&lum_b)
        });
    } else {
        palette.sort();
    }

    // Build index map
    let mut color_to_index = BTreeMap::new();
    for (i, color) in palette.iter().enumerate() {
        color_to_index.insert(color.clone(), i as i32);
    }

    PaletteAnalysis {
        use_palette: true,
        num_colors,
        palette,
        color_to_index,
    }
}

/// Apply the palette transform to a modular image (lossless).
///
/// Replaces channels `begin_c..begin_c+num_c` with:
/// - A palette meta-channel (width=nb_colors, height=num_c) at position `begin_c`
/// - An index channel (same width/height as original) at position `begin_c+1`
///
/// Returns the number of colors in the palette.
pub fn apply_palette(
    image: &mut ModularImage,
    begin_c: usize,
    num_c: usize,
    analysis: &PaletteAnalysis,
) -> Result<usize> {
    let width = image.width();
    let height = image.height();
    let nb_colors = analysis.palette.len();

    // Create palette meta-channel.
    // In JXL, the palette is stored as nb_colors wide, num_c high.
    // palette_channel.get(i, c) = palette[i][c]
    let mut palette_channel = Channel::new(nb_colors, num_c)?;
    for (i, color) in analysis.palette.iter().enumerate() {
        for (c, &val) in color.iter().enumerate() {
            palette_channel.set(i, c, val);
        }
    }

    // Create index channel
    let mut index_channel = Channel::new(width, height)?;
    for y in 0..height {
        for x in 0..width {
            let color: Vec<i32> = (begin_c..begin_c + num_c)
                .map(|c| image.channels[c].get(x, y))
                .collect();
            let index = analysis.color_to_index[&color];
            index_channel.set(x, y, index);
        }
    }

    // Replace the original channels with palette meta-channel + index channel.
    // Remove channels begin_c..begin_c+num_c, insert palette + index at begin_c.
    let mut new_channels = Vec::new();
    for (i, ch) in image.channels.iter().enumerate() {
        if i == begin_c {
            new_channels.push(palette_channel.clone());
            new_channels.push(index_channel.clone());
        }
        if i < begin_c || i >= begin_c + num_c {
            new_channels.push(ch.clone());
        }
    }

    image.channels = new_channels;

    Ok(nb_colors)
}

/// Check if palette transform is worthwhile for this image.
///
/// Returns `Some((begin_c, num_c))` if palette should be applied, `None` otherwise.
pub fn should_use_palette(image: &ModularImage) -> Option<(usize, usize)> {
    if image.channels.len() < 2 {
        // Grayscale — palette still helps for few-color images
        let analysis = analyze_palette(image, 0, 1, 256);
        if analysis.use_palette && analysis.num_colors <= 256 {
            return Some((0, 1));
        }
        return None;
    }

    // RGB or RGBA: palettize the color channels (not alpha)
    let num_color_channels = if image.has_alpha {
        image.channels.len() - 1
    } else {
        image.channels.len()
    };

    if num_color_channels < 2 {
        return None;
    }

    // For 8-bit images, max palette size is 256
    let max_colors = 1usize << image.bit_depth.min(12);
    let analysis = analyze_palette(image, 0, num_color_channels, max_colors);

    if analysis.use_palette {
        Some((0, num_color_channels))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::channel::ModularImage;

    #[test]
    fn test_palette_analysis_few_colors() {
        // 4x4 image with only 3 colors
        let data: Vec<u8> = vec![
            255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0,
            0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0,
        ];
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();
        let analysis = analyze_palette(&image, 0, 3, 256);
        assert!(analysis.use_palette);
        assert_eq!(analysis.num_colors, 3);
    }

    #[test]
    fn test_palette_analysis_too_many_colors() {
        // Each pixel is unique
        let mut data = Vec::new();
        for i in 0..64u8 {
            data.push(i);
            data.push(i);
            data.push(i);
        }
        let image = ModularImage::from_rgb8(&data, 8, 8).unwrap();
        let analysis = analyze_palette(&image, 0, 3, 16);
        assert!(!analysis.use_palette); // 64 colors > max_colors=16
    }

    #[test]
    fn test_palette_apply() {
        // 2x2 image with 2 colors: red and blue
        let data: Vec<u8> = vec![255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255];
        let mut image = ModularImage::from_rgb8(&data, 2, 2).unwrap();
        assert_eq!(image.channels.len(), 3);

        let analysis = analyze_palette(&image, 0, 3, 256);
        assert!(analysis.use_palette);
        assert_eq!(analysis.num_colors, 2);

        let nb_colors = apply_palette(&mut image, 0, 3, &analysis).unwrap();
        assert_eq!(nb_colors, 2);

        // After palette: palette_channel + index_channel = 2 channels
        assert_eq!(image.channels.len(), 2);

        // Palette channel: width=2, height=3 (num_c=3 for RGB)
        assert_eq!(image.channels[0].width(), 2);
        assert_eq!(image.channels[0].height(), 3);

        // Index channel: width=2, height=2
        assert_eq!(image.channels[1].width(), 2);
        assert_eq!(image.channels[1].height(), 2);

        // All pixels should map to valid indices
        for y in 0..2 {
            for x in 0..2 {
                let idx = image.channels[1].get(x, y);
                assert!(idx >= 0 && idx < nb_colors as i32);
            }
        }
    }

    #[test]
    fn test_palette_roundtrip_values() {
        // Verify palette stores correct values
        let data: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60];
        let mut image = ModularImage::from_rgb8(&data, 2, 2).unwrap();

        let analysis = analyze_palette(&image, 0, 3, 256);
        assert_eq!(analysis.num_colors, 2);

        let _nb = apply_palette(&mut image, 0, 3, &analysis).unwrap();

        // Check that we can reconstruct original values from palette + indices
        let palette = &image.channels[0];
        let indices = &image.channels[1];

        for y in 0..2 {
            for x in 0..2 {
                let idx = indices.get(x, y) as usize;
                let r = palette.get(idx, 0);
                let g = palette.get(idx, 1);
                let b = palette.get(idx, 2);
                let orig_r = data[(y * 2 + x) * 3] as i32;
                let orig_g = data[(y * 2 + x) * 3 + 1] as i32;
                let orig_b = data[(y * 2 + x) * 3 + 2] as i32;
                assert_eq!((r, g, b), (orig_r, orig_g, orig_b));
            }
        }
    }
}
