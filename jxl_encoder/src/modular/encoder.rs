// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Modular encoder - transforms images into entropy-coded bitstreams.

use super::channel::{Channel, ModularImage};
use super::predictor::{Neighbors, pack_signed};
use super::token::{Histogram, Token, TokenList};
use super::tree::{PixelProperties, Tree, gradient_tree, traverse_tree};
use crate::bit_writer::BitWriter;
use crate::entropy_coding::ans::{AnsDistribution, AnsEncoder};
use crate::error::Result;

/// Options for modular encoding.
#[derive(Debug, Clone)]
pub struct ModularEncoderOptions {
    /// Tree to use for prediction (None = use default gradient tree).
    pub tree: Option<Tree>,
    /// Number of contexts for entropy coding.
    pub num_contexts: usize,
}

impl Default for ModularEncoderOptions {
    fn default() -> Self {
        Self {
            tree: None,
            num_contexts: 1,
        }
    }
}

/// Modular encoder state.
pub struct ModularEncoder {
    /// Encoding options.
    options: ModularEncoderOptions,
    /// Decision tree for prediction.
    tree: Tree,
}

impl ModularEncoder {
    /// Creates a new modular encoder with default options.
    pub fn new() -> Self {
        Self::with_options(ModularEncoderOptions::default())
    }

    /// Creates a new modular encoder with custom options.
    pub fn with_options(options: ModularEncoderOptions) -> Self {
        let tree = options.tree.clone().unwrap_or_else(gradient_tree);
        Self { options, tree }
    }

    /// Encodes a modular image, producing tokens for entropy coding.
    pub fn encode_image(&self, image: &ModularImage) -> Result<EncodedModularData> {
        let mut tokens = TokenList::with_contexts(self.options.num_contexts);
        let mut histograms: Vec<Histogram> = vec![Histogram::new(); self.options.num_contexts];

        // Encode each channel
        for (channel_idx, channel) in image.channels.iter().enumerate() {
            self.encode_channel(channel, channel_idx as u32, &mut tokens, &mut histograms)?;
        }

        // Build ANS distributions from histograms
        let distributions = self.build_distributions(&histograms)?;

        Ok(EncodedModularData {
            tokens,
            distributions,
            tree: self.tree.clone(),
        })
    }

    /// Encodes a single channel.
    fn encode_channel(
        &self,
        channel: &Channel,
        channel_idx: u32,
        tokens: &mut TokenList,
        histograms: &mut [Histogram],
    ) -> Result<()> {
        let width = channel.width();
        let height = channel.height();

        for y in 0..height {
            let row = channel.row(y);
            let prev_row = if y > 0 {
                Some(channel.row(y - 1))
            } else {
                None
            };
            let prev_prev_row = if y > 1 {
                Some(channel.row(y - 2))
            } else {
                None
            };

            for x in 0..width {
                let actual = row[x];

                // Gather neighbors
                let neighbors = Neighbors::gather_fast(row, prev_row, prev_prev_row, x, width);

                // Compute properties for tree traversal
                let nww = if x > 1 && y > 0 {
                    prev_row.map_or(0, |r| if x >= 2 { r[x - 2] } else { 0 })
                } else {
                    0
                };

                let properties = PixelProperties::compute(
                    channel_idx,
                    0, // group_id
                    x,
                    y,
                    neighbors.n,
                    neighbors.w,
                    neighbors.nw,
                    neighbors.ne,
                    neighbors.nn,
                    neighbors.ww,
                    nww,
                );

                // Find leaf node in tree
                let leaf = traverse_tree(&self.tree, &properties);

                // Get prediction
                let prediction = leaf.predictor.predict_from_neighbors(&neighbors);

                // Compute residual
                let residual = actual - prediction;

                // Pack as unsigned (zig-zag encoding)
                let packed = pack_signed(residual);

                // Create token
                let context = leaf.context_id as usize;
                tokens.push(Token::new(context as u32, packed));

                // Update histogram
                if context < histograms.len() {
                    histograms[context].add(packed);
                }
            }
        }

        Ok(())
    }

    /// Builds ANS distributions from histograms.
    fn build_distributions(&self, histograms: &[Histogram]) -> Result<Vec<AnsDistribution>> {
        let mut distributions = Vec::with_capacity(histograms.len());

        for hist in histograms {
            if hist.total() == 0 {
                // Empty histogram - use flat distribution
                distributions.push(AnsDistribution::flat(256)?);
            } else {
                // Find max value to size the distribution
                let max_val = hist.iter().map(|(v, _)| v).max().unwrap_or(0);
                let alphabet_size = (max_val as usize + 1).clamp(2, 4096);

                let dense = hist.to_dense(max_val);
                if dense.iter().all(|&c| c == 0) {
                    distributions.push(AnsDistribution::flat(alphabet_size)?);
                } else {
                    distributions.push(AnsDistribution::from_frequencies(&dense)?);
                }
            }
        }

        Ok(distributions)
    }
}

impl Default for ModularEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoded modular data ready for bitstream writing.
pub struct EncodedModularData {
    /// Tokens for entropy coding.
    pub tokens: TokenList,
    /// ANS distributions for each context.
    pub distributions: Vec<AnsDistribution>,
    /// Decision tree used.
    pub tree: Tree,
}

impl EncodedModularData {
    /// Writes the encoded data to a BitWriter.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // Write tree (for now, just indicate we're using a simple tree)
        self.write_tree(writer)?;

        // Write distributions
        for dist in &self.distributions {
            dist.write(writer)?;
        }

        // Write tokens using ANS
        let mut encoder = AnsEncoder::new();

        // Process tokens in reverse order
        for token in self.tokens.tokens().iter().rev() {
            let dist = &self.distributions[token.context as usize % self.distributions.len()];
            if let Some(info) = dist.get(token.value as usize) {
                encoder.put_symbol(info);
            }
        }

        encoder.finalize(writer)?;

        Ok(())
    }

    /// Writes the decision tree to the bitstream.
    fn write_tree(&self, writer: &mut BitWriter) -> Result<()> {
        // For a simple single-node tree, we write minimal info
        // Full implementation would serialize the complete tree structure

        // Write tree size indicator (1 = single node)
        writer.write(1, 0)?; // Use simple tree encoding

        // For now, just indicate gradient predictor
        // In full implementation, would encode tree structure

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_simple_image() {
        // Create a simple 4x4 gradient image
        let data: Vec<u8> = (0..48).collect();
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let encoder = ModularEncoder::new();
        let encoded = encoder.encode_image(&image).unwrap();

        // Should have tokens for all pixels in all channels
        assert_eq!(encoded.tokens.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_encode_flat_image() {
        // Create a flat image (all same value)
        let data: Vec<u8> = vec![128; 48];
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let encoder = ModularEncoder::new();
        let encoded = encoder.encode_image(&image).unwrap();

        // All residuals should be small after prediction
        assert_eq!(encoded.tokens.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_write_encoded() {
        let data: Vec<u8> = (0..48).collect();
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();

        let encoder = ModularEncoder::new();
        let encoded = encoder.encode_image(&image).unwrap();

        let mut writer = BitWriter::new();
        encoded.write(&mut writer).unwrap();

        let bytes = writer.finish_with_padding();
        assert!(!bytes.is_empty());
    }
}
