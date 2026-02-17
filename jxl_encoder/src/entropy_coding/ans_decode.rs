// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// ANS decoder ported from jxl-rs for testing purposes.

use crate::error::{Error, Result};

const LOG_SUM_PROBS: usize = 12;
const SUM_PROBS: u16 = 1 << LOG_SUM_PROBS;
const RLE_MARKER_SYM: u16 = LOG_SUM_PROBS as u16 + 1;

/// Simple bit reader for testing.
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    pub fn read(&mut self, n: usize) -> Result<u64> {
        let mut val = 0u64;
        for i in 0..n {
            let byte_idx = self.bit_pos / 8;
            let bit_idx = self.bit_pos % 8;
            if byte_idx >= self.data.len() {
                return Err(Error::Bitstream("unexpected end of data".to_string()));
            }
            let bit = ((self.data[byte_idx] >> bit_idx) & 1) as u64;
            val |= bit << i;
            self.bit_pos += 1;
        }
        Ok(val)
    }

    pub fn peek(&mut self, n: usize) -> u64 {
        let old_pos = self.bit_pos;
        let val = self.read(n).unwrap_or(0);
        self.bit_pos = old_pos;
        val
    }

    pub fn consume(&mut self, n: usize) -> Result<()> {
        self.bit_pos += n;
        if self.bit_pos > self.data.len() * 8 {
            return Err(Error::Bitstream("unexpected end of data".to_string()));
        }
        Ok(())
    }

    pub fn bits_read(&self) -> usize {
        self.bit_pos
    }
}

/// Decoded ANS histogram for decoding.
#[derive(Debug)]
pub struct AnsHistogram {
    pub buckets: Vec<Bucket>,
    pub log_bucket_size: usize,
    pub bucket_mask: u32,
    pub single_symbol: Option<u32>,
    pub frequencies: Vec<u16>,
}

#[derive(Debug, Copy, Clone)]
pub struct Bucket {
    pub alias_symbol: u8,
    pub alias_cutoff: u8,
    pub dist: u16,
    pub alias_offset: u16,
    pub alias_dist_xor: u16,
}

impl AnsHistogram {
    pub fn decode(br: &mut BitReader, log_alpha_size: usize) -> Result<Self> {
        debug_assert!((5..=8).contains(&log_alpha_size));
        let table_size = 1usize << log_alpha_size;
        let log_bucket_size = LOG_SUM_PROBS - log_alpha_size;
        let bucket_size = 1u16 << log_bucket_size;
        let bucket_mask = bucket_size as u32 - 1;

        let mut dist = vec![0u16; table_size];
        let alphabet_size = if br.read(1)? != 0 {
            if br.read(1)? != 0 {
                Self::decode_dist_two_symbols(br, &mut dist)?
            } else {
                Self::decode_dist_single_symbol(br, &mut dist)?
            }
        } else if br.read(1)? != 0 {
            Self::decode_dist_evenly_distributed(br, &mut dist)?
        } else {
            Self::decode_dist_complex(br, &mut dist)?
        };

        let frequencies = dist.clone();

        if let Some(single_sym_idx) = dist.iter().position(|&d| d == SUM_PROBS) {
            let buckets = dist
                .into_iter()
                .enumerate()
                .map(|(i, dist)| Bucket {
                    dist,
                    alias_symbol: single_sym_idx as u8,
                    alias_offset: bucket_size * i as u16,
                    alias_cutoff: 0,
                    alias_dist_xor: dist ^ SUM_PROBS,
                })
                .collect();
            return Ok(Self {
                buckets,
                log_bucket_size,
                bucket_mask,
                single_symbol: Some(single_sym_idx as u32),
                frequencies,
            });
        }

        Ok(Self {
            buckets: Self::build_alias_map(alphabet_size, log_bucket_size, &dist),
            log_bucket_size,
            bucket_mask,
            single_symbol: None,
            frequencies,
        })
    }

    fn decode_dist_two_symbols(br: &mut BitReader, dist: &mut [u16]) -> Result<usize> {
        let table_size = dist.len();

        let v0 = Self::read_u8(br)? as usize;
        let v1 = Self::read_u8(br)? as usize;
        if v0 == v1 {
            return Err(Error::InvalidHistogram(
                "two symbols are the same".to_string(),
            ));
        }

        let alphabet_size = v0.max(v1) + 1;
        if alphabet_size > table_size {
            return Err(Error::InvalidHistogram("alphabet too large".to_string()));
        }

        let prob = br.read(LOG_SUM_PROBS)? as u16;
        dist[v0] = prob;
        dist[v1] = SUM_PROBS - prob;

        Ok(alphabet_size)
    }

    fn decode_dist_single_symbol(br: &mut BitReader, dist: &mut [u16]) -> Result<usize> {
        let table_size = dist.len();

        let val = Self::read_u8(br)? as usize;
        let alphabet_size = val + 1;
        if alphabet_size > table_size {
            return Err(Error::InvalidHistogram("alphabet too large".to_string()));
        }

        dist[val] = SUM_PROBS;

        Ok(alphabet_size)
    }

    fn decode_dist_evenly_distributed(br: &mut BitReader, dist: &mut [u16]) -> Result<usize> {
        let table_size = dist.len();

        let alphabet_size = Self::read_u8(br)? as usize + 1;
        if alphabet_size > table_size {
            return Err(Error::InvalidHistogram("alphabet too large".to_string()));
        }

        let base = SUM_PROBS as usize / alphabet_size;
        let remainder = SUM_PROBS as usize % alphabet_size;
        dist[0..remainder].fill(base as u16 + 1);
        dist[remainder..alphabet_size].fill(base as u16);

        Ok(alphabet_size)
    }

    fn decode_dist_complex(br: &mut BitReader, dist: &mut [u16]) -> Result<usize> {
        let table_size = dist.len();

        let mut len = 0usize;
        while len < 3 {
            if br.read(1)? != 0 {
                len += 1;
            } else {
                break;
            }
        }

        let shift = (br.read(len)? + (1 << len) - 1) as i16;
        if shift > 13 {
            return Err(Error::InvalidHistogram("shift too large".to_string()));
        }

        let alphabet_size = Self::read_u8(br)? as usize + 3;
        if alphabet_size > table_size {
            return Err(Error::InvalidHistogram("alphabet too large".to_string()));
        }

        let mut repeat_ranges = Vec::new();
        let mut omit_data: Option<(u16, usize)> = None;
        let mut idx = 0;
        while idx < alphabet_size {
            dist[idx] = Self::read_prefix(br)?;
            if dist[idx] == RLE_MARKER_SYM {
                let repeat_count = Self::read_u8(br)? as usize + 4;
                if idx + repeat_count > alphabet_size {
                    return Err(Error::InvalidHistogram("RLE overflow".to_string()));
                }
                repeat_ranges.push(idx..(idx + repeat_count));
                idx += repeat_count;
                continue;
            }
            match &mut omit_data {
                Some((log, pos)) => {
                    if dist[idx] > *log {
                        *log = dist[idx];
                        *pos = idx;
                    }
                }
                data => {
                    *data = Some((dist[idx], idx));
                }
            }
            idx += 1;
        }
        let Some((_, omit_pos)) = omit_data else {
            return Err(Error::InvalidHistogram("no omit position".to_string()));
        };
        if dist.get(omit_pos + 1) == Some(&RLE_MARKER_SYM) {
            return Err(Error::InvalidHistogram("RLE after omit".to_string()));
        }

        let mut repeat_range_idx = 0usize;
        let mut acc = 0;
        let mut prev_dist = 0u16;
        for (idx, code) in dist.iter_mut().enumerate() {
            if repeat_range_idx < repeat_ranges.len()
                && repeat_ranges[repeat_range_idx].start <= idx
            {
                if repeat_ranges[repeat_range_idx].end == idx {
                    repeat_range_idx += 1;
                } else {
                    *code = prev_dist;
                    acc += *code;
                    if acc >= SUM_PROBS {
                        return Err(Error::InvalidHistogram("sum overflow".to_string()));
                    }
                    continue;
                }
            }

            if *code == 0 {
                prev_dist = 0;
                continue;
            }
            if idx == omit_pos {
                prev_dist = 0;
                continue;
            }
            if *code > 1 {
                let zeros = (*code - 1) as i16;
                let bitcount = (shift - ((LOG_SUM_PROBS as i16 - zeros) >> 1)).clamp(0, zeros);
                *code = (1 << zeros) + ((br.read(bitcount as usize)? as u16) << (zeros - bitcount));
            }

            prev_dist = *code;
            acc += *code;
            if acc >= SUM_PROBS {
                return Err(Error::InvalidHistogram("sum overflow".to_string()));
            }
        }
        dist[omit_pos] = SUM_PROBS - acc;

        Ok(alphabet_size)
    }

    /// Public alias map builder for verification/testing.
    pub fn build_alias_map_from_freqs(
        alphabet_size: usize,
        log_bucket_size: usize,
        dist: &[u16],
    ) -> Vec<Bucket> {
        Self::build_alias_map(alphabet_size, log_bucket_size, dist)
    }

    fn build_alias_map(alphabet_size: usize, log_bucket_size: usize, dist: &[u16]) -> Vec<Bucket> {
        struct WorkingBucket {
            dist: u16,
            alias_symbol: u16,
            alias_offset: u16,
            alias_cutoff: u16,
        }

        let bucket_size = 1u16 << log_bucket_size;
        let mut buckets: Vec<_> = dist
            .iter()
            .enumerate()
            .map(|(i, &dist)| WorkingBucket {
                dist,
                alias_symbol: if i < alphabet_size { i as u16 } else { 0 },
                alias_offset: 0,
                alias_cutoff: dist,
            })
            .collect();

        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (idx, bucket) in buckets.iter().enumerate() {
            match bucket.dist.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(idx),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(idx),
            }
        }
        while let (Some(o), Some(u)) = (overfull.pop(), underfull.pop()) {
            let by = bucket_size - buckets[u].alias_cutoff;
            buckets[o].alias_cutoff -= by;
            buckets[u].alias_symbol = o as u16;
            buckets[u].alias_offset = buckets[o].alias_cutoff;
            match buckets[o].alias_cutoff.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(o),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(o),
            }
        }

        buckets
            .iter()
            .enumerate()
            .map(|(idx, bucket)| {
                if bucket.alias_cutoff == bucket_size {
                    Bucket {
                        dist: bucket.dist,
                        alias_symbol: idx as u8,
                        alias_offset: 0,
                        alias_cutoff: 0,
                        alias_dist_xor: 0,
                    }
                } else {
                    Bucket {
                        dist: bucket.dist,
                        alias_symbol: bucket.alias_symbol as u8,
                        alias_offset: bucket.alias_offset - bucket.alias_cutoff,
                        alias_cutoff: bucket.alias_cutoff as u8,
                        alias_dist_xor: bucket.dist ^ buckets[bucket.alias_symbol as usize].dist,
                    }
                }
            })
            .collect()
    }

    fn read_u8(br: &mut BitReader) -> Result<u8> {
        Ok(if br.read(1)? != 0 {
            let n = br.read(3)?;
            ((1 << n) + br.read(n as usize)?) as u8
        } else {
            0
        })
    }

    fn read_prefix(br: &mut BitReader) -> Result<u16> {
        #[rustfmt::skip]
        const TABLE: [(u8, u8); 128] = [
            (10, 3), (12, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (13, 7), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), (11, 6), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
            (10, 3), ( 0, 5), (7, 3), (3, 4), (6, 3), (8, 3), (9, 3), (5, 4),
            (10, 3), ( 4, 4), (7, 3), (1, 4), (6, 3), (8, 3), (9, 3), (2, 4),
        ];

        let index = br.peek(7) as usize;
        let (sym, bits) = TABLE[index];
        br.consume(bits as usize)?;
        Ok(sym as u16)
    }

    /// Decode a symbol and update state.
    pub fn read(&self, br: &mut BitReader, state: &mut u32) -> u32 {
        let idx = *state & 0xfff;
        let i = (idx >> self.log_bucket_size) as usize;
        let pos = idx & self.bucket_mask;

        let bucket = &self.buckets[i & (self.buckets.len() - 1)];
        let alias_symbol = bucket.alias_symbol as u32;
        let alias_cutoff = bucket.alias_cutoff as u32;
        let dist = bucket.dist as u32;

        let map_to_alias = (pos >= alias_cutoff) as u32;
        let offset = (bucket.alias_offset as u32) * map_to_alias;
        let dist_xor = (bucket.alias_dist_xor as u32) * map_to_alias;

        let dist = dist ^ dist_xor;
        let symbol = (alias_symbol * map_to_alias) | (i as u32 * (1 - map_to_alias));
        let offset = offset + pos;

        let next_state = (*state >> LOG_SUM_PROBS) * dist + offset;
        let select_appended = (next_state < (1 << 16)) as u32;
        let appended_bits = br.peek(16) as u32;
        let appended_state = (next_state << 16) | appended_bits;
        *state = (appended_state * select_appended) | (next_state * (1 - select_appended));
        if select_appended != 0 {
            br.consume(16).ok();
        }
        symbol
    }
}

/// ANS state reader.
pub struct AnsReader(pub u32);

impl AnsReader {
    pub const CHECKSUM: u32 = 0x130000;

    pub fn init(br: &mut BitReader) -> Result<Self> {
        let initial_state = br.read(32)? as u32;
        Ok(Self(initial_state))
    }

    pub fn check_final_state(&self) -> Result<()> {
        if self.0 == Self::CHECKSUM {
            Ok(())
        } else {
            Err(Error::Bitstream(format!(
                "ANS checksum mismatch: got 0x{:08x}, expected 0x{:08x}",
                self.0,
                Self::CHECKSUM
            )))
        }
    }

    pub fn state(&self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_writer::BitWriter;
    use crate::entropy_coding::ans::{ANSEncodingHistogram, ANSHistogramStrategy};
    use crate::entropy_coding::histogram::Histogram;

    #[test]
    fn test_decode_single_symbol() {
        // Create and write a single-symbol histogram
        let histo = Histogram::from_counts(&[100, 0, 0, 0]);
        let ans_histo =
            ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

        let mut writer = BitWriter::new();
        ans_histo.write(&mut writer).unwrap();
        let bytes = writer.finish_with_padding();

        println!("Single symbol histogram bytes: {:02x?}", bytes);

        // Decode it back
        let mut br = BitReader::new(&bytes);
        let decoded = AnsHistogram::decode(&mut br, 6).unwrap();

        println!("Decoded frequencies: {:?}", &decoded.frequencies[..4]);
        println!("Single symbol: {:?}", decoded.single_symbol);

        // Verify
        assert_eq!(decoded.single_symbol, Some(0));
        assert_eq!(decoded.frequencies[0], 4096);
    }

    #[test]
    fn test_decode_two_symbols() {
        // Create and write a two-symbol histogram
        let histo = Histogram::from_counts(&[100, 100, 0, 0]);
        let ans_histo =
            ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

        println!("Two symbol histogram: {:?}", ans_histo.counts);

        let mut writer = BitWriter::new();
        ans_histo.write(&mut writer).unwrap();
        let bytes = writer.finish_with_padding();

        println!("Two symbol histogram bytes: {:02x?}", bytes);

        // Decode it back
        let mut br = BitReader::new(&bytes);
        let decoded = AnsHistogram::decode(&mut br, 6).unwrap();

        println!("Decoded frequencies: {:?}", &decoded.frequencies[..4]);

        // Verify sum
        let sum: u16 = decoded.frequencies.iter().sum();
        assert_eq!(sum, 4096, "Sum should be 4096");

        // Verify the two non-zero entries match what we wrote
        assert_eq!(decoded.frequencies[0], ans_histo.counts[0] as u16);
        assert_eq!(decoded.frequencies[1], ans_histo.counts[1] as u16);
    }

    #[test]
    fn test_decode_general_histogram() {
        // Create and write a general histogram
        let histo = Histogram::from_counts(&[100, 50, 25, 10]);
        let ans_histo =
            ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

        println!("General histogram:");
        println!("  counts: {:?}", ans_histo.counts);
        println!(
            "  method: {}, alphabet_size: {}, omit_pos: {}",
            ans_histo.method, ans_histo.alphabet_size, ans_histo.omit_pos
        );

        let mut writer = BitWriter::new();
        ans_histo.write(&mut writer).unwrap();
        let bytes = writer.finish_with_padding();

        println!("  bytes ({} bytes): {:02x?}", bytes.len(), bytes);

        // Decode it back
        let mut br = BitReader::new(&bytes);
        let decoded = AnsHistogram::decode(&mut br, 6).unwrap();

        println!(
            "Decoded frequencies: {:?}",
            &decoded.frequencies[..ans_histo.alphabet_size]
        );

        // Verify sum
        let sum: u16 = decoded.frequencies.iter().sum();
        assert_eq!(sum, 4096, "Sum should be 4096");

        // Verify frequencies match what we wrote
        for i in 0..ans_histo.alphabet_size {
            assert_eq!(
                decoded.frequencies[i], ans_histo.counts[i] as u16,
                "Frequency mismatch at symbol {}",
                i
            );
        }
    }

    #[test]
    fn test_decode_sparse_histogram_roundtrip() {
        // Reproduce the exact histogram that fails:
        // alphabet_size=36, symbols at positions 1 (4092), 31 (2), 35 (2)
        // This is the histogram that caused gradient_256 tree learning to fail.
        let mut raw_counts = vec![0i32; 40]; // padded to HISTOGRAM_ROUNDING
        raw_counts[1] = 196000; // dominant symbol
        raw_counts[31] = 100; // rare symbol
        raw_counts[35] = 100; // rare symbol
        let histo = Histogram::from_counts(&raw_counts);

        let ans_histo =
            ANSEncodingHistogram::from_histogram(&histo, ANSHistogramStrategy::Precise).unwrap();

        println!("Sparse histogram:");
        println!(
            "  method={}, alphabet_size={}, omit_pos={}",
            ans_histo.method, ans_histo.alphabet_size, ans_histo.omit_pos
        );
        println!("  non-zero counts:");
        for (i, &c) in ans_histo.counts.iter().enumerate() {
            if c != 0 {
                println!("    [{}] = {}", i, c);
            }
        }

        let mut writer = BitWriter::new();
        ans_histo.write(&mut writer).unwrap();
        // Add padding so decoder's peek(7) doesn't read past end
        writer.write(8, 0).unwrap();
        writer.zero_pad_to_byte();
        let bytes = writer.finish();

        println!(
            "  encoded bytes ({} bytes): {:02x?}",
            bytes.len(),
            &bytes[..bytes.len().min(32)]
        );

        // Decode it back
        let mut br = BitReader::new(&bytes);
        let decoded = AnsHistogram::decode(&mut br, 6).unwrap();

        println!("  decoded frequencies:");
        for (i, &f) in decoded.frequencies.iter().enumerate() {
            if f != 0 {
                println!("    [{}] = {}", i, f);
            }
        }

        // Verify frequencies match
        let sum: u16 = decoded.frequencies.iter().sum();
        assert_eq!(sum, 4096, "Sum should be 4096 but got {}", sum);

        for i in 0..ans_histo.alphabet_size {
            assert_eq!(
                decoded.frequencies[i], ans_histo.counts[i] as u16,
                "Frequency mismatch at symbol {}: encoder wrote {}, decoder read {}",
                i, ans_histo.counts[i], decoded.frequencies[i]
            );
        }
    }
}
