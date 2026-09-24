// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossless greedy candidate from libjxl e8ff0976 (post-v0.12).
//! Keeps this encoder's window and match-length work limits.

use super::*;

const HASH_BITS: usize = 14;
const HASH_SIZE: usize = 1 << HASH_BITS;
const MAX_MATCH: usize = 1024;

#[derive(Clone)]
struct Bucket<const N: usize> {
    positions: [usize; N],
    next: usize,
    len: usize,
}

struct Matcher<'a, const N: usize> {
    tokens: &'a [Token],
    buckets: Vec<Bucket<N>>,
    special: [i32; NUM_SPECIAL_DISTANCES],
    num_special: usize,
}

impl<'a, const N: usize> Matcher<'a, N> {
    fn new(tokens: &'a [Token], multiplier: i32, fallible: bool) -> Result<Self> {
        assert!(N > 0);
        let mut buckets = vec_with_capacity_fallible(fallible, HASH_SIZE)?;
        buckets.resize(
            HASH_SIZE,
            Bucket {
                positions: [0; N],
                next: 0,
                len: 0,
            },
        );
        let special = core::array::from_fn(|i| special_distance(i, multiplier));
        Ok(Self {
            tokens,
            buckets,
            special,
            num_special: if multiplier == 0 {
                0
            } else {
                NUM_SPECIAL_DISTANCES
            },
        })
    }

    fn hash(&self, pos: usize) -> usize {
        if self.tokens.len() - pos < 3 {
            return 0;
        }
        let mut hash = 0u32;
        for token in &self.tokens[pos..pos + 3] {
            hash ^= token
                .value
                .wrapping_add(0x9e37_79b9)
                .wrapping_add(hash << 6)
                .wrapping_add(hash >> 2);
        }
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85eb_ca6b);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(0xc2b2_ae35);
        hash ^= hash >> 16;
        hash as usize & (HASH_SIZE - 1)
    }

    fn update(&mut self, pos: usize) {
        let hash = self.hash(pos);
        let bucket = &mut self.buckets[hash];
        bucket.positions[bucket.next] = pos;
        bucket.next = (bucket.next + 1) % N;
        bucket.len = (bucket.len + 1).min(N);
    }

    fn find_match(&mut self, pos: usize, min_length: usize) -> (usize, usize) {
        let bucket = &self.buckets[self.hash(pos)];
        let mut best_len = 0;
        let mut best_symbol = 0;
        let limit = (self.tokens.len() - pos).min(MAX_MATCH);
        for &candidate in &bucket.positions[..bucket.len] {
            let distance = pos - candidate;
            if distance == 0 || distance >= WINDOW_SIZE {
                continue;
            }
            let mut len = 0;
            while len < limit && self.tokens[candidate + len].value == self.tokens[pos + len].value
            {
                len += 1;
            }
            if len < min_length || len < best_len {
                continue;
            }
            let symbol = self.special[..self.num_special]
                .iter()
                .position(|&d| d == distance as i32)
                .unwrap_or(self.num_special + distance - 1);
            if len > best_len || symbol < best_symbol {
                best_len = len;
                best_symbol = symbol;
            }
        }
        self.update(pos);
        (best_symbol, best_len)
    }
}

pub(super) fn apply<const N: usize>(
    tokens: &[Token],
    num_contexts: usize,
    force_huffman: bool,
    distance_multiplier: i32,
    budget: Option<&Arc<MemoryBudget>>,
) -> Result<Option<(Vec<Token>, Lz77Params)>> {
    if tokens.is_empty() {
        return Ok(None);
    }
    let fallible = budget.is_some_and(|b| b.is_fallible());
    let mut lz77 = Lz77Params::new(num_contexts, force_huffman);
    let sce = SymbolCostEstimator::new(num_contexts, force_huffman, tokens, &lz77);
    let mut costs = vec_with_capacity_fallible(fallible, tokens.len() + 1)?;
    costs.push(0.0f32);
    for token in tokens {
        let encoded = UintCoder::encode(token.value);
        let cost = sce.symbol_cost(token.context() as usize, encoded.token as usize)
            + encoded.nbits as f32;
        costs.push(costs.last().unwrap() + cost);
    }
    let mut matcher = Matcher::<N>::new(tokens, distance_multiplier, fallible)?;
    let mut out = vec_with_capacity_fallible(fallible, tokens.len())?;
    let threshold = (tokens.len() as f32 * 0.2 + 16.0) * accept_scale();
    let mut decrease = 0.0;
    let min_length = lz77.min_length as usize;
    let mut pos = 0;
    while pos < tokens.len() {
        // Same upper bound as the hash-chain walk: disjoint matches can save
        // at most the remaining literal cost, even when a match loses bits.
        if !early_out_disabled()
            && !keep_best_enabled()
            && decrease + (costs[tokens.len()] - costs[pos]) <= threshold
        {
            return Ok(None);
        }
        out.push(tokens[pos]);
        if tokens.len() - pos <= 3 {
            pos += 1;
            continue;
        }
        let (symbol, len) = matcher.find_match(pos, min_length);
        if len < min_length {
            pos += 1;
            continue;
        }
        let length_value = (len - min_length) as u32;
        let match_cost = len_cost(length_value) + dist_cost(symbol as u32);
        decrease += costs[pos + len]
            - costs[pos]
            - match_cost
            - sce.add_symbol_cost(tokens[pos].context() as usize);
        let last = out.last_mut().unwrap();
        last.value = length_value;
        last.set_lz77_length(true);
        out.push(Token::new(lz77.distance_context, symbol as u32));
        for next in pos + 1..pos + len {
            matcher.update(next);
        }
        pos += len;
    }
    let accepted = if keep_best_enabled() {
        decrease > 0.0
            && lz77_beats_plain_on_real_cost(tokens, &out, num_contexts, &lz77, force_huffman)
    } else {
        decrease > threshold
    };
    if accepted {
        lz77.enabled = true;
        Ok(Some((out, lz77)))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "corpus-tests")]
    #[test]
    fn retained_float_screen_is_exact_in_both_decoders() {
        use std::path::PathBuf;
        let root = PathBuf::from(std::env::var_os("LZ77_BUCKET_SCREEN_ROOT").expect("screen root"));
        let mut checked = 0;
        let arms = std::env::var("LZ77_BUCKET_SCREEN_ARMS").expect("screen arms");
        for arm in arms.split(',') {
            let table = std::fs::read_to_string(root.join(format!("{arm}.tsv"))).unwrap();
            for line in table.lines().skip(1) {
                let fields: Vec<_> = line.split('\t').collect();
                if fields[3] != "f32" {
                    continue;
                }
                let n: usize = fields[7].parse().unwrap();
                let source = image::open(root.join("input").join(fields[1]))
                    .unwrap()
                    .to_rgb8();
                let (x0, y0) = (
                    (source.width() as usize - n) / 2,
                    (source.height() as usize - n) / 2,
                );
                let expected: Vec<_> = (0..n)
                    .flat_map(|y| {
                        let source = &source;
                        (0..n).flat_map(move |x| {
                            source
                                .get_pixel((x0 + x) as u32, (y0 + y) as u32)
                                .0
                                .map(|v| ((u16::from(v) << 8) as f32 / 65535.0).to_bits())
                        })
                    })
                    .collect();
                let artifact = PathBuf::from(fields[10]);
                let decoded =
                    crate::test_helpers::decode_with_jxl_rs(&std::fs::read(&artifact).unwrap())
                        .unwrap();
                assert_eq!((decoded.width, decoded.height, decoded.channels), (n, n, 3));
                assert_eq!(
                    decoded
                        .pixels
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected,
                    "{arm}/{} jxl-rs",
                    fields[1]
                );
                let pfm = artifact.with_extension("pfm");
                let output = std::process::Command::new(crate::test_helpers::djxl_path())
                    .args([artifact.as_os_str(), pfm.as_os_str()])
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let bytes = std::fs::read(pfm).unwrap();
                let mut start = 0;
                let mut header = Vec::new();
                for (i, byte) in bytes.iter().enumerate() {
                    if *byte == b'\n' {
                        header.push(std::str::from_utf8(&bytes[start..i]).unwrap());
                        start = i + 1;
                        if header.len() == 3 {
                            break;
                        }
                    }
                }
                assert_eq!(header[0], "PF");
                assert_eq!(header[1], format!("{n} {n}"));
                let scale: f32 = header[2].parse().unwrap();
                assert_eq!(scale.abs(), 1.0);
                assert_eq!(bytes.len() - start, n * n * 3 * 4);
                let mut decoded = Vec::new();
                for y in (0..n).rev() {
                    let offset = start + y * n * 3 * 4;
                    for &value in bytes[offset..offset + n * 3 * 4].as_chunks::<4>().0 {
                        decoded.push(if scale < 0.0 {
                            u32::from_le_bytes(value)
                        } else {
                            u32::from_be_bytes(value)
                        });
                    }
                }
                assert_eq!(decoded, expected, "{arm}/{} djxl", fields[1]);
                checked += 1;
            }
        }
        assert_eq!(checked, 4 * arms.split(',').count());
    }

    #[test]
    fn overlapping_matches_roundtrip_values_and_literal_contexts() {
        for multiplier in [0, 1, 7, 256] {
            for huffman in [false, true] {
                let tokens: Vec<_> = (0..8192)
                    .map(|i| Token::new((i % 3) as u32, ((i % 71) * 7919) as u32))
                    .collect();
                let (encoded, params) = apply::<3>(&tokens, 3, huffman, multiplier, None)
                    .unwrap()
                    .expect("repeated wide tokens must produce references");
                let mut decoded = Vec::new();
                let mut iter = encoded.iter();
                let mut matches = 0;
                while let Some(token) = iter.next() {
                    assert_eq!(token.context(), tokens[decoded.len()].context());
                    if token.is_lz77_length() {
                        matches += 1;
                        let distance = iter.next().unwrap();
                        assert_eq!(distance.context(), params.distance_context);
                        let distance =
                            if multiplier != 0 && distance.value < NUM_SPECIAL_DISTANCES as u32 {
                                special_distance(distance.value as usize, multiplier) as usize
                            } else {
                                distance.value as usize + 1
                                    - if multiplier == 0 {
                                        0
                                    } else {
                                        NUM_SPECIAL_DISTANCES
                                    }
                            };
                        let len = token.value as usize + params.min_length as usize;
                        assert!(len <= MAX_MATCH);
                        assert!(
                            distance > 0 && distance < WINDOW_SIZE && distance <= decoded.len()
                        );
                        for _ in 0..len {
                            decoded.push(decoded[decoded.len() - distance]);
                        }
                    } else {
                        decoded.push(token.value);
                    }
                }
                assert!(matches > 0);
                assert_eq!(decoded, tokens.iter().map(|t| t.value).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn ring_eviction_and_distance_symbol_ties() {
        let tokens = vec![Token::new(0, 77); 3000];
        let mut matcher = Matcher::<3>::new(&tokens, 7, false).unwrap();
        for pos in [1, 2, 3, 4] {
            matcher.update(pos);
        }
        let bucket = &matcher.buckets[matcher.hash(4)];
        assert_eq!(bucket.positions, [4, 2, 3]);
        // pos 9: distance 7 (symbol 0) beats the nearer distances 5 and 6.
        assert_eq!(matcher.find_match(9, 3), (0, MAX_MATCH));
    }

    #[test]
    fn expired_matches_and_short_tails_are_not_referenced() {
        let tokens = vec![Token::new(0, 77); WINDOW_SIZE + 8];
        let mut matcher = Matcher::<3>::new(&tokens, 0, false).unwrap();
        matcher.update(0);
        assert_eq!(matcher.find_match(WINDOW_SIZE, 3), (0, 0));
        assert_eq!(matcher.find_match(tokens.len() - 2, 3), (0, 0));
        for len in 0..=3 {
            assert!(
                apply::<3>(&tokens[..len], 1, false, 0, None)
                    .unwrap()
                    .is_none()
            );
        }
    }
}
