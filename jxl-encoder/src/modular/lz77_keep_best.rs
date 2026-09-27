// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Lossless LZ77 selection after prediction, with production entropy coding.

use alloc::{sync::Arc, vec::Vec};
use core::ops::Range;

use crate::{
    bit_writer::BitWriter,
    budget::MemoryBudget,
    entropy_coding::{
        encode_ans::{
            OwnedAnsEntropyCode, build_entropy_code_ans_with_options, write_entropy_code_ans,
            write_tokens_ans,
        },
        lz77::{Lz77Method, Lz77Params, apply_lz77, bucket, write_lz77_header},
        token::Token,
    },
    error::Result,
};

pub(super) struct Candidate {
    pub tokens: Vec<Token>,
    pub ranges: Vec<Range<usize>>,
    pub params: Option<Lz77Params>,
    pub code: OwnedAnsEntropyCode,
    num_contexts: usize,
}

impl Candidate {
    pub fn write_header(&self, writer: &mut BitWriter) -> Result<()> {
        if self.num_contexts + usize::from(self.params.is_some()) > 1 {
            write_lz77_header(self.params.as_ref(), writer)?;
            write_entropy_code_ans(&self.code, writer)
        } else {
            super::section::write_ans_modular_header(writer, &self.code)
        }
    }

    pub fn write_stream(&self, index: usize, writer: &mut BitWriter) -> Result<()> {
        write_tokens_ans(
            &self.tokens[self.ranges[index].clone()],
            &self.code,
            self.params.as_ref(),
            writer,
        )
    }
}

#[derive(Clone, Copy, Debug)]
enum Parse {
    Incumbent,
    Plain,
    Greedy,
    Bucket,
}

pub(super) struct Selection<'a> {
    pub streams: &'a [(&'a [Token], i32)],
    pub num_contexts: usize,
    pub total_pixels: usize,
    pub method: Lz77Method,
    pub budget: Option<&'a Arc<MemoryBudget>>,
}

impl Selection<'_> {
    fn build(&self, parse: Parse) -> Result<Candidate> {
        let count = self.streams.iter().map(|(tokens, _)| tokens.len()).sum();
        let mut tokens = crate::budget::vec_with_capacity_fallible(
            self.budget.is_some_and(|b| b.is_fallible()),
            count,
        )?;
        let mut ranges = Vec::with_capacity(self.streams.len());
        for &(plain, dm) in self.streams {
            let transformed = match parse {
                Parse::Incumbent | Parse::Greedy => apply_lz77(
                    plain,
                    self.num_contexts,
                    false,
                    if matches!(parse, Parse::Greedy) {
                        Lz77Method::Greedy
                    } else {
                        self.method
                    },
                    dm,
                    self.budget,
                )?,
                Parse::Plain => None,
                Parse::Bucket => {
                    bucket::apply::<3>(plain, self.num_contexts, false, dm, self.budget)?
                }
            };
            let start = tokens.len();
            tokens.extend_from_slice(transformed.as_ref().map_or(plain, |(t, _)| t));
            ranges.push(start..tokens.len());
        }
        let params = tokens.iter().any(Token::is_lz77_length).then(|| {
            let mut p = Lz77Params::new(self.num_contexts, false);
            p.enabled = true;
            p
        });
        let code = build_entropy_code_ans_with_options(
            &tokens,
            self.num_contexts + usize::from(params.is_some()),
            true,
            true,
            params.as_ref(),
            Some(self.total_pixels),
        );
        Ok(Candidate {
            tokens,
            ranges,
            params,
            code,
            num_contexts: self.num_contexts,
        })
    }

    /// The caller measures its real coding unit: prefixes, entropy header,
    /// separately flushed streams, padding and TOC. A candidate is never
    /// accepted on estimator cost, token count, or failed serialization.
    pub fn select(&self, measure: impl Fn(&Candidate) -> Result<usize>) -> Result<Candidate> {
        // Retained incumbent/candidate tokens plus transient parse buffers.
        // Baseline allocations still use the caller's ordinary preflight.
        let extra = self
            .streams
            .iter()
            .map(|(t, _)| t.len() as u64)
            .sum::<u64>()
            * core::mem::size_of::<Token>() as u64
            * 4
            + 1024 * 1024;
        let _reservation = MemoryBudget::reserve_opt(self.budget, extra)?;
        let mut best = self.build(Parse::Incumbent)?;
        let mut best_size = measure(&best)?;
        #[cfg(test)]
        observe("incumbent", best_size);
        for parse in [Parse::Plain, Parse::Greedy, Parse::Bucket] {
            if matches!(parse, Parse::Greedy) && self.method != Lz77Method::Optimal {
                continue;
            }
            let candidate = self.build(parse)?;
            let size = measure(&candidate)?;
            #[cfg(test)]
            observe(
                match parse {
                    Parse::Plain => "plain",
                    Parse::Greedy => "greedy",
                    Parse::Bucket => "bucket",
                    Parse::Incumbent => "incumbent",
                },
                size,
            );
            if size < best_size {
                best = candidate;
                best_size = size;
            }
        }
        Ok(best)
    }
}

#[cfg(test)]
std::thread_local! {
    static OBSERVATION: std::cell::RefCell<Option<Vec<(&'static str, usize)>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn observe(parse: &'static str, size: usize) {
    OBSERVATION.with_borrow_mut(|o| {
        if let Some(o) = o {
            o.push((parse, size));
        }
    });
}

#[cfg(all(test, feature = "corpus-tests"))]
pub(super) fn begin_observation() {
    OBSERVATION.with_borrow_mut(|o| *o = Some(Vec::new()));
}
#[cfg(all(test, feature = "corpus-tests"))]
pub(super) fn take_observation() -> Vec<(&'static str, usize)> {
    OBSERVATION.with_borrow_mut(|o| o.take().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_tokens() -> Vec<Token> {
        let source = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/images/frymire-srgb.png"
        ))
        .unwrap()
        .to_rgb8();
        source.as_raw()[..6144]
            .iter()
            .enumerate()
            .map(|(i, &v)| Token::new((i % 3) as u32, u32::from(v)))
            .collect()
    }

    #[test]
    fn tied_sizes_preserve_the_incumbent_and_writer_errors_propagate() {
        let tokens = real_tokens();
        let streams = [(&tokens[..], 64)];
        let selection = Selection {
            streams: &streams,
            num_contexts: 3,
            total_pixels: 2048,
            method: Lz77Method::Optimal,
            budget: None,
        };
        let incumbent = selection.build(Parse::Incumbent).unwrap();
        let kept = selection.select(|_| Ok(100)).unwrap();
        let emit = |c: &Candidate| {
            let mut writer = BitWriter::new();
            c.write_header(&mut writer).unwrap();
            c.write_stream(0, &mut writer).unwrap();
            writer.zero_pad_to_byte();
            writer.finish()
        };
        assert_eq!(emit(&kept), emit(&incumbent));
        let calls = std::cell::Cell::new(0);
        let result = selection.select(|_| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                Ok(100)
            } else {
                Err(crate::error::Error::InvalidInput(
                    "injected writer failure".into(),
                ))
            }
        });
        assert!(
            matches!(result, Err(crate::error::Error::InvalidInput(message)) if message == "injected writer failure")
        );
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn selection_reserves_extra_storage_before_building_candidates() {
        let tokens = real_tokens();
        let streams = [(&tokens[..], 64)];
        let budget = MemoryBudget::new(1);
        let selection = Selection {
            streams: &streams,
            num_contexts: 3,
            total_pixels: 2048,
            method: Lz77Method::Greedy,
            budget: Some(&budget),
        };
        assert!(matches!(
            selection.select(|_| panic!("budget refusal must precede measurement")),
            Err(crate::error::Error::AllocationLimit { .. })
        ));
    }

    #[test]
    fn toc_cost_includes_size_classes_and_shared_byte_padding() {
        use super::super::frame::FrameEncoder;
        // Frozen u2S widths: 12, 16, 24, 32 bits; entries share byte padding.
        for (sizes, bytes) in [
            (vec![0], 2),
            (vec![1023, 0], 1026),
            (vec![1024, 0], 1028),
            (vec![17407], 17409),
            (vec![17408], 17411),
            (vec![4211711], 4211714),
            (vec![4211712], 4211716),
        ] {
            assert_eq!(FrameEncoder::coded_sections_size(&sizes).unwrap(), bytes);
        }
    }
}
