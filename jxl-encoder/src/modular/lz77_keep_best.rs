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
        lz77::{
            Lz77Method, Lz77Params, Lz77Parse, apply_lz77, apply_lz77_optimal_keeping_greedy,
            bucket, skip_greedy, write_lz77_header,
        },
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

/// A candidate's transform stage output: the concatenated token stream with
/// per-stream ranges, before params derivation and entropy-code construction.
struct Parsed {
    tokens: Vec<Token>,
    ranges: Vec<Range<usize>>,
    /// Any stream's parse produced a transform. `false` means the token
    /// stream is the plain concatenation — identical to `Parse::Plain`'s.
    transformed: bool,
}

impl Selection<'_> {
    fn parse(&self, parse: Parse, greedy: &mut GreedyReuse) -> Result<Parsed> {
        let count = self.streams.iter().map(|(tokens, _)| tokens.len()).sum();
        let mut tokens = crate::budget::vec_with_capacity_fallible(
            self.budget.is_some_and(|b| b.is_fallible()),
            count,
        )?;
        let mut ranges = Vec::with_capacity(self.streams.len());
        let mut transformed = false;
        for (index, &(plain, dm)) in self.streams.iter().enumerate() {
            let parsed = match parse {
                Parse::Incumbent if greedy.recording() => {
                    let (optimal, byproduct) = apply_lz77_optimal_keeping_greedy(
                        plain,
                        self.num_contexts,
                        false,
                        dm,
                        self.budget,
                    )?;
                    greedy.record(byproduct);
                    optimal
                }
                Parse::Incumbent => apply_lz77(
                    plain,
                    self.num_contexts,
                    false,
                    self.method,
                    dm,
                    self.budget,
                )?,
                Parse::Greedy if greedy.replaying() => greedy.take(index),
                Parse::Greedy => apply_lz77(
                    plain,
                    self.num_contexts,
                    false,
                    Lz77Method::Greedy,
                    dm,
                    self.budget,
                )?,
                Parse::Plain => None,
                Parse::Bucket => {
                    bucket::apply::<3>(plain, self.num_contexts, false, dm, self.budget)?
                }
            };
            transformed |= parsed.is_some();
            let start = tokens.len();
            tokens.extend_from_slice(parsed.as_ref().map_or(plain, |(t, _)| t));
            ranges.push(start..tokens.len());
        }
        Ok(Parsed {
            tokens,
            ranges,
            transformed,
        })
    }

    fn finish(&self, parsed: Parsed) -> Candidate {
        let params = parsed.tokens.iter().any(Token::is_lz77_length).then(|| {
            let mut p = Lz77Params::new(self.num_contexts, false);
            p.enabled = true;
            p
        });
        let code = build_entropy_code_ans_with_options(
            &parsed.tokens,
            self.num_contexts + usize::from(params.is_some()),
            true,
            true,
            params.as_ref(),
            Some(self.total_pixels),
        );
        Candidate {
            tokens: parsed.tokens,
            ranges: parsed.ranges,
            params,
            code,
            num_contexts: self.num_contexts,
        }
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
        let mut greedy = if matches!(self.method, Lz77Method::Optimal) && !skip_greedy() {
            GreedyReuse::Recording(Vec::new())
        } else {
            GreedyReuse::Off
        };
        let incumbent = self.parse(Parse::Incumbent, &mut greedy)?;
        greedy.start_replay();
        let untransformed = !incumbent.transformed;
        let mut best = self.finish(incumbent);
        let mut best_size = measure(&best)?;
        let mut plain_size = untransformed.then_some(best_size);
        observe("incumbent", best_size);
        for parse in [Parse::Plain, Parse::Greedy, Parse::Bucket] {
            if matches!(parse, Parse::Greedy) && self.method != Lz77Method::Optimal {
                continue;
            }
            let size = if let (Parse::Plain, Some(size)) = (parse, plain_size) {
                size
            } else {
                let parsed = self.parse(parse, &mut greedy)?;
                let known = match plain_size {
                    Some(size) if !parsed.transformed => Some(size),
                    _ if parsed.ranges == best.ranges
                        && same_tokens(&parsed.tokens, &best.tokens) =>
                    {
                        Some(best_size)
                    }
                    _ => None,
                };
                match known {
                    Some(size) => size,
                    None => {
                        let untransformed = !parsed.transformed;
                        let candidate = self.finish(parsed);
                        let size = measure(&candidate)?;
                        if untransformed {
                            plain_size = Some(size);
                        }
                        if size < best_size {
                            best = candidate;
                            best_size = size;
                        }
                        size
                    }
                }
            };
            observe(
                match parse {
                    Parse::Plain => "plain",
                    Parse::Greedy => "greedy",
                    Parse::Bucket => "bucket",
                    Parse::Incumbent => "incumbent",
                },
                size,
            );
        }
        Ok(best)
    }
}

/// Whether two token streams are identical in every coding-relevant
/// respect: length, and each token's context, LZ77-length flag and value — the
/// whole of a token's state.
fn same_tokens(a: &[Token], b: &[Token]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.value == b.value
                && a.context() == b.context()
                && a.is_lz77_length() == b.is_lz77_length()
        })
}

/// Carries the greedy parse produced as a by-product of the `Optimal`
/// incumbent over to the `Greedy` candidate.
enum GreedyReuse {
    /// Not applicable: the incumbent is not an `Optimal` parse, or the greedy
    /// candidate is being suppressed diagnostically.
    Off,
    /// Collecting one entry per stream, in stream order.
    Recording(Vec<Lz77Parse>),
    /// Handing the collected entries back out, in stream order.
    Replaying(Vec<Lz77Parse>),
}

impl GreedyReuse {
    fn recording(&self) -> bool {
        matches!(self, Self::Recording(_))
    }

    fn replaying(&self) -> bool {
        matches!(self, Self::Replaying(_))
    }

    fn record(&mut self, parse: Lz77Parse) {
        if let Self::Recording(parses) = self {
            parses.push(parse);
        }
    }

    fn start_replay(&mut self) {
        if let Self::Recording(parses) = self {
            *self = Self::Replaying(core::mem::take(parses));
        }
    }

    fn take(&mut self, index: usize) -> Lz77Parse {
        match self {
            Self::Replaying(parses) => parses[index].take(),
            _ => None,
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static OBSERVATION: std::cell::RefCell<Option<Vec<(&'static str, usize)>>> = const { std::cell::RefCell::new(None) };
}

/// Records a candidate's measured coded size for the corpus regression, which
/// asserts on the full candidate ladder. Candidates whose size is equal to an
/// already-measured candidate by construction report that size.
#[cfg(not(test))]
fn observe(_parse: &'static str, _size: usize) {}

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

    /// Real sample values laid down as a short repeating cycle, so the
    /// matchers find backreferences and the parses actually diverge.
    fn repeating_tokens() -> Vec<Token> {
        let source = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/images/frymire-srgb.png"
        ))
        .unwrap()
        .to_rgb8();
        let cycle = &source.as_raw()[..300];
        (0..6144)
            .map(|i| Token::new((i % 3) as u32, u32::from(cycle[i % cycle.len()])))
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
        let incumbent = selection.finish(
            selection
                .parse(Parse::Incumbent, &mut GreedyReuse::Off)
                .unwrap(),
        );
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

    /// The dedup fast paths (untransformed-plain reuse, same-tokens reuse and
    /// the greedy handover) must leave the measured candidate ladder and the
    /// winning bytes identical to parsing and measuring every candidate
    /// outright.
    #[test]
    fn reused_parses_preserve_the_measured_ladder() {
        fn emit(candidate: &Candidate, streams: usize) -> Vec<u8> {
            let mut writer = BitWriter::new();
            candidate.write_header(&mut writer).unwrap();
            for index in 0..streams {
                candidate.write_stream(index, &mut writer).unwrap();
            }
            writer.zero_pad_to_byte();
            writer.finish()
        }

        /// The ladder `select` reports under observation plus the winning
        /// candidate's serialized bytes.
        fn ladder(selection: &Selection, streams: usize) -> (Vec<(&'static str, usize)>, Vec<u8>) {
            OBSERVATION.with_borrow_mut(|o| *o = Some(Vec::new()));
            let winner = selection.select(|c| Ok(emit(c, streams).len())).unwrap();
            let sizes = OBSERVATION.with_borrow_mut(|o| o.take().unwrap());
            (sizes, emit(&winner, streams))
        }

        /// The same ladder computed by parsing and measuring every candidate
        /// directly — no replay, no dedup.
        fn oracle(selection: &Selection, streams: usize) -> (Vec<(&'static str, usize)>, Vec<u8>) {
            let measure = |c: &Candidate| -> Result<usize> { Ok(emit(c, streams).len()) };
            let mut sizes = Vec::new();
            let mut best = selection.finish(
                selection
                    .parse(Parse::Incumbent, &mut GreedyReuse::Off)
                    .unwrap(),
            );
            let mut best_size = measure(&best).unwrap();
            sizes.push(("incumbent", best_size));
            for parse in [Parse::Plain, Parse::Greedy, Parse::Bucket] {
                if matches!(parse, Parse::Greedy) && selection.method != Lz77Method::Optimal {
                    continue;
                }
                let candidate =
                    selection.finish(selection.parse(parse, &mut GreedyReuse::Off).unwrap());
                let size = measure(&candidate).unwrap();
                sizes.push((
                    match parse {
                        Parse::Plain => "plain",
                        Parse::Greedy => "greedy",
                        Parse::Bucket => "bucket",
                        Parse::Incumbent => "incumbent",
                    },
                    size,
                ));
                if size < best_size {
                    best = candidate;
                    best_size = size;
                }
            }
            (sizes, emit(&best, streams))
        }

        let tokens = real_tokens();
        // Literal runs the matchers find nothing in, so every parse falls back
        // to the plain stream and the plain-equivalence path is exercised.
        let flat: Vec<Token> = (0..4096)
            .map(|i| Token::new((i % 3) as u32, (i * 7919) as u32))
            .collect();
        let repeating = repeating_tokens();
        for source in [&tokens, &flat, &repeating] {
            for method in [Lz77Method::Greedy, Lz77Method::Optimal] {
                let halves = [
                    (&source[..source.len() / 2], 64),
                    (&source[source.len() / 2..], 64),
                ];
                for streams in [&halves[..1], &halves[..]] {
                    let selection = Selection {
                        streams,
                        num_contexts: 3,
                        total_pixels: 2048,
                        method,
                        budget: None,
                    };
                    assert_eq!(
                        ladder(&selection, streams.len()),
                        oracle(&selection, streams.len()),
                        "{method:?} over {} stream(s)",
                        streams.len()
                    );
                }
            }
        }
    }

    #[test]
    fn the_optimal_parse_hands_over_the_greedy_parse_verbatim() {
        let tokens = repeating_tokens();
        let (optimal, reused) =
            apply_lz77_optimal_keeping_greedy(&tokens, 3, false, 64, None).unwrap();
        let direct = apply_lz77(&tokens, 3, false, Lz77Method::Greedy, 64, None).unwrap();
        assert!(optimal.is_some(), "frymire tokens must parse");
        let (reused_tokens, reused_params) = reused.expect("the optimal parse keeps its greedy");
        let (direct_tokens, direct_params) = direct.expect("greedy parses the same tokens");
        assert!(same_tokens(&reused_tokens, &direct_tokens));
        assert_eq!(
            alloc::format!("{reused_params:?}"),
            alloc::format!("{direct_params:?}")
        );
    }

    #[test]
    fn token_equality_rejects_any_coding_relevant_difference() {
        let tokens = real_tokens();
        assert!(same_tokens(&tokens, &tokens));
        let mutators: [fn(&mut Token); 3] = [
            |t| t.value ^= 1,
            |t| *t = Token::new(t.context() + 1, t.value),
            |t| *t = Token::lz77_length(t.context(), t.value),
        ];
        for mutate in mutators {
            let mut mutated = tokens.clone();
            mutate(&mut mutated[17]);
            assert!(!same_tokens(&tokens, &mutated));
        }
        assert!(!same_tokens(&tokens, &tokens[..tokens.len() - 1]));
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
