//! Production tokens against sparse positions observed in unmodified v0.12.
use super::*;
use crate::vardct::ac_strategy::{COVERED_X, COVERED_Y, STRATEGY_CODE_LUT};

struct SparseCase {
    raw: usize,
    position: usize,
    value: i32,
    channel: usize,
    offset: usize,
    custom: bool,
    strict: bool,
}

fn sparse_tokens(case: SparseCase, config: &ProgressivePassConfig) -> (Vec<Vec<Token>>, usize) {
    let SparseCase {
        raw,
        position,
        value,
        channel,
        offset,
        custom,
        strict,
    } = case;
    let code = STRATEGY_CODE_LUT[raw];
    let (px, py) = (COVERED_X[raw], COVERED_Y[raw]);
    let (cx, cy) = (px.max(py), px.min(py));
    let covered = cx * cy;
    let mut order = crate::vardct::ac_group::get_coeff_order(code).to_vec();
    if custom {
        order[covered..].reverse();
    }
    let rank = order.iter().position(|p| *p as usize == position).unwrap();
    let (width, height) = (offset + px, offset + py);
    let mut ac = core::array::from_fn(|_| vec![vec![[0_i32; 64]; width]; height]);
    let (x, y) = (position % (cx * 8), position / (cx * 8));
    let (sx, sy) = (x / 8, y / 8);
    let (bx, by) = if py > px { (sy, sx) } else { (sx, sy) };
    ac[channel][offset + by][offset + bx][(y % 8) * 8 + x % 8] = value;
    let nz = u16::from(rank >= covered);
    let mut nzeros = core::array::from_fn(|_| vec![vec![0_u8; width]; height]);
    let mut raw_nzeros = core::array::from_fn(|_| vec![vec![0_u16; width]; height]);
    raw_nzeros[channel][offset][offset] = nz;
    for row in &mut nzeros[channel][offset..] {
        row[offset..].fill(nz as u8);
    }
    let mut strategy = AcStrategyMap::new_dct8(width, height);
    strategy.set(offset, offset, raw as u8);
    let bucket = crate::vardct::coeff_order::strategy_bucket(code) as usize;
    let mut orders = vec![vec![Vec::<u32>::new(); 3]; 13];
    orders[bucket][channel] = order;
    let (group, groups) = if offset == 0 { (0, 1) } else { (3, 2) };
    let contexts = if strict {
        BlockCtxMap::libjxl_default()
    } else {
        BlockCtxMap::default()
    };
    let tokens = tokenize_ac_group(
        group,
        width,
        height,
        groups,
        &ac,
        &nzeros,
        &raw_nzeros,
        &vec![1; width * height],
        &strategy,
        &contexts,
        custom.then_some(orders.as_slice()),
        if custom { 1 << bucket } else { 0 },
        config,
    );
    (tokens, rank)
}

#[test]
fn progressive_sparse_ac_tokens_preserve_native_ranked_coefficients() {
    for line in include_str!("progressive_ac_v012.tsv")
        .lines()
        .filter(|l| l.starts_with("M\t"))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let code: u8 = fields[1].parse().unwrap();
        let raw = STRATEGY_CODE_LUT.iter().position(|c| *c == code).unwrap();
        let cx: usize = fields[2].parse().unwrap();
        let cy: usize = fields[3].parse().unwrap();
        assert_eq!(
            (
                COVERED_X[raw].max(COVERED_Y[raw]),
                COVERED_X[raw].min(COVERED_Y[raw])
            ),
            (cx, cy)
        );
        let prefix: Vec<u32> = fields[4].split(',').map(|v| v.parse().unwrap()).collect();
        assert_eq!(
            crate::vardct::ac_group::get_coeff_order(code)[..cx * cy],
            prefix
        );
    }
    let mut failures = Vec::new();
    let mut codes = std::collections::BTreeSet::new();
    for line in include_str!("progressive_ac_v012.tsv")
        .lines()
        .filter(|l| l.starts_with("V\t"))
    {
        let v: Vec<i32> = line
            .split('\t')
            .skip(1)
            .map(|v| v.parse().unwrap())
            .collect();
        let code = v[0] as u8;
        codes.insert(code);
        let raw = STRATEGY_CODE_LUT.iter().position(|c| *c == code).unwrap();
        let covered = COVERED_X[raw] * COVERED_Y[raw];
        let (position, native_rank, value) = (v[1] as usize, v[2] as usize, v[3]);
        let expected = v[4] * 2 + v[5];
        assert_eq!(
            expected,
            if native_rank < covered { 0 } else { value },
            "native weighted split"
        );
        for mode in [
            ProgressiveMode::Single,
            ProgressiveMode::QuantizedAcFullAc,
            ProgressiveMode::DcVlfLfAc,
        ] {
            let config = ProgressivePassConfig::from_mode(mode);
            for (custom, strict, offset, channel) in [
                (false, true, 0, 1),
                (true, true, 0, 0),
                (false, false, 32, 2),
                (true, false, 32, 1),
            ] {
                let result = std::panic::catch_unwind(|| {
                    sparse_tokens(
                        SparseCase {
                            raw,
                            position,
                            value,
                            channel,
                            offset,
                            custom,
                            strict,
                        },
                        &config,
                    )
                });
                let label = format!(
                    "code{code} pos{position} input{value} {mode:?} custom{custom} strict{strict} offset{offset} channel{channel}"
                );
                let Ok((tokens, rank)) = result else {
                    failures.push(format!("{label}: token panic"));
                    continue;
                };
                if !custom {
                    assert_eq!(rank, native_rank, "native sparse coefficient rank");
                }
                let header = match channel {
                    1 => 0,
                    0 => 1,
                    2 => 2,
                    _ => unreachable!(),
                };
                let mut recovered = 0;
                for (pass, tokens) in tokens.iter().enumerate() {
                    assert!(tokens[..header].iter().all(|t| t.value == 0));
                    let count = tokens[header].value;
                    if count == 0 {
                        assert_eq!(tokens.len(), 3, "three empty plane headers");
                        continue;
                    }
                    assert_eq!(count, 1, "one sparse coefficient");
                    assert!(rank >= covered, "LLF must have no AC tokens");
                    let index = header + 1 + rank - covered;
                    assert_eq!(
                        tokens.len(),
                        index + 3 - header,
                        "one coefficient and remaining empty planes"
                    );
                    assert!(tokens[header + 1..index].iter().all(|t| t.value == 0));
                    assert!(tokens[index + 1..].iter().all(|t| t.value == 0));
                    let packed = tokens[index].value;
                    let signed = if packed % 2 == 0 {
                        (packed / 2) as i32
                    } else {
                        -((packed / 2) as i32) - 1
                    };
                    recovered += signed << config.shift_for_pass(pass);
                }
                if recovered != expected {
                    failures.push(format!("{label}: recovered{recovered} expected{expected}"));
                }
            }
        }
    }
    assert_eq!(codes.len(), 19, "all offered native codes observed");
    assert!(
        failures.is_empty(),
        "actual sparse token failures: {failures:?}"
    );
}
