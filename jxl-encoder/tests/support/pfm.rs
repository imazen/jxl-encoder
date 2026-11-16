/// Declared-endianness and bottom-up PFM parsing, following lossy_f16.rs.
pub(crate) fn read_pfm_samples(path: &std::path::Path, dims: (usize, usize)) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let mut fields = bytes.splitn(4, |v| *v == b'\n');
    let channels = match fields.next().unwrap() {
        b"PF" => 3,
        b"Pf" => 1,
        value => panic!("invalid PFM signature: {value:?}"),
    };
    let dimensions: Vec<usize> = std::str::from_utf8(fields.next().unwrap())
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(dimensions, [dims.0, dims.1]);
    let scale: f32 = std::str::from_utf8(fields.next().unwrap())
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(scale.abs(), 1.0);
    let payload = fields.next().unwrap();
    assert_eq!(payload.len(), dims.0 * dims.1 * channels * 4);
    let samples: Vec<f32> = payload
        .chunks_exact(dims.0 * channels * 4)
        .rev()
        .flat_map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|raw| {
                    if scale < 0.0 {
                        f32::from_le_bytes(*raw)
                    } else {
                        f32::from_be_bytes(*raw)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(samples.iter().all(|v| v.is_finite()));
    samples
}
