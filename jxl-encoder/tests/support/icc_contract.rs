//! Shared retained-stream and profile controls for ICC admission.
#[path = "../../examples/distance_targeting_probe/decode.rs"]
mod decode;
pub(crate) fn decode_pixels(bytes: &[u8], width: usize, height: usize) -> Vec<f32> {
    decode::verify_jxl_rs(bytes, width, height)
}

pub(crate) fn retain_and_decode_stream(
    directory: &std::path::Path,
    label: &str,
    bytes: &[u8],
    dims: (usize, usize),
) -> (Option<Vec<f32>>, bool) {
    let input = directory.join(format!("{label}.jxl"));
    std::fs::write(&input, bytes).unwrap();
    let native = std::panic::catch_unwind(|| decode_pixels(bytes, dims.0, dims.1));
    let rust = match &native {
        Ok(samples) => {
            assert!(samples.iter().all(|v| v.is_finite()));
            format!("rendered {} samples", samples.len())
        }
        Err(_) => "decode refused; panic diagnostic retained in test log".into(),
    };
    let output = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(&input)
        .arg(directory.join(format!("{label}.pam")))
        .arg(format!(
            "--orig_icc_out={}",
            directory.join(format!("{label}.icc")).display()
        ))
        .arg("--num_threads=1")
        .output()
        .unwrap();
    if output.status.success() {
        image::open(directory.join(format!("{label}.pam"))).unwrap();
    }
    std::fs::write(
        directory.join(format!("{label}.decode.log")),
        format!(
            "jxl-rs: {rust}\ndjxl status: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ),
    )
    .unwrap();
    (native.ok(), output.status.success())
}

pub(crate) fn profiles() -> Vec<Vec<u8>> {
    include_str!("../../src/headers/color_profiles_v012.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.split_once('\t')
                .unwrap()
                .1
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect()
        })
        .collect()
}

pub(crate) fn new_directory(label: &str) -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("color-profile-contract");
    (0..)
        .find_map(|i| {
            let path = root.join(format!("{label}-{i}"));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => panic!("{}: {error}", path.display()),
            }
        })
        .unwrap()
}

pub(crate) fn embedded_icc(data: &[u8]) -> Vec<u8> {
    use jxl::api::{JxlColorProfile, JxlDecoder, JxlDecoderOptions, ProcessingResult, states};
    let mut input = data;
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete { result: decoder } = decoder.process(&mut input).unwrap()
    else {
        panic!("incomplete color header")
    };
    match decoder.embedded_color_profile() {
        JxlColorProfile::Icc(profile) => profile.clone(),
        _ => panic!("expected embedded ICC bytes"),
    }
}
