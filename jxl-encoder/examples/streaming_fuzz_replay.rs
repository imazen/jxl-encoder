//! Replay a valid streaming fuzz input without libFuzzer or mutation.
use std::io::Write as _;

#[path = "../tests/support/fuzz_entry.rs"]
mod entry;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let input = args.next().expect("input fuzz file");
    let output = args.next().expect("new output JXL path");
    assert!(args.next().is_none(), "expected input and output paths");
    let input = std::fs::read(input).expect("read fuzz input");
    let encoded = entry::streaming_roundtrip(&input).expect("valid case must encode");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .expect("create new output");
    file.write_all(&encoded).expect("retain encoded stream");
    let decoded = std::process::Command::new(jxl_encoder::test_helpers::djxl_path())
        .arg(output)
        .args(["--disable_output", "--num_threads=1"])
        .output()
        .expect("run pinned djxl");
    assert!(
        decoded.status.success(),
        "djxl: {}",
        String::from_utf8_lossy(&decoded.stderr)
    );
    println!(
        "streaming bytes and Rust pixels agree; djxl fully rendered {} bytes",
        encoded.len()
    );
}
