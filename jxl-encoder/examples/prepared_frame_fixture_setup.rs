//! Provision the real prepared-frame patch witness before corpus tests run.
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::path::PathBuf;

fn main() {
    let root_file = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .expect("usage: prepared_frame_fixture_setup <corpus-root-path-file>"),
    );
    let corpus = codec_corpus::Corpus::new().expect("initialize codec-corpus");
    let screenshots = corpus.get("gb82-sc").expect("provision gb82-sc");
    let source = std::fs::read(screenshots.join("imac_g3.png")).expect("required patch witness");
    let mut checksum = String::with_capacity(64);
    for byte in Sha256::digest(&source).iter() {
        write!(&mut checksum, "{byte:02x}").unwrap();
    }
    assert_eq!(
        checksum, "3b0d3eb1d6156bff4e71d6a4a8e087466a993478d095e51a14b93115ffa1918f",
        "prepared-frame source differs from the retained witness"
    );
    std::fs::write(
        root_file,
        screenshots.parent().unwrap().as_os_str().as_encoded_bytes(),
    )
    .expect("write corpus root for the caller");
}
