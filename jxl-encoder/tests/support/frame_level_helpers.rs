//! Shared artifact directory and independent container-level field observer.
pub(super) fn directory() -> std::path::PathBuf {
    let root = jxl_encoder::test_helpers::output_dir("frame-level-auxiliary-contract");
    std::fs::create_dir_all(&root).unwrap();
    (0..)
        .find_map(|i| {
            let p = root.join(format!("run-{i}"));
            match std::fs::create_dir(&p) {
                Ok(()) => Some(p),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(e) => panic!("{e}"),
            }
        })
        .unwrap()
}
pub(super) fn level(bytes: &[u8]) -> u8 {
    if bytes.starts_with(&[255, 10]) {
        return 5;
    }
    let mut offset = 0;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && offset + size <= bytes.len());
        if &bytes[offset + 4..offset + 8] == b"jxll" {
            assert_eq!(size, 9);
            return bytes[offset + 8];
        }
        offset += size;
    }
    5
}
