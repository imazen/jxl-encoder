#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "../entry.rs"]
mod entry;
fuzz_target!(|data: &[u8]| {
    let _ = entry::streaming_roundtrip(data);
});
