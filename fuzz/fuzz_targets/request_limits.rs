#![no_main]
use libfuzzer_sys::fuzz_target;
#[path = "../../jxl-encoder/tests/support/fuzz_entry.rs"]
mod entry;
fuzz_target!(|data: &[u8]| entry::request_limits(data));
