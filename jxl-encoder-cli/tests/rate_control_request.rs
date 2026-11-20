//! Actual CLI request/controller contracts across every offered CLI policy.
#![cfg(feature = "rate-control")]
use std::io::Write;
#[path = "../../jxl-encoder/tests/support/decoded_frames.rs"]
mod decoded_frames;
#[path = "../../jxl-encoder/tests/support/frame_level_helpers.rs"]
mod helpers;
use helpers::level;

#[derive(Clone, Copy, Debug)]
struct CliPolicy {
    strategy: &'static str,
    experimental: bool,
}
fn cli_policies() -> Vec<CliPolicy> {
    [
        "libjxl",
        "libjxl-exact",
        "libjxl-strict",
        "lean-faster",
        "zenjxl",
        "aggressive",
    ]
    .into_iter()
    .flat_map(|strategy| {
        [false, true].map(|experimental| CliPolicy {
            strategy,
            experimental,
        })
    })
    .collect()
}
fn cli_directory(policy: CliPolicy) -> std::path::PathBuf {
    let path = helpers::directory();
    std::fs::write(path.join("cli-policy.txt"), format!("{policy:?}\n")).unwrap();
    path
}
fn cli_binary() -> std::ffi::OsString {
    std::env::var_os("JXL_FRAME_LEVEL_CLI").unwrap_or_else(|| env!("CARGO_BIN_EXE_cjxl-rs").into())
}
fn apply_cli_policy(command: &mut std::process::Command, policy: CliPolicy) {
    command.args(["--strategy", policy.strategy]);
    if policy.experimental {
        command.arg("--experimental");
    }
}
#[path = "../../jxl-encoder/tests/frame_level_auxiliary_contract/cli_small.rs"]
mod cli_small;
