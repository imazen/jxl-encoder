// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
//
// Every module in this crate forbids `unsafe_code` at the file level
// EXCEPT `rusage.rs` which has a single libc::getrusage FFI shim
// guarded by a `# Safety` comment. The crate-level attribute below
// is intentionally a `deny` not a `forbid` so the single allowed
// site can compile; everything else is `forbid` per-file.
#![deny(unsafe_code)]

//! # zenjxl-tuning-runner — W44-212 sweep worker
//!
//! Single-cell worker for the upcoming tuning-sweep fleet. Given a
//! [`SweepCellSpec`] (one image × effort × distance × strategy × params),
//! the runner:
//!
//! 1. Loads the source PNG → 8-bit sRGB RGB(A) bytes
//! 2. Computes [`jxl_encoder::vardct::encoder::ZenanalyzeProxies`] +
//!    extended features (W44-91 / W44-96 / W44-164 discriminators)
//! 3. Optionally installs the [`SweepCellSpec::params_blob_path`]
//!    `RuntimeTuning` override (best-effort: see W44-211 note in
//!    [`crate::SCAFFOLDING_NOTE`])
//! 4. Encodes via [`jxl_encoder::LossyConfig::encode`] with CPU `rusage`
//!    + wall timing
//! 5. Decodes the JXL back via `jxl` (jxl-rs)
//! 6. Scores the decoded image against the source via:
//!    - GPU (preferred): shells out to `zen-metrics score` for
//!      `butter-norm3-gpu` / `ssim2-gpu` / `cvvdp-gpu`
//!    - CPU fallback (opt-in via `--features cpu-metrics`): in-process
//!      butteraugli + fast-ssim2; cvvdp stays null with backend `"skip"`
//! 7. Emits one [`SweepCellRow`] as a single-row Parquet file
//!
//! ## What this worker does NOT do
//!
//! - Fleet orchestration (chunk claim, atomic R2 upload, retries) —
//!   that lives in [`crate::SCRIPTS_README`]-referenced bash scripts
//! - Multi-cell sweep — the runner is single-cell by design so the
//!   fleet can fan out via shell parallelism + crash-isolation per cell
//! - GPU init / metric library calls — kept out-of-process so a GPU
//!   OOM in one metric cell doesn't kill the runner
//! - Anchor merging (canonical Parquet table) — that's W44-213
//!
//! ## Public API
//!
//! - [`SweepCellSpec`] — JSON-deserialisable input contract
//! - [`SweepCellRow`] — Parquet-emitted output row
//! - [`run_cell`] — top-level entry that wires steps 1–7

#![allow(clippy::too_many_arguments)]

pub mod features;
pub mod metrics;
pub mod params;
pub mod parquet_writer;
pub mod rusage;
pub mod spec;

pub use spec::{SweepCellRow, SweepCellSpec, run_cell};

/// W44-212 scaffolding note. The W44-211 [`crate::params::RuntimeTuning`]
/// override layer is installed in this worker via
/// [`jxl_encoder::tuning::runtime::install_from_postcard_file`], but no
/// production encoder code path reads from it yet (W44-211 commit
/// 7164197e shipped re-export hub + struct only). Until W44-213+ wires
/// consumers, the `params_blob` Parquet column captures the postcard
/// payload the worker INTENDED to apply; downstream MLP training must
/// treat tuning-axis variance for the W44-211 fields as zero until the
/// downstream consumer site exists. See `docs/TUNING_RELATIONS.md`
/// Section 0 for the canonical re-export hub paths.
pub const SCAFFOLDING_NOTE: &str = "\
W44-212 worker uses jxl_encoder::tuning::runtime::install_from_postcard_file. \
RuntimeTuning installation is a no-op at the encoder until consumer sites \
land in W44-213+.";

/// Pointer to the fleet scripts. The bash launcher + Dockerfile mirror
/// the [`zenmetrics`](https://github.com/imazen/zenmetrics) sweep
/// pattern (`Dockerfile.sweep.v26` + `scripts/launch_*.sh`).
pub const SCRIPTS_README: &str = "scripts/zenjxl-tuning-sweep/README.md";
