// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
#![forbid(unsafe_code)]

//! RuntimeTuning install + serialisation.
//!
//! W44-211 shipped [`jxl_encoder::tuning::runtime::RuntimeTuning`] +
//! [`jxl_encoder::tuning::runtime::install_from_postcard_file`]. W44-212
//! wraps both so the sweep runner can:
//!
//! 1. Read a postcard blob from `params_blob_path`, install it as the
//!    process-global tuning override, AND
//! 2. Capture the raw bytes for the Parquet `params_blob` column so
//!    downstream training can join cells by params.
//!
//! See `crate::SCAFFOLDING_NOTE`: as of W44-211, no encoder consumer
//! site reads the runtime override yet — installation is a no-op for
//! encoded bytes. The captured `params_blob` is still useful for the
//! Parquet schema because the next set of consumer-wire chunks
//! (W44-213+) will start producing differential rows.

use std::path::Path;

/// Materialised params for one cell. Always carries the postcard
/// blob (even on the default-path case where it's just the empty /
/// production-defaults serialisation).
#[derive(Clone, Debug)]
pub struct ParamsRecord {
    /// Postcard bytes that the runner installed (or would have
    /// installed). Empty if no override was specified AND the
    /// default-serialise mode is off.
    pub blob: Vec<u8>,
    /// True if the override was actually installed via
    /// `RuntimeTuning::install_from_postcard_file`.
    pub installed: bool,
}

/// Read + install (best-effort) the runtime tuning override.
///
/// - If `path` is `None`: returns an empty blob, `installed = false`.
/// - If `path` is `Some(p)`: reads the file, captures bytes, calls
///   [`jxl_encoder::tuning::runtime::install_from_postcard_file`]. If
///   the install fails because something else has already installed
///   this process-lifetime (the runner is single-cell-per-process so
///   this should never happen in practice), we return the captured
///   blob with `installed = false` and let the runner continue. If
///   the file is unreadable or postcard-corrupt, we return an error
///   so the runner aborts the cell.
pub fn materialise_params(path: Option<&Path>) -> Result<ParamsRecord, String> {
    let Some(path) = path else {
        return Ok(ParamsRecord {
            blob: Vec::new(),
            installed: false,
        });
    };
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    // Validate the bytes deserialise BEFORE installing — postcard
    // requires a known struct shape. The install path also validates,
    // but doing it here gives us a clean error message before mutating
    // the process-global state.
    let _: jxl_encoder::tuning::runtime::RuntimeTuning = postcard::from_bytes(&bytes)
        .map_err(|e| format!("postcard decode {}: {e}", path.display()))?;
    let installed = match jxl_encoder::tuning::runtime::install_from_postcard_file(path) {
        Ok(()) => true,
        Err(e) => {
            // Single-shot installer — if this process already installed,
            // log and proceed without re-installing. The blob is still
            // captured in the row for join-on-params downstream.
            eprintln!("[w44-212] runtime tuning install warning (continuing): {e}");
            false
        }
    };
    Ok(ParamsRecord {
        blob: bytes,
        installed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_path_yields_empty_blob() {
        let r = materialise_params(None).unwrap();
        assert!(r.blob.is_empty());
        assert!(!r.installed);
    }

    #[test]
    fn rejects_invalid_blob() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"NOT_POSTCARD").unwrap();
        let r = materialise_params(Some(tmp.path()));
        assert!(r.is_err(), "invalid postcard should error");
    }

    /// W44-211 [`jxl_encoder::tuning::runtime::RuntimeTuning`] only
    /// derives `serde::Deserialize`, not `Serialize`. The sweep
    /// generator (Python / Rust) MUST emit postcard bytes via its
    /// own writer mirroring the field order. For W44-212 runner
    /// validation we ship a hand-rolled "all-defaults" postcard blob
    /// for the 6 fields currently in the struct.
    ///
    /// Field order (from W44-211 `src/tuning.rs::RuntimeTuning`):
    /// 1. smart_zenjxl_photo_mask_p25_min     (f32 = 85.0)
    /// 2. screenshot_median_threshold         (f32 = 95.0)
    /// 3. buttloop_default_screenshot_qf_seed_scale  (f32 = 4.0)
    /// 4. buttloop_qf_seed_scale_min_distance (f32 = 3.5)
    /// 5. adaptive_quant_screenshot_qf_seed_scale_e5_e6 (f32 = 2.0)
    /// 6. adaptive_quant_screenshot_qf_seed_scale_e7 (f32 = 3.0)
    ///
    /// postcard encodes f32 as 4 bytes little-endian.
    #[test]
    fn hand_built_default_blob_decodes() {
        fn f32_le(x: f32) -> [u8; 4] {
            x.to_le_bytes()
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&f32_le(85.0));
        bytes.extend_from_slice(&f32_le(95.0));
        bytes.extend_from_slice(&f32_le(4.0));
        bytes.extend_from_slice(&f32_le(3.5));
        bytes.extend_from_slice(&f32_le(2.0));
        bytes.extend_from_slice(&f32_le(3.0));
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        let r = materialise_params(Some(tmp.path())).unwrap();
        assert_eq!(r.blob, bytes);
        // installed may or may not be true (single-shot global,
        // test ordering). Either way the call returned Ok.
    }
}
