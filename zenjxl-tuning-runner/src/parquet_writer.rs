// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later.
#![forbid(unsafe_code)]

//! Single-row Parquet emit.
//!
//! Per W44-212 design, each cell produces ONE Parquet file with ONE
//! row. The fleet coordinator (W44-213) concatenates these into per-
//! sweep canonical Parquet via `pyarrow.parquet.ParquetWriter` or
//! `cargo run -p zen-merge`.
//!
//! Why single-row files vs streaming:
//! - Crash isolation: a cell that segfaults mid-encode doesn't lose
//!   ALL prior rows in the same chunk.
//! - Atomic R2 upload: one file → one PUT request, no part-uploads.
//! - The Parquet overhead (~3-5 KB header per file) is fine because
//!   the row itself is ~500 B compressed; ~4 KB total per cell is
//!   negligible against the ~100 KB encoded.jxl.
//!
//! ## Schema versioning
//!
//! See [`SCHEMA_VERSION`]. Bump when adding columns; downstream
//! merge tools handle additive columns gracefully (missing → null).

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, Float32Array, Float64Array, RecordBatch, StringArray, UInt8Array,
    UInt32Array, UInt64Array,
};
// Needed in tests; re-export at module scope so `cargo test` doesn't
// warn about an unused import behind `#[cfg(test)]`.
#[cfg(test)]
use arrow_array::Array;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::spec::SweepCellRow;

/// Bumped any time the column schema gains/loses columns. Downstream
/// merge tools assert `schema_version == 1` for now.
pub const SCHEMA_VERSION: u32 = 1;

/// Write one [`SweepCellRow`] as a single-row Parquet file with zstd
/// compression. Returns the bytes written on success.
pub fn write_single_row_parquet(row: &SweepCellRow, path: &Path) -> Result<u64, String> {
    let schema = Arc::new(build_schema());
    let arrays = build_arrays(row);
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| format!("RecordBatch::try_new: {e}"))?;

    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_created_by(format!(
            "zenjxl-tuning-runner v{}",
            env!("CARGO_PKG_VERSION")
        ))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| format!("ArrowWriter::try_new: {e}"))?;
    writer
        .write(&batch)
        .map_err(|e| format!("ArrowWriter::write: {e}"))?;
    let _meta = writer
        .close()
        .map_err(|e| format!("ArrowWriter::close: {e}"))?;

    let bytes = std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|e| format!("stat {}: {e}", path.display()))?;
    Ok(bytes)
}

/// Build the Parquet/Arrow schema. ~40 columns.
fn build_schema() -> Schema {
    Schema::new(vec![
        // Identity / provenance
        Field::new("schema_version", DataType::UInt32, false),
        Field::new("sweep_id", DataType::Utf8, false),
        Field::new("chunk_claim_id", DataType::Utf8, false),
        Field::new("image_sha256", DataType::Utf8, false),
        Field::new("image_path", DataType::Utf8, false),
        Field::new("image_w", DataType::UInt32, false),
        Field::new("image_h", DataType::UInt32, false),
        // Inputs
        Field::new("effort", DataType::UInt8, false),
        Field::new("distance", DataType::Float32, false),
        Field::new("strategy", DataType::Utf8, false),
        Field::new("params_blob", DataType::Binary, false),
        // Features
        Field::new("feat_m3_colourfulness", DataType::Float32, false),
        Field::new("feat_fcbr", DataType::Float32, false),
        Field::new("feat_edge_density", DataType::Float32, false),
        Field::new("feat_luma_var", DataType::Float32, false),
        Field::new("feat_mask_p25", DataType::Float32, false),
        Field::new("feat_mask_median", DataType::Float32, false),
        Field::new("feat_mask_p75", DataType::Float32, false),
        Field::new("feat_luma_mean", DataType::Float32, false),
        Field::new("feat_n_pixels", DataType::Float32, false),
        Field::new("feat_aspect", DataType::Float32, false),
        Field::new("feat_bpp_source", DataType::Float32, false),
        Field::new("feat_byte_entropy_bits", DataType::Float32, false),
        // Output bytes
        Field::new("encoded_bytes", DataType::UInt32, false),
        // Quality (nullable; backend column is non-null with reason)
        Field::new("ssim2", DataType::Float32, true),
        Field::new("ssim2_backend", DataType::Utf8, false),
        Field::new("butter_norm3", DataType::Float32, true),
        Field::new("butter_norm3_backend", DataType::Utf8, false),
        Field::new("cvvdp", DataType::Float32, true),
        Field::new("cvvdp_backend", DataType::Utf8, false),
        // CPU cost
        Field::new("encode_ms", DataType::Float64, false),
        Field::new("encode_user_ms", DataType::UInt64, false),
        Field::new("encode_sys_ms", DataType::UInt64, false),
        Field::new("encode_peak_rss_mb", DataType::UInt32, false),
        Field::new("encode_threads", DataType::UInt8, false),
        Field::new("decode_ms", DataType::Float64, false),
        Field::new("decode_peak_rss_mb", DataType::UInt32, false),
        // GPU cost
        Field::new("gpu_peak_vram_mb", DataType::UInt32, false),
        Field::new("gpu_kernel_ms", DataType::Float64, false),
        // Provenance
        Field::new("runner_host", DataType::Utf8, false),
        Field::new("gpu_model", DataType::Utf8, false),
        Field::new("commit_sha", DataType::Utf8, false),
        Field::new("runner_version", DataType::Utf8, false),
    ])
}

fn build_arrays(row: &SweepCellRow) -> Vec<ArrayRef> {
    let arr_str = |s: &str| -> ArrayRef { Arc::new(StringArray::from(vec![s])) };
    let arr_u32 = |x: u32| -> ArrayRef { Arc::new(UInt32Array::from(vec![x])) };
    let arr_u8 = |x: u8| -> ArrayRef { Arc::new(UInt8Array::from(vec![x])) };
    let arr_u64 = |x: u64| -> ArrayRef { Arc::new(UInt64Array::from(vec![x])) };
    let arr_f32 = |x: f32| -> ArrayRef { Arc::new(Float32Array::from(vec![x])) };
    let arr_f64 = |x: f64| -> ArrayRef { Arc::new(Float64Array::from(vec![x])) };
    let arr_f32_opt = |x: Option<f32>| -> ArrayRef { Arc::new(Float32Array::from(vec![x])) };
    let arr_bin = |b: &[u8]| -> ArrayRef { Arc::new(BinaryArray::from(vec![b])) };

    vec![
        arr_u32(SCHEMA_VERSION),
        arr_str(&row.sweep_id),
        arr_str(&row.chunk_claim_id),
        arr_str(&row.image_sha256),
        arr_str(&row.image_path),
        arr_u32(row.image_w),
        arr_u32(row.image_h),
        arr_u8(row.effort),
        arr_f32(row.distance),
        arr_str(&row.strategy),
        arr_bin(&row.params_blob),
        arr_f32(row.features.m3_colourfulness),
        arr_f32(row.features.flat_color_block_ratio),
        arr_f32(row.features.edge_density),
        arr_f32(row.features.luma_var),
        arr_f32(row.features.mask_p25),
        arr_f32(row.features.mask_median),
        arr_f32(row.features.mask_p75),
        arr_f32(row.features.luma_mean),
        arr_f32(row.features.n_pixels),
        arr_f32(row.features.aspect),
        arr_f32(row.features.bpp_source),
        arr_f32(row.features.byte_entropy_bits),
        arr_u32(row.encoded_bytes),
        arr_f32_opt(row.ssim2),
        arr_str(&row.ssim2_backend),
        arr_f32_opt(row.butter_norm3),
        arr_str(&row.butter_norm3_backend),
        arr_f32_opt(row.cvvdp),
        arr_str(&row.cvvdp_backend),
        arr_f64(row.encode_ms),
        arr_u64(row.encode_user_ms),
        arr_u64(row.encode_sys_ms),
        arr_u32(row.encode_peak_rss_mb),
        arr_u8(row.encode_threads),
        arr_f64(row.decode_ms),
        arr_u32(row.decode_peak_rss_mb),
        arr_u32(row.gpu_peak_vram_mb),
        arr_f64(row.gpu_kernel_ms),
        arr_str(&row.runner_host),
        arr_str(&row.gpu_model),
        arr_str(&row.commit_sha),
        arr_str(&row.runner_version),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ExtendedFeatures;

    #[test]
    fn schema_has_43_columns() {
        let s = build_schema();
        // If you add a column, BUMP `SCHEMA_VERSION` and update this
        // count. The downstream merge tool reads this count to detect
        // additive schemas vs breaking schemas.
        assert_eq!(s.fields().len(), 43, "schema column count drift");
    }

    #[test]
    fn build_arrays_returns_one_per_field() {
        let row = SweepCellRow {
            sweep_id: "test".into(),
            chunk_claim_id: "claim1".into(),
            image_sha256: "deadbeef".into(),
            image_path: "/tmp/test.png".into(),
            image_w: 32,
            image_h: 32,
            effort: 7,
            distance: 1.0,
            strategy: "zenjxl".into(),
            params_blob: vec![1, 2, 3],
            features: ExtendedFeatures::default(),
            encoded_bytes: 1024,
            ssim2: Some(95.0),
            ssim2_backend: "gpu-cli-ssim2".into(),
            butter_norm3: Some(0.4),
            butter_norm3_backend: "gpu-cli-butteraugli-pnorm3".into(),
            cvvdp: None,
            cvvdp_backend: "skip".into(),
            encode_ms: 42.0,
            encode_user_ms: 35,
            encode_sys_ms: 5,
            encode_peak_rss_mb: 128,
            encode_threads: 8,
            decode_ms: 12.0,
            decode_peak_rss_mb: 64,
            gpu_peak_vram_mb: 512,
            gpu_kernel_ms: 6.0,
            runner_host: "test-host".into(),
            gpu_model: "test-gpu".into(),
            commit_sha: "abc123".into(),
            runner_version: env!("CARGO_PKG_VERSION").into(),
        };
        let arrays = build_arrays(&row);
        assert_eq!(arrays.len(), 43);
        for a in &arrays {
            assert_eq!(a.len(), 1, "every array must have exactly one row");
        }
    }

    #[test]
    fn writes_to_disk() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let row = SweepCellRow {
            sweep_id: "test".into(),
            chunk_claim_id: "claim1".into(),
            image_sha256: "deadbeef".into(),
            image_path: "/tmp/test.png".into(),
            image_w: 32,
            image_h: 32,
            effort: 7,
            distance: 1.0,
            strategy: "zenjxl".into(),
            params_blob: vec![],
            features: ExtendedFeatures::default(),
            encoded_bytes: 1024,
            ssim2: Some(95.0),
            ssim2_backend: "skip".into(),
            butter_norm3: None,
            butter_norm3_backend: "skip".into(),
            cvvdp: None,
            cvvdp_backend: "skip".into(),
            encode_ms: 42.0,
            encode_user_ms: 35,
            encode_sys_ms: 5,
            encode_peak_rss_mb: 128,
            encode_threads: 8,
            decode_ms: 12.0,
            decode_peak_rss_mb: 64,
            gpu_peak_vram_mb: 0,
            gpu_kernel_ms: 0.0,
            runner_host: "test-host".into(),
            gpu_model: "test-gpu".into(),
            commit_sha: "abc123".into(),
            runner_version: env!("CARGO_PKG_VERSION").into(),
        };
        let bytes = write_single_row_parquet(&row, &path).unwrap();
        assert!(bytes > 0);
        assert!(path.exists());
    }
}
