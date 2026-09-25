//! Dump the strict libjxl-parity InvDequantMatrix tables (same record
//! format as cjxl's JXL_QM_DUMP): {magic,strat,c,n} u32x4 then n f32.
#[cfg(feature = "__internals")]
fn main() {
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/ours_qm.bin").unwrap();
    for strat in 0..19usize {
        for c in 0..3usize {
            let t = jxl_encoder::__internals::inv_dequant_matrix_lj(strat, c);
            let hdr = [0x514d5442u32, strat as u32, c as u32, t.len() as u32];
            for w in hdr {
                f.write_all(&w.to_le_bytes()).unwrap();
            }
            for v in t {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
}
#[cfg(not(feature = "__internals"))]
fn main() {}
