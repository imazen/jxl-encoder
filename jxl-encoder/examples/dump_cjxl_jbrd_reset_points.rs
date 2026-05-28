//! Extract and parse the JBRD box from a JXL file (typically cjxl output)
//! and dump per-scan reset_points / extra_zero_runs counts.
//!
//! Stops parsing at the brotli-compressed tail (no brotli decode needed —
//! reset_points/extra_zero_runs are in the bit-packed prefix).
//!
//! Used to establish GROUND TRUTH for task #12 (jxl-encoder/zenjpeg):
//! the brief claimed pete-walls's 31583 reset_points were spurious; this
//! tool reads what cjxl actually emits so we can verify.

use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dump_cjxl_jbrd_reset_points <jxl_file> [<jxl_file> ...]");
        process::exit(2);
    }
    println!("file\tnum_scans\ttotal_reset_points\ttotal_extra_zero_runs");
    for path in &args[1..] {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("read error {path}: {e}");
                continue;
            }
        };
        let Some((box_off, box_size)) = find_jbrd_box(&bytes) else {
            eprintln!("no jbrd box in {path}");
            continue;
        };
        let payload = &bytes[box_off + 8..box_off + box_size];

        match dump_jbrd_prefix(payload) {
            Ok((scans, total_rp, total_ezr)) => {
                println!("{path}\t{scans}\t{total_rp}\t{total_ezr}");
            }
            Err(e) => eprintln!("parse error {path}: {e}"),
        }
    }
}

fn find_jbrd_box(bytes: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let size_be = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap());
        let typ = &bytes[i + 4..i + 8];
        let mut box_size = size_be as usize;
        if box_size == 0 {
            box_size = bytes.len() - i;
        } else if box_size == 1 {
            if i + 16 > bytes.len() {
                return None;
            }
            let large = u64::from_be_bytes(bytes[i + 8..i + 16].try_into().unwrap()) as usize;
            box_size = large;
        }
        if typ == b"jbrd" {
            return Some((i, box_size));
        }
        if box_size < 8 {
            return None;
        }
        i += box_size;
    }
    None
}

/// Parse the bit-packed JBRD prefix up through the per-scan reset_points
/// and extra_zero_runs lists. Returns (num_scans, total_reset_points, total_extra_zero_runs).
fn dump_jbrd_prefix(payload: &[u8]) -> Result<(usize, usize, usize), String> {
    let mut br = BitReader::new(payload);
    let is_gray = br.read(1)? == 1;
    let num_components: u32 = if is_gray { 1 } else { 3 };

    let mut num_scans = 0u32;
    let mut num_app = 0u32;
    let mut num_com = 0u32;
    let mut has_dri = false;
    loop {
        let marker = br.read(6)? as u8 + 0xC0;
        match marker {
            0xD9 => break,
            0xDA => num_scans += 1,
            0xE0..=0xEF => num_app += 1,
            0xFE => num_com += 1,
            0xDD => has_dri = true,
            _ => {}
        }
    }

    for _ in 0..num_app {
        let _ = read_u32_jbrd(&mut br, &[0, 1], &[(1, 2), (2, 4)])?;
        let _ = br.read(16)?;
    }
    for _ in 0..num_com {
        let _ = br.read(16)?;
    }
    let num_quant = read_u32_jbrd(&mut br, &[1, 2, 3, 4], &[])?;
    for _ in 0..num_quant {
        let _ = br.read(1)?;
        let _ = br.read(2)?;
        let _ = br.read(1)?;
    }
    let comp_type = br.read(2)?;
    if comp_type == 3 {
        let num_comp = read_u32_jbrd(&mut br, &[1, 2, 3, 4], &[])?;
        for _ in 0..num_comp {
            let _ = br.read(8)?;
        }
    }
    for _ in 0..num_components {
        let _ = br.read(2)?;
    }

    // Huffman codes
    let num_huff = read_u32_jbrd(&mut br, &[4], &[(3, 2), (4, 10), (6, 26)])?;
    for _ in 0..num_huff {
        let _ = br.read(1)?;
        let _ = br.read(2)?;
        let _ = br.read(1)?;
        let _ = read_u32_jbrd(&mut br, &[0, 1], &[(3, 2), (8, 0)])?;
        let mut counts = [0u32; 16];
        let mut max_depth_idx = 0;
        for (i, count) in counts.iter_mut().enumerate() {
            *count = read_u32_jbrd(&mut br, &[0, 1], &[(3, 2), (8, 0)])?;
            if *count > 0 {
                max_depth_idx = i;
            }
        }
        if counts[max_depth_idx] == 0 {
            return Err("huffman with no symbols".into());
        }
        counts[max_depth_idx] -= 1;
        let num_symbols: u32 = counts.iter().sum::<u32>() + 1;
        for _ in 0..num_symbols {
            let _ = read_u32_jbrd(&mut br, &[], &[(2, 0), (2, 4), (4, 8), (8, 1)])?;
        }
    }
    let mut scan_meta = Vec::new();
    for _ in 0..num_scans {
        let scan_nc = read_u32_jbrd(&mut br, &[1, 2, 3, 4], &[])?;
        let ss = br.read(6)? as u8;
        let se = br.read(6)? as u8;
        let al = br.read(4)? as u8;
        let ah = br.read(4)? as u8;
        for _ in 0..scan_nc {
            let _ = br.read(2)?;
            let _ = br.read(2)?;
            let _ = br.read(2)?;
        }
        let _last_needed_pass = read_u32_jbrd(&mut br, &[0, 1, 2], &[(3, 3)])?;
        scan_meta.push((ss, se, ah, al));
    }
    if has_dri {
        let _ = br.read(16)?;
    }
    let mut total_rp: usize = 0;
    let mut total_ezr: usize = 0;
    let dump_indices = std::env::var_os("DUMP_INDICES").is_some();
    for (i, &(ss, se, ah, al)) in scan_meta.iter().enumerate() {
        let num_rp = read_u32_jbrd(&mut br, &[0], &[(2, 1), (4, 4), (16, 20)])?;
        let mut rp_indices: Vec<u32> = Vec::new();
        // libjxl jpeg_data.cc:313-325 — encoding does `block_idx -= last_block_idx + 1`
        // before writing, where `last_block_idx` starts at -1. So the first entry
        // decodes via `block_idx = diff + 0`, subsequent via `block_idx = diff + prev + 1`.
        let mut last_block_idx_signed: i64 = -1;
        for _ in 0..num_rp {
            let diff = read_u32_jbrd(&mut br, &[0], &[(3, 1), (5, 9), (28, 41)])?;
            let block_idx = (last_block_idx_signed + 1 + diff as i64) as u32;
            rp_indices.push(block_idx);
            last_block_idx_signed = block_idx as i64;
        }
        let num_ezr = read_u32_jbrd(&mut br, &[0], &[(2, 1), (4, 4), (16, 20)])?;
        let mut ezr_indices: Vec<(u32, u32)> = Vec::new();
        let mut last_ezr_idx_signed: i64 = -1;
        for _ in 0..num_ezr {
            let runs = read_u32_jbrd(&mut br, &[1], &[(2, 2), (4, 5), (8, 20)])?;
            let diff = read_u32_jbrd(&mut br, &[0], &[(3, 1), (5, 9), (28, 41)])?;
            let block_idx = (last_ezr_idx_signed + 1 + diff as i64) as u32;
            ezr_indices.push((block_idx, runs));
            last_ezr_idx_signed = block_idx as i64;
        }
        eprintln!(
            "  scan[{i}] ss={ss} se={se} ah={ah} al={al} reset_points={num_rp} extra_zero_runs={num_ezr}"
        );
        if dump_indices {
            for v in &rp_indices {
                eprintln!("    RP scan={i} mcu={v}");
            }
            for (v, r) in &ezr_indices {
                eprintln!("    EZR scan={i} mcu={v} runs={r}");
            }
        }
        total_rp += num_rp as usize;
        total_ezr += num_ezr as usize;
    }
    Ok((scan_meta.len(), total_rp, total_ezr))
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }
    fn read(&mut self, nbits: u32) -> Result<u32, String> {
        let mut value: u32 = 0;
        for i in 0..nbits {
            if self.byte_pos >= self.data.len() {
                return Err(format!("EOF at byte {} after {i}/{nbits} bits", self.byte_pos));
            }
            let bit = ((self.data[self.byte_pos] >> self.bit_pos) & 1) as u32;
            value |= bit << i;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(value)
    }
}

fn read_u32_jbrd(
    br: &mut BitReader,
    vals: &[u32],
    bits_offsets: &[(u32, u32)],
) -> Result<u32, String> {
    let sel = br.read(2)? as usize;
    if sel < vals.len() {
        return Ok(vals[sel]);
    }
    let slot = sel - vals.len();
    if slot >= bits_offsets.len() {
        return Err(format!(
            "u32 selector {sel} out of range (vals={} bo={})",
            vals.len(),
            bits_offsets.len()
        ));
    }
    let (nbits, offset) = bits_offsets[slot];
    if nbits == 0 {
        return Ok(offset);
    }
    Ok(br.read(nbits)? + offset)
}
