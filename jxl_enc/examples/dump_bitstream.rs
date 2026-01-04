//! Bitstream dumper for JXL files.
//!
//! Parses a JXL file and shows the bitstream structure with bit positions,
//! similar to the trace output from our encoder.
//!
//! Usage:
//!   cargo run --example dump_bitstream <file.jxl>

use jxl::bit_reader::BitReader;
use std::env;
use std::fs;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file.jxl>", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];
    let mut file = fs::File::open(filename)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    println!("=== JXL Bitstream Dump: {} ===", filename);
    println!("File size: {} bytes\n", data.len());

    dump_bitstream(&data)?;

    Ok(())
}

fn dump_bitstream(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = BitReader::new(&data);

    // Parse container/signature
    println!("=== SIGNATURE ===");
    let sig1 = reader.read(8)?;
    println!("[{:6}] signature[0]: 0x{:02x} ({})",
        reader.total_bits_read() - 8, sig1,
        if sig1 == 0xFF { "0xFF - JXL codestream" } else { "unexpected" });

    let sig2 = reader.read(8)?;
    println!("[{:6}] signature[1]: 0x{:02x} ({})",
        reader.total_bits_read() - 8, sig2,
        if sig2 == 0x0A { "0x0A - JXL codestream" } else { "unexpected" });

    if sig1 == 0xFF && sig2 == 0x0A {
        println!("→ Bare codestream (no container)\n");
        dump_codestream(&mut reader)?;
    } else if sig1 == 0x00 && sig2 == 0x00 {
        println!("→ Container format detected");
        dump_container(&mut reader)?;
    } else {
        println!("→ Unknown format\n");
    }

    Ok(())
}

fn dump_container(reader: &mut BitReader) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== CONTAINER ===");

    // Read "JXLC" or remaining signature bytes
    let b3 = reader.read(8)?;
    let b4 = reader.read(8)?;
    let b5 = reader.read(8)?;
    let b6 = reader.read(8)?;
    let b7 = reader.read(8)?;
    let b8 = reader.read(8)?;
    let b9 = reader.read(8)?;
    let b10 = reader.read(8)?;
    let b11 = reader.read(8)?;

    println!("[{:6}] rest of signature: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
        reader.total_bits_read() - 72, b3, b4, b5, b6, b7, b8, b9, b10, b11);

    // Parse boxes (simplified - just show structure)
    println!("\n=== BOXES ===");
    println!("(Box parsing not fully implemented - look for 'jxlc' or 'jxlp' boxes)");

    // For now, just show hex dump of next 128 bytes
    let remaining = std::cmp::min(128, reader.total_bits_available() / 8);
    print!("[{:6}] next {} bytes: ", reader.total_bits_read(), remaining);
    for _ in 0..remaining {
        print!("{:02x} ", reader.read(8)?);
    }
    println!("\n");

    Ok(())
}

fn dump_codestream(reader: &mut BitReader) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FILE HEADER ===");

    // Read size header
    let pos = reader.total_bits_read();
    let size_header = reader.read(16)?;
    println!("[{:6}] size_header: {} ({:#018b})", pos, size_header, size_header);

    // Decode DIY (Div8) flags
    let div8 = (size_header >> 15) & 1;
    println!("  → div8: {}", div8);

    let ysize_div8 = (size_header >> 12) & 7;
    let xsize_div8 = (size_header >> 9) & 7;
    println!("  → ysize_div8: {}, xsize_div8: {}", ysize_div8, xsize_div8);

    // Parse dimensions based on size_header
    let (width, height) = if size_header == 0 {
        // Custom size
        let w = read_u32_0bits(reader)?;
        let h = read_u32_0bits(reader)?;
        println!("  → Custom size: {}x{}", w, h);
        (w, h)
    } else {
        // Encoded in size_header
        let h_ratio = (ysize_div8 + 1) as u32;
        let w_code = xsize_div8;

        let width = if w_code == 0 {
            h_ratio
        } else {
            1 + w_code as u32
        };

        println!("  → Encoded size: {}x{}", width, h_ratio);
        (width, h_ratio)
    };

    // Read ImageMetadata
    println!("\n=== IMAGE METADATA ===");
    let pos = reader.total_bits_read();
    let all_default = reader.read(1)?;
    println!("[{:6}] all_default: {}", pos, all_default);

    if all_default == 0 {
        // Read extra_fields
        let pos = reader.total_bits_read();
        let extra_fields = reader.read(1)?;
        println!("[{:6}] extra_fields: {}", pos, extra_fields);

        // Read orientation (if present)
        if extra_fields != 0 {
            let pos = reader.total_bits_read();
            let orientation = reader.read(3)?;
            println!("[{:6}] orientation: {} ({:03b})", pos, orientation, orientation);
        }

        // Read more metadata fields...
        println!("  (remaining metadata fields not fully parsed)");

        // Show next 32 bits for inspection
        let pos = reader.total_bits_read();
        let next_bits = reader.read(32)?;
        println!("[{:6}] next 32 bits: {:#034b}", pos, next_bits);
        // Jump to byte boundary
        reader.jump_to_byte_boundary()?;
    }

    println!("\n=== FRAME HEADER (first frame) ===");
    let pos = reader.total_bits_read();

    // Try to read frame header
    let all_default = reader.read(1)?;
    println!("[{:6}] frame.all_default: {}", pos, all_default);

    if all_default == 0 {
        let pos = reader.total_bits_read();
        let frame_type = reader.read(2)?;
        println!("[{:6}] frame_type: {} ({})", pos, frame_type,
            match frame_type {
                0 => "RegularFrame",
                1 => "LFFrame",
                2 => "ReferenceOnly",
                3 => "SkipProgressive",
                _ => "unknown"
            });

        let pos = reader.total_bits_read();
        let encoding = reader.read(1)?;
        println!("[{:6}] encoding: {} ({})", pos, encoding,
            if encoding == 0 { "VarDCT" } else { "Modular" });

        // Show more frame header fields
        println!("  (remaining frame header fields not fully parsed)");
    }

    println!("\n[{:6}] ... (end of partial parse)", reader.total_bits_read());
    println!("\nRemaining: {} bits ({} bytes)",
        reader.total_bits_available(), reader.total_bits_available() / 8);

    Ok(())
}

// Read U32(0, 0, 0, BitsOffset(30, 0)) - used for custom dimensions
fn read_u32_0bits(reader: &mut BitReader) -> Result<u32, Box<dyn std::error::Error>> {
    let pos = reader.total_bits_read();
    let selector = reader.read(2)?;

    let value = match selector {
        0 => 0,
        1 => 0,
        2 => 0,
        3 => {
            let bits = reader.read(30)?;
            bits as u32
        }
        _ => unreachable!()
    };

    println!("[{:6}] U32(0,0,0,BitsOffset(30,0)): selector={}, value={}", pos, selector, value);
    Ok(value)
}
