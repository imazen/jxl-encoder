/// Trace prefix code bits for different alphabet sizes

fn bit_reverse(value: u32, len: u8) -> u32 {
    if len == 0 {
        return 0;
    }
    let mut result = 0u32;
    let mut v = value;
    for _ in 0..len {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

fn write_complex_prefix_code_trace(alphabet_size: usize) {
    println!(
        "\n=== Tracing prefix code for alphabet_size={} ===",
        alphabet_size
    );

    let d = (usize::BITS - (alphabet_size - 1).leading_zeros()) as usize;
    let pow2_d = 1usize << d;

    let symbols_short = pow2_d.saturating_sub(alphabet_size);
    let symbols_long = alphabet_size.saturating_sub(symbols_short);

    let (depth_short, depth_long) = if d <= 1 { (1, 1) } else { (d - 1, d) };

    println!(
        "d={}, symbols_short={} (depth {}), symbols_long={} (depth {})",
        d, symbols_short, depth_short, symbols_long, depth_long
    );

    const STORAGE_ORDER: [u8; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    let pos_short = STORAGE_ORDER
        .iter()
        .position(|&x| x == depth_short as u8)
        .unwrap_or(0);
    let pos_long = STORAGE_ORDER
        .iter()
        .position(|&x| x == depth_long as u8)
        .unwrap_or(0);
    let min_pos = pos_short.min(pos_long);
    let max_pos = pos_short.max(pos_long);

    let skip = if min_pos >= 3 {
        3
    } else if min_pos >= 2 {
        2
    } else {
        0
    };

    println!(
        "pos_short={}, pos_long={}, skip={}",
        pos_short, pos_long, skip
    );

    // Simulate writing
    println!("\nBits written:");
    println!("  hskip = {} (2 bits): {:02b}", skip, skip);

    const CL_CODE_BITS: [u8; 6] = [0b00, 0b0111, 0b011, 0b10, 0b01, 0b1111];
    const CL_CODE_LENS: [u8; 6] = [2, 4, 3, 2, 2, 4];

    let mut bits_written = 2; // hskip
    let mut space = 32i32;

    println!(
        "  Code-length-code-lengths (positions {} to {}):",
        skip, max_pos
    );
    for (idx, &storage_val) in STORAGE_ORDER[skip..=max_pos].iter().enumerate() {
        let cl_cl = if storage_val as usize == depth_short || storage_val as usize == depth_long {
            1u8
        } else {
            0u8
        };
        let bits = CL_CODE_BITS[cl_cl as usize];
        let len = CL_CODE_LENS[cl_cl as usize];
        println!(
            "    storage[{}]={}: cl_cl={}, bits={:0w$b} ({} bits)",
            skip + idx,
            storage_val,
            cl_cl,
            bits,
            len,
            w = len as usize
        );
        bits_written += len as usize;
        if cl_cl != 0 {
            space -= 32 >> cl_cl;
        }
    }
    println!("  Space after cl-cl: {} (should be 0)", space);

    // Code length bits
    let (first_cl, _second_cl) = if pos_short < pos_long {
        (depth_short, depth_long)
    } else {
        (depth_long, depth_short)
    };

    println!("  Code lengths ({} bits, one per symbol):", alphabet_size);
    println!(
        "    Symbols 0-{}: depth {} -> code {} ({} zeros)",
        symbols_short.saturating_sub(1),
        depth_short,
        if depth_short == first_cl { 0 } else { 1 },
        symbols_short
    );
    println!(
        "    Symbols {}-{}: depth {} -> code {} ({} ones)",
        symbols_short,
        alphabet_size - 1,
        depth_long,
        if depth_long == first_cl { 0 } else { 1 },
        symbols_long
    );
    bits_written += alphabet_size;

    println!("\nTotal prefix code bits: {}", bits_written);

    // Show what Huffman codes result
    println!("\nResulting Huffman codes:");
    let mut code = 0u32;
    let mut prev_len = 0u8;
    for sym in 0..alphabet_size.min(60) {
        let len = if sym < symbols_short {
            depth_short
        } else {
            depth_long
        };
        if len as u8 > prev_len {
            code <<= (len as u8) - prev_len;
        }
        let rev_code = bit_reverse(code, len as u8);
        if sym < 15 || sym >= alphabet_size - 5 {
            println!(
                "  sym {:2}: depth={}, canonical={:0w$b}, reversed={:0w$b} ({})",
                sym,
                len,
                code,
                rev_code,
                rev_code,
                w = len
            );
        } else if sym == 15 {
            println!("  ...");
        }
        code += 1;
        prev_len = len as u8;
    }
}

fn main() {
    // Test alphabet sizes around the failure boundary
    for &size in &[46, 50, 53, 54, 55] {
        write_complex_prefix_code_trace(size);
    }
}
