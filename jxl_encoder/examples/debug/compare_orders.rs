#![allow(unused)]
use jxl_enc::vardct::tokenize::ZIGZAG_ORDER_8X8;

fn jxl_natural_order_8x8() -> Vec<usize> {
    // Computed from jxl-oxide const_compute_natural_order algorithm
    // for bw=8, bh=8
    let lbw = 1; // 8/8
    let y_scale = 1; // 8/8
    
    let mut order = Vec::with_capacity(64);
    
    // Position 0: DC at (0, 0)
    order.push(0);
    
    // Diagonal scan for HF coefficients
    for dist in 2..16 { // 2*8 = 16
        let margin = if dist > 8 { dist - 8 } else { 0 };
        for o in margin..(dist - margin) {
            let (x, y) = if dist % 2 == 1 {
                (o, dist - 1 - o)
            } else {
                (dist - 1 - o, o)
            };
            
            // Skip LLF positions (x < lbw && y < lbw)
            if x < lbw && y < lbw {
                continue;
            }
            
            // Skip if y not divisible by y_scale
            if y % y_scale != 0 {
                continue;
            }
            
            if x < 8 && y < 8 {
                let idx = y * 8 + x; // (x, y) -> row-major index
                order.push(idx);
            }
        }
    }
    
    order
}

fn main() {
    let jxl_order = jxl_natural_order_8x8();
    
    println!("Comparing ZIGZAG_ORDER_8X8 vs JXL natural order:");
    println!("{:>4} {:>7} {:>7} {:>5}", "pos", "ZIGZAG", "JXL", "match");
    println!("{}", "-".repeat(30));
    
    for pos in 0..20 {
        let zig = ZIGZAG_ORDER_8X8[pos];
        let jxl = jxl_order[pos];
        let zig_v = zig / 8;
        let zig_u = zig % 8;
        let jxl_v = jxl / 8;
        let jxl_u = jxl % 8;
        let matched = if zig == jxl { "✓" } else { "✗" };
        println!("{:4} {:3} ({},{}) {:3} ({},{})   {}", 
            pos, zig, zig_v, zig_u, jxl, jxl_v, jxl_u, matched);
    }
    
    // Find first difference
    for pos in 0..64 {
        if ZIGZAG_ORDER_8X8[pos] != jxl_order[pos] {
            println!("\nFirst difference at position {}: ZIGZAG={}, JXL={}", 
                pos, ZIGZAG_ORDER_8X8[pos], jxl_order[pos]);
            break;
        }
    }
}
