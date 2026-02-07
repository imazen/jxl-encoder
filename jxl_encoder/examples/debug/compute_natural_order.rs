//! Compute JXL natural order and compare with our ZIGZAG_ORDER_8X8

use jxl_encoder::vardct::tokenize::ZIGZAG_ORDER_8X8;

fn main() {
    // Compute JXL natural order for 8x8 using the jxl-oxide algorithm
    // From jxl-oxide/crates/jxl-vardct/src/hf_pass.rs
    let (bw, bh) = (8usize, 8usize);
    let y_scale = bw / bh; // 1
    let lbw = bw / 8; // 1
    let lbh = bh / 8; // 1
    
    let mut natural_order: Vec<(usize, usize)> = Vec::new();
    
    // First: LF region (DC)
    for idx in 0..(lbw * lbh) {
        let x = idx % lbw;
        let y = idx / lbw;
        natural_order.push((x, y));
    }
    
    // Second: AC region in diagonal scan
    for dist in 1..(2 * bw) {
        let margin = dist.saturating_sub(bw);
        for order in margin..(dist - margin) {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            
            // Skip LF region
            if x < lbw && y < lbw {
                continue;
            }
            // Skip y positions that don't align with y_scale
            if y % y_scale != 0 {
                continue;
            }
            natural_order.push((x, y / y_scale));
        }
    }
    
    println!("JXL Natural Order for 8x8 (first 20 positions):");
    println!("  pos -> (x, y) where x=horizontal, y=vertical");
    for (i, (x, y)) in natural_order.iter().enumerate().take(20) {
        let linear_idx = y * 8 + x;  // Convert (x,y) to linear index assuming row-major
        println!("  {:2}: ({}, {}) -> linear idx {} (zigzag: {})", 
                 i, x, y, linear_idx, ZIGZAG_ORDER_8X8[i]);
    }
    
    // Compare with our ZIGZAG_ORDER_8X8
    println!("\nComparing natural order linear idx vs ZIGZAG_ORDER_8X8:");
    let mut mismatches = 0;
    for (i, (x, y)) in natural_order.iter().enumerate() {
        let linear_idx = y * 8 + x;
        if linear_idx != ZIGZAG_ORDER_8X8[i] {
            println!("  MISMATCH at pos {}: natural=({},{})={} vs zigzag={}", 
                     i, x, y, linear_idx, ZIGZAG_ORDER_8X8[i]);
            mismatches += 1;
        }
    }
    
    if mismatches == 0 {
        println!("  All 64 positions match!");
    } else {
        println!("\nTotal mismatches: {}", mismatches);
    }
    
    // The natural order gives (x, y) coordinates
    // x is horizontal (column), y is vertical (row)
    // In our DCT output: dct[v*8+u] where v=row=vertical, u=col=horizontal
    // So (x, y) -> linear index = y*8+x = v*8+u if y=v and x=u
    // This means natural order (x,y) maps to frequency (u=x, v=y)
}
