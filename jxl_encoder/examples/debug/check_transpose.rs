#![allow(unused)]
use jxl_enc::vardct::tokenize::ZIGZAG_ORDER_8X8;

fn main() {
    println!("ZIGZAG_ORDER_8X8 interpretation:");
    println!("  ZIGZAG[1] = {} = row {}, col {}", ZIGZAG_ORDER_8X8[1], ZIGZAG_ORDER_8X8[1]/8, ZIGZAG_ORDER_8X8[1]%8);
    println!("  ZIGZAG[2] = {} = row {}, col {}", ZIGZAG_ORDER_8X8[2], ZIGZAG_ORDER_8X8[2]/8, ZIGZAG_ORDER_8X8[2]%8);
    
    println!("\nIf we interpret row=v (vertical freq), col=u (horizontal freq):");
    println!("  ZIGZAG[1] gives freq ({},{}) - {} frequency", 
        ZIGZAG_ORDER_8X8[1]%8, ZIGZAG_ORDER_8X8[1]/8,
        if ZIGZAG_ORDER_8X8[1]/8 == 0 { "horizontal" } else { "diagonal/other" });
    println!("  ZIGZAG[2] gives freq ({},{}) - {} frequency", 
        ZIGZAG_ORDER_8X8[2]%8, ZIGZAG_ORDER_8X8[2]/8,
        if ZIGZAG_ORDER_8X8[2]%8 == 0 { "vertical" } else { "diagonal/other" });
    
    println!("\nBut JXL natural order (dx, dy) where grid uses (x, y):");
    println!("  Position 1: (1, 0) -> x=1, y=0");
    println!("  Position 2: (0, 1) -> x=0, y=1");
    println!();
    println!("If x=column, y=row in grid storage:");
    println!("  grid[y*8+x] for (x=1,y=0) -> index 0*8+1 = 1");
    println!("  grid[y*8+x] for (x=0,y=1) -> index 1*8+0 = 8");
    println!();
    println!("But what if JXL means (x,y) = (v,u) instead of (u,v)?");
    println!("  Position 1: x=1, y=0 = (v=1, u=0) = VERTICAL freq at index 8");
    println!("  Position 2: x=0, y=1 = (v=0, u=1) = HORIZONTAL freq at index 1");
    println!();
    println!("Then ZIGZAG would need to be:");
    println!("  Position 1: index 8 (not 1)");
    println!("  Position 2: index 1 (not 8)");
    
    println!("\n=== What if we swap the interpretation? ===");
    println!("Maybe the issue is (x,y) means (row,col) not (col,row)?");
    println!("In that case, x=vertical, y=horizontal:");
    println!("  JXL position 1: (1,0) = row 1, col 0 -> index 8");
    println!("  JXL position 2: (0,1) = row 0, col 1 -> index 1");
    println!();
    println!("Our ZIGZAG:");
    println!("  Position 1: index 1");
    println!("  Position 2: index 8");
    println!();
    println!("That's SWAPPED!");
}
