use jxl_enc::vardct::quant_weights::generate_dct8_weights;

fn main() {
    let weights = generate_dct8_weights();
    
    println!("DCT8 weights for Y channel (should decrease with distance from DC):");
    println!("Position  Weight   Distance");
    println!("--------  -------  --------");
    
    for y in 0..8 {
        for x in 0..8 {
            let idx = 64 + y * 8 + x;  // Y channel offset
            let dist = ((x*x + y*y) as f32).sqrt();
            if x + y <= 4 {  // Just show first few
                println!("({},{})      {:.2}     {:.2}", x, y, weights[idx], dist);
            }
        }
    }
    
    // What about B channel?
    println!("\nB channel weights:");
    for pos in [0, 1, 7, 8, 63] {
        println!("  B[{}] = {:.2}", pos, weights[128 + pos]);
    }
}
