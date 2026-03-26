#![allow(unused)]
use jxl_encoder::entropy_coding::hybrid_uint::HybridUintConfig;

fn main() {
    // The config used in write_pass_group_clustered
    let hybrid_config = HybridUintConfig::new(4, 2, 0);
    
    // Test encoding the values we care about
    let test_values = [0u32, 4, 5, 7, 25, 297];
    
    println!("HybridUint encoding test (config: split_exp=4, msb=2, lsb=0):");
    for &val in &test_values {
        let (token, extra_bits, num_extra_bits) = hybrid_config.encode(val);
        println!("  value {:4} -> token={:2}, extra_bits=0b{:016b}, num_extra_bits={}", 
                 val, token, extra_bits, num_extra_bits);
        
        // Calculate what the decoder would see
        if num_extra_bits == 0 {
            println!("           -> decoder reads symbol {}, no extra bits", token);
        } else {
            println!("           -> decoder reads symbol {}, then {} extra bits", token, num_extra_bits);
        }
    }
    
    // What's the max token value?
    let max_packed = 297;
    let (token, _, _) = hybrid_config.encode(max_packed);
    println!("\nMax token for value {}: {}", max_packed, token);
    println!("This means alphabet_size needs to be >= {}", token + 1);
}
