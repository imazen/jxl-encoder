//! Minimal well-formed `strategy_def!` invocation: one strategy
//! (Zenjxl), one gate (bool), no env hook. Should compile cleanly.

fn main() {}

jxl_encoder_macros::strategy_def! {
    name = Minimal;
    default_strategy = Zenjxl;

    enums {}

    strategies {
        Zenjxl { the_only_gate = true, },
    }

    gates {
        the_only_gate: bool {},
    }
}
