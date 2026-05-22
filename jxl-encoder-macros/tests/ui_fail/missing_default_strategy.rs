//! Omitting `default_strategy = ...` is an error.

fn main() {}

jxl_encoder_macros::strategy_def! {
    name = Missing;

    enums {}

    strategies {
        Zenjxl { x = true, },
    }

    gates {
        x: bool {},
    }
}
