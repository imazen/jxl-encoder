//! `default_strategy = X;` where `X` isn't one of the declared
//! strategies is an error.

fn main() {}

jxl_encoder_macros::strategy_def! {
    name = BadDefault;
    default_strategy = NotAStrategy;

    enums {}

    strategies {
        Zenjxl { x = true, },
    }

    gates {
        x: bool {},
    }
}
