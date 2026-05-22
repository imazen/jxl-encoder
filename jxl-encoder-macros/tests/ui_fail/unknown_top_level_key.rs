//! Unknown top-level key in `strategy_def!` is an error.

fn main() {}

jxl_encoder_macros::strategy_def! {
    name = BadKey;
    default_strategy = Zenjxl;
    typo_key = 42;

    enums {}

    strategies {
        Zenjxl { x = true, },
    }

    gates {
        x: bool {},
    }
}
