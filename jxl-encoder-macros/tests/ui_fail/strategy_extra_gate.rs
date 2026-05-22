//! A strategy variant that lists a gate not declared in `gates {}` is
//! an error.

fn main() {}

jxl_encoder_macros::strategy_def! {
    name = ExtraGate;
    default_strategy = Zenjxl;

    enums {}

    strategies {
        Zenjxl { x = true, surprise = false, },
    }

    gates {
        x: bool {},
    }
}
