//! A strategy variant that doesn't list one of the declared gates is
//! an error.

fn main() {}

jxl_encoder_macros::strategy_def! {
    name = MissGate;
    default_strategy = Zenjxl;

    enums {}

    strategies {
        Zenjxl { x = true, /* missing y */ },
    }

    gates {
        x: bool {},
        y: bool {},
    }
}
