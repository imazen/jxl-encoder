//! Multi-strategy `strategy_def!` with one bool gate (env hook) and
//! one enum gate (env hook with custom parser). Should compile.

fn main() {
    // Drive the generated code so unused-code lints don't pollute
    // the trybuild output.
    let _ = MultiEncoderStrategy::default();
    let _ = MultiResolvedImprovements::libjxl();
    let _ = MultiResolvedImprovements::zenjxl();
    let _ = MultiResolvedImprovements::lean_faster();
    let _ = MultiResolvedImprovements::aggressive();
    let _ = MultiResolvedImprovements::from_custom(&MultiEncoderImprovements::default());
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Knob {
    #[default]
    Off,
    On,
}

fn bool_one(s: &str) -> Option<bool> {
    if s == "1" { Some(true) } else { None }
}

fn parse_knob(s: &str) -> Option<Knob> {
    match s {
        "off" => Some(Knob::Off),
        "on" => Some(Knob::On),
        _ => None,
    }
}

jxl_encoder_macros::strategy_def! {
    name = Multi;
    default_strategy = Zenjxl;

    enums {}

    strategies {
        Libjxl { flag = true, knob = Knob::On, },
        Zenjxl { flag = false, knob = Knob::Off, },
        LeanFaster { flag = false, knob = Knob::Off, },
        Aggressive { flag = false, knob = Knob::On, },
    }

    gates {
        flag: bool {
            env_hook = "MY_FLAG" => bool_one,
            divergence_section = "C",
            divergence_row_ref = "Multi/flag (test)",
        },
        knob: Knob {
            env_hook = "MY_KNOB" => parse_knob,
        },
    }
}
