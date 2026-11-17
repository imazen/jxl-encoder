//! Retain actual primary-decoder observations for gamma attribution controls.
#[path = "../tests/support/decoded_extras.rs"]
mod decoded_extras;
fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "color_gamma_decode <input.jxl> <new-output> [--single]"
    );
    if args.len() == 4 {
        assert_eq!(args[3], "--single");
    }
    let input = std::path::Path::new(&args[1]);
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir(out).unwrap();
    let data = std::fs::read(input).unwrap();
    let frames = if args.len() == 4 {
        vec![decoded_extras::decode_current_planes(
            &data,
            "gamma-attribution",
        )]
    } else {
        decoded_extras::decode_current_frames(&data, "gamma-attribution")
    };
    let mut observations = Vec::new();
    for (index, (color, extras)) in frames.into_iter().enumerate() {
        std::fs::write(
            out.join(format!("primary-{index}.f32")),
            color
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        observations.push(serde_json::json!({"frame":index,"samples":color.len(),"first_rgb":&color[..color.len().min(3)],
            "minimum":color.iter().copied().fold(f32::INFINITY,f32::min),
            "maximum":color.iter().copied().fold(f32::NEG_INFINITY,f32::max),
            "maximum_white_error":color.iter().map(|v|(v-1.0).abs()).fold(0_f32,f32::max),
            "extra_planes":extras.len()}));
    }
    std::fs::write(
        out.join("observed.json"),
        serde_json::to_vec_pretty(&observations).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string(&observations).unwrap());
}
