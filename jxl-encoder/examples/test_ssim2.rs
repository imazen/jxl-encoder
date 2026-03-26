//! Test SSIMULACRA2 calculation directly
use fast_ssim2::compute_frame_ssimulacra2;
use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic};

fn main() {
    let width = 300;
    let height = 300;

    // Create original horizontal gradient
    let original: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let val = x as f32 * 255.0 / width as f32 / 255.0;
            [val, val, val]
        })
        .collect();

    // Create decoded (simulated ~5% error)
    let decoded: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let base_val = x as f32 * 255.0 / width as f32 / 255.0;
            // Add some error similar to what we saw
            let error = (x as f32 / 300.0 - 0.5) * 0.1;
            let val = (base_val + error).clamp(0.0, 1.0);
            [val, val, val]
        })
        .collect();

    println!("Original[0]: {:?}", original[0]);
    println!("Original[100]: {:?}", original[100]);
    println!("Original[299]: {:?}", original[299]);
    println!("Decoded[0]: {:?}", decoded[0]);
    println!("Decoded[100]: {:?}", decoded[100]);
    println!("Decoded[299]: {:?}", decoded[299]);

    // Create Rgb structures
    let source = Rgb::new(
        original,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to create source");

    let distorted = Rgb::new(
        decoded,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to create distorted");

    let ssim2 = compute_frame_ssimulacra2(source, distorted).expect("SSIM2 failed");
    println!("SSIM2: {:.2}", ssim2);
}
