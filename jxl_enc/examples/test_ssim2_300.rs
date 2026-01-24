//! Test SSIMULACRA2 with 300x300 images
use fast_ssim2::compute_frame_ssimulacra2;
use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic};

fn main() {
    let width = 300;
    let height = 300;

    // Create original horizontal gradient
    let original: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let val = (x * 255 / width) as f32 / 255.0;
            [val, val, val]
        })
        .collect();

    // Create decoded with ~4% RMSE (matching test)
    let decoded: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let base_val = (x * 255 / width) as f32 / 255.0;
            // Add random-ish error to get ~4% RMSE
            let noise = ((x * 37 + y * 13) % 100) as f32 / 100.0 * 0.1 - 0.05;
            let val = (base_val + noise).clamp(0.0, 1.0);
            [val, val, val]
        })
        .collect();

    // Compute stats
    let mut diff_sq_sum = 0.0f64;
    for (o, d) in original.iter().zip(decoded.iter()) {
        diff_sq_sum += ((o[0] - d[0]) as f64).powi(2);
    }
    let rmse = (diff_sq_sum / (width * height) as f64).sqrt();
    println!("RMSE: {:.4}", rmse);

    // Create Rgb structures
    let source = Rgb::new(
        original.clone(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to create source");

    let distorted = Rgb::new(
        decoded.clone(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed to create distorted");

    let ssim2 = compute_frame_ssimulacra2(source, distorted).expect("SSIM2 failed");
    println!("SSIM2 with noise: {:.2}", ssim2);

    // Now test with identical images - should be 100
    let source2 = Rgb::new(
        original.clone(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed");
    let distorted2 = Rgb::new(
        original.clone(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("Failed");

    let ssim2_identical = compute_frame_ssimulacra2(source2, distorted2).expect("SSIM2 failed");
    println!("SSIM2 identical: {:.2}", ssim2_identical);
}
