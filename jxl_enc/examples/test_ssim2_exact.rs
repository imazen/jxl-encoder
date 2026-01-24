//! Test SSIMULACRA2 with exact values from failing test
use fast_ssim2::compute_frame_ssimulacra2;
use fast_ssim2::{ColorPrimaries, Rgb, TransferCharacteristic};

fn main() {
    let width = 300;
    let height = 300;

    // Create original horizontal gradient EXACTLY like the test
    let original: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            let val = (x * 255 / width) as f32 / 255.0;
            [val, val, val]
        })
        .collect();

    // Sample values from test output:
    // (x=0,y=0): orig=[0.000,0.000,0.000] dec=[0.000,0.000,0.000]
    // (x=50,y=0): orig=[0.165,0.165,0.165] dec=[0.129,0.129,0.129]
    // (x=100,y=0): orig=[0.333,0.333,0.333] dec=[0.353,0.353,0.353]
    // (x=150,y=0): orig=[0.498,0.498,0.498] dec=[0.557,0.557,0.557]
    // (x=200,y=0): orig=[0.667,0.667,0.667] dec=[0.620,0.620,0.620]
    // (x=250,y=0): orig=[0.831,0.831,0.831] dec=[0.796,0.796,0.796]
    // (x=299,y=0): orig=[0.996,0.996,0.996] dec=[1.000,1.000,1.000]

    // Verify original values are correct
    println!("Original[0]: {:?}", original[0]);
    println!("Original[50]: {:?}", original[50]);
    println!("Original[100]: {:?}", original[100]);

    // Create decoded by interpolating the known values
    let known_points = [
        (0, 0.000),
        (50, 0.129),
        (100, 0.353),
        (150, 0.557),
        (200, 0.620),
        (250, 0.796),
        (299, 1.000),
    ];

    let decoded: Vec<[f32; 3]> = (0..(width * height))
        .map(|i| {
            let x = i % width;
            // Find surrounding known points
            let mut val = 0.0f32;
            for j in 0..known_points.len() - 1 {
                if x >= known_points[j].0 && x <= known_points[j + 1].0 {
                    let t = (x - known_points[j].0) as f32
                        / (known_points[j + 1].0 - known_points[j].0) as f32;
                    val = known_points[j].1 * (1.0 - t) + known_points[j + 1].1 * t;
                    break;
                }
            }
            [val, val, val]
        })
        .collect();

    println!("Decoded[0]: {:?}", decoded[0]);
    println!("Decoded[50]: {:?}", decoded[50]);
    println!("Decoded[100]: {:?}", decoded[100]);

    // RMSE
    let mut sum_sq = 0.0f64;
    for (o, d) in original.iter().zip(decoded.iter()) {
        sum_sq += (o[0] - d[0]).powi(2) as f64;
    }
    let rmse = (sum_sq / (width * height) as f64).sqrt();
    println!("RMSE: {:.4}", rmse);

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
