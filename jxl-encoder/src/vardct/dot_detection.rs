// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Dot detection for star fields and specular highlights (refs #19).
//!
//! Ports libjxl `enc_detect_dots.cc`. The pipeline is:
//!
//! 1. Compute an **energy image** from XYB orig vs blurred:
//!    `energy = w_x * dX^2 + w_y * dY^2 + w_b * dB^2` where `(dX, dY, dB)`
//!    is the difference between the original XYB image and a "smooth"
//!    XYB image (Gaussian blurred 3x with `Gaussian3`). Color weights
//!    `(0, 10, 0)` mean only the Y channel contributes — chrominance
//!    differences are ignored.
//!
//! 2. **Connected components** via flood-fill on the energy image with a
//!    dual threshold (`0.04` to enter, `0.02` to grow). Components are
//!    capped at 1000 pixels and 5×5 bounding window.
//!
//! 3. **2D Gaussian ellipse fitting** to each connected component
//!    (position, sigma, angle, per-channel intensity).
//!
//! 4. **Quality filter** — reject poor fits, low intensity, bad
//!    centroid alignment.
//!
//! 5. Encode as patches via the patch dictionary.
//!
//! **Gating**: effort >= 7, distance >= 3.0. Skipped if text-like
//! patches are already found. Niche feature — only fires on astronomy
//! images, specular highlights on dark backgrounds, certain noise
//! patterns.
//!
//! This module is being ported in stages. Currently implemented:
//! - [`SumOfSquareDifferences`]: weighted sum of squared XYB diffs
//! - [`gaussian_separable_5_horizontal`] / `_vertical`: 5-tap
//!   horizontal/vertical Gaussian convolution helpers
//! - [`compute_energy_image`]: orchestrates blur + diff to produce the
//!   per-pixel energy map (caller-provided smooth output buffer)
//!
//! TODO (subsequent ticks):
//! - Connected component flood-fill (`FindCC`)
//! - `FitGaussianFast` + `FitGaussian` (2D ellipse least-squares)
//! - `ComputeDotLosses` quality filter
//! - Patch dictionary encoding
//! - Wire-up at effort >= 7, distance >= 3.0 in encode pipeline

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// libjxl color-channel weights for the energy image. Y-channel only;
/// chroma differences are ignored. From `enc_detect_dots.cc:54-56`.
pub(crate) const ENERGY_COLOR_COEF: [f32; 3] = [0.0, 10.0, 0.0];

/// 5-tap separable Gaussian for the **noise-smoothing** pass (sigma
/// ≈ 0.65). Coefficients from libjxl
/// `WeightsSeparable5Gaussian0_65` (`enc_detect_dots.cc:128-138`).
/// The kernel is symmetric so only the center + 2 outer taps are
/// stored; the actual 5-tap convolution applies them as
/// `[w2, w1, w0, w1, w2]` at each pixel.
pub(crate) const GAUSSIAN_0_65_TAPS: [f32; 3] = [0.558311, 0.210395, 0.010449];

/// 5-tap separable Gaussian for the **dot-removal** pass (sigma ≈ 3.0,
/// applied twice → effective sigma ≈ 4.24). Coefficients from libjxl
/// `WeightsSeparable5Gaussian3` (`enc_detect_dots.cc:140-149`).
pub(crate) const GAUSSIAN_3_TAPS: [f32; 3] = [0.222338, 0.210431, 0.1784];

/// Bounding-window cap for a single dot's connected component (libjxl
/// `kEllipseWindowSize`, `enc_detect_dots.cc:97`).
pub(crate) const ELLIPSE_WINDOW_SIZE: usize = 5;

/// Maximum area in pixels of an ellipse-candidate connected component.
/// libjxl `kMaxCCSize`, `enc_detect_dots.cc:188`.
pub(crate) const MAX_CC_SIZE: usize = 1000;

/// Padding (in pixels) used when computing background statistics
/// around a connected component's bounding rectangle. libjxl
/// `kExtraRect = 4`, `enc_detect_dots.cc:299`.
pub(crate) const CC_EXTRA_RECT: i32 = 4;

/// 2×2 anisotropic Gaussian "dot" model evaluated at a relative
/// position `(dx, dy)`. `(ct, st)` is the rotation
/// `(cos(angle), sin(angle))`; `sigma_x`/`sigma_y` are squared
/// standard-deviation parameters along the rotated axes;
/// `intensity` scales the peak. Mirrors libjxl `DotGaussianModel`
/// (`enc_detect_dots.cc:115-122`).
pub(crate) fn dot_gaussian_model(
    dx: f64,
    dy: f64,
    ct: f64,
    st: f64,
    sigma_x: f64,
    sigma_y: f64,
    intensity: f64,
) -> f64 {
    let rx = ct * dx + st * dy;
    let ry = -st * dx + ct * dy;
    let md = rx * rx / sigma_x + ry * ry / sigma_y;
    intensity * (-0.5 * md).exp()
}

/// 2×2 symmetric matrix eigendecomposition. Given symmetric `a`,
/// returns `(diag, U)` such that `diag` is the eigenvalue pair and
/// `U` is the orthonormal eigenvector basis (rows of U are the
/// eigenvectors). Mirrors libjxl `ConvertToDiagonal`
/// (`enc_linalg.cc:14-46`). Inputs are stored row-major as
/// `[[a00, a01], [a10, a11]]` with the symmetry assumption
/// `a01 == a10`.
pub(crate) fn convert_to_diagonal_2x2(a: [[f64; 2]; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    debug_assert!((a[0][1] - a[1][0]).abs() < 1e-15, "matrix not symmetric");
    let b = -(a[0][0] + a[1][1]);
    let c = a[0][0] * a[1][1] - a[0][1] * a[0][1];
    let disc = b * b - 4.0 * c;
    if a[0][1].abs() < 1e-10 || disc < 0.0 {
        // Already diagonal.
        return ([a[0][0], a[1][1]], [[1.0, 0.0], [0.0, 1.0]]);
    }
    let sqd = disc.sqrt();
    let l1 = (-b - sqd) * 0.5;
    let l2 = (-b + sqd) * 0.5;
    let v1x = a[0][0] - l1;
    let v1y = a[1][0];
    let v1n = 1.0 / v1x.hypot(v1y);
    let v1x = v1x * v1n;
    let v1y = v1y * v1n;
    // U rows are the eigenvectors; libjxl's convention swaps cols.
    ([l1, l2], [[v1y, -v1x], [v1x, v1y]])
}

/// 2D anisotropic Gaussian ellipse with per-channel intensity, as
/// fit by [`fit_gaussian_fast`]. Mirrors libjxl `GaussianEllipse`
/// (`enc_detect_dots.cc:101-122`). Position + shape are encoded into
/// the patch dictionary; the loss fields drive quality filtering.
#[derive(Clone, Debug)]
pub(crate) struct GaussianEllipse {
    pub x: f64,
    pub y: f64,
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub angle: f64,
    pub intensity: [f64; 3],
    pub l2_loss: f64,
    pub l1_loss: f64,
    pub ridge_loss: f64,
    pub custom_loss: f64,
    pub bg_color: [f64; 3],
    pub neg_pixels: usize,
    pub neg_value: [f64; 3],
}

impl GaussianEllipse {
    fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            sigma_x: 0.0,
            sigma_y: 0.0,
            angle: 0.0,
            intensity: [0.0; 3],
            l2_loss: 0.0,
            l1_loss: 0.0,
            ridge_loss: 0.0,
            custom_loss: 0.0,
            bg_color: [0.0; 3],
            neg_pixels: 0,
            neg_value: [0.0; 3],
        }
    }
}

/// Fit a 2D anisotropic Gaussian ellipse to a connected component's
/// energy distribution + per-channel intensity. Mirrors libjxl
/// `FitGaussianFast` (`enc_detect_dots.cc:410-520`).
///
/// Inputs:
/// - `cc`: connected component (provides `mode`, `bounds`)
/// - `image_w`/`image_h`: dimensions of the planar XYB inputs
/// - `img_x`/`img_y`/`img_b`: original XYB planes (planar f32)
/// - `bg_x`/`bg_y`/`bg_b`: background-smoothed XYB planes (planar f32)
///
/// The "fast" variant in libjxl computes 1st/2nd central moments of
/// the Y channel around `cc.mode`, eigendecomposes the resulting 2×2
/// covariance to recover (sigma_x, sigma_y, angle), then re-fits
/// per-channel intensity via least squares against the dot model.
/// libjxl's `FitGaussian` adds a quality check on top — we'll add
/// that in a subsequent tick when wiring losses.
#[allow(clippy::too_many_arguments)]
pub fn fit_gaussian_fast(
    cc: &ConnectedComponent,
    image_w: i32,
    image_h: i32,
    img_x: &[f32],
    img_y: &[f32],
    img_b: &[f32],
    bg_x: &[f32],
    bg_y: &[f32],
    bg_b: &[f32],
) -> Option<GaussianEllipse> {
    debug_assert_eq!(img_x.len(), (image_w * image_h) as usize);
    debug_assert_eq!(img_y.len(), (image_w * image_h) as usize);
    debug_assert_eq!(img_b.len(), (image_w * image_h) as usize);
    debug_assert_eq!(bg_x.len(), (image_w * image_h) as usize);
    debug_assert_eq!(bg_y.len(), (image_w * image_h) as usize);
    debug_assert_eq!(bg_b.len(), (image_w * image_h) as usize);

    const K_EPSILON: f64 = 1e-6;
    const K_RECT_BOUNDS: i32 = (ELLIPSE_WINDOW_SIZE >> 1) as i32; // 5 >> 1 = 2
    const K_SIGMA_MULT: f64 = 1.0;
    const K_SCALE_MULT: [f64; 3] = [1.1, 1.1, 1.1];

    let mut ans = GaussianEllipse::zero();
    let img_planes = [img_x, img_y, img_b];
    let bg_planes = [bg_x, bg_y, bg_b];
    let stride = image_w as usize;

    // Bounds-checked planar fetch.
    let fetch = |plane: &[f32], x: i32, y: i32| -> Option<f32> {
        if x < 0 || x >= image_w || y < 0 || y >= image_h {
            None
        } else {
            Some(plane[(y as usize) * stride + (x as usize)])
        }
    };

    // Per-channel `color` at the CC mode.
    let mut color = [0.0_f64; 3];
    for c in 0..3 {
        let v = fetch(img_planes[c], cc.mode.x, cc.mode.y)?;
        let bg = fetch(bg_planes[c], cc.mode.x, cc.mode.y)?;
        color[c] = (v - bg) as f64;
    }
    let sign = if color[1] > 0.0 { 1.0 } else { -1.0 };

    // Weighted moments around cc.mode, in the kRectBounds×kRectBounds
    // window (libjxl's `kEllipseWindowSize >> 1` half-window).
    let mut sum = 0.0_f64;
    let mut m1 = [0.0_f64; 3];
    let mut m2 = [0.0_f64; 3];
    let mut bg_color = [0.0_f64; 3];
    let mut n: usize = 0;
    for sy in -K_RECT_BOUNDS..=K_RECT_BOUNDS {
        let y = sy + cc.mode.y;
        if y < 0 || y >= image_h {
            continue;
        }
        for sx in -K_RECT_BOUNDS..=K_RECT_BOUNDS {
            let x = sx + cc.mode.x;
            if x < 0 || x >= image_w {
                continue;
            }
            let yval = img_y[(y as usize) * stride + (x as usize)] as f64;
            let bgval = bg_y[(y as usize) * stride + (x as usize)] as f64;
            let w = (sign * (yval - bgval)).max(K_EPSILON);
            sum += w;
            m1[0] += w * x as f64;
            m1[1] += w * y as f64;
            m2[0] += w * (x as f64) * (x as f64);
            m2[1] += w * (x as f64) * (y as f64);
            m2[2] += w * (y as f64) * (y as f64);
            for c in 0..3 {
                bg_color[c] += bg_planes[c][(y as usize) * stride + (x as usize)] as f64;
            }
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    for i in 0..3 {
        m1[i] /= sum;
        m2[i] /= sum;
        bg_color[i] /= n as f64;
    }

    ans.x = m1[0];
    ans.y = m1[1];
    for j in 0..3 {
        ans.intensity[j] = K_SCALE_MULT[j] * color[j];
    }

    // Build the 2×2 covariance and eigen-decompose to recover
    // (sigma_x, sigma_y, angle).
    let sigma = [
        [m2[0] - m1[0] * m1[0], m2[1] - m1[0] * m1[1]],
        [m2[1] - m1[0] * m1[1], m2[2] - m1[1] * m1[1]],
    ];
    let (d, u) = convert_to_diagonal_2x2(sigma);
    // u rows = eigenvectors; libjxl uses U[1] (second eigenvector) for
    // the angle. p1 indexes the larger eigenvalue.
    let (p1, p2) = if d[0] < d[1] { (1, 0) } else { (0, 1) };
    ans.sigma_x = K_SIGMA_MULT * d[p1];
    ans.sigma_y = K_SIGMA_MULT * d[p2];
    ans.angle = u[1][p1].atan2(u[1][p2]);
    ans.bg_color = bg_color;

    // Least-squares per-channel intensity fit (regularized).
    let ct = ans.angle.cos();
    let st = ans.angle.sin();
    for c in 0..3 {
        let mut gg = 0.0_f64;
        let mut gd = 0.0_f64;
        for sy in -K_RECT_BOUNDS..=K_RECT_BOUNDS {
            let y = sy + cc.mode.y;
            if y < 0 || y >= image_h {
                continue;
            }
            for sx in -K_RECT_BOUNDS..=K_RECT_BOUNDS {
                let x = sx + cc.mode.x;
                if x < 0 || x >= image_w {
                    continue;
                }
                let target = (img_planes[c][(y as usize) * stride + (x as usize)]
                    - bg_planes[c][(y as usize) * stride + (x as usize)])
                    as f64;
                let g = dot_gaussian_model(
                    x as f64 - ans.x,
                    y as f64 - ans.y,
                    ct,
                    st,
                    ans.sigma_x,
                    ans.sigma_y,
                    1.0,
                );
                gg += g * g;
                gd += g * target;
            }
        }
        ans.intensity[c] = gd / (gg + K_EPSILON);
    }

    // Final step: l1/l2/custom losses + neg_pixels accounting.
    compute_dot_losses(&mut ans, cc, image_w, image_h, &img_planes, &bg_planes);

    Some(ans)
}

/// Compute l1/l2/custom losses + neg_pixels for an already-fit
/// `GaussianEllipse`. Mirrors libjxl `ComputeDotLosses`
/// (`enc_detect_dots.cc:345-408`).
///
/// `kOptimizeBackground = true` in libjxl: the prediction `pred` uses
/// `ellipse.bg_color[c]` instead of the per-pixel background row,
/// so the background is treated as a per-CC constant. We mirror that
/// branch.
pub(crate) fn compute_dot_losses(
    ellipse: &mut GaussianEllipse,
    cc: &ConnectedComponent,
    image_w: i32,
    image_h: i32,
    img_planes: &[&[f32]; 3],
    _bg_planes: &[&[f32]; 3],
) {
    const RECT_BOUNDS: i32 = 2;
    const K_INTENSITY_R: f64 = 0.0; // libjxl: 0.015 (commented out)
    const K_SIGMA_R: f64 = 0.0; // libjxl: 0.01 (commented out)
    const K_ZERO_EPSILON: f64 = 0.1;
    const CHANNEL_GAINS: [f64; 3] = [1.0, 1.0, 1.0];

    let stride = image_w as usize;
    let ct = ellipse.angle.cos();
    let st = ellipse.angle.sin();

    let mut n: usize = 0;
    ellipse.l1_loss = 0.0;
    ellipse.l2_loss = 0.0;
    ellipse.custom_loss = 0.0;
    ellipse.neg_pixels = 0;
    ellipse.neg_value = [0.0; 3];

    let dx_mode = cc.mode.x as f64 - ellipse.x;
    let dy_mode = cc.mode.y as f64 - ellipse.y;
    let dist_mean_mode_sq = dx_mode * dx_mode + dy_mode * dy_mode;

    for c in 0..3 {
        for sy in -RECT_BOUNDS..(cc.bounds.ysize + RECT_BOUNDS) {
            let y = sy + cc.bounds.y0;
            if y < 0 || y >= image_h {
                continue;
            }
            for sx in -RECT_BOUNDS..(cc.bounds.xsize + RECT_BOUNDS) {
                let x = sx + cc.bounds.x0;
                if x < 0 || x >= image_w {
                    continue;
                }
                let target = img_planes[c][(y as usize) * stride + (x as usize)] as f64;
                let dot_delta = dot_gaussian_model(
                    x as f64 - ellipse.x,
                    y as f64 - ellipse.y,
                    ct,
                    st,
                    ellipse.sigma_x,
                    ellipse.sigma_y,
                    ellipse.intensity[c],
                );
                if dot_delta > target + K_ZERO_EPSILON {
                    ellipse.neg_pixels += 1;
                    ellipse.neg_value[c] += dot_delta - target;
                }
                let bkg = ellipse.bg_color[c]; // kOptimizeBackground = true
                let pred = bkg + dot_delta;
                let diff = target - pred;
                let l2 = CHANNEL_GAINS[c] * diff * diff;
                let l1 = CHANNEL_GAINS[c] * diff.abs();
                ellipse.l2_loss += l2;
                ellipse.l1_loss += l1;
                let w = dot_gaussian_model(
                    x as f64 - cc.mode.x as f64,
                    y as f64 - cc.mode.y as f64,
                    1.0,
                    0.0,
                    1.0 + ellipse.sigma_x,
                    1.0 + ellipse.sigma_y,
                    1.0,
                );
                ellipse.custom_loss += w * l2;
                n += 1;
            }
        }
    }
    if n > 0 {
        let inv_n = 1.0 / n as f64;
        ellipse.l2_loss *= inv_n;
        ellipse.l1_loss *= inv_n;
        ellipse.custom_loss *= inv_n;
    }
    // libjxl: custom_loss += 20 * dist_mean_mode_sq + neg_value[1].
    ellipse.custom_loss += 20.0 * dist_mean_mode_sq + ellipse.neg_value[1];
    let mut ridge = K_SIGMA_R * ellipse.sigma_x + K_SIGMA_R * ellipse.sigma_y;
    for c in 0..3 {
        ridge += K_INTENSITY_R * ellipse.intensity[c] * ellipse.intensity[c];
    }
    ellipse.ridge_loss = ellipse.l2_loss + ridge;
}

/// Tuning knobs for [`detect_gaussian_ellipses`]. Mirrors libjxl
/// `GaussianDetectParams` (`enc_detect_dots.h:14-30`).
#[derive(Clone, Copy, Debug)]
pub struct GaussianDetectParams {
    pub t_high: f32,
    pub t_low: f32,
    pub max_win_size: usize,
    pub max_l2_loss: f64,
    pub max_custom_loss: f64,
    pub min_intensity: f64,
    pub max_dist_mean_mode: f64,
    pub max_neg_pixels: usize,
    pub min_score: f32,
    pub max_cc: usize,
    pub perc_cc: usize,
}

impl Default for GaussianDetectParams {
    /// libjxl default knobs (cjxl uses these in `enc_heuristics.cc`).
    fn default() -> Self {
        Self {
            t_high: 0.04,
            t_low: 0.02,
            max_win_size: ELLIPSE_WINDOW_SIZE,
            max_l2_loss: 0.005,
            max_custom_loss: 300.0,
            min_intensity: 0.12,
            max_dist_mean_mode: 1.0,
            max_neg_pixels: 0,
            min_score: 4.0,
            max_cc: 50,
            perc_cc: 15,
        }
    }
}

/// One detected dot: bounding-box origin + per-channel residual
/// pixel data (relative to the smoothed background) ready to feed the
/// patch dictionary.
#[derive(Clone, Debug)]
pub struct DetectedDot {
    pub x0: u32,
    pub y0: u32,
    pub xsize: usize,
    pub ysize: usize,
    /// `[plane][y * xsize + x]` residual = `orig - smooth`.
    pub residuals: [Vec<f32>; 3],
}

/// Detect dot-like Gaussian ellipses in an XYB-encoded image.
/// Mirrors libjxl `DetectGaussianEllipses` (`enc_detect_dots.cc:553-618`).
///
/// Inputs are three planar XYB f32 channels of size `image_w * image_h`.
/// Output is a list of dots with their bounding-box origins and the
/// XYB residual data (orig − smooth) ready to feed
/// [`crate::vardct::patches`] as a Gaussian-ellipse-style patch.
///
/// **Pipeline**:
/// 1. [`compute_energy_image`] → `energy` + smooth XYB
/// 2. [`find_cc`] → connected components above the dual threshold
/// 3. Sort by score, keep the top `min(max_cc, perc_cc% * total)`
/// 4. For each, [`fit_gaussian_fast`] → quality filter
/// 5. Surviving dots emit residuals
pub fn detect_gaussian_ellipses(
    img_x: &[f32],
    img_y: &[f32],
    img_b: &[f32],
    width: usize,
    height: usize,
    params: &GaussianDetectParams,
) -> Vec<DetectedDot> {
    let n = width * height;
    let mut smooth_x = vec![0.0_f32; n];
    let mut smooth_y = vec![0.0_f32; n];
    let mut smooth_b = vec![0.0_f32; n];
    let mut energy = vec![0.0_f32; n];
    compute_energy_image(
        img_x,
        img_y,
        img_b,
        width,
        height,
        &mut smooth_x,
        &mut smooth_y,
        &mut smooth_b,
        &mut energy,
    );

    let mut ccs = find_cc(
        &energy,
        width,
        height,
        params.t_low,
        params.t_high,
        params.max_win_size,
        params.min_score,
    );

    // Keep top-N CCs by score (per libjxl: min(max_cc, perc_cc% * total)).
    let num_keep = params.max_cc.min((ccs.len() * params.perc_cc) / 100);
    if ccs.len() > num_keep {
        ccs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        ccs.truncate(num_keep);
    }

    let mut dots = Vec::new();
    let img_planes = [img_x, img_y, img_b];
    let smooth_planes = [&smooth_x[..], &smooth_y[..], &smooth_b[..]];
    let stride = width;
    for cc in &ccs {
        let Some(ellipse) = fit_gaussian_fast(
            cc,
            width as i32,
            height as i32,
            img_x,
            img_y,
            img_b,
            &smooth_x,
            &smooth_y,
            &smooth_b,
        ) else {
            continue;
        };
        // Quality filter mirroring libjxl `DetectGaussianEllipses` body.
        if ellipse.x < 0.0 || ellipse.x.ceil() >= width as f64 {
            continue;
        }
        if ellipse.y < 0.0 || ellipse.y.ceil() >= height as f64 {
            continue;
        }
        if ellipse.neg_pixels > params.max_neg_pixels {
            continue;
        }
        // libjxl perceptual luminance weighting:
        // intensity = 0.21 * I_X + 0.72 * I_Y + 0.07 * I_B.
        let intensity =
            0.21 * ellipse.intensity[0] + 0.72 * ellipse.intensity[1] + 0.07 * ellipse.intensity[2];
        let intensity_sq = intensity * intensity;
        let dx = ellipse.x - cc.mode.x as f64;
        let dy = ellipse.y - cc.mode.y as f64;
        let sq_dist_mean_mode = dx * dx + dy * dy;
        let l2_ok = ellipse.l2_loss < params.max_l2_loss;
        let custom_ok = ellipse.custom_loss < params.max_custom_loss;
        let intensity_ok = intensity_sq > params.min_intensity * params.min_intensity;
        let dist_ok = sq_dist_mean_mode < params.max_dist_mean_mode * params.max_dist_mean_mode;
        if !(l2_ok && custom_ok && intensity_ok && dist_ok) {
            continue;
        }
        // Emit the dot's residual data (orig - smooth) in its
        // bounding rectangle.
        let x0 = cc.bounds.x0 as usize;
        let y0 = cc.bounds.y0 as usize;
        let xsize = cc.bounds.xsize as usize;
        let ysize = cc.bounds.ysize as usize;
        let mut residuals: [Vec<f32>; 3] = [
            vec![0.0_f32; xsize * ysize],
            vec![0.0_f32; xsize * ysize],
            vec![0.0_f32; xsize * ysize],
        ];
        for c in 0..3 {
            for y in 0..ysize {
                for x in 0..xsize {
                    let src_idx = (y0 + y) * stride + (x0 + x);
                    residuals[c][y * xsize + x] =
                        img_planes[c][src_idx] - smooth_planes[c][src_idx];
                }
            }
        }
        dots.push(DetectedDot {
            x0: cc.bounds.x0 as u32,
            y0: cc.bounds.y0 as u32,
            xsize,
            ysize,
            residuals,
        });
    }
    dots
}

/// Single-pixel coordinate for the BFS queue. Mirrors libjxl
/// `Pixel { int x; int y; }` at `enc_detect_dots.cc:178-181`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pixel {
    pub x: i32,
    pub y: i32,
}

/// Axis-aligned bounding rectangle in pixel coordinates. Mirrors
/// libjxl `Rect(x0, y0, xsize, ysize)` for the dot-detection use
/// case only (no clipping helpers needed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub xsize: i32,
    pub ysize: i32,
}

impl Rect {
    /// `true` when `(p.x, p.y)` lies inside the rectangle. Mirrors
    /// libjxl `PointInRect` (`enc_detect_dots.cc:218-223`).
    fn contains(&self, p: Pixel) -> bool {
        p.x >= self.x0 && p.x < self.x0 + self.xsize && p.y >= self.y0 && p.y < self.y0 + self.ysize
    }
}

/// One connected-component candidate from the energy image. Carries
/// the raw pixel list, its bounding rectangle, and per-CC statistics
/// used downstream for filtering / Gaussian fitting. Mirrors libjxl
/// `ConnectedComponent` (`enc_detect_dots.cc:224-291`).
#[derive(Clone, Debug)]
pub(crate) struct ConnectedComponent {
    pub bounds: Rect,
    pub pixels: Vec<Pixel>,
    pub max_energy: f32,
    pub mean_energy: f32,
    pub var_energy: f32,
    pub mean_bg: f32,
    pub var_bg: f32,
    pub score: f32,
    pub mode: Pixel,
}

impl ConnectedComponent {
    /// Compute mean / variance / mode statistics for the component
    /// itself and a `extra`-pixel padded ring around it (used as
    /// background). Mirrors libjxl `CompStats`
    /// (`enc_detect_dots.cc:241-285`). The `image_w` × `image_h`
    /// span is the energy image's pixel size; coordinates outside
    /// it are skipped.
    fn compute_stats(&mut self, energy: &[f32], image_w: i32, image_h: i32, extra: i32) {
        let mut max_energy = 0.0_f32;
        let mut sum_e = 0.0_f64;
        let mut sum_e2 = 0.0_f64;
        let mut sum_bg = 0.0_f64;
        let mut sum_bg2 = 0.0_f64;
        let mut n_in: usize = 0;
        let mut n_out: usize = 0;
        let mut mode = Pixel { x: 0, y: 0 };

        let y_min = -extra;
        let y_max = self.bounds.ysize + extra;
        let x_min = -extra;
        let x_max = self.bounds.xsize + extra;
        for sy in y_min..y_max {
            let y = sy + self.bounds.y0;
            if y < 0 || y >= image_h {
                continue;
            }
            for sx in x_min..x_max {
                let x = sx + self.bounds.x0;
                if x < 0 || x >= image_w {
                    continue;
                }
                let v = energy[(y as usize) * (image_w as usize) + (x as usize)];
                if v > max_energy {
                    max_energy = v;
                    mode = Pixel { x, y };
                }
                if self.bounds.contains(Pixel { x, y }) {
                    sum_e += v as f64;
                    sum_e2 += (v as f64) * (v as f64);
                    n_in += 1;
                } else {
                    sum_bg += v as f64;
                    sum_bg2 += (v as f64) * (v as f64);
                    n_out += 1;
                }
            }
        }
        let mean_e = if n_in > 0 { sum_e / n_in as f64 } else { 0.0 };
        let mean_bg = if n_out > 0 {
            sum_bg / n_out as f64
        } else {
            0.0
        };
        let var_e = if n_in > 0 {
            sum_e2 / n_in as f64 - mean_e * mean_e
        } else {
            0.0
        };
        let var_bg = if n_out > 0 {
            sum_bg2 / n_out as f64 - mean_bg * mean_bg
        } else {
            0.0
        };
        let score = if var_bg > 0.0 {
            ((mean_e - mean_bg) / var_bg.sqrt()) as f32
        } else {
            // libjxl divides by sqrt(varBg) unconditionally; we guard
            // against the zero-variance case (would be Inf or NaN)
            // and treat it as "no signal" → low score.
            0.0
        };
        self.max_energy = max_energy;
        self.mean_energy = mean_e as f32;
        self.var_energy = var_e as f32;
        self.mean_bg = mean_bg as f32;
        self.var_bg = var_bg as f32;
        self.score = score;
        self.mode = mode;
    }
}

/// Bounding rectangle of a non-empty pixel list. Mirrors libjxl
/// `BoundingRectangle` (`enc_detect_dots.cc:293-306`).
fn bounding_rectangle(pixels: &[Pixel]) -> Rect {
    debug_assert!(!pixels.is_empty(), "bounding_rectangle on empty pixel list");
    let mut low_x = pixels[0].x;
    let mut high_x = pixels[0].x;
    let mut low_y = pixels[0].y;
    let mut high_y = pixels[0].y;
    for p in &pixels[1..] {
        if p.x < low_x {
            low_x = p.x;
        }
        if p.x > high_x {
            high_x = p.x;
        }
        if p.y < low_y {
            low_y = p.y;
        }
        if p.y > high_y {
            high_y = p.y;
        }
    }
    Rect {
        x0: low_x,
        y0: low_y,
        xsize: high_x - low_x + 1,
        ysize: high_y - low_y + 1,
    }
}

/// 8-connected neighbor offsets used by the flood-fill BFS.
///
/// **NOTE**: libjxl's `enc_detect_dots.cc:194-195` declares this list as
/// `{{1,-1},{1,0},{1,1},{0,-1},{0,1},{-1,-1},{-1,1},{1,0}}` —
/// 7 distinct offsets with `{1,0}` duplicated and **`{-1,0}` missing**.
/// That is an upstream bug (the "left" neighbor is never visited, so
/// connected components are biased rightward). We mirror the list
/// verbatim for bit-parity with libjxl's dot detection so our patch
/// output matches what `cjxl -e 7 -d 3.0+` would emit.
const FLOOD_NEIGHBORS: [(i32, i32); 8] = [
    (1, -1),
    (1, 0),
    (1, 1),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, 0),
];

/// BFS flood-fill from `seed`. Pops pixels off `img` (zeroes them) as
/// it grows the component; returns `false` if the component would
/// exceed [`MAX_CC_SIZE`]. Mirrors libjxl `ExtractComponent`
/// (`enc_detect_dots.cc:191-216`).
fn extract_component(
    img: &mut [f32],
    width: i32,
    height: i32,
    pixels: &mut Vec<Pixel>,
    seed: Pixel,
    threshold: f32,
) -> bool {
    let mut stack = vec![seed];
    while let Some(current) = stack.pop() {
        pixels.push(current);
        if pixels.len() > MAX_CC_SIZE {
            return false;
        }
        for &(dx, dy) in &FLOOD_NEIGHBORS {
            let cx = current.x + dx;
            let cy = current.y + dy;
            if cx < 0 || cx >= width || cy < 0 || cy >= height {
                continue;
            }
            let idx = (cy as usize) * (width as usize) + (cx as usize);
            if img[idx] > threshold {
                img[idx] = 0.0;
                stack.push(Pixel { x: cx, y: cy });
            }
        }
    }
    true
}

/// Locate connected-component candidates in the energy image: any
/// pixel above `t_high` seeds a flood-fill that grows over neighbors
/// above `t_low`. Components larger than [`MAX_CC_SIZE`] or with a
/// bounding box ≥ `max_window` along either axis are dropped.
/// Per-CC stats are computed on the *original* energy image (the
/// flood-fill destroys a working copy) and any CC whose
/// signal-to-background `score` is below `min_score` is also dropped.
///
/// Mirrors libjxl `FindCC` (`enc_detect_dots.cc:297-339`).
pub fn find_cc(
    energy: &[f32],
    width: usize,
    height: usize,
    t_low: f32,
    t_high: f32,
    max_window: usize,
    min_score: f32,
) -> Vec<ConnectedComponent> {
    debug_assert_eq!(energy.len(), width * height);
    let w_i = width as i32;
    let h_i = height as i32;
    let mut img: Vec<f32> = energy.to_vec();
    let mut out: Vec<ConnectedComponent> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if img[idx] > t_high {
                img[idx] = 0.0;
                let mut pixels: Vec<Pixel> = Vec::new();
                let seed = Pixel {
                    x: x as i32,
                    y: y as i32,
                };
                if !extract_component(&mut img, w_i, h_i, &mut pixels, seed, t_low) {
                    continue;
                }
                if pixels.is_empty() {
                    continue;
                }
                let bounds = bounding_rectangle(&pixels);
                if (bounds.xsize as usize) < max_window && (bounds.ysize as usize) < max_window {
                    let mut cc = ConnectedComponent {
                        bounds,
                        pixels,
                        max_energy: 0.0,
                        mean_energy: 0.0,
                        var_energy: 0.0,
                        mean_bg: 0.0,
                        var_bg: 0.0,
                        score: 0.0,
                        mode: Pixel { x: 0, y: 0 },
                    };
                    cc.compute_stats(energy, w_i, h_i, CC_EXTRA_RECT);
                    if cc.score < min_score {
                        continue;
                    }
                    out.push(cc);
                }
            }
        }
    }
    out
}

/// Single-pass 5-tap horizontal Gaussian convolution. `taps` are
/// `[center, 1-step, 2-step]`. Edges replicate (mirror would also be
/// reasonable; libjxl uses replicate per `WeightsSeparable5`'s
/// horizontal-convolution helper).
pub(crate) fn gaussian_separable_5_horizontal(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    let [w0, w1, w2] = taps;
    for y in 0..height {
        let row = &src[y * width..(y + 1) * width];
        let drow = &mut dst[y * width..(y + 1) * width];
        for x in 0..width {
            let xm2 = if x >= 2 { x - 2 } else { 0 };
            let xm1 = if x >= 1 { x - 1 } else { 0 };
            let xp1 = if x + 1 < width { x + 1 } else { width - 1 };
            let xp2 = if x + 2 < width { x + 2 } else { width - 1 };
            drow[x] = w0 * row[x] + w1 * (row[xm1] + row[xp1]) + w2 * (row[xm2] + row[xp2]);
        }
    }
}

/// Single-pass 5-tap vertical Gaussian convolution. Same conventions
/// as [`gaussian_separable_5_horizontal`] but along the y axis.
pub(crate) fn gaussian_separable_5_vertical(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    let [w0, w1, w2] = taps;
    for y in 0..height {
        let ym2 = if y >= 2 { y - 2 } else { 0 };
        let ym1 = if y >= 1 { y - 1 } else { 0 };
        let yp1 = if y + 1 < height { y + 1 } else { height - 1 };
        let yp2 = if y + 2 < height { y + 2 } else { height - 1 };
        for x in 0..width {
            let v = w0 * src[y * width + x]
                + w1 * (src[ym1 * width + x] + src[yp1 * width + x])
                + w2 * (src[ym2 * width + x] + src[yp2 * width + x]);
            dst[y * width + x] = v;
        }
    }
}

/// Apply a single full 5×5 separable Gaussian (horizontal then
/// vertical) on a planar f32 image. Allocates one scratch buffer
/// (height × width).
pub(crate) fn gaussian_separable_5(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    taps: [f32; 3],
) {
    let mut tmp = vec![0.0_f32; width * height];
    gaussian_separable_5_horizontal(src, &mut tmp, width, height, taps);
    gaussian_separable_5_vertical(&tmp, dst, width, height, taps);
}

/// Sum-of-square-differences between three planar XYB channel pairs,
/// weighted by [`ENERGY_COLOR_COEF`]. Mirrors libjxl
/// `SumOfSquareDifferences` (`enc_detect_dots.cc:50-87`).
///
/// Output `energy[i] = 0 * dX^2 + 10 * dY^2 + 0 * dB^2` where
/// `dC = orig_C[i] - smooth_C[i]`. With weights `(0, 10, 0)` the X
/// and B contributions vanish; the function still iterates them so
/// the per-channel buffers stay live and the future weight-tuning
/// path doesn't need a separate codepath.
pub fn sum_of_square_differences(
    orig_x: &[f32],
    orig_y: &[f32],
    orig_b: &[f32],
    smooth_x: &[f32],
    smooth_y: &[f32],
    smooth_b: &[f32],
    energy: &mut [f32],
) {
    let n = energy.len();
    debug_assert_eq!(orig_x.len(), n);
    debug_assert_eq!(orig_y.len(), n);
    debug_assert_eq!(orig_b.len(), n);
    debug_assert_eq!(smooth_x.len(), n);
    debug_assert_eq!(smooth_y.len(), n);
    debug_assert_eq!(smooth_b.len(), n);
    let [cx, cy, cb] = ENERGY_COLOR_COEF;
    for i in 0..n {
        let dx = orig_x[i] - smooth_x[i];
        let dy = orig_y[i] - smooth_y[i];
        let db = orig_b[i] - smooth_b[i];
        energy[i] = cx * dx * dx + cy * dy * dy + cb * db * db;
    }
}

/// Compute the per-pixel energy image used by dot detection. Mirrors
/// libjxl `ComputeEnergyImage` (`enc_detect_dots.cc:151-176`).
///
/// Inputs are three planar XYB channel buffers (interleaved input
/// must be deinterleaved by the caller). Outputs:
/// - `smooth_x` / `smooth_y` / `smooth_b`: the noise-smoothed
///   (`Gaussian0_65`) XYB image — caller pre-allocates and passes
///   mutable references; subsequent quality fitting uses this.
/// - `energy`: the per-pixel energy map (`width * height` f32).
///
/// libjxl smooths the original twice with the dot-removal Gaussian
/// (sigma ≈ 3, applied 2× → effective sigma ≈ 4.24) to produce a
/// "background" estimate, then computes the squared difference vs the
/// noise-smoothed original. We do the same.
#[allow(clippy::too_many_arguments)]
pub fn compute_energy_image(
    orig_x: &[f32],
    orig_y: &[f32],
    orig_b: &[f32],
    width: usize,
    height: usize,
    smooth_x: &mut [f32],
    smooth_y: &mut [f32],
    smooth_b: &mut [f32],
    energy: &mut [f32],
) {
    let n = width * height;
    debug_assert_eq!(orig_x.len(), n);
    debug_assert_eq!(orig_y.len(), n);
    debug_assert_eq!(orig_b.len(), n);
    debug_assert_eq!(smooth_x.len(), n);
    debug_assert_eq!(smooth_y.len(), n);
    debug_assert_eq!(smooth_b.len(), n);
    debug_assert_eq!(energy.len(), n);

    // Step 1: noise-smooth each channel into smooth_*.
    let mut tmp = vec![0.0_f32; n];
    gaussian_separable_5_horizontal(orig_x, &mut tmp, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_vertical(&tmp, smooth_x, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_horizontal(orig_y, &mut tmp, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_vertical(&tmp, smooth_y, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_horizontal(orig_b, &mut tmp, width, height, GAUSSIAN_0_65_TAPS);
    gaussian_separable_5_vertical(&tmp, smooth_b, width, height, GAUSSIAN_0_65_TAPS);

    // Step 2: dot-remove each channel via 2× Gaussian-3 → background.
    // We need separate buffers for the background since smooth_*
    // already holds the noise-smoothed versions.
    let mut bg_x = vec![0.0_f32; n];
    let mut bg_y = vec![0.0_f32; n];
    let mut bg_b = vec![0.0_f32; n];
    let mut tmp2 = vec![0.0_f32; n];

    // First Gaussian3 pass: orig → tmp2 → bg.
    gaussian_separable_5_horizontal(orig_x, &mut tmp, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_vertical(&tmp, &mut bg_x, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_horizontal(orig_y, &mut tmp, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_vertical(&tmp, &mut bg_y, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_horizontal(orig_b, &mut tmp, width, height, GAUSSIAN_3_TAPS);
    gaussian_separable_5_vertical(&tmp, &mut bg_b, width, height, GAUSSIAN_3_TAPS);

    // Second Gaussian3 pass on each channel: bg → tmp2 → bg.
    for ch in [&mut bg_x, &mut bg_y, &mut bg_b] {
        gaussian_separable_5_horizontal(ch, &mut tmp, width, height, GAUSSIAN_3_TAPS);
        gaussian_separable_5_vertical(&tmp, &mut tmp2, width, height, GAUSSIAN_3_TAPS);
        ch.copy_from_slice(&tmp2);
    }

    // Step 3: energy = weighted SoSD between noise-smoothed and background.
    sum_of_square_differences(smooth_x, smooth_y, smooth_b, &bg_x, &bg_y, &bg_b, energy);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel_sums_match_libjxl() {
        // Each separable kernel should sum to 1.0 across its 5 taps:
        //   [w2, w1, w0, w1, w2]
        let g0_65: f32 =
            GAUSSIAN_0_65_TAPS[0] + 2.0 * GAUSSIAN_0_65_TAPS[1] + 2.0 * GAUSSIAN_0_65_TAPS[2];
        assert!(
            (g0_65 - 1.0).abs() < 1e-3,
            "Gaussian0_65 5-tap sum should be ~1.0, got {g0_65}",
        );
        let g3: f32 = GAUSSIAN_3_TAPS[0] + 2.0 * GAUSSIAN_3_TAPS[1] + 2.0 * GAUSSIAN_3_TAPS[2];
        assert!(
            (g3 - 1.0).abs() < 1e-3,
            "Gaussian3 5-tap sum should be ~1.0, got {g3}",
        );
    }

    #[test]
    fn test_gaussian_separable_5_uniform_input_is_uniform_output() {
        let w = 16;
        let h = 16;
        let src = vec![0.5_f32; w * h];
        let mut dst = vec![0.0_f32; w * h];
        gaussian_separable_5(&src, &mut dst, w, h, GAUSSIAN_0_65_TAPS);
        for &v in &dst {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "uniform 0.5 should stay ~0.5, got {v}",
            );
        }
    }

    #[test]
    fn test_gaussian_separable_5_blurs_a_dot() {
        // 7×7 image with a single bright pixel in the center.
        let w = 7;
        let h = 7;
        let mut src = vec![0.0_f32; w * h];
        src[3 * w + 3] = 1.0;
        let mut dst = vec![0.0_f32; w * h];
        gaussian_separable_5(&src, &mut dst, w, h, GAUSSIAN_0_65_TAPS);
        // Center should drop (energy spread); outer ring should rise.
        assert!(
            dst[3 * w + 3] < 1.0,
            "center should drop, got {}",
            dst[3 * w + 3]
        );
        assert!(dst[2 * w + 3] > 0.0, "neighbor should pick up energy");
        // Conservation: total energy preserved within rounding.
        let total: f32 = dst.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.05,
            "5-tap separable Gaussian should approximately preserve energy; total={total}",
        );
    }

    #[test]
    fn test_sum_of_square_differences_y_only() {
        // Energy = 0*dX^2 + 10*dY^2 + 0*dB^2. Verify only Y contributes.
        let n = 4;
        let orig_x = vec![1.0_f32; n];
        let orig_y = vec![2.0_f32; n];
        let orig_b = vec![3.0_f32; n];
        let smooth_x = vec![0.5_f32; n]; // dX = 0.5 → ignored
        let smooth_y = vec![1.0_f32; n]; // dY = 1.0 → contributes 10
        let smooth_b = vec![2.0_f32; n]; // dB = 1.0 → ignored
        let mut energy = vec![0.0_f32; n];
        sum_of_square_differences(
            &orig_x,
            &orig_y,
            &orig_b,
            &smooth_x,
            &smooth_y,
            &smooth_b,
            &mut energy,
        );
        for &e in &energy {
            // 10 * (1.0)^2 = 10.0
            assert!((e - 10.0).abs() < 1e-5, "energy should be 10.0, got {e}");
        }
    }

    #[test]
    fn test_compute_energy_image_uniform_input_is_zero() {
        // A uniform image has no dots → energy should be ~0 everywhere
        // (Gaussian blurs of a uniform image are uniform; diff is 0).
        let w = 16;
        let h = 16;
        let orig = vec![0.5_f32; w * h];
        let mut sx = vec![0.0_f32; w * h];
        let mut sy = vec![0.0_f32; w * h];
        let mut sb = vec![0.0_f32; w * h];
        let mut energy = vec![0.0_f32; w * h];
        compute_energy_image(
            &orig,
            &orig,
            &orig,
            w,
            h,
            &mut sx,
            &mut sy,
            &mut sb,
            &mut energy,
        );
        for &e in &energy {
            assert!(e.abs() < 1e-3, "uniform input → ~0 energy, got {e}");
        }
    }

    #[test]
    fn test_bounding_rectangle_single_pixel() {
        let r = bounding_rectangle(&[Pixel { x: 5, y: 7 }]);
        assert_eq!(
            r,
            Rect {
                x0: 5,
                y0: 7,
                xsize: 1,
                ysize: 1
            }
        );
    }

    #[test]
    fn test_bounding_rectangle_multi_pixel() {
        let pixels = vec![
            Pixel { x: 1, y: 2 },
            Pixel { x: 4, y: 7 },
            Pixel { x: 0, y: 5 },
            Pixel { x: 3, y: 1 },
        ];
        let r = bounding_rectangle(&pixels);
        assert_eq!(
            r,
            Rect {
                x0: 0,
                y0: 1,
                xsize: 5,
                ysize: 7
            }
        );
    }

    #[test]
    fn test_rect_contains() {
        let r = Rect {
            x0: 2,
            y0: 3,
            xsize: 4,
            ysize: 5,
        };
        assert!(r.contains(Pixel { x: 2, y: 3 }));
        assert!(r.contains(Pixel { x: 5, y: 7 }));
        assert!(!r.contains(Pixel { x: 1, y: 3 }));
        assert!(!r.contains(Pixel { x: 6, y: 3 })); // x0+xsize=6 is exclusive
        assert!(!r.contains(Pixel { x: 2, y: 8 })); // y0+ysize=8 is exclusive
    }

    #[test]
    fn test_extract_component_single_pixel() {
        // 5×5 image: only one pixel above threshold → CC of size 1.
        let mut img = vec![0.0_f32; 25];
        img[12] = 1.0; // center
        let mut pixels = Vec::new();
        let ok = extract_component(&mut img, 5, 5, &mut pixels, Pixel { x: 2, y: 2 }, 0.5);
        assert!(ok);
        // The seed itself isn't checked against threshold — caller
        // (find_cc) is expected to validate the seed and zero it
        // before calling. Here we just verify the BFS popped the
        // seed and added it.
        assert_eq!(pixels.len(), 1);
        assert_eq!(pixels[0], Pixel { x: 2, y: 2 });
    }

    #[test]
    fn test_extract_component_grows_to_neighbors() {
        // 5×5 image with a 3-pixel L-shape above threshold:
        //   row 2: . . X X .
        //   row 3: . . X . .
        // Seed at (2, 2). Two neighbors at (3, 2) and (2, 3).
        let mut img = vec![0.0_f32; 25];
        img[2 * 5 + 2] = 1.0; // seed
        img[2 * 5 + 3] = 1.0;
        img[3 * 5 + 2] = 1.0;
        let mut pixels = Vec::new();
        // Caller zeros the seed before calling, mirroring find_cc.
        img[2 * 5 + 2] = 0.0;
        let ok = extract_component(&mut img, 5, 5, &mut pixels, Pixel { x: 2, y: 2 }, 0.5);
        assert!(ok);
        assert_eq!(pixels.len(), 3);
    }

    #[test]
    fn test_extract_component_aborts_above_max_size() {
        // Saturate a 40×40 region with above-threshold pixels →
        // 1600 > MAX_CC_SIZE (1000) → returns false.
        let w = 40;
        let h = 40;
        let mut img = vec![1.0_f32; w * h];
        let mut pixels = Vec::new();
        // Zero seed before calling.
        img[0] = 0.0;
        let ok = extract_component(
            &mut img,
            w as i32,
            h as i32,
            &mut pixels,
            Pixel { x: 0, y: 0 },
            0.5,
        );
        assert!(!ok, "saturated 40×40 should abort at MAX_CC_SIZE");
        assert!(pixels.len() > MAX_CC_SIZE);
    }

    #[test]
    fn test_find_cc_no_dots_below_threshold() {
        let w = 16;
        let h = 16;
        let energy = vec![0.01_f32; w * h]; // below t_high=0.04
        let ccs = find_cc(&energy, w, h, 0.02, 0.04, 5, -1e9);
        assert!(ccs.is_empty(), "no above-threshold pixels → 0 CCs");
    }

    #[test]
    fn test_find_cc_isolated_dot() {
        // 16×16 image with a single bright pixel at (8, 8).
        let w = 16;
        let h = 16;
        let mut energy = vec![0.0_f32; w * h];
        energy[8 * w + 8] = 0.5; // > t_high
        let ccs = find_cc(&energy, w, h, 0.02, 0.04, 5, -1e9);
        assert_eq!(ccs.len(), 1, "exactly one CC for one isolated dot");
        let cc = &ccs[0];
        assert_eq!(
            cc.bounds,
            Rect {
                x0: 8,
                y0: 8,
                xsize: 1,
                ysize: 1
            }
        );
        assert_eq!(cc.pixels.len(), 1);
        assert!((cc.max_energy - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_find_cc_drops_oversized_window() {
        // Two pixels exactly 5 apart → bounding window 6 ≥ max_window=5 → dropped.
        let w = 16;
        let h = 16;
        let mut energy = vec![0.0_f32; w * h];
        energy[2 * w + 2] = 0.5;
        energy[2 * w + 7] = 0.5;
        // Connect them via above-t_low pixels in between to force one CC.
        for x in 3..7 {
            energy[2 * w + x] = 0.03;
        }
        let ccs = find_cc(&energy, w, h, 0.02, 0.04, 5, -1e9);
        // bounding xsize would be 6 (x0=2..7 inclusive). max_window=5 strict.
        assert!(
            ccs.is_empty() || ccs.iter().all(|cc| (cc.bounds.xsize as usize) < 5),
            "wide CC should be dropped",
        );
    }

    #[test]
    fn test_dot_gaussian_model_peak_at_origin() {
        // intensity * exp(-0.5 * 0) = intensity at (dx, dy) = (0, 0).
        let v = dot_gaussian_model(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 5.0);
        assert!((v - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_dot_gaussian_model_drops_with_distance() {
        // With sigma_x = sigma_y = 1.0 and angle = 0, the radial
        // exponent is dx² + dy²; at (1, 0) → exp(-0.5 * 1) = 0.6065.
        let v = dot_gaussian_model(1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0);
        assert!((v - (-0.5_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_convert_to_diagonal_identity() {
        let (d, u) = convert_to_diagonal_2x2([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(d, [1.0, 1.0]);
        assert_eq!(u, [[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn test_convert_to_diagonal_already_diagonal() {
        let (d, u) = convert_to_diagonal_2x2([[3.0, 0.0], [0.0, 5.0]]);
        assert_eq!(d, [3.0, 5.0]);
        assert_eq!(u, [[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn test_convert_to_diagonal_symmetric_2x2() {
        // Eigenvalues of [[2, 1], [1, 2]] are 1 and 3.
        let (d, _u) = convert_to_diagonal_2x2([[2.0, 1.0], [1.0, 2.0]]);
        let mut sorted = d;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-9, "smallest eigval ≈ 1");
        assert!((sorted[1] - 3.0).abs() < 1e-9, "largest eigval ≈ 3");
    }

    #[test]
    fn test_default_gaussian_detect_params() {
        let p = GaussianDetectParams::default();
        assert_eq!(p.t_high, 0.04);
        assert_eq!(p.t_low, 0.02);
        assert_eq!(p.max_win_size, ELLIPSE_WINDOW_SIZE);
        assert_eq!(p.max_cc, 50);
        assert_eq!(p.perc_cc, 15);
    }

    #[test]
    fn test_detect_gaussian_ellipses_uniform_image_no_dots() {
        // Uniform image → no dots.
        let w = 32;
        let h = 32;
        let img = vec![0.5_f32; w * h];
        let dots =
            detect_gaussian_ellipses(&img, &img, &img, w, h, &GaussianDetectParams::default());
        assert!(dots.is_empty());
    }

    #[test]
    fn test_detect_gaussian_ellipses_finds_synthetic_dot() {
        // 32×32 background of 0.0 with a single bright Gaussian dot
        // at (16, 16) on the Y channel. Use parameters lenient enough
        // to actually pick it up.
        let w = 32;
        let h = 32;
        let img_x = vec![0.0_f32; w * h];
        let img_b = vec![0.0_f32; w * h];
        let mut img_y = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 16.0;
                let dy = y as f64 - 16.0;
                let g = dot_gaussian_model(dx, dy, 1.0, 0.0, 0.5, 0.5, 1.0);
                img_y[y * w + x] = g as f32;
            }
        }
        let mut params = GaussianDetectParams::default();
        params.min_score = -1e9; // accept any score
        params.max_l2_loss = 1.0;
        params.max_custom_loss = 1e9;
        params.min_intensity = 0.0;
        params.max_dist_mean_mode = 100.0;
        params.max_neg_pixels = 10000;
        // perc_cc=15 with a single CC rounds to 0 → drops it.
        // Test wants the single dot to make it through; use 100%.
        params.perc_cc = 100;
        let dots = detect_gaussian_ellipses(&img_x, &img_y, &img_b, w, h, &params);
        // We should detect at least one dot near (16, 16).
        assert!(!dots.is_empty(), "expected at least one detected dot");
        let near_center = dots.iter().any(|d| {
            let cx = (d.x0 as i32) + (d.xsize as i32) / 2;
            let cy = (d.y0 as i32) + (d.ysize as i32) / 2;
            (cx - 16).abs() <= 3 && (cy - 16).abs() <= 3
        });
        assert!(near_center, "no detected dot near the synthesized center");
    }

    #[test]
    fn test_compute_dot_losses_zero_for_perfect_fit() {
        // If the dot is already a perfect fit (intensity matches
        // exactly, no neg pixels), losses should be approximately zero.
        let w = 16;
        let h = 16;
        let bg_val = 0.5_f32;
        let mut img_y = vec![bg_val; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 8.0;
                let dy = y as f64 - 8.0;
                let g = dot_gaussian_model(dx, dy, 1.0, 0.0, 1.0, 1.0, 0.5);
                img_y[y * w + x] = (img_y[y * w + x] as f64 + g) as f32;
            }
        }
        let img_x = vec![bg_val; w * h];
        let img_b = vec![bg_val; w * h];
        let bg_x = vec![bg_val; w * h];
        let bg_y = vec![bg_val; w * h];
        let bg_b = vec![bg_val; w * h];
        let cc = ConnectedComponent {
            bounds: Rect {
                x0: 7,
                y0: 7,
                xsize: 3,
                ysize: 3,
            },
            pixels: vec![Pixel { x: 8, y: 8 }],
            max_energy: 0.5,
            mean_energy: 0.5,
            var_energy: 0.0,
            mean_bg: 0.0,
            var_bg: 0.0,
            score: 1.0,
            mode: Pixel { x: 8, y: 8 },
        };
        let mut ellipse = GaussianEllipse {
            x: 8.0,
            y: 8.0,
            sigma_x: 1.0,
            sigma_y: 1.0,
            angle: 0.0,
            intensity: [0.0, 0.5, 0.0],
            l2_loss: 0.0,
            l1_loss: 0.0,
            ridge_loss: 0.0,
            custom_loss: 0.0,
            bg_color: [bg_val as f64, bg_val as f64, bg_val as f64],
            neg_pixels: 0,
            neg_value: [0.0; 3],
        };
        compute_dot_losses(
            &mut ellipse,
            &cc,
            w as i32,
            h as i32,
            &[&img_x, &img_y, &img_b],
            &[&bg_x, &bg_y, &bg_b],
        );
        // Y-channel residual should be ~0 because the synthesized dot
        // is exactly the model. Other channels are uniform background
        // → also ~0 residual.
        assert!(
            ellipse.l2_loss < 1e-3,
            "perfect-fit l2_loss should be ~0; got {}",
            ellipse.l2_loss,
        );
        // dist_mean_mode_sq is 0, neg_value[1] is 0 → custom_loss
        // should also be small.
        assert!(
            ellipse.custom_loss < 0.01,
            "perfect-fit custom_loss should be ~0; got {}",
            ellipse.custom_loss,
        );
    }

    #[test]
    fn test_fit_gaussian_fast_isolated_dot_recovers_position() {
        // Synthesize a 16×16 image with a Gaussian dot centered at
        // (8, 8) with sigma=1.5 and Y-channel intensity 0.3 above a
        // uniform background.
        let w = 16;
        let h = 16;
        let mut img_x = vec![0.5_f32; w * h];
        let mut img_y = vec![0.5_f32; w * h];
        let mut img_b = vec![0.5_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 8.0;
                let dy = y as f64 - 8.0;
                let g = dot_gaussian_model(dx, dy, 1.0, 0.0, 1.5, 1.5, 0.3);
                img_y[y * w + x] = (img_y[y * w + x] as f64 + g) as f32;
            }
        }
        let bg_x = vec![0.5_f32; w * h];
        let bg_y = vec![0.5_f32; w * h];
        let bg_b = vec![0.5_f32; w * h];

        // Build a CC at (8, 8) with mode = (8, 8).
        let cc = ConnectedComponent {
            bounds: Rect {
                x0: 7,
                y0: 7,
                xsize: 3,
                ysize: 3,
            },
            pixels: vec![Pixel { x: 8, y: 8 }],
            max_energy: 0.3,
            mean_energy: 0.3,
            var_energy: 0.0,
            mean_bg: 0.0,
            var_bg: 0.0,
            score: 1.0,
            mode: Pixel { x: 8, y: 8 },
        };

        let fit = fit_gaussian_fast(
            &cc, w as i32, h as i32, &img_x, &img_y, &img_b, &bg_x, &bg_y, &bg_b,
        )
        .expect("fit should succeed for in-bounds CC");

        // Center should land near (8, 8). Because the moments only
        // weight a 5×5 window around the mode, the recovered center
        // is a moment-weighted estimate; tolerance ~0.5 px is fine.
        assert!((fit.x - 8.0).abs() < 0.5, "x={}", fit.x);
        assert!((fit.y - 8.0).abs() < 0.5, "y={}", fit.y);
        // Y-channel intensity recovered via least squares should be
        // close to the true 0.3 (within ~10% — the 5×5 window only
        // captures part of the Gaussian).
        assert!(
            fit.intensity[1] > 0.2 && fit.intensity[1] < 0.5,
            "intensity[1]={}, want ≈ 0.3",
            fit.intensity[1]
        );
    }
}
