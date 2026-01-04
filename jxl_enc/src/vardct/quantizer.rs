//! Quantizer for VarDCT encoding.
//!
//! The quantizer controls the trade-off between quality and file size.
//! It converts floating-point DCT coefficients to integers.

use crate::bit_writer::BitWriter;
#[allow(unused_imports)]
use crate::{trace_section, trace_write};

/// Denominator for global scale (1 << 16 = 65536).
pub const GLOBAL_SCALE_DENOM: u32 = 1 << 16;

/// Numerator for global scale scaling.
pub const GLOBAL_SCALE_NUMERATOR: u32 = 4096;

/// Maximum quantization value for per-block quant field.
pub const QUANT_MAX: i32 = 256;

/// Default quant value.
pub const DEFAULT_QUANT: i32 = 64;

/// DC quantization power constant.
const DC_QUANT_POW: f32 = 0.83;

/// DC quantization base constant.
#[allow(clippy::excessive_precision)]
const DC_QUANT: f32 = 1.095924047623553;

/// AC quantization base constant.
const AC_QUANT: f32 = 0.765;

/// Zero-bias defaults for quantizing channels X, Y, B.
pub const ZERO_BIAS_DEFAULT: [f32; 3] = [0.5, 0.5, 0.5];

/// Quantization bias for coefficient adjustment.
#[allow(clippy::excessive_precision)]
pub const QUANT_BIAS: [f32; 4] = [
    1.0 - 0.05465007330715401,
    1.0 - 0.07005449891748593,
    1.0 - 0.049935103337343655,
    0.145,
];

/// Quantizer parameters for serialization.
#[derive(Clone, Debug)]
pub struct QuantizerParams {
    /// Global scale (1-73728).
    pub global_scale: u32,
    /// DC quantization value (1-65552).
    pub quant_dc: u32,
}

impl Default for QuantizerParams {
    fn default() -> Self {
        Self {
            global_scale: GLOBAL_SCALE_DENOM / DEFAULT_QUANT as u32,
            quant_dc: DEFAULT_QUANT as u32,
        }
    }
}

impl QuantizerParams {
    /// Create quantizer params from a butteraugli distance target.
    pub fn from_distance(distance: f32) -> Self {
        let quant_dc = initial_quant_dc(distance);
        let quant_ac = AC_QUANT / distance;

        // Compute global scale based on target median quant
        let quant_field_target = 5.0;
        let mut scale = GLOBAL_SCALE_DENOM as f32 * quant_ac / quant_field_target;

        // Clamp scale to valid range
        scale = scale.clamp(1.0, (1 << 15) as f32);

        // Ensure quant_dc won't be too small
        let scaled_quant_dc = quant_dc * GLOBAL_SCALE_NUMERATOR as f32 * 1.6;
        if scale > scaled_quant_dc {
            scale = scaled_quant_dc.max(1.0);
        }

        let global_scale = scale as u32;
        let inv_global_scale = GLOBAL_SCALE_DENOM as f32 / global_scale as f32;

        // Compute quant_dc based on global scale
        let fval = (quant_dc * inv_global_scale + 0.5).min(65536.0);
        let quant_dc_int = fval as u32;

        Self {
            global_scale,
            quant_dc: quant_dc_int.max(1),
        }
    }

    /// Inverse of global scale (GLOBAL_SCALE_DENOM / global_scale).
    pub fn inv_global_scale(&self) -> f32 {
        GLOBAL_SCALE_DENOM as f32 / self.global_scale as f32
    }

    /// Inverse of DC quantization.
    pub fn inv_quant_dc(&self) -> f32 {
        self.inv_global_scale() / self.quant_dc as f32
    }

    /// Global scale as float (global_scale / GLOBAL_SCALE_DENOM).
    pub fn global_scale_float(&self) -> f32 {
        self.global_scale as f32 / GLOBAL_SCALE_DENOM as f32
    }

    /// Write quantizer params to bitstream.
    pub fn write(&self, writer: &mut BitWriter) {
        // global_scale: U32(BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097), BitsOffset(16, 8193))
        let gs = self.global_scale as u64;
        if (1..=2048).contains(&gs) {
            writer.write(2, 0).unwrap();
            writer.write(11, gs - 1).unwrap();
        } else if (2049..=4096).contains(&gs) {
            writer.write(2, 1).unwrap();
            writer.write(11, gs - 2049).unwrap();
        } else if (4097..=8192).contains(&gs) {
            writer.write(2, 2).unwrap();
            writer.write(12, gs - 4097).unwrap();
        } else {
            writer.write(2, 3).unwrap();
            writer.write(16, gs - 8193).unwrap();
        }

        // quant_dc: U32(Val(16), BitsOffset(5, 1), BitsOffset(8, 1), BitsOffset(16, 1))
        let qdc = self.quant_dc as u64;
        if qdc == 16 {
            writer.write(2, 0).unwrap();
        } else if (1..=32).contains(&qdc) {
            writer.write(2, 1).unwrap();
            writer.write(5, qdc - 1).unwrap();
        } else if (1..=256).contains(&qdc) {
            writer.write(2, 2).unwrap();
            writer.write(8, qdc - 1).unwrap();
        } else {
            writer.write(2, 3).unwrap();
            writer.write(16, qdc - 1).unwrap();
        }
    }

    /// Write quantizer params to bitstream with tracing.
    pub fn write_traced(&self, writer: &mut BitWriter) {
        trace_section!(begin "QUANTIZER_PARAMS", writer);

        let gs = self.global_scale as u64;
        if (1..=2048).contains(&gs) {
            trace_write!(writer, 2, 0, "global_scale.selector", "0 (1-2048)").unwrap();
            trace_write!(writer, 11, gs - 1, "global_scale.value", &format!("{}", gs)).unwrap();
        } else if (2049..=4096).contains(&gs) {
            trace_write!(writer, 2, 1, "global_scale.selector", "1 (2049-4096)").unwrap();
            trace_write!(writer, 11, gs - 2049, "global_scale.value", &format!("{}", gs)).unwrap();
        } else if (4097..=8192).contains(&gs) {
            trace_write!(writer, 2, 2, "global_scale.selector", "2 (4097-8192)").unwrap();
            trace_write!(writer, 12, gs - 4097, "global_scale.value", &format!("{}", gs)).unwrap();
        } else {
            trace_write!(writer, 2, 3, "global_scale.selector", "3 (8193+)").unwrap();
            trace_write!(writer, 16, gs - 8193, "global_scale.value", &format!("{}", gs)).unwrap();
        }

        let qdc = self.quant_dc as u64;
        if qdc == 16 {
            trace_write!(writer, 2, 0, "quant_dc", "selector=0 → 16").unwrap();
        } else if (1..=32).contains(&qdc) {
            trace_write!(writer, 2, 1, "quant_dc.selector", "1 (1-32)").unwrap();
            trace_write!(writer, 5, qdc - 1, "quant_dc.value", &format!("{}", qdc)).unwrap();
        } else if (1..=256).contains(&qdc) {
            trace_write!(writer, 2, 2, "quant_dc.selector", "2 (1-256)").unwrap();
            trace_write!(writer, 8, qdc - 1, "quant_dc.value", &format!("{}", qdc)).unwrap();
        } else {
            trace_write!(writer, 2, 3, "quant_dc.selector", "3 (1+)").unwrap();
            trace_write!(writer, 16, qdc - 1, "quant_dc.value", &format!("{}", qdc)).unwrap();
        }

        trace_section!(end "QUANTIZER_PARAMS", writer);
    }
}

/// Compute initial DC quantization from butteraugli distance.
pub fn initial_quant_dc(butteraugli_target: f32) -> f32 {
    const DC_MUL: f32 = 0.3; // Butteraugli target where non-linearity kicks in

    let butteraugli_target_dc = (0.5 * butteraugli_target).max(
        butteraugli_target.min(DC_MUL * ((1.0 / DC_MUL) * butteraugli_target).powf(DC_QUANT_POW)),
    );

    // Clamp to reasonable range
    (DC_QUANT / butteraugli_target_dc).min(50.0)
}

/// Compute initial AC quantization from butteraugli distance.
pub fn initial_quant_ac(butteraugli_target: f32) -> f32 {
    AC_QUANT / butteraugli_target
}

/// Quantizer for VarDCT encoding.
#[derive(Clone, Debug)]
pub struct Quantizer {
    /// Quantizer parameters (serializable).
    pub params: QuantizerParams,
    /// Cached inverse global scale.
    inv_global_scale: f32,
    /// Cached global scale as float.
    global_scale_float: f32,
    /// Cached inverse DC quantization.
    inv_quant_dc: f32,
    /// Zero bias per channel.
    pub zero_bias: [f32; 3],
}

impl Default for Quantizer {
    fn default() -> Self {
        Self::new(QuantizerParams::default())
    }
}

impl Quantizer {
    /// Create a new quantizer with the given parameters.
    pub fn new(params: QuantizerParams) -> Self {
        let inv_global_scale = params.inv_global_scale();
        let global_scale_float = params.global_scale_float();
        let inv_quant_dc = params.inv_quant_dc();

        Self {
            params,
            inv_global_scale,
            global_scale_float,
            inv_quant_dc,
            zero_bias: ZERO_BIAS_DEFAULT,
        }
    }

    /// Create a quantizer from a butteraugli distance.
    pub fn from_distance(distance: f32) -> Self {
        Self::new(QuantizerParams::from_distance(distance))
    }

    /// Get inverse global scale.
    pub fn inv_global_scale(&self) -> f32 {
        self.inv_global_scale
    }

    /// Get global scale as float.
    pub fn global_scale_float(&self) -> f32 {
        self.global_scale_float
    }

    /// Get inverse DC quantization.
    pub fn inv_quant_dc(&self) -> f32 {
        self.inv_quant_dc
    }

    /// Get inverse AC quantization for a given per-block quant value.
    pub fn inv_quant_ac(&self, quant: i32) -> f32 {
        self.inv_global_scale / quant as f32
    }

    /// Clamp a quantization value to valid range.
    pub fn clamp_val(val: f32) -> i32 {
        val.clamp(1.0, QUANT_MAX as f32) as i32
    }

    /// Compute per-block quant value from a float quant field value.
    pub fn quant_from_field(&self, field_val: f32) -> i32 {
        Self::clamp_val(field_val * self.inv_global_scale + 0.5)
    }

    /// Write quantizer parameters to bitstream.
    pub fn write(&self, writer: &mut BitWriter) {
        self.params.write(writer);
    }

    /// Write quantizer parameters to bitstream with tracing.
    pub fn write_traced(&self, writer: &mut BitWriter) {
        self.params.write_traced(writer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_quantizer() {
        let q = Quantizer::default();
        assert_eq!(
            q.params.global_scale,
            GLOBAL_SCALE_DENOM / DEFAULT_QUANT as u32
        );
        assert_eq!(q.params.quant_dc, DEFAULT_QUANT as u32);
    }

    #[test]
    fn test_initial_quant_dc() {
        // At distance 1.0, quant_dc should be around 1.1
        let qdc = initial_quant_dc(1.0);
        assert!(qdc > 0.5 && qdc < 2.0, "qdc = {}", qdc);

        // At distance 0.5, quant_dc should be higher (better quality)
        let qdc_high = initial_quant_dc(0.5);
        assert!(qdc_high > qdc, "qdc_high = {}, qdc = {}", qdc_high, qdc);

        // At high distance, quant_dc approaches 0
        let qdc_low = initial_quant_dc(10.0);
        assert!(qdc_low < qdc, "qdc_low = {}", qdc_low);
    }

    #[test]
    fn test_initial_quant_ac() {
        // At distance 1.0, quant_ac = 0.765
        let qac = initial_quant_ac(1.0);
        assert!((qac - 0.765).abs() < 0.001);

        // At distance 2.0, quant_ac = 0.765 / 2 = 0.3825
        let qac2 = initial_quant_ac(2.0);
        assert!((qac2 - 0.3825).abs() < 0.001);
    }

    #[test]
    fn test_from_distance() {
        let q = Quantizer::from_distance(1.0);
        assert!(q.params.global_scale > 0);
        assert!(q.params.quant_dc > 0);
    }

    #[test]
    fn test_inv_global_scale() {
        let params = QuantizerParams {
            global_scale: GLOBAL_SCALE_DENOM,
            quant_dc: 64,
        };
        let q = Quantizer::new(params);
        assert!((q.inv_global_scale() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_clamp_val() {
        assert_eq!(Quantizer::clamp_val(0.5), 1);
        assert_eq!(Quantizer::clamp_val(100.0), 100);
        assert_eq!(Quantizer::clamp_val(500.0), QUANT_MAX);
    }

    #[test]
    fn test_write_params() {
        use crate::bit_writer::BitWriter;

        let params = QuantizerParams {
            global_scale: 1024,
            quant_dc: 16,
        };

        let mut writer = BitWriter::new();
        params.write(&mut writer);

        // Should have written some bits
        assert!(writer.bits_written() > 0);
    }

    #[test]
    fn test_write_params_large() {
        use crate::bit_writer::BitWriter;

        let params = QuantizerParams {
            global_scale: 10000,
            quant_dc: 500,
        };

        let mut writer = BitWriter::new();
        params.write(&mut writer);

        // Should have written more bits for larger values
        assert!(writer.bits_written() > 10);
    }
}
