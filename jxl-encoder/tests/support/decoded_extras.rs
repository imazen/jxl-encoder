//! Full rendering through the pinned current jxl-rs API.
#[path = "decoded_frames.rs"]
mod frames;
pub(crate) use frames::decode_current_frames;
pub(crate) fn decode_current_planes(data: &[u8], label: &str) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut frames = decode_current_frames(data, label);
    assert_eq!(frames.len(), 1, "single displayed frame: {label}");
    frames.pop().unwrap()
}
