// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Production animation encoding through the unmodified libjxl v0.12 public API.
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
#include "jxl/encode.h"
#include "jxl/color_encoding.h"

int Run(const char* label, bool tiny, bool keep) {
  JxlEncoder* encoder = JxlEncoderCreate(nullptr);
  if (!encoder) return 1;
  auto check = [](JxlEncoderStatus s) { return s == JXL_ENC_SUCCESS; };
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = 3;
  info.ysize = 1;
  info.bits_per_sample = 32;
  info.exponent_bits_per_sample = 8;
  info.num_color_channels = 3;
  info.num_extra_channels = 1;
  info.alpha_bits = 32;
  info.alpha_exponent_bits = 8;
  info.have_animation = JXL_TRUE;
  if (!check(JxlEncoderSetBasicInfo(encoder, &info))) return 2;
  JxlColorEncoding color;
  JxlColorEncodingSetToLinearSRGB(&color, JXL_FALSE);
  if (!check(JxlEncoderSetColorEncoding(encoder, &color))) return 3;
  auto* settings = JxlEncoderFrameSettingsCreate(encoder, nullptr);
  if (!settings || !check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_EFFORT, 3))) return 4;
  if (!check(JxlEncoderFrameSettingsSetOption(settings, JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE, keep))) return 7;
  const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  const std::array<float, 12> pixels = tiny ? std::array<float,12>{
    0.f,0.f,1.f,1.f, 1.f,0.f,0.f,1.f/1024.f, 0.f,0.f,1.f,1.f} : std::array<float,12>{
    .25f,.25f,.25f,.0625f, .25f,.25f,.25f,.9375f, .25f,.25f,.25f,.1234567f};
  for (unsigned duration : {10u, 20u}) {
    JxlFrameHeader frame;
    JxlEncoderInitFrameHeader(&frame);
    frame.duration = duration;
    if (!check(JxlEncoderSetFrameHeader(settings, &frame)) ||
        !check(JxlEncoderAddImageFrame(settings, &format, pixels.data(), sizeof(pixels)))) return 5;
  }
  JxlEncoderCloseInput(encoder);
  std::vector<uint8_t> bytes(4096);
  size_t used = 0;
  for (;;) {
    uint8_t* next = bytes.data() + used;
    size_t avail = bytes.size() - used;
    auto result = JxlEncoderProcessOutput(encoder, &next, &avail);
    used = bytes.size() - avail;
    if (result == JXL_ENC_SUCCESS) break;
    if (result != JXL_ENC_NEED_MORE_OUTPUT) return 6;
    bytes.resize(bytes.size() * 2);
  }
  JxlEncoderDestroy(encoder);
  std::cout << label << '\t';
  for (size_t i = 0; i < used; ++i)
    std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(bytes[i]);
  std::cout << '\n';
  return 0;
}

int main() {
  if (int result = Run("float32", false, false)) return result;
  if (int result = Run("visible-on", true, false)) return result;
  return Run("visible-off", true, true);
}
