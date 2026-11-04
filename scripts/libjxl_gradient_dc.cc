// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Reference object retains its BSD license.
// Calls the unmodified libjxl v0.12 enc_modular.cc object, not a transcription.
#include <jxl/version.h>
#include <cstdint>
#include <iostream>
#include <vector>

static_assert(JPEGXL_MAJOR_VERSION == 0 && JPEGXL_MINOR_VERSION == 12 &&
              JPEGXL_PATCH_VERSION == 0);

namespace jxl {
int QuantizeGradient(const int32_t*, size_t, size_t, size_t, size_t, size_t,
                     float, float);
}

int main() {
  size_t width, height;
  if (!(std::cin >> width >> height) || width == 0 || height == 0 ||
      width > 256 || height > 256) return 1;
  const size_t stride = width + 3;
  std::vector<int32_t> quantized(stride * height, 123456);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      float value;
      if (!(std::cin >> value)) return 2;
      int32_t* row = quantized.data() + y * stride;
      row[x] = jxl::QuantizeGradient(row, stride, 1, x, y, width, value, 1.0f);
      std::cout << row[x] << ((x + 1 == width && y + 1 == height) ? '\n' : ',');
    }
  }
}
