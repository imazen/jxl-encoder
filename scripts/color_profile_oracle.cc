// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Generates valid profiles from the unmodified libjxl v0.12 CMS.
#include <iomanip>
#include <iostream>
#include "lib/jxl/color_encoding_internal.h"

jxl::Status Run() {
  for (auto space : {jxl::ColorSpace::kRGB, jxl::ColorSpace::kGray}) {
    jxl::ColorEncoding color;
    JXL_RETURN_IF_ERROR(color.SetSRGB(space));
    std::cout << static_cast<int>(space) << '\t';
    for (uint8_t byte : color.ICC())
      std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    std::cout << std::dec << '\n';
  }
  return true;
}
int main() { return Run() ? 0 : 1; }
