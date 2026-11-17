// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Calls the unmodified libjxl v0.12 ColorEncoding field writer.
#include <iomanip>
#include <iostream>
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/memory_manager_internal.h"

jxl::Status Run() {
  JxlMemoryManager memory;
  JXL_RETURN_IF_ERROR(jxl::MemoryManagerInit(&memory, nullptr));
  jxl::ColorEncoding color;
  JXL_RETURN_IF_ERROR(color.SetSRGB(jxl::ColorSpace::kRGB));
  auto external = color.ToExternal();
  external.white_point = JXL_WHITE_POINT_CUSTOM;
  external.white_point_xy[0] = .1;
  external.white_point_xy[1] = .2;
  external.primaries = JXL_PRIMARIES_CUSTOM;
  external.primaries_red_xy[0] = .3;
  external.primaries_red_xy[1] = .4;
  external.primaries_green_xy[0] = -.5;
  external.primaries_green_xy[1] = .7;
  external.primaries_blue_xy[0] = 1.2;
  external.primaries_blue_xy[1] = -1.5;
  JXL_RETURN_IF_ERROR(color.FromExternal(external));
  jxl::BitWriter writer(&memory);
  JXL_RETURN_IF_ERROR(jxl::Bundle::Write(color, &writer, jxl::LayerType::Header, nullptr));
  std::cout << writer.BitsWritten() << '\t';
  JXL_RETURN_IF_ERROR(writer.WithMaxBits(8, jxl::LayerType::Header, nullptr,
      [&]() -> jxl::Status { writer.ZeroPadToByte(); return true; }));
  for (uint8_t byte : writer.GetSpan())
    std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
  std::cout << '\n';
  return true;
}
int main() { return Run() ? 0 : 1; }
