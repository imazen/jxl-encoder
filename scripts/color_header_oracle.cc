// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Calls the unmodified libjxl v0.12 ColorEncoding writer.
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
  for (int icc : {0, 1}) {
    for (int space : {0, 1, 2, 3}) {
      // v0.12 refuses Unknown without ICC, independently of wire grammar.
      if (!icc && space == 3) continue;
      for (int intent : {0, 1, 2, 3}) {
        if (space == 2 && intent != 0) continue;
        jxl::ColorEncoding color;
        JXL_RETURN_IF_ERROR(color.SetSRGB(jxl::ColorSpace::kRGB));
        color.SetColorSpace(static_cast<jxl::ColorSpace>(space));
        color.SetRenderingIntent(static_cast<jxl::RenderingIntent>(intent));
        if (icc) color.SetICCRaw(jxl::IccBytes{0});
        jxl::BitWriter writer(&memory);
        JXL_RETURN_IF_ERROR(jxl::Bundle::Write(color, &writer, jxl::LayerType::Header, nullptr));
        const size_t bits = writer.BitsWritten();
        JXL_RETURN_IF_ERROR(writer.WithMaxBits(8, jxl::LayerType::Header, nullptr, [&]() -> jxl::Status {
          writer.ZeroPadToByte();
          return true;
        }));
        std::cout << icc << ':' << space << ':' << intent << '\t' << bits << '\t';
        for (uint8_t byte : writer.GetSpan())
          std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
        std::cout << std::dec << '\n';
      }
    }
  }
  return true;
}
int main() { return Run() ? 0 : 1; }
