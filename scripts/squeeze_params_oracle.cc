// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Calls the unmodified v0.12 explicit squeeze field writer.
#include <iostream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/squeeze.h"
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/transform/squeeze_params.h"
jxl::Status PrintDefaults(JxlMemoryManager& memory) {
    for (const auto shape : std::vector<std::pair<size_t, size_t>>{
        {1,17}, {17,1}, {8,8}, {16,16}, {259,17}, {17,259}, {8,65665}, {65665,8}}) {
      for (const int channels : {3, 19, 20, 21, 259}) {
        if (channels == 259 && shape != std::make_pair(size_t{16}, size_t{16})) continue;
        JXL_ASSIGN_OR_RETURN(auto image, jxl::Image::Create(&memory, shape.first, shape.second, 8, channels));
        std::vector<jxl::SqueezeParams> defaults;
        jxl::DefaultSqueezeParameters(&defaults, image);
        for (size_t index = 0; index < defaults.size(); ++index) {
          const auto& param = defaults[index];
          std::cout << shape.first << "\t" << shape.second << "\t" << channels
                    << "\t" << index << "\t" << param.horizontal << "\t"
                    << param.in_place << "\t" << param.begin_c << "\t" << param.num_c << "\n";
        }
      }
    }
  return true;
}
jxl::Status PrintWire(JxlMemoryManager& memory) {
  for (unsigned count : {0, 1, 2, 3, 4, 19}) {
    jxl::Transform transform(jxl::TransformId::kSqueeze);
    if (count != 0) {
      jxl::SqueezeParams parameter;
      parameter.horizontal = true; parameter.in_place = true;
      parameter.begin_c = 0; parameter.num_c = count;
      transform.squeezes.push_back(parameter);
    }
    jxl::BitWriter writer(&memory);
    JXL_RETURN_IF_ERROR(jxl::Bundle::Write(transform, &writer, jxl::LayerType::Header, nullptr));
    const size_t bits = writer.BitsWritten();
    JXL_RETURN_IF_ERROR(writer.WithMaxBits(8, jxl::LayerType::Header, nullptr, [&]() -> jxl::Status {
      writer.ZeroPadToByte(); return true;
    }));
    std::cout << count << "\t" << bits << "\t";
    for (uint8_t byte : writer.GetSpan())
      std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    std::cout << std::dec << "\n";
  }
  return true;
}
int main(int argc, char** argv) {
  JxlMemoryManager memory;
  if (!jxl::MemoryManagerInit(&memory, nullptr)) return 1;
  if (argc == 2 && std::string(argv[1]) == "--wire") return PrintWire(memory) ? 0 : 2;
  if (argc == 2 && std::string(argv[1]) == "--defaults") {
    return PrintDefaults(memory) ? 0 : 2;
  }
  if (argc == 2 && std::string(argv[1]) == "--transform-counts") {
    for (unsigned count : {0, 1, 16, 17, 72, 73, 296, 297}) {
      jxl::Transform transform(jxl::TransformId::kSqueeze);
      jxl::SqueezeParams parameter;
      parameter.horizontal = true; parameter.in_place = true;
      parameter.begin_c = 0; parameter.num_c = 1;
      transform.squeezes.assign(count, parameter);
      jxl::BitWriter writer(&memory);
      auto result = jxl::Bundle::Write(transform, &writer, jxl::LayerType::Header, nullptr);
      std::cout << count << "\t" << static_cast<bool>(result) << "\n";
    }
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--begin-fields") {
    for (unsigned begin : {0, 7, 8, 71, 72, 1095, 1096, 9287, 9288}) {
      jxl::SqueezeParams parameter;
      parameter.horizontal = true; parameter.in_place = true;
      parameter.begin_c = begin; parameter.num_c = 1;
      jxl::BitWriter writer(&memory);
      auto result = jxl::Bundle::Write(parameter, &writer, jxl::LayerType::Header, nullptr);
      std::cout << begin << "\t" << static_cast<bool>(result) << "\n";
    }
    return 0;
  }
  for (unsigned count : {0, 1, 2, 3, 4, 18, 19, 20, 21}) {
    jxl::SqueezeParams params;
    params.horizontal = true; params.in_place = true;
    params.begin_c = 0; params.num_c = count;
    jxl::BitWriter writer(&memory);
    auto result = jxl::Bundle::Write(params, &writer, jxl::LayerType::Header, nullptr);
    std::cout << count << "\t" << static_cast<bool>(result) << "\n";
  }
}
