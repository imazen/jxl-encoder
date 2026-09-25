// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Calls the unmodified libjxl v0.12 EstimateCost implementation.
// Input: case-name channel-count, followed by width height and i32 samples
// for each channel. Output: case-name cost (integer-valued f32).
#include <cstdint>
#include <iostream>
#include <string>
#include "lib/jxl/modular/transform/transform.h"
#include "lib/jxl/enc_modular_simd.h"
#include "lib/jxl/memory_manager_internal.h"

jxl::Status Run() {
  JxlMemoryManager memory;
  JXL_RETURN_IF_ERROR(jxl::MemoryManagerInit(&memory, nullptr));
  std::string name;
  size_t channels;
  while (std::cin >> name >> channels) {
    jxl::Image image(&memory);
    for (size_t c = 0; c < channels; ++c) {
      size_t w, h;
      if (!(std::cin >> w >> h)) return false;
      JXL_ASSIGN_OR_RETURN(auto channel, jxl::Channel::Create(&memory, w, h));
      for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
          if (!(std::cin >> channel.Row(y)[x])) return false;
        }
      }
      image.channel.push_back(std::move(channel));
    }
    JXL_ASSIGN_OR_RETURN(float cost, jxl::EstimateCost(image));
    std::cout << name << '\t' << std::fixed << cost << '\n';
  }
  return true;
}
int main() { return Run() ? 0 : 1; }
