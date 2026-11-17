// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Observe the pinned v0.12 implementation, without copying its bracket algorithm.
#include <iostream>
#include "lib/jxl/frame_header.h"

int main() {
  for (int mode = 0; mode < 3; ++mode) {
    jxl::Passes passes;
    passes.num_passes = mode == 0 ? 1 : mode == 1 ? 2 : 3;
    passes.num_downsample = mode;
    if (mode == 1) {
      passes.downsample[0] = 2;
      passes.last_pass[0] = 0;
    }
    if (mode == 2) {
      passes.downsample[0] = 8;
      passes.last_pass[0] = 0;
      passes.downsample[1] = 4;
      passes.last_pass[1] = 1;
    }
    for (size_t pass = 0; pass < passes.num_passes; ++pass) {
      int minimum, maximum;
      passes.GetDownsamplingBracket(pass, minimum, maximum);
      std::cout << mode << '\t' << pass << '\t' << minimum << '\t'
                << maximum << '\n';
    }
  }
}
