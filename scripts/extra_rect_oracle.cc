// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Unmodified v0.12 Rect clipping after channel shifting, as used by
// enc_modular.cc PrepareStreamParams with its unclipped AC group rectangles.
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "lib/jxl/base/rect.h"
int main(int argc, char** argv) {
  const bool lf = argc == 2 && std::string(argv[1]) == "--lf";
  std::cout << "# libjxl v0.12.0 a7a9c78 Rect / enc_modular.cc group geometry\n";
  for (size_t group_shift = 0; group_shift <= 3; ++group_shift) {
    const size_t group = (size_t{128} << group_shift) * (lf ? 8 : 1);
    for (size_t shift = lf ? 3 : 0; shift <= 3; ++shift) {
      const size_t factor = size_t{1} << shift;
      for (const auto& dimensions : std::vector<std::pair<size_t,size_t>>{
          {group-1, factor}, {group, factor}, {group+1, factor},
          {factor, group+1}, {group+1, group+1}}) {
        const size_t width = dimensions.first, height = dimensions.second;
        const size_t channel_width = (width + factor - 1) / factor;
        const size_t channel_height = (height + factor - 1) / factor;
        for (size_t y = 0; y < height; y += group) {
          for (size_t x = 0; x < width; x += group) {
            const jxl::Rect unclipped(x,y,group,group);
            const jxl::Rect local(unclipped.x0() >> shift,
                unclipped.y0() >> shift, unclipped.xsize() >> shift,
                unclipped.ysize() >> shift, channel_width, channel_height);
            std::cout << width << '\t' << height << '\t' << group_shift << '\t'
                << shift << '\t' << x << '\t' << y << '\t' << local.x0()
                << '\t' << local.y0() << '\t' << local.xsize() << '\t'
                << local.ysize() << '\n';
          }
        }
      }
    }
  }
}
