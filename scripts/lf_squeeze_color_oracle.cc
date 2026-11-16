// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Compare unmodified v0.12 forward squeeze on all channels and the XYB prefix.
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "lib/jxl/base/status.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"
#include "lib/jxl/modular/transform/squeeze.h"
#include "lib/jxl/modular/transform/transform.h"

jxl::StatusOr<jxl::Image> Make(JxlMemoryManager* memory, size_t w, size_t h, int channels) {
  JXL_ASSIGN_OR_RETURN(auto image, jxl::Image::Create(memory, w, h, 16, channels));
  for (int c = 0; c < channels; ++c) {
    image.channel[c].component = c < 3 ? c : -1;
    for (size_t y = 0; y < h; ++y) for (size_t x = 0; x < w; ++x)
      image.channel[c].Row(y)[x] = c < 3 ? static_cast<int>((x*37+y*97+c*211)%4096)-2048 : 0;
  }
  return image;
}

jxl::Status Run(JxlMemoryManager* memory) {
  for (auto shape : std::vector<std::pair<size_t,size_t>>{
      {1,1},{1,17},{17,1},{16,16},{17,9},{257,1},{1,257},{257,257}}) {
    for (int channels : {19,20,21}) {
      JXL_ASSIGN_OR_RETURN(auto full, Make(memory, shape.first, shape.second, channels));
      JXL_ASSIGN_OR_RETURN(auto color, Make(memory, shape.first, shape.second, channels));
      std::vector<jxl::SqueezeParams> params;
      jxl::DefaultSqueezeParameters(&params, full);
      auto restricted = params;
      for (auto& step : restricted) if (step.num_c == static_cast<unsigned>(channels)) step.num_c = 3;
      JXL_RETURN_IF_ERROR(jxl::FwdSqueeze(full, params, nullptr));
      JXL_RETURN_IF_ERROR(jxl::FwdSqueeze(color, restricted, nullptr));
      std::vector<const jxl::Channel*> a,b;
      for (const auto& ch : full.channel) if (ch.component >= 0 && ch.w && ch.h) a.push_back(&ch);
      for (const auto& ch : color.channel) if (ch.component >= 0 && ch.w && ch.h) b.push_back(&ch);
      JXL_RETURN_IF_ERROR(a.size() == b.size());
      size_t values = 0;
      for (size_t i = 0; i < a.size(); ++i) {
        JXL_RETURN_IF_ERROR(a[i]->w == b[i]->w && a[i]->h == b[i]->h &&
            a[i]->hshift == b[i]->hshift && a[i]->vshift == b[i]->vshift && a[i]->component == b[i]->component);
        for (size_t y = 0; y < a[i]->h; ++y)
          JXL_RETURN_IF_ERROR(std::equal(a[i]->Row(y), a[i]->Row(y)+a[i]->w, b[i]->Row(y)));
        values += a[i]->w * a[i]->h;
      }
      for (const auto* image : {&full, &color}) for (const auto& ch : image->channel)
        if (ch.component < 0) for (size_t y = 0; y < ch.h; ++y)
          for (size_t x = 0; x < ch.w; ++x) JXL_RETURN_IF_ERROR(ch.Row(y)[x] == 0);
      std::cout << shape.first << '\t' << shape.second << '\t' << channels << '\t'
                << a.size() << '\t' << values << "\tEXACT\n";
    }
  }
  return true;
}
int main() {
  JxlMemoryManager memory;
  if (!jxl::MemoryManagerInit(&memory, nullptr)) return 1;
  return Run(&memory) ? 0 : 2;
}
