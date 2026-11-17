// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Observe actual unmodified v0.12 default/forward squeeze band metadata.
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "lib/jxl/base/status.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"
#include "lib/jxl/modular/transform/squeeze.h"
#include "lib/jxl/modular/transform/transform.h"
jxl::Status Run(JxlMemoryManager* memory, bool thin_alpha) {
  const size_t group_dim = thin_alpha ? 256 : 128;
  const auto shapes = thin_alpha
      ? std::vector<std::pair<size_t,size_t>>{{1025,17},{17,1025}}
      : std::vector<std::pair<size_t,size_t>>{{2,16449},{16449,2}};
  for (auto shape : shapes) {
    for (int count : thin_alpha ? std::vector<int>{1} : std::vector<int>{1,2}) {
      JXL_ASSIGN_OR_RETURN(auto image,jxl::Image::Create(memory,shape.first,shape.second,8,count));
      for (auto& channel:image.channel) for(size_t y=0;y<channel.h;++y)
        for(size_t x=0;x<channel.w;++x) channel.Row(y)[x]=static_cast<int>((x+37*y)%256);
      std::vector<jxl::SqueezeParams> params;
      jxl::DefaultSqueezeParameters(&params,image);
      JXL_RETURN_IF_ERROR(jxl::FwdSqueeze(image,params,nullptr));
      bool global=true;
      for(size_t i=0;i<image.channel.size();++i) {
        const auto& ch=image.channel[i];
        if(ch.w>group_dim || ch.h>group_dim) global=false;
        if(!global && ch.w && ch.h) {
          const unsigned extent=std::min(ch.hshift,ch.vshift)>=3?8*group_dim:group_dim;
          JXL_RETURN_IF_ERROR((extent>>std::max(ch.hshift,ch.vshift))!=0);
        }
        std::cout<<shape.first<<'\t'<<shape.second<<'\t'<<count<<'\t'<<i<<'\t'
          <<ch.w<<'\t'<<ch.h<<'\t'<<ch.hshift<<'\t'<<ch.vshift<<'\t'<<global<<'\n';
      }
    }
  }
  return true;
}
int main(int argc, char** argv) {
  const bool thin_alpha = argc == 2 && std::string(argv[1]) == "--thin-alpha";
  if(argc != 1 && !thin_alpha) return 3;
  JxlMemoryManager memory;
  if(!jxl::MemoryManagerInit(&memory,nullptr)) return 1;
  return Run(&memory,thin_alpha)?0:2;
}
