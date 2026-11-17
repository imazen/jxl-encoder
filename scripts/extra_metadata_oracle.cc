// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Observe unmodified v0.12 bundle writing separately from public setter admission.
#include <iomanip>
#include <iostream>
#include "jxl/encode.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"

int main() {
  if (JxlEncoderVersion() != 12000) return 2;
  JxlMemoryManager memory;
  if (!jxl::MemoryManagerInit(&memory, nullptr)) return 3;
  struct Case { const char* name; JxlExtraChannelType type; uint32_t shift, cfa; float solidity; };
  for (const auto& c : {
      Case{"depth0", JXL_CHANNEL_DEPTH, 0, 0, 0},
      Case{"depth3", JXL_CHANNEL_DEPTH, 3, 0, 0},
      Case{"depth4", JXL_CHANNEL_DEPTH, 4, 0, 0},
      Case{"cfa0", JXL_CHANNEL_CFA, 0, 0, 0},
      Case{"cfa274", JXL_CHANNEL_CFA, 0, 274, 0},
      Case{"cfa275", JXL_CHANNEL_CFA, 0, 275, 0},
      Case{"spot0", JXL_CHANNEL_SPOT_COLOR, 0, 0, 0},
      Case{"spot1", JXL_CHANNEL_SPOT_COLOR, 0, 0, 1},
      Case{"spot2", JXL_CHANNEL_SPOT_COLOR, 0, 0, 2}}) {
    jxl::ExtraChannelInfo channel;
    channel.type = static_cast<jxl::ExtraChannel>(c.type);
    channel.dim_shift = c.shift;
    channel.cfa_channel = c.cfa;
    channel.spot_color[0] = -2.0f;
    channel.spot_color[1] = .5f;
    channel.spot_color[2] = 2.0f;
    channel.spot_color[3] = c.solidity;
    jxl::BitWriter writer(&memory);
    bool written = !!jxl::Bundle::Write(channel, &writer, jxl::LayerType::Header, nullptr);
    const auto bits = writer.BitsWritten();
    if (!writer.WithMaxBits(8, jxl::LayerType::Header, nullptr, [&]()->jxl::Status { writer.ZeroPadToByte(); return true; })) return 4;
    JxlEncoder* encoder = JxlEncoderCreate(nullptr);
    JxlBasicInfo basic; JxlEncoderInitBasicInfo(&basic);
    basic.xsize=32; basic.ysize=17; basic.num_extra_channels=1;
    if (JxlEncoderSetBasicInfo(encoder, &basic)!=JXL_ENC_SUCCESS) return 5;
    JxlExtraChannelInfo info; JxlEncoderInitExtraChannelInfo(c.type, &info);
    info.dim_shift=c.shift; info.cfa_channel=c.cfa; info.spot_color[3]=c.solidity;
    auto accepted=JxlEncoderSetExtraChannelInfo(encoder,0,&info)==JXL_ENC_SUCCESS;
    std::cout<<c.name<<'\t'<<accepted<<'\t'<<written<<'\t'<<bits<<'\t';
    for(uint8_t byte:writer.GetSpan()) std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(byte);
    std::cout<<std::dec<<'\n';
    JxlEncoderDestroy(encoder);
  }
}
