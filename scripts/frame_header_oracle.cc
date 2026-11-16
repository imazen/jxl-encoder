// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing
// Calls the unmodified libjxl v0.12 FrameHeader writer. No grammar transcription.
#include <iomanip>
#include <iostream>
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"

jxl::Status Run() {
  JxlMemoryManager memory;
  JXL_RETURN_IF_ERROR(jxl::MemoryManagerInit(&memory, nullptr));
  for (int modular : {0, 1}) {
    for (int alpha : {0, 1}) {
      for (int type : {0, 2, 3}) {
        for (int crop : {0, 1, 2, 3}) {
          jxl::CodecMetadata metadata;
          JXL_RETURN_IF_ERROR(metadata.size.Set(64, 32));
          metadata.m.xyb_encoded = !modular;
          metadata.m.have_animation = true;
          metadata.m.SetAlphaBits(alpha ? 8 : 0);
          jxl::FrameHeader frame(&metadata);
          frame.encoding = modular ? jxl::FrameEncoding::kModular : jxl::FrameEncoding::kVarDCT;
          frame.color_transform = modular ? jxl::ColorTransform::kNone : jxl::ColorTransform::kXYB;
          frame.frame_type = static_cast<jxl::FrameType>(type);
          frame.flags = modular ? 0 : 128;
          frame.x_qm_scale = 3;
          frame.b_qm_scale = 2;
          frame.loop_filter.gab = !modular;
          frame.loop_filter.epf_iters = modular ? 0 : 2;
          frame.is_last = false;
          frame.save_as_reference = 2;
          frame.save_before_color_transform = type == 2;
          frame.animation_frame.duration = 10;
          frame.name = "oracle";
          frame.custom_size_or_origin = crop != 0;
          frame.frame_origin.x0 = crop == 2 ? -1 : crop == 3 ? 1 : 0;
          frame.frame_origin.y0 = frame.frame_origin.x0;
          frame.frame_size.xsize = crop == 2 ? 66 : crop == 3 ? 32 : 64;
          frame.frame_size.ysize = crop == 2 ? 34 : crop == 3 ? 16 : 32;
          frame.blending_info.mode = jxl::BlendMode::kReplace;
          frame.blending_info.source = 1;
          frame.extra_channel_blending_info.resize(alpha);
          for (auto& blend : frame.extra_channel_blending_info) {
            blend.mode = jxl::BlendMode::kReplace;
            blend.source = 1;
          }
          jxl::BitWriter writer(&memory);
          JXL_RETURN_IF_ERROR(jxl::Bundle::Write(frame, &writer, jxl::LayerType::Header, nullptr));
          const size_t bits = writer.BitsWritten();
          JXL_RETURN_IF_ERROR(writer.WithMaxBits(8, jxl::LayerType::Header, nullptr, [&]() -> jxl::Status {
            writer.ZeroPadToByte();
            return true;
          }));
          std::cout << modular << ':' << alpha << ':' << type << ':' << crop << '\t' << bits << '\t';
          for (uint8_t byte : writer.GetSpan()) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
          }
          std::cout << std::dec << '\n';
        }
      }
    }
  }
  return true;
}
int main() { return Run() ? 0 : 1; }
