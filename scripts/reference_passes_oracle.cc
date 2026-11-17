// Observe the unmodified v0.12 FrameHeader writer, including implicit passes.
#include <iomanip>
#include <iostream>
#include <vector>
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"

jxl::Status Run() {
  JxlMemoryManager memory;
  JXL_RETURN_IF_ERROR(jxl::MemoryManagerInit(&memory, nullptr));
  for (int alpha : {0, 1}) {
    for (int type : {0, 2}) {
      std::vector<uint8_t> baseline;
      for (uint32_t count : {1, 2, 3}) {
        jxl::CodecMetadata metadata;
        JXL_RETURN_IF_ERROR(metadata.size.Set(64, 32));
        metadata.m.xyb_encoded = true;
        metadata.m.have_animation = true;
        metadata.m.SetAlphaBits(alpha ? 8 : 0);
        jxl::FrameHeader frame(&metadata);
        frame.encoding = jxl::FrameEncoding::kVarDCT;
        frame.color_transform = jxl::ColorTransform::kXYB;
        frame.frame_type = static_cast<jxl::FrameType>(type);
        frame.is_last = false;
        frame.save_as_reference = 2;
        frame.save_before_color_transform = type == 2;
        frame.animation_frame.duration = 10;
        frame.name = "passes-oracle";
        frame.passes.num_passes = count;
        frame.passes.num_downsample = 0;
        for (uint32_t i = 0; i < count - 1; ++i) frame.passes.shift[i] = 0;
        frame.extra_channel_blending_info.resize(alpha);
        jxl::BitWriter writer(&memory);
        JXL_RETURN_IF_ERROR(jxl::Bundle::Write(frame, &writer, jxl::LayerType::Header, nullptr));
        const size_t bits = writer.BitsWritten();
        JXL_RETURN_IF_ERROR(writer.WithMaxBits(8, jxl::LayerType::Header, nullptr, [&]() -> jxl::Status {
          writer.ZeroPadToByte(); return true;
        }));
        std::vector<uint8_t> bytes;
        for (uint8_t byte : writer.GetSpan()) bytes.push_back(byte);
        if (count == 1) baseline = bytes;
        else if ((bytes == baseline) != (type == 2)) return false;
        std::cout << alpha << ':' << type << ':' << count << '\t' << bits << '\t';
        for (uint8_t byte : bytes) std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
        std::cout << std::dec << '\n';
      }
    }
  }
  return true;
}
int main() { return Run() ? 0 : 1; }
