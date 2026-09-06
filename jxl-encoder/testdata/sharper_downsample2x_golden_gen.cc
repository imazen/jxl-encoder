// Golden-vector generator: calls libjxl's REAL DownsampleImage2_Sharper on
// procedurally generated planes and dumps the outputs as hex f32 bits.
#include <cstdio>
#include <cstdint>
#include <vector>
#include "lib/jxl/image.h"
#include "lib/jxl/enc_heuristics.h"
#include "lib/jxl/base/status.h"
#include <jxl/memory_manager.h>
#include <cstdlib>
#include <cstring>

// Identical PRNG in the Rust test. xorshift32 -> [0,1), then per-channel range.
static float gen_px(uint32_t &s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  return (float)(s >> 8) * (1.0f / 16777216.0f);
}
static float map_c(int c, float u) {
  if (c == 0) return -1.0f + u * 0.99f;   // ALL NEGATIVE: exposes max-init
  if (c == 1) return u;                    // [0,1)
  return u - 0.5f;                         // [-0.5,0.5)
}

int main() {
  const int shapes[][2] = {{16,16},{17,13},{33,9},{8,8},{31,31},{129,97},{256,256}};
  // Own manager: malloc/free with 64-byte alignment (libjxl only needs the
  // two callbacks; this avoids depending on a test-only helper).
  JxlMemoryManager mm_storage;
  mm_storage.opaque = nullptr;
  mm_storage.alloc = [](void*, size_t size) -> void* {
    void* p = nullptr;
    if (posix_memalign(&p, 64, size ? size : 1) != 0) return nullptr;
    return p;
  };
  mm_storage.free = [](void*, void* address) { ::free(address); };
  JxlMemoryManager* mm = &mm_storage;
  for (auto &sh : shapes) {
    int w = sh[0], h = sh[1];
    auto opsin_or = jxl::Image3F::Create(mm, w, h);
    if (!opsin_or.ok()) { fprintf(stderr, "create failed\n"); return 1; }
    jxl::Image3F opsin = std::move(opsin_or).value_();
    for (int c = 0; c < 3; c++) {
      uint32_t s = 0x9E3779B9u ^ (uint32_t)(w*73856093) ^ (uint32_t)(h*19349663) ^ (uint32_t)(c*83492791);
      if (s == 0) s = 1;
      for (int y = 0; y < h; y++) {
        float* row = opsin.PlaneRow(c, y);
        for (int x = 0; x < w; x++) row[x] = map_c(c, gen_px(s));
      }
    }
    if (!jxl::DownsampleImage2_Sharper(&opsin)) { fprintf(stderr, "downsample failed\n"); return 1; }
    // Small shapes: every value, exactly. Large shapes: an FNV-1a 64 hash of
    // the same bit stream, so scale coverage costs a line instead of a
    // megabyte (the repo forbids committing large files).
    bool big = (w * h) > 2048;
    uint64_t fnv = 0xcbf29ce484222325ULL;   // FNV-1a 64 offset basis
    printf("%s %d %d %zu %zu\n", big ? "HASH" : "SHAPE", w, h, opsin.xsize(), opsin.ysize());
    for (int c = 0; c < 3; c++) {
      for (size_t y = 0; y < opsin.ysize(); y++) {
        const float* row = opsin.PlaneRow(c, y);
        for (size_t x = 0; x < opsin.xsize(); x++) {
          uint32_t bits; memcpy(&bits, &row[x], 4);
          if (big) {
            for (int k = 0; k < 4; k++) {
              fnv ^= (uint64_t)((bits >> (8 * k)) & 0xff);
              fnv *= 0x100000001b3ULL;
            }
          } else {
            printf("%08x\n", bits);
          }
        }
      }
    }
    if (big) printf("%016llx\n", (unsigned long long)fnv);
  }
  return 0;
}
