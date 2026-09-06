#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "lib/jxl/enc_xyb.h"
namespace jxl { void ComputePremulAbsorb(float, float*); void LinearRGBRowToXYB(float*,float*,float*,const float*,size_t); }
static float gen_px(uint32_t &s){ s^=s<<13; s^=s>>17; s^=s<<5; return (float)(s>>8)*(1.0f/16777216.0f); }
int main(){
  const int n = 256;
  const int cap = n + 128;            // SIMD tail slack
  auto alloc = [&](float** p){ void* q=nullptr; if(posix_memalign(&q,128,sizeof(float)*cap)!=0) exit(1);
                               memset(q,0,sizeof(float)*cap); *p=(float*)q; };
  float *r,*g,*b; alloc(&r); alloc(&g); alloc(&b);
  uint32_t s = 0xC0FFEEu;
  for (int i=0;i<n;i++){ r[i]=gen_px(s); g[i]=gen_px(s); b[i]=gen_px(s); }
  void* pa_raw=nullptr;
  if (posix_memalign(&pa_raw, 128, sizeof(float)*12*32)!=0) return 1;
  float* premul = (float*)pa_raw;
  jxl::ComputePremulAbsorb(255.0f, premul);
  printf("XYB %d\n", n);
  for (int i=0;i<n;i++) printf("%08x %08x %08x\n",
      *(uint32_t*)&r[i], *(uint32_t*)&g[i], *(uint32_t*)&b[i]);
  jxl::LinearRGBRowToXYB(r, g, b, premul, n);
  printf("OUT\n");
  for (int i=0;i<n;i++) printf("%08x %08x %08x\n",
      *(uint32_t*)&r[i], *(uint32_t*)&g[i], *(uint32_t*)&b[i]);
  return 0;
}
