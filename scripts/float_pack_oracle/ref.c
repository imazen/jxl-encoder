// Standalone oracle: libjxl v0.12.0 (a7a9c787) float<->int modular packing.
// float_to_int  transcribed from lib/jxl/enc_modular.cc:157
// int_to_float  transcribed from lib/jxl/dec_modular.cc:128
// Semantics preserved exactly; only the JXL_FAILURE/JXL_ENSURE plumbing is
// replaced by return codes so the failure cases are observable.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef int32_t pixel_type;

#define OK 0
#define ERR_PRECISION 1
#define ERR_EXPONENT 2
#define ERR_SUBNORMAL_RANGE 3

static int float_to_int(const float *row_in, pixel_type *row_out, size_t xsize,
                        unsigned bits, unsigned exp_bits) {
  if (bits == 32) {
    // JXL_ENSURE(exp_bits == 8)
    memcpy((void *)row_out, (const void *)row_in, 4 * xsize);
    return OK;
  }
  int exp_bias = (1 << (exp_bits - 1)) - 1;
  int max_exp = (1 << exp_bits) - 1;
  uint32_t sign = (1u << (bits - 1));
  int mant_bits = bits - exp_bits - 1;
  int mant_shift = 23 - mant_bits;
  for (size_t x = 0; x < xsize; ++x) {
    uint32_t f;
    memcpy(&f, &row_in[x], 4);
    int signbit = (f >> 31);
    f &= 0x7fffffff;
    if (f == 0) {
      row_out[x] = (signbit ? sign : 0);
      continue;
    }
    int exp = (f >> 23) - 127;
    int mantissa = (f & 0x007fffff);
    if (exp == 128) {  // NaN or infinity
      f = (signbit ? sign : 0);
      f |= ((1 << exp_bits) - 1) << mant_bits;
      f |= mantissa >> mant_shift;
      row_out[x] = (pixel_type)f;
      continue;
    }
    exp += exp_bias;
    if (exp <= 0) {  // subnormal
      mantissa |= 0x00800000;
      if (exp < -mant_bits) return ERR_SUBNORMAL_RANGE;
      mantissa >>= 1 - exp;
      exp = 0;
    }
    if (exp >= max_exp) return ERR_EXPONENT;
    if (mantissa & ((1 << mant_shift) - 1)) return ERR_PRECISION;
    mantissa >>= mant_shift;
    f = (signbit ? sign : 0);
    f |= (exp << mant_bits);
    f |= mantissa;
    row_out[x] = (pixel_type)f;
  }
  return OK;
}

static int int_to_float(const pixel_type *row_in, float *row_out, size_t xsize,
                        int bits, int exp_bits) {
  if (bits == 32) {
    memcpy(row_out, row_in, xsize * sizeof(float));
    return OK;
  }
  int exp_bias = (1 << (exp_bits - 1)) - 1;
  int sign_shift = bits - 1;
  int mant_bits = bits - exp_bits - 1;
  int mant_shift = 23 - mant_bits;
  for (size_t x = 0; x < xsize; ++x) {
    uint32_t f;
    memcpy(&f, &row_in[x], 4);
    int signbit = (f >> sign_shift);
    f &= (1 << sign_shift) - 1;
    if (f == 0) {
      row_out[x] = (signbit ? -0.f : 0.f);
      continue;
    }
    int exp = (f >> mant_bits);
    int mantissa = (f & ((1 << mant_bits) - 1));
    if (exp == (1 << exp_bits) - 1) {  // NaN or infinity
      f = (signbit ? 0x80000000u : 0u);
      f |= 0xffu << 23;
      f |= (uint32_t)mantissa << mant_shift;
      memcpy(&row_out[x], &f, 4);
      continue;
    }
    mantissa <<= mant_shift;
    if (exp == 0 && exp_bits < 8) {  // subnormal, renormalize
      while ((mantissa & 0x800000) == 0) {
        mantissa <<= 1;
        exp--;
      }
      exp++;
      mantissa &= 0x7fffff;
    }
    exp = exp - exp_bias + 127;
    f = (signbit ? 0x80000000u : 0u);
    f |= (uint32_t)exp << 23;
    f |= (uint32_t)mantissa;
    memcpy(&row_out[x], &f, 4);
  }
  return OK;
}

static float bits2f(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t f2bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

static void emit(const char *name, float v, unsigned bits, unsigned exp_bits) {
  pixel_type packed = 0;
  int rc = float_to_int(&v, &packed, 1, bits, exp_bits);
  printf("%s\t%u\t%u\t0x%08x\t%d\t", name, bits, exp_bits, f2bits(v), rc);
  if (rc == OK) {
    float back = 0.f;
    int rc2 = int_to_float(&packed, &back, 1, bits, exp_bits);
    printf("0x%08x\t%d\t0x%08x\n", (uint32_t)packed, rc2, f2bits(back));
  } else {
    printf("-\t-\t-\n");
  }
}

int main(void) {
  printf("name\tbits\texp_bits\tin_f32_bits\tpack_rc\tpacked\tunpack_rc\tout_f32_bits\n");
  struct { const char *n; uint32_t b; } cases[] = {
    {"zero",              0x00000000},
    {"neg_zero",          0x80000000},
    {"one",               0x3f800000},
    {"neg_one",           0xbf800000},
    {"two",               0x40000000},
    {"half",              0x3f000000},
    {"f16_max_65504",     0x477fe000},
    {"f16_min_normal",    0x38800000},
    {"f16_max_subnormal", 0x387fc000},
    {"f16_min_subnormal", 0x33800000},
    {"f16_subnormal_mid", 0x35000000},
    {"inf",               0x7f800000},
    {"neg_inf",           0xff800000},
    {"nan_quiet",         0x7fc00000},
    {"nan_payload",       0x7f800001},
    {"precision_loss",    0x3f800001},
    {"exp_too_large",     0x7f000000},
    {"exp_too_small",     0x00800000},
    {"f32_min_subnormal", 0x00000001},
    {"f32_max_finite",    0x7f7fffff},
    {"pi",                0x40490fdb},
    {"f16_repr_pi",       0x40480000},
  };
  unsigned fmts[][2] = {{32, 8}, {16, 5}, {24, 8}, {16, 8}};
  for (unsigned fi = 0; fi < 4; fi++)
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
      emit(cases[i].n, bits2f(cases[i].b), fmts[fi][0], fmts[fi][1]);
  return 0;
}
