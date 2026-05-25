import math, struct

def f32(x): return struct.unpack('f', struct.pack('f', x))[0]
def f32_bits(x): return struct.unpack('I', struct.pack('f', x))[0]
sqrt2 = math.sqrt(2.0)

def fast_powf_approx(base, exp):
    """libjxl FastPowf — uses Log2/Pow2 polynomial approximations.
       Mean error ~1e-7, max ~3e-7 (5 ULP) in [1.0, 2.0].
       For this analysis use real pow but document the bias."""
    return f32(math.pow(base, exp))

def gen_table(params, num_bands, rows, cols, channel=1, use_f32=True):
    if use_f32:
        bands = [f32(params[0])]
        for i in range(1, num_bands):
            v = f32(params[i])
            mult = f32(1.0 + v) if v > 0.0 else f32(1.0 / f32(1.0 - v))
            bands.append(f32(bands[-1] * mult))
        scale = f32((num_bands - 1) / (f32(sqrt2) + 1e-6))
        rcpcol = f32(scale / (cols - 1))
        rcprow = f32(scale / (rows - 1))
        out = []
        for y in range(rows):
            dy = f32(y * rcprow)
            dy2 = f32(dy * dy)
            for x in range(cols):
                dx = f32(x * rcpcol)
                sd = f32(math.sqrt(f32(dx * dx + dy2)))
                idx = min(int(sd), num_bands - 2)
                frac = f32(sd - idx)
                a = bands[idx]; b = bands[idx + 1]
                ratio = f32(b / a)
                dequant = f32(a * fast_powf_approx(ratio, frac))
                out.append(f32(1.0 / dequant))
        return out
    else:
        bands = [params[0]]
        for i in range(1, num_bands):
            v = params[i]
            mult = (1.0 + v) if v > 0.0 else (1.0 / (1.0 - v))
            bands.append(bands[-1] * mult)
        scale = (num_bands - 1) / (sqrt2 + 1e-6)
        rcpcol = scale / (cols - 1)
        rcprow = scale / (rows - 1)
        out = []
        for y in range(rows):
            dy = y * rcprow
            dy2 = dy * dy
            for x in range(cols):
                dx = x * rcpcol
                sd = math.sqrt(dx * dx + dy2)
                idx = min(int(sd), num_bands - 2)
                frac = sd - idx
                a = bands[idx]; b = bands[idx + 1]
                dequant = a * math.pow(b / a, frac)
                out.append(f32(1.0 / dequant))
        return out

for name, params, nb, rows, cols in [
    ("DCT8 X",    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0], 6, 8, 8),
    ("DCT8 B",    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0], 6, 8, 8),
    ("DCT16 X",   [8996.8725711814115328, -1.3000777393353804, -0.49424529824571225, -0.439093774457103443, -0.6350101832695744, -0.90177264050827612, -1.6162099239887414], 7, 16, 16),
    ("DCT16 Y",   [3191.48366296844234752, -0.67424582104194355, -0.80745813428471001, -0.44925837484843441, -0.35865440981033403, -0.31322389111877305, -0.37615025315725483], 7, 16, 16),
    ("DCT16 B",   [1157.50408145487200256, -2.0531423165804414, -1.4, -0.50687130033378396, -0.42708730624733904, -1.4856834539296244, -4.9209142884401604], 7, 16, 16),
    ("DCT32 X",   [15718.40830982518931456, -1.025, -0.98, -0.9012, -0.4, -0.48819395464, -0.421064, -0.27], 8, 32, 32),
    ("DCT32 Y",   [7305.7636810695983104, -0.8041958212306401, -0.7633036457487539, -0.55660379990111464, -0.49785304658857626, -0.43699592683512467, -0.40180866526242109, -0.27321683125358037], 8, 32, 32),
    ("DCT32 B",   [3803.53173721215041536, -3.060733579805728, -2.0413270132490346, -2.0235650159727417, -0.5495389509954993, -0.4, -0.4, -0.3], 8, 32, 32),
]:
    f64_path = gen_table(params, nb, rows, cols, use_f32=False)
    f32_path = gen_table(params, nb, rows, cols, use_f32=True)
    ulp_diffs = [abs(f32_bits(f64_path[i]) - f32_bits(f32_path[i])) for i in range(len(f64_path))]
    max_ulp = max(ulp_diffs)
    mean_ulp = sum(ulp_diffs)/len(ulp_diffs)
    nonzero = sum(1 for u in ulp_diffs if u > 0)
    relerr = [abs(f64_path[i] - f32_path[i]) / abs(f64_path[i]) for i in range(len(f64_path))]
    max_rel = max(relerr)
    print(f"{name:<12} cells={len(f64_path):<5} max_ulp={max_ulp:<5} mean_ulp={mean_ulp:<6.2f} nonzero_ulp_count={nonzero:<5} max_relerr={max_rel:.3e}")
