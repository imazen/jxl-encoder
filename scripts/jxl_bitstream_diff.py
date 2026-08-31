#!/usr/bin/env python3
"""Field-level differential for two JPEG XL codestreams (T4 libjxl-mimic harness).

Answers the question a byte `cmp` cannot: *which field* diverged, not "byte 4711
onward". Parses the structural envelope of a bare codestream

    signature -> SizeHeader -> ImageMetadata -> CustomTransformData
              -> [ICC] -> FrameHeader -> TOC -> sections

and reports (a) the first field whose value differs and (b) the per-TOC-section
byte table with per-section deltas, which localises the residual to
`LfGlobal` / `LfGroup k` / `HfGlobal` / `HfGroup pass p group g`.

Deliberately a standalone re-implementation of the bit layout read straight from
libjxl `d089091a` (`fields.h`, `image_metadata.cc`, `frame_header.cc`,
`loop_filter.cc`, `toc.cc`, `frame_dimensions.h`) rather than a call into our own
`jxl-encoder` reader or into `jxl-rs`/`jxl-oxide`. A differential harness that
shares code with the encoder under test cannot see a bug the two share; this one
can. It is also dependency-free, so it runs anywhere without a build.

Usage
-----
    jxl_bitstream_diff.py trace  a.jxl
    jxl_bitstream_diff.py diff   cjxl.jxl ours.jxl
    jxl_bitstream_diff.py toc    a.jxl [b.jxl ...]     # TOC size table only
    jxl_bitstream_diff.py tsv    cjxl.jxl ours.jxl     # machine-readable diff

Exit status for `diff`: 0 identical, 1 fields differ, 2 parse failure.

Scope: everything up to and including the TOC is parsed field-by-field. Section
*contents* (ANS histograms, MA trees, token streams) are reported by size only —
localising to a section is the actionable unit, and a full entropy decoder here
would duplicate `jxl-rs` without the independence benefit (the section boundary
is checkable from the TOC alone, which is why it is the right stopping point).
"""

import sys
import struct

# ── bit reader ──────────────────────────────────────────────────────────────
# JXL packs bits LSB-first inside each byte, bytes in stream order
# (libjxl `BitReader`).


class BitReader:
    def __init__(self, data, byte_start=0):
        self.d = data
        self.pos = byte_start * 8

    def bit(self):
        if (self.pos >> 3) >= len(self.d):
            raise EOFError("ran off the end of the codestream")
        b = (self.d[self.pos >> 3] >> (self.pos & 7)) & 1
        self.pos += 1
        return b

    def bits(self, n):
        v = 0
        for i in range(n):
            v |= self.bit() << i
        return v

    def jump_to_byte_boundary(self):
        while self.pos & 7:
            self.bit()

    def u32(self, dists):
        """U32Coder: 2-bit selector picks one of four distributions.
        Each dist is ("V", value) | ("B", nbits) | ("O", (nbits, offset))."""
        sel = self.bits(2)
        kind, arg = dists[sel]
        if kind == "V":
            return arg
        if kind == "B":
            return self.bits(arg)
        n, off = arg
        return self.bits(n) + off

    def u64(self):
        """U64Coder (libjxl fields.cc)."""
        sel = self.bits(2)
        if sel == 0:
            return 0
        if sel == 1:
            return 1 + self.bits(4)
        if sel == 2:
            return 17 + self.bits(8)
        v = self.bits(12)
        shift = 12
        while self.bit():
            if shift == 60:
                v |= self.bits(4) << shift
                break
            v |= self.bits(8) << shift
            shift += 8
        return v

    def f16(self):
        b = self.bits(16)
        sign = (b >> 15) & 1
        exp = (b >> 10) & 0x1F
        mant = b & 0x3FF
        if exp == 0:
            val = (mant / 1024.0) * (2.0**-14)
        elif exp == 31:
            val = float("inf") if mant == 0 else float("nan")
        else:
            val = (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))
        return -val if sign else val

    def f32(self):
        return struct.unpack("<f", struct.pack("<I", self.bits(32)))[0]

    def enum(self, ):
        """Enum coder: U32(Val(0), Val(1), BitsOffset(4,2), BitsOffset(6,18))."""
        return self.u32([("V", 0), ("V", 1), ("O", (4, 2)), ("O", (6, 18))])


def div_ceil(a, b):
    return (a + b - 1) // b


# ── U32 distribution tables lifted from libjxl ──────────────────────────────

SIZE_DIST = [("B", 9), ("B", 13), ("B", 18), ("B", 30)]  # SizeHeader (+1)
TOC_DIST = [("B", 10), ("O", (14, 1024)), ("O", (22, 17408)), ("O", (30, 4211712))]
CROP_DIST = [("B", 8), ("O", (11, 256)), ("O", (14, 2304)), ("O", (30, 18688))]

COLOR_SPACE = {0: "kRGB", 1: "kGray", 2: "kXYB", 3: "kUnknown"}
WHITE_POINT = {1: "kD65", 2: "kCustom", 10: "kE", 11: "kDCI"}
PRIMARIES = {1: "kSRGB", 2: "kCustom", 9: "k2100", 11: "kP3"}
TRANSFER_FN = {1: "k709", 2: "kUnknown", 8: "kLinear", 13: "kSRGB",
               16: "kPQ", 17: "kDCI", 18: "kHLG"}
RENDERING_INTENT = {0: "kPerceptual", 1: "kRelative", 2: "kSaturation",
                    3: "kAbsolute"}
FRAME_TYPE = {0: "kRegularFrame", 1: "kLFFrame", 2: "kReferenceOnly",
              3: "kSkipProgressive"}
BLEND_MODE = {0: "kReplace", 1: "kAdd", 2: "kBlend", 3: "kAlphaWeightedAdd",
              4: "kMul"}


class Parser:
    """Walks the codestream envelope, appending (name, value) in stream order."""

    def __init__(self, data):
        self.d = data
        self.fields = []          # [(bitpos, name, value)]
        self.toc = None           # {"entries": [...], "labels": [...], ...}
        self.note = []

    def f(self, name, value):
        self.fields.append((self.br.pos, name, value))
        return value

    # ── SizeHeader ──────────────────────────────────────────────────────────
    def size_header(self, prefix="size"):
        br = self.br
        small = self.f(f"{prefix}.small", br.bit())
        if small:
            ys = (br.bits(5) + 1) * 8
        else:
            ys = br.u32(SIZE_DIST) + 1
        self.f(f"{prefix}.ysize", ys)
        ratio = self.f(f"{prefix}.ratio", br.bits(3))
        if ratio == 0:
            xs = (br.bits(5) + 1) * 8 if small else br.u32(SIZE_DIST) + 1
        else:
            xs = {1: ys, 2: ys * 12 // 10, 3: ys * 4 // 3, 4: ys * 3 // 2,
                  5: ys * 16 // 9, 6: ys * 5 // 4, 7: ys * 2}[ratio]
        self.f(f"{prefix}.xsize", xs)
        return xs, ys

    # ── BitDepth ────────────────────────────────────────────────────────────
    def bit_depth(self, p):
        br = self.br
        fp = self.f(f"{p}.floating_point", br.bit())
        if not fp:
            self.f(f"{p}.bits_per_sample",
                   br.u32([("V", 8), ("V", 10), ("V", 12), ("O", (6, 1))]))
        else:
            self.f(f"{p}.bits_per_sample",
                   br.u32([("V", 32), ("V", 16), ("V", 24), ("O", (6, 1))]))
            self.f(f"{p}.exponent_bits",
                   br.u32([("V", 8), ("V", 5), ("V", 10), ("O", (6, 1))]))

    # ── ColorEncoding ───────────────────────────────────────────────────────
    def custom_xy(self, p):
        br = self.br
        x = br.u32([("B", 19), ("B", 19), ("B", 19), ("B", 19)])
        y = br.u32([("B", 19), ("B", 19), ("B", 19), ("B", 19)])
        self.f(f"{p}.xy", (x, y))

    def color_encoding(self, p="color_encoding"):
        br = self.br
        if self.f(f"{p}.all_default", br.bit()):
            return
        want_icc = self.f(f"{p}.want_icc", br.bit())
        cs = br.enum()
        self.f(f"{p}.color_space", COLOR_SPACE.get(cs, cs))
        if not want_icc:
            if cs != 2:  # kXYB implies D65
                wp = br.enum()
                self.f(f"{p}.white_point", WHITE_POINT.get(wp, wp))
                if wp == 2:
                    self.custom_xy(f"{p}.white")
            if cs != 1 and cs != 2:  # not gray, not XYB
                pr = br.enum()
                self.f(f"{p}.primaries", PRIMARIES.get(pr, pr))
                if pr == 2:
                    for c in ("red", "green", "blue"):
                        self.custom_xy(f"{p}.{c}")
            have_gamma = self.f(f"{p}.have_gamma", br.bit())
            if have_gamma:
                self.f(f"{p}.gamma", br.bits(24))
            else:
                tf = br.enum()
                self.f(f"{p}.transfer_function", TRANSFER_FN.get(tf, tf))
            ri = br.enum()
            self.f(f"{p}.rendering_intent", RENDERING_INTENT.get(ri, ri))
        return want_icc

    # ── ImageMetadata ───────────────────────────────────────────────────────
    def extra_channel_info(self, i):
        br = self.br
        p = f"metadata.extra_channel[{i}]"
        if self.f(f"{p}.all_default", br.bit()):
            return
        self.f(f"{p}.type", br.enum())
        self.bit_depth(f"{p}.bit_depth")
        self.f(f"{p}.dim_shift",
               br.u32([("V", 0), ("V", 3), ("V", 4), ("O", (3, 1))]))
        nlen = self.f(f"{p}.name_len",
                      br.u32([("V", 0), ("O", (4, 1)), ("O", (5, 17)),
                              ("O", (10, 49))]))
        for _ in range(nlen):
            br.bits(8)

    def tone_mapping(self, p="metadata.tone_mapping"):
        br = self.br
        if self.f(f"{p}.all_default", br.bit()):
            return
        self.f(f"{p}.intensity_target", br.f16())
        self.f(f"{p}.min_nits", br.f16())
        self.f(f"{p}.relative_to_max_display", br.bit())
        self.f(f"{p}.linear_below", br.f16())

    def extensions(self, p):
        self.f(f"{p}.extensions", self.br.u64())

    def image_metadata(self):
        br = self.br
        self.have_animation = False
        self.num_extra_channels = 0
        self.xyb_encoded = True
        if self.f("metadata.all_default", br.bit()):
            return
        ef = self.f("metadata.extra_fields", br.bit())
        if ef:
            self.f("metadata.orientation", br.bits(3) + 1)
            if self.f("metadata.have_intrinsic_size", br.bit()):
                self.size_header("metadata.intrinsic")
            if self.f("metadata.have_preview", br.bit()):
                self.size_header("metadata.preview")
            self.have_animation = self.f("metadata.have_animation", br.bit())
            if self.have_animation:
                self.animation()
        self.bit_depth("metadata.bit_depth")
        self.f("metadata.modular_16bit_buffers", br.bit())
        nec = self.f("metadata.num_extra_channels",
                     br.u32([("V", 0), ("V", 1), ("O", (4, 2)), ("O", (12, 1))]))
        self.num_extra_channels = nec
        for i in range(nec):
            self.extra_channel_info(i)
        self.xyb_encoded = self.f("metadata.xyb_encoded", br.bit())
        self.want_icc = self.color_encoding("metadata.color_encoding")
        if ef:
            self.tone_mapping()
        self.extensions("metadata")

    def animation(self):
        br = self.br
        self.f("metadata.animation.tps_numerator",
               br.u32([("V", 100), ("V", 1000), ("O", (10, 1)), ("O", (30, 1))]))
        self.f("metadata.animation.tps_denominator",
               br.u32([("V", 1), ("V", 1001), ("O", (8, 1)), ("O", (10, 1))]))
        self.f("metadata.animation.num_loops",
               br.u32([("V", 0), ("B", 3), ("B", 16), ("B", 32)]))
        self.f("metadata.animation.have_timecodes", br.bit())

    # ── CustomTransformData ─────────────────────────────────────────────────
    def custom_transform_data(self):
        br = self.br
        if self.f("transform_data.all_default", br.bit()):
            return
        if self.xyb_encoded:
            if self.f("transform_data.opsin_inverse_matrix.all_default", br.bit()) == 0:
                for i in range(9):
                    self.f(f"transform_data.opsin.inverse[{i}]", br.f32())
                for i in range(3):
                    self.f(f"transform_data.opsin.bias[{i}]", br.f32())
                for i in range(3):
                    self.f(f"transform_data.opsin.cbrt_bias[{i}]", br.f32())
        mask = self.f("transform_data.custom_weights_mask", br.bits(3))
        for bit, count in ((1, 15), (2, 55), (4, 210)):
            if mask & bit:
                for i in range(count):
                    self.f(f"transform_data.upsampling{bit*2}_w[{i}]", br.f16())

    # ── FrameHeader ─────────────────────────────────────────────────────────
    def blending_info(self, p, num_ec, is_partial):
        br = self.br
        mode = br.u32([("V", 0), ("V", 1), ("V", 2), ("O", (2, 3))])
        self.f(f"{p}.mode", BLEND_MODE.get(mode, mode))
        if num_ec > 0 and mode in (2, 3):
            self.f(f"{p}.alpha_channel",
                   br.u32([("V", 0), ("V", 1), ("V", 2), ("O", (3, 3))]))
        if (num_ec > 0 and mode in (2, 3)) or mode == 4:
            self.f(f"{p}.clamp", br.bit())
        if mode != 0 or is_partial:
            self.f(f"{p}.source",
                   br.u32([("V", 0), ("V", 1), ("V", 2), ("V", 3)]))
        return mode

    def passes(self):
        br = self.br
        n = self.f("frame.passes.num_passes",
                   br.u32([("V", 1), ("V", 2), ("V", 3), ("O", (3, 4))]))
        if n != 1:
            nd = self.f("frame.passes.num_downsample",
                        br.u32([("V", 0), ("V", 1), ("V", 2), ("O", (1, 3))]))
            for i in range(n - 1):
                self.f(f"frame.passes.shift[{i}]", br.bits(2))
            for i in range(nd):
                self.f(f"frame.passes.downsample[{i}]",
                       br.u32([("V", 1), ("V", 2), ("V", 4), ("V", 8)]))
            for i in range(nd):
                self.f(f"frame.passes.last_pass[{i}]",
                       br.u32([("V", 0), ("V", 1), ("V", 2), ("B", 3)]))
        return n

    def loop_filter(self, is_modular):
        br = self.br
        if self.f("frame.loop_filter.all_default", br.bit()):
            return
        gab = self.f("frame.loop_filter.gab", br.bit())
        if gab:
            if self.f("frame.loop_filter.gab_custom", br.bit()):
                for n in ("x1", "x2", "y1", "y2", "b1", "b2"):
                    self.f(f"frame.loop_filter.gab_{n}", br.f16())
        epf = self.f("frame.loop_filter.epf_iters", br.bits(2))
        if epf > 0:
            if not is_modular:
                if self.f("frame.loop_filter.epf_sharp_custom", br.bit()):
                    for i in range(8):
                        self.f(f"frame.loop_filter.epf_sharp_lut[{i}]", br.f16())
            if self.f("frame.loop_filter.epf_weight_custom", br.bit()):
                for n in ("cs0", "cs1", "cs2", "p1zf", "p2zf"):
                    self.f(f"frame.loop_filter.epf_{n}", br.f16())
            if self.f("frame.loop_filter.epf_sigma_custom", br.bit()):
                if not is_modular:
                    self.f("frame.loop_filter.epf_quant_mul", br.f16())
                for n in ("p0sig", "p2sig", "border_sad_mul"):
                    self.f(f"frame.loop_filter.epf_{n}", br.f16())
            if is_modular:
                self.f("frame.loop_filter.epf_sigma_for_modular", br.f16())
        self.extensions("frame.loop_filter")

    def name_string(self, p):
        br = self.br
        n = self.f(f"{p}.name_len",
                   br.u32([("V", 0), ("O", (4, 1)), ("O", (5, 17)),
                           ("O", (10, 49))]))
        if n:
            self.f(f"{p}.name", bytes(br.bits(8) for _ in range(n)))

    def frame_header(self, image_xsize, image_ysize):
        br = self.br
        # Defaults, overwritten when all_default == 0
        self.frame = dict(is_modular=False, upsampling=1, num_passes=1,
                          group_size_shift=1, custom_size=False,
                          fx=image_xsize, fy=image_ysize, flags=0,
                          frame_type=0, is_last=True)
        if self.f("frame.all_default", br.bit()):
            return
        ft = self.f("frame.frame_type", FRAME_TYPE.get(br.bits(2), "?"))
        ft_i = {v: k for k, v in FRAME_TYPE.items()}.get(ft, 0)
        self.frame["frame_type"] = ft_i
        is_modular = self.f("frame.is_modular", br.bit())
        self.frame["is_modular"] = bool(is_modular)
        flags = self.f("frame.flags", br.u64())
        self.frame["flags"] = flags
        use_dc_frame = bool(flags & 0x20)  # kUseDcFrame
        color_transform = 2  # kXYB
        if not self.xyb_encoded:
            alt = self.f("frame.ycbcr", br.bit())
            color_transform = 1 if alt else 0  # kYCbCr : kNone
        if color_transform == 1 and not use_dc_frame:
            for c in range(3):
                self.f(f"frame.chroma_subsampling[{c}]", br.bits(2))
        if not use_dc_frame:
            up = self.f("frame.upsampling",
                        br.u32([("V", 1), ("V", 2), ("V", 4), ("V", 8)]))
            self.frame["upsampling"] = up
            for i in range(self.num_extra_channels):
                self.f(f"frame.ec_upsampling[{i}]",
                       br.u32([("V", 1), ("V", 2), ("V", 4), ("V", 8)]))
        if is_modular:
            self.frame["group_size_shift"] = self.f(
                "frame.group_size_shift", br.bits(2))
        if not is_modular and color_transform == 2:
            self.f("frame.x_qm_scale", br.bits(3))
            self.f("frame.b_qm_scale", br.bits(3))
        if ft_i != 2:  # not kReferenceOnly
            self.frame["num_passes"] = self.passes()
        if ft_i == 1:  # kDCFrame
            self.f("frame.dc_level",
                   br.u32([("V", 1), ("V", 2), ("V", 3), ("V", 4)]))
        is_partial = False
        if ft_i != 1:
            custom = self.f("frame.custom_size_or_origin", br.bit())
            self.frame["custom_size"] = bool(custom)
            if custom:
                if ft_i in (0, 3):
                    self.f("frame.origin.x0", unpack_signed(br.u32(CROP_DIST)))
                    self.f("frame.origin.y0", unpack_signed(br.u32(CROP_DIST)))
                fx = self.f("frame.size.xsize", br.u32(CROP_DIST))
                fy = self.f("frame.size.ysize", br.u32(CROP_DIST))
                self.frame["fx"], self.frame["fy"] = fx, fy
                is_partial = (fx < image_xsize or fy < image_ysize)
        if ft_i in (0, 3):
            mode = self.blending_info("frame.blending", self.num_extra_channels,
                                      is_partial)
            for i in range(self.num_extra_channels):
                self.blending_info(f"frame.ec_blending[{i}]",
                                   self.num_extra_channels, is_partial)
            if self.have_animation:
                self.f("frame.animation.duration", br.u32(
                    [("V", 0), ("V", 1), ("B", 8), ("B", 32)]))
            is_last = self.f("frame.is_last", br.bit())
            self.frame["is_last"] = bool(is_last)
        else:
            is_last = False
        if ft_i != 1 and not is_last:
            self.f("frame.save_as_reference",
                   br.u32([("V", 0), ("V", 1), ("V", 2), ("V", 3)]))
        if ft_i != 1:
            can_ref = (self.frame["frame_type"] != 0 or not is_last)
            if can_ref and mode == 0 and not is_partial and ft_i in (0, 3):
                self.f("frame.save_before_color_transform", br.bit())
            elif ft_i == 2:
                self.f("frame.save_before_color_transform", br.bit())
        self.name_string("frame")
        self.loop_filter(bool(is_modular))
        self.extensions("frame")

    # ── TOC ─────────────────────────────────────────────────────────────────
    def read_toc(self, image_xsize, image_ysize):
        br = self.br
        fr = self.frame
        up = fr["upsampling"]
        xs = div_ceil(fr["fx"], up)
        ys = div_ceil(fr["fy"], up)
        group_dim = 128 << fr["group_size_shift"]
        xsize_blocks = div_ceil(xs, 8)
        ysize_blocks = div_ceil(ys, 8)
        num_groups = div_ceil(xs, group_dim) * div_ceil(ys, group_dim)
        num_dc_groups = (div_ceil(xsize_blocks, group_dim)
                         * div_ceil(ysize_blocks, group_dim))
        num_passes = fr["num_passes"]
        if num_groups == 1 and num_passes == 1:
            n = 1
            labels = ["all-in-one"]
        else:
            n = 2 + num_dc_groups + num_groups * num_passes
            labels = ["LfGlobal"]
            labels += [f"LfGroup[{i}]" for i in range(num_dc_groups)]
            labels.append("HfGlobal")
            for p in range(num_passes):
                for g in range(num_groups):
                    labels.append(f"HfGroup[p{p} g{g}]")
        permuted = self.f("toc.permutation", br.bit())
        if permuted:
            self.note.append(
                "TOC carries an ANS-coded permutation; entry sizes below are "
                "in STORAGE order (the permutation itself is not decoded here)."
            )
            return None
        br.jump_to_byte_boundary()
        sizes = [br.u32(TOC_DIST) for _ in range(n)]
        br.jump_to_byte_boundary()
        self.toc = dict(entries=sizes, labels=labels, num_groups=num_groups,
                        num_dc_groups=num_dc_groups, num_passes=num_passes,
                        payload_start=br.pos // 8)
        # Self-consistency: header + TOC + every section size must account for
        # the whole file. If it does not, the parse is wrong somewhere upstream
        # and every number below it is fiction — say so loudly rather than
        # reporting a confident, wrong attribution.
        accounted = self.toc["payload_start"] + sum(sizes)
        if accounted != len(self.d):
            self.note.append(
                f"PARSE INCONSISTENT: header+TOC ({self.toc['payload_start']}) "
                f"+ sections ({sum(sizes)}) = {accounted} != file size "
                f"({len(self.d)}). Do not trust the section table.")
        self.lf_global_prefix(br.pos // 8, sizes[0] if labels[0] == "LfGlobal"
                              else None)
        self.hf_global_prefix()
        return self.toc

    # ── LfGlobal fixed prefix ───────────────────────────────────────────────
    def lf_global_prefix(self, start_byte, lf_global_bytes):
        """Parse LfGlobal up to where its modular sub-stream begins.

        Order (libjxl `dec_frame.cc::ProcessDCGlobal`): [patches] [splines]
        [noise] -> `DequantMatrices::DecodeDC` -> `Quantizer::Decode` ->
        `DecodeBlockCtxMap` -> `ColorCorrelation::DecodeDC` ->
        `ModularFrameDecoder::DecodeGlobalInfo`.

        Everything before the modular call is plain field coding; the modular
        stream is where the MA tree and the ANS histograms live. Splitting at
        that boundary turns "LfGlobal is +19 bytes" into "the global MA tree is
        +19 bytes", which is a different investigation. Bails (recording why)
        on any construct that needs an ANS decoder — patches, splines, noise,
        or a non-default block context map — rather than reporting a wrong
        offset."""
        br = self.br
        fr = getattr(self, "frame", {})
        flags = fr.get("flags", 0)
        if flags & 0x02 or flags & 0x10 or flags & 0x01:
            self.note.append("LfGlobal carries patches/splines/noise "
                             "(entropy-coded); modular-boundary split skipped")
            return
        if fr.get("is_modular"):
            return
        try:
            if self.f("lfglobal.dequant_dc.all_default", br.bit()) == 0:
                for c in range(3):
                    self.f(f"lfglobal.dequant_dc.quant[{c}]", br.f16())
            self.f("lfglobal.quantizer.global_scale",
                   br.u32([("O", (11, 1)), ("O", (11, 2049)), ("O", (12, 4097)),
                           ("O", (16, 8193))]))
            self.f("lfglobal.quantizer.quant_dc",
                   br.u32([("V", 16), ("O", (5, 1)), ("O", (8, 1)),
                           ("O", (16, 1))]))
            if self.f("lfglobal.block_ctx_map.is_default", br.bit()) == 0:
                # The thresholds are plain field coding; only the context map
                # that follows them is ANS. Report them — they say how the two
                # encoders SHAPE the block context map, which is a different
                # question from how many bytes the map costs.
                dct_dist = [("B", 4), ("O", (8, 16)), ("O", (16, 272)),
                            ("O", (32, 65808))]
                qf_dist = [("B", 2), ("O", (3, 4)), ("O", (5, 12)),
                           ("O", (8, 44))]
                for c in range(3):
                    n = self.f(f"lfglobal.block_ctx_map.dc_thresholds[{c}].n",
                               br.bits(4))
                    for k in range(n):
                        self.f(
                            f"lfglobal.block_ctx_map.dc_thresholds[{c}][{k}]",
                            unpack_signed(br.u32(dct_dist)))
                nqf = self.f("lfglobal.block_ctx_map.qf_thresholds.n",
                             br.bits(4))
                for k in range(nqf):
                    self.f(f"lfglobal.block_ctx_map.qf_thresholds[{k}]",
                           br.u32(qf_dist) + 1)
                # Relative to the section start, NOT the absolute file bit:
                # an absolute offset differs whenever anything upstream does,
                # which makes it a false-positive generator in a diff. This is
                # the LENGTH of LfGlobal's field-coded prefix, which is the
                # comparable quantity.
                self.f("lfglobal.block_ctx_map.ans_starts_at_bit",
                       br.pos - start_byte * 8)
                self.note.append("block_ctx_map context map is ANS-coded; "
                                 "LfGlobal modular-boundary split skipped "
                                 "(needs a DecodeContextMap port)")
                return
            if self.f("lfglobal.cmap_dc.all_default", br.bit()) == 0:
                self.f("lfglobal.cmap_dc.color_factor",
                       br.u32([("V", 84), ("V", 256), ("O", (8, 2)),
                               ("O", (16, 258))]))
                self.f("lfglobal.cmap_dc.base_correlation_x", br.f16())
                self.f("lfglobal.cmap_dc.base_correlation_b", br.f16())
                self.f("lfglobal.cmap_dc.ytox_dc", br.bits(8) - 128)
                self.f("lfglobal.cmap_dc.ytob_dc", br.bits(8) - 128)
        except EOFError:
            self.note.append("ran out of bytes inside LfGlobal prefix")
            return
        fixed_bits = br.pos - start_byte * 8
        self.f("lfglobal.fixed_prefix_bits", fixed_bits)
        if lf_global_bytes is not None:
            self.f("lfglobal.modular_stream_bytes",
                   lf_global_bytes - (fixed_bits + 7) // 8)

    # ── HfGlobal fixed prefix ───────────────────────────────────────────────
    def hf_global_prefix(self):
        """Parse HfGlobal up to where its first ANS-coded structure begins.

        Order (libjxl `dec_frame.cc::ProcessACGlobal`):
        `DequantMatrices::Decode` -> `num_histograms` -> per pass
        {`used_orders`, `DecodeCoeffOrders` (ANS), `DecodeHistograms` (ANS)}.

        The first two are plain field coding and the `used_orders` bitmask is a
        `U32Coder`, so three genuinely comparable numbers come out before any
        entropy decoding is needed: whether the AC quant matrices are default,
        how many histogram sets the frame uses, and WHICH of the 13 coefficient
        orders each pass customises. That last one is the informative one — two
        encoders can spend very different numbers of bytes here purely by
        choosing to customise more orders.
        """
        t = self.toc
        if not t or t["labels"][0] != "LfGlobal":
            return
        idx = 1 + t["num_dc_groups"]
        if t["labels"][idx] != "HfGlobal":
            return
        start = t["payload_start"] + sum(t["entries"][:idx])
        size = t["entries"][idx]
        if size == 0:
            return
        br = BitReader(self.d, start)
        saved, self.br = self.br, br
        try:
            if self.f("hfglobal.dequant_matrices.all_default", br.bit()) == 0:
                self.note.append("HfGlobal carries CUSTOM AC dequant matrices "
                                 "(17 QuantEncoding tables); prefix split "
                                 "stops here")
                return
            num_groups = t["num_groups"]
            # CeilLog2Nonzero(num_groups): 0 when num_groups == 1.
            nbits = max(0, (num_groups - 1).bit_length())
            self.f("hfglobal.num_histograms", br.bits(nbits) + 1)
            for p in range(t["num_passes"]):
                used = br.u32([("V", 0x5F), ("V", 0x13), ("V", 0),
                               ("B", 13)])
                self.f(f"hfglobal.used_orders[p{p}]",
                       f"0x{used:04x} ({bin(used).count('1')} of 13 custom)")
                self.f(f"hfglobal.ans_starts_at_bit[p{p}]", br.pos - start * 8)
                break  # later passes sit behind this pass's ANS data
        except EOFError:
            self.note.append("ran out of bytes inside HfGlobal prefix")
        finally:
            self.br = saved

    # ── driver ──────────────────────────────────────────────────────────────
    def parse(self):
        d = self.d
        if len(d) < 2:
            raise ValueError("file too short")
        if d[0] == 0 and d[1] == 0 and d[2] == 0 and d[3] == 0x0C:
            raise ValueError(
                "ISOBMFF container (starts with a JXL box) — this tool reads "
                "bare codestreams; re-encode without --container, or strip the "
                "boxes first")
        if not (d[0] == 0xFF and d[1] == 0x0A):
            raise ValueError(f"not a bare JXL codestream (magic {d[0]:02x}{d[1]:02x})")
        self.br = BitReader(d, 2)
        xs, ys = self.size_header()
        self.image_metadata()
        self.custom_transform_data()
        if getattr(self, "want_icc", False):
            self.note.append("ICC blob present — not parsed (field trace stops "
                             "at the ICC boundary)")
            return
        # The codestream headers are zero-padded to a byte boundary before the
        # first frame (libjxl `encode.cc:828 writer.ZeroPadToByte()`, decoder
        # side `decode.cc:1133 reader->JumpToByteBoundary()`). Missing this is
        # the single easiest way to derail a hand-written JXL parser: every
        # frame-header field still *decodes*, just shifted, so it produces
        # plausible-looking garbage (a 32x32 image reporting num_passes=2 and a
        # custom crop origin) rather than an error.
        self.br.jump_to_byte_boundary()
        self.f("<headers zero-padded to byte boundary>", self.br.pos // 8)
        self.frame_header(xs, ys)
        self.read_toc(xs, ys)
        return self


def unpack_signed(u):
    return (u >> 1) ^ -(u & 1)


def parse_file(path):
    with open(path, "rb") as fh:
        data = fh.read()
    p = Parser(data)
    p.parse()
    return p, data


# ── reporting ───────────────────────────────────────────────────────────────


def cmd_trace(path):
    p, data = parse_file(path)
    print(f"=== {path} ({len(data)} bytes) ===")
    for pos, name, val in p.fields:
        print(f"  bit{pos:6d} (byte {pos // 8:5d}.{pos % 8})  {name:44s} = {val}")
    if p.toc:
        print_toc(p, len(data))
    for n in p.note:
        print(f"  NOTE: {n}")
    return 0


def print_toc(p, total_len, other=None):
    t = p.toc
    print(f"  --- TOC: {len(t['entries'])} entries "
          f"({t['num_dc_groups']} dc-groups, {t['num_groups']} groups, "
          f"{t['num_passes']} passes), payload starts at byte {t['payload_start']}")
    ot = other.toc if other else None
    hdr = f"    {'section':22s} {'bytes':>9s}"
    if ot:
        hdr += f" {'other':>9s} {'delta':>9s}"
    print(hdr)
    for i, (lab, sz) in enumerate(zip(t["labels"], t["entries"])):
        line = f"    {lab:22s} {sz:9d}"
        if ot and i < len(ot["entries"]):
            o = ot["entries"][i]
            line += f" {o:9d} {sz - o:+9d}"
        print(line)
    tot = sum(t["entries"])
    line = f"    {'TOTAL(sections)':22s} {tot:9d}"
    if ot:
        otot = sum(ot["entries"])
        line += f" {otot:9d} {tot - otot:+9d}"
    print(line)
    line = f"    {'header+TOC':22s} {t['payload_start']:9d}"
    if ot:
        line += f" {ot['payload_start']:9d} {t['payload_start'] - ot['payload_start']:+9d}"
    print(line)


def cmd_toc(paths):
    for path in paths:
        p, data = parse_file(path)
        print(f"=== {path} ({len(data)} bytes) ===")
        if p.toc:
            print_toc(p, len(data))
        else:
            print("  (no TOC parsed)")
            for n in p.note:
                print(f"  NOTE: {n}")
    return 0


def align(fa, fb):
    """Align two field sequences by name, allowing insertions on either side.
    Returns rows of (name_a, val_a, name_b, val_b, status)."""
    rows = []
    i = j = 0
    na = [f[1] for f in fa]
    nb = [f[1] for f in fb]
    while i < len(fa) or j < len(fb):
        if i < len(fa) and j < len(fb) and na[i] == nb[j]:
            status = "=" if fa[i][2] == fb[j][2] else "!"
            rows.append((na[i], fa[i][2], nb[j], fb[j][2], status))
            i += 1
            j += 1
            continue
        # look ahead for a resync point
        ra = nb[j:j + 24].index(na[i]) if i < len(fa) and na[i] in nb[j:j + 24] else None
        rb = na[i:i + 24].index(nb[j]) if j < len(fb) and nb[j] in na[i:i + 24] else None
        if ra is not None and (rb is None or ra <= rb):
            for k in range(ra):
                rows.append((None, None, nb[j + k], fb[j + k][2], "B-only"))
            j += ra
        elif rb is not None:
            for k in range(rb):
                rows.append((na[i + k], fa[i + k][2], None, None, "A-only"))
            i += rb
        else:
            if i < len(fa):
                rows.append((na[i], fa[i][2], None, None, "A-only"))
                i += 1
            if j < len(fb):
                rows.append((None, None, nb[j], fb[j][2], "B-only"))
                j += 1
    return rows


def cmd_diff(pa, pb):
    a, da = parse_file(pa)
    b, db = parse_file(pb)
    print(f"A = {pa} ({len(da)} bytes)")
    print(f"B = {pb} ({len(db)} bytes)   delta {len(db) - len(da):+d} bytes")
    print()
    rows = align(a.fields, b.fields)
    ndiff = sum(1 for r in rows if r[4] != "=")
    first = None
    print("  status  field                                        A                    B")
    for name_a, va, name_b, vb, st in rows:
        if st == "=":
            continue
        if first is None:
            first = (name_a or name_b)
        nm = name_a or name_b
        print(f"  {st:6s}  {nm:44s} {str(va):20s} {str(vb):20s}")
    if ndiff == 0:
        print("  (envelope fields identical)")
    print()
    if first:
        print(f"  FIRST DIVERGING FIELD: {first}")
    if a.toc and b.toc:
        print()
        print("  Per-section byte table (A vs B):")
        print_toc(a, len(da), other=b)
    else:
        for n in a.note + b.note:
            print(f"  NOTE: {n}")
    return 1 if (ndiff or (a.toc and b.toc and a.toc["entries"] != b.toc["entries"])) else 0


def cmd_tsv(pa, pb):
    a, da = parse_file(pa)
    b, db = parse_file(pb)
    print("kind\tname\ta\tb")
    print(f"size\tfile_bytes\t{len(da)}\t{len(db)}")
    for name_a, va, name_b, vb, st in align(a.fields, b.fields):
        if st == "=":
            continue
        print(f"field\t{name_a or name_b}\t{va}\t{vb}")
    if a.toc and b.toc:
        for i, lab in enumerate(a.toc["labels"]):
            sa = a.toc["entries"][i]
            sb = b.toc["entries"][i] if i < len(b.toc["entries"]) else ""
            print(f"section\t{lab}\t{sa}\t{sb}")
        print(f"section\theader_and_toc\t{a.toc['payload_start']}\t{b.toc['payload_start']}")
    return 0


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd = argv[1]
    try:
        if cmd == "trace":
            return cmd_trace(argv[2])
        if cmd == "toc":
            return cmd_toc(argv[2:])
        if cmd == "diff":
            return cmd_diff(argv[2], argv[3])
        if cmd == "tsv":
            return cmd_tsv(argv[2], argv[3])
    except (ValueError, EOFError, KeyError, IndexError) as e:
        print(f"PARSE FAILURE: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
