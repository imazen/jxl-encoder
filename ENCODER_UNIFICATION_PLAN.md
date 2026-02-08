# Encoder Unification Plan

## Problem

We have two complete encoder implementations that share only BitWriter and error handling:

| | **Old Path (Modular/Lossless)** | **Tiny Path (VarDCT/Lossy)** |
|---|---|---|
| Struct | `Encoder` (encoder.rs, 301 lines) | `TinyEncoder` (tiny/encoder.rs, 1623 lines) |
| File header | `FileHeader::write()` (headers/file_header.rs) | `TinyEncoder::write_file_header()` (tiny/bitstream.rs) |
| Frame header | `FrameEncoder::write_frame_header_with_extra_channels()` | `write_frame_header()` (tiny/frame.rs) |
| Frame assembly | `FrameEncoder::encode_modular()` (frame/frame_encoder.rs) | Inline in bitstream.rs (two-pass + streaming) |
| Entropy coding | `entropy_coding/*` | `tiny/entropy_code.rs` (reimplemented) |
| ICC wiring | `encoder.icc_profile` → FileHeader | `tiny.icc_profile` → write_file_header |

Every feature (ICC, 16-bit, alpha, EXIF) must be wired twice. libjxl has ONE `EncodeFrame` with `if (VarDCT)` branches where modes diverge.

## Target Architecture

One encoder, one file header path, one frame assembly. `ModularFrameEncoder` is a component used by both VarDCT (for DC/metadata streams) and pure-modular paths — matching libjxl.

```
EncodeRequest::encode()
  → write_file_header()          // ONE path, parameterized
  → write_icc() if needed        // ONE path
  → zero_pad_to_byte()
  → if VarDCT:
      TinyEncoder pipeline (XYB, quant, DCT, strategies, butteraugli)
      write_frame(VarDCT, ...)   // shared frame assembly
    else:
      ModularImage pipeline (RCT, tree learning, squeeze, palette)
      write_frame(Modular, ...)  // shared frame assembly
```

## Phases

### Phase 1: Unified File Header (Low Risk)

**Goal:** One `write_file_header()` used by both paths.

The two implementations write nearly identical bitstreams. Differences:
- TinyEncoder always writes `xyb_encoded=1`, old path writes `xyb_encoded=0`
- TinyEncoder hardcodes RGB color space; old path supports Gray
- Both now handle ICC, extra channels, bit depth

**Steps:**

1. **Extend `FileHeader` to cover all TinyEncoder needs**
   - Already has: width/height, bit_depth, color_encoding, extra_channels, xyb_encoded
   - Add: nothing needed, it's already parameterized

2. **Make TinyEncoder build a `FileHeader` and call `FileHeader::write()`**
   - In `tiny/encoder.rs`, construct `FileHeader` from TinyEncoder fields
   - Replace `self.write_file_header(w, h, has_alpha, writer)` with `file_header.write(writer)`
   - Move ICC writing to after `FileHeader::write()` (both paths already do this)

3. **Delete `TinyEncoder::write_file_header()` and `write_size()` from bitstream.rs**
   - ~80 lines removed

4. **Verify**: All 655 tests pass. Byte-exact output for non-ICC files (header bits must not change).

**Files touched:**
- `tiny/encoder.rs` — build FileHeader, call its write()
- `tiny/bitstream.rs` — delete write_file_header(), write_size()
- `headers/file_header.rs` — possibly minor adjustments for xyb_encoded + ICC integration
- Tests — verify no regression

**Risk:** Low. Pure refactor, no functionality change. The two headers already write compatible bitstreams (both produce valid JXL).

### Phase 2: Unified Frame Header (Medium Risk)

**Goal:** One `write_frame_header()` used by both paths.

Frame headers differ more than file headers:
- VarDCT frames: `encoding=VarDCT`, gaborish flags, noise params, quant scales, epf_iters
- Modular frames: `encoding=Modular`, simpler header

**Steps:**

1. **Create `FrameConfig` struct** with all frame header parameters
   ```rust
   pub struct FrameConfig {
       pub encoding: FrameEncoding,  // VarDCT or Modular
       pub width: u32,
       pub height: u32,
       pub num_extra_channels: u32,
       pub gaborish: bool,
       pub epf_iters: u32,
       pub noise_params: Option<NoiseParams>,
       pub quant_scales: Option<QuantScales>,
       // ... other VarDCT-specific fields with sensible defaults
   }
   ```

2. **Write `FrameConfig::write()` that handles both modes**
   - Most fields are shared (dimensions, TOC, extra channels)
   - VarDCT-only fields gated on `encoding == VarDCT`

3. **Update both paths to build FrameConfig and call write()**

4. **Delete duplicate frame header writers**

**Files touched:**
- New or extended: `headers/frame_header.rs` or `tiny/frame.rs`
- `tiny/bitstream.rs` — use new frame header
- `frame/frame_encoder.rs` — use new frame header
- Tests — verify both paths still decode

**Risk:** Medium. Frame headers are complex and any bit-level mismatch breaks decoding. Need careful verification.

### Phase 3: Unified Frame Assembly / TOC (Medium Risk)

**Goal:** One frame assembly pipeline that handles both VarDCT and Modular section layouts.

Both paths need:
- Frame header
- TOC (table of contents with section sizes)
- DC groups (VarDCT: DC + AC metadata; Modular: pixel data groups)
- AC groups (VarDCT only)

**Steps:**

1. **Abstract section writing** — each section is a closure that writes to a BitWriter
2. **Shared TOC computation** — count sections, write TOC, then write sections
3. **VarDCT provides:** DC group writer, AC group writer
4. **Modular provides:** group writer (single type)

This is where the two-pass vs streaming distinction lives. TinyEncoder's two-pass path collects all tokens first, builds optimal codes, then writes. The modular path does something similar via FrameEncoder.

**Files touched:**
- `tiny/bitstream.rs` — extract frame assembly into shared code
- `frame/frame_encoder.rs` — use shared frame assembly
- Possibly new `frame/assembly.rs`

### Phase 4: Merge Encoder Structs (High Risk)

**Goal:** One `Encoder` struct that can produce both VarDCT and Modular frames.

**Steps:**

1. **Move TinyEncoder fields into a unified Encoder**
   - VarDCT-specific fields only populated when distance > 0
   - Modular-specific fields (tree_learning, squeeze, palette) present for both (used by VarDCT for DC/alpha too)

2. **One `encode()` entry point**
   ```rust
   impl Encoder {
       pub fn encode(&self, pixels: &[u8], width: u32, height: u32, layout: PixelLayout) -> Result<Vec<u8>> {
           let file_header = self.build_file_header(width, height, layout);
           file_header.write(&mut writer)?;
           self.write_icc(&mut writer)?;
           writer.zero_pad_to_byte();

           if self.distance > 0.0 {
               self.encode_vardct_frame(pixels, layout, &mut writer)?;
           } else {
               self.encode_modular_frame(pixels, layout, &mut writer)?;
           }
           Ok(writer.finish())
       }
   }
   ```

3. **Update API layer** — `encode_lossless()` and `encode_lossy()` both create the same Encoder type with different config

**Risk:** High. This is a massive refactor touching every test. Do only after Phases 1-3 prove the shared infrastructure works.

### Phase 5: Unified Entropy Coding (Deferred)

**NOT recommended for now.** The two entropy coding implementations have:
- Different context models (VarDCT block contexts vs modular pixel contexts)
- Different optimization strategies
- Different performance characteristics

The `tiny/entropy_code.rs` implementation is battle-tested for VarDCT. The `entropy_coding/*` modules serve modular well. Forcing them together risks performance regression in both paths.

**When to revisit:** After Phase 4 is stable and benchmarked, if maintaining two entropy paths becomes a pain point.

## Priority Order

| Phase | Risk | Effort | Value | Do When |
|-------|------|--------|-------|---------|
| 1: File Header | Low | 1 day | High (kills biggest duplication) | Now |
| 2: Frame Header | Medium | 1-2 days | Medium | After Phase 1 |
| 3: Frame Assembly | Medium | 2-3 days | Medium | After Phase 2 |
| 4: Merge Structs | High | 3-5 days | High (single encoder) | After Phase 3 |
| 5: Entropy | High | 1 week+ | Low (working fine separate) | Maybe never |

## Verification at Each Phase

1. All 655+ existing tests pass
2. Byte-exact output for files without new features (hash lock tests)
3. djxl, jxl-oxide, jxl-rs all decode output
4. RD regression test shows no quality/size changes
5. No performance regression (benchmark encode speed)

## Non-Goals

- **Don't merge LossyConfig/LosslessConfig yet.** The separate configs are good API design (prevents invalid state at compile time). The internal unification is orthogonal to the public API shape.
- **Don't unify pixel format handling yet.** TinyEncoder taking linear f32 and old Encoder taking ModularImage serve different needs. The conversion happens in the API layer and should stay there.
- **Don't touch the modular encoding pipeline.** `modular/improved/*` is working well. VarDCT uses parts of it for DC/alpha. Keep it as a component, not merged into VarDCT code.
