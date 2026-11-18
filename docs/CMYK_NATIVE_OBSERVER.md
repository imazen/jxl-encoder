# Native CMYK sample and layer observer

The issue-144 tests use the public libjxl v0.12 decoder API independently
of the Rust decoder. The helper refuses other native decoder versions.

The source-injected CMake target is jxl-cmyk-frames. CI builds it alongside
the original-profile observer on each native platform and exports
CMYK_RAW_FRAME_ORACLE. Existing checksum-pinned CLI controls remain selected
on Linux x64 and Windows x64.

The helper takes an input JXL, output prefix and mode:

| Mode | Observed color samples |
|---|---|
| linear | Native linear-sRGB output, requiring full-canvas Replace |
| raw-cmy | Stored normalized CMY, requiring full-canvas Replace |
| linear-layers | Native linear-sRGB layers plus crop/reference fields |
| raw-cmy-layers | Stored CMY layers plus crop/reference fields |
| original | Requests the native original profile, for output-selection diagnostics |

The raw modes keep the native ICC field parser and substitute a CMS callback
that copies its CMY input samples. The callback must actually execute.
This is a raw-sample observer, not a physical color transform. Physical
checks use the linear modes and separate LCMS source controls.

Black and optional alpha are decoded into separate float planes. The helper
finds Black by its declared extra-channel type, requires exactly one Black,
and writes explicit little-endian PFM samples. Layer modes support Replace
and Add. Their header table records duration, crop, both blend modes,
both source slots and the saved reference slot. Tests compose those decoded
layers independently; the helper does not silently enable native coalescing.

The native CLI's inability to coalesce selected cropped CMYK/ICC files does
not prove an invalid codestream. A completed layer decode and independent
composition are required before claiming complete displayed pixels. The
original-profile mode is diagnostic and is not assumed to succeed for CMYK.

Local source-build controls pass with both LCMS and skcms v0.12 libraries.
Native-platform CI remains a separate verdict.
