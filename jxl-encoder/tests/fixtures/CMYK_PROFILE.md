# CMYK fixture

`ps_cmyk.hex` contains the 5,340-byte Artifex PS CMYK Profile from Ghostscript,
with original description and copyright tags intact. SHA256 of decoded bytes:
`020236a6458406743848a12c6be280e54bcd0fd8e2d4bea9c4899410cef98cb2`.

Copyright Artifex Software 2010. Distributed under AGPL-3.0-or-later,
as part of Ghostscript's `iccprofiles` sources. This third-party test fixture
retains that license; it is not covered by the encoder's commercial licensing
offer. [Ghostscript sources](https://github.com/ArtifexSoftware/ghostpdl/tree/master/iccprofiles).

Native LCMS controls use relative-colorimetric conversion to linear sRGB,
without optimization or pixel caching. JXL ink values are inverted and scaled
to LCMS's 0–100 convention. The private source/Black tests separately check
stored sample codes and physical display colors.

`cmyk_lcms_legacy.hex` freezes the 1,024-pixel legacy gradient's LCMS 2.17
linear RGB output as three binary32 bit patterns per line. Generator:
`scripts/cmyk_source_oracle.cc`; profile identity is above. It covers the
approved replacement of the inverse naive-ink oracle.
SHA256: `18228f940d18f706f00784b8cf86e58a4ff5864c017bf68211016d95e72b19b2`.

cmyk_lcms_swatches.hex freezes the six source-oracle swatches in row order:
white, cyan, magenta, yellow, black and CMYK [64,128,192,224]. Values are
LCMS 2.17 linear RGB binary32 bit patterns, from the same pinned profile and
scripts/cmyk_source_oracle.cc. SHA256: f5362c26d627b106943c5194fc6596c058710f95d2bc861ce7babb4255d964ff.
