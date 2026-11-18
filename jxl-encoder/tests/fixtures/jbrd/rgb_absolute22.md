# Absolute RGB sampling refusal source

The text fixture is a 16x8 RGB JPEG with equal 2x2 SOF factors, one unit
quantization table and three single-component baseline scans. Each visible
block contains DC0/EOB. Separate scans avoid a twelve-block interleaved MCU.
`unusual::rgb_absolute22` binds the explicit builder to these exact bytes.

Source SHA256: `04ec49434cd03097620a388680ae46a4f8a8ab4e01c0f83ad10500e2fe2fe1aa`.
Independent decoder: `libjpeg-turbo version 2.1.5 (build 20260406)`.
Decoder SHA256: `981d101c298da97f9c8844b8387ba1c0950508353566dd034361d364b87ed296`.
Every selected unrelated-APP0 mutation preserves its complete RGB8 source
samples. Pinned cjxl v0.12 returns an encoder error for the original; the
encoder's shared typed refusal is tested across raw/parsed/bare/recompression/
embedded-gain-map and actual CLI paths. This is a source/admission control,
not a claim of JXL pixel output for unsupported RGB sampling.
