# LZ77 bucket matcher: unconditional adoption rejected

All five bucket capacities grow the 256×256 terminal screenshot's lossless
float output from 22,072 to 29,702 bytes (+34.57%). This rules out adopting
the tested matcher unconditionally. It does not rule out a future selection
policy evaluated against actual coded size.

The candidate and its decoder checks are preserved on
`abandoned/issue110-bucket-default` (`deac9c61e897`); production retains the hash-chain matcher.
The candidate changes the existing Greedy dispatcher, without a new strategy
or runtime override. That dispatcher also serves strict VarDCT tree coding
and ICC, so a future shipping implementation must narrow its wiring.

## Size screen

Four real sources, 256×256 center crops, lossless effort 8; u8, widened u16
and derived linear f32 input. Every u8/u16 cell is SHA256-identical to the
hash-chain baseline. Float results are:

| Source | Chain | Bucket 1 | Bucket 3 | Bucket 7 | Bucket 15 | Bucket 31 |
|---|---:|---:|---:|---:|---:|---:|
| codec_wiki | 10,797 | 10,797 | 10,797 | 10,797 | 10,797 | 10,797 |
| frymire | 60,333 | 56,374 | 54,222 | 53,194 | 52,234 | 51,827 |
| path-lossless | 315,516 | 343,163 | 323,442 | 313,536 | 311,105 | 310,160 |
| terminal | 22,072 | 29,702 | 29,702 | 29,702 | 29,702 | 29,702 |

These are bytes, not performance measurements. Arms ran in blocks, one
encode per cell; their recorded times are unsuitable for a wall-time verdict.
This is an early rejection screen, not the four-size/content acceptance grid
needed to adopt a new default. The wider calibration was not run because
every capacity already fails this screen. These floats derive from 8-bit
sources (`(u16(sample) << 8) / 65535`), not native HDR photography.

All 72 JXL artifacts have verified sizes and SHA256s. All 24 float outputs
reconstruct source samples bit-for-bit in both jxl-rs and djxl v0.12; the size
loss is not corruption. Matcher unit tests cover overlapping references,
literal contexts, ring eviction, distance-symbol ties, the window cutoff,
short tails and the 1024-token match bound.

An additional [48-cell shape check](lz77_bucket_shapes_2026-09-24.tsv) compares
the chain and bucket-3 on the same sources at 64×64 and multi-group 259×259.
All sixteen float outputs also reconstruct bit-exactly in both decoders.
The candidate passes 74/75 unchanged lock/drift tests. The e8 tiled RGB
512×512 lock fails: 72,849 → 73,177 bytes. No lock was regenerated.
All five strict byte-lock tests pass, which does not prove that replacing
their shared Greedy dispatcher preserves every strict input.

## Implementation and reproduction

The design follows libjxl's post-v0.12
[e8ff0976 rewrite](https://github.com/libjxl/libjxl/commit/e8ff09762481785938d8e4e01333ed3917571161):
14-bit Murmur hashing, ring buckets, longest match with distance-symbol tie
breaking, no lazy match and no per-match cost rejection. This port retains
the existing encoder's 1 Mi-token window, 1024-token match cap, and smallest
special-distance symbol for narrow images. It borrows token values and uses
a bounded special-distance array instead of a width-sized allocation.
It is not a claim of full upstream byte parity.

Baseline: `3ad5d9bb`, release `lz77_hash_ab` example, default Cargo features.
For each candidate, change only `bucket::apply::<N>` in
`entropy_coding/lz77.rs`, with N in 1/3/7/15/31, and rebuild the same example.
The archived selection is 3. The final bucket-3 rebuild reproduces all twelve
original bucket-3 hashes.

```
nice -n 19 cargo build --locked -p jxl-encoder --release --example lz77_hash_ab -j 4
nice -n 19 <binary> <input-dir> <out.tsv> --images 4 --size 256 --efforts 8 --lossy 0 --arm bucket3
just lz77-bucket-decode-check <screen-root> chain,bucket1,bucket3,bucket7,bucket15,bucket31
```

The bucket verification recipes exist on the archive commit.
Set `TMPDIR=$HOME/tmp`, `CARGO_BUILD_JOBS=4`, `RAYON_NUM_THREADS=4`.
Sources: committed `jxl-encoder/tests/images/frymire-srgb.png`, codec-corpus
`gb82/path-lossless.png`, `gb82-sc/{codec_wiki,terminal}.png`.
The [TSV](lz77_bucket_screen_2026-09-24.tsv) records source and output hashes;
artifact paths are relative to
`/Users/lilith/tmp/jxl-backlog/lz77-bucket-screen/`. Raw arm tables and logs
remain there. No artifact bytes are committed to git.
