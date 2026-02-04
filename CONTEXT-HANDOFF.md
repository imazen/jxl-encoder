# Context Handoff — Tree Learning Roundtrip Tests

**Date**: 2026-02-03
**Task**: Implementing content-adaptive MA tree learning for modular encoding (plan: `fizzy-doodling-marble.md`)

## Overall Progress

Tasks 1-5 are COMPLETE and committed. Task 6 (roundtrip tests) is IN PROGRESS.

| Step | Status | Commit |
|------|--------|--------|
| 1. Fix BFS child ordering in collect_tree_tokens | DONE | `74d0a74` |
| 2. Create tree_learn.rs (TreeSamples, gather_samples) | DONE | `73b5665` |
| 3. Implement compute_best_tree | DONE | `73b5665` |
| 4. Add write_tree + multi-context encoding | DONE | `0584a4c` |
| 5. Wire --tree-learning through CLI/FrameEncoder | DONE | `6b848a0` |
| 6. Roundtrip tests | **IN PROGRESS** | uncommitted |

## Current Test Results

**4 of 6 tree learning tests PASS, 2 FAIL** (run with `--release`):

| Test | Result | Tree | Notes |
|------|--------|------|-------|
| gray_constant_8x8 | PASS | 1 leaf | Single-context path |
| gray_gradient_8x8 | PASS | 1 leaf | Single-context path |
| gray_32x32 | PASS | Multi-leaf | After BFS context fix |
| rgb_checkerboard_8x8 | PASS | 13 nodes, 7 leaves, 3 histos | Multi-context path works |
| **rgb_gradient_128x128** | **FAIL** | 171 nodes, 86 leaves, 4 histos | SectionTooShort |
| **rgb_multigroup_300x300** | **FAIL** | Large tree | SectionTooShort (98 bytes for 300x300 RGB!) |

Run tests: `cargo test -p jxl_enc --lib --release "tree_learning_tests" -- --nocapture`

## Bugs Found and Fixed (Uncommitted)

### Bug 1: Double HybridUint Encoding (tree_learn.rs)
`collect_residuals_with_tree` applied HybridUint{4,2,0}.encode() to packed residuals before storing in AnsToken. But `build_entropy_code_ans` and `write_tokens_ans` ALSO apply UintCoder::encode (same HybridUint{4,2,0}) to Token.value. Double-encoded.
**Fix**: Store raw packed residuals in AnsToken.value (matching non-tree path in section.rs:407).

### Bug 2: Context Map Written for num_contexts=1 (improved.rs, section.rs)
`write_entropy_code_ans` always writes a context map. JXL spec: context map omitted when num_dist <= 1. jxl-rs decoder line 512: `if num_contexts > 1 { decode_context_map() }`.
**Fix**: Branch on num_contexts: multi-context uses `write_entropy_code_ans`; single-context uses `write_ans_modular_header` (in section.rs, made pub(super)).

### Bug 3: Double lz77.enabled=0 (improved.rs, section.rs)
`write_modular_stream_with_tree` wrote lz77=0, then `write_ans_modular_header` wrote lz77=0 again. `write_entropy_code_ans` does NOT write lz77 (caller handles it).
**Fix**: lz77=0 only in multi-context branch; single-context branch lets `write_ans_modular_header` handle it.

### Bug 4: BFS Context Ordering (tree.rs)
`assign_sequential_contexts` iterated nodes in storage order (0, 1, 2, ...), but the decoder assigns context IDs in BFS order (rchild first, then lchild). For single-leaf trees this doesn't matter; for multi-leaf trees context IDs were completely scrambled.
**Fix**: Changed `assign_sequential_contexts` to BFS traversal with same child ordering as `collect_tree_tokens` (rchild first, lchild second).

## Active Bug: SectionTooShort on Large Trees

The rgb_gradient_128x128 produces 171 tree nodes, 86 leaf contexts, clustered to 4 histograms.
Debug output for the failing 128x128:
```
[bit 2] after dc_quant + has_tree
[bit 1966] after write_tree (171 nodes)
[bit 1967] after lz77.enabled=0
[bit 2497] after write_entropy_code_ans (86 contexts, 4 histograms)
[bit 2510] after GroupHeader
[bit 4046] after write_tokens_ans (49152 tokens)
```
File is 517 bytes. Both jxl-rs and djxl reject it.

The multigroup 300x300 is only 98 bytes for a 300x300 RGB image — suspiciously tiny, suggesting the multigroup tree learning path barely encodes anything.

### Possible Causes (Not Yet Investigated)

1. **write_tree Huffman encoding for large alphabets**: Tree has ~510 distinct symbol values (pack_signed of split values up to 255). Huffman table for 510 symbols may be incorrectly encoded.

2. **Tree token count mismatch**: If `collect_tree_tokens` produces different tokens than decoder expects, decoder reads past tree boundary.

3. **Property index mapping**: Tree uses properties [0, 3, 4, 5, 6, 7] (from CANDIDATE_PROPERTIES). The decoder's property numbering must match exactly. Our properties are: channel(0), group_id(1), y(2), x(3), |N-NW|(4), |N-W|(5), N(6), FloorLog2(|N|)(7), N-NE(8), |N-NN|(9), W-WW(10), W-NW(11), NW-NNW(12), N-NW(13), N+NE-2NN(14), WP max error(15).

4. **Context map encoding**: 86 contexts with 4 histograms uses simple context map (nbits=2, 86 entries). Verify decoder reads exactly 86 entries.

5. **Multigroup-specific**: 300x300 at 98 bytes suggests `write_global_modular_section_with_tree` or `write_group_modular_section` with AnsWithTree variant has an encoding issue.

### Suggested Next Steps

1. Create a minimal failing test: handcraft a 3-leaf tree on a 128x128 image. If it fails, the issue is in multi-context data path. If it passes, the issue is in large tree serialization.

2. Verify tree serialization for the passing 13-node checkerboard vs the failing 171-node gradient. Specifically check `write_tree` -> `build_and_store_huffman_tree` for large alphabets.

3. Reduce `max_nodes` parameter (default 256) to something small like 8 to force a small tree on the 128x128 test and see if that passes.

4. Check the `write_tree` token encoding — verify that the 6 tree contexts are handled correctly. The tree uses single-histogram (all 6 contexts map to histogram 0), but if the decoder expects different formatting for multi-context tree encoding, that would explain the issue.

## Dirty Files (Uncommitted)

```
jxl_enc/src/modular/tree.rs        — BFS assign_sequential_contexts
jxl_enc/src/modular/tree_learn.rs  — Remove double HybridUint encoding
jxl_enc/src/modular/improved.rs    — Context map branch + lz77 fix + debug prints
jxl_enc/src/modular/section.rs     — pub(super) write_ans_modular_header + context map/lz77 fix
jxl_enc/src/encoder_tests.rs       — 6 tree learning roundtrip tests
```

All changes are working-tree modifications on top of commit `6b848a0`.

## Key Code Paths

### Single-group tree learning
`improved.rs::write_modular_stream_with_tree()` (line ~1666)
- gather_samples -> compute_best_tree -> collect_residuals_with_tree -> build_entropy_code_ans
- write dc_quant + has_tree
- write_tree (tree's own EntropyCode with Huffman, line 1574)
- write data EntropyCode (lz77 + context_map + ANS distributions)
- write GroupHeader
- write_tokens_ans

### Multi-group tree learning
`section.rs::write_global_modular_section_with_tree()` (line ~253)
- Same but gathers samples from ALL groups, writes tree + distributions globally
- Each group: `write_group_modular_section()` with `GlobalModularState::AnsWithTree`

### Tree serialization
`tree.rs::collect_tree_tokens()` (line 252) — BFS with rchild first, lchild second
`improved.rs::write_tree()` (line 1574) — Huffman prefix code, raw symbols (split_exp=15)

### Context assignment
`tree.rs::assign_sequential_contexts()` (line 381) — NOW uses BFS order matching collect_tree_tokens

## Debug Prints

`write_modular_stream_with_tree` in improved.rs currently has `eprintln!` statements showing tree structure, bit positions, and context/histogram info. These should be removed or gated behind a feature flag before final commit.
