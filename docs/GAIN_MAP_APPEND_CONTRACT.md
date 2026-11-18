# Appending one gain-map box

Part 2 clause 9.11 permits at most one `jhgm` box. The owner selected rejection
of an existing map, including its `brob` representation, and approved a
fallible low-level helper through a PR assigned to lilith.

The approved source-breaking signature is:

```rust,ignore
pub fn append_gain_map_box(
    jxl_data: &[u8],
    jhgm_payload: &[u8],
) -> Result<Vec<u8>, EncodeError>;
```

Callers must propagate or handle the result, for example
`let output = append_gain_map_box(input, payload)?;`. The root re-export has
the same signature. `append_gain_map_bundle` retains its existing Result
signature and propagates serialization and append errors.

Bare codestreams retain their wrapping behavior. Existing containers are
checked for valid top-level box boundaries and direct or compressed gain-map
presence before insertion. Duplicate maps and malformed boundaries return
InvalidInput; inputs are borrowed and remain unchanged. Valid ordinary
compressed metadata still permits the first append.

This is not complete container, codestream or bundle-content validation. The
raw payload is opaque. Unrelated box contents and ICC contents are not parsed.
The typed helper retains its existing serializer checks. No new requirement
to decompress `brob` data is introduced: its four-byte inner type identifies
the box. No strategy gate changes.

The retained original-source before run accepts second maps in raw and typed
helpers for direct, compressed, extended and to-end box forms. Current controls
exercise both base encoders and every offered policy, require the specific
duplicate-map error, retain existing file bytes, and fully render the existing
primary through jxl-rs and native v0.12. First-append controls retain exact
payload bytes and unchanged primary pixels. Boundary controls include truncated
ordinary/extended headers, undersized lengths and oversized lengths. Typed
serializer errors have a separate propagation control.

The approved signature is merged through [PR145](https://github.com/imazen/jxl-encoder/pull/145)
at `699ad7cd`. No crate publication is authorized by these checks.

Exact-source LLVM evidence (`gain-map-final/regions/regions.json`) observes
29/29 regions in the raw helper and 9/9 in the typed forwarding helper.
The existing top-level box parser observes 53/57 regions. Three remaining
error continuations convert slices to fixed arrays immediately after obtaining
exactly sized slices and cannot fail. The fourth is the u64-to-usize overflow
refusal, unavailable on this 64-bit host. The stable unit control includes
u64::MAX for the required i686 CI route; a local 32-bit coverage verdict is
not claimed.

Restored direct, compressed, boundary-check and typed-error-propagation faults
each fail at runtime, with source checksums restored afterward. The fault
wrapper passes in `gain-map-faults.log` (103 seconds; build/test peak RSS
1.10 GiB, minimum available 19,295 MiB). This verifies those regressions'
ability to detect the specified faults, not all possible box-parser defects.

The restored production archive passes, followed by the library, unchanged
normal/strict/divergence locks and compatibility surface, doctests, strict
workspace/all-target and focused Clippy, and append tests with all default
features disabled. `gain-map-final-validation.log` records rc0/172 seconds,
build/test peak RSS 1.69 GiB and minimum available 18,744 MiB. The final
restored executable, sources and profiles are retained in `gain-map-final`;
existing ignores, byte expectations and quality thresholds are unchanged.


The current-main integration also passes duplicate-box, complete native/primary
JPEG reconstruction and compatibility-surface checks against the frozen sibling
graph. Its six overlays restore exactly. Evidence:
pr145-reference-main-integration.log, rc0/182s, peak RSS1.08GiB,
minimum available18922MiB. The preceding review tip f928b171 passed complete CI,
including Windows ARM, macOS ARM and i686; the integrated tip has its own pending
CI verdict. These checks do not establish the separate C5 reconstruction-gamut
contract or expand raw ICC/bundle validation.
