# Gain-map color contract

Issue 144 C5 remains a reconstruction-color qualification gap. Part 2 9.11's
optional bundle color fields describe the reconstructed alternate image.
Leaving both fields unset selects the baseline image's color encoding. The
inner gain-map codestream carries its own image metadata independently.

`HdrFromSdrRequest::encode` supplies separate SDR/HDR primaries and transfer
functions to `compute_gainmap_slice`. It encodes the base with the selected
SDR color encoding and the gain plane as Gray8 or RGB8, then constructs a
bundle with both alternate color fields unset. The former comment justified
that choice using the inner gain plane's encoding; the comment is corrected.
No metadata, pixel arithmetic or API changes accompany that correction.

The owner selected the HDR alternate gamut with gain computation there on
November 19. The main implementation still has the source flow described above;
the candidate and retained numerical controls below establish the repair and
its remaining contract gates. The bundle's version/ICC-presence checks remain
independent of physical color reconstruction.

This request currently exposes packed pixel inputs. No strided-row or broad
HDR-source qualification is claimed here. C1's separate duplicate-box rejection
and approved fallible-helper PR remain independent of the C5 color contract.


## November 19 measured candidate and remaining decisions

The owner selected an explicitly tagged HDR alternate gamut and gain computation
in that gamut. The local candidate normalizes source transfers and converts the
base to those primaries. In the retained matched colored BT.2020 case, actual
decoded gain codes change [190,143,108] to [201,140,105], and independent
full-weight relative-linear-sRGB reconstruction error changes
0.20308315048595427 to 0.002938900545130796. These are candidate observations;
the implementation is not on main yet.

Colored HLG requires a luminance-based OOTF. The per-channel candidate produces
[38,127,237] rather than native [48,126,216]; sharing the encoder's luminance
OOTF corrects those gain codes. Independent reconstruction error changes
0.04939544133077409 to 0.0005363980673157576. A separate float-base before control
finds input 64/255 rendered as 0.53808534 (sRGB), 0.85212886 (PQ), 0.73932827 (HLG)
in native output. Typed PQ/HLG base layouts and private sRGB linearization pass
the selected layout/alpha controls. The associated logs are
gain-map-hlg-before-v2, gain-map-float-base-first-after and
gain-map-float-all-and-faults. The last includes four restored transfer faults;
these are selected contracts, not full HDR qualification.

Two decisions still prevent completing the candidate:

- The established small_pair_produces_container_with_jhgm fixture supplies
  SDR alpha 128/255 and HDR alpha 1 while expecting success. The proposed exact
  equal-alpha guard rejects it. A retained original mismatch with base alpha 1
  and HDR alpha 0.5 yields native alpha 65535/65535. The fixture is untouched while
  the owner decides whether to require equal alpha and migrate this smoke input
  or explicitly ignore alternate alpha.
- The post-transfer-fix candidate still mixes absolute PQ units with a relative
  linear base. Native v0.12 maps PQ 128/255 to 94.0746255219 nits; the actual base
  header white is 255 nits and the base sample is 0.05. Both decoders recover
  gain [0, 0, 0]. Independent full-weight reconstruction from those decoded planes
  gives 12.748323529958725 nits, maximum error 81.32630199194128 nits. Common base-white
  normalization requires code 163. The absolute/relative input-unit and automatic
  headroom contract is pending owner choice; its new failing regression remains.

Both individual codestreams fully render through primary Rust and pinned native
decoding before these calculations. The reconstructed HDR values above are
independently modeled, not observed compound HDR-player output. No independent
compound renderer has been qualified. Broad candidate validation also retains
the existing unequal-alpha library failure; no assertion, ignore, expected
byte or scientific threshold has been changed.
