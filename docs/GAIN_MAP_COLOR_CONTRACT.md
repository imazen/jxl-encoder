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

The remaining question is whether the producer's reconstructed alternate
samples are expressed in the baseline gamut when source primaries differ.
That requires physical color reconstruction controls against the producer's
actual contract. The current source flow and a comment correction alone do
not demonstrate wrong alternate-image pixels or justify a new metadata default.
The existing bundle's version/ICC-presence checks do not settle this question.

This request currently exposes packed pixel inputs. No strided-row or broad
HDR-source qualification is claimed here. C1's separate duplicate-box rejection
and approved fallible-helper PR remain independent of the C5 color contract.
