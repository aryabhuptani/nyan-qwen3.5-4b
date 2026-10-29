# Calibration sensitivity to original reviewers

Computed during validation generation, without inspecting validation replies or labels. These diagnostic reruns do not change the selected doses, validation job, or predictions.

Applying the identical frozen selection rule separately to the two original masked discovery reviews gives:

| Modifier | Final adjudicated fraction used | Original reviewer A | Original reviewer B |
|---|---:|---:|---:|
| Paws | .5 | .5 | .5 |
| Hugs | .25 | .25 | .25 |
| Genki | 1 | 1 | 1 |
| Sounds | .5 | .25 | .5 |
| Pride candidate | .25 | .25 | .25 |

Four of five selected doses agree across original reviewers. Sounds has dose uncertainty on this small four-prompt discovery set. The main experiment uses the already sealed adjudicated fraction .5 (normalized offset −.2625). It does not reselect after seeing validation. Each subfolder preserves full candidate counts and source label hashes.
