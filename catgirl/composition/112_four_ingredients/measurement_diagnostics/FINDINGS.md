# Completed measurement audit

All 24 fixed matched contrasts were recomputed from raw rows and agree exactly with the comparison CSV. Float32 means computations on the exactly cast original BF16 checkpoint values, with unchanged discovery doses.

| Ingredient | Operation | BF16 mean | Float32 mean | Positive cases BF16/float32 | Sign agreement |
|---|---|---:|---:|---:|---:|
| genki | canonical | +0.004998 | +0.001164 | 1/3; 3/3 | 1/3 |
| genki | attenuate | -0.002108 | -0.014580 | 1/3; 1/3 | 3/3 |
| hugs | canonical | +0.010502 | +0.001482 | 2/3; 2/3 | 3/3 |
| hugs | attenuate | +0.052928 | +0.056989 | 2/3; 2/3 | 3/3 |
| paws | canonical | +0.014145 | +0.005617 | 3/3; 3/3 | 3/3 |
| paws | attenuate | -0.032941 | -0.030969 | 0/3; 0/3 | 3/3 |
| female | canonical | -0.013299 | +0.000347 | 1/3; 2/3 | 2/3 |
| female | attenuate | -0.019294 | -0.002820 | 0/3; 1/3 | 2/3 |

Units: intervention-minus-baseline change in target-minus-control mean-token log probability, nat/token. Canonical and attenuation means are distinct and must not be pooled as a desired-behavior score.

No canonical float32 mean reaches the predeclared +0.01 nat/token marker. Paws retains positive enhancement and negative attenuation in all three contexts, but its mean enhancement shrinks from +0.014145 to +0.005617. Hugs attenuation increases target preference in two of three contexts, with mean +0.056989 in float32; this opposes a simple monotonic necessity interpretation. Genki canonical is tiny despite three positive float32 signs. Female canonical changes from a negative BF16 mean to a near-zero positive float32 mean.

Across all 24 contrasts, 20/24 nonzero signs agree. Mean absolute discrepancy is 0.014155, maximum 0.041120 nat/token.

Candidate continuation masses are extremely small (see all24 table), so these effects concern relative preferences among supplied completions. They do not measure the probability of spontaneously producing a catgirl role. Exact-sequence probabilities shrink with length; low mass does not show these meanings are impossible. Variable continuation length also limits mass comparisons across cases.

The unrelated same-layer Paws control has BF16 matched writing mean +0.013507 nat/token (3/3 positive), close to the native Paws mean +0.014145. Random writing controls were not repeated in float32. Native sign consistency alone is not evidence of semantic specificity or a clean mechanism. All matched random BF16 writing means are retained in precision_matched_table.csv.

## Unrelated-coordinate natural diagnostic

| Matched target context | Unrelated coordinate | Semantic contrast / own discovery final SD | Same-direction families | Descriptive marker |
|---|---|---:|---:|---|
| genki | L13 N7887 | -0.219799 | 1/3 | False |
| hugs | L4 N4897 | -0.072682 | 1/3 | False |
| paws | L23 N7626 | +0.112690 | 2/3 | False |
| female | L22 N6906 | -0.201906 | 0/3 | False |

All four fail the directional 0.25 SD plus two-of-three-family marker. This is a fixed four-control diagnostic, not a statistical population test or evidence that published target meanings are correct. Standardization uses each control coordinate’s own final-position population SD over all 60 discovery capture cases. Validation uses the 48 natural cases only. Control signs follow the paired target convention.

Family effects and all case scalars are retained separately. Control activation outcomes were not added to the original target forecasts. Source hashes and complete numerical tables are in summary.json.
