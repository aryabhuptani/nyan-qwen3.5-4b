# Audit of the frozen 103 activation association screen

The saved results reproduce the frozen computation exactly. They support an association between three already selected native coordinates and this synthetic joy-text contrast. They do not establish happiness, natural neuron function, or effective causal control. Narrative length and writing style remain material confounds. No ranking, corpus, script, or candidate selection was changed in this audit; no 104 results were read.

The raw capture has shape `[10 contexts, 5 conditions, 32 layers, 9216 neurons]`, contains only finite values, and matches the recorded SHA256. Corpus, plan, and published-norm hashes also match the capture metadata. Every saved statistics array and the full discovery order recomputes exactly from the raw activations. The compressed full ranking contains all 294,912 unique coordinates, with consecutive ranks. Confirmation values do not enter that discovery order.

All 50 token logs contain the same 17 measured suffix token IDs. Scored starts equal chat-prefix length plus narrative length, and each topic uses the same prefix length across its five conditions. The suffix is “For now, I am sitting at the table and thinking about what comes next.” It contains no direct emotion label. The capture metadata identifies the pinned original snapshot, and its configuration says BF16, 32 layers, hidden size 2560 and intermediate size 9216. The capture implementation checks unquantized BF16 MLP projections and leaves intervention fields unset. This is a metadata/computation audit, not an independent fingerprint of every model weight. Published residual norms establish provenance; they were not used to normalize these natural activation measurements.

The following means describe all ten topics; they are not intervention coefficients. Native activations can be negative.

| Condition | Narrative tokens, mean (range) | L22N1911 mean | L28N1478 mean | L31N33 mean |
|---|---:|---:|---:|---:|
| Joy | 29.2 (26–32) | 0.286 | −0.601 | 3.165 |
| Neutral | 22.6 (20–25) | 0.094 | −0.048 | 1.598 |
| Sadness | 28.9 (26–31) | 0.055 | −0.146 | 2.144 |
| Cute | 28.8 (27–31) | 0.150 | −0.129 | 2.064 |
| Agreeable | 24.0 (21–27) | 0.070 | −0.092 | 1.191 |

Joy narratives average 6.6 more tokens than neutral and 5.2 more than agreeable; sadness and cute are much closer in length. The shared suffix removes direct differences in the measured tokens, but its absolute positions and preceding narratives still differ. Across the 50 captures, after subtracting each topic's five-condition mean, activation/length Pearson correlations are **+0.424, −0.540, and +0.548** for the fixed first three discovery coordinates, respectively. Across all axes, the median absolute within-topic correlation is 0.296; 62,141 axes have absolute correlation at least 0.5. These descriptive correlations are not adjusted effects or independent causal evidence. The forty joy-minus-control pairs also reuse each joy narrative four times.

Length is not a complete explanation by inspection: joy and the sadness/cute controls have similar mean lengths, and confirmation joy narratives are slightly shorter on average than those two controls. Yet all three fixed coordinates retain their expected mean contrast signs against both. That observation does not separate event content, intensity, sentence structure, successful-goal language, arousal, and length. The corpus is manually authored, small, and stylistically related across its disjoint topics; confirmation is a topic holdout rather than an independent author or corpus.

For the frozen first three discovery coordinates, the sign below was fixed from discovery joy-minus-neutral. The displayed effects are signed mean differences divided by sample SD plus the frozen 0.05 native-unit floor; they are descriptive standardized contrasts, not inferential statistics.

| Fixed discovery rank / coordinate | Fixed sign | Confirmation vs neutral | vs sadness | vs cute | vs agreeable | Individual confirmation pairs in expected direction |
|---|---:|---:|---:|---:|---:|---:|
| 1 / L22N1911 | + | 0.749 | 1.218 | 0.130 | 0.737 | 14/16 |
| 2 / L28N1478 | − | 1.095 | 0.997 | 1.067 | 0.961 | 16/16 |
| 3 / L31N33 | + | 1.551 | 0.908 | 0.683 | 1.644 | 13/16 |

All four confirmation mean signs persist for each of these three. L22N1911 reverses against cute and agreeable in `baking`. L31N33 reverses against cute in `library`, and against sadness and cute in `baking`. All first-three discovery comparisons were direction-consistent across their six discovery topics; the weaker confirmation consistency illustrates selection effects. Among the fixed discovery top 64, 61 retain all four mean signs in confirmation, but only 15 retain the expected direction on every one of the sixteen topic/control pairs. Their median worst confirmation standardized contrast is 0.512. These are diagnostics of the frozen list, not a new ordering or recommendation.

**L28N1478 is naturally lower in these joy-conditioned captures**, opposite the positive happiness-word-writing pole described by the root's separate screen. Its discovery joy-minus-neutral mean is −0.660, and its confirmation mean is −0.393. Association under teacher-forced narratives and effects of adding activation during free generation can disagree. Neither direction should be relabeled to make the findings agree; this audit does not independently review that separate writing experiment.

The ranking tests 294,912 axes using only six discovery topics, without a permutation null, multiplicity correction, or representative emotion sampling. The 0.05 denominator floor also affects comparisons between axes with different activation scales. Raw captures and full rankings preserve those limitations for later review. The frozen single-coordinate intervention screen remains a separate causal test; confirmation diagnostics here must not be used to replace its already selected candidates.

Sources: [frozen plan](103_HAPPY_ACTIVATION_PLAN.json), [fixed corpus](103_HAPPY_CONTRAST_CORPUS.json), [capture metadata](results/103_happy_activation.capture.json), [saved summary](results/103_happy_activation.summary.json), and [compact audit diagnostics](results/103_association_diagnostics.json). Raw activation NPZ remains local; the compressed complete ranking preserves all original axes.
