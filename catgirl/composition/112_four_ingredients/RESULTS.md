# Four published neuron ingredients: a bounded-policy negative result

We completed the planned experiment on the laptop GPU. **These settings did not produce a catgirl, useful personality composition, or a dependable prediction advantage from the published descriptions.** Small writing-preference effects were numerically sensitive and did not establish semantic specificity. The semantic-answer assay failed its baseline check and cannot support an interpretation claim.

This tests four known coordinates under a new, deliberately bounded intervention policy. It does **not** retest the earlier successful showcase at its stronger settings, or show that those effects never occur.

## ELI5

We chose four neurons that people had described as influencing excited/cute expression, hugs, paws and tails, and female self-description. We measured them on new examples, changed their values modestly, and tried every combination. Other agents predicted the results before the test answers existed. Reviewers scored the answers without seeing which changes produced them.

The changes did not make Qwen adopt the intended personality. Some supplied sentences became slightly more likely, but that did not translate into the corresponding behavior. Higher-precision calculations and unrelated-neuron controls made those small effects less convincing. We also found a problem with our test for understanding: its baseline answer score was already inadequate.

We therefore have a documented negative result for these settings, rather than evidence that we understand a catgirl mechanism or have disproved interpretable neurons.

## Fixed methods and completed work

The model was original BF16 `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, using MLX on an Apple M3 Max. Coordinates are zero-based post-SwiGLU scalars before the down projection:

| Published ingredient | Coordinate | Canonical direction | Selected fraction | Discovery response rule |
|---|---|---|---:|---|
| Genki cute | L13 N4810 | positive | 1 | failed |
| Hugs | L4 N8016 | negative | 1 | passed |
| Paws & tail | L23 N4769 | positive | 1 | failed |
| Female, layer22 | L22 N8295 | negative | 0.25 | passed |

Exact original Neuronary wording and links are in [SOURCES.json](SOURCES.json). These are prior steering observations, not validated natural-activation explanations. No new neurons were discovered. Four strengths were calibrated on discovery data; model weights were unchanged.

The [plan](112_PLAN.json) was committed in `d9c7d1f` before discovery. All eight forecasting sessions and their jobs were sealed and backed up in `3d5e21d` before validation. The [clarifications](PREVALIDATION_CLARIFICATIONS.md) fix analysis and recipe-selection details prospectively. No doses, primary cases, forecasts or endpoints changed after seeing validation results.

| Phase | Completed |
|---|---:|
| Natural discovery, target and random coordinates | 60 contexts each |
| Discovery writing calibration | 13 conditions × 12 contexts = 156 rows |
| Natural validation, target and random coordinates | 60 contexts each; 48 primary meaning/word cases |
| Held-out writing probes, two timing scopes | 336 rows |
| Held-out semantic probes, two timing scopes | 1,344 rows |
| Free generation | 16 combinations × 20 prompts; 4 strong singletons × 20; random-four × 20 |
| Duplicate random-pass baselines | 20; all matched main baseline text exactly |
| Masked behavior reviews | 420 replies, two reviews each; 15 disputed feature cells adjudicated |
| Precision audit | 36 rows; all 24 matched intervention contrasts |
| Conversation/late-input extension | 45 replies; two reviews, two disputed cells adjudicated |

The intervention moves a value one-sidedly toward its discovery q95 or q05, leaving already-more-canonical values unchanged. Quantiles use neutral-readout-text positions in the synthetic discovery corpus. They are not universal native limits. Genki and Paws failed discovery calibration and were retained as explicitly nonresponsive diagnostics, as planned. See [all discovery values and failed doses](public_discovery/manifest.json).

## Every combination: no observed personality effect

Across **all 16 states**, each of the following was identical to baseline case by case:

| Endpoint | Positives per 20 replies, every state |
|---|---:|
| Bubbly/excited expression | 0 |
| Female assistant self-role | 0 |
| Sustained cute register | 0 |
| Personal affection | 0 |
| Own animal body | 0 |
| Feline assistant self-role | 0 |
| Performed meowing | 0 |
| Hug/cuddle vocabulary | 1 |
| Paw/tail/fur vocabulary | 2 |
| Own-voice emoji | 3 |
| Coherence | 20 |

The five control conditions had the same trait counts. Main-condition task relevance ranged from 18–20/20; baseline and controls were 19/20, and full-four was 18/20. Quality is not pooled with personality. Original reviewers and uncertainty sensitivity remain separate in the complete exports.

Thus every observed trait marginal effect and pair interaction is zero in the adjudicated main grid. These are sample results, not proof of a zero effect on all possible prompts. Degenerate bootstrap intervals on all-zero contrasts do not provide such a proof. Six overlapping pairs are not six independent neuron samples.

The hooks did change activations. Full-four achieved block RMS was about 0.1486, versus 0.0855 for random-four and 0.0863–0.1394 for strong singletons. Discovery matching therefore did not produce exact held-out matching. This block norm across layers is a nuisance budget, not functional equivalence. [The budget audit](SEMANTIC_AND_BUDGET_AUDIT.md) also reports unchanged fractions, original range departures, and a one-BF16-step strong-Paws endpoint overshoot; its high departure percentage should not be mistaken for large excursions.

## Natural activity and writing are different findings

Natural primary association is the canonical-signed meaning-without-keyword contrast at the final neutral readout, divided by that coordinate's discovery final-position SD:

| Ingredient | Semantic contrast / discovery SD | Positive families / 3 |
|---|---:|---:|
| Genki | +0.2455 | 2 |
| Hugs | +0.0863 | 3 |
| Paws | −0.0599 | 0 |
| Female | +0.0393 | 2 |

None meets the descriptive 0.25-SD plus two-family marker; Genki is close and its continuous value matters more than a pass/fail label. Keyword, carrier and readout-text diagnostics remain separate. These few synthetic templates and one position cannot identify every natural function. Random-coordinate diagnostics are included in [measurement diagnostics](measurement_diagnostics/FINDINGS.md).

Matched writing effects below are changes in target-minus-control **mean-token log probability**, nat/token, across three contexts. They concern supplied continuations, not spontaneous personality:

| Ingredient | Enhancement BF16 | Enhancement float32 | Attenuation BF16 | Attenuation float32 |
|---|---:|---:|---:|---:|
| Genki | +0.00500 | +0.00116 | −0.00211 | −0.01458 |
| Hugs | +0.01050 | +0.00148 | +0.05293 | +0.05699 |
| Paws | +0.01414 | +0.00562 | −0.03294 | −0.03097 |
| Female | −0.01330 | +0.00035 | −0.01929 | −0.00282 |

Float32 means computation on exactly cast original BF16 weight values, not a different checkpoint. All 24 contrasts were retained: 20 signs agree, while mean absolute discrepancy is 0.01416 nat/token. No enhancement mean reaches the original +0.01 marker in float32. Paws retains enhancement-positive and attenuation-negative signs in 3/3 contexts, but its BF16 random comparator also gains +0.01351 in 3/3. Random controls were not repeated in float32. We therefore cannot establish specificity. Hugs attenuation increases preference, opposing a simple monotonic necessity account.

Full sentence probabilities depend strongly on length. Their small candidate masses are recorded, but do not mean the corresponding meanings are impossible. Off-target contexts, target/control changes, both timing scopes and all candidate probabilities are preserved.

## The semantic interpretation assay was not adequate

Matched forced-choice accuracy was 50% at baseline and remained 50% for every intervention at both timing scopes. Small Yes/No likelihood shifts cannot establish improved understanding or absence of a neuron's semantic role. **Baseline assay competence should have been a gate before confirmation; we missed that design check.**

A separately documented [post-hoc diagnostic](POSTHOC_SEMANTIC_PLAN.json) generated baseline continuations for all 48 original contexts. Input token IDs matched the original probes exactly. Nineteen replies did not start with Yes/No; some acknowledged the account before answering, and capitalization varied. For example, one actual answer was `I have read the account.\n\nno`, although forcing capitalized Yes versus No at the first position favored Yes. Literal-word lists are another possible task confound. [All 48 replies](posthoc_semantic_diagnostic/replies.jsonl) are retained. This diagnosis does not replace the frozen results or constitute a new intervention finding.

## Did descriptions improve advance predictions?

Eight fresh sessions, two per information arm, predicted 192 activation outcomes, 48 writing changes, and 144 condition/family vectors of 12 behaviors each. Their 15,744 normalized forecast rows were locked before validation. Values below average **separate session losses**, not ensembled predictions; lower is better:

| Information | Natural Brier | Writing magnitude MAE | Writing sign Brier |
|---|---:|---:|---:|
| Correct descriptions + projections | 0.14674 | 0.06501 | 0.25971 |
| Projections only | 0.14345 | 0.09201 | 0.26242 |
| No neuron semantics | 0.13143 | 0.04797 | 0.25694 |
| Shuffled descriptions + projections | 0.15058 | 0.07965 | 0.26449 |

Correct descriptions improve writing-error scores over projections, but not over the no-semantics arm. There is no dependable advantage across alternatives. Natural prevalence, prediction calibration, per-session differences, all 12 behavioral endpoints and all reviewer modes remain in the CSVs. The absent personality effects make this a poor test of forecasting successful composition. Two sessions per arm and one underlying forecaster model further limit generalization.

Independent calculations matched all 504 score groups and endpoint comparisons. The [full audit](INDEPENDENT_PRIMARY_FULL_AUDIT.json) checked 9,160 numerical values; maximum discrepancy was 2.78×10⁻¹⁷. Numerical agreement verifies calculations, not the validity of the semantic assay.

## Conversation and placement

Because no state produced the strict composed role, the predeclared fallback selected full-four `1111`, alongside baseline `0000` and singleton `0001` (a tie, not a successful persona).

Each ran three actual four-turn histories plus three late-input cases. All 45 replies were coherent; no female/feline identity, cute register, personal affection or own animal body appeared. **Identity retention is unassessable because no identity was induced at turn 1.** Steering remained active throughout, so this does not test carryover after removal. Late inputs were 766, 893 and 818 tokens, not broad long-context tests. [Every turn, trait, quality score and input length](persistence_diagnostics/FINDINGS.md) is retained.

## Scientific conclusion and sharing decision

This is counterevidence to the narrow expectation that these four published descriptions straightforwardly predict useful, composable behavior under this bounded policy. It does not refute the descriptions at their original steering settings, estimate a fraction of clean neurons, establish superposition, or test subjective feelings or hidden thoughts.

[Goodfire's essay](https://already-clean.goodfire.com/) argues for studying native neurons with neuron-specific contexts and downstream mechanisms, and cautions against generic percentile standardization. Our test covers a much narrower policy and lacks that circuit evidence. Its null result is not a decisive test of the broader hypothesis.

The frozen sharing gate **fails**: no personality changes occurred in four or more states, so no forecast advantage can rescue that criterion. We have not produced a promotional result image. The defensible output is this transparent negative report, including the assay failure.

Main limits are four chosen coordinates, fitted doses, synthetic English contexts, three template families per ingredient, short greedy outputs, one model, shared agent-judge errors without human gold labels, imperfect strength matching, and limited precision sensitivity. A stronger follow-up must first validate its baseline semantic assay and discover neuron-specific effective settings on separate data, before freezing a fresh confirmation set.

## Data and reproduction

- [Complete scored replies and objective CSVs](public_results/manifest.json): all 420 main/control replies and 45 conversation/placement replies, original reviewers, uncertainty, adjudication and full histories.
- [Discovery measurements and calibration](public_discovery/manifest.json): all cases and failed doses.
- [Exact forecasts](forecasts_normalized.csv), [forecast lock](forecast_lock.json), [source descriptions](SOURCES.json), [environment](ENVIRONMENT.json), [hook integrity](GPU_INTEGRITY.json), [precision plan](PRECISION_PLAN.md).
- [Independent findings](INDEPENDENT_FINDINGS.md), [measurement audit](measurement_diagnostics/FINDINGS.md), [semantic/budget audit](SEMANTIC_AND_BUDGET_AUDIT.md), [persistence audit](persistence_diagnostics/FINDINGS.md).
- [4.05 MiB primary measurement archive](data/measurements.zip) and [README reproduction instructions](README.md): recompute the primary analysis without a GPU. Full agent transcripts and weights are excluded.
