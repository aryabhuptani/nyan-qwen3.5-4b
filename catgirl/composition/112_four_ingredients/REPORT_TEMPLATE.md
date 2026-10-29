# Four published neuron ingredients: what transferred, what composed, and what failed

**Status:** report skeleton, written before outcome analysis. Replace placeholders with locked results and cite exact artifacts. Do not omit planned negative results or promote exploratory diagnostics to primary findings.

## Plain-language result

We tested four previously published neuron descriptions in Qwen3.5-4B. We measured normal activity on examples that separated meaning from words, made bounded changes to individual neuron values, and tested all16 combinations of four selected settings. Separate agents predicted the effects before validation answers existed. Condition-masked reviewers scored the resulting answers.

**Result:** [State the strongest supported result in one sentence, or say the experiment was inconclusive.]

[Explain which measured behaviors changed and which did not, without a metaphor. Distinguish word occurrence, adopted role, coherence, and task performance. State whether descriptions helped predict outcomes beyond projections and no-description controls.]

## Scope and provenance

- Model: original BF16 `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; laptop GPU; exact environment and hardware from runtime metadata.
- Four deliberately selected, previously tested coordinates: Genki cute L13N4810; Hugs L4N8016; Paws & tail L23N4769; Female (layer22) L22N8295. These are neither newly discovered coordinates nor a random sample of neurons.
- Exact Neuronary labels and source text are in `SOURCES.json`. They describe prior **steering observations**, not established natural functions. Paws & tail predicts vocabulary, not necessarily feline identity; Hugs predicts wording, not necessarily performed affection.
- Four scalar intervention strengths were fitted by a fixed discovery-only rule. No weights or learned steering direction were trained. This still involves calibration, and must not be advertised as parameter-free discovery.
- Original studies110/111 remain unchanged. This new design does not retroactively rescue their inconclusive and failed-calibration results.
- Cite the frozen plan, pre-validation clarifications, source/case hashes, forecast lock and backup commit before validation, labels lock, runner hash and analysis/audit hashes.

## What was fixed before validation

| Stage | Frozen design | Completed count / artifact |
|---|---|---|
| Native discovery capture |60 cases:48 meaning×word cases plus12 writing prefixes; all four coordinates | [fill] |
| Per-coordinate calibration | Three bounded fractions; choose smallest meeting matched writing gain≥0.01nat/token and positive gain in≥2/3 discovery cases; retain strongest diagnostic if none | [fill] |
| Native validation capture |60 cases; primary natural association on48 meaning×word cases, all four coordinates | [fill] |
| Causal writing | All four enhancements and attenuations on all12 contexts; last-prompt and all-assistant-prediction scopes | [fill] |
| Causal semantic readout | All four enhancements and attenuations on48 cases; carrier and last-answer-prediction scopes | [fill] |
| Composition |16 states×20 fresh generation prompts; fixed decoding and128-token limit | [fill] |
| Controls | Discovery RMS matched same-layer random channels; nominal aggregate-strength singletons; disclose unattainable budgets | [fill] |
| Prospective forecasts | Four information arms×two fresh sessions;192 natural,48 lexical and144 factorial-family rows per session | [fill] |
| Reviews | Two independent masked reviews, third masked adjudication, original reviewer and uncertainty sensitivity | [fill] |
| Precision sensitivity | Prospective BF16 sensitivity audit; preserve its exact methods and designated scope | [fill] |
| Persistence and late input | Frozen deterministic recipe selection, three four-turn cases, three late-input cases; steering continuously active | [fill] |

List incomplete phases explicitly. Counts of tokens, prompts, coordinates and factorial pairs are different statistical units.

## Calibration and intervention fidelity

| Ingredient | Discovery support / count | Endpoint and selected fraction | Matched discovery gain | Nonresponsive? | Achieved validation change |
|---|---|---|---|---|---|
| Genki cute | [fill] | [fill] | [fill] | Discovery rule flagged nonresponsive; verify exact artifact | [fill] |
| Hugs | [fill] | [fill] | [fill] | [fill] | [fill] |
| Paws & tail | [fill] | [fill] | [fill] | Discovery rule flagged nonresponsive; verify exact artifact | [fill] |
| Female | [fill] | [fill] | [fill] | [fill] | [fill] |

A nonresponsive calibration remains a diagnostic at fraction1; it is not dropped, replaced, or silently set to zero. Quantiles come from discovery neutral-readout-text positions, not a universal natural distribution. The operation moves one-sidedly toward q95/q05; existing values beyond the endpoint remain unchanged. Report achieved BF16 changes, unchanged rates, own observed-range departures, actual down-column norms and control budget ratios. Multiple marginally bounded interventions can jointly be unusual.

Goodfire's essay cautions against global percentile standardization and emphasizes sparsity, neuron-specific context and downstream circuits. Our per-coordinate percentile policy is one deliberately bounded intervention policy, not a decisive test of those broader claims. A synthetic-corpus percentile does not identify the appropriate natural causal setting for a given prompt. Original Neuronary steering notes are not natural-activation explanations.

## Natural activity: meanings versus literal words

Primary: final post-context readout scalar, standardized by each coordinate's final-readout population SD across all60 discovery cases. Show semantic-without-keyword and keyword-without-semantics contrasts, all three held-out template families and individual cases. Readout-text, carrier and keyword-token summaries are separate diagnostics.

| Coordinate | Semantic contrast, raw / discovery SD | Literal-word contrast | Family sign agreement |≥0.25SD marker? | Carrier future-word negative control |
|---|---|---|---|---|---|
| [four rows] | [fill] | [fill] | [fill] | [fill] | [fill] |

Distinguish anticipated/contextual meaning from processing a word when it appears. Report whether the expected direction transferred; a threshold marker is not a definition of clean representation. Three template families give limited evidence of generalization.

## Surgical causal writing

Primary writing measure: intervention-minus-baseline change in unconditional mean-token-logprob(target)−mean-token-logprob(control). Show matched and off-target4×4 context/intervention matrix, enhancement and attenuation, both timing scopes. Preserve baseline preferences, separate target/control changes, summed/token log probabilities and candidate masses. High conditional preference at tiny candidate mass cannot establish natural semantic use.

| Intervention | Matched mean gain | Positive /3 contexts | Cross-context effects | Attenuation effect | Same-layer random comparator |
|---|---|---|---|---|---|
| [four rows per primary scope] | [fill] | [fill] | [fill] | [fill] | [fill] |

Suppression is one-sided attenuation toward a discovery median. Its effect alone does not establish necessity unless the relevant natural contexts actually recruit the coordinate.

## Semantic interpretation versus Yes bias

Report semantic×literal cell effects, Yes−No log-odds changes, expected-answer accuracy and semantic discrimination separately. Compare carrier and readout interventions and matched/off-target contexts. A uniform shift toward Yes is an interpretation bias, not evidence of improved understanding. Include baseline performance, candidate mass and precision sensitivity.

## All16 combinations and control conditions

Report each source-faithful endpoint separately: bubbly/excited expression, own-voice emoji, hug/cuddle words, paw/tail/fur words and female assistant self-role. Keep strict cute style, affection, own animal body, feline self-role, meowing, coherence and task performance distinct.

- Include the complete16×20 scored grid, every condition rate and all failures.
- Per ingredient: factorial average marginal effect, baseline-to-single effect and effect when other ingredients are on.
- Primary pair contrast with other ingredients off: `p11−p10−p01+p00`. Average-over-background interaction is secondary. Show all four cells, conditional effects and trait retention. Risk-scale interaction is not by itself evidence of a physical neural interaction.
- Compare nominal-strength singletons and random-coordinate condition with achieved strength/clipping shown. Failed control quality cannot establish semantic necessity when matching is imperfect.
- Bootstrap whole prompt units only; disclose authored-case assumptions and original-review sensitivity. Four chosen coordinates and six overlapping pairs are not independent population samples.

## Did original descriptions help prediction?

Four arms: correct words+own projections; deranged words+own projections; own projections only; no neuron-specific semantics. Two fresh sessions per arm; identical parameter/task information, no discovery replies or pass flags. Procedural isolation is not an OS security boundary.

| Endpoint / task | Correct words | Projection only | No semantics | Shuffled words | Per-session / reviewer stability |
|---|---|---|---|---|---|
| Natural activation Brier | [fill] | [fill] | [fill] | [fill] | [fill] |
| Lexical magnitude MAE | [fill] | [fill] | [fill] | [fill] | [fill] |
| Lexical positive-direction Brier | [fill] | [fill] | [fill] | [fill] | [fill] |
| Each of12 behavior endpoints separately | [fill] | [fill] | [fill] | [fill] | [fill] |

Average separate session losses rather than averaging probabilities first. Show prevalence, constant predictors, paired prompt uncertainty and exact per-endpoint differences. Never combine quality with persona to create a favorable score. Shuffled descriptions are misleading information, not a neutral control. A claim of practical persona prediction needs the frozen≥0.01 Brier benefit over projection/no-info, both-session and reviewer stability, and actual matched-baseline trait changes in≥4 factorial states. Report all comparisons even if this fails.

## BF16 sensitivity

The prospective audit covers all12 held-out lexical-writing contexts, with baseline and each context's matched enhancement and attenuation:36 rows and24 intervention-minus-baseline contrasts. It casts the identical pinned BF16 parameter values to float32, preserves the frozen doses and canonical tokenization, reuses the hook, and uses unchunked uncached teacher-forced scoring. This is a precision comparison of the same checkpoint, not a new trained model or independent scientific replication. Report all contrasts, absolute discrepancies, sign agreement and zeros; no cases or doses are selected by the audit.

[State whether signs/rankings changed. Report audit timing relative to outcomes. Do not imply a small lexical gain is reliable solely because many floating-point outputs were recorded. Preserve original BF16 analysis.]

## Persistence and late-input results

Use the strict **composed-role** conjunction: cute register+affection+own animal body+female self-role+coherence. This does not require a feline identity. Count all20 replies per state; ties use fewer active ingredients then lexical bit ID; if no state has any joint reply choose1111 as diagnostic. Singleton selection is fixed separately in `PREVALIDATION_CLARIFICATIONS.md`. Report female+feline+coherent separately.

[Show deterministic selection counts; all turns and late-input cases; traits, coherence and task quality. Continuous intervention measures maintained steering, not unassisted carryover after steering removal. No recipe tuning after this selection.]

## What this says about “Neural Networks Are Already Clean”

[Give bounded evidence/counterevidence for these exact published ingredients. Association, a causal wording effect, semantic interpretation, useful composition and forecast value are separate findings. Agreement among them is stronger evidence than a showy output; a dissociation can be informative too. Neither positive nor negative results here estimate model-wide cleanliness, identify every natural function, establish feelings, or decode a hidden thought process. Cite the original essay for its actual claims rather than treating it as a binary hypothesis.]

## Limitations and reproducibility

Four chosen known coordinates; four fitted strengths; synthetic English contexts; few independent template families; one model snapshot and greedy trajectory; limited output lengths; agent rather than human judges; shared model/rubric errors; uncertain scoring; purposeful source-word operationalization; native marginal support does not establish joint typicality; imperfect norm matching; possible small-effect precision sensitivity; prospective predictor procedural isolation only. List any post-freeze implementation amendments with invalidated and replacement runs.

Link frozen source descriptions, plans/clarifications, case files, selected-dose curves, complete scored CSVs, exact forecasts/locks, runtime hashes, analysis outputs, independent recomputation and reproduction commands. Publish key methods/results rather than duplicate large local runtime transcripts. State clearly whether the predeclared share threshold was met. If not, do not manufacture a promotional graphic; a transparent negative report remains valid.
