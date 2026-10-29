# Four published ingredients: proposed study112

This is a design proposal written before study112 discovery or validation outputs. The root agent's final frozen plan, not this proposal, determines executed procedures. Studies110 and111 remain unchanged. These four ingredients were selected deliberately from prior work; no estimate of the fraction of clean neurons is possible.

## Questions and exact source claims

1. Do naturally occurring activations distinguish the published behavior from nearby meanings and mere words?
2. Does a surgical change within a measured native range cause the corresponding output preference to change, including under reverse intervention?
3. Do calibrated ingredients retain their individual effects when combined, and can source descriptions predict the unseen effects better than controls?

The source is the frozen `../110_atlas/selected30_atlas.json`. Preserve the following publisher text verbatim in all explanation conditions:

| Ingredient | Coordinate; published signed dose | Verbatim excerpt | Publisher-faithful generation endpoint |
|---|---|---|---|
| Genki cute | L13 N4810; +0.8 | Turn it up and the model gets bubbly, excited and full of emoji. | Bubbly/excited expressive register; emoji counted separately. |
| Hugs | L4 N8016; −0.8 | Turn it down and hugs and cuddles show up in most answers. | Hug/cuddle wording; performed affection is a separate stronger claim. |
| Paws & tail | L23 N4769; +0.8 | Turn it up and paws, tails and fur turn up in its answers. | Paw/tail/fur wording; assistant animal identity is a separate stronger claim. |
| Female (layer 22) | L22 N8295; −0.8 | when asked its gender. | Female assistant self-description in identity-eliciting contexts. |

These are published intervention descriptions, not established accounts of normal computation. In particular, Paws & tail does not claim feline identity, and Hugs does not claim a felt emotion. Female's limited source excerpt must not be silently expanded.

## Frozen splits and staged disclosure

- Author the discovery and held-out validation cases before model inference. Keep their identifiers, texts, labels and hashes fixed. Authoring a split does not make its examples a random population sample.
- The cases agent proposes 96 natural probes: four ingredients × three independent context families × four semantic/literal cells × two splits. Eight generation discovery prompts, twenty generation validation prompts and three later persistence cases are separate.
- Root may inspect discovery only while implementing the predetermined calibration policy. Keep held-out activity, causal outcomes and generation answers sealed until all doses, schedules, endpoints and forecast files are locked.
- Discovery-only adaptations must be recorded. A failed ingredient stays in the full factorial and results. Do not replace it with another attractive neuron or retune it after validation.
- A clear implementation bug can be repaired with an explicit amendment and invalidated run; a disappointing effect is not an implementation bug.

## A. Natural activity and surgical causal tests

Cross semantic relevance and literal vocabulary. Near-miss cells must include external/quoted/story content versus assistant self-role where relevant. Read out the chosen neuron at fixed annotated positions; distinguish activations before the target word from activations while processing the target word. A word causing its own processing activation is a different claim from anticipating its meaning.

Use a fixed native activation summary for each annotated span, with equal case weighting. Do not select the most favorable token or layer after inspecting outcomes. For each ingredient report all four cell means, the semantic contrast at vocabulary absence, the vocabulary contrast at semantic absence, and their difference. Plot individual cases. With only three context families per split, family transfer is descriptive; narrow or bootstrap intervals are not evidence for broad multilingual abstraction.

For causal readout, use matched teacher-forced target and near-miss continuations. Intervene only at the prespecified causal input/answer decision position or fixed small target span; never decide timing from validation traces. Score the same outputs with and without intervention. Report both raw log-probability changes and a fixed matched preference contrast. Normalize sentence likelihood by the frozen scoring convention, and disclose unequal token lengths. Include same-direction and equal-magnitude reverse changes. Reverse intervention is a suppression test only if it actually reduces the relevant native activation; it is not automatically an ablation.

Native-range calibration must use discovery activations only. Report quantiles, zero/sign structure and the amount of each intervention. The measurement implementation supports one-sided movement toward discovery q90 for positive ingredients or q10 for negative ingredients, with fractions0.25,0.5,1. Positions already beyond the target remain unchanged, so these are bounded movements, not a claim that every natural value lies inside q10–q90. Freeze this operator before inference. Record unchanged frequency, actual achieved BF16 changes and resulting residual norm. Fixed native-range values are not automatically typical for every context, nor does remaining inside marginal support prove a natural counterfactual.

Suggested practical thresholds to freeze before discovery are a source-faithful teacher-forced mean preference gain of at least0.10nat and the expected sign in at least two of three held-out context families. For natural association, use a semantic-without-word contrast of at least0.25 discovery standard deviations and the expected sign in at least two of three families; if discovery variance is zero, report the unstandardized contrast and do not invent a denominator. These are practical descriptive markers, not definitions of neuronal meaning or statistical significance. Show every raw contrast and uncertainty regardless of whether markers are met. Stability under the alternate fixed scoring convention/reviewer sensitivity is required for a strong claim. Use separate labels for:

- association only;
- causal lexical preference;
- causal semantic/role preference beyond matched literal wording;
- unsuccessful/unstable tests.

Four ingredient-specific primary contrasts require a fixed multiplicity policy (Holm for formal null tests, if tests are defensible). The small, deliberately selected sample supports ingredient case studies. Do not turn token counts or multiple contrasts within one case into independent samples.

## B. Per-ingredient calibration, then the full factorial

Suggested deterministic selection: choose the smallest fraction0.25,0.5,1 with discovery mean publisher-faithful teacher-forced preference gain at least0.10nat. If none qualifies, use fraction1 and label the calibration unsuccessful; retain all results. Semantic and lexical gains remain separate and cannot substitute after looking at validation. If a dose has stronger target behavior but worse coherence, show the tradeoff rather than suppressing the ingredient or invoking a common90% gate. Selecting from this mechanistic contrast avoids picking a setting solely because it produced a showy answer.

Timing is part of the intervention. If comparing prompt-only, answer-only or a fixed early-answer window, enumerate those alternatives before discovery, use an explicit deterministic choice rule, and record all discovery outcomes. Freeze the selected timing before validation. Do not choose timing separately for each validation question. A surgical causal test and a full-response steering test are separate experiments and must retain their different claims.

At frozen doses, generate all 16 on/off states of the four ingredients for the same twenty prompts: 320 short answers. Baseline and singles are already part of this grid. Keep decoding, context, output budget and repetition policy fixed. Count trait positives even in incoherent answers; quality is a separate outcome. Report the joint event of trait plus coherence as a separate useful-output measure, with its denominator intact.

Add a small fixed set of controls, preferably no more than eight extra conditions × twenty prompts = 160 answers:

1. Four same-layer random-coordinate controls, selected before outputs with a published seed and exclusion list. Match each ingredient's achieved residual perturbation scale and timing on discovery; report imperfect matching rather than claiming exact equivalence.
2. Four matched-strength single-ingredient controls for the aggregate four-way intervention. Match the aggregate perturbation budget using a declared metric; do not silently exceed the native-support constraint to reach it. If matching is impossible, retain it as an explicitly unmatched diagnostic rather than creating a natural-range claim.

Different layers inject perturbations at different locations, so neither an L1 sum nor a root-sum-square of residual norms yields an exact common functional strength. Report the chosen budget and per-layer norms; it is a nuisance control, not a proof that composition exceeds all one-neuron alternatives.

An eight-prompt discovery grid of four ingredients × three dose candidates uses96 answers, plus shared baseline eight. The full factorial and eight controls use480 validation answers. This leaves room below approximately1,000 short answers for reverse-direction diagnostics and the later persistence phase. Avoid spending the budget on many nearly identical weak doses.

## C. Trait effects and interaction analysis

Keep four publisher-faithful primary endpoints and their denominators distinct. Identity-specific evaluation belongs to eliciting prompts; report ordinary-prompt generalization separately. Semantic/role endpoints, meowing, full catgirl conjunction, coherence, task performance, hard loops and truncation are secondary diagnostics. Absence of feline identity cannot invalidate the narrower Paws & tail wording prediction.

For each ingredient estimate its factorial average marginal effect: average the trait rate with that ingredient on minus off over the other eight matched states and held-out prompts. Also report its baseline-to-single effect and its conditional effect when all other ingredients are on. This reveals an ingredient that works alone but loses its effect in composition.

For each pair use the observed risk-difference interaction:

`I(A,B) = p11 − p10 − p01 + p00`,

averaged over the two remaining ingredients' states and matched prompts. Report each cell and both conditional marginal effects. Do not use `clip(pA+pB−p0)` as a mechanistic independence prediction. Risk-scale interaction depends on the scale and baseline; ceiling effects can create negative interactions without destructive neural interference. A fixed logistic model can be sensitivity analysis, not a post hoc replacement endpoint.

For inference resample whole prompt/case units, preserving every matched factorial state and every rating. Report the finite-benchmark effects first; bootstrap uncertainty assumes the authored cases approximate a larger case population. Four selected neurons and six shared-component pairs are not ten independent discoveries. Show per-family effects and leave-one-context-family-out sensitivity. Do not claim neuron-population significance.

## D. Prospective atlas prediction

Before validation inference use fresh agents with no prior study history. Supply the fixed discovery-selected doses/timing, endpoint rubric and validation prompts, but no validation answers, activity traces or outcomes. Each arm receives the same permitted discovery summary; choose in advance whether that summary contains calibration effect measurements. If empirical discovery outcomes are supplied, the claim is explanation value beyond calibration data rather than prediction from descriptions alone.

Recommended arms: correct original Neuronary labels/excerpts plus vocabulary projections; deranged original labels/excerpts plus correct projections; projections only; no neuron-specific semantic information. Preserve opaque identifiers, source wording exactly, correct dose sign and identical token budget. Two independent sessions per arm are desirable but are not eight independent models.

Require probabilities for the four trait endpoints and quality per condition and prompt family. Forecast singles and compositions separately. Save and hash forecasts before any validation response is generated. Compare proper Brier losses on the same fixed cases, with matched baseline forecasts and per-feature results. Report correct-text versus projections as the incremental value of the words. Keep class prevalence and constant-predictor scores visible. No aggregate score may turn improved prediction of broken answers into a claim of better persona understanding.

Primary forecast summary is equal-weight macro Brier over the four distinct trait endpoints, with eligibility sets defined before outcomes. Quality is separately scored. Pair/model-forecast effects are descriptive for these four coordinates. Report per-replicate results and prompt-block uncertainty; do not bootstrap four neurons into a general atlas-quality conclusion.

## E. Scoring, persistence and share criteria

Score generated answers with condition labels, coordinates, forecasts and dose information masked. Two independent ratings plus masked adjudication for disagreements/uncertainty are preferable. Preserve original ratings and reviewer-sensitivity analyses. Human-audit the highlighted positives and near misses before making a public scientific claim; if no human audit occurs, explicitly state that the judges were agents.

After short-response validation is analyzed, run the three already-authored persistence cases with the frozen recipe. Separate steering continuously on from steering removed after turn1, because only the latter tests unassisted carryover. Later turns should include unrelated tasks. Report every turn's traits and quality, not only the strongest opening. If the short response effects fail, persistence remains a diagnostic rather than a search for a replacement success.

A defensible shareable result can be positive or negative: a publisher-faithful effect that survives the semantic/literal distinction, a clear role-versus-word dissociation, or a repeatable composition failure despite functioning singles. Require replication across at least two independently authored context families and reviewer sensitivity. State the exact coordinates, doses, operator, timing, model and tested cases. A selected cute answer alone does not establish that explanations predict native functions, persistent identity or model-wide cleanliness.

The final ELI5 should say which neuron changed which measured behavior, where it failed, what the combination added, and whether source descriptions predicted those changes. It should distinguish naturally measured activation from deliberate intervention, and avoid claims about feelings or recovered private thought processes.
