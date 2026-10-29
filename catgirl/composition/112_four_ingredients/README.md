# Four native ingredients: wording, meaning and composition

This study separates three questions: whether published Neuronary descriptions predict natural activity, whether interventions change particular writing preferences, and whether combinations produce the corresponding conversational behaviors. These questions can have different answers.

The fixed ingredients are [Genki cute](https://neuronary.org/r/5v8axayv), [Hugs](https://neuronary.org/r/pybhb985), [Paws & tail](https://neuronary.org/r/dg2d4u26), and [Female, layer22](https://neuronary.org/r/6waz9uek). Exact original source wording and zero-based coordinates are in [SOURCES.json](SOURCES.json). These are previously published and previously tested coordinates, not new discoveries. Publisher steering descriptions are hypotheses about transfer, not established natural functions.

## Prospective stages

1. **Discovery measurement:** capture all four native post-SwiGLU/pre-down-projection scalars on60 cases. Forty-eight cross semantic content with literal vocabulary; twelve provide writing-choice contexts. Distinct held-out templates are fixed separately.
2. **Independent calibration:** for each neuron, test fractions0.25/0.5/1 of a one-sided move toward its discovery q95 or q05, in the publisher's direction. Select the smallest fraction meeting the frozen writing-effect rule on three matched discovery contexts. Retain nonresponsive ingredients at the highest tested fraction and label them as such. Four intervention strengths are fitted on discovery data; model weights remain unchanged.
3. **Forecast lock:** eight fresh agents receive correct source words, shuffled words, vocabulary projections, or no neuron semantics. Every arm receives identical prompts, thresholds, doses and scoring definitions. All15,744 normalized forecast rows are sealed before validation, including natural-activation probabilities, writing-change estimates and per-endpoint factorial forecasts. The original packets are preserved in `forecast_inputs.zip`.
4. **Held-out measurements:** natural activity, enhancement and attenuation at distinct positions, all16 on/off combinations on20 fresh prompts, and strength controls. Semantic Yes bias is separated from interpretation accuracy. Lexical probability changes are separated from free-generation behavior.
5. **Masked review and persistence:** two independent agent reviews, third adjudication for disputes, preserved uncertainty and evidence. A deterministic rule then selects conditions for three four-turn conversations and three late-input cases. Actual assistant history is retained. No dose changes are permitted after validation.

The authoritative [112_PLAN.json](112_PLAN.json) was committed in `d9c7d1f` before discovery. All eight forecasts and derived intervention jobs were backed up in `3d5e21d` before any validation output. [Pre-validation clarifications](PREVALIDATION_CLARIFICATIONS.md) resolve endpoint, control-budget and persistence-selection details; they do not change doses or cases.

## What is measured

Natural primary association uses the final neutral assistant-readout scalar; carrier, keyword and readout-text means are separate diagnostics. The keyword exercise occurs after the carrier, so an earlier carrier cannot respond to those future words. Every coordinate is captured on every context; forecasts do not receive context-to-ingredient labels.

Writing probes retain unconditional continuation log probabilities, per-token probabilities, candidate mass and conditional preferences. A low-mass forced continuation can change probability without becoming a likely answer. Semantic probes place their question in the user message; increasing Yes across both positive and negative cases is not improved interpretation.

Behavioral endpoints distinguish bubbly expression, emoji, hug/cuddle vocabulary, paws/tail/fur vocabulary, female self-reference, sustained cute style, personal affection, own animal body, feline self-role and enacted meowing. Coherence and task fulfillment remain separate. A quoted hug can count as vocabulary without counting as assistant affection.

Factorial interactions use the observed paired contrast `p11 − p10 − p01 + p00`; shared ingredients and hand-authored prompts limit inference. Control offsets match discovery RMS perturbation budgets through down-projection column norms, with achieved held-out budgets reported. This does not establish identical downstream effects. Strong singletons stay within observed marginal endpoint ranges; some may be unable to match the full-composition budget. Random controls are explicitly not constrained to their own native range.

## Verification and reproduction

The exact checkpoint is original BF16 `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. MLX on Apple Silicon is required for inference. Model weights are downloaded separately. Empty wrappers matched unmodified-model logits exactly; cached-decoding checks and actual achieved intervention values are preserved in [GPU_INTEGRITY.json](GPU_INTEGRITY.json). A prospective [float32 sensitivity check](PRECISION_PLAN.md) casts the original weight values to higher precision and repeats every matched held-out writing contrast.

```sh
.venv/bin/python catgirl/composition/112_four_ingredients/measurement_selftest.py
.venv/bin/python catgirl/composition/112_four_ingredients/analyze.py --selftest
python3 catgirl/composition/112_four_ingredients/run_validation.py
# In a clean replication output directory, with the pinned snapshot downloaded:
env HF_HOME=/path/to/model-cache .venv/bin/python \
  catgirl/composition/112_four_ingredients/run_validation.py \
  --model /path/to/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output-root /path/to/fresh-replication --run
```

The runner refuses changed frozen inputs and differing existing outputs. New replications must preserve the published decision artifacts and use separate output paths. Raw activation arrays and full agent transcripts remain local; compact scientific outputs, complete scored replies, source words, frozen forecasts, methods and hashes are the publication bundle.

## Interpretation limits

Four chosen coordinates cannot establish a model-wide cleanliness fraction. Discovery ranges are measured on synthetic contexts, not universal native limits, and individually observed values can form unnatural joint states. Single English greedy trajectories cannot establish multilingual, stochastic or broad conversational robustness. Same-model agent reviewers can share errors; no human gold standard is claimed. Neither generated persona language nor a word-probability change identifies subjective feelings or reads the model's thought process.

The earlier [mild-dose pilot110](../110_atlas/RESULTS.md) and [failed global-dose calibration111](../111_atlas_dose/RESULTS.md) remain separate and unchanged. Results may support or challenge the tested descriptions and intervention policy; stronger claims require additional evidence.
