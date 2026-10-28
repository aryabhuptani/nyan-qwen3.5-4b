# Reproduce the L22 N1911 pride versus wording test

This study tests one fixed native coordinate in original BF16 Qwen3.5-4B. It does not train a model, fit a direction, combine coordinates, or search for a replacement candidate. Read [the prospective plan](109_PRIDE_PLAN.json), [independent method audit](109_METHOD_AUDIT.md), and [results report](109_PRIDE_REPORT.md).

The manually authored [132-case corpus](109_PRIDE_FRESH_CASES.json) contains four calibration and four validation event families, each crossed with five interpretations in English, Spanish and Chinese. Three English lexical diagnostic families separately cross own-accomplishment satisfaction with explicit pride-related wording. Translations are related observations, not independent topics; no independent native-speaker review was performed.

## Run

Use the existing project Python environment and pinned original model snapshot; see [the native reproduction guide](REPRODUCE.md) for installation. CPU validation never imports MLX:

```sh
.venv/bin/python catgirl/composition/109_pride_probe.py
```

Collect unsteered per-token activations and calibration quantiles into a **new output directory**:

```sh
.venv/bin/python catgirl/composition/109_pride_probe.py \
  --run --calibration-only --output /tmp/pride-reproduction
```

The default model is the local original BF16 `Qwen/Qwen3.5-4B` snapshot at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Override `--model` with the same snapshot on another machine. Apple Silicon and MLX are required. The runner checks all MLP projections are unquantized BF16.

Inspect and freeze the numeric calibration before causal comparisons. All numeric thresholds come exclusively from the 60 predetermined calibration cases; none are chosen from validation effects. Then run:

```sh
.venv/bin/python catgirl/composition/109_pride_probe.py \
  --run --causal-only --output /tmp/pride-reproduction
```

Each phase refuses to overwrite its existing output files. The causal phase checks the corpus, plan, revision, selected channels and runner identity against the natural-capture metadata. Committed reference results live in [results](results/).

## Exact intervention

Indices are zero based: layer 22, gated MLP channel 1911, at `silu(gate_proj(x))*up_proj(x)` before `down_proj`. Signed raw activation values are used directly. The three fixed-seed same-layer controls are N4581, N3753 and N7525; each uses its own calibration range.

Carrier operations replace the scalar at the predetermined neutral carrier positions inside the user passage, before the question and choices. They do not directly modify subsequent question tokens or generated answers. The altered computation and cache can still influence later predictions. Conditions use calibration q10, q50 or q90, zero-ablation, or conditional suppression of values above q90 to q50. Output-only conditions replace the scalar at the final prompt position predicting answer token one, with separately calibrated task-specific quantiles. No published-dose normalization is applied to these raw range clamps.

Constant replacement stays near a calibration quantile in magnitude but need not reproduce natural temporal dynamics. Zero is contribution ablation, not an opposite semantic pole. Random controls are normalized to their own natural ranges, not matched in residual perturbation norm; achieved changes and down-projection norms are recorded.

## Outcomes and integrity

Natural records retain all token IDs, carrier/story boundaries, and achieved pre/post scalars at the target and controls. Causal tests retain every fresh case, both A/B orders, all 14 conditions and exact label log probabilities. The nonlexical endpoint scores the own-accomplishment interpretation against the alternative; word choice separately scores English ` proud` against ` pleased` at a fixed assistant continuation. Both lexical candidates must be whole single tokens. Conditional probabilities must be interpreted alongside their absolute token probability mass.

The small readable confirmation generates 36 replies from 12 prospectively selected cases under three carrier conditions. It uses greedy decoding, seed 42, 96-token budgets, repetition penalty 1.1 over 128 tokens, and no direct output steering. Those are illustrations separate from the objective likelihood outcomes.

`109_implementation_checks.json` records exact baseline/restoration agreement and a cached-versus-full-forward diagnostic. Causal comparisons all use independent, identically segmented full forwards; they do not mix full and cached baselines. BF16 chunking can introduce arithmetic differences, so cached diagnostic discrepancies are reported separately.

`109_completion.json` records expected completed counts and output SHA256 identities. The CPU analyzer produces separate natural and causal CSVs and summaries, preserving multilingual strata, lexical controls, random controls, eligible high events, and individual effects. Small synthetic sets and label-free carrier tests support bounded claims only; neither a positive nor a null result settles global monosemanticity or subjective emotion.
