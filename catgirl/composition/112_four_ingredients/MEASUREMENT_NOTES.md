# Study112 native measurements

`measure.py` adapts the raw native hook used by `109_pride_probe.py`. It loads the pinned, unquantized Qwen3.5-4B BF16 snapshot only when `--run` is present. It does not instantiate the normalized-dose Runner or alter its calibration files. No learned steering vectors, optimization, or source-label-based activation selection is implemented here.

```sh
.venv/bin/python catgirl/composition/112_four_ingredients/measurement_selftest.py
.venv/bin/python catgirl/composition/112_four_ingredients/measure.py \
  --cases PATH_TO_CASES.json --output PATH_TO_NEW_OUTPUT_DIRECTORY --phase natural
# Add --run only in the single authorized GPU process.
```

Each of the four declared coordinates is measured on **every** case. A case has `case_id`, `split`, `text_prefix`, `readout_text`, and exact unique `target_span` inside the prefix. Optional `keyword_span` separately identifies literal-word context. The assistant prefix is teacher forced, not generated. We capture:

- Every token's raw signed `silu(gate_proj(x))*up_proj(x)` channel immediately before down projection.
- Carrier, literal-keyword, full neutral-readout-text, final prediction-position, and all-input spans separately.
- Actual canonical token IDs, offset-mapped boundary policy, fully rendered model input, and raw scalar arrays. Carrier positions cannot attend to later words. The neutral readout may attend to all previous input.

The neutral system is `You are a helpful assistant.` Thinking is disabled using the same chat-template argument as prior studies. The full rendered text is canonically tokenized once. Character spans map to all canonical tokens overlapping them; tokens straddling a span boundary are retained and explicitly listed. A diagnostic compares canonical encoding with separately encoded segments and records any difference, but only canonical IDs enter the model. The exact resulting token sequence is held identical across intervention conditions. This procedure measures responses to synthetic inputs, not a universal natural activation distribution. CPU checks with the actual local pinned tokenizer found no segment-versus-canonical differences on the 48 discovery natural cases in either natural or semantic-probe format.

Natural output includes discovery-only quantiles for each channel and each scope. All discovery cases are pooled, not merely cases labeled with that channel's presumed meaning. The number of tokens is retained. Quantiles are token-weighted, so longer spans carry more weight. Validation rows never determine these quantiles. Case-weighted association analysis belongs in the separate analysis script.

## Frozen intervention job

For `--phase probe` or `--phase generation`, provide `--job` with:

```json
{
  "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "case_ids": ["explicit_case_id"],
  "seed": 42,
  "temperature": 0,
  "max_tokens": 96,
  "repetition_penalty": 1.1,
  "conditions": [
    {"condition_id": "baseline", "operations": []},
    {"condition_id": "example_only_not_a_frozen_dose", "operations": [
      {"ingredient": "genki", "mode": "toward", "direction": 1,
       "value": 0.123, "fraction": 0.5, "scope": "readout"}
    ]}
  ]
}
```

The numerical example above is only interface documentation, not a recommended dose. The root protocol must freeze numeric endpoints from discovery data, the case list, timing, and hashes before validation. One operation per ingredient per condition prevents order-dependent changes to the same scalar.

Modes: `add` requires explicit `[lower, upper]` clipping; `replace`/`clamp` set an explicit raw value; `suppress_high` or `suppress_low` move only scalar values beyond an explicit threshold to a specified value; `restore` is a no-op; `toward` moves a fixed fraction `(0,1]` toward an endpoint, only when that endpoint is in the specified direction. Natural values already beyond a directional endpoint remain unchanged. Consequently a toward operation is bounded relative to the endpoint but does not force every natural value inside the discovery interval. Addition with clipping can itself change out-of-range values in an unintended direction; prefer directional toward for the primary bounded probes.

Zero replacement is not automatically physiological or suppressive. Only use zero when it lies inside the frozen discovery interval, otherwise use the nearest frozen interval bound and explicitly record that substitution. Negative published steering direction need not imply that zeroing a signed neuron suppresses its function. The direction and wording must be stated precisely in the report.

Scopes: `carrier`, `keyword`, `readout_text`, `readout`, `all`; `generation` begins at the last prompt position (the position predicting the first answer token) and continues through subsequent answer predictions. In generation phase `all` includes input and generated positions. In teacher-forced probe phase `all` includes the input, not candidate-continuation tokens, unless the fixed job explicitly chooses generation scope. No hidden schedule changes occur.

Teacher-forced semantic cases add `probe_question` to the **user message**, after a fixed two-newline delimiter; in this phase the neutral assistant acknowledgment is omitted. The template then emits a fresh assistant generation prefix. Final `readout` means the position predicting the Yes/No answer; neutral `readout_text` is empty in semantic-probe phase. Carrier and keyword spans remain within the original user context. Lexical-writing cases instead use `messages` plus `assistant_prefix`, omit `probe_question`, and supply `choices: {label: continuation_text}`. For every choice, output includes each unconditional token log probability, full continuation log probability, mean token log probability, total candidate mass, and normalization over the declared choices. Candidate mass is not a semantic probability; prefix-overlapping continuations are not disjoint events. Use equal-format, prefix-disjoint continuations when treating their sum as probability mass. Conditional contrast can be large while unconditional mass is tiny, so report both. Uniform Yes preference shifts must be stratified by true semantic presence and reported separately from correctness.

All intervention runs record the requested operation, token scope, and **achieved** pre/post BF16 scalar values. Down-projection column norms are retained for residual-change diagnostics. Changes at other measured channels downstream are genuine propagated effects, not evidence that the hook directly changed extra coordinates. Quantile clipping or replacement arithmetic can round in BF16; use achieved values for implementation checks.

Outputs refuse to overwrite a phase's raw JSONL. Completion records include raw, case, runner, and job hashes plus versions/configuration. A partial failed phase should be preserved and rerun in a fresh output directory. CPU tests exercise selector math, exact boundary construction, direction-aware suppression, and absence of GPU imports; GPU execution must additionally check no-op identity and observed achieved ranges on discovery data before interpreting results.


For unrelated same-layer control runs, `--coordinates coordinates.json` replaces the default mapping with explicit `name: [layer, neuron, direction]` triples. One coordinate per layer is allowed in each process; use a separate identically configured pass for same-layer controls. The coordinate file hash enters metadata. Names in frozen condition operations must match the replacement map. No random control selection occurs inside this runner.

## Discovery selection and matching jobs

`build_jobs.py --stage discovery` constructs the frozen 13-condition writing calibration grid. `--stage select --writing PATH_TO_DISCOVERY_PROBE_JSONL --control-calibration PATH_TO_CONTROL_CALIBRATION_JSON` applies the preregistered selection rule, preserving every curve and selecting fraction1 diagnostically if no fraction passes. It builds native, unrelated-control and factorial validation jobs without reading validation outcomes.

Matching budgets use all discovery `readout_text` scalars (explicit final-readout fallback when empty), transformed by each selected one-sided toward operation. Nominal scalar RMS times its down-projection column norm gives the per-coordinate residual RMS. This calculation uses float64 arithmetic on observed BF16 input values; held-out actual BF16 achievements and budget ratios must be reported separately. Actual writing-branch RMS is retained as a separate discovery diagnostic, not substituted into the matching formula.

Before validation, strong-singleton construction was clarified: use the same one-sided toward mechanism, with each coordinate's observed discovery minimum/maximum endpoint; choose its fraction to target the full four-coordinate nominal block energy, capped at1. The cap and any unattainable nominal budget are explicit. Full-four block energy is the square root of the sum of per-layer squared residual RMS values, not the norm of a combined same-layer vector. Unrelated controls use constant additive offsets matched to each selected native residual RMS; their ±1e30 clip is a machine guard and does **not** make those controls native-range interventions. Report their own-range excursions and limitations.

Natural activation forecast thresholds use separately recorded `final_readout_q25`/`final_readout_q75`, not the dose-support `readout_text` quartiles.
