# Analysis interface, fixed before validation

`analyze.py --manifest analysis_inputs.json --output analysis/` consumes explicitly listed files; it does not search for runtime outcomes. Authoring and CPU self-tests must not read study112 validation results. Root supplies the outcome manifest only after the label lock. Paths are resolved relative to the manifest file. Raw file hashes and the executed analyzer hash are saved in the summary.

Manifest sections are optional while developing; the final report must list which planned sections were executed. Example (paths are placeholders):

```json
{
  "discovery": {"raw": "results.local/discovery/natural.jsonl"},
  "natural": {"raw": "results.local/validation/natural.jsonl", "cases": "CASES_natural_validation.json"},
  "lexical": [{"raw": "results.local/lexical_generation/probe.jsonl", "cases": "CASES_lexical_validation.json", "scope": "generation"}],
  "semantic": [{"raw": "results.local/semantic_carrier/probe.jsonl", "cases": "CASES_natural_validation.json", "scope": "carrier"}],
  "factorial": {
    "labels_csv": "labels.csv",
    "conditions": "generation_conditions.json",
    "endpoint_columns": {
      "bubbly_excited_expression": "bubbly_excited_expression",
      "lexical_hugs_cuddles": "lexical_hugs_cuddles",
      "lexical_paws_tail_fur": "lexical_paws_tail_fur",
      "female_assistant_selfrole": "female_assistant_selfrole"
    },
    "extra_endpoints": ["emoji_in_assistant_voice", "sustained_cute_register", "explicit_virtual_affection", "own_animal_body", "feline_assistant_selfrole", "performed_feline_vocalization", "coherent", "task_relevant"]
  },
  "forecasts_csv": "forecasts_normalized.csv"
}
```

Natural/probe rows follow `measure.py`: `case_id,condition_id,spans,activations`; probe rows also have `choices`, candidate mass and operations. Default condition IDs are `baseline`, `canonical_<ingredient>`, `attenuate_<ingredient>`; explicit per-section `conditions` metadata can override these using objects `{condition_id, ingredient, kind}`. Scope is explicit per input section, never guessed from file names. Only baseline natural rows enter association. Discovery SD is population SD of FINAL readout activation over all60 discovery cases for each coordinate. Cases carry frozen semantic, keyword, family and intended ingredient metadata; forecast packets must omit that metadata.

Natural primary: final readout scalar. Readout-text mean, carrier mean and keyword-span mean are separate diagnostics. Compute canonical-signed semantic difference when literal words are absent and canonical-signed literal-word difference when semantics are absent; paired by template family. Carrier future-keyword differences are a negative control. No pseudoreplication over tokens. The0.25SD marker also needs expected direction in at least two of three families; missing/zero discovery SD is reported rather than repaired.

Lexical primary: intervention minus baseline change in **mean-token-logprob(target) minus mean-token-logprob(control)**. Save target/control gains, sum-logprob contrast, candidate masses and per-case results too. Every coordinate is tested on every writing case; primary matched-ingredient results and cross-ingredient specificity matrix are separate. Enhancement's practical marker is matched gain≥0.01nat/token and positive gain in at least2/3 cases. Attenuation is reported with its own direct sign; it is not automatically necessity. Controls must be marked `kind: random` to avoid conflating them with target coordinates.

Semantic primary diagnostics: signed `sum_logprob(yes)−sum_logprob(no)` change, by true semantic/literal cell. Expected-answer accuracy and semantic discrimination are separate. A uniform Yes shift is never called improved interpretation. All coordinates run on all48 contexts; publisher-matched contexts and off-target contexts remain separately visible.

`labels.csv` has one row per condition and case, with `condition_id,case_id`, optional `family,reviewer`, and endpoint columns. Values are0/1 or blank/`uncertain`. `reviewer` defaults to `final`; independent reviewer values can be additional rows for sensitivity, or set `labels_by_reviewer` to `{reviewer_name: path}` instead of `labels_csv`. Four source-faithful endpoint columns are required by the final root manifest; additional strict-role and quality columns may be included via `extra_endpoints`. Endpoint names must match `FEATURE_DEFINITIONS.json` and forecast CSV exactly; do not alias them to ingredient names. Include all12 endpoints to score all frozen forecasts, still reporting each separately. Unknowns are counted and never silently turned into zeros. Each review must contain the entire16×20 factorial grid. `generation_conditions.json` is a list (or object with `conditions`) of `{condition_id,mask}` where `mask` is four bits in `genki,hugs,paws,female` order; controls use `kind: control` and omit mask. Factorial effects use matched case units and complete endpoint tuples.

Factorial effects: average marginal effect across the eight backgrounds, baseline-to-single effect, all-others-on effect, pair risk-difference interaction with other ingredients OFF primary, and interaction averaged over remaining states secondary. Report each endpoint separately. Quality never enters a persona macro score. Paired prompt bootstrap is descriptive; four ingredients and six shared-component pairs are not independent neuron samples.

`forecasts_normalized.csv` is a long interchange format. Columns: `arm,replicate,task,ingredient,case_id,condition_id,family,endpoint,probability,predicted_change`. Task `natural` uses case+ingredient with probability of canonical final activity crossing the frozen discovery q75/q25 threshold. Task `lexical` uses case+ingredient, predicts canonical-generation preference change in nat/token; save `predicted_change` and optional probability of positive change. Task `factorial` uses condition+family (or condition+case) and endpoint with probability. Exactly the same scored units must exist across compared arms/replicates; mismatches fail validation. Brier scores remain per endpoint/task, lexical magnitude uses MAE, and sign scoring is separate. No quality/persona macro. Aggregate correct-arm comparisons must identify whether they average separate session losses or probabilities; the analyzer uses separate session losses.

Outputs: compact summary JSON; case-level association/causal CSVs; factorial effects CSV; per-session forecast scores and paired arm contrasts. Every output is reconstructible from the declared inputs. Root determines the final scientific wording and sharing gate; analyzer values do not create an automatic cleanliness claim.

Optional `perturbations` is a list of `{raw,metadata,label,selection,calibration,coordinate_kind}`. `metadata` is the runner's phase metadata containing down-column norms. `selection` is frozen `SELECTED_DOSES.json`, providing target discovery matching budgets and, for `coordinate_kind: target`, own observed ranges. For `coordinate_kind: random`, also supply that control run's discovery `calibration` to obtain its own observed ranges. Directly operated channels are analyzed separately from propagated changes in other measured channels. Input-position and each teacher-forced candidate branch are distinct variants; their repeated context positions are never presented as independent evidence. Output reports achieved scalar/residual RMS, unchanged fractions, own-range departures and matching ratios. Four-bit condition IDs can directly specify factorial masks when the conditions file is the frozen factorial job.
