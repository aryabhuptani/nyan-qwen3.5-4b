#109 pride probe: CPU analysis

This report is descriptive. It does not establish a pride neuron or choose a new model intervention. Primary results use only the fresh validation split; calibration and lexical diagnostics remain separate.

Captured 132/132 natural cases and 2464 causal order records. Missing/incomplete data and the literal frozen gate calculations are exposed in109_summary.json.

Calibration quantiles are linear/type7 quantiles over calibration carrier tokens only. Lastmean is the final carrier-token scalar. Wholetextmax uses logged story boundaries, with explicit fallback warnings. Natural pre/post equality is an implementation check.

| Primary own-accomplishment semantic condition | Cases | Families | Mean probability change | Mean log-odds change | Positive fraction | Negative fraction |
|---|---:|---:|---:|---:|---:|---:|
|baseline|12|4|0|0|0.000|0.000|
|control_0_carrier_q10|12|4|0.0019937704|0.0026041667|0.333|0.250|
|control_0_carrier_q90|12|4|0.00085740342|0.013020833|0.417|0.417|
|control_1_carrier_q10|12|4|0.001161148|-0.0026041667|0.250|0.417|
|control_1_carrier_q90|12|4|0.00011073545|0.0078125|0.500|0.417|
|control_2_carrier_q10|12|4|-0.0016280498|-0.010416667|0.333|0.500|
|control_2_carrier_q90|12|4|-0.00065607651|-0.0078125|0.333|0.500|
|target_carrier_q10|12|4|-0.00046434592|0.0026041667|0.333|0.417|
|target_carrier_q50|12|4|0.0024682311|0.0052083333|0.250|0.250|
|target_carrier_q90|12|4|-0.0006835551|-0.0078125|0.250|0.417|
|target_carrier_suppress_high|12|4|-0.00079027996|-0.0026041667|0.250|0.333|
|target_carrier_zero|12|4|0.00066277703|0.0026041667|0.333|0.333|
|target_output_q10|12|4|0.003350885|0.010416667|0.417|0.417|
|target_output_q90|12|4|0.00045454959|0.0026041667|0.250|0.250|

Frozen association gate: True. Frozen semantic gate: False. These are prospective descriptive thresholds, not statistical significance.

The primary semantic gate retains all12 own-accomplishment family-language cases. Conditional target suppression is compared with the prespecified downward proxy of each random all-carrier q10 clamp. These operations and residual perturbations are unmatched; no matched-operation claim is made.

Both semantic choice orders are averaged before condition-minus-baseline pairing. Each family cluster averages its related language/category rows; translations are not independent replications. Positive/negative fractions and individual effects are preserved, including zeros and failed predictions. High-event suppression is reported on all cases and separately on independently selected natural-high cases, without silently changing denominators.

Semantic and lexical probabilities are conditional on the two specified answer tokens. Their absolute total token mass, exp(logP(option1))+exp(logP(option2)), is reported by condition/language/category, alongside unconditional pride-option probability. Low mass limits free-response interpretation; the frozen endpoint is not changed because mass is low.

Previous published additive dose+0.4 corresponded to raw offset about19.6603, roughly51.2times the calibration carrier q90-q50 spread of about0.38423. That context/scope-specific additive displacement is not identical to a natural-range replacement and is not a standard-deviation claim.

All random controls, lexical tasks, language/category strata, natural contrasts and q90-minus-q10 differences are in [summary](109_summary.json), [natural CSV](109_natural.csv) and [causal CSV](109_causal.csv). Suppression is expected to decrease pride odds; a lexical effect alone does not establish the nonlexical semantic claim. Apply the separately frozen gate, including every prescribed random control, without post-hoc selections.
