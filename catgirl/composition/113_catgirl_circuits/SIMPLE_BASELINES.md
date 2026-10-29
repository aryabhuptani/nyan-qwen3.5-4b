# Deterministic forecast controls

Specified before discovery labels were joined or any held-out output was generated.

In addition to the two fresh sessions in each information arm, retain a deterministic additive forecast. For each of the seven forecast features, estimate the fixed-base rate and each selected-single rate from the four discovery prompts (unknown judgments omitted with their counts retained; use 0.5 only if there are no scored observations). For a combination, predict:

`clip(base_rate + sum(selected_single_rate - base_rate for each enabled modifier), 0, 1)`

Use the same prediction for all six held-out families. This intentionally simple control asks whether the agents predict composition better than adding observed singleton changes. It does not estimate real neural additivity or learn an intervention. Freeze its complete prediction CSV and discovery counts before held-out inference. Give it its own deterministic-control arm, never count it as a fresh agent session. The constant discovery-base forecast remains separately available as planned.
