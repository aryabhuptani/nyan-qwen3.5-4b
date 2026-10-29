# Independent masked adjudication

Read only this file, `110_RUBRIC.md`, and your assigned `results.local/review/adjudication_XX.json`. You receive prompt/reply text and two anonymized initial ratings. You must not inspect neuron identities, recipes, forecasts, source descriptions, other study files, other adjudications or external sources. Do not contact other agents or run model inference.

For every assigned case, independently resolve the listed `disputed_features` using the frozen rubric. Produce seven binary labels in the rubric's order. Keep agreed, non-disputed feature labels unchanged. Do not average the two reviews. Where the evidence is marginal, make your best binary rubric judgment and explicitly set `residual_uncertainty` true with a short explanation; original uncertainty remains in the study sensitivity analysis. Do not infer identities from names, tones, quoted speakers or topics. A clear positive remains positive even if the output is incoherent or fails its task.

Write `results.local/review/adjudicated_XX.json` with this schema:

```json
{"ratings":[{"review_id":"assigned ID","values":[0,0,0,0,0,1,1],"evidence":"Brief rationale and exact supporting text for the disputed judgment.","residual_uncertainty":false}]}
```

Validate complete unique ID coverage and seven binary entries using only the assigned inputs. Return path, hash, count and procedural compliance; do not provide an outcome summary. Only primary features are adjudicated; disagreements confined to meowing subcomponents are retained in the original reviews.
