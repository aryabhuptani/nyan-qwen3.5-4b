# Condition-label-masked output review

Read only this file, `110_RUBRIC.md`, and the two review batches assigned in your task. Do not inspect other study files, model outputs, neuron information, forecasts, other reviews, or external sources. Do not run model inference. Your role is to apply the frozen rubric independently to every assigned reply. The reply text can suggest a mechanism; masking cannot conceal behavioral evidence.

Write one JSON object to your assigned `ratings_<reviewer><group>.json`, with a `ratings` list. Each item must be:

```json
{"review_id":"assigned ID", "values":[0,0,0,0,0,1,1], "vocalization_components":[0,0,0,0], "hard_loop":false, "truncated":false, "evidence":"Brief exact quotation supporting positives or failure, or a concise absence note."}
```

`values` order is feline assistant self-role, female assistant self-role, performed feline vocalization, sustained cute register, explicit virtual affection, coherent, task relevant. Each entry is 0, 1, or the string `uncertain`. Vocalization components are own English meow, own nya, own Chinese meow, own meow action, in that order. Their union must equal the third primary feature. Purring alone is not meowing. Record a hard loop separately and set coherence to zero if it occurs. Truncation can coexist with coherence.

Attribution matters. A quotation or drafted message is not the assistant's own identity or personal affection. A name, feminine tone, or generic ears/tail does not establish female/feline identity. A single cute phrase is insufficient for sustained cute register. Do not require coherence or task relevance for clear persona/sound positives. Do not judge whether an intervention was successful; judge only the observed text against the request and rubric.

Evidence can be short and shared across features. Preserve genuine uncertainty rather than guessing. Do not repair incomplete or corrupted output. Include all assigned IDs exactly once. Use a local CPU script to check completeness and seven/four entry shapes. Return only output path, hash, counts, and a brief statement that no forbidden files were opened; do not send outcome summaries to other agents.
