# Independent masked calibration review

Read only this instruction, the unchanged `../110_atlas/110_RUBRIC.md` feature definitions, and your assigned `calibration_review.local.jsonl`. The rubric's111-use is the same seven feature definitions and attribution rules; its old110 validation denominators do not apply. Do not inspect mappings, source descriptions, doses, plan, job, prior results, or other reviewers. No inference or external lookup.

Score all120 replies independently. Each is a separate conversation. Write one JSON object with a `ratings` list containing each review ID exactly once:

```json
{"review_id":"assigned ID","values":[0,0,0,0,0,1,1],"hard_loop":false,"evidence":"Exact short evidence or a concise absence note."}
```

The seven values are feline assistant self-role, female assistant self-role, performed feline vocalization, sustained cute register, explicit virtual affection, coherent, and task relevant. Each is integer0/1 or literal `uncertain`. Preserve genuine ambiguity rather than guessing. A hard loop forces coherence0 but does not erase an earlier clear self-role or sound. A fictional narrator, quoted character, drafted speaker, feminine name, or external animal is not automatically the assistant. A sound alone does not establish feline identity, ordinary warmth does not establish sustained cute register, and helping language alone does not establish personal affection. Natural truncation does not erase a coherent completed clause.

Return only the ratings file path, SHA256 and coverage count. Do not send feature summaries or candidate opinions. This is a single independent masked calibration review, not confirmatory validation or a blinded human study.
