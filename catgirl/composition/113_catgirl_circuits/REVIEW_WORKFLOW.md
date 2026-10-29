# Masked review and complete exports

Run `python3 review_selftest.py` for the CPU integrity test. No GPU or forecasts are read.

After the complete validation grid exists:

```
python3 prepare_review.py mask --raw results.local/validation/generation.jsonl --cases CASES_validation.json --job JOB_validation.json --out-dir results.local/review --batch-size 66
```

The mask command requires the exact job-condition × case grid and exact full frozen messages. It shuffles replies, replaces case/condition identifiers with opaque item IDs, and supplies only complete messages, text, finish reason and descriptive quality criteria to reviewers. Each masked row is rated independently; sources, conditions, predictions, other reviews, and mapping.local.json must stay outside reviewer context. Assign each batch once to reviewer pool A and once independently to pool B, in fresh agents. Supply REVIEW_RUBRIC.md and FEATURE_DEFINITIONS.json only as reference. The latter discloses hypothesis categories but no per-item conditions. Do not reuse forecasters as reviewers.

Each reviewer file uses the required object format in REVIEW_RUBRIC.md: reviewer_id; items with item_id, ratings object over all14features, and evidence for each positive or null rating. Preserve null when undecidable. Capability and relevance are descriptive rather than acceptance gates.

```
python3 prepare_review.py adjudicate --out-dir results.local/review --review-a A_01.json A_02.json --review-b B_01.json B_02.json
```

Pass every A/B batch filename, not only the two illustrative filenames above. Every item must have exactly one A and one B review. This seals original reviews and produces condition-masked adjudication files for every disagreement or null cell. A fresh third reviewer reads these with the rubric, preserves undisputed fields, and returns the same items/ratings/evidence format. Third ratings may remain null; no forced certainty is introduced. Adjudicators see original A/B ratings, but no experimental condition or prediction.

```
python3 prepare_review.py join --out-dir results.local/review --review-a A_01.json A_02.json --review-b B_01.json B_02.json --adjudications third_01.json --labels-dir public_results
```

Join refuses incomplete reviews/adjudications and changed existing artifacts. Outputs include final labels.csv; both independent reviews; null-as-zero and null-as-one sensitivity tables; immutable original evidence; labels_lock.json; and complete replies_and_labels JSONL/CSV with context, condition, case, family and final ratings. The source raw hashes, batches, rubric, reviews and adjudications are all bound by hashes. Condition mapping is joined only after labels are sealed. Null sensitivity tables bound genuinely uncertain cells, not every resolved disagreement; both original reviews expose disagreement sensitivity independently.

Mechanical lexical diagnostics count whole-word body, affection, feline-sound and pride vocabulary in all uses, including quotations. They are distinct from manual assistant-own behavior or integrated actions. No regex result establishes composition.
