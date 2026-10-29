# Merge the420primary behavior-review replies

Run these commands only after all three generation jobs finish. The commands below assume the working directory is this study112 directory. They read completed outputs for packaging and integrity checks but do not score or summarize behavior.

```sh
python3 merge_behavior.py --factorial results.local/validation_factorial/generation.jsonl --strong results.local/validation_strong_singletons/generation.jsonl --random results.local/validation_random_four/generation.jsonl
python3 prepare_review.py mask --raw results.local/review_input.jsonl --cases CASES_generation_validation.json --job BEHAVIOR_REVIEW_JOB.json --out-dir results.local/review --tag factorial --batch-size 70
```

`BEHAVIOR_REVIEW_JOB.json` is metadata solely for the masking utility. Never execute it as a GPU job. It includes21review conditions: the16factorial conditions, four stronger singletons, and `random_1111`. The random run's `random_0000` condition is excluded from review only after all20generated texts exactly match main `0000` texts case by case. A mismatch aborts packaging; do not discard or repair it. Full condition×case grids are required in every source file.

The merger preserves messages, current text, finish reason and generated-token count. Activation arrays remain in the immutable raw source; merged rows contain absolute source path, SHA256, one-based line number and source condition/case keys, sufficient to retrieve the complete arrays. `review_input.manifest.json` hashes all three source files, merged output, case corpus and review metadata; it records the20successful duplicate-baseline comparisons. The merger refuses differing overwrites.

Masking creates exactly six immutable batches of70replies with fixed condition-independent shuffle seed11221. Reviewers receive only assigned batches, `BEHAVIOR_RUBRIC.md`, `FEATURE_DEFINITIONS.json` and `REVIEW_INSTRUCTIONS.md`; never give them `mapping.local.json`, review metadata, source jobs, forecasts or raw outputs.

Assign independent groups A1/B1 to batches01and02, A2/B2 to03and04, and A3/B3 to05and06. Each reviewer rates140items. Each B reviewer must be a fresh session, with no access to A ratings or condition assignments. Pairing the same batches allows direct inter-reviewer comparison; neither reviewer sees the pair's other output.

After all six review files exist, run:

```sh
python3 prepare_review.py adjudicate --out-dir results.local/review --review-a results.local/review/ratings_A1.json results.local/review/ratings_A2.json results.local/review/ratings_A3.json --review-b results.local/review/ratings_B1.json results.local/review/ratings_B2.json results.local/review/ratings_B3.json
```

This locks full coverage and original ratings before producing masked dispute packets. Assign those packets to fresh third adjudicators. After all disputes are resolved, use the same review arguments with `join`, supply all `adjudicated_*.json` via `--adjudications`, and set `--labels-dir results.local/scored`. Preserve the original reviewer-specific and uncertainty-bound files for sensitivity. No condition or forecast outcomes are examined in packaging or assigned grading sessions.
