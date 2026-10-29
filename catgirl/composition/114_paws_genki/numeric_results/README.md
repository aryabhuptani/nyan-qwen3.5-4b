# Numeric labels and CPU reproduction

These are the original numeric labels and hashes, with no generated text or conversations. Keep the final labels, both original reviews and both uncertainty bounds separate.

From the repository root, using an unused output directory:

```sh
python3 catgirl/composition/114_paws_genki/analyze_replication.py \
  --labels catgirl/composition/114_paws_genki/numeric_results/labels.csv \
  --forecasts catgirl/composition/114_paws_genki/FORECASTS.csv \
  --out-dir /tmp/study114-public-replay
python3 catgirl/composition/114_paws_genki/independent_audit.py \
  --labels catgirl/composition/114_paws_genki/numeric_results/labels.csv \
  --analysis /tmp/study114-public-replay \
  --forecasts catgirl/composition/114_paws_genki/FORECASTS.csv \
  --out-dir /tmp/study114-public-audit
```

Substitute each other labels CSV for reviewer and uncertainty sensitivity. For GPU inference, use the recorded MLX environment and pinned original BF16 snapshot:

```sh
python3 catgirl/composition/114_paws_genki/run_replication.py \
  --job catgirl/composition/114_paws_genki/JOB_validation.json \
  --cases catgirl/composition/114_paws_genki/CASES_validation.json \
  --sources catgirl/composition/114_paws_genki/SOURCES.json \
  --output /tmp/study114-generation.jsonl \
  --model /PATH/TO/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Raw generated replies and review evidence are retained locally pending publication approval. Their hashes remain bound by the original scoring/export locks. Inspecting those judgments requires the raw archive; the numeric data alone checks arithmetic.
