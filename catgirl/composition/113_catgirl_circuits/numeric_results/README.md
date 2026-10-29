# Numeric validation labels

These files contain binary/uncertain feature judgments, opaque item identifiers, condition/case/family identifiers and hashes. They contain no generated reply text or conversation messages. All408 outputs are represented; both original reviewers and uncertainty sensitivities remain separate. The label lock is byte-identical to the original scoring lock.

From the repository root, reproduce the primary tables without a GPU:

```sh
python3 catgirl/composition/113_catgirl_circuits/analysis_guard.py \
  --labels catgirl/composition/113_catgirl_circuits/numeric_results/labels.csv \
  --forecasts catgirl/composition/113_catgirl_circuits/forecasts_normalized.csv \
  --baseline-rates catgirl/composition/113_catgirl_circuits/simple_forecasts/discovery_base_rates.json \
  --out /tmp/study113-public-replay
```

Use a fresh output directory. For sensitivities substitute the corresponding `labels_reviewer_A.csv`, `labels_reviewer_B.csv`, `labels_uncertain_zero.csv` or `labels_uncertain_one.csv`. The public tables allow checking numerical analysis; inspecting the underlying semantic judgments additionally requires the raw reply/evidence archive, retained locally pending publication approval.
