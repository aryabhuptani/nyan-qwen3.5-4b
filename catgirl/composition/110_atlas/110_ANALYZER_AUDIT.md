# Analyzer audit before outcome access

This audit used the frozen plan, analyzer source, intervention metadata, and synthetic data. It did not inspect generated replies, actual forecast probabilities, or reviewer labels.

## Findings and corrections

- **Primary weighting matches the plan.** Complete inputs require every one of 73 conditions to share the exact eight selected prompts, two per family, and all seven primary labels. Each coordinate receives equal weight; canonical/opposite signs, families, prompts, and seven features receive equal weight within it. Random 24 and known six anchors are separate. Joint is secondary and cannot enter the seven-feature primary macro.
- **Replicates remain distinct.** The primary averages the two separate-session squared losses. The secondary ensemble squares the error after averaging probabilities. Synthetic disagreeing predictions verify these differ. Paired control contrasts are formed within replicate, then averaged within coordinate; no session is substituted for a neuron.
- **Bootstrap matches the plan.** The 10,000 seed-110 resamples select the 24 coordinate-level mean contrasts, keeping signs, prompts, features and sessions together. Identical seeded draws across contrasts preserve the pairing. Replicate-specific and ensemble intervals are labeled separately. No reply-level resampling or pair p-values are used.
- **Pair comparisons use matched canonical components.** Lookup uses component IDs, not condition list order. The singles comparator is the frozen clipped probability sum minus baseline. Leave-anchor-out comparisons remove every pair containing the excluded anchor, including anchor-anchor pairs. Shared-anchor dependence is not converted into twelve independent samples.
- **Corrected a diagnostic weighting discrepancy.** `positive_brier` and `negative_brier` now mean loss across the positive/negative observations, respectively. The earlier implementation averaged these class-conditioned losses equally across neurons containing that class, which changes the requested conditional distribution. This correction does not alter unconditional primary Brier or its bootstrap. A synthetic unequal-class-count test prevents regression.
- **Hardened exact input identity.** The analyzer checks the frozen selected-prompts hash and exact ID-to-family mapping, not merely eight internally consistent IDs. It also requires unique review IDs and validates the forecast lock and intervention hash. A synthetic wrong-ID grid is explicitly rejected.

No substantive discrepancy remains against the frozen primary analysis. The proposed design document contains exploratory recommendations; `110_PLAN.json` governs the implemented analysis. The 0.01 threshold is reported as an engineering effect-size flag, not a new significance claim or automatic publication decision.

## Reviewer sensitivity interface

Optional `--sensitivity-labels NAME=PATH` can be repeated. Each file uses the same joined 584-row schema as primary `labels.csv`, with identical condition/prompt/family keys and all eight feature columns. Binary cells use `0`/`1`; uncertainty can be `uncertain`, `?`, empty, `NA`, or `null`. The joint field must follow feline AND female AND coherent: any definite-negative conjunct determines zero; three positives determine one; otherwise it remains uncertain.

Sensitivity excludes uncertain labels separately for each feature, records confident/total coverage, and writes `sensitivity_NAME_scores.csv`. Compact summary includes its mean-session overall scores, source hash, and coverage. It does not impute uncertain cells, overwrite adjudicated primary labels, change the primary sample, or introduce another primary CI. With missing labels, sensitivity macro weights reflect available confident observations and need not equal the complete seven-feature primary grid; per-feature scores and coverage are therefore essential. These are reviewer-robustness diagnostics, not an alternative winning metric.

## Exact commands

CPU synthetic regression suite, including the full 584-label / 18,688-forecast grid:

```sh
python3 catgirl/composition/110_atlas/analyze_benchmark.py --self-test
```

After all masked reviews/adjudication are locked and joined, primary analysis:

```sh
python3 catgirl/composition/110_atlas/analyze_benchmark.py
```

Primary plus separately joined original reviewers (example filenames; create via the root's identity join only after locking):

```sh
python3 catgirl/composition/110_atlas/analyze_benchmark.py --sensitivity-labels reviewer1=catgirl/composition/110_atlas/reviewer1_labels.csv --sensitivity-labels reviewer2=catgirl/composition/110_atlas/reviewer2_labels.csv
```

The first command passed during this audit. The actual-data commands have **not** been run. No GPU process, source packets, locked forecasts, prompts, or runtime protocol were changed.
