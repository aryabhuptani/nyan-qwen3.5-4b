# Frozen Study114 analysis interface

`analyze_replication.py` uses only Python's standard library. It reads no generation text, makes no inference calls, and requires no GPU. Its primary endpoint is `performed_feline_vocalization`: assistant-own performed meow/mew/nya/purr or clear language equivalent, as defined by the unchanged Study113 review rubric. Capability is not a gate. Coherence and adopted cat/female roles remain descriptive features.

Run the synthetic test before inference:

```sh
python3 catgirl/composition/114_paws_genki/analyze_replication.py --self-test
```

Analyze one independent label export at a time:

```sh
python3 catgirl/composition/114_paws_genki/analyze_replication.py \
  --labels LABELS/labels.csv --out-dir ANALYSIS/adjudicated
```

Defaults are this directory's `CASES_validation.json` and `JOB_validation.json`; `--cases` and `--job` can override them. Repeat separately for `labels_reviewer_A.csv`, `labels_reviewer_B.csv`, `labels_uncertain_zero.csv`, and `labels_uncertain_one.csv`, with distinct output directories. The review utility preserves original reviews, evidence, adjudication and uncertainty exports; the analyzer never changes any labels. Outputs may be overwritten within the explicitly selected output directory.

## Input contract

Cases must supply twelve unique case IDs in six families of two cases each, all `validation`. The job must supply exactly eight unique `condition_id` values: `00000` (base), `10000` (Paws), `00100` (Genki), `10100` (Paws+Genki), and `random_1` through `random_4` (four norm-matched random pairs). Dose and norm matching are defined and audited upstream; this analyzer checks the registered condition IDs and case-family mapping.

Labels CSV requires `condition_id,case_id,family_id` and every feature in `FEATURE_DEFINITIONS.json`. Exactly the 96 distinct condition/case combinations must exist. Optional headers are the review utility's `item_id,split,text_sha256,lexical_body,lexical_affection,lexical_feline_sound,lexical_pride,adjudicated`. No other, duplicate, or missing headers are accepted; malformed row widths and incorrect families/splits are rejected. Labels accept literal `0`, `1`, empty, or case-insensitive `null`. Empty/null means unknown and is never converted to zero. Every feature is validated even if it is not the primary endpoint. Optional metadata do not change estimands.

## Estimands and missingness

`feature_rates.csv` reports positives, known denominators, unknown counts, and rates for all fourteen features, eight conditions, overall and all six families. There is no coherence filter.

`case_contrasts.csv` retains every case, including incomplete cases, raw primary-endpoint cell labels, completeness flags, and missing contrasts as empty cells. Canonical contrasts require all four cells known for that case, even if a contrast algebraically uses only two cells. With B, P, G, PG denoting endpoint labels:

- Interaction: PG − P − G + B.
- Pair minus Paws, Genki, and base: PG − P, PG − G, PG − B.
- `pair_minus_case_better_single`: PG − max(P,G) per case, descriptive.
- `pair_minus_better_single`: mean(PG) − max(mean(P),mean(G)) on the same four-cell complete cohort, reported separately in `contrast_summary.csv`. Bootstrap resamples whole case records and recomputes the maximum each replicate. This aggregate estimand differs from the mean per-case better-single contrast.

Specificity is PG − mean(random_1,…,random_4) within case. All five labels must be known. Missing any random label excludes the case from this contrast; the other random conditions are never reweighted. Specificity completeness is independent of canonical completeness. Each individual PG−random_j contrast uses its own two-known-label cohort and is descriptive. All individual random condition rates and deltas are retained.

`contrast_summary.csv` reports these contrasts overall and for all six families, including families with zero complete cases, with `n_total,n_complete,n_missing`. Canonical cohort cell rates are also reported. All intervals are 95% percentile intervals from 5,000 paired case bootstrap draws using seed 114, linear interpolation at quantiles .025/.975. Each interval restarts the seeded generator; family intervals resample within family. No interval is produced for zero complete cases. With two cases per family, family intervals are highly discrete and descriptive.

## Planned tests

Exactly two tests are retained in `planned_tests.csv`: canonical endpoint interaction and endpoint PG-minus-mean-random specificity. Each is an exact two-sided sign-flip test of absolute summed paired case contrasts, enumerating all 2^n sign assignments, including zero contrasts. Empty cohorts produce null p-values. These tests rely on the sign-exchangeability null for the case contrasts; they do not establish broad population generalization.

Holm adjustment always uses the fixed planned family size two. A null/unestimable test keeps its slot: the lone available p-value is multiplied by two. With two observed p-values, sorted multipliers are two and one with monotonic adjustment. No other feature, family, individual random pair, identity, capability, or contrast enters this confirmatory family. All other rates/contrasts/intervals are descriptive. These tests do not gate on capability.

## Optional condition forecasts

`--forecasts FILE.csv` accepts exactly the ordered headers `condition_id,probability`, exactly eight unique registered conditions, and finite probabilities in [0,1]. Each condition probability is fixed across all twelve cases. Only the primary endpoint receives a Brier score. `forecast_case_scores.csv` retains unknown outcomes with null scores; `forecast_condition_scores.csv` averages known case scores within condition with known/unknown counts. These are condition forecasts and make no claim about agent arms or forecast-session independence.

`analysis_summary.json` retains all tables and analysis parameters. JSON null and empty CSV fields preserve missingness throughout. The original A/B, adjudicated and uncertainty analyses must be retained and compared using separate invocations; none is silently substituted for another.
