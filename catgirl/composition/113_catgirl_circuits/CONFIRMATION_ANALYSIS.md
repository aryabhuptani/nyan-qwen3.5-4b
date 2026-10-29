# Reserved four-cell confirmation analysis

This separate CPU-only, standard-library helper analyzes the one canonical pair chosen by the frozen protocol. It does not select a pair, rerun the validation screen, introduce a confirmation threshold, or change the frozen factorial analyzer.

```sh
python3 catgirl/composition/113_catgirl_circuits/confirmation_analysis.py --selftest
python3 catgirl/composition/113_catgirl_circuits/confirmation_analysis.py \
  --labels path/to/locked_confirmation_labels.csv \
  --selection path/to/selection.json \
  --out path/to/new_confirmation_analysis
```

The selection JSON must contain exactly `{"pair_id": "paws_hugs"}` (replace the value with the selected canonical ID). Allowed IDs: `paws_hugs`, `sounds_genki`, `sounds_hugs`, `genki_hugs`, `pride_hugs`. Keep selection rationale and reviewer-direction evidence in a separate provenance record. The helper derives the integrated primary endpoint, modifier ordering, and four condition bitstrings from `FEATURE_DEFINITIONS.json`; endpoint overrides are rejected. In `y10` modifier A is the first frozen `modifier_indices` entry, and in `y01` modifier B is the second.

Labels require `case_id,condition_id,family_id` and all 14 frozen feature columns. Exactly 48 unique case-condition rows must cover the selected four states on 12 cases, in six consistent families of two cases. Missing outputs, extra states, duplicate rows, missing feature columns and nonbinary labels fail. Labels `0/1` are accepted; empty, `null`, `none`, `nan` and `uncertain` remain unknown. The output directory must be new.

Outputs:

- `state_rates.csv`: every state × feature, positive/scored/unknown/total counts, all-reply rates and coherent-only descriptive rates with their denominators.
- `four_cell_contrasts.csv`: all features, primary endpoint flagged, paired complete-case cell means (`y00,y10,y01,y11`), pair minus each single, pair minus base, pair minus better single, and interaction. Cell means here use the same complete cases across all four cells; `state_rates.csv` uses each state's available labels.
- `case_contrasts.csv`: all cases × features, retaining incomplete cases and their unknown cells, with paired contrasts where complete.
- `family_contrasts.csv`: all six families × features with complete/incomplete denominators and paired mean contrasts, even when a family has no complete cases.
- `analysis_summary.json`: canonical selection, endpoint, table sizes, resampling settings and interpretation caveats.

Percentile 95% intervals use 5,000 paired case draws with seed 113, separately per feature, from that feature's complete cases. Every bootstrap draw resamples whole four-cell case records; the better-single statistic recomputes the maximum of the two sampled single means. The interaction test exactly enumerates two-sided sign flips of the complete per-case interactions, including zeros. No Holm correction is applied: this is one selected, reserved primary confirmation. Tests for other features are explicitly descriptive and unadjusted. No inference is returned when all cases are unknown for a feature.

Unknowns are never silently zero. Complete-case omissions can change the estimand; report the counts. Coherent-only rates are selected descriptive subsets. Sign exchangeability and the convenience prompt sample limit inference; zero-width intervals on all-zero data do not establish absence elsewhere. Run separate output directories for final locked labels and each immutable original masked reviewer label set. Confirmation stays separate from the validation screen, including failures and reviewer sensitivity.
