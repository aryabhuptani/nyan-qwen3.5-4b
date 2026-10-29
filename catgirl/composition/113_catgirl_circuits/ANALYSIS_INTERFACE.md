# CPU analysis interface

Use Python3 standard library only. Inputs must be final locked labels, or immutable original reviewer labels for sensitivity. Run:

```
python3 catgirl/composition/113_catgirl_circuits/analyze.py --selftest
python3 catgirl/composition/113_catgirl_circuits/analyze.py --labels path/to/labels.csv --forecasts path/to/forecasts_normalized.csv --out path/to/analysis
```

Optional `--baseline-rates path/to/discovery_base_rates.json` supplies probabilities keyed by the seven forecast feature names. They must be frozen from discovery-base labels before validation; no fitting to heldout outcomes.

Label CSV requires case_id, condition_id, family_id and every feature in FEATURE_DEFINITIONS.json. Binary labels may be0/1; empty/null/none are unknown, never silently zero. Main conditions are five-character bitstrings in the frozen left-to-right Paws,Hugs,Genki,Sounds,Pride order. Additional condition IDs such as plain and positive_control_genki_hugs appear in state_rates but are excluded from factorial effects. Main grid must contain384 unique state-case rows,12 cases and six families of two; missing outputs are a failure to complete the planned experiment, not silently reduced samples.

Forecast CSV requires session_id, arm, family_id, condition_id, feature, probability. All588 rows per session must be present:14 states ×six families ×seven endpoints. Duplicate rows, missing coverage, invalid probabilities and forecasts with no observations fail. Each family prediction is scored against each individual reply; uncertainty labels are omitted with their scored denominator reported. Session losses are averaged before arm losses. Primary integrated endpoint losses remain separate from cat-role and coherence losses.

Tables:

- state_rates: every condition×14features, all-reply rate, unknown count implicit in n_total−n_scored, and coherent-only descriptive subset with denominator.
- singleton_targets: five intended singleton effects versus base, paired by case.
- primary_pairs: five canonical four-cell outcomes, differences, paired bootstrap intervals, exact sign-flip interaction tests and five-test Holm adjustment.
- primary_case_contrasts and primary_family_contrasts: transparent case/family decomposition of each primary effect.
- all_background_interactions: each named interaction averaged equally over eight other-modifier backgrounds within each case, then across complete cases. Requires all eight backgrounds for a case.
- larger_circuit_deletions: PHG,PHGS,11111 versus base and each immediate deletion, for every feature. These are predeclared secondary comparisons without multiplicity-corrected discovery claims.
- forecast_case_scores, forecast_session_scores, forecast_arm_scores: Brier losses, never an ensemble probability score. Optional discovery-base rate comparison is recorded per case/session.

The practical screen is first evaluated numerically (≥3/12 pair-over-each-single extra positives and at least two positive families). The final screen also requires both original reviewers to show the same positive pair-over-each-single direction; the analyzer deliberately labels its automatic column practical_screen_without_reviewer_check. Independently compare those reviewer analyses before stating the screen passed. Passing is an invitation for fresh confirmation, not proof of synergy or generality. An observed positive interaction and a positive pair-versus-single effect are distinct quantities.

The exact sign-flip test uses per-case interaction signs and is two-sided; its exchangeability assumption and the convenience prompt sample limit inference. Bootstrap draws are paired prompt-cluster draws (5,000; seed113), not independent condition draws. All-zero data can produce zero-width intervals and still fail to establish absence outside the tested prompts.

Protocol changes and source/hash freezes must be recorded before heldout inference. Analyzer changes after data are visible must be disclosed, justified and preserve the original outputs; no favorable endpoint or threshold may replace the frozen ones.
