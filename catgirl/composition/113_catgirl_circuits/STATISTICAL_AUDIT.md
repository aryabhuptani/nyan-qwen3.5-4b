# Pre-outcome statistical audit

This CPU-only, blinded audit reviewed PROTOCOL.md, ANALYSIS_INTERFACE.md,
FEATURE_DEFINITIONS.json, analyze.py and confirmation_analysis.py. Synthetic
examples and both built-in selftests were used. No actual labels, heldout replies,
discovery outputs, numeric forecast probabilities or result tables were inspected.
The subsequent guard addition consulted frozen job/case metadata, packet_lock.json,
forecast_packets.py and only the header and distinct session/arm identifiers from
forecasts_normalized.csv. Frozen analysis scripts were not changed.

## Verified calculations

Paired percentile bootstrap intervals use 5,000 case-level draws and seed 113.
Exact two-sided sign flips include zero contrasts correctly. Canonical four-cell
contrasts use shared complete cases. All-background contrasts require all eight
backgrounds per case. Brier losses score each constituent binary case, average
within session before averaging sessions within arm, and retain separate endpoints.
The numeric practical screen correctly requires complete cases, at least 3/12 net
extra positives over each single and positives in at least two families. Reviewer
direction agreement is intentionally an external requirement. Confirmation uses
the canonical endpoint/cells and the same contrast definitions; its CSV and
missingness guards are stricter than those in the frozen main analyzer.

## Defects and limits in the frozen analyzer

- analyze.py:79–85 adjusts across only non-null p-values. If a primary endpoint
  is entirely untestable, this weakens the predeclared five-hypothesis family.
  Synthetic input [.01, null, null, null, null] yields .01 rather than .05.
  Complete five-endpoint analyses are unaffected.
- analyze.py:276–282 checks within-session coverage but does not enforce the
  planned sessions or a unique arm within each session. A complete one-session
  packet passes; an empty packet also passes. analyze.py:219–227 assigns each
  session/feature to its first scored row's arm, so inconsistent arms can silently
  misassign scores. The four LLM sessions and the additional deterministic
  additive session must be checked separately from the frozen implementation.
- analyze.py:43–44 silently maps missing feature columns to unknown labels.
  Missing headers cannot be distinguished from explicit uncertain judgments.
  confirmation_analysis.py:45–51 correctly rejects absent feature columns.
- analyze.py:155–158 omits families without complete primary four-cell cases.
  Confirmation preserves all families and incomplete denominators (159–167).
- analyze.py:127 supplies no interval for pair-minus-better-single. Confirmation
  correctly bootstraps that nonlinear statistic (98–112). No retrospective main
  interval was added by this audit.
- analyze.py:213–216 does not validate discovery-baseline bounds or finiteness.

## Separate guard and supplements

analysis_guard.py requires all 14 label headers and exactly the frozen 34-condition
by 12-case grid (408 rows), including both controls and exact case/family mapping.
Labels use the frozen analyzer's 0/1/unknown parser. Forecasts must have exactly
2,940 unique cells: 588 per session, with these metadata identifiers:

| Session | Arm |
|---|---|
| descriptions_1 | descriptions |
| descriptions_2 | descriptions |
| observations_only_1 | observations_only |
| observations_only_2 | observations_only |
| additive_singletons | deterministic_additive |

All forecast and optional seven-feature baseline probabilities must be finite
and within [0,1]. The guard does not assess whether a baseline was genuinely
frozen before validation; provenance remains a separate requirement.

```
python3 analysis_guard.py --selftest
python3 analysis_guard.py --labels LOCKED.csv --forecasts FORECASTS.csv
python3 analysis_guard.py --labels LOCKED.csv --forecasts FORECASTS.csv --out NEW_DIRECTORY
python3 analysis_guard.py --labels LOCKED.csv --forecasts FORECASTS.csv --analysis EXISTING_DIRECTORY
```

Optional --baseline-rates validates a frozen baseline and forwards it when running
the analyzer. --out validates before invoking the unchanged analyze.py and requires
a fresh directory. --analysis adds supplements to existing output. Both paths
preserve original tables and refuse to overwrite existing supplements:

- corrected_fixed_five_holm.csv retains the original adjusted p-value, supplies
  a fixed-five adjustment and explicitly flags discrepancies. Untestable hypotheses
  remain in the family (conservatively equivalent to p=1 for tested hypotheses),
  while their own adjusted results remain unknown.
- primary_family_missingness_supplement.csv preserves all six families for each
  primary pair, with complete/incomplete case and unknown-cell denominators.

These additions were prepared without examining study outcomes. They do not
change endpoints, practical thresholds, contrasts, selection or frozen tables.
The guard's synthetic selftest covers full valid inputs, missing headers, missing
sessions/rows, inconsistent arms, duplicate cells, nonfinite probabilities,
invalid baselines, fixed-five Holm, original-table preservation and families with
no complete cases. Existing built-in analyzer selftests also passed in this audit.
