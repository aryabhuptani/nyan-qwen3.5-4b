# Independent Study115 discovery selection audit

The independent reconstruction imports no frozen selector code. It verifies each label export against its hash in `labels_lock.json`, reads numeric feature cells only, checks all 60 unique condition/case rows, computes every feature and derived positive/zero/unknown count, and applies the frozen ordering using decimal dose arithmetic.

Every cell in the primary 20-state summary matches the independent reconstruction. Candidate, baseline, full ranking and confirmation decision also match.

The primary selected condition is Paws +0.1 / Genki +0.4. It has clean catgirl with sound 1/3, tied with base 1/3; clean catgirl is 3/3 versus base 2/3. Its idol-or-mascot count is 0/3, as for base. It therefore meets the frozen strict secondary-improvement rule and warrants the unchanged selected-dose confirmation. Discovery selection does not show improved sound frequency over base.

| Label export | Selected Paws / Genki | Clean with sound | Clean | Confirmation allowed |
|---|---|---:|---:|---|
| Adjudicated primary | +0.1 / +0.4 | 1/3 | 3/3 | Yes |
| Original reviewer A | +0.1 / +0.4 | 1/3 | 3/3 | Yes |
| Original reviewer B | +0.1 / +0.4 | 1/3 | 3/3 | Yes |
| Unknown cells forced zero | +0.1 / +0.8 | 2/3 | 3/3 | Yes |
| Unknown cells forced one | +0.1 / +0.4 | 1/3 | 3/3 | Yes |

The selected dose is stable across the two original reviewers, but changes when every unknown is forced to zero. That sensitivity can convert unknown idol/mascot identity into known absence, allowing additional clean cases. This is a descriptive robustness check; it does not replace the adjudicated, frozen primary choice.

All 20 states are preserved in `independent_discovery_summary.csv`; variant subdirectories contain their full summaries and rankings. `selection_sensitivities.json` records the comparison. The audit does not evaluate reply text or reviewer evidence and cannot establish reviewer independence or correctness from numeric labels alone.
