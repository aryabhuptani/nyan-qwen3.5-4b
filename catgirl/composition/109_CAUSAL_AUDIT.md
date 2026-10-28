# Independent raw-data causal audit of 109

I independently recomputed these counts from [109_causal.jsonl](results/109_causal.jsonl), the frozen corpus, and the natural capture, without using `109_summary`. No GPU work or runner/data changes. The requested counts agree with the raw records.

## Completeness and pairing

There are **2,464 unique rows**, **14 conditions**, and **72 fresh cases**:60 core validation cases and12 lexical diagnostics. Every condition has176 rows:144 semantic rows (72 cases × both A/B orders) and32 English lexical rows. Thus totals are2,016 semantic and448 lexical rows. All expected condition/case/order combinations are present, with no duplicated keys or calibration cases scored causally. Semantic probabilities below were averaged over the two orders **within each case before** calculating paired intervention-minus-baseline differences. Four heldout families translated into three languages produce12 related own-accomplishment cases, not12 independent topic replications.

## Recomputed primary contrasts

| Endpoint | Paired mean difference | Direction counts |
|---|---:|---|
| Natural own-accomplishment minus neutral carrier mean | +0.0534043101 native units | Positive10/12 |
| Carrier q90, semantic own-accomplishment cases | −0.0006835551 probability (−0.068356 percentage points) | Positive3/12; negative5; tied4 |
| High-event carrier suppression, same semantic cases | −0.0007902800 probability (−0.079028 percentage points) | Negative4/12; positive3; tied5 |
| Output q90, English core proud-versus-pleased choice | +3.93576457 percentage points | Positive20/20 |
| Output q10, English core proud-versus-pleased choice | −4.70881430 percentage points | Negative16/20; tied4 |

All12 own-accomplishment suppression cases have eligible/changed target events. There are **23 changed carrier positions**, counted once per case; repeating the same carrier in the two semantic choice orders would double-count these events. Carrier q90 changes140 positions once per case. Natural association meets its descriptive10/12 count, but the prespecified causal semantic direction gates do not meet75%: q90 increases only3/12 and suppression decreases only4/12. The small observed semantic shifts do not establish absence of semantic contribution under other tasks, magnitudes, or intervention windows.

## Lexical baseline counterparts and specificity

For the20 English core cases, baseline conditional `P(proud | proud or pleased)` averages **0.7584763165**. Output q90 raises it to **0.7978339622**; output q10 lowers it to **0.7113881736**. These are pair-conditioned word-choice probabilities, not rates of spontaneous pride language.

| Core category, four cases each | Baseline conditional P(proud) | Output q90 change, percentage points | Output q10 change, percentage points |
|---|---:|---:|---:|
| Own accomplishment | 0.806398 | +1.613697 | −6.475017 |
| Neutral achievement | 0.761024 | +3.354051 | −9.495671 |
| Joy | 0.379256 | +10.730273 | −3.249156 |
| Confidence | 0.868497 | +3.225830 | −4.324228 |
| Arrogance | 0.977207 | +0.754973 | 0.000000 |

The output-position lexical effect also occurs on neutral/joy contexts; it is not restricted to appropriate accomplishment satisfaction. Arrogance's four cases account for the four q10 ties. The strong lexical result uses **20 English core cases**, while complete lexical coverage also includes12 separate English diagnostic cases; those denominators must not be interchanged.

## Probability mass and inference limits

The semantic endpoint normalizes only the A/B token pair. In the120 baseline core semantic rows, their **combined unconditional next-token mass** has minimum0.000310752, median0.012536383, mean0.054858940, and maximum0.319111640. For24 diagnostic semantic rows the median is0.001067976. Thus conditional A/B accuracy is not unconstrained answer-generation accuracy; much of the model's next-token probability lies elsewhere. Similarly, proud+pleased combined baseline mass in20 core lexical rows has median0.004366169 (mean0.005500197; range0.000172118–0.021067620). Apparent large pair-conditioned percentages need that qualification.

The three random controls have **carrier** q10/q90 operations only: there are **no matched random output-position controls**. Therefore the lexical output effect is a reproducible causal result for this coordinate and fixed task, but the study does not establish its specificity versus other native coordinates at that output position.

Carrier and output interventions use **different calibrated distributions and different positions**. Target carrier q10/q50/q90 are approximately−0.220898/−0.040283/+0.343945; lexical-output q10/q90 are+0.420703/+2.350000. These are replacements, not equal additive increments or norm-matched perturbations. A positive output-position lexical result and weak carrier semantic result therefore do not isolate “lexical rather than semantic” as a clean matched double dissociation. They support a narrower description: strong conditional lexical sensitivity under the tested final-position clamp, with the frozen natural-range carrier semantic gate unmet.

Retain the [implementation audit](109_IMPLEMENTATION_AUDIT.md) limits: small authored templates, four semantic families with related translations, category/length/syntax confounds, BF16 achieved-clamp precision, identity-only restoration check, and residual-norm-unmatched random controls. Neither language effects nor causal word-choice effects establish a natural uniquely semantic pride neuron, subjective feelings, or universal necessity.

Raw causal SHA256: `7769d3bd678cca7ace78882d874688c935143cde5b88235528fd716babe1b397`.
