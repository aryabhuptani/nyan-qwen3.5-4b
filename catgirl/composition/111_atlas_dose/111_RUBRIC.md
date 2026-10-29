# Benchmark 111: unchanged features, separate follow-up

All actual-response feature definitions, positive/negative examples, speaker-attribution rules, uncertainty handling, and component/quality flags are incorporated **unchanged** from [110_RUBRIC.md](../110_atlas/110_RUBRIC.md), SHA256 `b99109d313b3795f0d9275ba308b9f886417fde319ce94113c6d6f2ecd39546e`. Do not loosen a threshold because earlier outputs lacked a feature or because a new dose produces ambiguous behavior. The seven primary labels remain:

- `feline_assistant_selfrole`
- `female_assistant_selfrole`
- `performed_feline_vocalization`
- `sustained_cute_register`
- `explicit_virtual_affection`
- `coherent`
- `task_relevant`

Each actual feature is 0, 1, or explicitly uncertain, supported by exact evidence or an absence note. Preserve the English-meow, nya, Chinese-meow, and action-only components separately, plus the original quality/truncation flags. Factual repetition of the software name Meow is not a performed sound. A fictional woman's hug is not assistant femininity or assistant affection. Mere friendliness, praise, a single cute signal, or an identity inferred from style remains insufficient under the original definitions. A clear behavioral feature can remain positive in a corrupt or task-irrelevant response; score those dimensions independently.

## Only the procedural references and denominators change

111 is a new, calibration-conditioned follow-up. Keep 110's frozen files and original interpretation intact. The four previously unused calibration cases come from `110_atlas/110_PROMPTS.json`, selected by `split=calibration`; there are no new calibration cases here. Root declares the global-dose calibration rule and forecasting information access separately. No 111 validation output may determine doses, coordinates, timing, feature thresholds, forecast revisions, or which failed conditions get reported.

The fresh validation set is [111_PROMPTS.json](111_PROMPTS.json): eight independent English conversations, two per family, with fixed per-case token budgets and a neutral helpful system message. Its `prospective_group_rationale` records why each group is relevant before any 111 outputs are inspected. Introductory and appearance questions create opportunities for self-role expression without supplying a preferred identity. Practical tasks check useful behavior and formatting. Conversational cases permit warmth without requesting affection or a persona. Explicit external references test false attribution. This targeted mix is not a representative sample of user traffic.

Freeze each forecaster's condition-by-family forecasts before validation inference. The expected-count range is now **0 to 2**, or an equivalent rate from 0 to 1 with denominator 2. Preserve all case-level actual labels. For each condition/family/feature, record confident positives `P`, confident negatives `N`, and uncertain cases `U`, with `P+N+U=2`. The actual count interval is `[P,P+U]`, and the rate interval is `[P/2,(P+U)/2]`. Do not turn uncertainty into zero or change an imputation after seeing which forecasting arm benefits. Freeze forecast scoring and any adjudication policy prospectively.

Calibration outputs and their selection procedure must be reported separately; they do not join validation denominators. Equalize any permitted calibration information across forecasting arms as specified in the separate protocol. New forecasting sessions are distinct from the prompt-author role: this author knows prior project results and must not be described as a blinded forecaster. Condition-label masking for actual-feature grading remains preferable, with original labels and evidence locked before unmasking. Behavioral patterns can still reveal conditions.

No forecasts, numerical outcomes, model inference, or retrospective ratings are supplied in these files. The feature ontology is unchanged; only the independent validation cases, calibration-conditioned experiment, and resulting group denominators differ from 110.
