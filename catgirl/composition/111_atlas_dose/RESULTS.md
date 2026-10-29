# Dose calibration failed its predeclared feasibility gate

**No tested global dose qualified. No study111 forecasting sessions or confirmatory validation were run.** This is an exploratory calibration result following the inconclusive [mild-dose study110](../110_atlas/RESULTS.md), which remains unchanged.

## ELI5, without an analogy

The first experiment changed the neurons only mildly and produced almost none of the target personality behaviors. We next tested stronger changes on four different questions reserved for calibration. Before generating those answers, we wrote a rule: choose the lowest strength that makes several settings show target traits while keeping at least nine out of ten answers understandable.

Stronger changes produced more cute language or affection, but also many broken and repetitive answers. None passed the rule. A second reviewer, who did not know the strengths or the first reviewer's scores, reached almost the same counts. We stopped before making new predictions or testing the new validation questions. We did not change the rule after seeing the failure.

## Frozen rule and observed results

The plan was committed and pushed in `66707f1` before calibration inference. It tests six familiar single-neuron ingredients and four previously tested two-neuron compositions, using exactly the four unused calibration prompts from110. Each global multiplier applies to the **original published signed reference**, not the110 half-dose. All offsets act on every prompt and answer position, under the original pinned BF16 model, public normalization, greedy decoding and unchanged feature definitions. Thirty settings × four prompts yield120 replies.

A qualifying level needed all three: at least4/10 settings with any confidently observed persona feature, at least two distinct persona-feature types, and at least36/40 coherent replies. Choose the lowest passing multiplier. Uncertainty counts conservatively zero for selection; primary labels remain unchanged. A diagnostic fallback cannot permit confirmation.

| Global multiplier of published reference | Settings with a target trait | Coherent replies | Hard loops | Task-relevant replies |
|---|---:|---:|---:|---:|
| 1.0× | 3/10 | 31/40 | 7/40 | 11/40 |
| 1.5× | 5/10 | 19/40 | 17/40 | 2/40 |
| 2.0× | 5/10 | 12/40 | 23/40 | 0/40 |

The observed persona types were **sustained cute style and explicit affection**. No feline identity, female identity or performed feline vocalization was confidently observed by the calibration reviewer. The gate permits traits in different answers; it never establishes a coherent catgirl personality. Task relevance was measured but was not a selection gate. Clear trait positives were retained even in broken outputs.

The primary rule selected `stop_no_coherent_dose`, with no multiplier and `confirmatory_allowed=false`. Full counts and original hashes are in [dose_selection.json](dose_selection.json). An [independent calculation](calibration_independent_audit.json) reproduced the complete decision and verified the120-row grid and raw-output hash.

## Reviewer sensitivity

The preregistered selection used one fresh condition-label-masked calibration reviewer. After that decision was fixed, a second fresh masked reviewer scored the same120 replies independently, without seeing doses, settings or the first ratings. This is **post-selection sensitivity**, not a replacement selection procedure or a human gold standard.

The second reviewer counted coherence31/40,18/40,12/40; settings with target traits3/10,5/10,5/10; hard loops7/40,16/40,23/40; and task relevance11/40,2/40,0/40. No level would have passed. Original uncertainty is preserved in both [primary](calibration_primary_observations.csv) and [sensitivity](calibration_sensitivity_observations.csv) scored-output CSVs. [Machine-readable sensitivity](calibration_review_sensitivity.json) records both sets and their hashes. Shared agent-model/rubric errors remain possible.

## What this establishes and what it does not

This is evidence that **one global multiplier across these ten particular steering settings failed our feasibility policy**. It is not a finding that no useful dose exists for any individual neuron. The90% coherence gate is an engineering choice, not a definition of scientific validity or native-neuron cleanliness. Heterogeneous coordinates can require different scales, and published additive reference doses are not calibrated natural activation percentiles. Four purpose-built English prompts and ten familiar settings are not representative samples of neurons, conversations or users.

Together,110 and111 show why a prospective atlas test needs an informative dose and endpoint before confirmation. The first study barely elicited the target behaviors; the stronger shared-dose policy elicited some styles while substantially degrading output. These results **do not validate or refute Goodfire's broader Already Clean hypothesis** ([original essay](https://already-clean.goodfire.com/)). They do not identify natural semantic functions, explain a thought process, measure a cleanliness fraction, or establish superposition. The earlier four-coordinate catgirl showcase uses a different recipe and schedule; this calibration does not re-evaluate that exact demonstration.

No new atlas accuracy claim can be made from111: its prospective confirmatory phase was blocked. Its authored fresh validation questions remain unused. Any future per-neuron calibration would require its own explicit discovery procedure, new locked predictions and independently held-out validation; it cannot retroactively rescue these studies.

**The image/tweet gate was not met. No positive-result share graphic was produced.** A transparent report of an inconclusive forecast pilot and a failed dose policy is supported; a headline claiming validation or refutation of native cleanliness is not.

## Key artifacts and reproduction

- [Pre-inference calibration plan](111_CALIBRATION_PLAN.json), [fixed masking/selection code](mask_calibration.py), and [primary decision](dose_selection.json).
- [Every calibration reply and primary score](calibration_primary_observations.csv), [second-review scores](calibration_sensitivity_observations.csv), and the original compact rating JSONs.
- [Independent mathematical/provenance verification](calibration_independent_audit.json).
- `calibration_inputs.zip` contains the exact job whose hash is pinned in the plan. It contains no raw catalog snapshot or agent transcript. Decode the original model using the pinned [environment](../110_atlas/environment.json).

```sh
python3 catgirl/composition/111_atlas_dose/verify_calibration.py
# A new GPU replication requires a separate study copy with clean output/decision paths:
python3 -m zipfile -e catgirl/composition/111_atlas_dose/calibration_inputs.zip catgirl/composition/111_atlas_dose
env HF_HOME=/path/to/model-cache .venv/bin/python catgirl/composition_worker.py \
  --model /path/to/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --job catgirl/composition/111_atlas_dose/job.local.json \
  --results-dir catgirl/composition/111_atlas_dose/results.local
python3 catgirl/composition/111_atlas_dose/mask_calibration.py \
  --raw catgirl/composition/111_atlas_dose/results.local/job.local.jsonl
# Obtain a fresh masked review using CALIBRATION_REVIEW.md, then:
python3 catgirl/composition/111_atlas_dose/mask_calibration.py \
  --ratings catgirl/composition/111_atlas_dose/calibration_ratings.local.json
python3 catgirl/composition/111_atlas_dose/export_calibration.py
```

`verify_calibration.py` reproduces all reported counts and the frozen stop directly from public CSVs, without model weights or private runtime files. Existing decisions are immutable: a new GPU replication needs a separate study copy initialized with the frozen plan and job, without overwriting the published decision. The raw hash includes timing metadata and is not expected to match across replications. The primary and second rating bytes are preserved publicly as `calibration_primary_ratings.json` and `calibration_sensitivity_ratings.json`. Full agent transcripts and raw runtime records remain local; complete reply text is in the compact CSVs. The follow-up source-code harness was prepared but not executed and is not a completed validation result.
