# Paws + Genki increased feline sounds in a fresh test, with important limits

## ELI5, without a metaphor

We found an unexpected result in Study113: adding the neurons called Paws and Genki together made the model produce many more feline sounds than either addition alone. We then kept those settings fixed, wrote12 new questions, and tested again. We also tested four random pairs of neurons with matching requested perturbation strengths in the same layers. Reviewers judged the answers without knowing their settings.

The direction repeated, but the effect was smaller. Paws+Genki produced feline sounds in7/12 fresh answers, versus1/12 for Paws and3/12 for Genki. The random pairs produced0,0,1,0 out of12. This supports a specific behavioral effect worth investigating. It does not establish that the two named meanings combine into a general catgirl personality: explicit female identity was1/12, and the stronger test for more-than-additive interaction remained inconclusive.

## Selection, freezing and exact settings

This is a separately registered, post-selection replication. Study113's five primary hypotheses all failed. Paws+Genki was selected after inspecting its full grid, where vocalizations were11/12 versus base1/12, Paws2/12 and Genki1/12. It was not one of113's14 forecast states. We did not repurpose113's reserved confirmation prompts. [Fresh cases](CASES_validation.json), [methods](PROTOCOL.md), [sources](SOURCES.json), random draws, predictions and analysis were sealed/pushed in commit `001aa22` before114 inference.

Original BF16 `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Fixed base: L9N9009+.7, L26N6173+.3, L26N8930+.7, L22N8295−.4; the female coordinate uses the original prefill/first16 answer-prediction schedule. Paws L23N4769+.4 and Genki L13N4810+.8 stay active throughout input/output. The pair therefore uses **six distinct coordinates total**, not two on an unsteered model. No additional Animal sounds coordinate is used. Doses use the original published residual/column normalization and were not changed after113.

Greedy generation, seed42, max192 tokens, repetition1.1/window128. Eight conditions×12 prompts=96 replies and17,729 output tokens. Inputs were48–51 tokens. These are independent short replies, not a test of persistence, long context, other languages or other models. Capability was not a gate.

Four random pairs were drawn prospectively with seed114: L23 [3947,1603,5511,3989] paired respectively with L13 [781,2515,8498,8694]. Each receives the same requested per-layer normalized dose, timing and base. This matches requested residual perturbation norms through the existing column normalization; it does not match natural activation distributions, meanings or downstream nonlinear effects. Four controls do not characterize every random pair.

## Fresh results

The primary endpoint is assistant-performed feline vocalization: mew/nya/meow/purr or an equivalent, excluding descriptions and quoted characters. It is broader than the literal word “meow.” All primary adjudicated labels are known.

| Setting, always on the same base | Vocalization | Coherent | Explicit female identity |
|---|---:|---:|---:|
| Base | 2/12 | 12/12 | 7/11 scored, one unknown |
| + Paws | 1/12 | 12/12 | 5/12 |
| + Genki | 3/12 | 10/12 | 6/12 |
| + Both | **7/12** | 11/12 | **1/12** |
| + Random pair1 | 0/12 | 9/12 | 4/12 |
| + Random pair2 | 0/12 | 10/12 | 0/12 |
| + Random pair3 | 1/12 | 7/12 | 1/10 scored, two unknown |
| + Random pair4 | 0/12 | 11/12 | 12/12 |

All seven positive pair replies were coherent under the rubric, and positives span five of six families. “Coherent” means interpretable communication, not factual accuracy or task success. Some responses use repetitive character introductions or fanciful explanations. Positive vocalization does not imply a stable self-role, emotion or useful answer. The very different identity effects of random controls also show why a sound effect alone does not establish persona specificity.

Pair−Paws is6/12; pair−Genki is4/12; pair−base is5/12. Both original reviewers give positive pair-over-each-single direction, meeting the descriptive practical-size criterion. The paired95% bootstrap interval for pair−Genki is [−1/12,+9/12], so sampling uncertainty remains substantial.

| Planned contrast | Estimate | Paired bootstrap95% interval | Exact two-sided p | Fixed Holm2 p |
|---|---:|---|---:|---:|
| Interaction: pair−Paws−Genki+base | +.4167 | [−.0833,.9167] | .234375 | .234375 |
| Pair−mean of four random pairs | +.5625 | [.2917,.8130] | .015625 | .03125 |

The first test does not establish superadditivity. The adjudicated random-control contrast passes.05 after the two-test correction, **but that significance is not robust to reviewer uncertainty**: Holm p=.0625 in both original reviews and the uncertain-as-zero sensitivity. Uncertain-as-one gives.00390625. The correct conclusion is a repeated directional effect with reviewer-sensitive statistical strength, not a definitive mechanistic discovery. The sign-flip null requires sign-exchangeability; selected convenience prompts limit population interpretation.

## Predictions, reviewers and independent checks

Predictions were simple fixed extrapolations from113, not new description-only agent forecasts. Jeffreys-smoothed probabilities for base/Paws/Genki/pair were.1154/.1923/.1154/.8846; each random pair used.1154. The observed pair rate was.5833, so the prediction was overconfident (pair Brier.3338). Correct directional prediction is different from accurate calibration. Study113's prospective source-description comparison is reported separately, including its failures.

Two fresh masked reviewers rated every output with independently shuffled order.75 of1,344 feature judgments in44 replies went to a fresh third adjudicator. Primary labels are complete after adjudication; ten other cells remain unknown (six cat identity, three female identity, one own-body). Original sound positives for the pair are7 and6, with one and four uncertain respectively. The uncertainty-bound count is6–10/12; this is a sensitivity range, not a confidence interval. All original judgments and evidence are retained locally. Shared agent/rubric errors remain possible; these are not human gold ratings.

The independent verifier imports no primary numeric functions. It authenticated label locks and frozen metadata and matched rates, complete-case/family contrasts, all bootstrap intervals, exact integer-scaled sign flips, fixed-Holm2 and Brier scores:7,996 comparisons per mode,39,980 across five modes, zero discrepancies at1e−12. See [audit](audit/). Numerical agreement does not validate the semantic judgments or assumptions.

## What this teaches us

Named native-coordinate interventions can have a repeatable combined output effect, including one not obvious from their labels. Increasing the number of coordinates did not preserve every desired trait: more sounds coexisted with weaker explicit female identity. Simple label-based composition predictions mostly failed in113. These results favor testing each composition and its deletions directly over assuming a label specifies a portable behavior.

This is related to Goodfire's emphasis on forming explanations and testing concrete predictions of native mechanisms ([essay](https://already-clean.goodfire.com/)). It neither validates nor refutes model-wide cleanliness or recursive generalization. We did not establish natural activation selectivity, necessity within natural ranges, a minimal connected circuit, a full atlas, or a thought-process decoder. A useful next test would identify which downstream computations mediate the Paws/Genki effect and whether suppressing the corresponding natural activity reduces spontaneous vocalization.

## Assets and reproduction

[Numeric labels](numeric_results/) and [all primary tables](analysis/) allow CPU reproduction. The [SVG](share/paws_genki_fresh.svg), [PNG](share/paws_genki_fresh.png) and [image provenance](share/IMAGE_PROVENANCE.json) contain numeric results, not a selected generated quote. Both original reviewer analyses and uncertainty analyses remain in parallel folders. The full reply/evidence/raw-trace bundle is retained locally pending explicit publication approval after automatic review rejected its public upload. A numeric-only reproduction does not permit inspecting semantic judgments; full inference remains reproducible with the pinned model and frozen job.

Clean-directory recovery verified134 archived files and reproduced all30 analysis CSVs byte-for-byte from public numeric labels, including five independent replay audits. [Replay record](recovery/CPU_REPLAY.json). The complete local archive SHA-256 is `e500e513bb666bef87aa4e4d1029a353634b995b39bd975c2e9a970a8292f571`; its public upload remains pending.

An updated [chat-example PNG](share/paws_genki_chat.png) / [editable SVG](share/paws_genki_chat.svg) shows an exact selected reply prefix and explicitly labels the two added coordinates over the four-neuron base. [Selected-excerpt provenance](share/CHAT_IMAGE_PROVENANCE.json) preserves the prompt and exact text. This image is illustrative; aggregate results above remain unchanged.
