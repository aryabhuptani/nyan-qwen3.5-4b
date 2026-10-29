# Result: the mild-dose pilot did not meaningfully test catgirl prediction

All 584 preregistered replies were generated and scored. **No reply adopted a feline self-role, performed a feline vocalization, sustained cute style, or expressed personal affection under the frozen rubric.** Two replies explicitly used a female self-role; none met the feline-and-female-and-coherent joint endpoint. The study therefore does not establish that an atlas predicts catgirl behavior or composition.

## ELI5, without an analogy

We gave separate agents different information about the same neurons. Some received Neuronary's real descriptions. Others received incorrect descriptions, lists of related vocabulary, or no neuron descriptions. They predicted what would happen before we changed the model. We then changed the neurons and generated 584 answers to eight questions. Separate agents scored the answers without seeing the neuron settings or predictions.

The changes were mild: half the published dose for each neuron. Almost no target personality behavior appeared. Agents with the correct descriptions made slightly better predictions overall, mainly about whether answers stayed understandable and completed the task. The difference was too small to meet our rule for an interesting result. We learned that this mild-dose test cannot answer our main catgirl question. We must preserve this finding rather than call it evidence that the descriptions work or do not work in general.

## Frozen primary comparison

Lower Brier score means smaller squared probability error. These scores average the two forecasting sessions' separate losses across seven features, both signs, eight prompts and **24 random published-catalog coordinates**. They do not average forecast probabilities first. The six familiar ingredients are separate.

| Prediction information | Random-coordinate Brier |
|---|---:|
| Correct descriptions + correct projections | 0.020969 |
| Shuffled descriptions + correct projections | 0.022082 |
| Correct projections only | 0.022564 |
| Prompts, without neuron semantics | 0.023588 |

| Atlas minus control | Difference | Paired coordinate-bootstrap 95% interval |
|---|---:|---:|
| Shuffled | −0.001113 | [−0.002125, −0.000230] |
| Projection only | −0.001595 | [−0.003588, +0.000209] |
| Prompts only | −0.002619 | [−0.004606, −0.000912] |

All three point estimates favored descriptions in both forecasting sessions. Every comparison missed the predeclared **0.01 absolute Brier reduction** threshold. The projection contrast's interval included zero. The intervals condition on the fixed prompts, source catalog, doses, forecaster model and labeling procedure; they do not establish model-wide neuron interpretability. No multiple-comparison-adjusted significance claim is made.

All five persona-related features had **zero positives in the 384 random-coordinate replies**. Consequently, this comparison mostly scores correct negative predictions and answer-quality forecasts. Class prevalence, positive/negative losses, constant predictors, separate sessions and family/sign diagnostics are included in [scores.csv](scores.csv).

## Composition and observed behavior

Across all 584 replies, 576 were coherent and 479 fulfilled their task. The corresponding counts were 379/384 and 321/384 for the random-coordinate stratum. These are rubric judgments about short replies, not general capability evaluations.

For the twelve fixed pairs, atlas Brier was **0.048266**, versus **0.043704** for prompts without neuron-specific semantics. Direct atlas pair predictions were slightly worse than its fixed singles-based comparator: 0.048266 versus 0.047521. The pairs share familiar anchors; these are descriptive benchmark scores, not twelve independent composition discoveries. No catgirl composition success was demonstrated at these doses.

## Review and verification

Every reply received two fresh condition-label-masked agent reviews. There were 22 disputed/uncertain primary feature cells across 21 replies, out of 4,088 primary cells. A third fresh masked reviewer adjudicated these before labels were joined to condition identities. Original ratings, evidence and uncertainty remain in [label_evidence.json](label_evidence.json). Agreement alone does not validate accuracy, especially with rare positives; same-model agent judges can share errors. There is no independent human gold standard.

Separate sensitivity analyses using each original review, and bounding original uncertainty at zero/one, retained the small random-coordinate advantage and did not rescue the practical threshold. Reviewer-specific scores exclude their uncertain cells and explicitly report coverage, so those diagnostic means need not equal complete-grid means. Meowing-component disagreements were retained but were not separately adjudicated when their primary union agreed.

An independent stdlib implementation reproduced all 64 macro score cells and all twelve random-coordinate bootstrap intervals within 1.7×10⁻¹⁷; see [verification](110_INDEPENDENT_VERIFICATION.json). The scoring and review pipeline passed synthetic complete-grid tests. Forecasts were publicly backed up before generation; labels were locked before primary analysis.

## Scientific meaning and relation to Goodfire

Goodfire argues that studying native neurons can yield useful predictive explanations ([original essay](https://already-clean.goodfire.com/)). Our prospective controls test a narrow consequence: whether published words add predictive value for particular interventions on fresh prompts. This pilot finds a small accuracy benefit on unfamiliar published coordinates, but it **does not provide informative evidence about catgirl behavior**, because the intended behaviors barely occurred.

It neither confirms nor refutes native-neuron cleanliness, a model-wide cleanliness percentage, monosemanticity, recursive generalization during training, or absence of superposition. Published steering descriptions, doses outside measured native ranges, a curated catalog, eight English prompts and one forecaster model substantially limit generalization. The intervention experiment does not read a model's thoughts or establish internal feelings.

## Decision and follow-up

**The preregistered tweet/image gate failed. No celebratory output image is justified for this pilot.** Preserve it as an inconclusive mild-dose result.

A separately labeled study111 will use independent calibration prompts to choose one global dose under a frozen feasibility rule, then lock fresh forecasts before new validation prompts and newly held-out pairs. Study110's outcomes and thresholds remain unchanged. This is an explicit follow-up motivated by a failed feasibility assumption, not a replacement of the original result.

## Reproduction artifacts

- [Complete scored replies](observations.csv), all 584, including output text, conditions, labels and token counts.
- [Frozen protocol and methods](README.md), [original source words](selected30_atlas.json), [locked forecasts](locked_forecasts.csv), and [forecast lock](forecast_lock.json).
- [Primary/sensitivity labels and hashes](labels_lock.json), [compact machine-readable analysis](compact_summary.json), and [per-coordinate differences](perunit_differences.csv).
- [Pinned environment](environment.json) and [runtime completion/hash](runtime_completion.json). The laptop GPU generated 15,209 output tokens in 448.6 seconds; model loading was outside that timer.

The small reproduction archive contains exact predictor packets and runtime inputs. Raw mutable-site snapshots and full agent transcripts remain local with recorded hashes. See README for restore/run/review commands.
