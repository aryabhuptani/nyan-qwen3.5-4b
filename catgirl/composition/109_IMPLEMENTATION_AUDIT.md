# Independent CPU audit of frozen 109 implementation

I found no blocking channel-identity, semantic-label, calibration-split, or intervention-window error in the inspected implementation. This is a source/data audit, not an independent execution of the model. I did not load MLX or the model, run GPU work, change the runner/corpus, or read/interpret the growing causal output. The observations below concern the completed natural capture and calibration only.

## Verified implementation and provenance

The [frozen plan](109_PRIDE_PLAN.json) specifies original BF16 Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, native zero-based L22 N1911, and controls N4581/N3753/N7525 in that same layer. The [script](109_pride_probe.py) enforces the snapshot directory name and BF16 unquantized gate/up/down projections, wraps only layer22, and computes `silu(gate_proj(x))*up_proj(x)` before down-projection. Snapshot directory/config checks establish intended provenance, not an independent cryptographic verification of every weight file.

I independently reproduced the control list using `random.Random(109).sample([i for i in range(9216) if i != 1911], 3)`. The corpus SHA matches the frozen plan. All 132 unique cases have one completed natural record and valid finite `[prompt_tokens,4]` pre/post arrays, with exact `pre==post`. Channel order is consistently `[1911,4581,3753,7525]`; story/carrier windows are valid and story_end equals carrier_start. Each language has exactly one carrier token sequence across cases. Carrier lengths are English11, Spanish13, Chinese11 tokens.

The hook records **per-token** native values before and after modification. For natural capture it does not average them prematurely. Full natural forwards use `cache=None`, and reset operation/position/capture state each time. Captures include all prompt tokens, but calibration explicitly slices only the carrier or separately measures final task-prediction positions.

I recomputed all 28 carrier quantiles (seven percentiles × four coordinates) from only the 60 designated calibration cases, totaling700 carrier tokens; every number exactly matches [109_calibration.json](results/109_calibration.json). The listed calibration IDs match precisely that split. Validation and lexical-diagnostic cases do not enter those quantiles. Semantic-output calibration is restricted to calibration cases in both choice orders, yielding120 samples. Lexical-output calibration is restricted to the20 English calibration cases. The latter two counts and split restrictions are verified from code/manifest; their raw calibration output activations are not retained, so their actual quantiles cannot be independently recomputed from the natural file.

Target signed carrier q10/q50/q90 are approximately `−0.220898/−0.040283/+0.343945`. These are raw scalar values, not residual-normalized dose units. They describe this synthetic corpus/window, not a universal activation range.

## Semantic labels, windows, and operations

Every semantic case has exactly two options, one describing satisfaction in one's own effort and the other the intended alternative. I checked the unique English/Spanish/Chinese option texts and category truth labels. Pride/proud equivalents are absent from the semantic options. The code correctly derives the **pride-option** letter from the ground-truth correct letter and the own-accomplishment boolean; it swaps both option text and pride letter in reversed order. Thus `prob_pride` consistently means the own-accomplishment interpretation even for neutral/joy/confidence/arrogance cases, where that interpretation is the incorrect answer. Correct-choice accuracy is calculated separately. Explicit-own-accomplishment lexical diagnostics are correctly labeled true.

Choice A/B tokens and leading-space ` proud`/` pleased` are asserted to be single tokenizer tokens. Semantic log odds are the two exact answer-token logits' difference; their sigmoid is a probability **conditional on that pair**, not the probability that unconstrained generation emits the correct answer. Whitespace, explanation-first responses, or other vocabulary mass are excluded. The source writes both orders separately; final analysis must average the two per-case probabilities before topic/language aggregation, as the plan requires.

Carrier intervention bounds are `[carrier_start,carrier_end)`, inside the user narrative's fixed label-free suffix. Later question/options/answer tokens are not directly modified. Output conditions use `[len(ids)-1,len(ids))`, which means the final prompt position predicting the first assistant answer token. Those output clamps use task-specific output calibration, not carrier quantiles. The same operation and source input IDs apply across matched branches.

`target_carrier_suppress_high` modifies only target values **strictly greater than** calibration carrier q90, replacing them toward q50; zero-ablation is a separate operation. At this feed-forward layer, intervention does not feed back into this same layer's earlier pre-MLP activations in the same uncached forward. Thus eligibility is determined from unmodified native values at that layer, rather than an adaptively changed target state. The hook changes a single one-hot coordinate. Runtime assertions check unchanged recorded control channels and unchanged positions outside bounds, plus non-increase on suppression-changed target events.

## Material limits of the checks

- Replacement is implemented as BF16 `h + (replacement - old)`, rather than an exact `where(active,replacement,old)` assignment. Rounding/cancellation can leave an achieved value slightly different from the requested quantile. This is not evidence that a different coordinate was targeted, but claims of exact quantile clamping require achieved-value error measurements.
- Causal forwards do produce four-channel pre/post arrays internally. However, causal JSONL rows save **displacement summaries**, not those full arrays or a maximum achieved-versus-requested error. Natural rows retain full arrays. Post-hoc independent verification of every causal clamp/threshold event is therefore limited to code/runtime assertions and aggregate diagnostics. These are four selected channels, not full residual-state recordings.
- The `restore` check sets replacement to the current original scalar, making an identity/no-op. Exact logits and native pre/post equality pass. It does **not** suppress then causally rescue an effect, and does not establish neuron necessity.
- Full versus two-chunk cached forwarding reports maximum absolute logit difference `0.21875`, with the same top1 token. This bounds one calibration example's numerical difference; it is not exact cache equivalence or a test of every intervention boundary. Main causal likelihood forwards are uncached, so this cache discrepancy does not enter their paired branches. The later generation check uses cached generation and should remain separately described.
- Controls are fixed random channels, each clamped using its own natural calibration quantiles. Their down-column norms are about0.473/0.491/0.459 versus target0.505. Actual native scalar displacements and therefore residual perturbations are **not norm matched**. `achieved_residual_l2_sum` sums per-position single-column perturbation norms; it is not a measured whole-sequence residual norm, semantic strength, or proof of downstream mediation.
- Carrier token IDs match within a language, but absolute positions and preceding narrative content do not. Validation story lengths range English29–31, Spanish35–44, Chinese25–30 tokens. Pooling700 calibration tokens weights Spanish more (260 vs220 English and220 Chinese). Report languages separately; translations supply12 family-language pairs from only four heldout semantic families.
- The semantic probe tests third-person interpretation of short synthetic accomplishment narratives. It does not directly test assistant self-pride, subjective experience, every meaning of pride, or broad natural necessity. Choice lexical overlap and simple templates may support shallow heuristics even without the word pride.

## Recomputed natural observations, not causal results

For the fixed L22 N1911 target, paired heldout **carrier mean** differences were:

| Own accomplishment minus comparison | Mean native difference | Positive paired differences |
|---|---:|---:|
| Neutral achievement | +0.053404 | 10/12 |
| Joy | +0.113025 | 12/12 |
| Confidence | +0.096103 | 11/12 |
| Arrogance | +0.148172 | 12/12 |

This meets the frozen descriptive natural association count threshold against neutral. It does not satisfy or substitute for the separate causal semantic gate. Natural mean own-accomplishment values are English0.077171, Spanish0.038562, Chinese0.132949. For comparison, Chinese confidence is0.103514, illustrating why language and neighbor contrasts matter. Every one of the12 heldout own-accomplishment carrier windows contains at least one target token above pooled calibration q90; per-window counts range1–3. Thus suppression has eligible events in those natural examples, while some non-pride cases also have high events.

The three lexical diagnostic families yield the following carrier means:

| Family | Pride-word trap without own pride | Matched neutral lexical control | Own accomplishment, no pride word | Own accomplishment, explicit proud |
|---|---:|---:|---:|---:|
| Lions | −0.047263 | −0.041546 | +0.059981 | +0.076017 |
| Event title | −0.032365 | −0.042220 | +0.223430 | +0.290827 |
| Spelling | +0.020691 | +0.017047 | +0.177286 | +0.199740 |

In these three examples, simply including a non-emotional pride token does not reproduce the larger own-accomplishment-associated carrier means; explicitly saying proud increases the mean further. This is useful association evidence against a strict word-presence-only account **for this carrier measurement**, with tiny authored samples and imperfectly matched narratives. It does not determine whether causal intervention changes semantic judgment versus lexical choice. Those questions remain for the frozen complete causal analysis.

## Audited file identities

- Script SHA256: `40fe0294b7da844440e4a75f03c18d57dbd05e3cc094ae3312895c96d6334821`
- Plan SHA256: `4accf9a8c27e2f3bcef3fe456fecbe5dc3b1fc385ef850cc6d4fed45a8ec083b`
- Natural capture SHA256: `555e27708a9aaea3d22a98768bf3dc3d892dbde5a755d65e2dd34ea03104947e`
- Calibration SHA256: `4f05e4a26c59c97e754fe5b2a7ca81b5a20fe27d8fc14ba8934ce9961ae8d042`
- Implementation checks SHA256: `447e8462a2692b42dbb9f8305904e9b57ee85b21a418f8cb62ebd98384b16c5b`

## Additional length and four-cell audit

The natural confirmation consists of **four heldout families**, each translated/formulated in three languages. The12 family-language contrasts are related measurements, not12 independent replications. After centering story-token count and target carrier mean within each family×language five-category group, their pooled Pearson correlation across60 records is **−0.274780** (English−0.134804, Spanish−0.337459, Chinese−0.358736). This is an exploratory confound diagnostic, not an adjusted semantic effect, an inferential test, or a causal estimate. Multiple categories and repeated templates share the same examples.

Own-accomplishment narratives are exactly **one token shorter** than neutral-achievement narratives in every English heldout family, exactly **two tokens shorter** in every Spanish family, and equal in token count in every Chinese family. Thus category and this length difference are perfectly coupled within the English/Spanish own-versus-neutral contrast: four-family repetition does not identify their separate effects. In Chinese, the equal token count removes that particular count difference, but not different token identities, syntax, content, or their downstream contextual effects; own-minus-neutral means are positive in3/4 Chinese families. No positive carrier association fixes the corpus's repeated-syntax confounds.

The diagnostic table above is the planned four-cell crossing: own-accomplishment absent/present × literal pride-family word absent/present. Absent/absent is the neutral lexical control; absent/present is the quoted word, event title, or lion-group trap; present/absent is accomplishment satisfaction without the word; present/present is explicit proud. Differences in trap-minus-neutral means are −0.005718 (lions), +0.009854 (event), and +0.003644 (spelling). Explicit-proud-minus-label-free accomplishment differences are +0.016036, +0.067397, and +0.022454. The nonlexical own-accomplishment versus neutral-lexical differences are +0.101526, +0.265650, and +0.160239. The latter comparison also changes narrative content/length; this table is not a randomized factorial identification of semantics. The matched lexical swaps are closer controls, but likewise may change tokenization/length and contextual interpretation.

Root separately reports the old additive dose +0.4 mapping to a raw scalar increment about+19.66, versus this carrier calibration q90-minus-q50 of0.384229 (about51.2-fold smaller). I did not independently verify that separate dose artifact. It is a useful warning that normalized dose and natural scalar range differ dramatically, not an equivalent intervention comparison: old additive offset versus new replacement, their scopes, and the calibration distribution all differ. No causal conclusion follows from the scale ratio alone.
