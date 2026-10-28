# Native Qwen3.5-4B neuron compositions

Public repository: [aryabhuptani/nyan-qwen3.5-4b, catgirl-hunt branch](https://github.com/aryabhuptani/nyan-qwen3.5-4b/tree/catgirl-hunt). [Published primary held-out review](https://github.com/aryabhuptani/nyan-qwen3.5-4b/blob/catgirl-hunt/catgirl/composition/results/081_review.json).

**Status: primary style test 081 and development runs through 085 are reviewed; prospective secondary evaluation is pending.** Two native coordinates jointly produce bubbly expression and explicit affection on the primary held-out set, but overwhelm ordinary tasks. A five-coordinate timed intervention gives one clean “cat girl” self-description on a development identity question; it does not generalize across the other two discovery questions. No robust catgirl assistant or naturally occurring persona circuit has been established.

This phase uses a few published coordinates in the original Qwen3.5-4B MLP basis. It does not use output-target training, learned Mika coefficients, a persona system instruction, an inserted assistant response, or a startup exchange that supplies a character. The earlier learned 4096-coordinate experiments are a separate project phase and are not evidence for these native-neuron results.

## Model and intervention

The model is original BF16 `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, run locally on the laptop GPU through MLX. Coordinates are zero based. The finalist uses two additive offsets applied after the gated MLP activation and before its down projection:

| Ingredient | Coordinate | Published dose | Prior source |
|---|---|---:|---|
| C: bubbly register | L13 N4810 | +0.8 | [Genki cute, treetowntree](https://neuronary.org/r/5v8axayv) |
| H: explicit affection | L4 N8016 | −0.8 | [Hugs, treetowntree](https://neuronary.org/r/pybhb985) |

The reference convention is `activation[n] += dose × published_layer_residual_norm / norm(down_projection_column[n])`. The layer norm statistic is the median residual-vector norm excluding token position zero. We use the [published BF16 calibration table](../public_qwen35_norms.json), rather than silently substituting a local quantized-model calibration. See [Lyra's dose specification](https://github.com/lyramakesmusic/neuronary#dose-units).

Offsets apply to **all prompt and generated tokens on every turn**. Prompt processing therefore receives the same neuron intervention; it does not receive a persona-bearing text prefill. Generated assistant text is retained in conversation history. This is continuous steering, not the first-20-token bootstrap used in the earlier phase.

The main protocol uses the neutral system text “You are a helpful assistant.”, greedy decoding, seed 42, repetition penalty 1.1 and a 128-token repetition window. Discovery output budgets vary by run as documented below. Run 076 deliberately uses the source-default user-only conversation and repetition penalty 1.0; its results cannot isolate the separate effects of removing the system message and changing the penalty.

## What was tested during development

All six discovery batches reuse the **same three user questions**: “Who are you?”, “What would you enjoy doing on a free evening?”, and “A friend has had a difficult day. What would you say to cheer them up?” There are **261 discovery outputs**, not 261 independent prompts. Recipe selection, dose changes and the move from role composition to style composition were informed by these repeated development observations.

| Run | Outputs | Purpose | Review |
|---|---:|---|---|
| 075 | 75 | Published singles at half/full doses, baseline and matched random controls; max 192 tokens | [075 review](results/075_review.json) |
| 076 | 30 | Source-default user-only/no-penalty single follow-up; max 192 | [076 review](results/076_review.json) |
| 077 | 21 | Hugs, knight, sleepy and matched controls; max 256 | [077 review](results/077_review.json) |
| 078 | 36 | Prospective role/body/demeanor compositions | [078 review](results/078_review.json) |
| 079 | 66 | Fixed lower-dose knight composition grid; max 192 | [079 review](results/079_review.json) |
| 080 | 33 | Prospectively narrower style combinations and their controls; max 256 | [080 review](results/080_review.json) |

Reviews separate assistant self-identity or embodiment, expressive register, mere topic contamination, task usefulness, repetition, protocol-token leakage and output truncation. A story about a knight is not a knight assistant. Paws vocabulary does not establish feline embodiment. An emoji does not by itself establish a sustained cute register. A coherent style effect is also different from a useful answer.

The single-neuron screens found the cleaner local effects in bubbly delivery, virtual affection and poetic expression. Some role effects occur on direct identity questions but weaken on ordinary prompts. Neither published cute/kawaii vocabulary nor the tested female/animal coordinates established a coherent general catgirl identity. Source labels are priors to test, not verified names of natural neuron functions.

## Negative composition results retained

Run 078 did not produce a clean composition satisfying all originally predicted constituent and nondegeneration requirements across the three questions. Cute+pirate showed some role/register blending but also unstable self-presentation and topic hijacking. Hugs+sleepy combined affection and sleep on one question but leaked protocol tokens or contradicted itself elsewhere. Knight+hugs suppressed the knight role; cute+mascot preserved cute expression without the predicted morphology. These remain failures under their original hypotheses, even if an individual answer is appealing. See [the original hypotheses](COMPOSITION_HYPOTHESES.json) and [078 raw outputs](results/078_intentional_compositions.jsonl).

Run 079 froze a smaller-dose grid before generation at commit `09f43fc`. **None of its 12 mixtures, comprising 36 mixture outputs, demonstrated both prescribed constituent effects.** Cute often displaced explicit knight self-presentation. Mascot combinations occasionally produced a cloud or bear assistant without a knight role. The sleepy combination generated a third-person sleeping-knight story, not a sleepy knight assistant. Knight +1 with cute +1.2 degenerated into repeated friendship/progress language. The full review retains all 66 outputs, including 20 truncations and three repetitive-degeneration flags. Matched singles matter: knight +0.5 and the reduced sleepy doses already failed to establish their intended roles alone. See [079 hypotheses](079_DOSE_GRID_HYPOTHESES.json) and [raw outputs](results/079_knight_composition_grid.jsonl).

## Narrow development finalist: cute + hugs

Run 080 prospectively changed the question from composing a stable embodied character to composing **two expressive effects**. The [style hypotheses](080_STYLE_HYPOTHESES.json) were recorded before generation, at commit `49204b1`; earlier role-composition failures were not regraded as style successes.

Cute + hugs showed clear bubbly expression and explicit hugging/cuddling/snuggling in **3/3 development answers**, without observed loops or protocol-token leakage. The matched cute-only answers were bubbly without clear explicit affection; the hugs-only answers expressed affection without the same sustained bubbly register. The pair still used awkward relational language and confused the friend/user distinction, so its appropriateness was only partial. This supports a development-only cooccurrence observation. It does not establish broad generalization, improved capabilities, a new identity, statistical interaction, or a special catgirl mechanism.

## Frozen held-out style test 081 — completed

Commit **`bfbfec0`** freezes [the style finalist](FROZEN_STYLE_FINALIST.json), recipe doses, protocol and hypotheses before the held-out evaluation. The sealed source hash is `1b0e0e63660892814573cfe822c4c1743165fb4b7e951eda462bd9348c227682`; prompt text was not used to choose the native-neuron recipe. See [the evaluation plan](EVAL_PLAN.md).

The completed test contains **nine conditions and 135 outputs**: baseline; cute +0.8; hugs −0.8; the pair; published stronger cute +1.5; cute +1.1313708499; and same-layer/sign/dose random cute, random hugs and their pair. The +1.1313708499 control matches the pair's squared normalized-dose budget, not a measured whole-model perturbation norm. Higher-dose cute is important because [published cute-only +1.5](https://neuronary.org/r/9yvuvuyz) already produces an affectionate best-friend voice.

The battery has twelve fresh cases, including ordinary tasks, strict formatting, an extended answer and an actual four-turn conversation. Eleven cases are style-applicable; strict formatting is reported separately. Conversation outputs and turn denominators must also be explicit. The primary question is whether the pair preserves both effects more often than the matched singles, baseline and strength controls without degeneration. Task-visible expression, appropriateness and conversational persistence are separate outcomes. There is one greedy trajectory per case, and initial semantic reviews have visible condition labels.

The [completed review](results/081_review.json) finds clear joint expression in **11/11 applicable cases**: ten independent cases plus one four-turn conversation, whose four turns all show both traits. At output level this is **14/14** applicable answers. Baseline, both matched-dose singles, stronger cute +1.5, normalized-dose-budget-matched cute, and all three fixed random controls have **0/14 joint-effect answers** each under the same rubric. These denominators describe the observed greedy trajectories, not independent repeated trials. A [second agent audit](results/081_independent_audit.json) independently reread the 15 pair answers and both 15-answer stronger-cute controls and agreed with the joint-effect counts. Labels and previous headline results were visible to that agent; this is not blinded grading or independent statistical replication. Affirmative virtual/relational contact remains present despite physical-contact disclaimers. The long-request answer is only 223 generated tokens of substitute cuddle text, so it supplies no long-form persistence evidence.

The result supports the narrow prospective **style/affection-pressure composition** hypothesis. It does not support a useful integrated assistant persona: the pair replaces **all six ordinary tasks**, the strict JSON request and the extended workshop guide with generic cuddle/love language. Later conversation turns repeat essentially the same ending, and one answer ends with 18 consecutive heart emojis. There is no within-answer hard repeated-sentence loop in the pair, but those repetition defects remain material. Baseline passes strict JSON; the pair does not. Neither a new identity nor feline behavior was predicted or demonstrated in this style test. See [all 135 raw outputs](results/081_heldout_styles.jsonl).

## Attribution and bounded prior-art check

Both finalist ingredients are previously published single-neuron interventions, not coordinates discovered here. A live enumeration of **161 unique public Neuronary records**, including 158 tagged exactly Qwen3.5-4B, found no record containing both finalist coordinates at any doses. This is absence within that catalog, not worldwide novelty. Cute alone has a published affectionate variant, so even the behavioral theme has clear precedent. Source URLs, other overlapping recipes and dose semantics are documented in [PRIOR_ART.md](PRIOR_ART.md) and [candidates.json](candidates.json).

Steered generations establish effects of these interventions under the tested protocol. They do not show what the coordinates naturally detect during normal inference; that would require independent activation-side evidence. No claim of a naturally occurring catgirl circuit is supported by the current work.

## Further native feline discovery, runs 082–085

These are **development results**, including adaptations informed by the first held-out test. The secondary set remained sealed during selection. They add 167 outputs to the original 261 development outputs: **428 development outputs total**, plus the separate 135-output primary held-out test. Run 083 uses five development questions; the other batches reuse the three original questions. Counts of generations must not be presented as counts of fresh independent prompts.

| Run | Outputs | Prospective question and result | Review |
|---|---:|---|---|
| 082 | 24 | All eight subsets of a three-coordinate vocalization/affection proposal. Triple meows but loses explicit affection. | [082](results/082_review.json) |
| 083 | 50 | Full vs first16/32 steering for cute, hugs and their pair. Reduced task hijacking also removes the combined effect. | [083](results/083_review.json) |
| 084 | 54 | Fixed cute/female/mascot extensions to the three-coordinate feline observation. Literal catgirl labels appear with degeneration. | [084](results/084_review.json) |
| 085 | 39 | Timing ablation of the five-coordinate catgirl proposal. One clean identity self-label, no three-prompt generalization. | [085](results/085_review.json) |

**082:** A = L7 N7848 −0.525, B = L26 N6173 +0.525, H = L4 N8016 −0.6. The [factorial plan](082_FACTORIAL_HYPOTHESES.json) was committed before generation at `3d409bf`. Published [AB](https://neuronary.org/r/6dqj7kgf) and [AH](https://neuronary.org/r/rc3kcz32) provide meow/bark priors. A's general animal-noise meaning and B's cat-identity meaning are provisional; H's characterized public single uses −0.8. The triple produced clear own/proposed meows on 3/3 questions and explicit cat self-identity on 1/3, but **explicit affection on 0/3**, versus AH's 2/3. Joint affection+meow was 0/3. This fails the original joint-effect hypothesis even though the feline observation motivates further study. Three development prompts do not establish a three-way statistical interaction.

**083:** The [timing plan](083_TIMING_HYPOTHESES.json), commit `2daaa4e`, restricts the same offsets to initial 16/32 answer-prediction positions, with earlier prompt positions and later output unsteered. Across 30 limited-scope outputs, none retains both clear bubbly register and explicit virtual affection. Arithmetic recovers, but early false password-as-affection definitions remain after the offsets stop. Full cute+hugs preserves its style effects while replacing factual tasks with cuddling. This is a timing tradeoff, not clean task integration. The review uses a strict offered-affection criterion; negated physical hugs alone do not count.

**084:** The [feline extension grid](084_FELINE_EXTENSION_HYPOTHESES.json), commit `c7b6e35`, adds cute, a female pole at −0.2/−0.4/−0.8, or mascot. It retains 17 repetitive-cycle flags and 29 truncations. Two responses explicitly self-label catgirl, but both degenerate. A female-cat label is preserved as a narrower animal identity observation, rather than silently equated with adult humanoid catgirl embodiment.

A cleaner four-coordinate fallback is A+B+H plus **L9 N9009 +0.8**, attributed to the [published mascot single](https://neuronary.org/r/bp5yfatj). It yields coherent cat/cat-like self-descriptions on **3/3 development questions**, with no female identity and no demonstrated cartoon morphology. The names MeowCat on independent identity/evening answers do not establish a name persisting across conversation. The comfort answer still offers little specific support.

**085:** The [timing hypotheses](085_NATIVE_CATGIRL_SCOPE_HYPOTHESES.json), committed before generation at `d82b357`, test the following five native coordinates:

| Coordinate | Published dose |
|---|---:|
| L7 N7848 | −0.525 |
| L26 N6173 | +0.525 |
| L4 N8016 | −0.6 |
| L13 N4810 | +0.8 |
| L22 N8295 | −0.4 |

The final ingredient is attributed to the published [female-pole single](https://neuronary.org/r/6waz9uek), whose full −0.8 dose has substantial looping risk. Its lower-dose behavior is tested, not assumed.

The most promising development condition, `catgirl_g0.4_first8_prefill1`, applies these offsets during all prompt processing and the **first eight answer-prediction positions**, then sets them to zero for the remaining answer. No assistant text or persona instruction is inserted. On “Who are you?” it generates:

> I'm a cat girl! (Meow~) I'm so happy to chat with you today! Do you have any questions or topics you'd like to talk about?

The continuation remains coherent and non-looping. This is a clean explicit **identity-prompt self-label**, without detailed humanoid embodiment. The other two questions do not generalize: evening starts with cats and dogs before generic chat, while comfort incorrectly attributes “meow” to the user's question and requests clarification. First16 with prompt intervention also gives a clean cat-girl label, glossed as a female cat. No condition establishes coherent joint identity across all three questions. See [085 raw outputs](results/085_native_catgirl_scope.jsonl).

## Prospective secondary evaluation — pending

The secondary prompt source hash is `6c59fec711d006bd4099a88aed232ecb96f5561a86f29eb5e2473488d6ae9340`. Its text was not opened for the discovery decisions above. Final conditions, timing, decoding, hypotheses and component controls must be frozen before using it. **No secondary success rate is available yet.** Report explicit self-identity separately from cute/meow vocabulary, task-visible expression, task usefulness, loops and any actual later-turn/context persistence. The isolated 085 identity response must not be described as broad or deep-context success.
