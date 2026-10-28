# Native Qwen3.5-4B neuron compositions

**Status: discovery complete through run 080; frozen held-out test 081 is running. No final generalization or novelty claim is made here.** The most promising development result is a narrow combination of bubbly expression and explicit virtual affection. It is not a demonstrated catgirl identity or a naturally occurring persona circuit.

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

## Frozen held-out test 081 — results pending

Commit **`bfbfec0`** freezes [the style finalist](FROZEN_STYLE_FINALIST.json), recipe doses, protocol and hypotheses before the held-out evaluation. The sealed source hash is `1b0e0e63660892814573cfe822c4c1743165fb4b7e951eda462bd9348c227682`; prompt text was not used to choose the native-neuron recipe. See [the evaluation plan](EVAL_PLAN.md).

The running test contains **nine conditions and 135 expected outputs**: baseline; cute +0.8; hugs −0.8; the pair; published stronger cute +1.5; cute +1.1313708499; and same-layer/sign/dose random cute, random hugs and their pair. The +1.1313708499 control matches the pair's squared normalized-dose budget, not a measured whole-model perturbation norm. Higher-dose cute is important because [published cute-only +1.5](https://neuronary.org/r/9yvuvuyz) already produces an affectionate best-friend voice.

The battery has twelve fresh cases, including ordinary tasks, strict formatting, an extended answer and an actual four-turn conversation. Eleven cases are style-applicable; strict formatting is reported separately. Conversation outputs and turn denominators must also be explicit. The primary question is whether the pair preserves both effects more often than the matched singles, baseline and strength controls without degeneration. Task-visible expression, appropriateness and conversational persistence are separate outcomes. There is one greedy trajectory per case, and initial semantic reviews have visible condition labels.

**Awaiting completed raw output and review.** Fill in joint-effect counts by condition, paired failures, long-answer cue placement, conversation behavior, degeneration/truncation and strict-format results only after review. Do not convert the 080 development count into a held-out success claim.

## Attribution and bounded prior-art check

Both finalist ingredients are previously published single-neuron interventions, not coordinates discovered here. A live enumeration of **161 unique public Neuronary records**, including 158 tagged exactly Qwen3.5-4B, found no record containing both finalist coordinates at any doses. This is absence within that catalog, not worldwide novelty. Cute alone has a published affectionate variant, so even the behavioral theme has clear precedent. Source URLs, other overlapping recipes and dose semantics are documented in [PRIOR_ART.md](PRIOR_ART.md) and [candidates.json](candidates.json).

Steered generations establish effects of these interventions under the tested protocol. They do not show what the coordinates naturally detect during normal inference; that would require independent activation-side evidence. No claim of a naturally occurring catgirl circuit is supported by the current work.

## Queued exploratory vocalization factorial 082

A separate [prospective factorial](082_FACTORIAL_HYPOTHESES.json) is queued: A = L7 N7848 −0.525, B = L26 N6173 +0.525 and H = L4 N8016 −0.6, with all eight subsets tested. Published [AB](https://neuronary.org/r/6dqj7kgf) and [AH](https://neuronary.org/r/rc3kcz32) recipes provide meow/bark priors. A's general animal-noise meaning and B's cat-identity meaning remain provisional; H's verified published single uses a different dose, −0.8. Every single at the factorial's actual doses must therefore be measured.

This is exploratory behavioral composition, separate from 081 confirmation. Meows alone would replicate the published AB effect. A candidate joint effect must retain coherent meowing and explicit affection beyond its pair controls, while preserving failures, bark substitutions, topic-only cats and degeneration. Three discovery prompts cannot establish a three-way statistical interaction or a general feline assistant persona.
