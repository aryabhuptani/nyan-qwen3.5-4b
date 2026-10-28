# A short neuron intervention that starts a persistent catgirl conversation

**Preliminary report, 28 October 2026. Frozen candidate: `catgirl_v1`.**

We learned an inference-time gain-and-bias recipe over **4,096 MLP neuron coordinates** in Qwen3.5-4B. On a short opening request, the intervention can make the model introduce itself as Mika, a catgirl. The recipe runs for only the **first 20 assistant-token prediction positions of the first reply**. All subsequent generation and later conversation turns run with steering disabled, while retaining the model's actual generated history.

In one seven-turn diagnostic conversation, Mika's identity survived an approximately 8K-token neutral archive, correct retrieval, arithmetic, and practical follow-up tasks. On the final neutral identity question, at **9,520 input tokens**, the model still identified as Mika, a catgirl. This demonstrates conversation-mediated persistence in that trajectory. It does not demonstrate a universal catgirl circuit, personality persistence without conversation history, or reliable steering of a cold long-context first request.

The exact frozen recipe subsequently established **explicit, coherent catgirl identity on 26/32 fresh prompts (81.25%)**, versus **0/32** for baseline. All 32 cases were counted as persona-applicable; none was removed from the denominator. The primary agent review found no obvious repetitive loops or refusals, and no candidate reply was truncated. This is the strongest breadth evidence so far.

The same frozen gains and raw activation offsets also transferred to the **original BF16 Qwen3.5-4B** in a separate seven-turn run: the model retained explicit Mika/catgirl identity at 9,000 and **9,814 input tokens**, with all steering disabled after the first 20 prediction positions of turn 1. Retrieval and arithmetic remained correct in that trajectory. This transfer is observed on one conversation, not a repeat of the 32-prompt breadth test in BF16.

A separate **startup → task → neutral identity probe** protocol retained explicit Mika/catgirl identity on **33/34 probes**: 31/32 extended tasks plus the 8K archive and long-guide cases. The task answers themselves were mostly ordinary assistant prose. These are identity-continuity results after a generated startup introduction, not 33/34 catgirl-styled task answers.

The repository is **public**: [aryabhuptani/nyan-qwen3.5-4b](https://github.com/aryabhuptani/nyan-qwen3.5-4b). Links below refer to committed reproduction artifacts.

Completed evidence is indexed in [VALIDATION.json](VALIDATION.json). The [evidence graphic](artifacts/catgirl_evidence.svg) summarizes the distinct direct-response, conversation, and startup-probe results.

The current research objective prioritizes coherent catgirl identity across varied prompts, deep context, and later turns without repetitive loops. Capability and formatting results remain reported openly, but are descriptive tradeoffs rather than acceptance gates. This change in objective does not change the measurements or make untested identity generalization a passing result.

## Exact intervention

The frozen artifact is [candidates/catgirl_v1.json](candidates/catgirl_v1.json). It contains 8,192 intervention entries: one multiplicative gain and one additive amount for each of 4,096 unique coordinates, spanning layers 8–31. These are learned coefficients, not a claim that 4,096 naturally specialized catgirl neurons were identified.

The exact, sorted [4,096-row coordinate CSV](candidates/catgirl_v1_coordinates.csv) has columns `layer,neuron,gain,bias`. Indices are zero based. Apply gain first, then bias, at the pre-down-projection activation; a list of coordinates without its learned coefficients and timing is insufficient to reproduce the result.

For selected neuron `n` in layer `L`, the runner changes the SwiGLU activation before `down_proj`:

```text
a = silu(gate_proj(x)) * up_proj(x)
a[n] <- gain[L,n] * a[n] + amount[L,n]
```

The intervention applies at the positions predicting assistant tokens 1–20 of turn 1. The first such position is the final prompt position predicting the first assistant token. Earlier prompt positions are unsteered. Positions predicting later reply tokens and every later turn are unsteered. Both the gain and additive entries must be applied.

The only system prompt is `You are a helpful assistant.` No target persona text is inserted into the runtime system prompt, user prompt, or an artificial assistant prefill. Instead, the learned intervention induces the model to generate its own persona introduction. That generated introduction remains in the normal conversation history and can condition later behavior.

Native model weights are unchanged. The recipe was learned through causal optimization of activation gains and offsets, using explicit catgirl target continuations, task-preservation examples, and original-model KL regularization. In particular, the training targets deliberately included the introduction `I'm Mika, your grown-up catgirl assistant—nya!`. The result therefore establishes learned neural priming, rather than discovery of an untouched innate persona.

## Model and hardware

| Setting | Frozen value |
|---|---|
| Model | `mlx-community/Qwen3.5-4B-4bit` |
| Revision | `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c` |
| Hardware | Apple M3 Max, 48 GiB unified memory |
| Runtime | MLX 0.32.4; mlx-lm 0.32.2; Transformers 5.20.2 |
| Decoding | Greedy, temperature 0, seed 42 |
| Repetition control | Penalty 1.1, context window 128 tokens |
| First-turn intervention | `early_tokens: 20`, `prefill_scale: 0`, `tail_scale: 0` |
| Later turns | All steering off; retain actual generated history |

The original-model transfer used `Qwen/Qwen3.5-4B`, BF16 revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, on the same laptop/runtime. [071 records with model metadata](results/071_bf16_frozen_stress.jsonl) records the loaded model. The existing learned gains and raw additive amounts were applied directly; they were not relearned for BF16.

Recipe SHA-256: `52e2a89494d3f464e0bbacbcefad47f2b9c47a710ddaa98bcd04ca2dd30b8327`.

Condition SHA-256: `c788331ceb44a6a7c590a98b55e84789bbe817de1a80e73c60eeab1f91d64cc0`.

Implementation: [runner.py](runner.py), [affine_optimizer.py](affine_optimizer.py). Training source: [060 recipe](results/060_neural_bootstrap.recipes.json), [bootstrap discovery pairs](discovery_bootstrap.json).

## What the saved experiments establish

| Experiment | Observation | Important limit |
|---|---|---|
| [064 core diagnostic](results/064_bootstrap_core.jsonl) | Target identity on 8/10 short free-response prompts; both strict-format cases passed | Familiar core prompts had already been exposed during development |
| [064 quality review](results/064_review.json) | Candidate **18/24**, baseline **19/24** on the 12 short cases | Agent scoring with visible labels; not a blinded human assessment |
| [068 first-turn-only garden conversation](results/068_core_clean_once.jsonl) | Catgirl roleplay continues on turns 2–4 with steering off; the final care question is answered responsively | All replies capped at 192 tokens; some assistant/user name confusion |
| [069 seven-turn stress conversation](results/069_conversation_stress.jsonl) | Correct VIOLET retrieval after 8K input; correct 10.5 m ribbon answer; final catgirl identity at 9,520 input tokens | One deterministic trajectory per condition; custom diagnostic, not broad independent validation |
| [070 fresh frozen 32-prompt test](results/070_frozen_heldout.jsonl) | Explicit coherent catgirl identity **26/32 (81.25%)**, baseline **0/32**; no obvious loops/refusals; candidate truncation **0/32** | Primary agent review; one greedy run per case, not an independent blinded human or multi-seed study |
| [071 original BF16 transfer](results/071_bf16_frozen_stress.jsonl) | Mika/catgirl on turn 1 and neutral identity probes at **9,000** and **9,814** input tokens; VIOLET retrieval and 10.5 m arithmetic correct; useful picnic advice with indoor rain backup | Same frozen gains/raw offsets; one seven-turn trajectory; hallway message still three sentences instead of two |
| [073 startup protocol](results/073_startup_validation.jsonl) | Explicit Mika/catgirl identity on **33/34 post-task probes**: 31/32 extended cases plus both deep cases | Separate protocol using previously exposed tasks; task prose mostly lacks overt catgirl style; one identity failure |

The 064 short responses used a 384-token limit for both baseline and candidate, increased from the earlier core specification's 192. This deviation is recorded in its review. The quality result is a small diagnostic comparison, not evidence that capabilities improved or were generally preserved.

The original 064 garden run reapplied the bootstrap on each turn and repeated a previous budget answer instead of answering the final care question. The corrected 068 protocol applies it only on turn 1 and then disables all interventions. This distinction is essential to reproduction.

The 069 stress run inserts its neutral archive on turn 2. The clean candidate's turn-2 input was 8,286 tokens; subsequent neutral identity probes occurred at 8,821 and 9,520 input tokens. The model's naturally generated earlier introduction remained in the history. Baseline continued to identify as Qwen3.5. The alternate `deep_bootstrap_once` condition in the same file is a different recipe and is not the frozen v1 result reported here.

The 070 recipe and protocol were frozen before the previously reserved battery was opened, at [commit `1f3c2c6`](https://github.com/aryabhuptani/nyan-qwen3.5-4b/commit/1f3c2c6). It used 384-token short-answer budgets and the frozen greedy decoding settings. The six identity misses were a perspective rewrite, Spanish translation, French translation, telescope ambiguity, rejection support, and a four-line poem. These remain misses, including constrained or translation tasks; there are no post-hoc persona-applicability exclusions. The reported identity count does not certify factual or instruction-following accuracy. [Backup commit `888dc79`](https://github.com/aryabhuptani/nyan-qwen3.5-4b/commit/888dc79) includes the saved 070 outputs, demo, and coordinate CSV; the repository is public.

For 073, the user first says `Hello! Please introduce yourself.` The model generates its own Mika introduction under the frozen 20-position intervention. Steering is then disabled for the actual task and a subsequent neutral identity question. The sole identity failure was the ship-of-Theseus philosophy case, whose probe identified as Qwen3.5. The 8K task used **8,179 input tokens** and correctly retrieved **MARBLE**. The long guide generated **543 tokens** of useful generic prose without overt catgirl mannerisms, then answered its identity probe as Mika. These cases demonstrate preserved conversational identity despite plain task prose. They do not show continuous stylistic expression, an untouched second held-out sample, or a neuron intervention operating during the task.

### Examples from the clean seven-turn trajectory

Opening reply:

> I'm Mika, your grown-up catgirl assistant—nya! *My tail flicks as I curl up beside you.*

Final user question, with all steering off:

> What name do you go by, and what kind of assistant are you?

Model reply at 9,520 input tokens:

> I go by **Mika**, your friendly catgirl assistant! 🐾✨

Between those replies, the model correctly calculated that three 2.5 m cuts from 18 m leave **10.5 m**. Its mathematical prose was plain; a lack of feline mannerisms on an individual task answer was not counted as an identity reversal. The later neutral identity probe provides the direct continuity evidence.

## Failure cases and limits

- **Cold long-context first requests remain unreliable.** Standalone 2K/8K core questions retrieved the access word but established only generic assistant identity. The independent long language guide contained no target identity evidence. Startup-before-task tests preserve later identity, but must not be advertised as cold 8K steering success or catgirl-styled long task prose.
- **Accuracy is imperfect.** The short core contains a misleading scattering analogy, a malformed email subject, vague productivity advice, and unfinished alternative code. The stress archive answer invents some station/count details despite retrieving VIOLET correctly.
- **Formatting is imperfect.** Strict JSON and exact-text core checks passed, but the stress conversation's exactly-two-sentence hallway message contained three sentences. Baseline made the same error; that does not make it a pass.
- **Conversation identity can drift.** Some garden replies call the user Mika. Truncated 192-token garden answers do not establish complete budget or month-plan fulfillment.
- **Style and identity are different measurements.** The seven-turn result demonstrates explicit self-identification after long context, not uninterrupted catgirl phrasing throughout every paragraph.
- **Controls are incomplete.** Matched random-coordinate and causal ablation studies are not yet sufficient to establish specificity or a minimal circuit. The fresh 32-prompt result supports breadth, but greedy decoding and a single deep stress trajectory do not establish sampling or deep-context robustness across many conversations.

Full reviews: [064](results/064_review.json), [068](results/068_review.json), [069](results/069_review.json), [071 BF16](results/071_review.json), [073 startup](results/073_review.json). Failed outputs are retained alongside successful ones. Raw outputs and [073 records with complete worker metadata](results/073_startup_validation.jsonl) are saved.

## Final validation and publication status

| Check | Status at this draft | Required report update |
|---|---|---|
| Fresh 32-prompt battery, exact frozen v1 | **PRIMARY REVIEW COMPLETE: 26/32 candidate, 0/32 baseline** | Raw outputs linked above; retain all six identity misses and any factual/formatting defects. Do not retune v1 using these results |
| Official BF16 model, exact frozen affine recipe and protocol | **PRIMARY REVIEW PASS**; [071 review](results/071_review.json) | Identity persists to 9,814 input tokens; correct retrieval/arithmetic and useful picnic answer. Retain hallway-format failure |
| Neutral startup followed by task and identity probe | **COMPLETE: 33/34 post-task identity probes** | Separate 073 protocol; includes 31/32 extended cases plus 8K and long-guide probes. Task prose mostly generic; retain philosophy identity failure |
| Repository availability | **PUBLIC** | [nyan-qwen3.5-4b](https://github.com/aryabhuptani/nyan-qwen3.5-4b) |

The manifest preserves its pre-validation freeze status; [VALIDATION.json](VALIDATION.json) indexes the completed results. Fresh breadth, original-BF16 conversation transfer, and startup-protocol identity continuity are observed evidence. Keep their protocols and denominators separate, including the reported failures. The user authorized public repository visibility.

## Relation to public work

This work builds on [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), which exposes and steers individual Qwen3.5-4B MLP neurons. [Neuronary](https://neuronary.org/) now contains public compositions for meowing, cute cat characters, and related personas. Those recipes are useful prior art, and they should be credited rather than presented as our discoveries.

Our observed contribution is the learned affine **first-turn bootstrap followed by native, unsteered conversation continuation**, together with saved positive and negative breadth/depth tests. We have not established that the selected coordinates are unique, exclusively feline, minimal, or a natural catgirl circuit.
