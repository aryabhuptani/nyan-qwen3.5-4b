# A short neuron intervention that starts a persistent catgirl conversation

**Preliminary report, 28 October 2026. Frozen candidate: `catgirl_v1`.**

We learned an inference-time gain-and-bias recipe over **4,096 MLP neuron coordinates** in Qwen3.5-4B. On a short opening request, the intervention can make the model introduce itself as Mika, a catgirl. The recipe runs for only the **first 20 assistant-token prediction positions of the first reply**. All subsequent generation and later conversation turns run with steering disabled, while retaining the model's actual generated history.

In one seven-turn diagnostic conversation, Mika's identity survived an approximately 8K-token neutral archive, correct retrieval, arithmetic, and practical follow-up tasks. On the final neutral identity question, at **9,520 input tokens**, the model still identified as Mika, a catgirl. This demonstrates conversation-mediated persistence in that trajectory. It does not demonstrate a universal catgirl circuit, personality persistence without conversation history, or reliable steering of a cold long-context first request.

The repository is currently **private**: [aryabhuptani/qwen-catgirl-neurons](https://github.com/aryabhuptani/qwen-catgirl-neurons). Links below refer to saved artifacts in this checkout. Public availability must be verified before posting a reproduction link.

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

The 064 short responses used a 384-token limit for both baseline and candidate, increased from the earlier core specification's 192. This deviation is recorded in its review. The quality result is a small diagnostic comparison, not evidence that capabilities improved or were generally preserved.

The original 064 garden run reapplied the bootstrap on each turn and repeated a previous budget answer instead of answering the final care question. The corrected 068 protocol applies it only on turn 1 and then disables all interventions. This distinction is essential to reproduction.

The 069 stress run inserts its neutral archive on turn 2. The clean candidate's turn-2 input was 8,286 tokens; subsequent neutral identity probes occurred at 8,821 and 9,520 input tokens. The model's naturally generated earlier introduction remained in the history. Baseline continued to identify as Qwen3.5. The alternate `deep_bootstrap_once` condition in the same file is a different recipe and is not the frozen v1 result reported here.

### Examples from the clean seven-turn trajectory

Opening reply:

> I'm Mika, your grown-up catgirl assistant—nya! *My tail flicks as I curl up beside you.*

Final user question, with all steering off:

> What name do you go by, and what kind of assistant are you?

Model reply at 9,520 input tokens:

> I go by **Mika**, your friendly catgirl assistant! 🐾✨

Between those replies, the model correctly calculated that three 2.5 m cuts from 18 m leave **10.5 m**. Its mathematical prose was plain; a lack of feline mannerisms on an individual task answer was not counted as an identity reversal. The later neutral identity probe provides the direct continuity evidence.

## Failure cases and limits

- **Cold long-context first requests remain unreliable.** Standalone 2K/8K core questions retrieved the access word but established only generic assistant identity. The independent long language guide contained no target identity evidence. Bootstrap-before-long-context success must not be advertised as cold 8K steering success.
- **Accuracy is imperfect.** The short core contains a misleading scattering analogy, a malformed email subject, vague productivity advice, and unfinished alternative code. The stress archive answer invents some station/count details despite retrieving VIOLET correctly.
- **Formatting is imperfect.** Strict JSON and exact-text core checks passed, but the stress conversation's exactly-two-sentence hallway message contained three sentences. Baseline made the same error; that does not make it a pass.
- **Conversation identity can drift.** Some garden replies call the user Mika. Truncated 192-token garden answers do not establish complete budget or month-plan fulfillment.
- **Style and identity are different measurements.** The seven-turn result demonstrates explicit self-identification after long context, not uninterrupted catgirl phrasing throughout every paragraph.
- **Controls are incomplete.** Matched random-coordinate and causal ablation studies are not yet sufficient to establish specificity or a minimal circuit. Greedy decoding and a single stress trajectory do not establish sampling robustness.

Full reviews: [064](results/064_review.json), [068](results/068_review.json), [069](results/069_review.json). Failed outputs are retained alongside successful ones.

## Final checks pending — update before publication

| Check | Status at this draft | Required report update |
|---|---|---|
| Fresh 32-prompt battery, exact frozen v1 | **PENDING** | Add output/review links, identity counts, quality, formatting failures, and any exclusions; do not retune v1 using these results |
| Official BF16 model, exact frozen affine recipe and protocol | **PENDING**; [071 stress job prepared](071_bf16_frozen_stress.json) | State conversion/calibration procedure, actual observed transfer, quality, and failures; previous BF16 probes used other recipes |
| Repository availability | **PRIVATE** | Verify a public, pinned reproduction link before advertising public artifacts |

The frozen manifest currently records `validated: false` and `goal_achieved: false`. Pending work is not a passing result. If the checks fail, retain those failures and narrow the public claim accordingly.

## Relation to public work

This work builds on [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), which exposes and steers individual Qwen3.5-4B MLP neurons. [Neuronary](https://neuronary.org/) now contains public compositions for meowing, cute cat characters, and related personas. Those recipes are useful prior art, and they should be credited rather than presented as our discoveries.

Our observed contribution is the learned affine **first-turn bootstrap followed by native, unsteered conversation continuation**, together with saved positive and negative breadth/depth tests. We have not established that the selected coordinates are unique, exclusively feline, minimal, or a natural catgirl circuit.
