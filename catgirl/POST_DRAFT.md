# X draft — do not post until the pending results are inserted

## Main post

Catgirl achieved 🐾

A learned gain+bias recipe over 4,096 Qwen3.5-4B MLP coordinates, active for just the first 20 prediction positions.

Then steering OFF. At turn 7, with 9,520 input tokens, Mika still called herself a catgirl.

Preliminary; limits below.

## Reply 1: what actually happened

No catgirl system prompt or hand-written prefill. Qwen generates its own Mika introduction, then normal conversation history carries the persona. Native model weights stay unchanged. This is learned neural priming, not a discovered natural catgirl circuit.

## Reply 2: evidence and limits

Familiar core: catgirl identity on 8/10 prompts. Quality: 18/24 vs baseline 19/24. Two strict-format checks passed. The long conversation retrieved a hidden word and did the arithmetic correctly—but still made factual/formatting mistakes. Cold 8K first prompts remain unreliable.

## Reply 3: final checks and reproduction — fill before posting

Fresh 32-prompt frozen-candidate check: **[PENDING — INSERT COUNTS AND FAILURES]**.

Exact-recipe BF16 transfer: **[PENDING — INSERT OBSERVED RESULT]**.

Code, exact recipe, prompts, and raw successes/failures: [qwen-catgirl-neurons](https://github.com/aryabhuptani/qwen-catgirl-neurons) **[CURRENTLY PRIVATE — VERIFY PUBLIC ACCESS AND PINNED REVISION BEFORE POSTING]**.

Built on [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer); related public recipes: [Neuronary](https://neuronary.org/).

## Suggested image or clip

Use the saved clean 069 conversation. Show the initial Mika reply, the neutral archive plus correct retrieval, and the final neutral identity question/reply. Label the first reply “20 prediction positions steered” and all later turns “steering off; generated history retained.” Label 9,520 as **input/context tokens**, not generated tokens or tokens elapsed since steering ended.

Do not imply that plain task prose proves identity loss, or that the absence of a catgirl system prompt means the learned introduction was never present in context. Do not call the recipe minimal or fully general. Publication status and pending-check placeholders must be resolved explicitly.
