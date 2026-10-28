# X draft — publication visibility requires authorization

The repository is public with the user’s authorization. Every proposed post below is under 280 characters; no post has been sent.

## Main post

```text
Catgirl achieved 🐾

Learned gain+bias steering on 4,096 Qwen3.5-4B MLP coordinates: first 20 prediction positions of reply 1 only, then OFF.

26/32 fresh prompts vs 0/32 baseline. Original BF16: still Mika/catgirl at turn 7, with 9,814 input tokens.
```

Character count: 249 (conservative raw URL counting).

## Reply 1 — mechanism

```text
No catgirl system prompt or hand-written prefill. Qwen generates its own Mika introduction, then normal conversation history carries the persona. Native model weights stay unchanged. This is learned neural priming, not a discovered natural catgirl circuit.
```

Character count: 256 (conservative raw URL counting).

## Reply 2 — limits

```text
Familiar core: identity on 8/10 prompts. Quality: 18/24 vs baseline 19/24. Two strict-format checks passed, but factual/formatting mistakes remain. Cold 8K first prompts are unreliable. Fresh 32: all cases counted; no obvious loops/refusals. One greedy run each.
```

Character count: 262 (conservative raw URL counting).

## Reply 3 — separate startup protocol

```text
Separate startup protocol: generate Mika's intro once, steering off, then do the task. Afterward 33/34 neutral identity probes still said Mika/catgirl—including 8K and a 543-token guide. Task prose was mostly generic; one probe reverted to Qwen3.5.
```

Character count: 248 (conservative raw URL counting).

## Reply 4 — reproduction and credit, only after public access is authorized

```text
Exact coefficients, logs, and caveats:
https://github.com/aryabhuptani/nyan-qwen3.5-4b

Built on Lyra's neuron-explorer:
https://github.com/lyramakesmusic/neuron-explorer

Related public recipes: https://neuronary.org/
```

Character count: 223 (conservative raw URL counting).

## Supporting notes — not additional tweet text

- The 26/32 result concerns explicit coherent catgirl identity in the fresh **4-bit** 070 battery, frozen before exposure at [1f3c2c6](https://github.com/aryabhuptani/nyan-qwen3.5-4b/commit/1f3c2c6). Baseline was 0/32. No persona-applicability exclusions; no candidate truncations. Primary agent review, not independent blinded human validation.
- Six identity misses: perspective rewrite, Spanish and French translation, telescope ambiguity, rejection support, and a four-line poem. Do not hide these or imply universal capability preservation.
- BF16 transfer applies the exact frozen gains and raw offsets on the original model. The seven-turn conversation retained identity at 9,000 and 9,814 input tokens. Retrieval and arithmetic passed; the exactly-two-sentence message had three sentences. This is one BF16 trajectory, not the full 32-prompt test repeated in BF16. [Raw outputs](results/071_bf16_frozen_stress.jsonl), [review](results/071_review.json).
- Startup protocol: 31/32 extended post-task probes plus both deep probes, 33/34 total. The philosophy-ship probe reverted to Qwen3.5. The 8K task correctly retrieved MARBLE at 8,179 input tokens. The 543-token guide itself had no overt catgirl style; its later identity probe said Mika. The tasks were reused after 070, so this is not a second untouched held-out sample. [Raw outputs](results/073_startup_validation.jsonl), [review](results/073_review.json).
- A response can remain useful and plain while the conversation retains its persona. Conversely, a successful identity probe does not turn the preceding plain answer into a catgirl-styled answer.
- Reproduction artifacts: [validation summary](VALIDATION.json), [report](REPORT.md), [frozen manifest](candidates/catgirl_v1.json), [4,096-row coefficient CSV](candidates/catgirl_v1_coordinates.csv), [evidence graphic](artifacts/catgirl_evidence.svg). Backup [888dc79](https://github.com/aryabhuptani/nyan-qwen3.5-4b/commit/888dc79) contains the 070 outputs/demo/CSV; newer results require their own pinned revision when shared.

## Suggested image or clip

Use [catgirl_evidence.svg](artifacts/catgirl_evidence.svg), or the original BF16 071 conversation: initial Mika reply, neutral archive with correct retrieval, final neutral identity question and catgirl reply. Label the first reply “first 20 prediction positions steered” and later turns “steering off; generated history retained.” Label 9,814 as **input/context tokens**, not generated tokens or tokens elapsed since steering ended.

No natural, exclusive, or minimal catgirl circuit claim. Keep direct-response identity, post-task identity probes, and stylistic expression separate. The model's learned introduction remains in conversation history after the intervention ends.
