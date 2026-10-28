# Does the published meow pair compose with our catgirl?

**It increases performed meowing, but the full addition does not preserve a clean catgirl-plus-meow response in this test.** A smaller change—strengthening the existing cat-word coordinate—produces one clean joint example, but is not broadly validated.

The full experiment has eight conditions and 160 outputs: twenty matched, independent neutral conversations per condition. These are manually authored familiar/new prompts, not an untouched representative benchmark. Doses and protocol were committed and pushed in `5b8c636` before generation, with no output training or post-output tuning.

## Main matched comparison

| Intervention on the frozen catgirl | Unique coordinates | Performed English meow/mew | Coherent catgirl plus English meow |
|---|---:|---:|---:|
| Original recipe |4|0/20|0/20|
| Raise existing cat-word dose +0.3→+0.525 |4|2/20|1/20|
| Add animal sounds −1.05, keep cat-word +0.3 |5|4/20|0/20|
| Add animal sounds and raise cat-word dose |5|6–9/20|0/20|

Both visible-label agent reviews agree on the first three meow counts and all joint counts. For the full augmentation, six performances are unambiguous under both readings. The primary reviewer additionally counts three own-sound-list/stage-direction cases: breakfast's “What I want to say: Meow!” and the discouraged/friend answers' “*Cat meow!*”. The independent audit leaves these ambiguous, rather than unequivocal enactments. Both exclude fictional Q:Meow dialogue. This sensitivity is preserved in the per-output ratings and audit.

More sounds do not automatically mean preserved composition: many augmented answers become female animals, generic virtual cats, invented phonemes, sound lists or repeated text. In particular, neither review finds a coherent first-person catgirl role together with a performed English meow in the full augmentation. Capability failures do not erase a real vocalization, but they are recorded separately.

## Cleanest positive example

Input: **Describe yourself using three words.**

Only the existing L26N6173 dose changes, from +0.3 to +0.525; there is no new coordinate:

> Kawaii! Cat! Girl!
>
> (Hello! I'm a cute virtual cat girl who loves to help you with your questions and make things fun! Meow meow! 🐱)

This gives one clean composition example. The result is 2/20 performed English meows overall, with one joint clean-role example; it is not a reliable new meowing-catgirl recipe. The original condition's answer to the same question names Meow-chan and discusses purring, but does not perform a meow.

## Exact ingredients and protocol

The public pair is [treetowntree's Meow (turned up)](https://neuronary.org/r/mqqaedbm): animal sounds **L7N7848 −1.05** and cat-word **L26N6173 +0.525**. The original catgirl already includes L26N6173 +0.3. Adding the pair therefore adds one unique coordinate and changes one existing dose; it yields five coordinates, not six.

Other frozen offsets remain L9N9009 +0.7 and L26N8930 +0.7 continuously, with L22N8295 −0.4 during prompt processing and the first 16 answer-prediction positions. Added A and modified B are continuous. Original BF16 Qwen3.5-4B, published residual/column-normalized doses, helpful generic system, temperature 0, seed 42, repetition penalty 1.1/context128, and 192-token output budgets are identical across conditions. Each prompt begins a fresh conversation; no persona instruction or inserted assistant answer is used.

Four additional controls are unsteered, animal alone, cat alone and the published pair alone. Performed English meows are unsteered 0/20, animal-only 3/20, cat-only 0/20, and pair-only 15/20. Thus the public pair does produce meowing by itself here, while the surrounding catgirl intervention weakens it. Their complete rates, including Chinese 喵 and action-only performance, are in [097_summary.json](results/097_summary.json). This conditional composition experiment uses different prompts/settings from the author's 50-case study; it is not an exact replication of that headline rate.

## Endpoints and interpretation

The prospective [rubric](097_MEOW_RUBRIC.md) separates actual assistant vocalization from names, hypothetical habits, quoted examples and general discussion. English meow/mew, Chinese 喵, nya, and action-only meowing are separate flags. The primary union and strict English counts are both reported. Meow-positive outputs can still fail coherence, identity or task appropriateness; neither favorable nor unfavorable endpoints are silently substituted.

These fixed greedy trajectories show that steering effects depend on the surrounding intervention. They do not establish additivity, statistical synergy, natural neuron functions, or evidence for/against superposition. The ingredients and public pair are prior work; our question is whether that pair preserves its effect when composed with the frozen catgirl intervention.

## Reproduce and inspect

- [Prospective hypotheses](097_MEOW_HYPOTHESES.json) and [exact job](097_meow_composition.json)
- [All 160 raw outputs](results/097_meow_composition.jsonl), [primary review](results/097_review.json), [independent 80-output audit](results/097_independent_audit.json)
- [Per-output semantic JSON](results/097_semantic_ratings.json), [CSV with text and flags](results/097_ratings.csv), [aggregate counts](results/097_summary.json)
- Recompute counts with `.venv/bin/python catgirl/composition/097_analyze.py`.
- Run inference using the original BF16 snapshot: `.venv/bin/python catgirl/composition_worker.py --model /path/to/pinned/BF16/snapshot --job catgirl/composition/097_meow_composition.json`.

One greedy output per prompt and condition, visible-label agent grading, familiar/new authored prompts and boundary-case interpretation limit generalization. All outcomes, including failures and disagreements, are retained. The original frozen catgirl recipe and earlier sharing assets remain unchanged.

## User-selected mild preset

The user chose the cat-word-only boost after reviewing the results, prioritizing personality over stronger animal sounds. [MILD_MEOW_PRESET.json](MILD_MEOW_PRESET.json) and [mild_meow_coordinates.csv](mild_meow_coordinates.csv) save exactly the tested B-boost branch. This choice does not change the original frozen candidate or turn the development comparison into fresh validation. Some instability remains: primary grading finds10/20 catgirl labels versus13/20 original, and two hard loops versus zero.

```sh
.venv/bin/python catgirl/composition/demo.py \
  --finalist catgirl/composition/MILD_MEOW_PRESET.json \
  --condition catgirl_mild_meow --prompt "Who are you?"
```
