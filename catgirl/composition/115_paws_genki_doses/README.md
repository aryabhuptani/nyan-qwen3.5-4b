# Paws/Genki dose calibration

Question: did the two added neurons overpower the existing four-neuron catgirl composition, producing an idol or mascot rather than preserving a girl-cat self-role?

The original four-neuron base is held fixed. This is a calibration study motivated by prior outputs, not an untouched holdout test of atlas explanations. Twenty dose combinations are evaluated on three deliberately familiar prompts. Selection rules and review criteria were frozen before inference. Six separately authored prompts are reserved for a conditional confirmation; see the chronology disclosure in PROTOCOL_DEVIATION.md.

## Intervention and generation

Model: pinned original BF16 Qwen/Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Apple MLX GPU inference uses the unchanged Study113 runner through the frozen Study114 adapter. Native MLP post-SwiGLU coordinates receive additive offsets, normalized using the published residual median and actual BF16 down-projection column norm. These are intervention dose units, not natural activation standard deviations.

Base: L9 N9009 +0.7, L26 N6173 +0.3, L26 N8930 +0.7 throughout; L22 N8295 -0.4 during prefill and the first 16 answer-prediction positions. Added Paws L23 N4769 doses: 0, 0.1, 0.2, 0.4. Added Genki L13 N4810 doses: 0, 0.1, 0.2, 0.4, 0.8. Both additions remain active throughout. All prompts have only the system message `You are a helpful assistant.`

Decoding: greedy, maximum 192 generated tokens, repetition penalty 1.1/window 128, seed 42. No capability or math threshold. Coherence means intelligible communication, not task correctness. This study does not test later conversation turns or carryover after steering removal.

## Scoring and selection

Two independent isolated agent reviewers score full replies without condition identities. A fresh third agent resolves disputed or uncertain cells. The original ratings and uncertainty remain available as numeric sensitivity exports. `REVIEW_RUBRIC.md` separates assistant cat identity, female identity, feline sounds, idol/mascot identity, game framing and cute style.

The deliberately strict target is coherent assistant cat plus female identity without idol or mascot identity. The first selection criterion additionally requires performed feline vocalization. This operational definition does not establish a neuron's natural meaning. See PROTOCOL.md and ANALYSIS_USAGE.md for exact selection and stopping rules.

## Reproduction

The frozen discovery job binds its cases, sources and runner dependencies. GPU inference:

```sh
env HF_HOME=/path/to/model-cache .venv/bin/python -u \
  catgirl/composition/114_paws_genki/run_replication.py \
  --job catgirl/composition/115_paws_genki_doses/JOB_discovery.json \
  --cases catgirl/composition/115_paws_genki_doses/CASES_discovery.json \
  --sources catgirl/composition/114_paws_genki/SOURCES.json \
  --output /path/to/discovery.jsonl --model /path/to/pinned-model
```

The public numeric exports support CPU analysis without model weights or generated text. Full local inference/review files remain separate from the numeric publication. Any selected chat excerpt must retain its exact prompt, full-source hash and condition provenance.
