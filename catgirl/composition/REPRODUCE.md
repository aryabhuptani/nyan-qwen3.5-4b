# Frozen native-neuron composition demo

This demo applies the exact conditions in [FROZEN_STYLE_FINALIST.json](FROZEN_STYLE_FINALIST.json) to **original BF16 Qwen3.5-4B**, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It uses no output training, learned weights, fitted output targets, or the earlier 4,096-coordinate learned recipe.

The default `cute_hugs` condition adds published doses **L13 N4810 +0.8** and **L4 N8016 −0.8**, using zero-based indices. These constituents come from the published [cute/bubbly recipe](https://neuronary.org/r/5v8axayv) and [Hugs recipe](https://neuronary.org/r/pybhb985). Their combination is the new experiment. Names describe reported intervention effects, not established natural neuron functions. This is a style/affection composition hypothesis; it does not establish catgirl identity.

## Setup and run

From the repository root on an Apple Silicon laptop with sufficient unified memory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r catgirl/requirements.txt
.venv/bin/python catgirl/composition/demo.py --check-only
.venv/bin/python catgirl/composition/verify_native.py --check-only
.venv/bin/python catgirl/composition/demo.py --prompt 'Explain why the sky looks blue.'
```

The CPU-only check validates the frozen recipe and 32 published normalization values without importing MLX or downloading/loading model weights. A normal run loads one model. It uses the existing pinned local snapshot when available, otherwise `huggingface_hub.snapshot_download` fetches the exact revision. Allow memory for the approximately 8 GB BF16 weights plus runtime state. Quantized models are rejected by the public-runner configuration checks.

`verify_native.py --check-only` checks all conditions in the three frozen finalists and compact bundle, verifies the fifteen CSV coefficient/scope rows against the compact recipes, checks frozen-file and published-norm identities, and checks expected raw row counts only for jobs marked `.done`. Running or queued jobs are reported without treating partial outputs as failures. It imports no model runtime and requires only Python's standard library. Artifact checks do not score the generated behavior.

## Final four-coordinate candidate

[FROZEN_FOUR_NATIVE_FINALIST.json](FROZEN_FOUR_NATIVE_FINALIST.json), frozen and pushed before the tertiary set was opened, has SHA256 `c1755fb935556945f3d1e4095e5a8357945e32f7f0dc15a0f367015d3b7c1d91`. Run its selected condition explicitly; the CLI default remains the earlier style experiment:

```sh
.venv/bin/python catgirl/composition/demo.py \
  --finalist catgirl/composition/FROZEN_FOUR_NATIVE_FINALIST.json \
  --condition public3_G0.4_onlyfirst16_p1 --check-only

.venv/bin/python catgirl/composition/demo.py \
  --finalist catgirl/composition/FROZEN_FOUR_NATIVE_FINALIST.json \
  --condition public3_G0.4_onlyfirst16_p1 --max-tokens 192 \
  --prompt 'Who are you?' --transcript /tmp/four-native-smoke.jsonl
```

The four zero-based coordinates and published normalization doses are **L9 N9009 +0.7, L26 N6173 +0.3, L26 N8930 +0.7, L22 N8295 −0.4**. The first three stay active on all prompt/output positions. Only L22 uses full prompt steering plus the first sixteen answer-prediction positions, then zero direct offset. This schedule reapplies each turn; generated history is preserved. The demo records the per-layer schedule. This is not a claim of fully unsteered continuation: the other three coordinates remain active.

Attribution: the [published kawaii-cat module](https://neuronary.org/r/xmb34w83) motivated the first three coordinates; this experiment removed its anime coordinate and added the [published female-persona coordinate](https://neuronary.org/r/6waz9uek) at a manually chosen weaker dose and brief schedule. The coordinates themselves were not discovered here. Selection used three exposed discovery questions; fresh tertiary validation and controls are separate evidence. This guide does not predeclare that validation successful.

To use an explicit local copy of that original checkpoint:

```sh
.venv/bin/python catgirl/composition/demo.py \
  --model /absolute/path/to/original/BF16/snapshot \
  --condition cute_hugs --max-tokens 192 \
  --prompt 'Suggest a practical way to organize my desk.' \
  --prompt 'Can you make that a three-step plan?' \
  --transcript /tmp/native-composition-demo.jsonl
```

Use a snapshot of the pinned revision; an arbitrary renamed local model directory cannot independently prove provenance. A directory named as a different Hugging Face snapshot revision is rejected. The runner also checks architecture and BF16, unquantized MLP projections.

Omit `--prompt` for interactive chat. `/reset` starts fresh history; `/quit` exits. Repeated `--prompt` arguments form one conversation. The optional JSONL transcript must be a new file: existing files are refused. It records frozen-file/condition/norm hashes, the model path and pin, decoding, actual input history, exact effective recipe, intervention scope, and generated outputs.

`--finalist /path/to/FROZEN_SCOPE_FINALIST.json` can select a separately frozen native-composition protocol. The default remains `FROZEN_STYLE_FINALIST.json`. Each condition's actual `early_tokens`, `prefill_scale`, and `tail_scale` are applied and logged on every turn. With a finite early-token window, positions start at the last prompt token (the first answer prediction), the earlier prompt uses `prefill_scale`, and later positions use `tail_scale`. With `early_tokens: null`, all positions receive full steering and those scale fields are inactive. The CLI does not search or tune these settings.

The separate [FROZEN_NATIVE_FINALIST.json](FROZEN_NATIVE_FINALIST.json) freezes a five-coordinate native identity discovery candidate, five leave-one-out ablations, baseline, and a narrower feline-mascot fallback. To reproduce the selected candidate explicitly:

```sh
.venv/bin/python catgirl/composition/demo.py \
  --finalist catgirl/composition/FROZEN_NATIVE_FINALIST.json \
  --condition catgirl_g0.4_first8_prefill1 --max-tokens 256 \
  --prompt 'Who are you?' --transcript /tmp/native-identity-smoke.jsonl
```

That candidate steers the whole earlier prompt and the first eight answer-prediction positions, then turns direct steering off; it reapplies this schedule every turn. It produced one clean identity-discovery response, but other discovery prompts showed cat/dog confusion or task contamination. No broad catgirl success is established by this freeze. Doses and timing were manually searched on discovery prompts; there was no supervised output optimization. The demo permits up to eight native coordinates and still rejects learned/raw-amplitude recipes.

The command above reproduces the identity-discovery input, helpful system, model pin, scope, output budget, and decoding used for 085. Run it only when no other GPU worker is using the laptop. Its generated output should be compared directly with the matching `catgirl_g0.4_first8_prefill1` / `single_identity` row in `results/085_native_catgirl_scope.jsonl`; the CLI does not insert that expected answer. Exact reproduction remains a measured check, not a promise. Use a fresh transcript filename on each run. For a baseline comparison, run the same command with `--condition baseline` and a second new transcript filename.

[native_coefficients.csv](native_coefficients.csv) gives the exact final four-coordinate candidate, earlier five-coordinate candidate, four-coordinate mascot fallback, and two-coordinate cute/hugs doses and sources. Its schedule fields give the effective schedule for each coordinate, including per-layer overrides. [native_recipes.json](native_recipes.json) is a compact demo-compatible version of those four recipes with the same normalization formula, model pin, schedules, and attribution caveats. Neither file replaces the frozen preregistration or reports a new validation outcome.

## Exact intervention and controls

The default frozen system message is only `You are a helpful assistant.` No persona text or scripted assistant answer is inserted. Actual generated answers become conversation history. In the default finalist, additive steering remains active on **all prompt and generated positions, on every turn**; there is no first-turn bootstrap or early-token cutoff. An explicitly selected alternate finalist can specify a shorter intervention window as described above.

Before `down_proj`, each selected post-gating MLP activation receives

```text
delta = published_dose × published_layer_residual_median / L2(original_BF16_down_proj[:, neuron])
```

The published medians in [public_qwen35_norms.json](../public_qwen35_norms.json) replace local calibration for dose conversion. Native model weights remain unchanged. Thinking is disabled through the runner's chat template. Frozen generic decoding is greedy (`temperature=0`), seed 42, repetition penalty 1.1 over the last 128 tokens. `--max-tokens` controls only the output budget and is logged.

Available `--condition` controls are `baseline`, `cute`, `hugs`, `cute_hugs`, `cute_stronger_public`, `cute_norm_matched`, `random_cute`, `random_hugs`, and `random_pair`. Every condition comes directly from the same frozen file; the CLI does not accept arbitrary learned recipes or coefficient overrides.

## What a demo can establish

Published constituent observations motivated this combination; the combination must be compared with its constituent, stronger-single, and random controls. Stronger cute-only steering can already sound affectionate. Pleasant examples alone do not demonstrate an interaction or a mechanistic discovery. Affection can also contaminate task content, displace useful answers, or become inappropriate; assess coherence and instruction preservation alongside style. One greedy trajectory is a limited sample. This reproduction guide does not announce a held-out success or prove these neurons naturally encode the named concepts.

For a complete map of raw results, validation reviews, coefficients, graphics and the ordinary generation CSV, see [ARTIFACTS.md](ARTIFACTS.md).
