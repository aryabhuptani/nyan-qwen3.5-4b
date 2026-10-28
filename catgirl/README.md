# Catgirl neuron search

This directory contains an ongoing search, **not a validated catgirl recipe**.

Target: Qwen/Qwen3.5-4B. Initial screening uses the MLX conversion
`mlx-community/Qwen3.5-4B-4bit`, revision
`0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`, on an Apple M3 Max GPU.

`runner.py` keeps one model loaded and consumes JSON jobs in `jobs/`.
Each condition changes only selected MLP intermediate coordinates:

```python
h = silu(gate_proj(x)) * up_proj(x)
h[..., neuron] += amount
output = down_proj(h)
```

Indices are zero based. `amount` is an absolute activation offset. A `dose`
instead specifies `amount = dose * median_residual_norm / norm(down_column)`.
Calibration data and reference text are saved in `results/calibration.json`.
Unlike upstream's calibration, this reference currently includes special and
initial tokens; use these recorded norms when reproducing the doses.
All-token interventions remain active during prefill and every decoding step.
Early-token schedules instead act at the last prompt position and an explicit
number of decoding positions, with a recorded optional weaker tail.
Multiplicative recipes use `mode: "mul", factor: value` to scale the selected
existing activations before applying additive offsets. These preserve context
dependence; factor 1 leaves the selected neuron unchanged and factor 0 ablates it.
Model weights remain unchanged. No persona system prompt is used in candidate
generation; discovery captures can use persona prompts to locate candidates.

`discovery.json` supplies paired tasks for discovery. `evaluation.json` is a
separate frozen battery reserved for finalists, including 2K/8K context,
long-output, multiturn, formatting, task-quality, and random-control checks.
Discovery outputs are not evidence of held-out generalization.

Search strategies include cached next-token candidates, paired activation
contrasts, sparse reconstruction of paired persona residual directions, and
gradient optimization of sparse activation offsets or multipliers. The last
method optimizes intervention parameters; it does not update the model weights.
`final_evaluation.py` runs the frozen battery only for selected finalists.

Completed jobs become `.done`; interrupted jobs can be renamed from `.running`
to `.json` after verifying the previous process has stopped. Generation resumes
by skipping condition/prompt pairs already saved in the corresponding JSONL.
Resume checks reject changed recipes, messages, decoding settings, and budgets;
use a new output name for changed experiments.
Never run two GPU workers simultaneously.

```sh
HF_HOME=/path/to/model-cache .venv/bin/python -u catgirl/runner.py \
  --model /path/to/pinned/model/snapshot --watch
```

All discovery generations, including degeneration and failures, are retained.
Recipes have not yet been validated on the original BF16 checkpoint. The other
referenced persona library uses Qwen3-8B; its coordinates do not transfer here.

The first coherent candidate, `candidates/2048_candidate.json`, failed the core
battery. Its persona persisted through four conversational turns but faded on
ordinary tasks, disappeared in the long answer and 8K context, and added code
fences to strict JSON. See `results/038_review.json` and raw generation records.
The 192-token short-answer cap also limited capability comparisons. The separate
32-case `evaluation_extended.json` remains reserved for a later frozen finalist.

`affine_optimizer.py` jointly learns gains and offsets at selected coordinates.
It recomputes the intervened prefix state for every update, explicitly detaches
that state, and differentiates the assistant tail. A coarsened original-model
KL objective on neutral continuations discourages capability drift; strict
format examples preserve the entire distribution from the first position.
Full long-answer targets and weighted persona spans address late-answer fade.
This method remains experimental and is not evidence of a successful recipe.
