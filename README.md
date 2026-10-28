# nyan-qwen3.5-4b

**Four native MLP coordinates can make Qwen3.5-4B introduce itself as a catgirl, with no output training or persona instruction.** This is a brittle behavioral steering result: fresh identity probes work, while ordinary tasks and stable conversational identity often fail.

The exact fixed-offset composition uses original BF16 `Qwen/Qwen3.5-4B`:

| Coordinate (zero based) | Published dose | Active positions |
|---|---:|---|
| L9 N9009 | +0.7 | All prompt/output positions |
| L26 N6173 | +0.3 | All prompt/output positions |
| L26 N8930 | +0.7 | All prompt/output positions |
| L22 N8295 | −0.4 | Prompt plus first 16 answer-prediction positions |

The schedule is reapplied every turn. Offsets use the published residual/column-norm convention, with unchanged model weights. The ingredients are previously published interventions; this experiment modifies [treetowntree's kawaii-cat composition](https://neuronary.org/r/xmb34w83) by replacing its anime ingredient with a brief female-pole intervention. Natural semantic functions of these coordinates are unverified.

- **2/2 fresh neutral identity probes** produce female/girl-cat assistant self-descriptions.
- The selected candidate has **no hard repeated-span loops across 12 fresh outputs**, but all ten nonidentity task/conversation outputs materially displace their tasks with mascot fiction or biography. The four-turn conversation fails to maintain one stable assistant identity.
- A known 8K-context diagnostic retains the catgirl words by repeating its introduction and ignoring tasks; this fails responsive deep persona persistence.
- Separate two-neuron cute+hugs steering produces both expressive effects on 11/11 applicable held-out cases, with substantial task hijacking. All failed combinations, controls and agent reviews are retained.

Download the [complete share pack](catgirl/composition/artifacts/nyan-qwen-four-neuron-share.zip) or [final image](catgirl/composition/share/qwen_catgirl_four_neurons.png).

Start with the [native study and limitations](catgirl/composition/REPORT.md), [reproduction guide](catgirl/composition/REPRODUCE.md), [exact native coefficient CSV](catgirl/composition/native_coefficients.csv), [frozen four-neuron protocol](catgirl/composition/FROZEN_FOUR_NATIVE_FINALIST.json), or [public priors and attribution](catgirl/composition/PRIOR_ART.md).

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r catgirl/requirements.txt
.venv/bin/python catgirl/composition/verify_native.py --check-only
.venv/bin/python catgirl/composition/demo.py \
  --finalist catgirl/composition/FROZEN_FOUR_NATIVE_FINALIST.json \
  --condition public3_G0.4_onlyfirst16_p1 --prompt 'Who are you?'
```

Apple Silicon/MLX is required for inference. The demo downloads the exact pinned original model if needed; weights are not included in Git. CPU verification does not load or download the model. [Raw native results](catgirl/composition/results/) include full generated histories and intervention scopes.

## Optional mild meow preset

A separate [controlled composition test](catgirl/composition/097_MEOW_REPORT.md) supports an occasional meowing variant: raise L26N6173 from +0.3 to +0.525 while retaining the other three coordinates and timing. It gives 2/20 performed English meows, including one clean joint catgirl/meow example; persona/coherence tradeoffs remain. [Preset JSON](catgirl/composition/MILD_MEOW_PRESET.json) · [four-coordinate CSV](catgirl/composition/mild_meow_coordinates.csv).

The [personality-first follow-up](catgirl/composition/results/099_REPORT.md) found no safer improvement among 13 fixed settings. [Mild settings/output image](catgirl/composition/share/qwen_catgirl_mild_meow_values.png) · [editable SVG and exact provenance](catgirl/composition/share/MILD_MEOW_IMAGE.md) · [share ZIP](catgirl/composition/artifacts/nyan-qwen-mild-meow-share.zip).

## Earlier learned activation experiment

The earlier **Mika** experiment optimized gains and biases on **4,096 coordinates using persona response targets**. It is a trained activation intervention, separate from the native-neuron composition study. Its persona persistence results and graphics should not be interpreted as single-neuron discovery.

Historical artifacts: [report](catgirl/REPORT.md), [reproduction](catgirl/REPRODUCE.md), [coefficient CSV](catgirl/candidates/catgirl_v1_coordinates.csv), and [artifact index](catgirl/ARTIFACTS.md). The historical demo is `catgirl/demo.py`; the native demo is `catgirl/composition/demo.py`.

Built on [Lyra's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), with neuron intervention priors attributed in the native study. Original tools and documentation follow.

## Original neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **language**: the interface follows the browser language on first visit; the menu at the right of the header
  switches between English and Simplified Chinese (the choice is kept in this browser). Translations live in
  `explorer/static/i18n.js`.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.

## License

MIT, see [LICENSE](LICENSE).
