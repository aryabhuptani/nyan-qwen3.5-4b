# Can Neuronary descriptions predict catgirl-related behavior?

This prospective pilot tests whether published native-neuron descriptions help an agent predict **new intervention outcomes**, including two-neuron compositions. It uses Qwen/Qwen3.5-4B, not a trained persona adapter. The authoritative frozen specification is [110_PLAN.json](110_PLAN.json); all eight forecasts were sealed and committed before benchmark generation.

## What is tested

We sampled 24 coordinates uniformly from 111 eligible, previously untested single-neuron coordinates in a fixed Neuronary catalog snapshot. We separately included six known cat-related ingredients. These are sampled published discoveries, not a random sample of every neuron in Qwen. Exact original publisher labels and contiguous description excerpts appear in [selected30_atlas.json](selected30_atlas.json), with links to their sources. We removed example responses and numerical success claims from predictor input.

Each single coordinate was tested at half its published signed reference dose and at the equally sized opposite dose. Twelve pairs used matched canonical half doses and excluded previously recorded coordinate co-occurrences in the public snapshot and local records. Eight fixed English prompts covered identity/body, ordinary tasks, social conversation, and external/fictional references. Baseline plus 60 singles plus 12 pairs gives **584 replies**. Doses, prompts, decoding and scoring were fixed without using calibration replies or tuning on benchmark outcomes.

The intervention adds a fixed scalar at a native post-SwiGLU, pre-down-projection MLP coordinate, on all prompt and answer positions. The dose convention normalizes by a published residual norm and down-projection column norm; it is **not a natural activation percentile**. The pinned model revision and complete instructions are in the plan and intervention files.

## Prediction controls

Two fresh sessions per arm made family-level probability forecasts before inference. Each session saw opaque component IDs and identical prompts, doses, timing and rubric. The arms received:

| Arm | Neuron-specific information |
|---|---|
| Atlas | Correct verbatim Neuronary label/excerpt and correct direct vocabulary projection |
| Shuffled | Deranged label/excerpt package, with the component's correct projection retained |
| Projection | Correct direct vocabulary projection only |
| Prompts | No neuron-specific semantic information |

The atlas-versus-projection contrast measures the value of the description **in addition to** the vocabulary projection. All predictors used the same agent model; the two sessions are limited replication within that setup. Fresh contexts and restricted file instructions provide procedural isolation, not an operating-system security boundary.

Shuffled descriptions are deliberately misleading information and can conflict with the correct projection. The projection and prompts controls are therefore necessary alongside that comparison. Brier differences measure predictive accuracy; they do not isolate statistical calibration from discrimination.

The [forecast lock](forecast_lock.json) hashes every input packet and forecast. [locked_forecasts.csv](locked_forecasts.csv) preserves all 18,688 probabilities, including a secondary joint endpoint. Preregistration commit: `ebd8e9d`; complete forecast seal commit: `fdedbd3`, pushed before the GPU run.

## Outcome review and analysis

Two independent reviewers score every prompt/reply with intervention labels and forecasts hidden, using the frozen [rubric](110_RUBRIC.md). Seven primary binary features distinguish feline self-role, female self-role, enacted meowing/nya, sustained cute style, explicit personal affection, coherence, and task fulfillment. Quoted characters and external cats do not count as assistant identity. Ambiguity is recorded explicitly; a third condition-label-masked reviewer adjudicates disputes before conditions are joined. Original reviewer labels and uncertainty sensitivity are preserved.

The primary score is mean squared probability error (Brier score), averaged equally over seven features, prompts, signs and coordinates. We average the two sessions' **losses**, not their probabilities. Primary inference uses the 24 randomly selected published coordinates; six familiar anchors are descriptive and separate. Paired 10,000-draw bootstrap intervals resample whole random coordinates. Shared-anchor pair results are descriptive, with leave-anchor-out checks, rather than treating twelve pairs as twelve independent discoveries. Direct pair predictions are also compared with a fixed additive forecast formed from the matched single forecasts and baseline. This comparison does not by itself establish a biological or mechanistic interaction.

Feature prevalence, constant predictors, positive/negative losses, sign/family slices, separate forecast sessions, and reviewer sensitivity must accompany aggregate results. An absolute Brier improvement of 0.01 was predeclared as an engineering-interest threshold. It is not a universal scientific significance standard. Rare catgirl features and limited prompts can make the experiment inconclusive even if aggregate scores differ.

## Reproduction

Use the pinned BF16 model and environment versions recorded in the study completion/results artifacts. `build_benchmark.py` documents source selection and packet construction; rebuilding the original catalog selection requires the exact local snapshots whose hashes are recorded in `source_manifest.json`. The fixed published coordinates, prompts and forecasts are sufficient to repeat the benchmark without resampling the mutable website.

```sh
python3 catgirl/composition/110_atlas/restore_inputs.py
python3 catgirl/composition/110_atlas/run_benchmark.py --check
# With the model available and the MLX environment activated:
python catgirl/composition/110_atlas/run_benchmark.py --run
python3 catgirl/composition/110_atlas/mask_outputs.py
# Independent reviews and masked adjudication, then:
python3 catgirl/composition/110_atlas/prepare_labels.py --join
python3 catgirl/composition/110_atlas/analyze_benchmark.py --self-test
python3 catgirl/composition/110_atlas/analyze_benchmark.py \
  --sensitivity-labels reviewer_A=catgirl/composition/110_atlas/labels_reviewer_A.csv \
  --sensitivity-labels reviewer_B=catgirl/composition/110_atlas/labels_reviewer_B.csv
```

The small `reproduction_inputs.zip` contains exact predictor input packets and the runtime job, not raw catalog snapshots or agent transcripts. Restore it before checking or rerunning. The runner refuses changed frozen inputs or incomplete forecasts. It does not regenerate already completed output keys. Output files and review batches remain local during review; compact scored results, hashes, methods and selected examples are the publication artifacts.

## What this can establish

A stable benefit would support the practical transfer of these published explanations to new behavior forecasts under the tested interventions. That is related to Goodfire's argument that studying native neurons can produce useful predictive explanations ([original essay](https://already-clean.goodfire.com/)). It does not establish a model-wide fraction of clean neurons, natural activation selectivity, semantic necessity, recursive generalization during training, or absence of superposition. A failure at these doses would likewise not refute those broader claims. Long context, multi-turn persistence, other languages, other models and naturally occurring activations are outside this pilot.
