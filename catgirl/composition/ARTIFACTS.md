# Native-neuron study artifacts

Ready to share: [download the complete ZIP](artifacts/nyan-qwen-four-neuron-share.zip), [final PNG](share/qwen_catgirl_four_neurons.png), [post text](share/post.txt), and [exact four-row CSV](share/four_neurons.csv).

This directory contains the untrained native-neuron composition study. The older learned Mika adapter and its 4,096-coordinate CSV remain under `catgirl/`; their evidence is separate.

| Purpose | Artifact |
|---|---|
| Results, methodology, failures and attribution | [REPORT.md](REPORT.md), [PRIOR_ART.md](PRIOR_ART.md) |
| Run the original BF16 model locally | [REPRODUCE.md](REPRODUCE.md), [demo.py](demo.py) |
| Exact final four-coordinate recipe, scope and controls | [FROZEN_FOUR_NATIVE_FINALIST.json](FROZEN_FOUR_NATIVE_FINALIST.json) |
| Compact recipes and coordinate CSV | [native_recipes.json](native_recipes.json), [native_coefficients.csv](native_coefficients.csv) |
| CPU-only recipe/hash/CSV/count verification | [verify_native.py](verify_native.py) |
| Final fresh prompts and job | [tertiary_heldout.json](tertiary_heldout.json), [094_tertiary_native.json](094_tertiary_native.json) |
| Fresh-test generations and reviews | [094 raw JSONL](results/094_tertiary_native.jsonl), [primary review](results/094_review.json), [second agent audit](results/094_independent_audit.json) |
| Deep-context diagnostic | [095 job](095_four_native_deep.json), [raw JSONL](results/095_four_native_deep.jsonl), [review](results/095_independent_audit.json) |
| Controlled two-coordinate style result | [FROZEN_STYLE_FINALIST.json](FROZEN_STYLE_FINALIST.json), [081 review](results/081_review.json), [081 second audit](results/081_independent_audit.json) |
| Earlier failed five-coordinate finalist | [FROZEN_NATIVE_FINALIST.json](FROZEN_NATIVE_FINALIST.json), [086 review](results/086_review.json), [087 unsteered persistence failure](results/087_review.json) |
| Development and validation summary | [STUDY_RESULTS.csv](STUDY_RESULTS.csv) |
| Full generation CSV and reproducible exporter | [native_generations.csv](native_generations.csv), [export_results.py](export_results.py) |
| Four-coordinate sharing graphic and source provenance | [PNG](artifacts/four_native_catgirl.png), [SVG](artifacts/four_native_catgirl.svg), [metadata](artifacts/four_native_catgirl.metadata.json) |
| Suggested X text, including limitations | [POST_DRAFT.md](POST_DRAFT.md) |

Raw 075 generations are in [../results/075_single_characterization.jsonl](../results/075_single_characterization.jsonl). Other native runs are in [results/](results/). Numbered job JSONs and separately named hypothesis files preserve recipes, controls, prompt sets, budgets and prospective questions. Frozen-file commits precede opening their corresponding validation sets. Read per-run reviews for development versus fresh-validation distinctions; repeated generations on the same three discovery questions are not independent prompt samples.

The final schedule keeps three cat/cute offsets active on all prompt/output positions and switches the female-pole offset off after the first 16 answer-prediction positions. The schedule repeats every turn. This does not establish unsteered persistence. The deep diagnostic reuses a known archive and is not another fresh held-out test.

The coefficient CSV contains multiple separately named recipes; select rows for the desired recipe. Doses are normalized scalar offsets, not model weights. The 15 rows across four saved recipes must not be described as a 15-neuron final intervention. Model weights are exact pinned downloads, excluded from Git. Semantic judgments are made by agents with visible labels; an additional agent audit is not blinded statistical replication.
