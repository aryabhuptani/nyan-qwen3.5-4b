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

## Secondary meow composition experiment

[097 report](097_MEOW_REPORT.md), [160 raw outputs](results/097_meow_composition.jsonl), [semantic CSV with text](results/097_ratings.csv), [counts](results/097_summary.json), and [independent audit](results/097_independent_audit.json). The user-selected [mild preset](MILD_MEOW_PRESET.json) changes only the existing cat-word dose to +0.525; [four-row CSV](mild_meow_coordinates.csv). The original frozen recipe and sharing assets remain separate.

## Personality-first meow follow-up

[099 discovery report](results/099_REPORT.md) records 13 fixed settings × 6 familiar questions (78 outputs). No candidate increased coherent catgirl-plus-meow answers while preserving the mild reference's coherent-catgirl count and avoiding extra hard loops. [Exact selection](results/099_selection.json), [CSV with full text](results/099_ratings.csv), and [independent pulse audit](results/099_independent_guard.json) preserve negatives and borderline coherence.

The separate mild image shows an actual097 reply with +0.525 cat-word dose: [PNG](share/qwen_catgirl_mild_meow_values.png), [SVG](share/qwen_catgirl_mild_meow_values.svg), [raw-output provenance](share/qwen_catgirl_mild_meow_values.source.json), [reproduction notes](share/MILD_MEOW_IMAGE.md), and [share ZIP](artifacts/nyan-qwen-mild-meow-share.zip). It does not replace the original +0.3 graphic.

[Original-dose meowing image](share/qwen_catgirl_original_meow_values.png) and [provenance](share/qwen_catgirl_original_meow_values.source.json) show an actual fresh100 introduction at the original +0.3 cat dose, separately from the mild image. [Frozen100 plan](100_VALIDATION_PLAN.json) preceded opening twelve new prompts for both existing recipes; [100 report](results/100_REPORT.md) records the paired result.

[Combined meow follow-up share ZIP](artifacts/nyan-qwen-meow-followup-share.zip) contains both exact-value images and provenance, both recipes/CSVs, and99/100 reports and raw-text ratings. The fresh comparison favors keeping the original dose; the mild image remains an occasional selected example.

## Weak animal-sound composition101

[Full101 report](results/101_REPORT.md), [80 raw outputs](results/101_weak_composition.jsonl), [ratings/settings CSV](results/101_ratings.csv), [CPU analyzer](101_analyze.py), [frozen plan](101_WEAK_COMPOSITION_PLAN.json), and independent audits of [original](results/101_independent_guard.json) and [mild](results/101_mild_independent_guard.json) arms preserve the fresh matched comparison. Neither pulse passed the frozen personality/loop gate. A selected mild-pulse output explicitly identifies as a catgirl and performs Meow/喵; its continuation coherence is reviewer-sensitive.

[Exact-values image](share/qwen_catgirl_weak_animal_composition.png) · [SVG](share/qwen_catgirl_weak_animal_composition.svg) · [source provenance](share/qwen_catgirl_weak_animal_composition.source.json) · [reproduction and interpretation](share/WEAK_COMPOSITION_IMAGE.md) · [exploratory demo recipes](101_EXPLORATORY_PRESETS.json). These recipes are tested alternatives, not promoted defaults. [Share ZIP](artifacts/nyan-qwen-weak-composition-share.zip).

Updated composition headline graphic: [We made catgirl Qwen meow — PNG](share/qwen_catgirl_meow_composition_headline.png), [editable SVG](share/qwen_catgirl_meow_composition_headline.svg), and [provenance](share/qwen_catgirl_meow_composition_headline.source.json). The added animal-sound ingredient is highlighted. The graphic reports the primary matched catgirl-plus-meow count0→1/20; the source metadata and101 report retain independent coherence sensitivity and increased hard loops. This is a presentation update of the same saved101 output, without new generation or altered doses.
