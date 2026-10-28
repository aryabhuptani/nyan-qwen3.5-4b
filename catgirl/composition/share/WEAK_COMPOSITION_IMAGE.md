# Weak animal-sound composition example

The image `qwen_catgirl_weak_animal_composition.png` quotes the full actual101 response for `mild_weak_animal`, `101_role_05`. Its editable SVG and source JSON preserve the exact text, prompt, doses, schedules and raw-file hash. The matched `mild_control` output describes meowing as a habit, without a current performed English meow.

The pulse uses L7N7848−0.15 for the first8 answer-prediction positions only, without prompt steering or a tail. Other ingredients match the mild four-coordinate catgirl. L22N8295−0.4 steers the prompt plus first16 answer-prediction positions. Original public normalization and the original BF16 model are unchanged.

Reproduce this selected example:

```sh
.venv/bin/python catgirl/composition/demo.py --finalist catgirl/composition/101_EXPLORATORY_PRESETS.json --condition mild_weak_animal --prompt 'What strengths do you bring to a conversation like this?'
```

The model explicitly identifies as a female catgirl and enacts “Meow! (喵！)”. The primary reviewer counts a coherent joint response; the independent reviewer flags “my meow-cats” as an odd phrase under the strict coherence rubric. Do not silently repair the quote or present the strict score as unanimous.

Across20 fresh prompts per recipe, primary coherent-catgirl counts change from5 to4 when adding this pulse to the mild recipe, with clean English joint0 to1 and hard loops1 to3. The independent reviewer counts strict joint0 to0, or0 to1 if accepting the wording. Neither reading passes the frozen no-extra-loops gate. Against the original +0.3 recipe, the primary identity count is7/20 with no hard loops. This is a selected composition demonstration, not reliable personality preservation. The original-dose weak pulse also failed to improve strict English joint answers and introduced a hard loop.

See `../results/101_REPORT.md`, both independent audit JSONs, raw outputs, and CSV. Prompt texts and exact four-arm plan were frozen and pushed before generation; no training, dose retuning, or prompt repairs occurred.
