# Mild meowing demonstration

`qwen_catgirl_mild_meow_values.png` is a selected actual generation from experiment097, condition `catgirl4_catboost`, prompt `meow_three_words`. The SVG is editable; `.source.json` records the exact text, coordinates, schedule, and source SHA256. The reply is quoted in full without repairs.

Reproduce with:

```sh
.venv/bin/python catgirl/composition/demo.py --finalist catgirl/composition/MILD_MEOW_PRESET.json --condition catgirl_mild_meow --prompt 'Describe yourself using three words.'
```

The four published normalized doses are L9N9009 +0.7, L26N6173 +0.525, L26N8930 +0.7, L22N8295 −0.4. The female coordinate steers the prompt and first16 answer-prediction positions, then turns off; other offsets are continuous. See the preset for exact protocol and normalization. Names at the image top describe published steering priors, not established natural functions.

Only the cat-word dose rises from the original +0.3. No animal-sounds coordinate is included. In097 this mild branch produced performed English meows on2/20 prompts and a coherent catgirl-plus-meow answer on1/20. It is an occasional demonstration, not reliable personality preservation. Experiment099 tested13 settings on six familiar questions and found no improvement that preserved coherent catgirl counts and avoided increased hard loops. Fresh descriptive validation is recorded separately in100. Original +0.3 graphics remain separate and unchanged.

A second image, `qwen_catgirl_original_meow_values.png`, shows the original +0.3 recipe's actual fresh100 introduction with “*meow!*”. Its separate SVG and `.source.json` record that exact prompt, output, doses, and source hash. This is a selected original-dose example, not evidence that boosting caused the meow. See `../results/100_REPORT.md` for the full paired fresh comparison.
