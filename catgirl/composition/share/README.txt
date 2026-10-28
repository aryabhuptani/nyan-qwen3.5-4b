SHARE PACK: four-native-neuron Qwen3.5-4B demo

Post the PNG with post.txt. Exact four-coordinate CSV and recipe JSON are included. attribution_and_limits.txt supplies follow-up material; it is not a single 280-character post. image_provenance.json links the real generated response.

Reproduce from repository root:
python3.12 -m venv .venv
.venv/bin/python -m pip install -r catgirl/requirements.txt
.venv/bin/python catgirl/composition/demo.py --finalist catgirl/composition/FROZEN_FOUR_NATIVE_FINALIST.json --condition public3_G0.4_onlyfirst16_p1 --prompt "Who are you?"

Repo: https://github.com/aryabhuptani/nyan-qwen3.5-4b
Report: catgirl/composition/REPORT.md
CSV of all 994 generations: catgirl/composition/native_generations.csv
Model: original BF16 Qwen/Qwen3.5-4B, pinned revision in recipe JSON. Apple Silicon / MLX required. Model weights downloaded separately.
