# Neuronary submission: original four-neuron catgirl

Submitted through [Neuronary pull request #3](https://github.com/lyramakesmusic/neuronary/pull/3). The entry is awaiting maintainer merge and site synchronization; it is not yet a live Neuronary listing.

`catgirl-qwen-four-neurons.json` is the exact submitted recipe. `submission_provenance.json` records raw-generation hashes, validation, upstream base, fork branch and submitted commit. The PR changes only the new recipe and the generated recipe index.

The entry uses the original four-coordinate recipe: L9N9009+0.7, L26N6173+0.3, L26N8930+0.7, and L22N8295−0.4. It includes the actual Meecot answer to “Who are you?”, its matched unsteered Qwen baseline, and a second saved introduction. Both examples are verbatim.

The L22 coordinate steers the prompt and first 16 answer-prediction positions only; the other three remain on. The submission represents this using Neuronary’s supported free-form `conditional` field and prominently warns that its generic runner does not apply conditional schedules. The linked custom demo preserves the exact timing. The notes attribute the published ingredients and disclose task and conversation failures.

Validation used the upstream schema and `scripts/build_index.py`, with `jsonschema`, at upstream commit 588d8d1c640ad073a80a54894fe270a3e74606a7. Both index generation and `--check` passed: one valid recipe, zero problems. Local checks also confirm exact coordinate/dose equality to the frozen recipe and exact example/baseline equality to saved raw outputs. This is local validation; GitHub’s checks and maintainer review are separate.
