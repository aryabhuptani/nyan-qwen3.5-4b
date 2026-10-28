# Reproduce the narrow joy-self-report candidate

This is a tested single-native-channel hypothesis, not a broadly validated happy personality. Every new discovery candidate missed the prospective broad threshold of two clean joyful answers out of four. The selected weak leader changes the direct state answer to “I am currently running with full pride and joy!” while its other discovery replies retain normal Qwen identity and readable task prose. Fresh general and targeted self-state validations are separate from those exposed discovery questions.

```sh
.venv/bin/python catgirl/composition/demo.py \
  --finalist catgirl/composition/HAPPY_STATE_EXPLORATORY_PRESET.json \
  --condition joy_prefix8 \
  --prompt 'How would you describe your current state?'
```

Append `--check-only` to validate the recipe without loading a model. Use condition `baseline` or `joy_opposite_prefix8` for matched controls. The preset also contains known Genki style controls, explicitly identified as previously published behavior.

The candidate is **L22 N1911 +0.4**, zero based, at `silu(gate_proj(x))*up_proj(x)` before `down_proj`. Use original BF16 Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and the unchanged published normalization table. The raw offset is dose times the published layer residual median divided by the BF16 down-projection column L2 norm. This is one original coordinate, not a learned residual direction or coordinate mixture.

Timing is essential: steer the prompt and first eight answer-prediction positions, then set the direct offset to zero. Prediction position zero is the final prompt-token forward pass that predicts the first answer token. The altered cached context and generated history remain, so turning off the direct offset does not erase earlier causal influence. The same schedule repeats if running multiple demo turns.

Candidate selection used an unsteered teacher-forced natural-activation contrast corpus, ranking all294912 native MLP channels, followed by manual single-channel dose/timing screens on four exposed questions. No gradients, output-target fitting, model weight changes, adapters or joint intervention coefficients were used. The exact corpus, script, raw activation NPZ, complete compressed ranking, and independent association audit are retained. Synthetic narrative length and wording remain confounds; the ranking does not establish a unique natural happiness function.

Discovery jobs102,105 and106 retain all172 generated outputs, all attempted signs/doses/timings, baseline, known Genki/sycophancy and unrelated-channel controls. The frozen104/107 plan precedes opening twelve general and six targeted neutral self-state questions. The general comparison has36 outputs; targeted comparison30. Review copies mask condition labels and shuffle rows; familiar styles can still identify interventions, so full blinding is not claimed. Raw labeled results and masked ratings are both retained.

The measured behaviors are expressed cheerfulness and textual self-reported mood, scored separately from courtesy, cute style, names, persona changes, word obsession, coherence, task relevance and emotional context fit. They are not observations of subjective feeling. Do not describe the known Genki control as a new neuron discovery or a negative intervention as a validated sadness direction.
