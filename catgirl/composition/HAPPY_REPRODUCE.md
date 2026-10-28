# Reproduce the happiness study and its negative validation result

The new candidate **failed fresh validation**: both reviewers found1/6 strong positive self-reports, consisting of pride alone, and0/6 joy-specific reports on new self-state probes. No new candidate met the broad discovery bar. Read [HAPPY_REPORT.md](HAPPY_REPORT.md) for exact primary/independent counts and limitations.

The preset below preserves the tested single-native-channel hypothesis for reproduction, not a recommended robust happy personality. Its “I am currently running with full pride and joy!” response is a selected exposed-discovery quote that failed to generalize. A separate CLI smoke reproduces that same question; it is not independent validation.

The known Genki control L13N4810+1.2 produced5–6/6 cheerful and3–4/6 joy-specific reports, with6/6 sustained cute style, on targeted probes. It remains a previously published cute/friendship effect, not a new happiness discovery or a broad-task validation.

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


To reproduce the known positive/style control using the same model and decoding, replace `--condition joy_prefix8` with `--condition genki_positive_control`. The matched negative control is `genki_negative_control`; its surreal responses do not establish sadness. Do not change coefficients or reinterpret the frozen validation to promote a failed candidate.

Final measured evidence: [summary JSON](results/104_107_summary.json), [general raw36](results/104_happy_general_validation.jsonl), [targeted raw30](results/107_happy_mood_validation.jsonl), and [primary](results/104_107_primary_unmasked.csv)/[independent](results/104_107_independent_unmasked.csv) ratings. Recalculate aggregate counts with `.venv/bin/python catgirl/composition/104_107_analyze.py`; this is CPU analysis, not model generation.
