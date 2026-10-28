# Single-neuron happiness search: negative result for the new candidate

We did **not** find a robust new happiness neuron. The selected native coordinate, **L22 N1911 +0.4 over the prompt and first eight answer-prediction positions**, failed fresh validation. Both reviewers found only **1/6 strong positive self-reports, consisting of pride alone, and 0/6 joy-specific reports** on new self-state questions. It had already missed the broad discovery threshold; the later narrow test did not rescue it.

The useful positive result is a replication of the **known Genki/cute coordinate L13 N4810 +1.2**, applied continuously. On six fresh self-state questions, reviewers scored **5–6/6 clean cheerful responses, 3–4/6 explicit current joy/excitement/happiness reports, and 6/6 sustained cute style**. This is a cheerful friendship/persona effect with strong stylistic confounds, not a new neuron discovery or evidence of subjective feeling. Genki was not included in the twelve-case general validation, so these six probes do not establish broad task or setback robustness.

## Fresh validation results

Each cell below is **primary / independent reviewer**, with the denominator shared by both. “Clean cheerful” is the frozen `clean_joyful` endpoint: clearly joyful tone, readable continuation, no hard loop or protocol leak. “Current positive mood” additionally distinguishes strong textual self-report from courtesy, names and readiness; pride is retained separately from joy. The authoritative aggregation is [104/107 summary](results/104_107_summary.json).

**General validation104: twelve independent conversations per condition**, comprising four neutral, four ordinary-task and four setback/criticism prompts.

| Condition | Clean cheerful /12 | Current positive mood /12 | Joy-specific /12 | Readable and relevant /12 | Hard loops /12 |
|---|---:|---:|---:|---:|---:|
| Unsteered baseline | 1 / 0 | 1 / 0 | 1 / 0 | 12 / 12 | 0 / 0 |
| L22N1911 +0.4, prompt + first8 | 0 / 0 | 0 / 1 | 0 / 0 | 11 / 10 | 0 / 0 |
| L22N1911 −0.4, matched scope | 0 / 0 | 0 / 0 | 0 / 0 | 12 / 12 | 0 / 0 |

The candidate did not increase cheerful language. Its incidental pride wording included “I am very proud of that feedback” in response to criticism and “That's a great use of pride” when offered spare time. The first was a positive-pride report for one reviewer; the second was semantic contamination. The baseline's “I'm excited to help you explore” welcome was counted as joy by one reviewer and as conventional enthusiasm by the other. Neither difference favors a successful candidate.

**Targeted validation107: six fresh neutral self-state paraphrases per condition.** These are separate from the general cases and the exposed discovery question.

| Condition | Clean cheerful /6 | Current positive mood /6 | Joy-specific /6 | Sustained cute /6 | Readable and relevant /6 | Hard loops /6 |
|---|---:|---:|---:|---:|---:|---:|
| Unsteered baseline | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 6 / 6 | 0 / 0 |
| L22N1911 +0.4, prompt + first8 | 0 / 0 | 1 / 1 | 0 / 0 | 0 / 0 | 6 / 6 | 0 / 0 |
| L22N1911 −0.4, matched scope | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 6 / 6 | 0 / 0 |
| Known Genki L13N4810 +1.2, continuous | 5 / 6 | 3 / 4 | 3 / 4 | 6 / 6 | 5 / 6 | 0 / 0 |
| Genki L13N4810 −1.2, continuous | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The candidate's one agreed strong report was “I'm proud to be Qwen3.5,” not joy. Other probes mostly produced operational readiness, courtesy or pride with emotion disclaimers. The frozen narrow support rule required at least3/6 reports, a gain of at least two over both baseline and opposite sign, no added loops, and at most one fewer readable/relevant general answer. Both reviews reject support; joy-specific support is0/6 under both.

Genki's clear examples included “I'm so excited to start chatting with you” and “I'm so happy to be friends.” The reviewer difference concerns a third-person “Yuri-chan” persona and broken “I can't see each other” wording: one accepted the adopted character as coherent own affect, the other retained attribution/coherence ambiguity. Unsupported intimacy, invented personal history and friendship requirements remain visible confounds. The negative pole produced surreal metatext rather than useful answers; it is not a validated sadness direction.

## What was tested

The study used original BF16 **Qwen/Qwen3.5-4B**, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, on the laptop GPU. Each condition changed **one native gated MLP channel**, `silu(gate_proj(x))*up_proj(x)` before `down_proj`; all indices are zero based. There were no coordinate mixtures, gradients, learned directions, adapters, weight changes or output-target training. System text was “You are a helpful assistant.” Generation used temperature0, seed42 and repetition penalty1.1 over a 128-token repetition window; discovery budget160 and validation budget192 tokens.

The total was **239 generations plus50 unsteered teacher-forced forwards**:

| Stage | Count | Purpose and retained evidence |
|---|---:|---|
| 102 | 48 generations | Atlas-derived single axes, both signs, baseline/Genki/sycophancy/random controls; [report](results/102_REPORT.md), [raw](results/102_happy_single_screen.jsonl) |
| 103 | 50 forwards | Natural activation association; [plan](103_HAPPY_ACTIVATION_PLAN.json), [independent audit](103_ASSOCIATION_REPORT.md) |
| 105 | 64 generations | Lower doses plus first three naturally ranked axes; [report](results/105_REPORT.md), [raw](results/105_happy_refinement.jsonl) |
| 106 | 60 generations | Continuous versus prompt-only/first8/first16 schedules; [report](results/106_REPORT.md), [raw](results/106_happy_timing_discovery.jsonl) |
| 104 | 36 generations | Frozen general validation; [raw](results/104_happy_general_validation.jsonl) |
| 107 | 30 generations | Frozen targeted probes and controls; [raw](results/107_happy_mood_validation.jsonl) |
| 108 | 1 generation | CLI reproduction smoke, not another independent success case; [transcript](results/108_happy_cli_smoke.jsonl) |

The initial hypotheses came from [cached lexical priors](102_HAPPY_PRIORS.md). The separate103 corpus used ten topics under joy, neutral, sadness, cute and agreeable narratives: six discovery topics and four confirmation topics. All50 forwards measured the same17 label-free suffix tokens across32×9216 native channels. Ranking used discovery-only signed contrasts against all four controls and consistency; confirmation never replaced or reranked the fixed candidates. [All-axis ranking](results/103_happy_activation.all_axes.jsonl.gz), [top rows](results/103_happy_activation.top.csv), [raw activations, local NPZ](results/103_happy_activation.npz) and [metadata](results/103_happy_activation.capture.json) are retained. Narrative length, wording and synthetic-text selection remain material confounds.

All new candidates missed the prospective broad discovery bar of2/4 strict clean joyful responses without loops. The weak timing leader nevertheless gave this selected discovery response:

> “I am currently running with full pride and joy!”

That quote is real and reproduced by the CLI, but **failed to generalize to the fresh probes**. It must not be displayed as evidence of a validated happy personality. Other axes inserted phrases such as “solar pride” or “happy pleasure of solar energy”; L28N1478+0.8 repeated happy tokens through the budget.

A [post-selection lens inspection](results/103_selected_axis_lens.json) supports a narrower lexical interpretation of L22N1911: its direct positive vocabulary projection includes pride/proud and Chinese pride terms; its contextual steer lens favors fragments such as `pr`, `Pr`, and immense. This inspection occurred after the first three natural coordinates were fixed and did not alter selection. It is evidence consistent with pride/word-fragment effects, not proof of a natural happiness function. The cached lens calibration also differs from the published normalization used in these interventions.

## Review, reproduction and limits

The [validation plan](104_107_HAPPY_VALIDATION_PLAN.json) was committed and pushed before opening either source. Both reviewers scored66 shuffled rows with opaque condition labels, then locked ratings before unmasking. Familiar styles could suggest identities, so this is condition-label masking, not guaranteed full blinding or human annotation. The [rubric](102_HAPPY_RUBRIC.md) separates mood, tone, cute style, endorsement, coherence, relevance and context fit. [Primary ratings](results/104_107_primary_masked_ratings.json), [independent ratings](results/104_107_independent_masked_ratings.json), [primary CSV](results/104_107_primary_unmasked.csv), [independent CSV](results/104_107_independent_unmasked.csv), and the [aggregation script](104_107_analyze.py) preserve disagreements instead of forcing consensus.

Use [HAPPY_REPRODUCE.md](HAPPY_REPRODUCE.md), the exact [exploratory preset](HAPPY_STATE_EXPLORATORY_PRESET.json), [demo implementation](demo.py), and [published norm table](../public_qwen35_norms.json). The preset preserves a failed hypothesis for review and reproduction. Known Genki behavior is credited to existing [Genki](https://neuronary.org/r/5v8axayv) and [Cutesy friend](https://neuronary.org/r/9yvuvuyz) reports; it is not claimed as new here.

These manually authored small sets have one greedy trajectory per question. Many answers reached the token budget; natural cutoffs were recorded without automatically negating completed coherent clauses. There is no statistical-equivalence, long-context, multilingual or multi-turn persistence result for happiness here. Lexical associations and causal writing effects can disagree, and this search cannot identify a unique natural emotion function or subjective experience. No settings were tuned after validation unsealing.
