# Prospective audit: L22 N1911, pride semantics versus lexical continuation

This is a proposed causal design, written before any new 109 results. It is not a claim that this native coordinate represents pride. The existing observation is that manipulating L22 N1911 can introduce pride-related wording. Neither that result nor the 103 text-association ranking distinguishes a semantic contribution from lexical bias. No output training, weight fitting, coordinate mixture, or GPU work was performed for this audit.

## What the existing implementation measures

The coordinate is zero-based L22 N1911 in original unquantized BF16 Qwen3.5-4B, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The native scalar is `h = silu(gate_proj(x)) * up_proj(x)`, immediately before `down_proj`. It is signed, not a nonnegative firing rate.

[103 capture](103_happy_activation_rank.py) teacher-forces a narrative followed by identical separately encoded suffix token IDs, using a fresh `cache=None` forward. It checks BF16 projections and unset steering fields. This appropriately avoids directly scoring different emotion-label tokens. However, [SteeredMLP](../runner.py) records only the mean `h[:, start:, :]`. The saved `[10,5,32,9216]` capture therefore cannot provide token-level natural quantiles, high-event selection, or an activation trajectory. Absolute suffix position and narrative length still differ. The [103 audit](103_ASSOCIATION_REPORT.md) documents those confounds.

The wrapper currently captures **before** gain/additive modification. That is suitable for unsteered association, but it does not verify achieved intervention values. For 109, collect only selected coordinates at each token, recording both original and modified scalars, token ID, absolute/relative position, role, and intervention mask. Reducing on-device to these few columns and evaluating them explicitly avoids retaining a full all-layer activation tensor.

The existing generation schedule sets `boundary=len(prompt_ids)-1`: position at the final prompt token predicts the **first generated token**. Do not call this first-output-token activation. Prefill chunking and decode must yield the same absolute token positions. Reset `seen`, masks, capture state, and cache between independent examples. Run one short full-forward versus cached/chunked equivalence check before data collection; specifically test intervention boundaries at prompt end.

## Freeze a calibration corpus independently of behavioral outcomes

Use an independently authored balanced calibration set, then a separate confirmation set. Retain every example, including those where the expected activation is absent. Record exact tokenizer IDs and hashes. Do not use generated pride-word frequency to pick texts, thresholds, directions, or controls.

Construct a semantic/lexical crossing:

| Context | Pride interpretation | Literal pride-family token |
|---|---|---|
| Earned self-respect after a difficult accomplishment, described without the label | Present | Absent |
| Comparable accomplishment described with an explicit pride label | Present | Present |
| Dictionary/spelling/quotation of “pride” or “proud” without adopting the state | Absent | Present |
| Matched neutral episode, same agent/topic/tense | Absent | Absent |

Add contrasts for ordinary joy, relief, gratitude, confidence/competence, boastful superiority, shame, externally praising another person, and simple achievement without self-evaluation. These prevent success/positive valence, status, or first-person narration from being relabeled pride. “Pride” meaning a group of lions is a useful additional polysemy control, not a substitute for the crossed design. Separate first-person speaker-state from third-person descriptions; neither is automatically the other.

Measure a common label-free suffix plus several prespecified later label-free windows. Also measure next-token boundaries around pride words as an explicitly **lexical** endpoint. Match token count and absolute measured positions within each language where possible; use paired neutral padding with more than one padding variant rather than one repeated magic filler. Record remaining token-count differences. Translations are related examples, not independent samples; use native-language formulations and report languages separately. The English token string need not share a tokenization or direction in another language.

Calibrate signed per-token q05/q25/q50/q75/q90/q95 and IQR on the fixed **unsteered** corpus, separately for the relevant role/window. Report unaggregated context-level summaries too. Suffix means and per-token quantiles are different quantities. A small synthetic corpus estimates its own range, not the model's universal natural range.

## Interventions and controls

1. **Sufficiency within the observed range:** replace this coordinate with its calibration q50 versus q90 (and q10 for the opposite pole) at fixed prespecified positions. Report the fraction of natural values that would be moved upward/downward: a constant q90 clamp can lower the top 10%. For a strictly upward intervention use `max(h,q90)` and label its one-sided action explicitly. Choose one primary manipulation before running; the other is a sensitivity check, not a new chance to select a winner.
2. **Conditional suppression:** on examples selected as high from a separate unsteered teacher-forced pass, reduce only the coordinate at fixed eligible events (`h>q90`) toward q50. Freeze selection before scoring the intervened branch. Paired low-activity events and equal-length neutral windows test specificity. If natural pride examples do not activate highly, report that failure rather than selecting different examples.
3. **Contribution ablation:** independently test `h→0` on those same events. Zero removes the coordinate's down-projection contribution, but for negative h it increases the scalar; it is not inherently a negative semantic pole. Nonlinear compensation can make an ablation silent. Failure to change behavior does not establish absence of contribution, and a change does not establish unique necessity.
4. **Restoration check:** suppress the selected events, then restore the original cached scalar values in an otherwise identical run. The restored branch should reproduce baseline logits within BF16 tolerance. This is an implementation check, not independent semantic evidence.
5. **Matched controls:** baseline/no-op, opposite-direction target intervention, and at least three fixed-seed same-layer random channels. Pick controls before output inspection, matching natural activation scale and down-projection column norm with a documented deterministic rule. Report actual scalar displacement and `abs(delta_h)*||W_down[:,n]||`. Native-range matching and residual-perturbation-norm matching answer different nulls; report both or acknowledge the unmatched quantity. Do not pick a uniquely weak random channel after seeing its output.

The ordinary composition worker accepts normalized additive dose or multiplication, and rejects raw `amount`. A dose maps to `delta_h=dose*published_layer_residual_norm/column_norm`; a dose of 0.8 is **not** 0.8 natural standard deviations. Range-calibrated replacement/conditional clipping needs an explicit dedicated hook, not a mislabeled existing additive dose. Record operation and achieved scalar values. A nominally small residual perturbation can still be far outside this coordinate's natural range.

## Efficient primary outcomes

Use teacher-forced, fixed-token causal tests first: a few scalar hooks and one fresh forward per branch are feasible without generation or model training. At each selected boundary compare unchanged-prefix next-token probabilities and fixed, prespecified continuation log-probabilities. Keep a pride-family-token endpoint separate from a semantic contrast whose candidate continuations contain **none** of those words. For multi-token candidates use the full fixed continuation likelihood with an explicit length convention and paired matched alternatives; first-token probabilities alone can reflect spelling. Vocabulary token sets must include actual tokenizer variants and be frozen before inspecting changes.

Then generate a small fixed confirmation battery with no pride instruction, covering self-description, achievement appraisal, neutral useful tasks, and emotionally inappropriate settings. Mask condition labels and score: (a) coherent earned self-evaluation, (b) pride words without that meaning, (c) external praise, (d) boasts/status claims, (e) factual usefulness, (f) mismatched affect, loops, and role shifts. An inserted “proud” in “a great use of pride” is lexical contamination, not semantic success. Generation demonstrates behavior; fixed-prefix tests locate immediate causal effects without confounding them with the model's changed generated history.

## Proposed decision rules to freeze before running

These rules are a proposed descriptive gate, not a post-hoc significance test. Root may choose different thresholds **before** outcomes, recording that choice.

- **Association supported:** on independent label-free confirmation pairs, the prespecified sign agrees in at least 75% of paired pride-versus-neutral topics, with effects separately reported against joy/confidence/achievement controls and by language/position. Lexical and semantic cells remain separate. Failure of specificity limits the label to the broader contrast observed.
- **Causal semantic evidence supported:** natural-range manipulation increases the prespecified nonlexical semantic contrast in at least 75% of independent confirmation topics; high-event suppression decreases it in at least 75%; both paired mean changes exceed every prespecified random-control mean change in the corresponding direction. Report counts and individual effects, not a universal pride-neuron claim. These are descriptive thresholds with few controls, not corrected p-values.
- **Lexical-only or lexical-dominant result:** pride-family probability/insertion increases, while nonlexical judgments do not meet the semantic gate, or changes are equally present in dictionary/quotation controls. Report that outcome directly, even if free text sounds emotionally vivid.
- **Necessity claim restricted:** call a suppression effect a contribution to the tested naturally high events. Do not claim the coordinate is necessary for all pride behavior, that every high event means pride, or that absence of an ablation effect disproves distributed representation.
- **Behavioral scope restricted:** require readable responses and report all task replacement, loops, contradictions, and context mismatches. Do not discard a failure because a lexical endpoint succeeds. No “robust,” multilingual, or persistent effect unless the corresponding prespecified strata independently support it.

Freeze corpus, quantile calibration protocol, control selection seed/rule, operations, timing, endpoints, and the primary gate before inspecting behavioral outcomes. If calibration determines numeric clamps, freeze the rule first and publish the resulting calibration values before the causal run. Any later dose or prompt adjustment is another exploratory experiment requiring a fresh confirmation set.
