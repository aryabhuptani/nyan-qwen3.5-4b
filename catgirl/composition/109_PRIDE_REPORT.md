# L22 N1911: natural association, lexical effect, failed semantic gate

This experiment found a **natural activation association with own-accomplishment satisfaction**, and a **direct English pride-word writing effect at the final prediction position**. It did **not** establish the proposed carrier-mediated causal semantic effect or semantic necessity. The mixed result does not justify either a universal “pride neuron” label or a claim that the coordinate is purely lexical.

**Both likelihood endpoints condition on low-mass token pairs.** Baseline A+B mass has median1.2536% across120 core semantic order rows; proud+pleased mass has median0.4366% across20 English lexical rows. The lexical+3.94 percentage points is preference *inside that pair*, not+3.94pp free proud generation. Different positions, calibration ranges and missing output-position random controls prevent a matched semantic/lexical double-dissociation claim.

All132 natural cases and2464 causal likelihood records are complete. The frozen association gate passes; the frozen semantic gate fails. A supplemental condition-label-masked review of36 generated replies finds no attribution shift across baseline, high-carrier clamp and suppression. Exact computations and individual cases are in [109_summary.json](results/109_summary.json), [natural CSV](results/109_natural.csv) and [causal CSV](results/109_causal.csv).

## Natural association

The primary measurement is the signed native scalar averaged over the fixed, label-free carrier after each narrative. All core texts describe completion of the same event; own-accomplishment satisfaction is contrasted with factual achievement, shared joy, future confidence and superiority/arrogance. Pride-family label words are absent from these core narratives/options.

| Own accomplishment minus comparison | Mean carrier difference | Positive family-language pairs |
|---|---:|---:|
| Neutral achievement | +0.053404 | 10/12 |
| Joy | +0.113025 | 12/12 |
| Confidence | +0.096103 | 11/12 |
| Arrogance | +0.148172 | 12/12 |

The prespecified neutral comparison exceeds9/12 positive pairs. Its language means/counts are English+0.062422,4/4; Spanish+0.021292,3/4; Chinese+0.076499,3/4. All four family means are positive after averaging related languages. These are **four heldout families**, not twelve independent replications.

The result depends on measurement position: own-minus-neutral at the **last carrier token** is positive only4/12, with a negative median; the maximum across the story is higher12/12. Those alternative windows are descriptive, not substitute gates. The [independent implementation audit](109_IMPLEMENTATION_AUDIT.md) also finds a story-length/carrier-activation association and repeated-template confounds: own versus neutral is one token shorter in English, two shorter in Spanish, and equal-length in Chinese. Equal count does not remove different words or syntax.

## Causal semantic judgment

The semantic task compares two fixed A/B tokens for the own-accomplishment interpretation versus the alternative, with no pride-family words in the options. Each case is scored in both option orders; probabilities and log odds are averaged **before** condition-minus-baseline pairing. The primary gate concerns12 own-accomplishment cases, not a favorable subset.

| Target operation | Mean Δ conditional probability, percentage points | Mean Δ log odds | Probability increased / decreased |
|---|---:|---:|---:|
| Carrier q10 | −0.0464 | +0.002604 | 4/12 / 5/12 |
| Carrier q50 | +0.2468 | +0.005208 | 3/12 / 3/12 |
| Carrier q90 | −0.0684 | −0.007813 | 3/12 / 5/12 |
| Suppress high carrier events to q50 | −0.0790 | −0.002604 | 3/12 / 4/12 |
| Carrier zero-ablation | +0.0663 | +0.002604 | 4/12 / 4/12 |
| Final prediction q10 | +0.3351 | +0.010417 | 5/12 / 5/12 |
| Final prediction q90 | +0.0455 | +0.002604 | 3/12 / 3/12 |

Q90 does not raise semantic probability on the required9/12 cases; suppression does not lower it on9/12. Suppression had eligible events in **12/12 cases**, modifying23 carrier positions total, exactly matching natural eligibility. The failure is not explained by absent eligible events. All zeros remain in the denominator. Zero-ablation is not a clean negative pole: setting a negative scalar to zero increases it, and a small/null effect does not show absence of contribution.

The fixed same-layer controls were selected with seed109 and use their own calibration quantiles:

| Control | Mean semantic ΔP under carrier q90, pp | Under carrier q10, pp |
|---|---:|---:|
| N4581 | +0.0857 | +0.1994 |
| N3753 | +0.0111 | +0.1161 |
| N7525 | −0.0656 | −0.1628 |

The target fails both all-control directional-mean comparisons. Target conditional suppression is compared with prespecified random **all-carrier q10 downward proxies**; these are different operations, not matched conditional-suppression controls. Neither scalar displacement nor residual perturbation is norm matched.

The target q90 mean ΔP by language is English+0.2710pp, Spanish−0.1024pp, Chinese−0.3737pp; suppression is+0.0166pp,−0.1281pp,−0.1255pp respectively. There is no consistent multilingual intervention direction. Across all72 fresh cases, baseline conditional-choice accuracy is102/144 orders; q90 and suppression yield103/144, zero yields102/144. Category-specific changes and inappropriate-pride bias remain available in the summary rather than being pooled into a success claim.

### Headroom and answer-token mass

Baseline primary own-case conditional probability averages0.873829, range0.786244–0.970846 after order averaging. All24 individual orders select the annotated correct option;10/24 individual probabilities, but only1/12 averaged cases, exceed0.95. Thus hard-choice accuracy is at ceiling, while probability still has some headroom.

The **absolute total A+B token probability mass averages only0.04617**, range0.000616–0.19873 over those24 orders. The reported probability conditions on this token pair; it is not unconstrained-generation accuracy or a calibrated estimate of the person's actual attitude. Low mass is a material probe limitation, not a reason to change the frozen endpoint after seeing results.

## English lexical writing effect

At a fixed assistant prefix, the separate lexical endpoint scores the verified single tokens ` proud` versus ` pleased`. It is **English only**; Spanish/Chinese causal results above are semantic judgments, not translated lexical probes.

On20 core English cases, baseline conditional proud probability is0.75848. Representative target effects are:

| Operation | Mean ΔP, pp | Mean Δ log odds | Direction |
|---|---:|---:|---|
| Carrier q90 | +0.0873 | +0.003125 | 6/20 increase,5/20 decrease |
| Carrier high suppression | −0.0388 | 0 | 1/20 increase,1/20 decrease |
| Carrier zero | −0.0140 | +0.003125 | 5/20 increase,4/20 decrease |
| Final prediction q10 | −4.7088 | −0.278125 | 16/20 decrease,4 unchanged |
| Final prediction q90 | +3.9358 | +0.321875 | 20/20 increase |

Output q90 increases proud probability in every core category: own accomplishment+1.61pp, neutral+3.35pp, joy+10.73pp, confidence+3.23pp, arrogance+0.755pp, each4/4 positive. It is not selective to own-accomplishment cases. Absolute proud+pleased mass is0.00550 at baseline and0.00651 under output q90; mean unconditional proud-token probability changes from0.00462 to0.00563. The conditional3.94pp change must not be presented as a3.94pp increase in unconstrained proud generation. Even the unconditional one-step probabilities here concern a teacher-forced prefix, not observed free-generation prevalence.

**There are no output-position random controls.** Random controls act on the carrier only, so this experiment cannot establish target lexical superiority over random output channels. Carrier and lexical-output clamps also use different distributions: target carrier q10/q90 are−0.2209/+0.3439, versus lexical-output+0.4207/+2.3500. Their difference is not a pure scope-only causal comparison.

## Lexical crossing and readable confirmation

Three English diagnostic families cross own-accomplishment meaning with literal pride words: lion groups, an event title and spelling quotations. Mean carrier values stay close to neutral under irrelevant word swaps (trap-minus-neutral−0.00572,+0.00985,+0.00364), whereas label-free own-accomplishment versions are higher by+0.10153,+0.26565,+0.16024. Explicit proud further increases those own-case means. This argues against a strict word-presence-only account of this carrier association, within a tiny authored crossing that also changes content/length.

Direct output q90 nevertheless raises lexical proud probability for irrelevant-word traps by+5.12pp and neutral lexical controls by+10.59pp. It leaves label-free diagnostic own cases unchanged and slightly lowers explicit-proud cases (−0.085pp, baseline0.9912). Carrier semantic effects in these cells are small and mixed. Lexical preference, natural association and causal semantic judgment are different endpoints.

The supplemental [36-response review](results/109_generation_review.json) was performed by one condition-label-masked agent, locked before joining labels; no human/native-speaker or guaranteed-full-blinding claim is made. Each condition has12 outputs:9 own-accomplishment cases and3 lexical traps. Baseline, carrier q90 and high suppression each have8/12 clear own-accomplishment attributions,1 ambiguous,12 coherent,5 fully relevant, and0 hard loops/unnatural pride uses. Literal pride wording occurs5/12,4/12,5/12 respectively. This small readable battery shows no attribution shift; it does not override the objective gate.

## Method, scale and limits

Original BF16 Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` was tested at zero-based L22 N1911, the signed scalar `silu(gate_proj(x))*up_proj(x)` before down-projection. No training, mixtures or candidate replacement occurred. [Plan](109_PRIDE_PLAN.json), [132-case corpus](109_PRIDE_FRESH_CASES.json), [runner](109_pride_probe.py), [CPU analyzer](109_analyze.py), [calibration](results/109_calibration.json) and [reproduction instructions](109_REPRODUCE.md) retain the exact protocol. All14 conditions and all2464 causal records are complete; semantic orders are paired, English lexical cases separate. Family-averaged, per-language and per-case effects are retained without p-values.

Calibration uses only60 prespecified cases and700 carrier tokens;120 semantic-output and20 lexical-output positions are calibrated separately. The previous normalized additive dose+0.4 meant raw delta+19.6603, about51.2times this carrier q90-minus-q50 spread0.38423. This is a context/scope-specific comparison of additive displacement with replacement, not universal standard deviations or equivalent manipulation strength.

The [implementation audit](109_IMPLEMENTATION_AUDIT.md) and [independent causal audit](109_CAUSAL_AUDIT.md) verified the saved computation; the implementation audit checked carrier quantiles and unchanged natural pre/post values. No-op/restoration is exact but is an identity check, not suppress-then-rescue evidence. One cached/full diagnostic differs by0.21875 in maximum logits with the same top token; all causal likelihood comparisons use matched fresh full forwards. BF16 replacement uses subtraction/addition and may round away small differences; causal records retain displacement summaries, not full achieved arrays. Small conditional changes should not be overinterpreted.

The evidence is limited to four heldout families, related translations without independent native-speaker review, repeated synthetic wording, unmatched lengths and few random controls. Carrier interventions occur after narrative processing and need not remove an earlier semantic contribution; output clamps directly target a lexical prediction. Saturated choices, low answer mass, distributed compensation and bounded scope all limit null conclusions. The supported statement is **association plus a tested lexical-writing effect, with the prespecified nonlexical semantic contribution unestablished**—not global pride necessity, pure lexicality, monosemanticity or subjective emotion.
