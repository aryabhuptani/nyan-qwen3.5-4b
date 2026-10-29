# Independent pre-inference design audit

This is a CPU/source review, not evidence that the GPU implementation or experiment has succeeded. Reviewed the initial `measure.py`, `measurement_selftest.py`, and authored discovery schema before validation inference. Files can change afterward; final runner hashes identify the executed version.

## Checks completed

- CPU reference tests passed for one-sided movement toward a percentile target, reverse movement, clipping, restore identity, declared token boundaries, and avoiding model/GPU import during validation.
- The initial hook operates at `silu(gate_proj(x))*up_proj(x)` before the down projection and records achieved BF16 scalar values after intervention arithmetic.
- Candidate branches reset probe counters and operations. Scoring includes absolute per-token, summed and mean log-probabilities and candidate probability mass, not only a renormalized candidate preference.
- Original source text distinguishes vocabulary claims from stronger persona claims. Paws & tail and Hugs must not be scored solely through feline identity and performed affection.

## Issues requiring explicit final choices

1. **Yes/no probe direction.** The authored semantic probes use Yes/No continuations. A uniform increase in Yes probability can raise accuracy on positive cases and reduce it on negative cases. Freeze signed Yes-versus-No log odds as an interpretation-bias measure, stratify by the semantic/literal cells, and report expected-answer accuracy separately. Do not select a dose by whichever coding happens to look favorable. Uniform Yes bias does not establish semantic specificity.
2. **Probe conversational format.** Initial code appended the probe question after the teacher-forced assistant acknowledgment. A new user question and new assistant prefix would be a cleaner interpretation task. If the continuation format is retained, describe it accurately and show baseline performance and candidate mass; do not call it ordinary conversational QA.
3. **Token boundaries.** Independently encoding annotated spans can differ from canonical whole-message tokenization. Fixed encoding supports matched intervention comparisons, but a natural-activity interpretation needs a boundary-difference audit or canonical-position mapping. Store rendered text, token IDs and spans.
4. **Calibration pool.** Quantiles initially pool tokens over all discovery cases, separately by carrier/readout/keyword scope. Freeze which scope sets generation targets. Assistant readout quantiles fit answer-only steering more directly than user-carrier quantiles, but a short repeated acknowledgment may poorly cover normal answer activity. Report support counts and this limitation.
5. **Controls.** The initial fixed four-coordinate job validator cannot run independent same-layer random-coordinate controls. Add explicit generic coordinate support, or run separate same-layer replacement passes with audited coordinate metadata. Do not silently omit controls while describing them as executed.
6. **Actual perturbation strength.** One-sided percentile movement is context-dependent. Sum squared per-layer down-projected norms is a declared nominal budget, not exact functional equivalence. Record achieved changes, unchanged positions and imperfect matching.

The root agent was notified of these issues before results. They are design amendments to settle before freezing, not reasons to reinterpret disappointing validation outcomes afterward.
