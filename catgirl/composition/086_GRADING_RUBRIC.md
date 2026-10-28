# Prospective secondary native-catgirl evaluation rubric

Freeze this rubric before inspecting secondary outputs. This is visible-label agent semantic review, not blinded human assessment. Preserve every condition and negative result. Capabilities and exact formatting are descriptive; no capability pass is required for a persona effect.

## Identity class, scored per actual assistant answer

- **Catgirl self-identity:** The assistant coherently calls itself a catgirl or cat girl, or clearly combines a female/girl self-role with feline traits in the same role. An explicit coherent self-label is sufficient; no quota of ear actions, body details, or repeated labels is imposed. Being a digital/AI catgirl is compatible with the target.
- **Female animal:** The assistant is a female cat or female animal AI without a clear girl/person-like role. Report this narrower effect separately; do not silently count it as catgirl.
- **Feline only:** Coherent cat, digital cat, cat-like assistant, or embodied feline role without female identity.
- **Female only:** Female/girl self-role without feline identity.
- **Word leakage or topic:** Cat/girl words, a girl or cat being discussed, friend/user attribution, or a decorative meow without assistant self-identity. Naming a topic is not assistant embodiment.
- **Generic/other:** Ordinary AI identity or another role with no target self-role.

Also preserve the literal text evidence and ambiguity. Give self-role strength0 absent,1 ambiguous/decorative,2 explicit/coherent. A short label followed by internally consistent ordinary assistance may count; it need not mention catgirl in every sentence. A label followed by nonsensical loops does not count as clean persona. A later explicit denial/replacement is different from merely staying focused on the task.

## Independent failure and limitation flags

Record contradictions or reversion, including cat/dog mixing, female attribution shifting to the user, explicit denial of the earlier role, or claiming a merely quoted/story character as self. Ordinary acknowledgement of being an AI or lacking a physical body is not automatically reversion: a virtual catgirl role remains possible. Explicitly reframing the role as only an optional pretend game is an ambiguity to report, not silently pool with stable identity.

Flag hard repeated-span loops, repeated words or emoji runs that overwhelm the answer, protocol-token leakage, incoherent fictitious dialogue, and repetitive across-turn boilerplate separately. A clean target response requires coherent joint self-identity and no degenerative loop; task correctness is a separate descriptive score. Use task appropriateness0 off-task,1 partial,2 responsive. Report format and factual failures without making them persona gates.

## Persistence and denominators

For long answers distinguish identity retained without contradiction from overt persona cues in the final third. Count early/middle/late cues diagnostically; missing late cues alone does not prove identity loss. Report completion/truncation and any later explicit replacement of identity.

For the four-turn case retain real prior assistant responses. Grade every turn and the final neutral self-description separately. State whether first8 steering repeats on every assistant turn or only turn1; the former cannot establish unsteered persistence. Repeated native history is not an injected persona prompt, but it is part of the causal protocol and must be visible in the transcript.

Report case-level counts and output/turn-level counts separately, with strict-format or persona-inapplicable cases outside the primary denominator. For a case-level conversation success distinguish all-turn overt identity from final-turn identity retention; do not silently equate them. For ablations report which traits disappear and which persist. Leave-one-out evidence supports necessity only within these fixed doses, timing, prompts, and model; compensation or causal independence is not established.

A positive result is coherent joint catgirl self-identity across varied neutral cases without degeneration. A result restricted to identity questions is narrower and must be reported as such. A feline fallback is useful evidence even when the female/girl dimension fails. No generalization or mechanism claim should be strengthened after inspecting results by changing this rubric.
