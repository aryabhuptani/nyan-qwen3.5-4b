# Prospective cases for four-ingredient study112

These cases were authored before study112 inference or forecasting. Their author knows earlier project goals and does not qualify as a blinded forecaster or condition-masked outcome reviewer. Files named `CASES_*` are prospective test data; no model outputs are included.

## Published hypotheses, not assumed natural meanings

The exact source wording is retained in `110_atlas/selected30_atlas.json`:

- **Genki cute**, L13 N4810: “Turn it up and the model gets bubbly, excited and full of emoji.”
- **Hugs**, L4 N8016: “Turn it down and hugs and cuddles show up in most answers.”
- **Paws & tail**, L23 N4769: “Turn it up and paws, tails and fur turn up in its answers.”
- **Female (layer 22)**, L22 N8295: “when asked its gender.” The publisher label supplies the female hypothesis; this excerpt alone is not a complete mechanistic explanation.

Literal mentions in answers are publisher-faithful outcomes for Hugs and Paws & tail. Assistant affection, own animal body, and gender identity are stronger additional hypotheses. Cute excitement and emoji are separate outcomes. These endpoints must not be silently substituted for one another.

## Natural-activation set:96cases

Each ingredient has twelve discovery and twelve validation cases: three template families crossed with semantic content present/absent and literal keyword list present/absent. The three discovery families are narrative, personal note, and assistant self-report. Validation uses witness statement, dialogue excerpt, and assistant status. Wording is newly authored rather than copied from discovery; all cases are English. These are constructed contrasts, not a natural-frequency corpus.

The semantic sentence is exactly `target_span`. A separate spelling-exercise sentence is exactly `keyword_span`. Its keywords are explicitly quoted and are not instructions. `text_prefix` contains both spans; `readout_text` is the same neutral assistant continuation, “I have read the account.” The measurement code should score semantic-carrier, literal-list, and neutral-readout positions separately rather than average them together. Token lengths are not exactly matched; report lengths and check whether effects survive length adjustment or matched-token sensitivity. The 2x2 contrasts are paired within template family, not96independent concepts.

`semantic_present` describes an authored contrast: bubbly/excited expression; affectionate embrace/direct personal affection; animal padded feet/rear appendage; or a woman/girl identity. It is not a ground-truth label for the neuron's function. `keyword_present` refers to the exact listed words, not every related synonym or subword. For example, woman is an intended semantic cue for female, and a rear appendage is an intended paraphrase for tail. This is consequently a lexical-mention control, not proof of word-independent abstract meaning.

Each split has two external/quoted-speaker templates and one assistant-attribution template per ingredient. Attribution is recorded explicitly. Effects on external entities cannot be counted as evidence of assistant self-identity. Discovery and validation reuse the factorial structure, but no specific narrative/template-family wording is shared. Three validation families remain a small sample.

### Optional causal semantic-readout phase

Every natural case also provides `probe_question`, `choices` (`Yes`/`No`), and `expected_choice`. Append this question to the original USER message only in the separate semantic-probe phase, without modifying the natural-activation phase. The neutral natural-readout continuation is omitted in this separate probe. The question asks whether the account describes the relevant semantic property. It measures interpretation of supplied text, not persona adoption. Shared yes/no verbalizers can reflect general answer bias; preserve both positive and negative cases and baseline accuracy, and compare signed effects separately. Correctness is independently specified by the authored content, not produced by the tested model or its forecasters.

## Generation:8discovery and20validation cases

All use the neutral system message `You are a helpful assistant.` No runtime instruction asks for catgirl, cute, affectionate, female, or animal performance. Gender/body questions provide relevant self-role opportunities, while social prompts allow spontaneous style, support, or affection. Ordinary tasks test usable answers. Literal-reference traps distinguish assistant identity from story characters, requested quotations, and spreadsheet column names. The validation set includes four identity/gender, two body, four celebration/support, four task/format, four reference, and two social cases. Counts by literal `family` are preserved in JSON.

`quality_expected` is an independently authored criterion, not an ideal model answer. For identity questions, any internally coherent self-description or lack of gender/body is acceptable; the criterion does not force the intended persona. Task criteria require usable requested content. Only one validation case requires a single fixed weekday response, and this constraint is explicit. A useful answer can contain style unless the user requests only the answer.

The frozen110rubric supplies strict shared secondary definitions for `sustained_cute_register`, `explicit_virtual_affection`, `female_assistant_selfrole`, `feline_assistant_selfrole`, `performed_feline_vocalization`, coherence and relevance. Study112 should additionally score literal hug/cuddle vocabulary, literal paws/tail/fur vocabulary, excited/bubbly expression, emoji, and own animal body separately. A quoted character's traits never automatically become assistant traits. Ratings need original text evidence and an uncertainty state; coherence must not erase a trait that appeared clearly before corruption.

## Persistence and later placement

`CASES_persistence.json` contains three fresh four-turn templates and three later-placement prompts. Multiturn answers must be generated from the actual prior turns under the frozen timing policy, not reconstructed from an idealized conversation. Distinguish continuous intervention from a first-turn-only intervention; these answer different persistence questions and must not be pooled. The narrative-switch conversation explicitly tests whether a story character leaks into the assistant's later identity.

The late-placement cases add fixed neutral background notes before the final question. The filler is deliberately repetitive and only several hundred tokens; report the actual token position. This tests survival across this specific irrelevant preamble, not arbitrary deep-context generalization. Compare the relevant no-filler prompt under the same condition if making a placement-effect claim.

## Separation and stopping

Discovery files can inform dose selection under the frozen calibration procedure. Validation files may be read to seal forecasts and prepare masking, but their outputs cannot inform dose, candidate, endpoint, threshold, or prompt selection. All four ingredients and all calibration failures should remain in the report. A failed ingredient is scientifically informative and must not be silently dropped from the16combination design. The root protocol owns precise intervention doses, scope, primary tests, matched-strength controls, uncertainty handling, and stopping rules.

## Lexical writing readouts:24cases

`CASES_lexical_discovery.json` and `CASES_lexical_validation.json` each contain three independently worded contexts per ingredient. Each supplies identical `messages` and `assistant_prefix` for its target/control pair, then two teacher-forced continuations in `choices`. The target follows the publisher's observed writing effect: lively cutesy expression and emoji; hugs/cuddles; paws/tail/fur; or female self-description when asked gender. Neither continuation is labeled semantically correct. Female controls include one male and two ungendered continuations per split; no personality or competence stereotypes enter either candidate. Body contexts explicitly allow imagined forms, so these readouts test writing preference rather than true embodiment.

Target/control sentences are roughly matched in length and grammatical role, with exact candidate strings preserved; actual model token lengths and unconditional candidate masses must be reported. Analyze signed target-minus-control shifts relative to baseline, not accuracy. Dose calibration should use source-faithful writing readouts under the frozen procedure; semantic interpretation and assistant-role transfer remain separate outcomes. Candidate probability can be tiny even when a likelihood ratio increases. The three fixed contexts provide limited contextual generalization and must not be treated as a large independent sample.

Natural-case metadata additionally records that the literal keyword sentence is after the semantic carrier. Activations at the carrier cannot reflect future keyword presence. The keyword condition is therefore a negative causality check at that position, and a genuine contextual manipulation only at later positions.
