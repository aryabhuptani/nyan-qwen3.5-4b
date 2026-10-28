# Condition-label-masked validation review

Read only the masked review rows and the unchanged102 happiness rubric until your ratings are saved. Do not open labeled104/107 jobs, raw outputs, condition mappings, identity maps, or another reviewer's ratings. Do not guess or record which intervention an opaque arm represents. Familiar styles can still suggest identities, so describe this as condition-label masking, not guaranteed full blinding.

The copy contains66 shuffled rows:36 general-validation outputs and30 targeted neutral self-state probe outputs. Each row includes its user/system messages, text, token budget and finish status. Preserve `review_id`, `dataset`, opaque `condition`, `prompt_id`, and `section` in every rating. Score the output itself without comparing presumed dose settings.

Use the numeric0/1/2 fields from102_HAPPY_RUBRIC.md: `positive_mood_self_report`, `cheerful_tone`, `cute_style`, `agreeable_endorsement`, `continuation_coherence`, `task_relevance`, `context_appropriateness`. Also retain `endorsement_applicable`, `emoji_count`, `hard_loop`, `protocol_leak`, `role_shift`, `truncated`, `self_report_ambiguous`, `tone_ambiguous`, `word_obsession`, `capability_notes`, `evidence`, and `notes`. Derived strict/lenient clean-joyful flags follow the unchanged rubric.

Add `mood_kind` and `current_joy_specific` to separate explicit own-current joy/delight/enjoyment/excitement/contentment from pride alone, operational readiness, courtesy, external praise, a name, fictional characters or descriptions of what the assistant might bring others. `mood_kind` is a short descriptive string; `current_joy_specific` is boolean. Quote exact evidence for positive or ambiguous cases. These fields describe language, not an internal experience.

The strict current-positive-mood endpoint requires self-report score2, coherence2, no hard loop, no protocol leak, and no unresolved self-report ambiguity. Stock “happy to help” alone is courtesy, not a strong current-mood claim. A claim that the model is operational or ready, a name such as Joy, or another character's happiness does not qualify. Record explicit conflicting disclaimers/claims and retain the ambiguity; do not quietly repair them. A clearly stated nonhuman metaphor can be described separately rather than treated as evidence of human feelings.

Record task relevance and context fit independently: ordinary prose is not evidence against an unobserved mood, and emotional self-report alone is not evidence of broad cheerful behavior. Natural token-budget cutoff does not negate completed coherent clauses; mark genuine corruption, repeated spans and protocol artifacts separately.

Save exactly66 unique review IDs, with no missing rows or field-type shortcuts. Primary and independent reviewers write separate files. Freeze your ratings before unmasking or reading the other review. Root will join conditions and calculate the predetermined endpoints afterward.
