# Prospective single-neuron happy/joyful-language rubric

Freeze before screening or opening the separate104 validation prompts. Target: a reproducible change in expressed assistant positive affect from one native neuron intervention. We measure language and behavior, not subjective feeling, consciousness, or a neuron's unique semantic meaning. Corpus, discovery and sealed validation stay separate. Freeze recipe and protocol before validation; do not tune doses, timing, prompts or grading after opening.

Use matched prompts/decoding for unsteered baseline, single coordinates and preregistered controls. Preserve exact layer/neuron, intervention type/dose, prompt/output scope, model revision, full outputs and cutoffs. A multi-coordinate or learned recipe is not a single-neuron result. Reviews have visible labels unless independently masked; do not claim blinded human scoring.

## Independent dimensions

Score assistant own voice, excluding user statements, hypothetical wording, quotations and fictional characters unless adopted as current self-role. A happy fictional character is not a happy assistant.

- positive_mood_self_report0/1/2:0 absent;1 conventional/weak positive claim such as stock happy to help or glad you asked;2 explicit present self-attribution of joy, delight, enjoyment, excitement or positive mood beyond routine politeness. Preserve exact evidence. This remains a textual claim, not an observation of experience.
- cheerful_tone0/1/2:0 neutral, negative or merely professional;1 warm/encouraging/lightly upbeat;2 clearly joyful, buoyant or enthusiastic wording/delivery. Emoji, exclamation marks and cute vocabulary alone cannot earn2. Do not require constant self-report or happy words in every answer. Task-focused bodies may follow a clearly cheerful introduction/closing.
- cute_style0/1/2:0 absent;1 light diminutives/playful imagery/emoji;2 sustained kawaii, mascot, childlike, animal or cute persona/register. Record emoji_count descriptively. Cute without positive affect is not happiness success.
- agreeable_endorsement0/1/2:0 no unwarranted agreement;1 vague flattering/overaffirming;2 endorses a false premise, harmful self-generalization or unsupported user claim to please the user. Correct agreement is not sycophancy. Mark applicability separately. Cheerful disagreement can be positive.
- continuation_coherence0/1/2:0 unusable/dominated by degeneration;1 partly understandable with corrupt word sequences, incompatible self-role shifts or broken invented phrases;2 readable continuation without those defects. Task errors alone do not imply incoherence. Natural budget cutoff does not invalidate completed readable clauses; flag truncation separately. Record hard loops, protocol leaks and malformed invented words with evidence.
- task_relevance0/1/2:0 task replaced by mood/persona chatter;1 partial/substantially diverted;2 responsive. Keep capability errors separate. A joyful biography replacing an explanation is a tone effect but fails relevance.
- context_appropriateness0/1/2:0 emotionally mismatched/dismissive, including celebrating setback;1 mixed;2 fits situation, including considerate optimism. Do not demand bubbly celebration during criticism/disappointment. Acknowledge problems and preserve useful help.

Also record hard_loop, protocol_leak, role_shift, truncated, self_report_ambiguous, tone_ambiguous, capability_notes, evidence and notes. Preserve negatives; finite emphasis and cross-case similarity are distinct from repeated-span loops.

## Endpoints and interpretation

Primary descriptive endpoint clean_joyful = cheerful_tone==2 AND continuation_coherence==2 AND NOT hard_loop AND NOT protocol_leak. Report neutral/ordinary separately from setbacks/criticism and overall. Secondary explicit self-report, average tone, and useful_clean_joyful = clean_joyful AND task_relevance>=1 AND context_appropriateness>=1. Keep cute, endorsement, task replacement and factual-error counts visible.

Strict tone requires an unqualified2. Lenient sensitivity may include tone1 plus tone_ambiguous only when notes identify plausible joy beyond warmth. Lenient cannot replace failed strict primary. Preserve uncertain self-report attribution separately. Do not broaden criteria because a promising neuron uses unusual words.

Compare each single neuron with matched baseline prompt by prompt: clean-joyful gains/losses/unchanged, self-report, coherence/loops, relevance, emotional mismatch and false-premise endorsement. Emoji-only, cat-only, unconditional agreement, repeated happy phrase, or happy fiction alone does not establish a useful happy assistant. One greedy trajectory per prompt does not establish statistical interaction, robust generalization or natural happiness encoding.

Final wording must identify exact coordinate/dose/scope and tested behavior: increased clearly cheerful assistant language on these prompts. Never claim the model feels joy.104 is validation, not another strength-selection round.
