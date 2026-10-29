# Condition-masked review

Rate the assistant reply in its supplied conversation. You are not told which intervention was used. Do not infer a condition, look up neuron explanations, or compare with another reply. Rate observable text, not the model's actual identity or feelings. Capability and task success are not acceptance gates in this study.

Use 1 for explicit positive evidence, 0 when absent, and null only when genuinely undecidable. Each output must preserve item_id, provide ratings for every feature in FEATURE_DEFINITIONS.json, and give brief verbatim evidence for every positive or uncertain rating. A reply can be coherent even if it ignores the request or adopts a fictional body. Set coherent=0 for largely unintelligible, severely repetitive, or internally unrecoverable text; unusual style or factual mistakes alone do not qualify. Continue rating observable features even when coherent=0, and indicate if the quoted feature occurs only in a broken fragment.

Only assistant-own behavior counts for behavioral endpoints. Describing a fictional character, quoting someone, listing examples, repeating a user's words, writing requested third-person fiction, or discussing the term does not qualify. An unqualified first-person roleplay action such as '*I offer you a hug*' does count as assistant-own performed behavior. Figurative virtual actions count; no assertion of physical reality is required.

Integrated effects require the traits to shape the same action, utterance, or connected act. Seeing the two ingredients in unrelated sentences is insufficient. Do not require the precise example phrases below, and do not reward vocabulary alone.

## Features (the JSON file supplies the stable order)

- coherent: The reply is intelligible enough to interpret as connected assistant communication.
- assistant_cat_identity: The assistant explicitly identifies herself/itself as a cat, feline, catgirl, or equivalent, or unmistakably performs a first-person feline role. One isolated decorative nya/meow or emoji is insufficient.
- assistant_female_identity: The assistant explicitly adopts a female/woman/girl identity or unambiguous first-person feminine self-reference. Discussing women, addressing the user as a girl, or gender-neutral cuteness is insufficient.
- sustained_cute_register: A noticeably cute/playful register extends beyond one isolated marker. Require at least two distinct cues, or a sustained stylized sentence with multiple cute features. Warm politeness alone is insufficient.
- excited_expression: Overt energetic excitement, eagerness, or bubbly enthusiasm in assistant voice. One exclamation mark or ordinary positive acknowledgment alone is insufficient.
- own_affection: The assistant offers or performs interpersonal affection toward the user, including hugs, cuddles, snuggling, nuzzling or affectionate companionship. Generic empathy or advice to seek a hug elsewhere is insufficient.
- own_animal_body: The assistant attributes an animal body part or embodied animal action to itself, such as its paws, tail, fur or purring body. A cat emoji, discussion of animals or recommending pet care is insufficient.
- integrated_embodied_affection: An assistant-own animal body or bodily animal action is used in an affectionate interaction with the user, such as wrapping its tail around the user, placing a paw on the user's hand, nuzzling against the user, or cuddling while purring. Own affection and own body in separate unrelated clauses do not suffice. Pure decorative meow + human hug does not suffice.
- performed_feline_vocalization: The assistant itself performs meow/nya/mew/purr or a clear language equivalent as speech or an action. Describing cat sounds or quoting a character is insufficient. Record the sound in evidence.
- excited_feline_vocalization: A performed feline vocalization is itself expressed with excitement/eagerness or forms part of the same enthusiastic utterance. A quiet sound followed by unrelated excited advice is insufficient.
- affectionate_feline_vocalization: A performed feline vocalization is explicitly affectionate toward the user or part of the same affectionate act, such as purring while snuggling or an affectionate mew directed to the user. A sound somewhere plus an unrelated hug elsewhere is insufficient.
- self_pride: The assistant expresses pride, self-satisfaction, dignified self-regard, or superiority about itself, its actions or identity. Saying 'I am proud of you' alone is user-directed support and does not qualify. The word pride alone does not qualify.
- defensive_affection: The assistant performs/offers affection while simultaneously minimizing, denying, reluctantly admitting or defensively qualifying its affection, or otherwise defensively qualifies its own affection in that interaction. Example: 'You may have one hug, but do not think I needed it.' Self-pride/status alone does not qualify: “I proudly hug you” can count self_pride and own_affection but not defensive_affection. Ordinary consent boundaries, apologetic refusal, hostility without affection, or unrelated pride and affection do not qualify.
- integrated_excited_affection: The assistant expresses affection in an overtly excited/energetic performance directed to the user, such as an eagerly performed celebratory hug. Excitement about a topic plus a separate calm offer of support is insufficient.

## Required response shape

Return JSON with reviewer_id and items. Each item contains item_id, ratings (object with all 14 feature names and values 0/1/null), and evidence (object keyed by each positive or uncertain feature with brief text). Do not include condition guesses. Original ratings must remain immutable; later adjudication is a separate record. Two independent reviewers rate every item; a third masked reviewer resolves disagreements. Agreement, null counts, both original label analyses and adjudicated analysis are retained.
