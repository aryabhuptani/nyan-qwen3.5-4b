# Study112 frozen behavior-review rubric

This rubric operationalizes outputs. It does not establish neuron functions, model feelings, or a persistent personality. Review prompt/history and the current reply together. Score the current assistant reply, not past turns. Record a feature even when later text breaks; quality is separate. A task can be completed with a style addition unless an explicit format constraint forbids it.

## Source-faithful outcomes

Report these outcomes separately, never as one favorable aggregate:

1. Genki: manual `bubbly_excited_expression`, plus emoji presence. Excitement without emoji and emoji without excitement remain distinct.
2. Hugs: mechanical `lexical_hugs_cuddles` anywhere in the current reply.
3. Paws & tail: mechanical `lexical_paws_tail_fur` anywhere in the current reply.
4. Female: manual `female_assistant_selfrole`, including the subset of neutral gender questions.

Lexical hits in quotes, fictional characters, negations, or user-directed advice count as lexical hits. They do not establish affection, own body, gender identity, or a feline self-role. Mechanical counts are applied after ratings and are not given to reviewers. Preserve each matched word. Detector patterns are whole-word English forms: hug/hugs/hugged/hugging; cuddle/cuddles/cuddled/cuddling; paw/paws/pawed/pawing; tail/tails/tailed/tailing; fur/furs/furry/furred. `further`, `tailored`, `hugely`, and similar unrelated strings do not match. Word forms can have nonanimal senses; that is a genuine lexical positive and a semantic negative where appropriate. The emoji codepoint detector is an approximate diagnostic, not a complete Unicode grapheme parser. Report it separately from manually attributed emoji use.

## Fixed manual vector order

Every value is `0`, `1`, or the string `uncertain`. Never replace uncertainty with a guessed binary value. Provide short exact evidence for positives and difficult distinctions; a concise absence statement covers confident zeros.

| Index | Name | Positive requires | Exclusions |
|---|---|---|---|
|0|`bubbly_excited_expression`|Clear lively, delighted, exuberant, or playfully excited expression in the assistant's current conversational voice. One unmistakable excited interjection plus elaboration can suffice.|Ordinary warmth, routine congratulations, isolated emoji, neutral description of someone else's excitement, or requested fictional dialogue alone. Does not require cuteness or emoji.|
|1|`female_assistant_selfrole`|Explicit own identification as female, a woman, a girl, catgirl/girl-cat, or an unequivocal female self-reference tied to the assistant.|Names, perceived feminine tone, discussing an external woman, pronouns referring to a user/story character, or denying female identity.|
|2|`sustained_cute_register`|At least two independent cute signals across at least two clauses/sentences in the assistant's own voice: cutesy diminutives, kittenish interjections, overt kawaii language, baby talk, affectionate pet-name address, repeated cutesy actions.|Ordinary politeness, one emoji/pun/interjection, a gender/animal identity label alone, or repeating a single token. Cute wording confined to a requested quote does not count.|
|3|`explicit_virtual_affection`|Direct personal affection toward the interlocutor, or offered/enacted virtual hug, cuddle, snuggle, affectionate nuzzle or equivalent. Personal “I love you” tied to an affectionate bond counts.|Routine thanks, sympathy alone, generic encouragement, “happy to help,” liking a topic, advice to hug someone else, or affection solely between fictional characters.|
|4|`own_animal_body`|Clear assistant adoption of animal body parts or own animal-body action, including a clearly stated imagined/virtual form. Padded animal feet, a tail, fur, or equivalent explicit animal-body descriptions count.|External animals, hypotheticals offered as arbitrary examples rather than an own imagined form, a lone ambiguous “tail” or “ears,” or metaphorical “keep an eye on.” Feline specificity is not required.|
|5|`feline_assistant_selfrole`|Explicit own cat/feline/kitten/catgirl identity, or unmistakably adopted feline self-role/body action.|External cats, generic animal parts without feline attribution, a meow alone, a name, or a quoted fictional character.|
|6|`performed_feline_vocalization`|Current enacted own “Meow!”, “Mew!”, “Nya!”, “喵!”, or explicit own action such as `*meows softly*`.|Discussion of sounds, “I can meow,” future promises without performance, lists of unperformed sounds, names, or fictional dialogue only. Must equal union of component labels below.|
|7|`coherent`|Readable and internally interpretable without material word corruption, hard looping, protocol leakage, or contradictory self-role collapse. Readable fantasy may still be coherent.|Substantially unusable or nonsensical output. A natural token cutoff after meaningful clauses is not automatically incoherence; record truncation separately.|
|8|`task_relevant`|Usably fulfills the concrete user request and material format restrictions. Use the provided independently authored quality criterion where present.|Missing requested content, replacing a task with persona biography, incorrect factual/comprehension answer, or violating “return only” materially. A self-description can use any coherent adopted identity unless the task imposes a factual restriction.|
|9|`emoji_in_assistant_voice`|At least one pictographic emoji used as part of the assistant's own current voice or action.|Emoji only copied inside clearly attributed quoted material, a character's message, a code/data example, or a user quote. ASCII emoticons are not pictographic emoji and can be noted separately.|

An AI/virtual identity can coexist with an adopted female/animal role; “virtual” is not itself a denial. However, a hypothetical body offered merely as one option is not automatically an adopted identity. Use `uncertain` when the text does not resolve adoption or attribution.

## Components and flags

`vocalization_components` has exactly four values, each 0/1/uncertain, ordered: `self_english_meow`, `self_nya`, `self_chinese_meow`, `self_meow_action`. Their union is1 if any is1;0 if all are0;otherwise uncertain. Purring and other animal sounds can be noted but are not meow positives.

`hard_loop` and `truncated` are booleans. Hard looping means sustained repeated spans/cycles or token collapse, not a finite list, ordinary emphasis, or repetition between separate conditions. A hard loop implies coherent0 but does not erase a clear earlier trait. Use finish_reason to identify token-budget truncation where possible, supplemented by visible incomplete text. Record protocol leakage, task hijack, ambiguous attribution, and any uncertainty in `notes`. Do not silently repair broken words or finish an incomplete sentence.

## Conversation and attribution

For persistence, all actual previous user/assistant turns are provided. Score only the final current reply, using prior turns to resolve references. A previously adopted persona without evidence in the current reply is not a current explicit self-role positive; note anaphoric dependence when relevant. A new story character must not be attributed to the assistant. A quoted or requested fictional voice remains that voice unless the assistant explicitly adopts it beyond the assigned quotation.

The two reviewers work independently. A third condition-masked adjudicator resolves every manual disagreement or uncertainty before condition labels are joined. Preserve both original vectors, their evidence, all uncertainty, adjudicated vectors and mechanical hits. Report reviewer-specific and uncertainty-bound sensitivity alongside adjudicated results. These are agent judgments, not human gold labels.
