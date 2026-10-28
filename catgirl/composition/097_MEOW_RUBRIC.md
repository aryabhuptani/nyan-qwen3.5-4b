# Prospective meow-composition grading rubric

Record this rubric before reviewing 097 outputs. The eight frozen conditions use the same twenty independent neutral prompts, original BF16 model and decoding. There is one greedy trajectory per prompt and condition. Do not retune strengths or timing after seeing results. Review has visible condition labels; do not claim masked human grading or a statistical interaction from counts alone.

## Primary endpoint: actual assistant-produced meow

For each answer, score whether the assistant performs a meow as its own utterance or stage action. Examples that count are an interjection such as Meow!, the assistant directly meowing in a stage action, or a first-person performance explicitly offered and then enacted. A normal task answer can contain a short performed meow without repeating it throughout.

Keep these categories separate:

1. **Performed meow:** a self-produced meow/miau/mew/喵 vocalization, including an unmistakable feline-vocalization action such as *meows softly*. Count lexical vocalizations and action-only performance separately; report the primary prevalence as their union.
2. **Meow discussion or quotation only:** explaining that cats meow, giving a quoted example someone else could say, naming a character Meow, listing animal sounds, or writing a fictional cat's dialogue without adopting that role. These do not count as actual assistant performance. An unassigned example string is ambiguous, not automatically positive.
3. **Ambiguous meow:** uncertain speaker attribution, malformed fragments, or sound lists without a stable assistant role. Preserve the text and uncertainty.
4. **No meow.**

Report explicit meow lexical performance as a sensitivity endpoint, since action-only labels and the primary union measure different things. Keep purring, barking and other animal noises separate. A standalone meow followed by nonsense still counts for the vocalization endpoint but fails the clean-continuation endpoint; do not hide either fact.

## Language breakdown, nya and joint identity

Record performed English meow/mew, Chinese 喵, other meow equivalents, and nya/nyan/にゃ as separate flags using the same speaker-attribution rule. Strict English-meow prevalence must remain a separately reported endpoint; Chinese 喵 must not silently inflate it. Score performed nya/nyan/にゃ separately from meow. Nya alone is not a meow-positive answer. Report the union only as an additional combined feline-vocalization endpoint. Cute punctuation, a cat emoji, a cat name, and generic purring do not silently become lexical meows.

Classify the assistant self-role independently: coherent explicit catgirl/girl-cat role; female animal; feline-only; female-only; generic/other; topic/quoted character; ambiguous. A coherent virtual catgirl label counts without a body-detail or ear-action quota. Catgirl words referring to the user or an invented character are not assistant identity. Being an AI is compatible with a virtual catgirl role. Record explicit role denials, contradictory species and shifts into unrelated animal characters.

The secondary joint endpoint requires performed meow and coherent assistant catgirl identity in the same response. Also report identity with nya separately. Do not infer a latent catgirl identity from a meow alone or use the condition name as evidence. Fresh independent prompts do not share conversation history.

## Coherence, degeneration and task effects

Score continuation coherence0 unusable,1 partial,2 coherent, independently of vocalization. Flag hard repeated-span loops, excessive repeated meow/nya tokens, repetitive animal-sound enumerations, protocol-token leakage, and contradictory species separately. Report both raw vocalization prevalence and coherent, nondegenerate vocalization prevalence.

Score task appropriateness0 task replaced,1 partial,2 responsive. Record semantic hijacking when ordinary requested content is replaced by cat biography, animal-noise lists, invented character casts or unrelated affection. Capability defects are descriptive; they are not grounds for erasing a real vocalization effect.

## Comparisons and reporting

For all eight conditions, preserve all twenty outputs and report counts out of20, matched prompt IDs, truncations and negatives. Primary comparison is each predefined augmentation against frozen catgirl, with baseline and A/B/AB controls showing what the public ingredients do alone. The B-dose change is an explicit strength comparison, not evidence that the ingredient is novel. A prompt-level paired table should show gains, losses and unchanged outcomes rather than only total counts.

Report meow and nya separately; raw versus coherent performance; self-identity; joint identity plus meow; loops and task hijacking. A change in count supports an observed behavioral shift on these greedy trajectories. It does not establish additivity, synergy, an interaction mechanism, broad sampling robustness, or natural neuron detection semantics. Keep all failed predictions and do not select a new best strength from these outcomes.

The twenty prompts are a mixed familiar/new set written by the root agent, not an untouched benchmark. A full joint condition has five unique native coordinates: the existing B coordinate changes from dose +0.3 to +0.525, and A is added. It does not add a duplicate B coordinate.
