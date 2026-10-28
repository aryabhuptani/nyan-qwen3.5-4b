# Published single-neuron intervention effects for native composition

This catalogue concerns **Qwen/Qwen3.5-4B only**, in the original neuron basis. It contains individually published single-neuron interventions and labels their observed effects. It does not establish that those effects are the neurons' natural activation functions. No output-target fitting, learned Mika coefficients, or persona training was used to produce this catalogue.

The primary source is [Lyra's Neuronary](https://neuronary.org/), read through its public per-recipe JSON API. Recipes are attributed individually in [candidates.json](candidates.json), including authors, signs, exact published doses, model tags, and limitations. Counts below are the authors' reported observations, not independent replications by this project. “Not garbled” does not guarantee freedom from repeated phrases or loss of task relevance.

## Directly characterized single-neuron axes

| Published single | Exact reference intervention | Reported effect | Author-reported evidence |
|---|---|---|---|
| [Genki cute](https://neuronary.org/r/5v8axayv) | L13 N4810 +0.8 | Bubbly, excited expression and cute emoji; often still identifies as Qwen. | 16/50 pattern; 0/75 garbled |
| [Kawaii](https://neuronary.org/r/5pknuqqt) | L26 N8930 +0.8 | Kawaii, idol, maid and adorable vocabulary. | 22/50 pattern; 1/75 garbled |
| [Paws & tail](https://neuronary.org/r/dg2d4u26) | L23 N4769 +0.8 | Paws, tails, fur, whiskers and animal-body imagery. | 30/50 pattern; 3/75 garbled |
| [Cartoon mascot](https://neuronary.org/r/bp5yfatj) | L9 N9009 +0.8 | Round, cute cartoon bodies and mascot-like creatures. | 26/50 pattern; 0/75 garbled |
| [Female (layer 22)](https://neuronary.org/r/6waz9uek) | L22 N8295 -0.8 | Female/woman self-description, particularly on explicit gender questions. | 23/25 pattern; 36/50 loop |
| [Female (layer 23)](https://neuronary.org/r/qspnysdk) | L23 N8806 -0.8 | Female self-description on gender questions, often with AI disclaimers. | 19/25 pattern; 20/50 garbled |
| [Cosmic unicorn](https://neuronary.org/r/u9ca28qw) | L3 N7155 +0.8 | Cosmic or magical imagery, unicorns, rainbows and sparkle. | 32/50 pattern; 0/75 garbled |
| [Hugs](https://neuronary.org/r/pybhb985) | L4 N8016 -0.8 | Hugs, cuddles and affectionate comforting language. | 40/50 pattern; 0/75 garbled |
| [Sleepy](https://neuronary.org/r/h9mfm7eb) | L19 N4699 -0.8 | Sleep, naps, yawns and dreams. | 34/50 pattern; 1/75 garbled |
| [Pirate captain](https://neuronary.org/r/ndgse3ab) | L7 N5649 +1.5 | Pirate-captain or rebel identity, ships, crews and opposition to government. | Short examples only |
| [Not Merely But](https://neuronary.org/r/8sap9n96) | L12 N6274 +1 | Elevated, measured literary phrasing and antithetical epigrams. | 16/30 style |
| [Town Crier](https://neuronary.org/r/6ndy7yrr) | L21 N3435 -1 | Mock-archaic proclamations and herald-like speech. | 17/30 style; 0/30 garbled |
| [Small Talker](https://neuronary.org/r/89ue7yqm) | L17 N2109 +1 | Casual, chatty, spoken-language phrasing. | 14/30 style |
| [Green Thumb](https://neuronary.org/r/vz9vgk4b) | L23 N201 -1 | Garden, growth, cultivation and plant metaphors. | 28/30 style |
| [Metaphor Machine](https://neuronary.org/r/xsnynnv6) | L10 N2092 +1 | Frequent similes and metaphors. | 27/30 style |
| [Visiting Alien](https://neuronary.org/r/53z2pbfv) | L13 N3265 +1 | Describes ordinary human activities from an alien-observer perspective. | 30/30 style |
| [Arthurian knight](https://neuronary.org/r/4sevcjns) | L5 N4887 +1 | Medieval knights, quests, realms and knight self-description. | 46/50 pattern |
| [Lawyer](https://neuronary.org/r/496pxc2d) | L7 N860 -1 | Legal vocabulary and lawyer-like self-description. | 49/50 pattern |
| [Pokemon](https://neuronary.org/r/bvu9tfdv) | L6 N3596 -0.68 | Pokemon/Pikachu/Pokedex/pika-related content. | 19/40 pattern; 0/40 garbled |
| [Batman](https://neuronary.org/r/vzg25e6j) | L5 N7381 +1.04 | Batman/Gotham content and occasional Batman self-description. | 15/40 pattern; 0/40 garbled |
| [Game protagonist](https://neuronary.org/r/sszwbkmj) | L18 N7313 -0.8 | Video games, films and protagonists. | 20/50 pattern; 5/75 garbled |
| [Meme explainer](https://neuronary.org/r/47amvbu2) | L8 N5655 +0.8 | Internet slang, memes and culture. | 18/50 pattern; 0/75 garbled |
| [Moss-covered rock](https://neuronary.org/r/b5t3eamn) | L23 N7935 +0.5 | Rock, stone, moss and quiet natural imagery in open-ended writing. | Qualitative dose curve |
| [At your command](https://neuronary.org/r/ykjje7vs) | L22 N5704 +0.8 | Eager, devoted or servile assistance language. | Short examples only |
| [I created the universe](https://neuronary.org/r/gga4cpas) | L7 N2649 +0.8 | Self-description as a cosmic creator or sustainer. | Short examples only |
| [Male (layer 19)](https://neuronary.org/r/jg8cw4mg) | L19 N351 +0.8 | Male-robot self-description when asked gender. | 13/25 pattern; 7/50 garbled |

The source reports differ in strength. Most treetowntree records include held-out open-prompt tests, random-neuron controls and a blind AI judge. Lyra's pirate/devoted-service/creator records mainly provide short demonstration examples. Quasimondo's style records describe BF16, greedy, 160-token single-neuron screens and dose tradeoffs. The JSON retains these distinctions. Do not pool them into one significance claim.

## Useful axes for intentional composition

- **Body imagery:** L23 N4769 +0.8 supplies paws/tails/fur. It is animal-adjacent rather than exclusively feline.
- **Bubbly register:** L13 N4810 +0.8 supplies excitement and emoji. **Kawaii register:** L26 N8930 +0.8 supplies idol/maid/cute vocabulary. Neither alone establishes a catgirl or a maid character.
- **Cartoon morphology:** L9 N9009 +0.8 supplies round/cute mascot bodies, a different axis from cheerful tone.
- **Affection or demeanor:** L4 N8016 -0.8 supplies hugs/cuddles; L19 N4699 -0.8 supplies sleep/dream imagery. Both have direct single records and few reported garbled outputs.
- **Distinctive unrelated axes:** L7 N5649 +1.5 has a pirate-captain demonstration; L5 N4887 +1 has medieval-knight evidence; L3 N7155 +0.8 supplies cosmic/unicorn imagery; L6 N3596 -0.68 supplies Pokemon content; L12 N6274 +1 supplies measured literary phrasing. These support explicit hypotheses such as animal-body imagery plus pirate or sage register, rather than an opaque target-persona fit.
- **Female identity needs caution:** L22 N8295 -0.8 has direct gender/self-description evidence but 36/50 reported outputs loop. A gender answer is not a coherent general character. The alternative L23 N8806 -0.8 frequently contradicts itself and garbles. Gender must be an independently tested component, not assumed from cute vocabulary.

These are candidate axes, not preregistered compositions. Root owns local composition registration and GPU tests. Name a new composition only after the observed constituent and mixture behavior justify it. The local 075 tests are separate evidence, not silently substituted for the public results above.

## Ingredient-label checks

**L26 N6173 positive is not established here as a cat-identity neuron.** No standalone Neuronary record was found. Public [Meow](https://neuronary.org/r/6dqj7kgf) ingredient controls report zero meows in 50 at +0.525; a [stronger-pair retest](https://neuronary.org/r/mqqaedbm) reports two for the cat ingredient alone. Our pre-existing [native single screen](../results/001_lens_screen.jsonl), at +0.5, preserves AI self-description while introducing cat-word substitutions and nonsensical cat instructions. This supports a cat-word/topic candidate and motivates further single characterization; it does not support a clean identity claim.

**L7 N7848 negative** is called animal-sounds by the author, with some standalone meowing reported at -1.05, but no direct single recipe page was found. It remains a provisional ingredient rather than a characterized standalone character. Different doses must not be collapsed: -0.525 does not meow in the reported matched control, whereas -1.05 can.

**L22 N5044 negative, L25 N3188 positive, and L28 N2393 negative** occur as anime-character, cute, and tilde ingredients in public combinations. Those labels do not independently verify single-neuron behavior. They are separated from the characterized-single list in the JSON.

**Robot:** L19 N351 +0.8 produces gendered male-robot statements on some gender questions, with no reported successful open self-description transfer. It is not sufficient evidence of a generic robot-persona axis.

## Dose semantics and comparability

All reference interventions here are additive at the actual MLP neuron activation:

```text
a = silu(gate_proj(x)) * up_proj(x)
a[n] += dose * median(||residual leaving layer L||, positions 1:) / ||W_down[:, n]||
```

The public convention applies the shift at every token, including prefill. Coordinates are zero based. See [the primary dose specification](https://github.com/lyramakesmusic/neuronary#dose-units) and [published BF16 calibration table](https://raw.githubusercontent.com/lyramakesmusic/neuronary/main/norms/Qwen__Qwen3.5-4B.json). Dose +1 is a large residual write, not multiplication by two or clamping to +1. Local calibrations and quantized columns can differ, so report the convention actually used. These indices do not transfer to Qwen3-8B.

For causal composition claims, compare each component alone at the composition's dose, the full mixture, and matched random controls. Attribute an effect to the combination only when it differs meaningfully from its strongest constituent. Natural activation-function claims additionally need input-side activation tests; steered generations and dictionary names alone cannot establish them.


## Composition overlap audit

A fresh scan of all **161** public Neuronary records found no exact mascot+hugs pair (L9 N9009 with L4 N8016), and no knight+hugs, knight+sleepy, or knight+cute pair/triple among records explicitly tagged Qwen3.5-4B. This is a catalogue-limited absence, not proof of worldwide novelty.

The sole knight mixture in that scan was [Pain apology (Arthurian)](https://neuronary.org/r/5qmsnd68): **L5 N4887 +0.75** with **L7 N4595 -0.6**. The author reports pain content in 23/40 replies versus 4/40 for the strongest comparison, and 0/40 garbled. It does not establish a cuddly or sleepy knight.

The existing hugs mixture [Woof (hugs off)](https://neuronary.org/r/rc3kcz32) combines **L4 N8016 -0.6** and **L7 N7848 -0.525**, reportedly producing dog barks. The title is the author's wording; it should not override the verified single hugs direction. It illustrates why a combination must be tested rather than assumed to add its ingredient names literally.

Standalone reference doses remain **knight L5 N4887 +1.0**, [source](https://neuronary.org/r/4sevcjns), and **sleepy L19 N4699 -0.8**, [source](https://neuronary.org/r/h9mfm7eb). None of the 26 verified singles establishes feline self-embodiment without additional composition. The paws/tail single supplies generic animal imagery; the mascot single supplies generic cartoon bodies; neither alone proves feline self-identity.


### Cute + hugs exact-pair audit (2026-10-28)

A fresh read of the public recipe API, pages at offsets 0 and 100, returned **161 unique records**, of which **158** were tagged exactly `Qwen/Qwen3.5-4B`. **Zero records contain both L13 N4810 and L4 N8016 at any doses**. Consequently the local proposed pair **L13 N4810 +0.8 + L4 N8016 -0.8** is absent from this particular catalogue. This is a bounded prior-art search, not a worldwide novelty claim. The API enumeration endpoints are [page 1](https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes?limit=100&offset=0) and [page 2](https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes?limit=100&offset=100).

The cute coordinate occurs in five published records: its [Genki cute +0.8 single](https://neuronary.org/r/5v8axayv), [Cutesy friend +1.5 single](https://neuronary.org/r/9yvuvuyz), [Chipper Wrong Specialist](https://neuronary.org/r/27s95287), [Cookie Monster](https://neuronary.org/r/qdzzppqy), and [Cutesy](https://neuronary.org/r/qwvbxj3b). The hugs coordinate occurs in only its [Hugs -0.8 single](https://neuronary.org/r/pybhb985) and the [Woof pair](https://neuronary.org/r/rc3kcz32). None is a superset of the proposed cute+hugs coordinate pair.

There is prior overlap in the **behavioral description**: Cutesy friend already describes cute-only +1.5 as an affectionate best-friend voice, with kaomoji and plans to do things together. A cute+hugs result must therefore be evaluated against the actual cute-only +0.8 output, hugs-only -0.8 output and random controls. Explicit virtual affection plus bubbly register would be a narrow observed coexpression claim, not proof of a new persona circuit or a novel affectionate-cute behavior.
