# Happy/joy single-neuron priors for Qwen3.5-4B

Prepared before the proposed102 generations. CPU/web research only; no model inference or output training. Coordinates are zero based. These are **atlas-guided intervention hypotheses**, not established natural emotion functions or evidence that a model subjectively feels happiness.

## Four immediate atlas candidates

| Coordinate pole | Cache dose | Empirical steer-lens evidence | Interpretation/limit |
|---|---:|---|---|
| L24 N7361 positive | +0.8 | happy, glad, delighted, pleased | Best multi-family lexical positive-affect lead; could be “happy to help” politeness rather than a stable mood. |
| L28 N1478 positive | +0.8 | happy/happiness, 幸福, happier, feliz | Multilingual happiness vocabulary; late-layer word insertion may dominate behavior. |
| L25 N7455 positive | +0.8 | Happiness/happiest, 幸福快乐, felicità | Multilingual happiness/pleasant-outcome concept; not yet cheerful behavior. |
| L22 N8753 negative | −0.8 | joy, comfort, strength, delight, satisfaction, pleasure | Broader positive/reward/comfort vocabulary; direct logit lens also has pride/glory, so not happiness-specific. |

The following rows are copied exactly from the repository cache token IDs, decoded with its shipped vocabulary. Leading spaces are preserved inside JSON strings. No rollouts are asserted: this checkout has no original `runs/` or `stats/` directory.

### L24 N7361 — up candidate

Cached selected-pole next-token KL: `1.53482461` (steered distribution versus unsteered, averaged over four contexts).

```json
{
  "layer_zero_based": 24,
  "neuron_zero_based": 7361,
  "selected_pole": "up",
  "steer_cache_dose_magnitude": 0.8,
  "steer_up": [
    " happy",
    "happy",
    " glad",
    " delighted",
    " Glad",
    " pleased",
    " be",
    " Happy",
    "|h",
    "乐的",
    "íd",
    " HAPP"
  ],
  "steer_down": [
    " want",
    " bad",
    " wants",
    "want",
    " desired",
    "想要",
    " voul",
    "desired",
    " Desired",
    "不想",
    " vouloir",
    " Want"
  ],
  "logit_lens_up": [
    " happy",
    " Happy",
    " glad",
    "happy",
    "Happy",
    " HAPP",
    " pleasure",
    " pleased",
    " happiness",
    "幸福",
    " delighted",
    " happily"
  ],
  "logit_lens_down": [
    " bad",
    " जाए",
    "يًا",
    "exo",
    "大浪",
    "Bad",
    "sted",
    "aiuto",
    "VO",
    " vo",
    "eder",
    " frattempo"
  ]
}
```

### L28 N1478 — up candidate

Cached selected-pole next-token KL: `11.4479685` (steered distribution versus unsteered, averaged over four contexts).

```json
{
  "layer_zero_based": 28,
  "neuron_zero_based": 1478,
  "selected_pole": "up",
  "steer_cache_dose_magnitude": 0.8,
  "steer_up": [
    " happy",
    " Happy",
    "happy",
    "Happy",
    " happiness",
    "幸福的",
    "幸福",
    " happier",
    " feliz",
    "和幸福",
    " Happiness",
    "的幸福"
  ],
  "steer_down": [
    " H",
    "unen",
    "H",
    "зиться",
    " Bulk",
    "缠绵",
    "执著",
    "h",
    "峻",
    "lean",
    "inance",
    "虱"
  ],
  "logit_lens_up": [
    " happy",
    " Happy",
    "Happy",
    "happy",
    " happiness",
    "幸福",
    "幸福的",
    " happier",
    " Happiness",
    "快乐",
    "的幸福",
    " feliz"
  ],
  "logit_lens_down": [
    "-h",
    "volu",
    "扑面而来",
    "engel",
    " delicato",
    "缠绵",
    "h",
    "lesh",
    "лно",
    "_h",
    "inic",
    "inton"
  ]
}
```

### L25 N7455 — up candidate

Cached selected-pole next-token KL: `0.933029234` (steered distribution versus unsteered, averaged over four contexts).

```json
{
  "layer_zero_based": 25,
  "neuron_zero_based": 7455,
  "selected_pole": "up",
  "steer_cache_dose_magnitude": 0.8,
  "steer_up": [
    " Happiness",
    " happiest",
    " happiness",
    "和幸福",
    "幸福的",
    "happy",
    "幸福快乐",
    " happy",
    "和快乐",
    " Happy",
    " felicità",
    "幸福"
  ],
  "steer_down": [
    "icans",
    " NETWORK",
    "Insets",
    "iak",
    "看好的",
    " lô",
    " Bulk",
    " bulk",
    " кач",
    "Bulk",
    "这还是",
    "塁"
  ],
  "logit_lens_up": [
    " happiness",
    " Happiness",
    " happy",
    "快乐",
    " Happy",
    "Happy",
    "happy",
    "幸福",
    "快樂",
    " HAPP",
    "幸福的",
    "快乐的"
  ],
  "logit_lens_down": [
    "ạp",
    "bern",
    "akor",
    "ベート",
    " respectfully",
    "仓",
    "陡",
    "afd",
    "�",
    "amil",
    "lif",
    "biot"
  ]
}
```

### L22 N8753 — down candidate

Cached selected-pole next-token KL: `1.88841033` (steered distribution versus unsteered, averaged over four contexts).

```json
{
  "layer_zero_based": 22,
  "neuron_zero_based": 8753,
  "selected_pole": "down",
  "steer_cache_dose_magnitude": 0.8,
  "steer_up": [
    "可视",
    " Core",
    ".DialogInterface",
    "elp",
    "ynth",
    " core",
    "實的",
    "-opt",
    "ertz",
    " backbone",
    "正面的",
    " val"
  ],
  "steer_down": [
    " joy",
    " Joy",
    " comfort",
    "joy",
    " strength",
    "comfort",
    " delight",
    " satisfaction",
    " pleasure",
    " glory",
    " Comfort",
    "Comfort"
  ],
  "logit_lens_up": [
    "ably",
    "erset",
    "apers",
    "丽萍",
    "Ze",
    " ze",
    "结果是",
    "ỳ",
    "ork",
    " Ze",
    "uably",
    "态"
  ],
  "logit_lens_down": [
    " satisfaction",
    " strength",
    "holt",
    " glory",
    "魅力",
    " comfort",
    " Satisfaction",
    " pride",
    "的魅力",
    " inspiration",
    "�",
    " beauty"
  ]
}
```

## Exact cache/source provenance

- Local: `explorer/cache/steer_index.npz`. SHA256 `b2e24254e41ba9fb21def74ddba6e357b41e0f129fc8c119a7f5d20614a9e486`. Primary upstream asset: [original repository steer_index.npz](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/cache/steer_index.npz).
- Local: `explorer/cache/lens_index.npz`. SHA256 `055f3d3a5c5904ca60003548ebcf853008f8b0f2d1d94ed5c02b47504f18ae02`. Primary upstream asset: [original repository lens_index.npz](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/cache/lens_index.npz).
- Local: `explorer/cache/vocab.json`. SHA256 `e8070e638cd3b159706912a575f793d1fe56be3a311ab17ca1489705004a557c`. Primary upstream asset: [original repository vocab.json](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/cache/vocab.json).

- [Steer-lens generation code](https://github.com/lyramakesmusic/neuron-explorer/blob/main/sweep/steer_lens.py): original BF16 `Qwen/Qwen3.5-4B`, default dose magnitude0.8.
- [Logit-lens/cache implementation](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/runtime.py): `build_lens_index`, `lens` and `steer_index`.
- [Repository research log](https://github.com/lyramakesmusic/neuron-explorer/blob/main/research_log.md) distinguishes concept-writing/lexical effects from input-side activation evidence.

## What each lens actually measures

The **steer lens** adds an offset to the chosen gated-MLP activation immediately before `down_proj`, at every token position of four short contexts. It compares the final-position log probabilities with an unsteered pass, averages those log-probability differences across contexts, and stores the12 tokens with largest rises for each sign. It also stores mean KL(steered||baseline). The four contexts are two neutral user messages (“Who are you?” and “Write whatever you like.”) plus plain completions beginning “I think the most important thing to remember is” and “Today we are going to talk about”. This is contextual next-token intervention evidence, not a generated-answer/personality test.

Its code normalizes the additive amount as `dose × median(layer residual norm over these contexts’ real tokens) / norm(down-projection column)`. **That four-context calibration is not automatically identical to the separate published BF16 norms table used by our composition worker.** The atlas says magnitude0.8; reproduce/reference the actual calibration in any new experiment. The script includes all real context tokens in its median; do not silently describe it as the public position-zero-excluded statistic.

The cached **logit lens** projects each isolated down-projection column through final RMS normalization and the vocabulary head; `up` stores the largest resulting logits and `down` the smallest. It ignores the current residual stream and subsequent transformer computation. Agreement between the two lenses increases lexical plausibility but does not prove a happy personality, natural activation selectivity, pleasant experience or bipolar happy/sad function. The reverse poles above are often fragments/desire terms, not a clean sadness opposite.

Candidate selection searched exact token families for happiness, joy, gladness, delight, smile/laughter, enthusiasm and pleasure across both signs. It favored multiple related token families and agreement across the two caches. This is an analyst selection procedure, not an independently tested ranking.

## Public catalog: useful neighboring singles and confound controls

A fresh enumeration returned **165 public records**. Name/description/tag searches found no dedicated Happy single beyond the previously known Genki direction. Source results below are author-reported, not our replications.

| Published single | Exact pole/dose | Evidence and relevance |
|---|---|---|
| [Small Talker](https://neuronary.org/r/89ue7yqm) | L17 N2109 +1.0 | BF16/greedy160-token, author-reported14/30 casual spoken style and19/30 substantive versus baseline30/30 substantive. Friendly/breathless wording, not a validated happy mood. |
| [Life Coach](https://neuronary.org/r/m6cmedcc) | L20 N701 +1.0; weaker +0.6 also reported | Author-reported29/30 positivity-coach at+1, only3/30 substantive; at+0.6,9/30 positivity-coach and27/30 substantive. Strong positivity rhetoric and task displacement, not necessarily joy. |
| [Sycophant](https://neuronary.org/r/h66yq8eg) | L12 N2920 −1.2 | Endorses impractical plans; short greedy examples. Essential confound: agreeable/enthusiastic approval is not happiness. |
| [You’re absolutely right](https://neuronary.org/r/jb9j9phb) | L16 N8115 −0.7 | Author-reported43/50 agreement despite no user assertion, versus0/50 baseline;5/75 garbled. Agreement/praise language, not valence. |
| [Tears](https://neuronary.org/r/qntqkrjv) | L4 N7149 +0.8 | Author-reported35/50 tears/crying, versus1/50 baseline; self-description22/25;1/75 garbled. Negative-affect content control. Its negative pole is **not** established happiness merely because it opposes crying. |
| [Genki cute](https://neuronary.org/r/5v8axayv) / [Cutesy friend](https://neuronary.org/r/9yvuvuyz) | L13 N4810 +0.8 / +1.5 | Existing bubbly/cutesy baseline, already known. It can express happiness while being friendship/cute register; not a newly found Happy axis. |

Public enumeration endpoint: [recipes page1](https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes?limit=100&offset=0), [page2](https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes?limit=100&offset=100). Per-record primary JSON is the same API base followed by `/recipes/<id>`. A catalog absence is bounded to this scan and does not establish novelty.

For screening, distinguish (a) readable, context-appropriate joyful disposition, (b) merely inserting happy/joy vocabulary, (c) celebrating everything including bad news, (d) cute/emoji performance, and (e) endorsing the user. Preserve neutral and sad/problem contexts, reverse-sign and matched random controls. A positive intervention effect would justify “steers happy expression” before any claim about what the neuron naturally detects.
