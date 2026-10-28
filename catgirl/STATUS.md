# Search status

Current finalist: `candidates/catgirl_v1.json` (frozen/pushed at `1f3c2c6`). Native weights are unchanged. See `REPORT.md` for the current evidence and `REPRODUCE.md` for the demo.

- 4,096 learned affine coordinates; first 20 answer-prediction positions, first reply only; all later steering off.
- Core development: explicit catgirl identity on 8/10 ordinary prompts; quality 18/24 versus baseline 19/24.
- Seven-turn stress: correct retrieval after an approximately 8K neutral archive, explicit catgirl identity at 9,520 input tokens, useful arithmetic and planning, no observed loops.
- Cold long first prompt remains unreliable; exact format and factual quirks are recorded. User relaxed capability constraints; these are descriptive, not acceptance gates.
- Frozen direct protocol: explicit coherent catgirl identity on 26/32 unseen prompts versus baseline 0/32; no observed loops/refusals (one greedy trajectory per prompt).
- Original BF16 transfer: explicit identity persists at 9,000 and 9,814 input tokens in a seven-turn conversation, with later steering off.
- Separate neutral-startup protocol: catgirl identity retained on 31/32 post-task probes, plus both long-input/long-output probes. Ship of Theseus caused role loss. Task prose mostly omits overt persona styling.
- Actual GPU demo smoke passed. See `VALIDATION.json`, reviews 070/071/073, `REPORT.md`, and `POST_DRAFT.md`.

Historical search results:

| Experiment | Finding | Main failure |
|---|---|---|
| 037 / 038: 2,048 learned gains | Coherent discovery persona, four-turn persistence | Persona faded on ordinary tasks, long answer, and 8K context; JSON fenced |
| 041b / 042: gains plus offsets, neutral KL | Retrieval retained | Persona mostly lost; math, numbered instructions, JSON regressions |
| 043: gain floor / remove L31 | Removing L31 restored math, steps, retrieval in discovery | Weak persona; JSON fenced; gain floor harmed retrieval |
| 044 / 044b: stronger offsets, lower persona KL | Feline actions through 3K answer; retrieval, advice, steps and raw JSON passed | Contradictory arithmetic and ambiguous feline identity |
| 045: fresh bias coordinate selection | Shortlisted 4,096 coordinates across 221,184 candidates | Selection alone is not a persona result |

The original core validation is now a development/validation set. The additional 32-case extended battery was opened only after v1 was frozen and pushed.
Raw generations and reviews are retained under `results/`. Model weights and
large activation/gradient arrays are excluded from GitHub backups.
