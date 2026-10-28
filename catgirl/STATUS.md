# Search status

Current finalist: `candidates/catgirl_v1.json` (frozen/pushed at `1f3c2c6`). Native weights are unchanged. See `REPORT.md` for the current evidence and `REPRODUCE.md` for the demo.

- 4,096 learned affine coordinates; first 20 answer-prediction positions, first reply only; all later steering off.
- Core development: explicit catgirl identity on 8/10 ordinary prompts; quality 18/24 versus baseline 19/24.
- Seven-turn stress: correct retrieval after an approximately 8K neutral archive, explicit catgirl identity at 9,520 input tokens, useful arithmetic and planning, no observed loops.
- Cold long first prompt remains unreliable; exact format and factual quirks are recorded. User relaxed capability constraints; these are descriptive, not acceptance gates.
- Fresh 32-case evaluation is complete and awaiting semantic review; final BF16 transfer test is running. Do not interpret pending checks as passes.

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
