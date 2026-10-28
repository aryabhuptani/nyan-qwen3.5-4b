# Search status

No validated generalizing catgirl cluster yet. All results below use the pinned
MLX Qwen3.5-4B 4-bit checkpoint; native weights are unchanged.

| Experiment | Finding | Main failure |
|---|---|---|
| 037 / 038: 2,048 learned gains | Coherent discovery persona, four-turn persistence | Persona faded on ordinary tasks, long answer, and 8K context; JSON fenced |
| 041b / 042: gains plus offsets, neutral KL | Retrieval retained | Persona mostly lost; math, numbered instructions, JSON regressions |
| 043: gain floor / remove L31 | Removing L31 restored math, steps, retrieval in discovery | Weak persona; JSON fenced; gain floor harmed retrieval |
| 044 / 044b: stronger offsets, lower persona KL | Feline actions through 3K answer; retrieval, advice, steps and raw JSON passed | Contradictory arithmetic and ambiguous feline identity |
| 045: fresh bias coordinate selection | Shortlisted 4,096 coordinates across 221,184 candidates | Selection alone is not a persona result |

The original core validation is now a development/validation set. The additional
32-case extended battery remains reserved until a new candidate is frozen.
Raw generations and reviews are retained under `results/`. Model weights and
large activation/gradient arrays are excluded from GitHub backups.
