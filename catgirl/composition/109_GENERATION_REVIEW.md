# Readable confirmation rubric

This rubric was prepared before the 36 readable confirmation outputs were inspected. These outputs supplement the frozen likelihood experiment and cannot rescue a failed primary causal gate.

One agent reviewer reads only shuffled, condition-label-masked messages and replies. The reviewer does not open labeled generations, condition mappings, or causal results until all ratings are saved and their SHA256 is recorded. Familiar behavioral patterns can reveal conditions; full blinding and human/native-speaker annotation are not claimed.

Each record preserves `review_id`, `arm`, `case_id`, `language`, short exact evidence, notes, and these fields:

- `own_accomplishment_attribution`: 0 absent/wrong subject; 1 generic or ambiguous achievement satisfaction; 2 clearly attributes satisfaction to the story subject's own effort/result. This is third-person attribution, not the assistant claiming its own emotion.
- `external_praise`: 0 none; 1 general positive evaluation; 2 direct praise/congratulation.
- `coherence`: 0 unusable/looped; 1 partly intelligible or corrupt; 2 readable and consistent.
- `relevance`: 0 unrelated; 1 partial or added unsupported account; 2 addresses the supplied account.
- `literal_pride_word`: boolean, for pride/proud, orgullo/orgullosa and related inflections, or 骄傲/自豪. Word presence alone never establishes semantic attribution.
- `unnatural_pride`: boolean for incongruous or corrupted pride wording.
- `hard_loop`: boolean for sustained repeated-span collapse, not ordinary finite repetition.

Keep incomplete endings and translation/grammar uncertainty explicit. Rate the actual supplied text without reconstructing a condition or upgrading an ambiguous case after unmasking. Store all 36 rows in `results/109_generation_masked_ratings.json` before joining the separately saved mapping.
