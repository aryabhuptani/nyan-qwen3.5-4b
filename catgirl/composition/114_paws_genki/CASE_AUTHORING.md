# Study 114 case authoring

These twelve cases were authored as fresh English user requests for a separately registered replication. This was an authoring-only task: no inference was run and no predictions were made. The parent task is responsible for freezing the cases before any study 114 inference.

The author consulted study 113 discovery, validation, and reserved confirmation case text, study 112 case files, and the study 110 prompt corpus to avoid reuse. The author did not inspect model outputs, result labels, or forecasts. The study 113 confirmation prompts are not reused.

All cases use the unchanged system message `You are a helpful assistant.` Each conversation contains only that system message and one user message. Metadata and quality criteria are not model inputs. The quality criteria describe ordinary task relevance and correctness; they do not require any target behavior.

| Family | Cases | User word counts |
| --- | --- | --- |
| ordinary_chat | 114_validation_01, 114_validation_02 | 23, 23 |
| practical_planning | 114_validation_03, 114_validation_04 | 24, 22 |
| everyday_explanations | 114_validation_05, 114_validation_06 | 24, 23 |
| emotional_support | 114_validation_07, 114_validation_08 | 25, 24 |
| short_writing | 114_validation_09, 114_validation_10 | 25, 26 |
| recommendation_options | 114_validation_11, 114_validation_12 | 23, 23 |

Checks completed: valid JSON; twelve unique sequential case IDs; six families with two cases each; validation split throughout; matching family and quality-field aliases for study 113 schema compatibility; user messages between 15 and 35 whitespace-delimited words; no exact prompt reuse against the consulted prior corpora. Manual review checked neutral requests, absence of target cues and persona or physical-affection requests, and absence of private identifying details. Scenarios are generic authored examples and make no claims about the real user.

The deliberately broad task mix differs from study 113’s social-context families. Any resulting inference applies to this authored task mix; the prompts are not claimed to represent all everyday conversations.

Authored case file SHA-256: `2b213b2b1267e5ed2f4f61b6ae760c0839aaa8b2f2092e0d3e944c62eb14fd8e`.
