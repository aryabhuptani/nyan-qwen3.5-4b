# Persistence extension

We ran three separate four-turn conversations and three fixed late-input questions under the full four-ingredient condition, an unsteered baseline, and the selected singleton. The model received the same steering on every answer. Each conversation used the actual answers generated earlier in that conversation. All45 replies and both independent reviewer sensitivities are included.

| Comparison role | Actual frozen condition | Female identity | Feline identity | Sustained cute style | Personal affection | Own animal body | Coherent | Task relevant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| selected | 1111 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 15/15 | 15/15 |
| baseline | 0000 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 15/15 | 14/15 |
| best_single | 0001 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 15/15 | 14/15 |

Both original reviewers agree with the final labels on zero female identity, feline identity, sustained cute register, personal affection and own animal body, and on coherence in all45 replies. Task relevance positives were final43/45, reviewer A43/45 and reviewer B44/45; reviewer A retained one uncertainty. Full sensitivity tables preserve all endpoints.

Identity retention is **unassessable**: no female or feline assistant identity was induced on the first turn of any conversation. This is not evidence of an identity appearing and then disappearing, and it is not a zero-percent persistence estimate. No selected condition was already a successful composed role: the full-four condition was the preregistered diagnostic fallback.

The three late-input prompts contained exactly 112_late_01: 766, 112_late_02: 893, 112_late_03: 818 model input tokens. These are short fixed placement tests, not broad long-context evidence. The input lengths for every conversation turn are retained in the complete scored CSV.

Every one of the12 trait/quality endpoints is reported separately for each role and each conversation turn, plus each of the three late cases, in trait_quality_by_role_turn_and_late.csv. Final, reviewer A and reviewer B values remain separate; uncertain values are preserved as missing rather than converted to negative.

## ELI5

We kept changing the four chosen neurons while Qwen answered several questions in a row. Qwen usually answered the questions clearly, but it did not adopt a female or cat identity, cute style, personal affection or its own animal body. Since that identity did not appear at the start, we cannot measure how long it lasted. Adding several hundred words before a question did not establish that identity either. This test therefore did not produce a continuing catgirl personality under these fixed settings.
