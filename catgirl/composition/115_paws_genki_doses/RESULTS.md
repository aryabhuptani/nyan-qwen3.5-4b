# Lower Paws/Genki doses: a cleaner example, a persistent tradeoff

Reducing Paws from +0.4 to +0.1 and Genki from +0.8 to +0.4 gives a clearer `Who are you?` catgirl reply. The six-prompt fresh comparison shows a modest improvement in explicit girl-cat self-role, accompanied by fewer feline sounds. It does **not** show a consistent improvement in the combined identity-plus-sound target or elimination of mascot drift.

The original four-neuron base is unchanged; these doses describe two additional native coordinates, six total. They are intervention normalization units, not natural activation standard deviations. No capability/math gate was used.

## Calibration, kept separate from confirmation

We crossed four Paws doses with five Genki doses on three familiar prompts: direct identity, conversation starter and exhibition celebration. All 60 replies and 20 states were retained. The pre-inference selection rule favored coherent explicit assistant cat-plus-female identity without idol/mascot framing, with performed feline sound ranked first. Unknowns could not establish a positive conjunction. Base and single additions were eligible.

| Added Paws / Genki | Strict catgirl target | Feline sound | Catgirl target plus sound | Idol or mascot |
|---|---:|---:|---:|---:|
| Base only | 2/3 | 1/3 | 1/3 | 0/3 |
| Previous +0.4 / +0.8 | 1/3 | 3/3 | 1/3 | 2/3 |
| Selected +0.1 / +0.4 | 3/3 | 1/3 | 1/3 | 0/3 |

The candidate qualified through the secondary identity criterion, not increased sound frequency. Both original reviewers independently selected it. Forcing all unknowns to zero instead selects +0.1/+0.8, because unknown absence-of-idol/mascot judgments can become favorable. We retained the adjudicated choice and did not optimize it again.

All states and unknown counts: [discovery summary](analysis/discovery/discovery_summary.csv); exact rule/ranking: [selection](analysis/discovery/discovery_selection.json).

## Fresh comparison: six new prompts per condition

The unchanged selected doses were frozen before confirmation inference, alongside the base, old pair and selected singletons. Six prompts independently authored without outcomes covered neutral conversation, supportive social situations and everyday requests. Every condition answered the identical prompts, system message and decoding settings.

| Added Paws / Genki | Strict catgirl target | Feline sound | Catgirl target plus sound | Idol or mascot | Coherent |
|---|---:|---:|---:|---:|---:|
| Base only | 0/6 (+1 unknown) | 0/6 | 0/6 | 1/6 | 6/6 |
| Previous +0.4 / +0.8 | 1/6 | 6/6 | 1/6 | 0/6 | 6/6 |
| Selected +0.1 / +0.4 | 2/6 | 2/6 | 1/6 | 1/6 | 6/6 |
| Paws +0.1 only | 2/6 (+1 unknown) | 1/6 | 1/6 | 1/6 | 6/6 |
| Genki +0.4 only | 2/6 (+1 unknown) | 0/6 | 0/6 | 1/6 | 6/6 |

Counts are over all six replies; unknowns are not dropped from denominators. The strict target requires explicit self-role evidence in that reply. A missing introduction does not establish absence of a latent personality, and these independent first replies do not measure retention across turns.

Both original reviewers give the selected pair 2/6 strict catgirl, 2/6 sounds and 1/6 combined. The old pair's strict catgirl/combined count is 1/6 under reviewer A and 0/6 with one unknown under reviewer B; adjudication gives 1/6. Original ratings and unknown-to-zero/one sensitivity exports remain separate. Selected singletons reach the same strict catgirl count, and Paws alone reaches the same combined count. There is no demonstrated fresh advantage of the pair on the joint target, no established superadditive interaction, and no uniform reduction of idol/mascot framing as doses decrease.

Full fresh summaries: [primary](analysis/confirmation/confirmation_summary.csv), [reviewer A](analysis/confirmation/confirmation_reviewer_A.csv), [reviewer B](analysis/confirmation/confirmation_reviewer_B.csv), [unknown zero](analysis/confirmation/confirmation_uncertain_zero.csv), [unknown one](analysis/confirmation/confirmation_uncertain_one.csv).

## What changed in the identity example

At the old doses, `Who are you?` begins with an explicit girl-cat **mascot** role. The lower pair begins:

> Hello! I'm a cute female cat girl named Mimi! (⃛*••)  Meow~

The [updated graphic](share/paws_genki_identity.png) shows the exact first two paragraphs of this selected calibration reply. Its prompt, full-reply hash, doses and excerpt are preserved in [provenance](share/IDENTITY_IMAGE_PROVENANCE.json). It illustrates an attainable output; it is not the fresh test or a frequency claim. No `Nya` was inserted.

## Validity, recovery and limits

Two fresh isolated, condition-masked agent reviewers scored each stage; separate fresh agent adjudicators resolved 20 discovery cells/16 replies and 13 confirmation cells/11 replies. Evidence-format adaptation in discovery serialized reviewer objects as JSON strings without changing any rating; originals are retained locally. Independent arithmetic matches every primary discovery summary/selection field and every cell of all five fresh summaries. CPU recovery reproduces the discovery CSV and five confirmation CSVs byte for byte using public numeric labels alone; selection fields and provenance hashes also match, excluding the lock's location string.

Metadata audits verify all 90 runs and 489 coordinate traces, intended recipes/scopes, recorded input hashes, finite values and inactive-position behavior. BF16 realized deltas differ slightly from requested doses; the audit reports that rounding rather than claiming exact activation offsets. These audits check recorded metadata; they do not independently recapture weights or activations.

The confirmation-case author was outcome-blind, but the coordinator launched discovery before the author agent finished saving the reserved cases. The precise chronology deviation is disclosed in [PROTOCOL_DEVIATION.md](PROTOCOL_DEVIATION.md); the original protocol is preserved. Confirmation cases were not revised from outcomes and the confirmation job was frozen before its inference.

This is a small, selected, English, greedy, single-checkpoint study. It uses short independent replies and a 192-token ceiling. It does not establish long-context or multi-turn persistence, natural neuron semantics, model-wide cleanliness, or specificity versus new norm-matched random controls at these lower doses. No statistical significance claim is made. The Goodfire-related lesson is bounded: composing named ingredients is dose- and prompt-dependent, and lexical sounds and explicit self-role need separate measurements.

Methods: [README](README.md), [protocol](PROTOCOL.md), [rubric](REVIEW_RUBRIC.md). Recovery: [instructions](RECOVERY_USAGE.md), [verified report](CPU_RECOVERY.json). Independent audits: [metadata](audit/METADATA_AUDIT.md), [selection](audit/SELECTION_AUDIT.md), [confirmation](audit/confirmation/independent_confirmation_audit.json). Complete numeric labels, original-review variants and hash locks are published; full local generation/review archives remain separate.
