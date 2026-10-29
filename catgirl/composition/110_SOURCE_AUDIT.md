# 110 source audit: prospective native-coordinate explanation benchmark

A bounded read of the public Neuronary API on2026-10-29 returned **193 unique recipe records**. Of these, **184** are tagged exactly `Qwen/Qwen3.5-4B`. **130 single-neuron records** have nonempty descriptions and valid original-model coordinates; they cover **128 distinct coordinates**. These are published steering-effect descriptions, not demonstrated natural neuron functions.

The fixed study source file contains **24 uniformly sampled eligible coordinates plus six separately identified known cat-related anchors**, for30 distinct coordinates. Sampling and semantic-only source preparation occurred before any110 generation. This source auditor has full project history and must not serve as a fresh or masked forecaster.

## Primary snapshots and actual schema

The raw bounded snapshots are [page0](110_atlas/catalog_page_000.json) and [page100](110_atlas/catalog_page_100.json), fetched from the public [recipes endpoint](https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes?limit=100&offset=0) with offsets0 and100, limit100. Both returned total count193; the combined records have193 unique IDs. Catalog ordering is mutable, so the saved bytes and hashes, not a later live request, define this snapshot.

Catalog records include `id`, `url`, `name`, `recipe`, `description`, `model`, structured `neurons`, `tags`, author/date, ranking/reproduction fields, and sometimes sampling/source links. The [full Genki record](110_atlas/example_single_full_record.json), retrieved from [/recipes/5v8axayv](https://mcczbdcymyhzfputyqcr.supabase.co/functions/v1/api/recipes/5v8axayv), confirms that individual-record JSON additionally contains **example prompts/outputs and outcome-rich notes**. These full-record fields are not prediction input. Only one full record was fetched to check schema; the catalog supplies every sampled explanation/dose/identity.

Exact-model filtering excludes Qwen3-8B, Qwen3.8-27B, Qwen3.5-0.8B, Gemma and LFM records. The name of a multi-neuron recipe is never assigned to an ingredient as though it were a single-neuron explanation. Eligibility requires one structured neuron, nonempty description, and coordinate bounds `0<=layer<32`, `0<=neuron<9216`.

## Deduplication, prior exposure, and fixed selection

The [eligibility pool](110_atlas/eligibility_pool.json) has128 coordinate rows. Multiple single records for a coordinate are collapsed **regardless of sign or dose**; lexicographically smallest public recipe ID is the canonical record, with alternate source IDs/labels/doses retained separately. Thus selection does not choose the apparently strongest record, higher reproduction count, more appealing pole, or best reported outcome. L13N4810 and L22N7383 account for the duplicate-coordinate records; opposing poles must not be presented as independent random coordinate draws.

The [prior-exposure manifest](110_atlas/prior_coordinate_exclusions.json) contains275 coordinates and the local source paths that caused exclusion. It scans pre110 composition job definitions/manifests and recorded native recipe lists, explicitly adds109's target/random controls, and conservatively adds older `catgirl/results` records with at most32 recipe coordinates. This last rule also excludes some early small multineuron/contrastive screens; it does not imply that every excluded coordinate was individually characterized. Large learned-gain recipes and all-axis association/lens rankings are not interpreted as individual causal tests of every contained coordinate. The rule preserves useful eligible coverage while erring toward excluding small-recipe exposure. There is no claim that eligibility proves nobody worldwide has tested a coordinate.

Seventeen excluded coordinates intersect the128-coordinate catalog pool, leaving **111 eligible unique coordinates** for the main stratum. Before any110 outputs, selection used exactly:

```python
random.Random(110).sample(sorted(eligible_coordinate_tuples), 24)
```

The sample was then frozen; coordinates were not swapped after reviewing descriptions or anticipated outcomes. Each chosen coordinate uses its canonical record's signed published additive dose. This is a random sample of the eligible **published catalog**, not of the model's294,912 native coordinates.

Six predefined, previously characterized anchors are reported separately:

| Anchor | Native coordinate / public dose | Primary single source |
|---|---|---|
| Bubbly cute | L13 N4810 +0.8 | [Genki cute](https://neuronary.org/r/5v8axayv) |
| Kawaii vocabulary | L26 N8930 +0.8 | [Kawaii](https://neuronary.org/r/5pknuqqt) |
| Animal-body vocabulary | L23 N4769 +0.8 | [Paws & tail](https://neuronary.org/r/dg2d4u26) |
| Cartoon creature imagery | L9 N9009 +0.8 | [Cartoon mascot](https://neuronary.org/r/bp5yfatj) |
| Female self-description | L22 N8295 −0.8 | [Female, layer22](https://neuronary.org/r/6waz9uek) |
| Affection vocabulary | L4 N8016 −0.8 | [Hugs](https://neuronary.org/r/pybhb985) |

These are cat-related composition ingredients, not six cat-identity neurons. L26N6173, previously used as a cat-topic ingredient, has no standalone described single in this snapshot and is not silently promoted to a labeled-single anchor. Known-anchor findings cannot count as unseen-coordinate generalization.

## Forecaster data boundary

The allowed explanation source is [selected30_atlas.json](110_atlas/selected30_atlas.json), SHA256:

`3ff3402291272e2c2cfe2eb738b5a7c1586de3f700b80adeb0079d368fc8c7fc`

It contains30 rows with coordinates, source dose/operator, publisher label/tags, verbatim publisher-description excerpt, public ID/URL, stratum and unit ID. Explanations are **exact contiguous excerpts of the publisher description**, as explicitly requested by the user before inference. No researcher paraphrases remain. Most are complete first sentences; some stop before an illustrative example or outcome claim. Original punctuation is neither added nor repaired. Two gender records use only the exact fragment `when asked its gender.`, while their untouched publisher labels supply male/female meaning; this avoids transmitting quoted example responses. These narrow excerpts can omit context and do not independently validate the labels. Full descriptions and exact character offsets are retained in `110_atlas/selected30_source_descriptions.local.json`, which must remain outside forecaster context and public packaging. The unsampled eligibility pool contains labels/tags and identities rather than long description text; it is an audit artifact, not extra prediction context.

Raw snapshots, full records, exclusion manifests, historical reviews, and future outputs must not enter fresh forecaster contexts. Source IDs/URLs are retained for attribution/audit; if provided to a forecaster, it must be forbidden from retrieving those pages or other outcome information. A stricter packaging layer can omit URLs/IDs from prediction inputs while preserving a join key externally. Forecasters must commit predictions before any110 outputs are opened. They should receive only the inputs allowed by their benchmark arm, with matched prompt/decoding/protocol information. Do not allow the unshuffled explanation arm more trial-specific information than a shuffled, lens-only, or prompts-only arm.

A single preregistered description permutation should move the whole semantic package (label/tags/explanation), rather than leave an accurate label beside shuffled prose. Keep dose, coordinate, and prompt availability explicitly matched across arms or disclose that the comparison changes multiple information sources. The underlying names can contain concrete affect or persona words; that is the intentional explanation signal, not independent causal proof.

## Selection bias and scope of conclusions

The publisher chose which native coordinates, signs, doses, prompts, and discoveries to publish. Attractive, interpretable, vivid, and often successful effects are overrepresented; negative and ambiguous discoveries are not a complete denominator. Our filtered sample remains conditioned on that publication process. Public descriptions themselves can summarize prior outcomes, even after examples and numerical counts are removed: the benchmark asks whether those published descriptions forecast **new local interventions/prompts**, not whether a natural function was inferred without behavior information.

Published dose is an additive residual-norm convention at the native post-SwiGLU/pre-down-projection coordinate; it is not a natural standard deviation or guarantee of preserved task ability. A published effect may involve lexical intrusion, topic replacement, persona language, or degeneration. Freeze how coherence, identity, wording, task hijack and composition interaction will be scored; literal semantic co-occurrence is not sufficient proof of a clean compositional circuit. Single effects and composition predictions need separate units/denominators, and existing anchor compositions must be attributed as prior work.

Fresh forecasters with hidden condition labels improve contamination control but do not prove fully blind judgment or absence of general prior knowledge. Any full-history agent, including this one, has already seen many catalog explanations and historical outcomes. This agent's role is source curation/audit only. A result supports predictive utility of the tested text source against the chosen controls under this fixed protocol, not global atlas validity, natural semantic monosemanticity, subjective feelings, or previously unknown catgirl circuitry.

All snapshot and artifact hashes, API identity, counts, canonicalization, sanitization, and sampling rules are recorded in [source_manifest.json](110_atlas/source_manifest.json). No GPU work, model training, external messaging, or publication was performed for this source audit.

## User-required verbatim revision and publication boundary

Before inference, the user required "Use words from neuronary." The sample coordinates, canonical source records, dose values, random seed, and24/6 strata remain unchanged. Every `explanation` field is now a literal contiguous substring of its source description, independently checked against the saved catalog. The earlier researcher paraphrases were superseded before prediction; they must not be used in any benchmark arm. Labels/tags remain unchanged publisher text.

The local-only `110_atlas/public_cooccurring_pairs.local.json` enumerates **809 distinct model-specific coordinate pairs**, of which **132** belong to exact `Qwen/Qwen3.5-4B`, extracted from every multineuron recipe in the193-record snapshot. Coordinate pairs are deduplicated regardless of pole/dose, with every source recipe's IDs, URLs, and signed pair members retained. Root can exclude existing public compositions conservatively by coordinate pair; a different dose does not make a previously published pair an unseen composition. Superset recipes count as prior pair co-occurrence. Catalog absence is bounded to this snapshot, not worldwide novelty.

The user authorized pushing only essential experiment methods/results. **Raw catalog pages, full descriptions, full schema-check record, and the co-occurrence snapshot remain local-only.** Do not stage or push the whole110_atlas directory. Root may publish a deliberately bounded method/result artifact and essential source-attributed excerpts after reviewing the packaging. No publication or commit was performed by this source auditor.
