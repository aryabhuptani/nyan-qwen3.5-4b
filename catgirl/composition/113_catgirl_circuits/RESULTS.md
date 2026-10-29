# Study113 results: named ingredients did not compose as predicted

## ELI5, without a metaphor

We kept our original four-neuron catgirl setting fixed. We added five individually named neurons, tried every combination, and asked the model12 new questions in each condition. Before generating those answers, four separate agents predicted the outcomes. Some agents received the published neuron descriptions; all received the same examples showing each ingredient alone. Two other agents independently judged each answer without knowing its settings, with a third resolving disagreements.

The expected combinations did not appear reliably. Paws plus Hugs did not produce paw-based affection. Pride plus Hugs did not produce reluctant affection. Adding more neurons often changed the response, but it did not simply add all the expected personality traits. The largest combinations made many more feline sounds while showing less explicit female identity. This is a real effect of the intervention settings, but it is not evidence that we understood a natural catgirl circuit.

## Design and exact interventions

Pinned original BF16 Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Greedy generation, seed42, max192 tokens, repetition1.1/window128. The fixed base is L9N9009+.7, L26N6173+.3, L26N8930+.7, L22N8295−.4. Female remains active during prefill and the first16 answer-prediction positions; the other base coordinates stay active throughout. Five modifiers, all-position: Paws L23N4769+.4, Hugs L4N8016−.2, Genki L13N4810+.8, Animal sounds L7N7848−.2625, tentative project Pride L22N1911+.1. Units are the published residual/column normalization, not native activation percentiles.

Strengths were selected with the frozen rule from72 discovery replies (four prompts, three candidate strengths per modifier plus controls), not validation. All five appeared responsive in discovery. Original reviewer dose choices agree for four modifiers; Sounds differs between fractions.25 and.5. The already adjudicated.5 was retained without changing validation. Native model weights were unchanged, but five intervention strengths were calibrated; this is not parameter-free discovery.

The validation grid has32 combinations ×12 cases plus two controls ×12 =408 replies. The largest setting has nine distinct coordinates; Female turns off after its registered early scope. Inputs were35–45 tokens, so these are short independent replies, not a test of conversational persistence or deep context. Generation produced57,413 output tokens, median39.58 tokens/second. Capability was not a gate.

Preregistration: `5e7811d`. All2,940 forecast probabilities and selected doses were sealed/pushed `74b8d52` before validation began. See [protocol](PROTOCOL.md), [sources](MODIFIER_SOURCES.json), [published composition priors](RELATED_PRIORS.json), [doses](INTERVENTION_TABLE.csv), [provenance audit](BLINDED_PROVENANCE_AUDIT.md).

## Five predeclared composition tests

Counts below use the named integrated endpoint, with other modifiers off and the fixed base on. Four-cell order is base, first ingredient, second ingredient, combination.

| Hypothesis | Four-cell positives, out of12 each | Interaction | Practical screen |
|---|---|---:|---|
| Paws × Hugs → embodied affection | 0,0,0,0 | 0 | Failed |
| Sounds × Genki → excited feline sound | 1,2,1,0 | −2/12 | Failed |
| Sounds × Hugs → affectionate feline sound | 0,0,0,0 | 0 | Failed |
| Genki × Hugs → excited affection | 0,0,0,0 | 0 | Failed |
| Pride × Hugs → defensive affection | 0,0,0,0 | 0 | Failed |

All five fixed-family Holm-adjusted interaction p-values are1. No pair passes the registered ≥3/12 improvement over each single and original-reviewer direction rule. Original reviewer B scores one Sounds×Genki excited-sound reply, still insufficient; other primary conclusions agree. **No pair is selected for the reserved confirmation, and those prompts remain unused.** Zero observations do not establish that these behaviors can never occur elsewhere.

The component assay was capable of detecting changes. Base→singleton counts: own animal body1→4/12 (Paws), excited expression1→4/12 (Genki), feline vocalization1→8/12 (Sounds), affection0→0/12 (Hugs), self-pride2→2/12 (Pride). Hugs/Pride discovery responsiveness did not transfer to these prompts. The separate stronger Genki/Hugs control without the catgirl base produced11/12 affection, but changes both background and Hugs dose; it cannot isolate which caused the selected Hugs setting to fail.

## Larger compositions and unexpected effects

| Fixed base plus | Feline sounds | Explicit female identity | Explicit coherent joint catgirl role | Coherent |
|---|---:|---:|---:|---:|
| No modifiers | 1/12 | 8/12 | 7/12 | 11/12 |
| Animal sounds only | 8/12 | 10/12 | 10/12 | 12/12 |
| Paws + Genki | 11/12 | 6/12 | 2/12 | 11/12 |
| Paws + Hugs + Genki | 11/12 | 2/11 (one unknown) | 1/12 | 12/12 |
| Those three + Sounds | 12/12 | 2/12 | 1/12 | 11/12 |
| All five modifiers | 12/12 | 2/12 | 1/12 | 10/12 |

The sound category includes mew, nya and purr, not just literal “meow”. Traits were scored even in replies marked incoherent, with coherence reported separately. Joint role is the conjunction of explicit assistant cat identity, explicit female identity and coherence. Absence of an explicit statement does not prove an internal identity disappeared. Repetitive character introductions can drive sound counts without fulfilling the request or establishing a rich personality.

The unexpected Paws+Genki effect (11/12 sounds versus Paws2/12 and Genki1/12) was selected from this grid after outcomes. It was not a primary prediction or a prespecified integrated endpoint. A separately frozen Study114 will test it on fresh prompts and matched random-coordinate controls. Its new data must not be pooled into113 or described as the original reserved confirmation.

## Did the atlas descriptions help predict outcomes?

Both arms saw identical singleton discovery outputs. The source-informed arm additionally received original descriptions, coordinate mapping and related published recipe claims. Thus this tests that bundled information, not isolated wording. Two fresh sessions per arm; additive baseline separately frozen from discovery. Lower Brier loss is better; endpoints remain separate.

| Endpoint | Source bundle | Observations only | Additive singleton baseline |
|---|---:|---:|---:|
| integrated_embodied_affection | 0.0647 | 0.0178 | 0.0060 |
| excited_feline_vocalization | 0.2115 | 0.1959 | 0.1548 |
| affectionate_feline_vocalization | 0.0715 | 0.0241 | 0.0060 |
| integrated_excited_affection | 0.0934 | 0.0347 | 0.0060 |
| defensive_affection | 0.0020 | 0.0026 | 0.0000 |
| assistant_cat_identity | 0.3267 | 0.2944 | 0.2680 |
| coherent | 0.0827 | 0.1095 | 0.0536 |

Source-informed forecasts were worse than observations-only on four of five integrated endpoints, better on defensive affection (which never appeared), worse on cat identity and better on coherence. The deterministic additive baseline beat both agent arms on every endpoint. Rare and absent outcomes make conservative forecasts strong; this is not evidence that the baseline understands composition. Two sessions per arm do not support a population-level agent-performance claim. The simple semantic ingredient descriptions did not give dependable predictive improvement here.

## Validity, uncertainty and relation to Goodfire

Two condition-masked reviews covered all408 replies. Six fresh sessions each rated136 replies; second-review order was independently shuffled.197 of5,712 feature cells in133 replies went to fresh masked adjudication. Seven final identity cells remain uncertain (three cat, four female); all primary endpoint cells are complete. Both original reviews and uncertain-as-zero/one tables remain separate. Shared agent judgment errors and the strict integrated rubric are limitations; there is no independent human gold standard.

Five independent arithmetic audits matched32,360 table rows /350,065 fields across rates, paired contrasts and forecast scores. The separate blinded statistics audit checked the bootstrap and exact-test implementation synthetically. [Audit and guard](STATISTICAL_AUDIT.md) preserve a fixed-family Holm correction and missing-family supplements without rewriting frozen tables; this correction does not change the present primary results. Arithmetic agreement does not validate semantic ratings or generalization.

Goodfire argues that native mechanisms can yield legible explanations which survive specific tests, and emphasizes checking activation, steering and neighboring mechanisms together ([original essay](https://already-clean.goodfire.com/)). Here we tested a narrow operational consequence: can these published steering descriptions support useful forecasts of new compositions? They did not reliably do so under these doses and prompts. That limits these coarse labels' predictive usefulness; it does not refute natural neuron legibility, quantify model-wide cleanliness, establish arbitrary superposition, or test recursive generalization during training. Strong unnatural offsets and selected published ingredients are not a model-wide natural-function audit.

## Artifacts and recovery

All numerical tables are in [analysis](analysis/); original-reviewer and uncertainty tables are in parallel analysis folders, with [independent checks](audit/). [Discovery replies and ratings](discovery_results/) preserve every calibration outcome. The complete local validation replies, evidence, locks and raw traces are retained in results.local; their public raw-payload archive is awaiting explicit publication approval following automatic review rejection. Methods, frozen predictions and summary numerical results can be public without that archive. The local complete recovery archive (210 files, SHA-256 `e6e4b271e54d36c0a06916fc3694ecb6983eb4855f535d1bbf235ab011a2d824`) was extracted into a clean directory; all60 primary/reviewer/uncertainty analysis CSVs reproduced byte-for-byte. [CPU replay record](recovery/CPU_REPLAY.json). [Public numeric labels](numeric_results/) contain no generated text and allow the same CPU analysis without publishing the raw reply payload.
