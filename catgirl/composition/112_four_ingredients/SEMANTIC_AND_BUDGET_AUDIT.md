# Post-lock semantic-assay and perturbation interpretation

All four matched ingredients are retained. This note interprets the completed primary outputs; it does not change settings, endpoints or cases.

## Independent numerical comparison

- `INDEPENDENT_PRIMARY_SCORE_AUDIT.json`: independently reconstructs objective outcomes, all504 separate-session/per-reviewer forecast score groups and per-endpoint arm differences. Maximum score discrepancy is2.78×10⁻¹⁷, below the fixed10⁻¹² tolerance.
- `INDEPENDENT_PRIMARY_FULL_AUDIT.json`: compares immutable independent results to primary scores and objective outcomes, and independently recomputes all336 lexical and1,344 semantic causal CSV rows directly from raw outputs. It checks9,160 numerical values, including240 unique primary natural/lexical objective outcomes. Maximum discrepancy is2.78×10⁻¹⁷. No study metrics or settings changed.

The extended audit initially needed a bookkeeping correction to distinguish repeated baseline rows from target-coordinate and random-coordinate passes. It now matches their explicit input order. This was an auditor indexing issue, not a change to the primary analysis or measured data. The original independent result files remain unchanged.

## Semantic interpretation assay is uninformative at baseline

Matched Yes/No accuracy is0.5 before intervention and remains0.5 for every enhancement, attenuation and random control, at both carrier and answer-readout scopes. The assay is biased toward Yes and does not provide a reliable test of improved semantic interpretation. Its small likelihood shifts must not be described as meaning changes or as negative evidence for a neuron's semantic function.

Below are descriptive changes in Yes-minus-No summed log probability. “Discrimination change” is the enhancement's difference between semantic-present and semantic-absent log-odds shifts, not an accuracy improvement.

| Scope | Ingredient | Enhancement mean Yes shift | Attenuation mean Yes shift | Enhancement discrimination change |
|---|---|---:|---:|---:|
| Carrier | Genki |0.01042|0|0.02083|
| Carrier | Hugs |−0.03125|0|0.10417|
| Carrier | Paws |−0.01042|−0.03125|−0.02083|
| Carrier | Female |−0.01042|0.02083|0.06250|
| Answer readout | Genki |0|0|−0.04167|
| Answer readout | Hugs |0|0|0|
| Answer readout | Paws |0|0.01042|0|
| Answer readout | Female |0|0.01042|0|

The raw cell values, off-target effects and random comparators remain in `analysis/semantic_cases.csv` and `analysis/semantic_effects.csv`. Any later baseline generation sanity check is a post-hoc diagnosis of this failed assay and cannot replace the frozen primary result.

## Generation interventions were applied, with imperfect strength matching

The following are case means across20 full-four replies. Residual RMS uses achieved native scalar change times that coordinate's down-projection column norm. Ratios compare case-mean achieved RMS with the frozen discovery matching budget for that coordinate.

| Full-four ingredient | Mean achieved residual RMS | Ratio to discovery budget | Mean unchanged positions |
|---|---:|---:|---:|
| Genki |0.03867|1.217|14.22%|
| Hugs |0.05633|1.035|4.89%|
| Paws |0.12432|2.175|1.44%|
| Female |0.01687|1.805|14.56%|

Thus the generation null is not explained by hooks universally doing nothing. However, native-range one-sided operations change strength with context, so controls matched on discovery are not perfectly matched on held-out generation.

The frozen full-four nominal discovery block budget is0.08560. For comparison, the root-mean-square of the20 achieved per-case block-RMS values is:

| Condition | Achieved block RMS |
|---|---:|
| Full-four |0.14859|
| Random-four |0.08548|
| Strong Genki |0.08993|
| Strong Hugs |0.08625|
| Strong Paws |0.13938|
| Strong Female |0.08834|

Strong Paws could reach only76.2% of the requested discovery budget within its frozen fraction≤1 cap, while its held-out achieved strength was larger. These differing reference contexts explain why a nominal discovery budget and held-out achieved budget must both be shown. Summing squared norms across different layers is a declared nuisance measure, not exact functional equivalence.

## Range departures need both percentage and magnitude

Random coordinates are additive controls and are explicitly not constrained to their own native ranges; their held-out post-intervention departure rates cannot be interpreted as physiological necessity tests. Target operations leave naturally already-more-canonical values unchanged even if those values exceed the synthetic discovery support.

The strong-Paws summary reports about26% of positions outside observed discovery support. Direct raw inspection finds456 newly outside positions among1,786 operated positions, with maximum introduced excess only0.0009765625 above the discovery maximum0.1513671875: one BF16 endpoint step. Larger post values were already outside discovery support. Other strong singleton operations introduce zero new own-support departures. This rounding detail limits a literally exact boundedness claim, but the percentage alone would misleadingly suggest large newly induced excursions.

The native quantities are neither no-op nor perfectly strength-matched, and the Yes/No assay is not functioning as an adequate interpretation test. The report should retain both qualifications alongside the unchanged generated personality traits.
