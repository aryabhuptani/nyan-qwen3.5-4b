# Frozen discovery selector

Run after masked independent reviews and adjudication have exported `labels.csv`:

```sh
python3 catgirl/composition/115_paws_genki_doses/analyze_discovery.py \
  --labels /path/to/scoring/labels.csv \
  --out-dir /path/to/discovery_analysis
```

The script reads the frozen discovery job, cases and feature definitions beside it. It checks the cases hash, all 20 dose states, exactly one row for each of the 60 condition/case combinations, distinct item IDs, and binary or blank feature cells. It checks an adjacent `labels_lock.json` if present; use `--labels-lock PATH` to require another lock location. A verified lock must authenticate the adjudicated `labels.csv` and describe 60 items. Lock absence is recorded explicitly in selection provenance. No reply text is read or used for selection.

`discovery_summary.csv` preserves all 20 states. Each feature and derived outcome has positive, zero and unknown counts over three cases. Clean catgirl requires coherence, assistant cat identity, female identity, and both idol and mascot identity absent. Clean catgirl with sound also requires performed feline vocalization. Unknown cells never establish a conjunction. Idol-or-mascot counts replies with either known positive; potentially drifting replies with no known positive and an unknown identity cell remain separately unknown.

`discovery_selection.json` records the candidate, baseline, full ranking, fixed rule, strict improvement decision and hashes of inputs and this script. Ranking maximizes clean catgirl with sound, then clean catgirl, minimizes known idol-or-mascot replies, then total dose, Genki and Paws. Decimal arithmetic avoids dose tie rounding. Only a strict gain in the first outcome, or the second with the first tied, permits confirmation. A drift-only or dose-only improvement does not qualify. Base and single modifiers remain eligible.

Repeated runs must reproduce existing artifacts exactly; changed artifacts are refused. Use a separate destination for another scoring export. This script performs discovery selection only and does not optimize from confirmation results. The lock verifies exported label bytes, not reviewer independence or correctness.
