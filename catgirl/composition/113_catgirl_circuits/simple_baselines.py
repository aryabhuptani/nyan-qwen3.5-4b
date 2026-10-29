"""Freeze a transparent singleton-addition forecast from locked discovery labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ['paws', 'hugs', 'genki', 'sounds', 'pride']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--labels-lock', type=Path, required=True)
    p.add_argument('--selected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    lock = json.loads(a.labels_lock.read_text())
    assert lock['labels_sha256'][a.labels.name] == sha(a.labels)
    selected = json.loads(a.selected.read_text())
    features = json.loads((ROOT/'FEATURE_DEFINITIONS.json').read_text())
    with a.labels.open() as f:
        rows = list(csv.DictReader(f))
    rates, counts = {}, {}
    conditions = ['base'] + [f'{n}_{selected[n]:g}' for n in NAMES]
    for condition in conditions:
        rr = [r for r in rows if r['condition_id'] == condition]
        assert len(rr) == 4
        rates[condition], counts[condition] = {}, {}
        for feature in features['forecast_features']:
            xs = [int(r[feature]) for r in rr if r[feature] in ('0','1')]
            rates[condition][feature] = sum(xs)/len(xs) if xs else .5
            counts[condition][feature] = len(xs)
    families = sorted({c['family_id'] for c in json.loads((ROOT/'CASES_validation.json').read_text())['cases']})
    predictions = []
    for state in features['forecast_states']:
        for family in families:
            for feature in features['forecast_features']:
                base = rates['base'][feature]
                q = base + sum(rates[conditions[i+1]][feature]-base for i,b in enumerate(state) if b == '1')
                predictions.append(dict(session_id='additive_singletons', arm='deterministic_additive',
                    condition_id=state, family_id=family, feature=feature, probability=max(0.,min(1.,q))))
    assert len(predictions) == 588
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    with (a.output/'forecasts.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
    (a.output/'discovery_base_rates.json').write_text(json.dumps(rates['base'],indent=2)+'\n')
    (a.output/'lock.json').write_text(json.dumps(dict(rates=rates, counts=counts,
        input_sha256={p.name:sha(p) for p in [a.labels,a.labels_lock,a.selected,Path(__file__)]},
        forecasts_sha256=sha(a.output/'forecasts.csv'), probabilities=588),indent=2)+'\n')

if __name__ == '__main__':
    main()
