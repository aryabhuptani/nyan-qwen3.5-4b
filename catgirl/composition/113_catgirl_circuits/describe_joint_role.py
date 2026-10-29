"""Descriptive joint-role rates; never substitutes for the frozen primary endpoints."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROLE = ('assistant_cat_identity', 'assistant_female_identity', 'coherent')


def conjunction(values):
    values = list(values)
    if 0 in values:
        return 0
    if None in values:
        return None
    return 1


def label(value):
    if value in ('', 'null', 'None', 'none'):
        return None
    if value not in ('0', '1'):
        raise ValueError('Invalid label: ' + value)
    return int(value)


def summarize(rows):
    endpoints = [p['endpoint'] for p in json.loads((HERE / 'FEATURE_DEFINITIONS.json').read_text())['primary_pairs']]
    groups = defaultdict(list)
    keys = set()
    for row in rows:
        key = (row['condition_id'], row['case_id'])
        if key in keys:
            raise ValueError('Duplicate condition/case')
        keys.add(key)
        role = conjunction(label(row[f]) for f in ROLE)
        for feature in ['explicit_joint_catgirl'] + endpoints:
            value = role if feature == 'explicit_joint_catgirl' else conjunction((role, label(row[feature])))
            groups[(row['condition_id'], feature)].append(value)
    return [dict(condition_id=c, feature=f, n_total=len(v), n_scored=sum(x is not None for x in v),
                 positives=sum(x == 1 for x in v), unknowns=sum(x is None for x in v),
                 rate=(sum(x == 1 for x in v) / sum(x is not None for x in v)) if any(x is not None for x in v) else None)
            for (c, f), v in sorted(groups.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        assert conjunction([1, 1, 1]) == 1
        assert conjunction([1, None, 1]) is None
        assert conjunction([0, None, 1]) == 0
        print('Joint role three-valued conjunction passed.')
        return
    if not a.labels or not a.output:
        p.error('--labels and --output are required')
    if a.output.exists():
        raise ValueError('Refusing to overwrite descriptive artifact')
    rows = summarize(list(csv.DictReader(a.labels.open())))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(
        labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),
        definition='cat identity AND female identity AND coherence; each integrated endpoint additionally ANDed with this role',
        scope='Descriptive only; explicit statements in independent replies do not establish latent identity or turn persistence.',
        rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
