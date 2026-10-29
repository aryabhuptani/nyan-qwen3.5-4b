#!/usr/bin/env python3
"""Frozen Study114 analysis; standard library only, no inference or network access."""
import argparse
import csv
import itertools
import json
import random
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEATURES = json.loads((HERE / 'FEATURE_DEFINITIONS.json').read_text())['feature_order']
ENDPOINT = 'performed_feline_vocalization'
CANONICAL = ('00000', '10000', '00100', '10100')
RANDOM = tuple(f'random_{i}' for i in range(1, 5))
CONDITIONS = CANONICAL + RANDOM
BOOTSTRAPS = 5000
SEED = 114
REQUIRED = {'condition_id', 'case_id', 'family_id', *FEATURES}
OPTIONAL = {'item_id', 'split', 'text_sha256', 'lexical_body', 'lexical_affection',
            'lexical_feline_sound', 'lexical_pride', 'adjudicated'}
METRICS = ('interaction', 'pair_minus_paws', 'pair_minus_genki', 'pair_minus_base',
           'pair_minus_case_better_single')


def mean(values):
    return statistics.fmean(values) if values else None


def read_json(path):
    return json.loads(Path(path).read_text())


def metadata(cases_path, job_path):
    cases = read_json(cases_path)['cases']
    mapping = {c['case_id']: c['family_id'] for c in cases}
    if len(cases) != 12 or len(mapping) != 12 or sorted(Counter(mapping.values()).values()) != [2]*6:
        raise ValueError('Cases must have twelve unique IDs, six families, two cases per family')
    if any(c['split'] != 'validation' for c in cases):
        raise ValueError('All cases must be validation cases')
    conditions = read_json(job_path)['conditions']
    ids = [c['condition_id'] for c in conditions]
    if len(ids) != 8 or set(ids) != set(CONDITIONS):
        raise ValueError('Job must contain exactly the eight registered conditions')
    return mapping


def label(value):
    if value is None or value.strip().lower() in ('', 'null'):
        return None
    if value not in ('0', '1'):
        raise ValueError(f'Expected literal 0, 1 or null/empty, got {value!r}')
    return int(value)


def read_labels(path, mapping):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not REQUIRED <= set(headers) or set(headers) - REQUIRED - OPTIONAL:
            raise ValueError('Missing, duplicated or unexpected labels headers')
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed CSV row width')
        key = row['condition_id'], row['case_id']
        if key in seen or key[0] not in CONDITIONS or key[1] not in mapping:
            raise ValueError(f'Duplicate or unregistered row {key}')
        seen.add(key)
        if row['family_id'] != mapping[key[1]]:
            raise ValueError(f'Incorrect family mapping {key}')
        if 'split' in row and row['split'] != 'validation':
            raise ValueError('Incorrect split')
        for feature in FEATURES:
            row[feature] = label(row[feature])
    if seen != set(itertools.product(CONDITIONS, mapping)):
        raise ValueError('Labels must contain exactly the full 8 by 12 grid')
    return rows


def quantile(sorted_values, q):
    position = (len(sorted_values)-1)*q
    low = int(position)
    return sorted_values[low] + (sorted_values[min(low+1, len(sorted_values)-1)]-sorted_values[low])*(position-low)


def bootstrap(records, statistic=mean):
    if not records:
        return None, None
    rng = random.Random(SEED)
    estimates = sorted(statistic([records[rng.randrange(len(records))] for _ in records])
                       for _ in range(BOOTSTRAPS))
    return quantile(estimates, .025), quantile(estimates, .975)


def signflip(values):
    if not values:
        return None
    if len(values) > 12:
        raise ValueError('Exact test is registered for at most twelve cases')
    observed = abs(sum(values))
    return sum(abs(sum(s*x for s, x in zip(signs, values))) >= observed-1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / 2**len(values)


def holm_two(pvalues):
    """Null tests retain their slot in the two-test planned family."""
    if len(pvalues) != 2:
        raise ValueError('The registered family has exactly two tests')
    out = [None, None]
    previous = 0
    for rank, (p, index) in enumerate(sorted((p, i) for i, p in enumerate(pvalues) if p is not None)):
        previous = max(previous, min(1., p*(2-rank)))
        out[index] = previous
    return out


def better(records):
    return mean([r['y11'] for r in records])-max(mean([r['y10'] for r in records]), mean([r['y01'] for r in records]))


def estimate(values):
    low, high = bootstrap(values)
    return {'n_complete': len(values), 'mean': mean(values), 'ci_low': low, 'ci_high': high}


def analyze(rows, mapping):
    lookup = {(r['condition_id'], r['case_id']): r for r in rows}
    rates = []
    for family in [None] + sorted(set(mapping.values())):
        for condition in CONDITIONS:
            selected = [r for r in rows if r['condition_id'] == condition and (family is None or r['family_id'] == family)]
            for feature in FEATURES:
                values = [r[feature] for r in selected if r[feature] is not None]
                rates.append(dict(family_id=family or 'ALL', condition_id=condition, feature=feature,
                                  n_total=len(selected), n_scored=len(values), n_unknown=len(selected)-len(values),
                                  positive=sum(values), rate=mean(values)))
    details = []
    for case, family in sorted(mapping.items()):
        yy = [lookup[c, case][ENDPOINT] for c in CANONICAL]
        rr = [lookup[c, case][ENDPOINT] for c in RANDOM]
        complete = all(y is not None for y in yy)
        specific = yy[3] is not None and all(y is not None for y in rr)
        row = dict(case_id=case, family_id=family, canonical_complete=int(complete), specificity_complete=int(specific),
                   **dict(zip(('y00', 'y10', 'y01', 'y11'), yy)),
                   **dict(zip(RANDOM, rr)))
        row.update({k: None for k in METRICS})
        if complete:
            base, paws, genki, pair = yy
            row.update(interaction=pair-paws-genki+base, pair_minus_paws=pair-paws,
                       pair_minus_genki=pair-genki, pair_minus_base=pair-base,
                       pair_minus_case_better_single=pair-max(paws, genki))
        row['pair_minus_mean_random'] = yy[3]-mean(rr) if specific else None
        for condition, value in zip(RANDOM, rr):
            row['pair_minus_'+condition] = yy[3]-value if yy[3] is not None and value is not None else None
        details.append(row)
    summaries = []
    for family in [None] + sorted(set(mapping.values())):
        selected = [d for d in details if family is None or d['family_id'] == family]
        complete = [d for d in selected if d['canonical_complete']]
        for metric in METRICS + ('pair_minus_mean_random',) + tuple('pair_minus_'+c for c in RANDOM):
            values = [d[metric] for d in selected if d[metric] is not None]
            summaries.append(dict(family_id=family or 'ALL', metric=metric, n_total=len(selected),
                                  n_missing=len(selected)-len(values), **estimate(values)))
        low, high = bootstrap(complete, better)
        summaries.append(dict(family_id=family or 'ALL', metric='pair_minus_better_single',
                              n_total=len(selected), n_complete=len(complete), n_missing=len(selected)-len(complete),
                              mean=better(complete) if complete else None, ci_low=low, ci_high=high))
        for cell in ('y00', 'y10', 'y01', 'y11'):
            summaries.append(dict(family_id=family or 'ALL', metric=cell+'_canonical_complete_rate',
                                  n_total=len(selected), n_missing=len(selected)-len(complete),
                                  **estimate([d[cell] for d in complete])))
    tests = []
    for metric in ('interaction', 'pair_minus_mean_random'):
        values = [d[metric] for d in details if d[metric] is not None]
        tests.append(dict(test=metric, endpoint=ENDPOINT, n_complete=len(values), n_total=12,
                          p_exact_two_sided=signflip(values)))
    for result, adjusted in zip(tests, holm_two([r['p_exact_two_sided'] for r in tests])):
        result['p_holm_fixed_two'] = adjusted
    return dict(feature_rates=rates, case_contrasts=details, contrast_summary=summaries, planned_tests=tests)


def forecast_scores(rows, path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['condition_id', 'probability']:
            raise ValueError('Forecast headers must be condition_id,probability')
        predictions = list(reader)
    probabilities = {}
    for row in predictions:
        if set(row) != {'condition_id', 'probability'} or row['condition_id'] in probabilities or row['condition_id'] not in CONDITIONS:
            raise ValueError('Malformed, duplicate or unregistered forecast condition')
        value = float(row['probability'])
        if not 0 <= value <= 1:
            raise ValueError('Forecast probability must be finite in [0,1]')
        probabilities[row['condition_id']] = value
    if len(probabilities) != 8:
        raise ValueError('Exactly eight condition probabilities required')
    scored = []
    for row in rows:
        y = row[ENDPOINT]
        p = probabilities[row['condition_id']]
        scored.append(dict(condition_id=row['condition_id'], case_id=row['case_id'], family_id=row['family_id'],
                           probability=p, outcome=y, brier=(p-y)**2 if y is not None else None))
    summary = []
    for condition in CONDITIONS:
        values = [r['brier'] for r in scored if r['condition_id'] == condition and r['brier'] is not None]
        summary.append(dict(condition_id=condition, n_scored=len(values), n_unknown=12-len(values), brier=mean(values)))
    return scored, summary


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    import tempfile
    mapping = {f'case{i}': f'family{i//2}' for i in range(12)}
    rows = [dict(condition_id=c, case_id=k, family_id=f, **{feature: 0 for feature in FEATURES})
            for c in CONDITIONS for k, f in mapping.items()]
    for row in rows:
        row[ENDPOINT] = int(row['condition_id'] == '10100')
    result = analyze(rows, mapping)
    assert len(result['case_contrasts']) == 12
    assert all(d['interaction'] == 1 and d['pair_minus_mean_random'] == 1 for d in result['case_contrasts'])
    assert signflip([1]*12) == 2/4096 and signflip([0]*12) == 1
    assert better([{'y11': 1, 'y10': 1, 'y01': 0}, {'y11': 1, 'y10': 0, 'y01': 1}]) == .5
    assert holm_two([None, .03]) == [None, .06]
    assert holm_two([.03, .04]) == [.06, .06]
    assert all(label(x) is None for x in ('', 'null', 'NULL'))
    for bad in ('uncertain', 'NaN', '2', '0.0', 'True'):
        try:
            label(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid label accepted')
    rows[0][ENDPOINT] = None
    next(r for r in rows if r['condition_id'] == 'random_1' and r['case_id'] == 'case1')[ENDPOINT] = None
    result = analyze(rows, mapping)
    assert result['planned_tests'][0]['n_complete'] == 11
    assert result['planned_tests'][1]['n_complete'] == 11
    assert result['case_contrasts'][0]['interaction'] is None
    assert result['case_contrasts'][0]['pair_minus_mean_random'] == 1
    assert result['case_contrasts'][1]['pair_minus_mean_random'] is None
    assert result['case_contrasts'][1]['pair_minus_random_2'] == 1
    for row in rows:
        if row['family_id'] == 'family0':
            row[ENDPOINT] = None
    result = analyze(rows, mapping)
    assert any(r['family_id'] == 'family0' and r['n_complete'] == 0 and r['mean'] is None for r in result['contrast_summary'])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'labels.csv'
        save_csv(path, rows)
        assert len(read_labels(path, mapping)) == 96
        duplicate = rows[:-1] + [rows[0]]
        save_csv(path, duplicate)
        try:
            read_labels(path, mapping)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate grid row accepted')
        save_csv(path, rows[:-1])
        try:
            read_labels(path, mapping)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete grid accepted')
        malformed = [dict(r) for r in rows]
        malformed[0]['family_id'] = 'wrong'
        save_csv(path, malformed)
        try:
            read_labels(path, mapping)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong family accepted')
    print('Study114 synthetic self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--cases', type=Path, default=HERE/'CASES_validation.json')
    parser.add_argument('--job', type=Path, default=HERE/'JOB_validation.json')
    parser.add_argument('--forecasts', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.labels is None or args.out_dir is None:
        parser.error('--labels and --out-dir are required unless --self-test')
    mapping = metadata(args.cases, args.job)
    rows = read_labels(args.labels, mapping)
    result = analyze(rows, mapping)
    if args.forecasts:
        result['forecast_case_scores'], result['forecast_condition_scores'] = forecast_scores(rows, args.forecasts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in result.items():
        save_csv(args.out_dir/(name+'.csv'), records)
    (args.out_dir/'analysis_summary.json').write_text(json.dumps(dict(endpoint=ENDPOINT, seed=SEED,
        bootstraps=BOOTSTRAPS, label_input=str(args.labels), case_input=str(args.cases), job_input=str(args.job),
        results=result), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
