#!/usr/bin/env python3
"""Independent CPU recomputation for Study114; no primary numeric imports.

Prepare/test without outcomes:
  python3 independent_audit.py --self-test
After labels are locked and outcome access is authorized:
  python3 independent_audit.py --labels LABELS/labels.csv --analysis ANALYSIS \
      --out-dir NEW_AUDIT [--forecasts FORECASTS.csv]

The adjacent labels_lock.json must authenticate the label CSV. Authenticate that
lock with --labels-lock-sha256 HEX, or its adjacent export_manifest.json. The
checked-in FROZEN_VALIDATION.json authenticates cases, job, feature definitions,
primary analyzer and optional forecasts. Independent outputs and a comparison
report go to a fresh directory; original files are never changed. Comparisons
use absolute tolerance 1e-12 and require identical missing values and row keys.

Pre-outcome review found no numerical mismatch in the frozen implementation.
Its primary CLI does not itself verify labels_lock.json or frozen metadata
hashes (analyze_replication.py:302-306); this verifier supplies those provenance
checks. The primary output directory is intentionally overwriteable (307-309),
as documented by ANALYSIS_INTERFACE.md. Neither is a demonstrated wrong result.
The statistical assumptions and generalization limits remain those in PROTOCOL.
"""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CANONICAL = ('00000', '10000', '00100', '10100')
CONTROLS = tuple('random_' + str(i) for i in range(1, 5))
CONDITIONS = CANONICAL + CONTROLS
ENDPOINT = 'performed_feline_vocalization'
METRICS = ('interaction', 'pair_minus_paws', 'pair_minus_genki', 'pair_minus_base',
           'pair_minus_case_better_single')
OPTIONAL = {'item_id', 'split', 'text_sha256', 'lexical_body', 'lexical_affection',
            'lexical_feline_sound', 'lexical_pride', 'adjudicated'}
TOLERANCE = 1e-12
KEYS = {
    'feature_rates': ('family_id', 'condition_id', 'feature'),
    'case_contrasts': ('case_id',), 'contrast_summary': ('family_id', 'metric'),
    'planned_tests': ('test',), 'forecast_case_scores': ('condition_id', 'case_id'),
    'forecast_condition_scores': ('condition_id',),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def check_digest(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError('SHA256 mismatch: ' + str(path))


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError('Missing/duplicate CSV header: ' + str(path))
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Malformed CSV row: ' + str(path))
    return fields, rows


def metadata(cases_path, job_path, features_path, freeze_path):
    frozen = load_json(freeze_path)['input_sha256']
    for name, path in [('CASES_validation.json', cases_path), ('JOB_validation.json', job_path),
                       ('FEATURE_DEFINITIONS.json', features_path),
                       ('analyze_replication.py', HERE/'analyze_replication.py')]:
        check_digest(path, frozen[name])
    cases = load_json(cases_path)['cases']
    mapping = {row['case_id']: row['family_id'] for row in cases}
    if len(cases) != 12 or len(mapping) != 12 or sorted(Counter(mapping.values()).values()) != [2]*6:
        raise ValueError('Expected twelve unique cases in six two-case families')
    if any(row['split'] != 'validation' for row in cases):
        raise ValueError('Non-validation case')
    if any(not isinstance(x, str) or not x or x == 'ALL' for x in mapping.values()):
        raise ValueError('Invalid/reserved family ID')
    job = load_json(job_path)
    ids = [row['condition_id'] for row in job['conditions']]
    if len(ids) != 8 or set(ids) != set(CONDITIONS):
        raise ValueError('Incorrect eight-condition job')
    check_digest(cases_path, job['cases_sha256'])
    features = load_json(features_path)['feature_order']
    if len(features) != 14 or len(set(features)) != 14 or ENDPOINT not in features:
        raise ValueError('Expected fourteen unique features including the canonical endpoint')
    return mapping, features, frozen


def locked_labels(path, lock_path, expected_lock_digest, mapping, features):
    check_digest(lock_path, expected_lock_digest)
    lock = load_json(lock_path)
    check_digest(path, lock['labels_sha256'][Path(path).name])
    fields, rows = csv_rows(path)
    required = {'case_id', 'condition_id', 'family_id', *features}
    if not required <= set(fields) or set(fields)-required-OPTIONAL:
        raise ValueError('Missing/unexpected label header')
    seen = set()
    expected = set(itertools.product(CONDITIONS, mapping))
    for row in rows:
        key = row['condition_id'], row['case_id']
        if key not in expected or key in seen:
            raise ValueError('Unexpected/duplicate condition-case label')
        seen.add(key)
        if row['family_id'] != mapping[row['case_id']] or row.get('split', 'validation') != 'validation':
            raise ValueError('Incorrect label family/split')
        for feature in features:
            raw = row[feature]
            if raw.strip().lower() in ('', 'null'):
                row[feature] = None
            elif raw in ('0', '1'):
                row[feature] = int(raw)
            else:
                raise ValueError('Labels must be literal 0/1/empty/null')
    if seen != expected or len(rows) != 96:
        raise ValueError('Incomplete 8 x 12 label grid')
    return rows


def average(values):
    return math.fsum(values)/len(values) if values else None


def exact_signs(values):
    """Enumerate 2**n assignments using integer quarter-units, with no tolerance."""
    if not values:
        return None
    if len(values) > 12:
        raise ValueError('At most twelve sign-flip cases')
    units = [int(round(4*v)) for v in values]
    if any(4*v != u for v, u in zip(values, units)):
        raise ValueError('Study114 contrasts must be exact quarter-integers')
    observed = abs(sum(units))
    hits = 0
    for mask in range(1 << len(units)):
        signed = sum(-u if mask & (1 << i) else u for i, u in enumerate(units))
        hits += abs(signed) >= observed
    return hits / (1 << len(units))


def adjust_two(pvalues):
    if len(pvalues) != 2:
        raise ValueError('Exactly two planned tests')
    observed = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    result = [None, None]
    for rank, (p, i) in enumerate(observed):
        adjusted = min(1.0, (2-rank)*p)
        if rank:
            adjusted = max(adjusted, result[observed[rank-1][1]])
        result[i] = adjusted
    return result


def aggregate_better(records):
    return average([r['y11'] for r in records])-max(
        average([r['y10'] for r in records]), average([r['y01'] for r in records]))


def interval(records, function=average):
    if not records:
        return None, None
    generator = random.Random(114)
    draws = []
    for _ in range(5000):
        replicate = [records[generator.randrange(len(records))] for _ in range(len(records))]
        draws.append(function(replicate))
    draws.sort()
    bounds = []
    for fraction in (.025, .975):
        location = fraction * 4999
        lower = math.floor(location)
        bounds.append(draws[lower] + (location-lower)*(draws[lower+1]-draws[lower]))
    return tuple(bounds)


def recompute(rows, mapping, features):
    rates, cases, summary = [], [], []
    families = ['ALL'] + sorted(set(mapping.values()))
    lookup = {(row['condition_id'], row['case_id']): row for row in rows}
    for family in families:
        for condition in CONDITIONS:
            selected = [r for r in rows if r['condition_id'] == condition and (family == 'ALL' or r['family_id'] == family)]
            for feature in features:
                known = [r[feature] for r in selected if r[feature] is not None]
                rates.append(dict(family_id=family, condition_id=condition, feature=feature,
                                  n_total=len(selected), n_scored=len(known), n_unknown=len(selected)-len(known),
                                  positive=sum(known), rate=average(known)))
    for case in sorted(mapping):
        base, paws, genki, pair = [lookup[c, case][ENDPOINT] for c in CANONICAL]
        randoms = [lookup[c, case][ENDPOINT] for c in CONTROLS]
        canonical_ok = None not in (base, paws, genki, pair)
        random_ok = pair is not None and None not in randoms
        row = dict(case_id=case, family_id=mapping[case], canonical_complete=int(canonical_ok),
                   specificity_complete=int(random_ok), y00=base, y10=paws, y01=genki, y11=pair,
                   **dict(zip(CONTROLS, randoms)))
        row.update(dict.fromkeys(METRICS))
        if canonical_ok:
            row.update(interaction=base+pair-paws-genki, pair_minus_paws=pair-paws,
                       pair_minus_genki=pair-genki, pair_minus_base=pair-base,
                       pair_minus_case_better_single=pair-max(paws, genki))
        row['pair_minus_mean_random'] = pair - sum(randoms)/4 if random_ok else None
        for control, value in zip(CONTROLS, randoms):
            row['pair_minus_'+control] = pair-value if pair is not None and value is not None else None
        cases.append(row)
    metrics = METRICS + ('pair_minus_mean_random',) + tuple('pair_minus_'+c for c in CONTROLS)
    for family in families:
        selected = [r for r in cases if family == 'ALL' or r['family_id'] == family]
        canonical = [r for r in selected if r['canonical_complete']]
        for metric in metrics + ('pair_minus_better_single',) + tuple(c+'_canonical_complete_rate' for c in ('y00','y10','y01','y11')):
            if metric == 'pair_minus_better_single':
                sample, function = canonical, aggregate_better
            elif metric.endswith('_canonical_complete_rate'):
                sample, function = [r[metric[:3]] for r in canonical], average
            else:
                sample, function = [r[metric] for r in selected if r[metric] is not None], average
            low, high = interval(sample, function)
            summary.append(dict(family_id=family, metric=metric, n_total=len(selected), n_missing=len(selected)-len(sample),
                                n_complete=len(sample), mean=function(sample) if sample else None, ci_low=low, ci_high=high))
    tests = []
    for metric in ('interaction', 'pair_minus_mean_random'):
        values = [r[metric] for r in cases if r[metric] is not None]
        tests.append(dict(test=metric, endpoint=ENDPOINT, n_complete=len(values), n_total=12,
                          p_exact_two_sided=exact_signs(values)))
    for row, p in zip(tests, adjust_two([r['p_exact_two_sided'] for r in tests])):
        row['p_holm_fixed_two'] = p
    return dict(feature_rates=rates, case_contrasts=cases, contrast_summary=summary, planned_tests=tests)


def recompute_forecasts(rows, path):
    fields, forecasts = csv_rows(path)
    if fields != ['condition_id', 'probability']:
        raise ValueError('Incorrect forecast headers')
    predictions = {}
    for row in forecasts:
        condition = row['condition_id']
        p = float(row['probability'])
        if condition not in CONDITIONS or condition in predictions or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Duplicate/unregistered/nonfinite/out-of-range forecast')
        predictions[condition] = p
    if set(predictions) != set(CONDITIONS):
        raise ValueError('Missing forecast condition')
    scores = []
    for row in rows:
        p, y = predictions[row['condition_id']], row[ENDPOINT]
        scores.append(dict(condition_id=row['condition_id'], case_id=row['case_id'], family_id=row['family_id'],
                           probability=p, outcome=y, brier=None if y is None else (y-p)*(y-p)))
    summaries = []
    for condition in CONDITIONS:
        known = [r['brier'] for r in scores if r['condition_id'] == condition and r['brier'] is not None]
        summaries.append(dict(condition_id=condition, n_scored=len(known), n_unknown=12-len(known), brier=average(known)))
    return dict(forecast_case_scores=scores, forecast_condition_scores=summaries)


def compare(tables, analysis):
    differences, comparisons = [], 0
    for name, records in tables.items():
        fields, observed = csv_rows(Path(analysis)/(name+'.csv'))
        expected_fields = set(records[0])
        if set(fields) != expected_fields:
            differences.append(dict(table=name, problem='column mismatch'))
            continue
        keys = KEYS[name]
        expected = {tuple(str(r[k]) for k in keys): r for r in records}
        actual = {tuple(r[k] for k in keys): r for r in observed}
        if len(actual) != len(observed) or set(actual) != set(expected):
            differences.append(dict(table=name, problem='duplicate/missing/unexpected row keys'))
            continue
        for key, row in expected.items():
            for column, value in row.items():
                comparisons += 1
                raw = actual[key][column]
                if value is None:
                    matches = raw == ''
                elif isinstance(value, (int, float)):
                    try:
                        parsed = float(raw)
                        matches = math.isfinite(parsed) and abs(parsed-value) <= TOLERANCE
                    except ValueError:
                        matches = False
                else:
                    matches = raw == value
                if not matches:
                    differences.append(dict(table=name, key=list(key), column=column,
                                            expected=value, observed=raw))
    return dict(passed=not differences, tolerance=TOLERANCE, compared_cells=comparisons,
                differences=differences, tables={name: len(rows) for name, rows in tables.items()})


def save_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    features = load_json(HERE/'FEATURE_DEFINITIONS.json')['feature_order']
    mapping = {f'case{i:02}': 'family'+str(i//2) for i in range(12)}
    rows = [dict(condition_id=c, case_id=k, family_id=f, **dict.fromkeys(features, 0))
            for c in CONDITIONS for k,f in mapping.items()]
    for row in rows:
        row[ENDPOINT] = int(row['condition_id'] == '10100')
    assert exact_signs([1]*12) == 2/4096
    assert exact_signs([0]*12) == 1 and exact_signs([1, -1]) == 1
    assert exact_signs([.25, .5]) == .5 and exact_signs([]) is None
    assert adjust_two([None,.03]) == [None,.06]
    assert adjust_two([.03,.04]) == [.06,.06]
    assert aggregate_better([dict(y11=1,y10=1,y01=0),dict(y11=1,y10=0,y01=1)]) == .5
    # Independently incomplete canonical and random cohorts; one all-unknown family.
    for row in rows:
        if row['family_id'] == 'family0' or (row['case_id'] == 'case02' and row['condition_id'] == '00000') or (row['case_id'] == 'case03' and row['condition_id'] == 'random_1'):
            row[ENDPOINT] = None
    tables = recompute(rows, mapping, features)
    tests = {r['test']: r for r in tables['planned_tests']}
    assert tests['interaction']['n_complete'] == tests['pair_minus_mean_random']['n_complete'] == 9
    assert len(tables['feature_rates']) == 8*14*7
    assert all(r['n_complete'] == 0 for r in tables['contrast_summary'] if r['family_id'] == 'family0')
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        labels, lock = temp/'labels.csv', temp/'labels_lock.json'
        save_csv(labels, rows)
        lock.write_text(json.dumps(dict(labels_sha256={labels.name:digest(labels)})))
        parsed = locked_labels(labels, lock, digest(lock), mapping, features)
        assert parsed == rows
        forecasts = temp/'forecasts.csv'
        save_csv(forecasts, [dict(condition_id=c, probability=.75) for c in CONDITIONS])
        tables.update(recompute_forecasts(parsed, forecasts))
        # Primary code is executed only on these generated synthetic inputs.
        cases, job = temp/'cases.json', temp/'job.json'
        cases.write_text(json.dumps(dict(cases=[dict(case_id=k,family_id=f,split='validation') for k,f in mapping.items()])))
        job.write_text(json.dumps(dict(conditions=[dict(condition_id=c) for c in CONDITIONS])))
        primary = temp/'primary'
        subprocess.run([sys.executable, '-B', str(HERE/'analyze_replication.py'), '--labels', str(labels),
                        '--cases', str(cases), '--job', str(job), '--forecasts', str(forecasts), '--out-dir', str(primary)], check=True)
        report = compare(tables, primary)
        assert report['passed'], report
        # A numerical discrepancy and an altered label file must both fail.
        path = primary/'planned_tests.csv'
        fields, changed = csv_rows(path)
        changed[0]['p_exact_two_sided'] = '.987'
        path.unlink(); save_csv(path, changed)
        assert not compare(tables, primary)['passed']
        labels.write_text(labels.read_text()+'\n')
        try:
            locked_labels(labels, lock, digest(lock), mapping, features)
        except ValueError:
            pass
        else:
            raise AssertionError('Modified locked labels accepted')
    print('Independent synthetic self-test passed: counts, all-family unknowns, cohorts, contrasts/intervals, exact tests/Holm2, Briers, primary comparison, discrepancy detection and label hash rejection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test', '--selftest', action='store_true')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--labels-lock', type=Path)
    parser.add_argument('--labels-lock-sha256')
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--forecasts', type=Path)
    parser.add_argument('--cases', type=Path, default=HERE/'CASES_validation.json')
    parser.add_argument('--job', type=Path, default=HERE/'JOB_validation.json')
    parser.add_argument('--features', type=Path, default=HERE/'FEATURE_DEFINITIONS.json')
    parser.add_argument('--freeze', type=Path, default=HERE/'FROZEN_VALIDATION.json')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.labels or not args.analysis or not args.out_dir:
        parser.error('--labels, --analysis and fresh --out-dir are required')
    try:
        if args.out_dir.exists():
            raise ValueError('Audit output directory must be fresh')
        mapping, features, frozen = metadata(args.cases, args.job, args.features, args.freeze)
        lock = args.labels_lock or args.labels.parent/'labels_lock.json'
        expected = args.labels_lock_sha256
        if expected is None:
            expected = load_json(lock.parent/'export_manifest.json')['labels_lock_sha256']
        rows = locked_labels(args.labels, lock, expected, mapping, features)
        tables = recompute(rows, mapping, features)
        if args.forecasts:
            check_digest(args.forecasts, frozen['FORECASTS.csv'])
            tables.update(recompute_forecasts(rows, args.forecasts))
        report = compare(tables, args.analysis)
        report.update(label_sha256=digest(args.labels), labels_lock_sha256=digest(lock),
                      case_sha256=digest(args.cases), job_sha256=digest(args.job),
                      independent_script_sha256=digest(__file__),
                      forecast_comparison=bool(args.forecasts), primary_numeric_imports=False)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        for name, records in tables.items():
            save_csv(args.out_dir/(name+'.csv'), records)
        (args.out_dir/'comparison_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(passed=report['passed'], compared_cells=report['compared_cells'],
                          discrepancies=len(report['differences']), tolerance=TOLERANCE)))
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
