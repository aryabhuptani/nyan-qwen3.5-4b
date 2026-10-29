#!/usr/bin/env python3
"""Pre-outcome input guard and separate supplements for the frozen analyzer.

Validate only: python3 analysis_guard.py --labels locked.csv --forecasts forecasts.csv
Validate and run: add --out NEW_DIRECTORY [--baseline-rates frozen_rates.json]
Supplement an existing analysis: add --analysis EXISTING_DIRECTORY
Synthetic tests only: python3 analysis_guard.py --selftest
"""
import argparse
import copy
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from analyze import FEATURES, PAIRS, SPEC, bitstate, val

HERE = Path(__file__).resolve().parent
EXPECTED_SESSIONS = {
    'descriptions_1': 'descriptions', 'descriptions_2': 'descriptions',
    'observations_only_1': 'observations_only', 'observations_only_2': 'observations_only',
    'additive_singletons': 'deterministic_additive',
}


def read_csv(path, required):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not set(required) <= set(headers):
            raise ValueError('Missing or duplicate CSV headers')
        rows = list(reader)
    if any(None in r or any(r[k] is None for k in required) for r in rows):
        raise ValueError('Malformed CSV row')
    return rows


def design():
    job = json.loads((HERE / 'JOB_validation.json').read_text())
    cases = json.loads((HERE / 'CASES_validation.json').read_text())['cases']
    states = [r['condition_id'] for r in job['conditions']]
    case_family = {r['case_id']: r['family_id'] for r in cases}
    if len(states) != 34 or len(set(states)) != 34 or len(cases) != 12 or len(case_family) != 12:
        raise ValueError('Frozen design is not the expected 34-condition, 12-case design')
    families = set(case_family.values())
    if len(families) != 6 or any(list(case_family.values()).count(f) != 2 for f in families):
        raise ValueError('Frozen design must contain six families of two cases')
    return states, case_family


def probability(value):
    if isinstance(value, bool):
        raise ValueError('Boolean probability')
    p = float(value)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Probability must be finite and in [0,1]')
    return p


def validate_labels(path, states, case_family):
    rows = read_csv(path, {'case_id', 'condition_id', 'family_id', *FEATURES})
    expected = set(itertools.product(states, case_family))
    seen = set()
    for row in rows:
        key = row['condition_id'], row['case_id']
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate label condition/case')
        seen.add(key)
        if row['family_id'] != case_family[row['case_id']]:
            raise ValueError('Label family differs from frozen case family')
        for feature in FEATURES:
            row[feature] = val(row[feature])
    if seen != expected or len(rows) != 408:
        raise ValueError('Labels require the complete 34 x 12 grid (408 rows)')
    return rows


def validate_forecasts(path, case_family):
    rows = read_csv(path, {'session_id', 'arm', 'family_id', 'condition_id', 'feature', 'probability'})
    coverage = set(itertools.product(set(case_family.values()), SPEC['forecast_states'], SPEC['forecast_features']))
    expected = {(session, *key) for session in EXPECTED_SESSIONS for key in coverage}
    seen = set()
    for row in rows:
        session = row['session_id']
        key = session, row['family_id'], row['condition_id'], row['feature']
        if session not in EXPECTED_SESSIONS or row['arm'] != EXPECTED_SESSIONS[session]:
            raise ValueError('Unexpected session or inconsistent session arm')
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicate forecast cell')
        seen.add(key)
        probability(row['probability'])
    if seen != expected or len(rows) != 2940:
        raise ValueError('Forecasts require all five sessions, each with 588 cells (2940 rows)')


def validate_baseline(path):
    rates = json.loads(Path(path).read_text())
    if not isinstance(rates, dict) or set(rates) != set(SPEC['forecast_features']):
        raise ValueError('Baseline must contain exactly the seven forecast features')
    for value in rates.values():
        probability(value)


def fixed_five_holm(ps):
    if len(ps) != 5:
        raise ValueError('Exactly five primary hypotheses required')
    adjusted = [None] * 5
    previous = 0
    for rank, (p, index) in enumerate(sorted((p, i) for i, p in enumerate(ps) if p is not None)):
        previous = max(previous, min(1, (5-rank)*p))
        adjusted[index] = previous
    return adjusted


def write_new(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def supplement(analysis, labels, case_family):
    analysis = Path(analysis)
    primary = read_csv(analysis / 'primary_pairs.csv',
                       {'pair', 'interaction_p_signflip', 'interaction_p_holm_five'})
    by_pair = {r['pair']: r for r in primary}
    if len(primary) != 5 or set(by_pair) != {p['id'] for p in PAIRS}:
        raise ValueError('Analysis must contain the five canonical primary pairs')
    ordered = [by_pair[p['id']] for p in PAIRS]
    ps = [None if r['interaction_p_signflip'] == '' else probability(r['interaction_p_signflip']) for r in ordered]
    corrected = []
    for row, adjusted in zip(ordered, fixed_five_holm(ps)):
        original = None if row['interaction_p_holm_five'] == '' else probability(row['interaction_p_holm_five'])
        corrected.append(dict(pair=row['pair'], interaction_p_signflip=row['interaction_p_signflip'],
                              original_interaction_p_holm_five=original, corrected_fixed_five_holm=adjusted,
                              differs_from_original=original != adjusted, family_size=5,
                              note='Separate pre-outcome audit correction; original table retained. Untestable hypotheses remain in the five-test family.'))
    lookup = {(r['condition_id'], r['case_id']): r for r in labels}
    missingness = []
    for pair in PAIRS:
        a, b = pair['modifier_indices']
        states = [bitstate(s) for s in ([], [a], [b], [a,b])]
        for family in sorted(set(case_family.values())):
            cc = [c for c, f in case_family.items() if f == family]
            yy = [[lookup[s,c][pair['endpoint']] for s in states] for c in cc]
            complete = sum(all(y is not None for y in ys) for ys in yy)
            missingness.append(dict(pair=pair['id'], feature=pair['endpoint'], family_id=family,
                                    n_total_cases=len(cc), n_complete=complete, n_incomplete=len(cc)-complete,
                                    n_unknown_cells=sum(y is None for ys in yy for y in ys)))
    paths = [analysis / 'corrected_fixed_five_holm.csv', analysis / 'primary_family_missingness_supplement.csv']
    if any(p.exists() for p in paths):
        raise FileExistsError('Audit supplements already exist; refusing to overwrite')
    write_new(paths[0], corrected)
    write_new(paths[1], missingness)


def selftest():
    states, case_family = design()
    labels = [dict(condition_id=s, case_id=c, family_id=f, **dict.fromkeys(FEATURES, 0))
              for s in states for c, f in case_family.items()]
    forecasts = [dict(session_id=s, arm=a, family_id=f, condition_id=c, feature=t, probability=.5)
                 for s,a in EXPECTED_SESSIONS.items()
                 for f,c,t in itertools.product(set(case_family.values()), SPEC['forecast_states'], SPEC['forecast_features'])]
    assert fixed_five_holm([.01, None, None, None, None]) == [.05, None, None, None, None]
    assert fixed_five_holm([.01,.02,.03,.04,.05]) == [.05,.08,.09,.09,.09]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lp, fp = tmp/'labels.csv', tmp/'forecasts.csv'
        write_new(lp, labels); write_new(fp, forecasts)
        parsed = validate_labels(lp, states, case_family)
        validate_forecasts(fp, case_family)
        def rejected(fn):
            try:
                fn()
            except (ValueError, TypeError):
                return
            raise AssertionError('Invalid synthetic input accepted')
        bad = tmp/'missing_header.csv'
        write_new(bad, [{k:v for k,v in r.items() if k != FEATURES[0]} for r in labels])
        rejected(lambda: validate_labels(bad, states, case_family))
        variants = [forecasts[588:], copy.deepcopy(forecasts), copy.deepcopy(forecasts), forecasts[:-1]+forecasts[:1]]
        variants[1][0]['arm'] = 'wrong'
        variants[2][0]['probability'] = 'nan'
        for i, rows in enumerate(variants):
            bad = tmp/f'bad{i}.csv'; write_new(bad, rows)
            rejected(lambda: validate_forecasts(bad, case_family))
        bad = tmp/'badlabels.csv'; write_new(bad, labels[:-1])
        rejected(lambda: validate_labels(bad, states, case_family))
        baseline = tmp/'baseline.json'
        baseline.write_text(json.dumps(dict.fromkeys(SPEC['forecast_features'], .5)))
        validate_baseline(baseline)
        baseline.write_text(json.dumps(dict.fromkeys(SPEC['forecast_features'], 2)))
        rejected(lambda: validate_baseline(baseline))
        endpoint = PAIRS[0]['endpoint']
        for row in parsed:
            row[endpoint] = None
        primary = [dict(pair=p['id'], interaction_p_signflip='' if i else '.01',
                        interaction_p_holm_five='' if i else '.01') for i,p in enumerate(PAIRS)]
        write_new(tmp/'primary_pairs.csv', primary)
        original = (tmp/'primary_pairs.csv').read_bytes()
        supplement(tmp, parsed, case_family)
        assert (tmp/'primary_pairs.csv').read_bytes() == original
        out = read_csv(tmp/'corrected_fixed_five_holm.csv', {'corrected_fixed_five_holm'})
        assert float(out[0]['corrected_fixed_five_holm']) == .05
        families = read_csv(tmp/'primary_family_missingness_supplement.csv', {'pair', 'n_complete'})
        assert len(families) == 30 and all(r['n_complete'] == '0' for r in families if r['pair'] == PAIRS[0]['id'])
    print('Synthetic selftest passed: complete grid, headers, sessions, arms, probabilities, baseline, fixed-five Holm and all-family missingness. No study outcomes read.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--forecasts', type=Path)
    parser.add_argument('--baseline-rates', type=Path)
    parser.add_argument('--out', type=Path, help='Fresh output directory; runs frozen analyze.py after validation')
    parser.add_argument('--analysis', type=Path, help='Existing frozen output directory; add separate audit supplements')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest(); return
    if not args.labels or not args.forecasts:
        parser.error('--labels and --forecasts are required')
    if args.out and args.analysis:
        parser.error('Use --out or --analysis, not both')
    try:
        states, case_family = design()
        labels = validate_labels(args.labels, states, case_family)
        validate_forecasts(args.forecasts, case_family)
        if args.baseline_rates:
            validate_baseline(args.baseline_rates)
        if args.out:
            if args.out.exists():
                raise FileExistsError('Output directory must be fresh')
            command = [sys.executable, '-B', str(HERE/'analyze.py'), '--labels', str(args.labels),
                       '--forecasts', str(args.forecasts), '--out', str(args.out)]
            if args.baseline_rates:
                command.extend(['--baseline-rates', str(args.baseline_rates)])
            subprocess.run(command, check=True)
            supplement(args.out, labels, case_family)
        elif args.analysis:
            supplement(args.analysis, labels, case_family)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print('Guard passed: 408 label rows; five sessions and 2940 forecast rows. Original analysis tables are preserved.')


if __name__ == '__main__':
    main()
