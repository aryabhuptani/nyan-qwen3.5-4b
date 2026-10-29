#!/usr/bin/env python3
"""Analyze the reserved four-cell confirmation; standard library only.

python3 confirmation_analysis.py --selftest
python3 confirmation_analysis.py --labels locked.csv --selection selection.json --out analysis
Selection: {"pair_id": "paws_hugs"}; endpoints and cells come from the frozen spec.
See CONFIRMATION_ANALYSIS.md for denominators and interpretation.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import statistics
import tempfile

from analyze import SPEC, FEATURES, PAIRS, bitstate, val, signflip, quantile_sorted, write_csv

DRAWS = 5000
SEED = 113
CELL_KEYS = ('y00', 'y10', 'y01', 'y11')
DIFF_KEYS = ('interaction', 'pair_minus_a', 'pair_minus_b', 'pair_minus_base')


def mean(xs):
    return statistics.fmean(xs) if xs else None


def selection_pair(selection):
    if not isinstance(selection, dict) or set(selection) != {'pair_id'}:
        raise ValueError('Selection must be an object with exactly one key: pair_id')
    for pair in PAIRS:
        if pair['id'] == selection['pair_id']:
            return pair
    raise ValueError('pair_id must name one of the five canonical primary pairs')


def cells(pair):
    a, b = pair['modifier_indices']
    return [bitstate(s) for s in ([], [a], [b], [a, b])]


def read_labels(path):
    required = {'case_id', 'condition_id', 'family_id', *FEATURES}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Missing CSV columns: ' + ', '.join(sorted(required - set(reader.fieldnames or []))))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Duplicate CSV column names')
        rows = list(reader)
    for row in rows:
        if None in row or any(row[k] is None for k in required):
            raise ValueError('Malformed CSV row')
        for feature in FEATURES:
            row[feature] = val(row[feature])
    return rows


def validate_design(rows, pair):
    expected = set(cells(pair))
    seen, by_case = set(), {}
    for row in rows:
        case, state, family = (row[k] for k in ('case_id', 'condition_id', 'family_id'))
        if not all(isinstance(x, str) and x.strip() for x in (case, state, family)):
            raise ValueError('Case, condition and family IDs must be nonempty strings')
        if state not in expected:
            raise ValueError(f'Unexpected confirmation condition: {state}')
        if (case, state) in seen:
            raise ValueError(f'Duplicate condition/case: {state}, {case}')
        seen.add((case, state))
        by_case.setdefault(case, []).append(row)
        for feature in FEATURES:
            if feature not in row or (row[feature] is not None and row[feature] not in (0, 1)):
                raise ValueError(f'Missing or nonbinary feature: {feature}')
    if len(rows) != 48 or len(by_case) != 12:
        raise ValueError('Expected exactly 48 rows on 12 reserved confirmation cases')
    families = {}
    for case, rr in by_case.items():
        if {r['condition_id'] for r in rr} != expected:
            raise ValueError(f'Incomplete four-cell coverage: {case}')
        ff = {r['family_id'] for r in rr}
        if len(ff) != 1:
            raise ValueError(f'Inconsistent family for case: {case}')
        families.setdefault(next(iter(ff)), set()).add(case)
    if len(families) != 6 or any(len(cc) != 2 for cc in families.values()):
        raise ValueError('Expected six families of two cases')


def estimates(complete):
    result = {k: mean([r[k] for r in complete]) for k in CELL_KEYS + DIFF_KEYS}
    result['pair_minus_better_single'] = (
        result['y11'] - max(result['y10'], result['y01']) if complete else None)
    return result


def intervals(complete):
    keys = CELL_KEYS + DIFF_KEYS + ('pair_minus_better_single',)
    if not complete:
        return {k + suffix: None for k in keys for suffix in ('_ci_low', '_ci_high')}
    rng = random.Random(SEED)
    samples = {k: [] for k in keys}
    for _ in range(DRAWS):
        sample = [complete[rng.randrange(len(complete))] for _ in complete]
        for k, value in estimates(sample).items():
            samples[k].append(value)
    result = {}
    for k, values in samples.items():
        values.sort()
        result[k + '_ci_low'] = quantile_sorted(values, .025)
        result[k + '_ci_high'] = quantile_sorted(values, .975)
    return result


def analyze(rows, pair):
    validate_design(rows, pair)
    states = cells(pair)
    cases = sorted({r['case_id'] for r in rows})
    lookup = {(r['case_id'], r['condition_id']): r for r in rows}
    rates, summaries, details, families = [], [], [], []
    for state in states:
        rr = [r for r in rows if r['condition_id'] == state]
        for feature in FEATURES:
            scored = [r[feature] for r in rr if r[feature] is not None]
            coherent = [r for r in rr if r['coherent'] == 1]
            coherent_scored = [r[feature] for r in coherent if r[feature] is not None]
            rates.append(dict(condition_id=state, feature=feature, n_total=len(rr),
                              n_scored=len(scored), n_unknown=len(rr)-len(scored),
                              positive=sum(scored), rate=mean(scored),
                              coherent_only_n_total=len(coherent), coherent_only_n=len(coherent_scored),
                              coherent_only_n_unknown=len(coherent)-len(coherent_scored),
                              coherent_only_positive=sum(coherent_scored),
                              coherent_only_rate=mean(coherent_scored)))
    for feature in FEATURES:
        ff = []
        for case in cases:
            rr = [lookup[case, state] for state in states]
            ys = [r[feature] for r in rr]
            complete = all(y is not None for y in ys)
            d = dict(pair=pair['id'], feature=feature, is_primary_endpoint=feature == pair['endpoint'],
                     case_id=case, family_id=rr[0]['family_id'], complete=complete,
                     n_unknown_cells=sum(y is None for y in ys), **dict(zip(CELL_KEYS, ys)))
            d.update(dict.fromkeys(DIFF_KEYS))
            if complete:
                y00, y10, y01, y11 = ys
                d.update(interaction=y11-y10-y01+y00, pair_minus_a=y11-y10,
                         pair_minus_b=y11-y01, pair_minus_base=y11-y00)
            ff.append(d)
        details.extend(ff)
        complete = [r for r in ff if r['complete']]
        summary = dict(pair=pair['id'], feature=feature, is_primary_endpoint=feature == pair['endpoint'],
                       **dict(zip(('state00', 'state10', 'state01', 'state11'), states)),
                       n_total_cases=12, n_complete=len(complete), n_incomplete=12-len(complete),
                       n_unknown_cells=sum(r['n_unknown_cells'] for r in ff),
                       **estimates(complete), **intervals(complete),
                       interaction_p_signflip=signflip([r['interaction'] for r in complete]))
        summaries.append(summary)
        for family in sorted({r['family_id'] for r in ff}):
            all_family = [r for r in ff if r['family_id'] == family]
            cc = [r for r in all_family if r['complete']]
            families.append(dict(pair=pair['id'], feature=feature,
                                 is_primary_endpoint=feature == pair['endpoint'], family_id=family,
                                 n_total_cases=len(all_family), n_complete=len(cc),
                                 n_incomplete=len(all_family)-len(cc),
                                 n_unknown_cells=sum(r['n_unknown_cells'] for r in all_family),
                                 **estimates(cc)))
    return dict(state_rates=rates, four_cell_contrasts=summaries,
                case_contrasts=details, family_contrasts=families)


def selftest():
    pair = selection_pair({'pair_id': 'paws_hugs'})
    rows = []
    for i in range(12):
        for state in cells(pair):
            row = dict(case_id=f'c{i:02}', family_id=f'f{i//2}', condition_id=state,
                       **dict.fromkeys(FEATURES, 0))
            row['coherent'] = 1
            row[pair['endpoint']] = int(state == '11000')
            rows.append(row)
    out = analyze(rows, pair)
    target = next(r for r in out['four_cell_contrasts'] if r['is_primary_endpoint'])
    assert target['interaction'] == target['pair_minus_better_single'] == 1
    assert target['interaction_ci_low'] == target['interaction_ci_high'] == 1
    assert target['interaction_p_signflip'] == 2 / 4096
    assert len(out['state_rates']) == 56 and len(out['case_contrasts']) == 168
    assert len(out['family_contrasts']) == 84
    assert signflip([0]*12) == 1 and signflip([1, -1]) == 1
    unknown = copy.deepcopy(rows)
    unknown[0][pair['endpoint']] = None
    out = analyze(unknown, pair)
    target = next(r for r in out['four_cell_contrasts'] if r['is_primary_endpoint'])
    assert target['n_complete'] == 11 and target['n_unknown_cells'] == 1
    assert target['interaction_p_signflip'] == 2 / 2048
    rate = next(r for r in out['state_rates'] if r['feature'] == pair['endpoint'] and r['condition_id'] == '00000')
    assert rate['n_unknown'] == 1 and rate['n_scored'] == 11
    for row in unknown:
        row[pair['endpoint']] = None
    out = analyze(unknown, pair)
    target = next(r for r in out['four_cell_contrasts'] if r['is_primary_endpoint'])
    assert target['interaction'] is None and target['interaction_p_signflip'] is None
    assert len([r for r in out['family_contrasts'] if r['is_primary_endpoint']]) == 6
    for bad in (rows[:-1], rows + rows[:1]):
        try:
            validate_design(bad, pair)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid coverage accepted')
    bad = copy.deepcopy(rows)
    bad[0]['family_id'] = 'wrong'
    try:
        validate_design(bad, pair)
    except ValueError:
        pass
    else:
        raise AssertionError('Inconsistent family accepted')
    for bad in ({'pair_id': 'unknown'}, {'pair_id': 'paws_hugs', 'endpoint': 'coherent'}):
        try:
            selection_pair(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid selection accepted')
    # Exercise CSV parsing and round-trip unknowns without using any study output.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'synthetic.csv'
        write_csv(path, unknown)
        assert read_labels(path)[0][pair['endpoint']] is None
        path.write_text('case_id,condition_id,family_id\nc,00000,f\n')
        try:
            read_labels(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing features accepted')
    assert intervals([dict(y00=0,y10=1,y01=0,y11=1,interaction=0,pair_minus_a=0,pair_minus_b=1,pair_minus_base=1)])['pair_minus_better_single_ci_low'] == 0
    print('Selftest passed: four-cell effects, paired intervals, exact tests, unknowns, families, selection, coverage and CSV guards.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not all((args.labels, args.selection, args.out)):
        parser.error('--labels, --selection and --out are required')
    try:
        selection = json.loads(args.selection.read_text())
        pair = selection_pair(selection)
        rows = read_labels(args.labels)
        tables = analyze(rows, pair)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    # Require a fresh destination so earlier analyses cannot be silently overwritten.
    args.out.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        write_csv(args.out / (name + '.csv'), table)
    summary = dict(selection=selection, canonical_primary_pair=pair,
                   label_rows=len(rows), features=FEATURES, bootstrap_draws=DRAWS, bootstrap_seed=SEED,
                   tables={name: len(table) for name, table in tables.items()},
                   inference='One reserved canonical primary endpoint; other feature tests are descriptive and unadjusted.',
                   caveats=['Paired complete-case feature analyses omit cases with any unknown cell; see denominators.',
                            'Coherent-only rates are descriptive selected subsets, not replacements for all-reply results.',
                            'Exact two-sided sign flips assume case-level sign exchangeability.',
                            'Convenience prompts limit generalization; zero-width intervals do not establish general absence.',
                            'No selection or confirmation gate is evaluated; analyze original reviewer labels separately.'])
    (args.out / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
