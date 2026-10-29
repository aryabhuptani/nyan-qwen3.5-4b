"""Independently recompute primary point estimates; never imports analyze.py.

Requires an immutable label lock for real data. This checks arithmetic and data
joins, not the correctness of judgments, prompt representativeness, or causality.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csvrows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def binary(value):
    if value is None or str(value).strip().lower() in ('', 'null', 'none', 'nan', 'uncertain'):
        return None
    if value in (0, '0', '0.0', 'False'):
        return 0
    if value in (1, '1', '1.0', 'True'):
        return 1
    raise ValueError('Invalid binary judgment')


def average(values):
    return sum(values) / len(values) if values else None


def recompute(raw_rows, spec, forecasts=None, base_rates=None):
    features = spec['feature_order']
    lookup = {}
    for raw in raw_rows:
        row = dict(raw)
        for feature in features:
            row[feature] = binary(row.get(feature))
        key = row['condition_id'], row['case_id']
        if key in lookup:
            raise ValueError('Duplicate observation')
        lookup[key] = row
    rows = list(lookup.values())
    states = sorted({r['condition_id'] for r in rows})
    rates = []
    for state in states:
        selected = [r for r in rows if r['condition_id'] == state]
        for feature in features:
            values = [r[feature] for r in selected if r[feature] is not None]
            readable = [r[feature] for r in selected if r['coherent'] == 1 and r[feature] is not None]
            rates.append(dict(condition_id=state, feature=feature, positive=sum(values),
                              n_scored=len(values), n_total=len(selected), rate=average(values),
                              coherent_only_positive=sum(readable), coherent_only_n=len(readable),
                              coherent_only_rate=average(readable)))
    cases = sorted({case for state, case in lookup if len(state) == 5 and set(state) <= {'0', '1'}})
    pairs = []
    case_contrasts = []
    for pair in spec['primary_pairs']:
        a, b = pair['modifier_indices']
        conditions = []
        for active in ((), (a,), (b,), (a, b)):
            conditions.append(''.join(str(int(i in active)) for i in range(5)))
        complete = []
        families = set()
        for case in cases:
            cells = [lookup.get((condition, case)) for condition in conditions]
            if any(r is None or r[pair['endpoint']] is None for r in cells):
                continue
            y00, y10, y01, y11 = [r[pair['endpoint']] for r in cells]
            effects = dict(y00=y00, y10=y10, y01=y01, y11=y11,
                           interaction=y11-y10-y01+y00, pair_minus_a=y11-y10,
                           pair_minus_b=y11-y01, pair_minus_base=y11-y00)
            complete.append(effects)
            if y11:
                families.add(cells[0]['family_id'])
            case_contrasts.append(dict(pair=pair['id'], feature=pair['endpoint'], case_id=case,
                                       family_id=cells[0]['family_id'], **effects))
        result = dict(pair=pair['id'], feature=pair['endpoint'], n_complete=len(complete),
                      n_total_cases=len(cases), positive_pair_families=len(families),
                      **{name: conditions[i] for i, name in enumerate(('state00', 'state10', 'state01', 'state11'))})
        for key in ('y00', 'y10', 'y01', 'y11', 'interaction', 'pair_minus_a', 'pair_minus_b', 'pair_minus_base'):
            result[key] = average([r[key] for r in complete])
        result['pair_minus_better_single'] = (result['y11'] - max(result['y10'], result['y01'])) if complete else None
        pairs.append(result)
    output = dict(state_rates=rates, primary_pairs=pairs, primary_case_contrasts=case_contrasts)
    if forecasts is not None:
        losses = {}
        seen = set()
        case_scores = []
        for forecast in forecasts:
            session, arm, feature = forecast['session_id'], forecast['arm'], forecast['feature']
            family, state = forecast['family_id'], forecast['condition_id']
            unique = session, family, state, feature
            if unique in seen:
                raise ValueError('Duplicate forecast')
            seen.add(unique)
            probability = float(forecast.get('probability', forecast.get('p')))
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid probability')
            selected = [r for r in rows if r['condition_id'] == state and r['family_id'] == family]
            if not selected:
                raise ValueError('Forecast without observations')
            for row in selected:
                outcome = row[feature]
                if outcome is None:
                    continue
                score = dict(session_id=session, arm=arm, feature=feature, family_id=family,
                             condition_id=state, case_id=row['case_id'], probability=probability,
                             outcome=outcome, brier=(probability-outcome)**2)
                if base_rates is not None:
                    score.update(discovery_base_probability=float(base_rates[feature]),
                                 discovery_base_brier=(float(base_rates[feature])-outcome)**2)
                case_scores.append(score)
                losses.setdefault((session, arm, feature), []).append(score)
        session_scores = []
        arm_losses = {}
        for (session, arm, feature), values in sorted(losses.items()):
            score = average([r['brier'] for r in values])
            session_scores.append(dict(session_id=session, arm=arm, feature=feature,
                                       n_case_scores=len(values), brier=score,
                                       discovery_base_brier=average([r['discovery_base_brier'] for r in values])
                                       if base_rates is not None else None))
            arm_losses.setdefault((arm, feature), []).append(score)
        output.update(forecast_case_scores=case_scores, forecast_session_scores=session_scores,
                      forecast_arm_scores=[dict(arm=arm, feature=feature, n_sessions=len(values),
                                                mean_session_brier=average(values))
                                           for (arm, feature), values in sorted(arm_losses.items())])
    return output


KEYS = {'state_rates': ('condition_id', 'feature'), 'primary_pairs': ('pair',),
        'primary_case_contrasts': ('pair', 'case_id'),
        'forecast_case_scores': ('session_id', 'family_id', 'condition_id', 'case_id', 'feature'),
        'forecast_session_scores': ('session_id', 'feature'), 'forecast_arm_scores': ('arm', 'feature')}


def compare(independent, primary_dir):
    findings = []
    for name, rows in independent.items():
        primary = csvrows(Path(primary_dir) / (name + '.csv'))
        keys = KEYS[name]
        index = {tuple(row[k] for k in keys): row for row in primary}
        if len(index) != len(primary) or len(primary) != len(rows):
            raise ValueError('Primary row count/duplicates: ' + name)
        checked = 0
        for row in rows:
            key = tuple(str(row[k]) for k in keys)
            actual = index[key]
            for field, value in row.items():
                candidate = actual[field]
                if value is None:
                    okay = candidate == ''
                elif isinstance(value, (int, float)):
                    okay = candidate != '' and math.isclose(float(candidate), value, rel_tol=1e-12, abs_tol=1e-12)
                else:
                    okay = str(value) == candidate
                if not okay:
                    raise ValueError(f'Mismatch {name}/{key}/{field}: independent={value!r}, primary={candidate!r}')
                checked += 1
        findings.append(dict(table=name, rows=len(rows), fields_checked=checked, match=True))
    return findings


def selftest():
    spec = json.loads((HERE / 'FEATURE_DEFINITIONS.json').read_text())
    rows = []
    for i in range(12):
        for bits in itertools.product((0, 1), repeat=5):
            row = dict(case_id=str(i), family_id=str(i//2), condition_id=''.join(map(str, bits)))
            row.update({feature: 0 for feature in spec['feature_order']})
            row['coherent'] = 1
            row['integrated_embodied_affection'] = bits[0] * bits[1]
            rows.append(row)
    forecasts = [dict(session_id='s', arm='a', family_id='0', condition_id='11000',
                      feature='integrated_embodied_affection', probability=.75)]
    out = recompute(rows, spec, forecasts)
    assert out['primary_pairs'][0]['interaction'] == 1
    assert out['primary_pairs'][0]['pair_minus_better_single'] == 1
    assert out['forecast_session_scores'][0]['brier'] == .0625
    rows[0]['integrated_embodied_affection'] = None
    assert recompute(rows, spec)['primary_pairs'][0]['n_complete'] == 11
    try:
        recompute(rows + rows[:1], spec)
        raise AssertionError('Duplicate accepted')
    except ValueError:
        pass
    print('Independent synthetic audit passed: AND interaction, Brier, unknown-case exclusion, duplicates.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path)
    p.add_argument('--labels-lock', type=Path)
    p.add_argument('--forecasts', type=Path)
    p.add_argument('--baseline-rates', type=Path)
    p.add_argument('--primary-dir', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if not all([a.labels, a.labels_lock, a.primary_dir, a.output]):
        p.error('--labels, --labels-lock, --primary-dir, --output required')
    lock = json.loads(a.labels_lock.read_text())
    if lock['labels_sha256'].get(a.labels.name) != sha(a.labels):
        raise ValueError('Labels do not match immutable lock')
    spec = json.loads((HERE / 'FEATURE_DEFINITIONS.json').read_text())
    output = recompute(csvrows(a.labels), spec, csvrows(a.forecasts) if a.forecasts else None,
                       json.loads(a.baseline_rates.read_text()) if a.baseline_rates else None)
    findings = compare(output, a.primary_dir)
    result = dict(pass_all=True, label_sha256=sha(a.labels), labels_lock_sha256=sha(a.labels_lock),
                  forecast_sha256=sha(a.forecasts) if a.forecasts else None,
                  independent_code_sha256=sha(__file__), checks=findings,
                  limitation='Point-estimate arithmetic audit; primary bootstrap intervals and hypothesis-test assumptions are not independently validated here.')
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
