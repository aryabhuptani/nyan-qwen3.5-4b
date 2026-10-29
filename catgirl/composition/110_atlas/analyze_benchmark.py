#!/usr/bin/env python3
"""CPU-only frozen 110 analysis. Reads joined labels, never raw model outputs.

Usage: python3 analyze_benchmark.py --self-test
       python3 analyze_benchmark.py [--root PATH] [--labels labels.csv]
Mean separate-session loss is primary; probability ensembles are secondary.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ('atlas', 'shuffled', 'projection', 'prompts')
JOINT = 'joint_catgirl_coherent'


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def unit_mean(rows, field='loss'):
    """Equal intervention-unit weight even if per-unit row counts differ."""
    units = defaultdict(list)
    for r in rows:
        units[r['unit']].append(r[field])
    return mean(mean(v) for v in units.values())


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - x) + values[hi] * (x - lo) if hi != lo else values[lo]


def bootstrap(values, draws=10000, seed=110):
    """One value per independent coordinate; resample entire coordinate blocks."""
    rng = random.Random(seed)
    n = len(values)
    samples = [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws)]
    return dict(mean=mean(values), lower95=quantile(samples, .025),
                upper95=quantile(samples, .975), clusters=n, draws=draws, seed=seed)


def metadata(interventions):
    components = interventions['components']
    result = {}
    canonical = {}
    for c in interventions['conditions']:
        ids = c['component_ids']
        if c['kind'] == 'single':
            component = components[ids[0]]
            stratum = ('single_random' if component['stratum'] == 'random_previously_untested'
                       else 'single_anchor')
            sign = 'canonical' if c['directions'] == [1] else 'opposite'
            unit = ids[0]
            if sign == 'canonical':
                canonical[unit] = c['id']
        else:
            stratum = c['kind']
            sign = 'canonical' if c['kind'] == 'pair' else 'baseline'
            unit = c['id']
        anchors = [i for i in ids if components[i]['stratum'] == 'known_cat_related_anchor']
        result[c['id']] = dict(stratum=stratum, sign=sign, unit=unit,
                                components=ids, anchors=anchors)
    return result, canonical


def load_inputs(root, label_path, allow_uncertain=False):
    plan = json.loads((root / '110_PLAN.json').read_text())
    features = plan['primary_features'] + [JOINT]
    interventions = json.loads((root / 'interventions.json').read_text())
    meta, canonical = metadata(interventions)
    for name, key in [('interventions.json', 'interventions_sha256'), ('selected_prompts.json', 'prompts_sha256')]:
        if key in plan and sha(root / name) != plan[key]:
            raise ValueError('Frozen input hash mismatch: ' + name)
    forecast_path = root / 'locked_forecasts.csv'
    lock = json.loads((root / 'forecast_lock.json').read_text())
    if lock['plan_sha256'] != sha(root/'110_PLAN.json') or lock['compact_forecasts_sha256'] != sha(forecast_path):
        raise ValueError('Forecast lock hash mismatch')
    forecasts = {}
    for r in read_csv(forecast_path):
        key = (r['arm'], str(r['replicate']), r['condition_id'], r['family'], r['feature'])
        p = float(r['probability'])
        if key in forecasts or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Duplicate or invalid forecast: ' + str(key))
        forecasts[key] = p
    expected = {(a, str(rep), c, fam, f) for a in ARMS for rep in (1, 2)
                for c in meta for fam in plan['families'] for f in features}
    if set(forecasts) != expected:
        raise ValueError('Forecast grid incomplete or contains unexpected entries')
    selected = json.loads((root/'selected_prompts.json').read_text())['prompts']
    expected_prompt_families = {r['id']:r['family'] for r in selected}
    if len(expected_prompt_families) != len(selected):
        raise ValueError('Duplicate frozen prompt IDs')
    labels = read_csv(label_path)
    if len({r['review_id'] for r in labels}) != len(labels):
        raise ValueError('Missing or duplicate review_id provenance')
    seen = set()
    prompt_families = {}
    condition_prompts = defaultdict(set)
    for r in labels:
        key = (r['condition_id'], r['prompt_id'])
        if key in seen or r['condition_id'] not in meta or r['family'] not in plan['families']:
            raise ValueError('Duplicate/unrecognized label: ' + str(key))
        seen.add(key)
        condition_prompts[r['condition_id']].add(r['prompt_id'])
        previous = prompt_families.setdefault(r['prompt_id'], r['family'])
        if previous != r['family']:
            raise ValueError('Prompt family mismatch')
        for f in features:
            value = str(r[f]).strip().lower()
            if allow_uncertain and value in ('uncertain', '?', '', 'na', 'null'):
                r[f] = None
            elif value in ('0', '1'):
                r[f] = int(value)
            else:
                raise ValueError('Binary label required (uncertain allowed only for sensitivity): ' + str((key, f)))
        parts = [r[features[0]], r[features[1]], r['coherent']]
        joint = 0 if 0 in parts else (1 if all(v == 1 for v in parts) else None)
        if r[JOINT] != joint:
            raise ValueError('Joint endpoint must be feline AND female AND coherent')
        if 'label_uncertflag' not in r:
            raise ValueError('Missing label_uncertflag provenance column')
    if len(labels) != plan['expected_outputs'] or set(condition_prompts) != set(meta):
        raise ValueError('Wrong joined label count/condition coverage')
    if len(prompt_families) != 8 or Counter(prompt_families.values()) != Counter({f: 2 for f in plan['families']}):
        raise ValueError('Expected two fixed prompts per family')
    if prompt_families != expected_prompt_families:
        raise ValueError('Joined labels do not match frozen selected prompt IDs/families')
    if any(v != set(prompt_families) for v in condition_prompts.values()):
        raise ValueError('Every condition must share all eight exact prompt IDs')
    return plan, features, meta, canonical, forecasts, labels


def make_rows(features, meta, canonical, forecasts, labels):
    rows = []
    for label in labels:
        c, family = label['condition_id'], label['family']
        m = meta[c]
        for arm in ARMS:
            for rep in ('1', '2', 'mean_sessions', 'ensemble'):
                for feature in features:
                    if label[feature] is None:
                        continue
                    ps = [forecasts[(arm, str(i), c, family, feature)] for i in (1, 2)]
                    p = mean(ps) if rep in ('ensemble', 'mean_sessions') else ps[int(rep) - 1]
                    y = label[feature]
                    loss = mean((v-y)**2 for v in ps) if rep == 'mean_sessions' else (p-y)**2
                    row = dict(arm=arm, replicate=rep, condition_id=c, prompt_id=label['prompt_id'],
                               family=family, feature=feature, stratum=m['stratum'], sign=m['sign'],
                               unit=m['unit'], y=y, p=p, loss=loss)
                    rows.append(row)
    return rows


def grouped_scores(rows, primary_features):
    groups = defaultdict(list)
    # Separate marginal dimensions, not a bulky all-dimensions Cartesian table.
    for r in rows:
        feature_groups = [r['feature']]
        if r['feature'] in primary_features:
            feature_groups.append('macro_primary7')
        for f in feature_groups:
            for dimension, value in [('overall', 'all'), ('family', r['family']), ('sign', r['sign'])]:
                groups[(r['arm'], r['replicate'], r['stratum'], f, dimension, value)].append(r)
    result = []
    for key, group in sorted(groups.items()):
        positive = [r for r in group if r['y'] == 1]
        negative = [r for r in group if r['y'] == 0]
        result.append(dict(zip(('arm', 'replicate', 'stratum', 'feature', 'dimension', 'value'), key)) |
                      dict(brier=unit_mean(group), units=len({r['unit'] for r in group}),
                           observations=len(group), positives=len(positive), negatives=len(negative),
                           prevalence=unit_mean(group, 'y'), allzero_brier=unit_mean(group, 'y'),
                           allone_brier=1-unit_mean(group, 'y'),
                           positive_brier=mean(r['loss'] for r in positive), negative_brier=mean(r['loss'] for r in negative)))
    return result


def unit_differences(rows, primary_features):
    grouped = defaultdict(list)
    for r in rows:
        if r['feature'] in primary_features:
            grouped[(r['arm'], r['replicate'], r['stratum'], r['unit'])].append(r['loss'])
    means = {k: mean(v) for k, v in grouped.items()}
    out = []
    for (arm, rep, stratum, unit), value in sorted(means.items()):
        if arm == 'atlas':
            for control in ARMS[1:]:
                out.append(dict(control=control, replicate=rep, stratum=stratum, unit=unit,
                                atlas_brier=value, control_brier=means[(control, rep, stratum, unit)],
                                difference=value-means[(control, rep, stratum, unit)]))
    return [r for r in out if r['replicate'] != 'mean_sessions']


def composition_scores(labels, features, meta, canonical, forecasts):
    groups = defaultdict(list)
    pairunits = defaultdict(list)
    for label in labels:
        c, fam = label['condition_id'], label['family']
        if meta[c]['stratum'] != 'pair':
            continue
        a, b = [canonical[i] for i in meta[c]['components']]
        for arm in ARMS:
            for rep in ('1', '2'):
                for f in features:
                    def get(cond):
                        return forecasts[(arm, rep, cond, fam, f)]
                    direct = get(c)
                    mechanical = min(1., max(0., get(a)+get(b)-get('baseline')))
                    y = label[f]
                    values = ((direct-y)**2, (mechanical-y)**2)
                    groups[(arm, rep, f)].append(values)
                    if f != JOINT:
                        pairunits[(arm, rep, c)].append(values[0]-values[1])
    result = []
    for (arm, rep, f), values in sorted(groups.items()):
        d, s = mean(v[0] for v in values), mean(v[1] for v in values)
        result.append(dict(arm=arm, replicate=rep, feature=f, direct_brier=d,
                           singles_only_brier=s, direct_minus_singles=d-s))
    pairrows = [dict(arm=a, replicate=r, pair=c, difference=mean(v))
                for (a, r, c), v in sorted(pairunits.items())]
    return result, pairrows


def summarize(differences, meta, draws):
    results = []
    for control in ARMS[1:]:
        for stratum in ('single_random', 'single_anchor', 'pair', 'baseline'):
            for rep in ('mean_sessions', '1', '2', 'ensemble'):
                group = [r for r in differences if r['control'] == control and r['stratum'] == stratum
                         and (r['replicate'] in ('1', '2') if rep == 'mean_sessions' else r['replicate'] == rep)]
                units = defaultdict(list)
                for r in group:
                    units[r['unit']].append(r['difference'])
                values = {u: mean(v) for u, v in units.items()}
                rec = dict(control=control, stratum=stratum, replicate=rep, difference=mean(values.values()),
                           units=len(values), units_benefiting=sum(v < 0 for v in values.values()),
                           engineering_reduction_at_least_001=mean(values.values()) <= -.01)
                if stratum == 'single_random':
                    rec['paired_coordinate_bootstrap'] = bootstrap(list(values.values()), draws)
                if stratum == 'pair':
                    anchors = sorted({a for m in meta.values() for a in m['anchors']})
                    rec['leave_anchor_out'] = {a: mean(v for u, v in values.items() if a not in meta[u]['anchors'])
                                              for a in anchors}
                results.append(rec)
    return results


def integration_self_test():
    """Full synthetic grid, including frozen-ID rejection and uncertain reviewer coverage."""
    with tempfile.TemporaryDirectory(prefix='atlas110-analyzer-test-') as tmp:
        root = Path(tmp)
        components = {f'N{i:02}':dict(stratum='random_previously_untested' if i < 24 else 'known_cat_related_anchor') for i in range(30)}
        conditions = [dict(id='baseline', kind='baseline', component_ids=[], directions=[])]
        for component in components:
            for sign, direction in [('canonical',1), ('opposite',-1)]:
                conditions.append(dict(id=f'{component}_{sign}', kind='single', component_ids=[component], directions=[direction]))
        for i in range(12):
            conditions.append(dict(id=f'P{i:02}', kind='pair', component_ids=[f'N{24+i%6:02}', f'N{i:02}'], directions=[1,1]))
        features = ['feline', 'female', 'vocalization', 'cute', 'affection', 'coherent', 'relevant']
        allfeatures = features + [JOINT]
        families = [f'family{i}' for i in range(4)]
        prompts = [dict(id=f'{fam}_{i}', family=fam) for fam in families for i in range(2)]
        (root/'interventions.json').write_text(json.dumps(dict(components=components, conditions=conditions)))
        (root/'selected_prompts.json').write_text(json.dumps(dict(prompts=prompts)))
        plan = dict(primary_features=features, families=families, expected_outputs=584,
                    interventions_sha256=sha(root/'interventions.json'), prompts_sha256=sha(root/'selected_prompts.json'))
        (root/'110_PLAN.json').write_text(json.dumps(plan))
        forecasts = [dict(arm=a, replicate=r, condition_id=c['id'], family=fam, feature=f,
                          probability=.25 if r==1 else .75) for a in ARMS for r in (1,2)
                     for c in conditions for fam in families for f in allfeatures]
        write_csv(root/'locked_forecasts.csv', forecasts)
        (root/'forecast_lock.json').write_text(json.dumps(dict(plan_sha256=sha(root/'110_PLAN.json'),
                                      compact_forecasts_sha256=sha(root/'locked_forecasts.csv'))))
        labels = [dict(review_id=str(i*8+j), condition_id=c['id'], prompt_id=p['id'], family=p['family'],
                       **{f:0 for f in allfeatures}, label_uncertflag=0)
                  for i,c in enumerate(conditions) for j,p in enumerate(prompts)]
        write_csv(root/'labels.csv', labels)
        plan, ff, meta, canonical, predictions, ll = load_inputs(root, root/'labels.csv')
        assert len(predictions)==18688 and len(ll)==584
        rr = make_rows(ff, meta, canonical, predictions, ll)
        scores = grouped_scores(rr, features)
        assert all(r['brier']==.3125 for r in scores if r['replicate']=='mean_sessions')
        assert all(r['brier']==.25 for r in scores if r['replicate']=='ensemble')
        differences = unit_differences(rr, features)
        assert all(r['difference']==0 for r in differences)
        labels[0]['cute']='uncertain'
        write_csv(root/'reviewer.csv', labels)
        _, _, _, _, _, reviewer = load_inputs(root, root/'reviewer.csv', allow_uncertain=True)
        assert sum(r['cute'] is not None for r in reviewer)==583
        assert reviewer[0][JOINT]==0  # Other definite-negative conjuncts determine the joint.
        # Eight invented but internally consistent IDs must fail the frozen identity check.
        for r in labels:
            r['prompt_id']='wrong_'+r['prompt_id']
        write_csv(root/'wrong.csv', labels)
        try:
            load_inputs(root, root/'wrong.csv', allow_uncertain=True)
        except ValueError as exc:
            assert 'frozen selected prompt' in str(exc)
        else:
            raise AssertionError('Unfrozen prompt IDs accepted')


def self_test():
    # Weight units equally rather than letting an overrepresented unit dominate.
    assert unit_mean([dict(unit='a', loss=0), dict(unit='a', loss=0), dict(unit='b', loss=1)]) == .5
    # Ensemble loss is not mean separate-session loss.
    assert mean([(0-1)**2, (1-1)**2]) == .5 and (.5-1)**2 == .25
    b = bootstrap([.2]*24, 100)
    assert abs(b['mean']-.2) < 1e-12 and abs(b['lower95']-.2) < 1e-12
    # Pair aligns to canonical components by explicit IDs, not list order or reverse sign.
    meta = {'p': dict(stratum='pair', components=['B', 'A'])}
    forecasts = {}
    for arm in ARMS:
        for rep in ('1', '2'):
            for c, p in [('a_plus', .6), ('b_plus', .5), ('baseline', .2), ('p', .9), ('a_minus', 0.)]:
                forecasts[(arm, rep, c, 'f', 'feature')] = p
    result, _ = composition_scores([dict(condition_id='p', family='f', feature=1)],
                                  ['feature'], meta, {'A':'a_plus', 'B':'b_plus'}, forecasts)
    assert all(abs(r['direct_minus_singles']) < 1e-12 for r in result)
    # Full score rows distinguish primary mean-session loss from ensemble loss.
    simple_meta = {'x':dict(stratum='single_random', sign='canonical', unit='A')}
    ff = {(a, str(rep), 'x', 'f', 'feature'): float(rep-1) for a in ARMS for rep in (1,2)}
    rr = make_rows(['feature'], simple_meta, {}, ff, [dict(condition_id='x', family='f', prompt_id='z', feature=1)])
    assert next(r for r in rr if r['replicate']=='mean_sessions')['loss'] == .5
    assert next(r for r in rr if r['replicate']=='ensemble')['loss'] == .25
    # Class-conditional loss pools positive observations, not equal weight per positive-containing neuron.
    diagnostic_rows = [dict(arm='atlas', replicate='1', stratum='single_random', feature='f',
       family='family', sign='canonical', unit=u, loss=loss, y=y)
       for u, loss, y in [('a',0.,1), ('a',0.,0), ('a',0.,0), ('a',0.,0),
                          ('b',1.,1), ('b',1.,1), ('b',1.,1), ('b',0.,0)]]
    diagnostic = next(r for r in grouped_scores(diagnostic_rows, ['f']) if r['feature']=='f' and r['dimension']=='overall')
    assert diagnostic['positive_brier'] == .75
    assert diagnostic['brier'] == .375
    # Shared anchors: removing A removes both pairs that contain A, including A+B.
    m = {'p1':dict(anchors=['A','B']), 'p2':dict(anchors=['A']), 'p3':dict(anchors=['B'])}
    diffs = [dict(control=c, stratum=s, replicate=r, unit=u, difference=v)
             for c in ARMS[1:] for s, units in [('single_random', {'n':.1}),
              ('single_anchor', {'a':.1}), ('baseline', {'baseline':.1}),
              ('pair', {'p1':1., 'p2':2., 'p3':3.})] for r in ('1','2','ensemble') for u,v in units.items()]
    summaries = summarize(diffs, m, 10)
    pair = next(r for r in summaries if r['stratum']=='pair' and r['replicate']=='mean_sessions')
    assert pair['leave_anchor_out'] == {'A':3., 'B':2.}
    integration_self_test()
    print('Self-tests passed: full synthetic grid and frozen prompt IDs, uncertain sensitivity, class diagnostics, unit weighting, session/ensemble distinction, block bootstrap, canonical pair alignment, shared-anchor exclusion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--sensitivity-labels', action='append', default=[], metavar='NAME=PATH',
                        help='Optional joined reviewer CSV; uncertain entries excluded feature by feature; never changes primary scores')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.bootstrap_draws != 10000:
        parser.error('Frozen benchmark requires 10000 bootstrap draws; alternate draws are self-test only')
    root = args.root
    label_path = args.labels or root/'labels.csv'
    plan, features, meta, canonical, forecasts, labels = load_inputs(root, label_path)
    rows = make_rows(features, meta, canonical, forecasts, labels)
    scores = grouped_scores(rows, plan['primary_features'])
    differences = unit_differences(rows, plan['primary_features'])
    composition, pairrows = composition_scores(labels, features, meta, canonical, forecasts)
    # Compact aggregate composition includes mean separate-session losses, never ensemble substitution.
    compmacro = []
    for arm in ARMS:
        selected = [r for r in composition if r['arm']==arm and r['feature'] in plan['primary_features']]
        compmacro.append(dict(arm=arm, direct_brier=mean(r['direct_brier'] for r in selected),
                              singles_only_brier=mean(r['singles_only_brier'] for r in selected),
                              direct_minus_singles=mean(r['direct_minus_singles'] for r in selected)))
    summary = dict(protocol='110_PLAN.json', label_rows=len(labels), forecast_entries=len(forecasts),
                   primary='mean separate-session loss; seven features; 24 random coordinates',
                   differences=summarize(differences, meta, args.bootstrap_draws),
                   composition_macro=compmacro, composition_by_feature=composition,
                   composition_pair_differences=pairrows,
                   initial_uncertainty_flags=Counter(str(r['label_uncertflag']) for r in labels),
                   input_sha256={'plan':sha(root/'110_PLAN.json'), 'interventions':sha(root/'interventions.json'),
                                 'forecasts':sha(root/'locked_forecasts.csv'), 'prompts':sha(root/'selected_prompts.json'), 'labels':sha(label_path)},
                   cautions=['Intervals condition on fixed prompts and one forecaster model.',
                             'Pairs share anchors; leave-anchor-out sensitivity is descriptive, not independent-pair inference.',
                             'Conditional positive/negative loss is diagnostic; balanced reweighting is not primary.',
                             'Uncertainty flags preserve pre-adjudication ambiguity; joined binary labels are primary.',
                             'No natural neuron cleanliness or interaction claim.'])
    write_csv(root/'scores.csv', scores)
    write_csv(root/'perunit_differences.csv', differences)
    sensitivity = []
    primary_keys = {(r['condition_id'], r['prompt_id'], r['family']) for r in labels}
    names = set()
    for specification in args.sensitivity_labels:
        if '=' not in specification:
            parser.error('--sensitivity-labels requires NAME=PATH')
        name, path = specification.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            parser.error('Sensitivity names must be unique letters/numbers/underscore/hyphen')
        names.add(name)
        _, _, smeta, scanonical, sf, slabels = load_inputs(root, Path(path), allow_uncertain=True)
        if {(r['condition_id'], r['prompt_id'], r['family']) for r in slabels} != primary_keys:
            raise ValueError('Reviewer sensitivity and primary label keys must match')
        sr = make_rows(features, smeta, scanonical, sf, slabels)
        ss = grouped_scores(sr, plan['primary_features'])
        write_csv(root/f'sensitivity_{name}_scores.csv', ss)
        coverage = {f:dict(confident=sum(r[f] is not None for r in slabels), total=len(slabels)) for f in features}
        overall = [r for r in ss if r['replicate']=='mean_sessions' and r['dimension']=='overall']
        sensitivity.append(dict(name=name, labels_sha256=sha(Path(path)), coverage=coverage,
                                scores=overall, rule='Confident labels only, feature-by-feature; diagnostic; no primary imputation or replacement. Macro weights reflect available confident observations and can differ from complete seven-feature primary grid.'))
    summary['reviewer_sensitivity'] = sensitivity
    (root/'compact_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(label_rows=len(labels), forecast_entries=len(forecasts), score_rows=len(scores),
                          outputs=['compact_summary.json','scores.csv','perunit_differences.csv'])))


if __name__ == '__main__':
    main()
