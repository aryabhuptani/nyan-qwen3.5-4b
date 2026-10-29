#!/usr/bin/env python3
"""CPU-only paired composition analysis. Inputs are locked case-level labels, not model calls."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'FEATURE_DEFINITIONS.json').read_text())
FEATURES = SPEC['feature_order']
PAIRS = SPEC['primary_pairs']

def mean(xs):
    return statistics.fmean(xs) if xs else None

def bitstate(indices):
    return ''.join('1' if i in indices else '0' for i in range(5))

def val(x):
    if x is None or str(x).strip().lower() in ('', 'null', 'none', 'nan', 'uncertain'):
        return None
    if str(x) in ('0', '0.0', 'False'):
        return 0
    if str(x) in ('1', '1.0', 'True'):
        return 1
    raise ValueError(f'Nonbinary label {x!r}')

def read_labels(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for row in rows:
        key = row['condition_id'], row['case_id']
        if key in seen:
            raise ValueError(f'Duplicate condition/case {key}')
        seen.add(key)
        row.setdefault('family_id', row.get('family', ''))
        for feature in FEATURES:
            row[feature] = val(row.get(feature))
    return rows

def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def quantile_sorted(xs, q):
    k = (len(xs) - 1) * q
    a = int(k)
    return xs[a] + (xs[min(a + 1, len(xs) - 1)] - xs[a]) * (k - a)

def ci(xs, seed=113):
    if not xs:
        return None, None
    rng = random.Random(seed)
    samples = sorted(mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(5000))
    return quantile_sorted(samples, .025), quantile_sorted(samples, .975)

def signflip(xs):
    if not xs:
        return None
    observed = abs(sum(xs))
    # n=12 in the frozen design; exact enumeration, including zero differences.
    if len(xs) > 20:
        raise ValueError('Exact sign-flip implementation expects at most20 cases')
    return sum(abs(sum(s*x for s, x in zip(signs, xs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(xs))) / 2**len(xs)

def holm(ps):
    out = [None] * len(ps)
    good = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    prev = 0
    for rank, (p, i) in enumerate(good):
        prev = max(prev, min(1, p * (len(good) - rank)))
        out[i] = prev
    return out

def analyze(rows):
    lookup = {(r['condition_id'], r['case_id']): r for r in rows}
    cases = sorted({r['case_id'] for r in rows if re.fullmatch('[01]{5}', r['condition_id'])})
    rates, contrasts, details, background = [], [], [], []
    for condition in sorted({r['condition_id'] for r in rows}):
        selected = [r for r in rows if r['condition_id'] == condition]
        for feature in FEATURES:
            xs = [r[feature] for r in selected if r[feature] is not None]
            coh = [r[feature] for r in selected if r['coherent'] == 1 and r[feature] is not None]
            rates.append({'condition_id': condition, 'feature': feature, 'positive': sum(xs),
                          'n_scored': len(xs), 'n_total': len(selected), 'rate': mean(xs),
                          'coherent_only_positive': sum(coh), 'coherent_only_n': len(coh),
                          'coherent_only_rate': mean(coh)})
    for pair in PAIRS:
        a, b = pair['modifier_indices']
        feature = pair['endpoint']
        states = [bitstate(s) for s in ([], [a], [b], [a,b])]
        valid = []
        for case in cases:
            rr = [lookup.get((state, case)) for state in states]
            if any(r is None or r[feature] is None for r in rr):
                continue
            yy = [r[feature] for r in rr]
            family = rr[0]['family_id']
            d = {'pair': pair['id'], 'feature': feature, 'case_id': case, 'family_id': family,
                 'y00': yy[0], 'y10': yy[1], 'y01': yy[2], 'y11': yy[3],
                 'interaction': yy[3]-yy[1]-yy[2]+yy[0],
                 'pair_minus_a': yy[3]-yy[1], 'pair_minus_b': yy[3]-yy[2],
                 'pair_minus_base': yy[3]-yy[0]}
            valid.append(d)
            details.append(d)
        result = {'pair': pair['id'], 'feature': feature, 'state00': states[0], 'state10': states[1],
                  'state01': states[2], 'state11': states[3], 'n_complete': len(valid), 'n_total_cases': len(cases)}
        for key in ('y00','y10','y01','y11','interaction','pair_minus_a','pair_minus_b','pair_minus_base'):
            xs = [d[key] for d in valid]
            result[key] = mean(xs)
            if not key.startswith('y'):
                result[key+'_ci_low'], result[key+'_ci_high'] = ci(xs)
        result['interaction_p_signflip'] = signflip([d['interaction'] for d in valid])
        result['pair_minus_better_single'] = None if not valid else result['y11'] - max(result['y10'], result['y01'])
        result['positive_pair_families'] = len({d['family_id'] for d in valid if d['y11'] == 1})
        result['practical_screen_without_reviewer_check'] = bool(
            len(valid) == len(cases) and valid and
            result['pair_minus_better_single'] >= 3/12 - 1e-12 and result['positive_pair_families'] >= 2)
        contrasts.append(result)
        others = [i for i in range(5) if i not in (a,b)]
        bg_case = []
        for case in cases:
            interactions = []
            for bg in itertools.product((0,1), repeat=3):
                active = {i for i, on in zip(others,bg) if on}
                ss = [bitstate(active | set(extra)) for extra in ([],[a],[b],[a,b])]
                rr = [lookup.get((state,case)) for state in ss]
                if any(r is None or r[feature] is None for r in rr):
                    continue
                ys = [r[feature] for r in rr]
                interactions.append(ys[3]-ys[1]-ys[2]+ys[0])
            # Require all8 backgrounds so missing labels cannot change background weighting.
            if len(interactions) == 8:
                bg_case.append(mean(interactions))
        lo,hi = ci(bg_case)
        background.append({'pair':pair['id'],'feature':feature,'n_complete_cases':len(bg_case),
                           'background_averaged_interaction':mean(bg_case),'ci_low':lo,'ci_high':hi})
    for r,p in zip(contrasts,holm([r['interaction_p_signflip'] for r in contrasts])):
        r['interaction_p_holm_five'] = p
    families=[]
    for pair in PAIRS:
        for family in sorted({d['family_id'] for d in details if d['pair']==pair['id']}):
            rr=[d for d in details if d['pair']==pair['id'] and d['family_id']==family]
            families.append({'pair':pair['id'],'family_id':family,'n_complete':len(rr),**{
                k:mean([d[k] for d in rr]) for k in ('y00','y10','y01','y11','interaction','pair_minus_a','pair_minus_b')}})
    larger=[]
    for state in ('11100','11110','11111'):
        for feature in FEATURES:
            targets=[r for r in rows if r['condition_id']==state]
            comparisons=['00000']+[state[:i]+'0'+state[i+1:] for i,b in enumerate(state) if b=='1']
            for other in comparisons:
                ds=[]
                for r in targets:
                    ref=lookup.get((other,r['case_id']))
                    if ref and r[feature] is not None and ref[feature] is not None:
                        ds.append(r[feature]-ref[feature])
                lo,hi=ci(ds)
                larger.append({'state':state,'comparison':other,'feature':feature,'n_complete':len(ds),
                               'paired_difference':mean(ds),'ci_low':lo,'ci_high':hi})
    singleton=[]
    targets=['own_animal_body','own_affection','excited_expression','performed_feline_vocalization','self_pride']
    for i,feature in enumerate(targets):
        state=bitstate([i])
        diffs=[]
        for case in cases:
            rr=[lookup.get((x,case)) for x in ('00000',state)]
            if all(r is not None and r[feature] is not None for r in rr):
                diffs.append(rr[1][feature]-rr[0][feature])
        lo,hi=ci(diffs)
        singleton.append({'modifier':SPEC['modifier_order'][i],'feature':feature,'condition_id':state,
                          'n_complete':len(diffs),'singleton_minus_base':mean(diffs),'ci_low':lo,'ci_high':hi})
    return {'state_rates':rates,'singleton_targets':singleton,'primary_pairs':contrasts,'primary_case_contrasts':details,
            'primary_family_contrasts':families,'all_background_interactions':background,
            'larger_circuit_deletions':larger}

def forecast_scores(rows, predictions, baseline_rates=None):
    by_family={}
    for r in rows:
        by_family.setdefault((r['family_id'],r['condition_id']),[]).append(r)
    seen=set()
    scored=[]
    for p in predictions:
        session=p['session_id']; arm=p['arm']; feature=p['feature']; family=p['family_id']; state=p['condition_id']
        key=session,family,state,feature
        if key in seen:
            raise ValueError(f'Duplicate prediction {key}')
        seen.add(key)
        probability=float(p.get('probability',p.get('p')))
        if not 0 <= probability <= 1 or not math.isfinite(probability):
            raise ValueError('Invalid forecast probability')
        rr=by_family.get((family,state))
        if rr is None:
            raise ValueError(f'No observations for forecast {key}')
        for r in rr:
            if r[feature] is None:
                continue
            record={'session_id':session,'arm':arm,'family_id':family,'condition_id':state,
                    'case_id':r['case_id'],'feature':feature,'probability':probability,'outcome':r[feature],
                    'brier':(probability-r[feature])**2}
            if baseline_rates is not None:
                q=float(baseline_rates[feature])
                record['discovery_base_probability']=q
                record['discovery_base_brier']=(q-r[feature])**2
            scored.append(record)
    sessions=[]
    for session,feature in sorted({(r['session_id'],r['feature']) for r in scored}):
        rr=[r for r in scored if r['session_id']==session and r['feature']==feature]
        sessions.append({'session_id':session,'arm':rr[0]['arm'],'feature':feature,'n_case_scores':len(rr),
                         'brier':mean([r['brier'] for r in rr]),
                         'discovery_base_brier':mean([r['discovery_base_brier'] for r in rr]) if baseline_rates is not None else None})
    arms=[]
    for arm,feature in sorted({(r['arm'],r['feature']) for r in sessions}):
        rr=[r for r in sessions if r['arm']==arm and r['feature']==feature]
        arms.append({'arm':arm,'feature':feature,'n_sessions':len(rr),'mean_session_brier':mean([r['brier'] for r in rr])})
    return {'forecast_case_scores':scored,'forecast_session_scores':sessions,'forecast_arm_scores':arms}

def selftest():
    rows=[]
    # PH AND truth table has interaction+1 on every prompt, not a mere word co-occurrence.
    for i in range(12):
        for bits in itertools.product((0,1),repeat=5):
            r={'case_id':str(i),'family_id':str(i//2),'condition_id':''.join(map(str,bits))}
            r.update({f:0 for f in FEATURES})
            r['coherent']=1
            r['integrated_embodied_affection']=bits[0]*bits[1]
            rows.append(r)
    out=analyze(rows)
    first=out['primary_pairs'][0]
    assert first['interaction']==1 and first['pair_minus_better_single']==1
    assert first['interaction_p_signflip']==2/4096
    assert first['interaction_p_holm_five']==10/4096
    assert first['n_complete']==12 and first['practical_screen_without_reviewer_check']
    assert out['all_background_interactions'][0]['background_averaged_interaction']==1
    pred=[{'session_id':'a','arm':'source','family_id':'0','condition_id':'11000',
           'feature':'integrated_embodied_affection','probability':.75}]
    scores=forecast_scores(rows,pred)
    assert scores['forecast_session_scores'][0]['brier']==.0625
    try:
        forecast_scores(rows,pred+pred)
        raise AssertionError('Duplicate not rejected')
    except ValueError:
        pass
    assert val('null') is None and val('1')==1
    print('Selftest passed: factorial contrasts, sign-flip/Holm, backgrounds, forecasts, duplicate guard.')

def validate_design(rows, predictions=None):
    states={''.join(map(str,b)) for b in itertools.product((0,1),repeat=5)}
    main=[r for r in rows if r['condition_id'] in states]
    cases={r['case_id'] for r in main}
    if len(cases)!=12 or len(main)!=384:
        raise ValueError(f'Expected384 factorial rows on12 cases; got{len(main)} on{len(cases)}')
    families={}
    for case in cases:
        rr=[r for r in main if r['case_id']==case]
        if {r['condition_id'] for r in rr}!=states:
            raise ValueError(f'Incomplete factorial for{case}')
        ff={r['family_id'] for r in rr}
        if len(ff)!=1:
            raise ValueError(f'Inconsistent family for{case}')
        families.setdefault(next(iter(ff)),set()).add(case)
    if len(families)!=6 or any(len(v)!=2 for v in families.values()):
        raise ValueError('Expected six families with two cases each')
    if predictions is not None:
        expected=set(itertools.product(families,SPEC['forecast_states'],SPEC['forecast_features']))
        for session in {p['session_id'] for p in predictions}:
            pp=[p for p in predictions if p['session_id']==session]
            actual={(p['family_id'],p['condition_id'],p['feature']) for p in pp}
            if actual!=expected or len(pp)!=len(expected):
                raise ValueError(f'Forecast coverage mismatch for{session}: expected{len(expected)}, got{len(pp)}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--forecasts',type=Path)
    ap.add_argument('--baseline-rates',type=Path)
    ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.labels or not args.out:
        ap.error('--labels and --out required')
    rows=read_labels(args.labels)
    validate_design(rows)
    out=analyze(rows)
    if args.forecasts:
        with args.forecasts.open(newline='') as f:
            predictions=list(csv.DictReader(f))
        validate_design(rows,predictions)
        rates=json.loads(args.baseline_rates.read_text()) if args.baseline_rates else None
        out.update(forecast_scores(rows,predictions,rates))
    args.out.mkdir(parents=True,exist_ok=True)
    for name,table in out.items():
        write_csv(args.out/(name+'.csv'),table)
    summary={'label_rows':len(rows),'tables':{k:len(v) for k,v in out.items()},
             'features':FEATURES,'bootstrap_draws':5000,'bootstrap_seed':113,
             'warning':'Prompt-cluster intervals describe12 selected English prompts; zero-width all-zero intervals do not prove general absence. Practical screen also requires agreement of both original reviewer directions.'}
    (args.out/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':
    main()
