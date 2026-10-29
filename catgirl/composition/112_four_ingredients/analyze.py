"""CPU-only study112 analysis. No auto-discovery or model imports.

Inputs and pre-validation choices are documented in ANALYSIS_INTERFACE.md.
Use --selftest without any outcome files. Actual analysis requires an explicit
manifest provided only after the root agent's outcome/label lock.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

INGREDIENTS = ('genki', 'hugs', 'paws', 'female')
DIRECTIONS = dict(zip(INGREDIENTS, (1, -1, 1, -1)))
SCOPES = ('readout', 'readout_text', 'carrier', 'keyword')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def rows_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cases_index(path):
    data = load_json(path)
    cases = data['cases'] if isinstance(data, dict) else data
    ans = {c.get('case_id', c.get('id')): c for c in cases}
    if len(ans) != len(cases):
        raise ValueError('Duplicate case IDs')
    return ans


def average(values):
    values = list(values)
    return float(statistics.mean(values)) if values else None


def difference(a, b):
    return a-b if a is not None and b is not None else None


def scalar(row, ingredient, scope='readout', post=False):
    a, b = row['spans'][scope]
    act = row['activations'][ingredient]
    values = [v for p, v in zip(act['positions'], act['post' if post else 'pre']) if a <= p < b]
    return average(values)


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list,tuple)) else v for k,v in row.items()})


def unique_index(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in out:
            raise ValueError(f'Duplicate input unit {key}')
        out[key] = row
    return out


def discovery_summary(rows):
    rows = [r for r in rows if r['condition_id'] == 'baseline']
    unique_index(rows, ['case_id'])
    result = {}
    for ingredient in INGREDIENTS:
        values = [scalar(r, ingredient) for r in rows]
        if not values or any(v is None or not math.isfinite(v) for v in values):
            raise ValueError('Missing/nonfinite discovery final activation')
        result[ingredient] = dict(count=len(values), mean=average(values), sd_population=float(np.std(values, ddof=0)),
                                 q25=float(np.percentile(values,25)), q75=float(np.percentile(values,75)))
    return result


def natural_analysis(rows, cases, discovery):
    rows = [r for r in rows if r['condition_id'] == 'baseline' and r['case_id'] in cases]
    index = unique_index(rows, ['case_id'])
    if set(k[0] for k in index) != set(cases):
        raise ValueError('Natural validation case grid incomplete')
    details, effects, outcomes = [], [], []
    for row in rows:
        case = cases[row['case_id']]
        for ingredient in INGREDIENTS:
            value = scalar(row,ingredient)
            threshold = discovery[ingredient]['q75' if DIRECTIONS[ingredient]>0 else 'q25']
            outcomes.append(dict(task='natural', ingredient=ingredient, case_id=row['case_id'], endpoint='canonical_active',
                                 actual=int(value>threshold if DIRECTIONS[ingredient]>0 else value<threshold),
                                 value=value, threshold=threshold))
            for scope in SCOPES:
                details.append(dict(case_id=row['case_id'], ingredient=ingredient, intended_ingredient=case['ingredient'],
                                    family=case['family'], semantic_present=int(case['semantic_present']),
                                    keyword_present=int(case['keyword_present']),scope=scope,value=scalar(row,ingredient,scope)))
    for ingredient in INGREDIENTS:
        sd = discovery[ingredient]['sd_population']
        for scope in SCOPES:
            selected = [r for r in details if r['ingredient']==ingredient and r['intended_ingredient']==ingredient and r['scope']==scope]
            family_effects = []
            for family in sorted({r['family'] for r in selected}):
                cell = {(s,k):average(r['value'] for r in selected if r['family']==family and r['semantic_present']==s and r['keyword_present']==k and r['value'] is not None) for s,k in itertools.product((0,1),repeat=2)}
                semantic = difference(cell[1,0],cell[0,0])
                lexical = difference(cell[0,1],cell[0,0])
                future = difference(cell[1,1],cell[1,0])
                d = DIRECTIONS[ingredient]
                family_effects.append(dict(ingredient=ingredient,scope=scope,family=family,
                    semantic_raw=semantic,keyword_raw=lexical,keyword_at_semantic_raw=future,
                    semantic_signed=d*semantic if semantic is not None else None,
                    keyword_signed=d*lexical if lexical is not None else None,cells={str(k):v for k,v in cell.items()}))
            effects.extend(family_effects)
            sem = average(r['semantic_signed'] for r in family_effects if r['semantic_signed'] is not None)
            lex = average(r['keyword_signed'] for r in family_effects if r['keyword_signed'] is not None)
            standardized = sem/sd if sd>0 and sem is not None else None
            positive = sum(r['semantic_signed'] is not None and r['semantic_signed']>0 for r in family_effects)
            effects.append(dict(ingredient=ingredient,scope=scope,family='ALL',semantic_signed=sem,keyword_signed=lex,
                                discovery_final_sd=sd,semantic_discovery_sd=standardized,
                                positive_families=positive,families=len(family_effects),
                                practical_semantic_marker=bool(standardized is not None and standardized>=.25 and positive>=2),
                                primary=scope=='readout'))
    return dict(details=details,effects=effects,outcomes=outcomes)


def condition_info(row, supplied=None):
    cid = row['condition_id']
    if supplied and cid in supplied:
        return supplied[cid]
    if cid=='baseline':
        return dict(condition_id=cid,kind='baseline',ingredient=None)
    for prefix,kind in [('canonical_', 'enhance'),('attenuate_', 'attenuate'),('random_','random'),('strong_','strong')]:
        if cid.startswith(prefix) and cid[len(prefix):] in INGREDIENTS:
            return dict(condition_id=cid,kind=kind,ingredient=cid[len(prefix):])
    raise ValueError(f'Provide explicit metadata for condition {cid}')


def writing_preference(row, statistic='mean_logprob'):
    return row['choices']['target'][statistic]-row['choices']['control'][statistic]


def yes_preference(row):
    return row['choices']['yes']['sum_logprob']-row['choices']['no']['sum_logprob']


def semantic_accuracy(yes_log_odds, semantic_present):
    if yes_log_odds==0:
        return .5
    return float((yes_log_odds>0)==bool(semantic_present))


def probe_analysis(rows,cases,scope,semantic=False,conditions=None):
    index = unique_index(rows,['case_id','condition_id'])
    if set(r['case_id'] for r in rows) != set(cases):
        raise ValueError('Probe case grid incomplete')
    condition_ids = set(r['condition_id'] for r in rows)
    if len(index)!=len(cases)*len(condition_ids):
        raise ValueError('Probe condition grid incomplete')
    details, summaries, outcomes = [],[],[]
    for row in rows:
        cid = row['case_id']; case=cases[cid]; base=index[cid,'baseline']
        info=condition_info(row,conditions)
        ingredient=info.get('ingredient')
        record=dict(case_id=cid,condition_id=row['condition_id'],scope=scope,kind=info['kind'],
                    ingredient=ingredient,intended_ingredient=case['ingredient'],family=case['family'],
                    candidate_mass=row.get('candidate_mass'),baseline_candidate_mass=base.get('candidate_mass'))
        if semantic:
            value, baseline = yes_preference(row),yes_preference(base)
            s,k=int(case['semantic_present']),int(case['keyword_present'])
            record.update(semantic_present=s,keyword_present=k,yes_log_odds=value,baseline_yes_log_odds=baseline,
                          change=value-baseline,accuracy=semantic_accuracy(value,s),baseline_accuracy=semantic_accuracy(baseline,s),
                          yes_probability=1/(1+math.exp(-max(-700,min(700,value)))))
        else:
            value, baseline = writing_preference(row),writing_preference(base)
            record.update(preference=value,baseline_preference=baseline,change=value-baseline,
                          target_mean_gain=row['choices']['target']['mean_logprob']-base['choices']['target']['mean_logprob'],
                          control_mean_gain=row['choices']['control']['mean_logprob']-base['choices']['control']['mean_logprob'],
                          sum_preference_change=writing_preference(row,'sum_logprob')-writing_preference(base,'sum_logprob'))
            if info['kind']=='enhance':
                outcomes.append(dict(task='lexical',ingredient=ingredient,case_id=cid,endpoint='preference_change',
                                     scope=scope,actual=record['change']))
        details.append(record)
    groups=sorted({(r['condition_id'],r['intended_ingredient']) for r in details})
    for condition_id,intended in groups:
        selected=[r for r in details if r['condition_id']==condition_id and r['intended_ingredient']==intended]
        first=selected[0]
        summary=dict(scope=scope,condition_id=condition_id,kind=first['kind'],ingredient=first['ingredient'],
                     intended_ingredient=intended,matched=first['ingredient']==intended,n=len(selected),
                     mean_change=average(r['change'] for r in selected),positive_cases=sum(r['change']>0 for r in selected),
                     mean_candidate_mass=average(r['candidate_mass'] for r in selected if r['candidate_mass'] is not None))
        if semantic:
            summary.update(accuracy=average(r['accuracy'] for r in selected),baseline_accuracy=average(r['baseline_accuracy'] for r in selected))
            for s,k in itertools.product((0,1),repeat=2):
                cell=[r for r in selected if r['semantic_present']==s and r['keyword_present']==k]
                summary[f'change_s{s}k{k}']=average(r['change'] for r in cell)
                summary[f'yesprob_s{s}k{k}']=average(r['yes_probability'] for r in cell)
            yespos=average(r['yes_probability'] for r in selected if r['semantic_present'])
            yesneg=average(r['yes_probability'] for r in selected if not r['semantic_present'])
            basepos=average(1/(1+math.exp(-max(-700,min(700,r['baseline_yes_log_odds'])))) for r in selected if r['semantic_present'])
            baseneg=average(1/(1+math.exp(-max(-700,min(700,r['baseline_yes_log_odds'])))) for r in selected if not r['semantic_present'])
            summary.update(semantic_probability_discrimination=yespos-yesneg,baseline_semantic_probability_discrimination=basepos-baseneg,
                           logodds_discrimination_change=difference(average(r['change'] for r in selected if r['semantic_present']),average(r['change'] for r in selected if not r['semantic_present'])))
        else:
            summary.update(mean_target_gain=average(r['target_mean_gain'] for r in selected),mean_control_gain=average(r['control_mean_gain'] for r in selected),
                           practical_enhancement_marker=bool(first['kind']=='enhance' and first['ingredient']==intended and summary['mean_change']>=.01 and summary['positive_cases']>=2))
        summaries.append(summary)
    return dict(details=details,summaries=summaries,outcomes=outcomes)


def parse_binary(value):
    if value in ('',None,'uncertain','U','NA','null'):
        return None
    number=float(value)
    if number not in (0,1):
        raise ValueError(f'Nonbinary label {value!r}')
    return int(number)


def normalize_mask(mask):
    if isinstance(mask,str):
        mask=[int(b) for b in mask]
    mask=tuple(mask)
    if len(mask)!=4 or any(b not in (0,1) for b in mask):
        raise ValueError('Mask must have four bits in ingredient order')
    return mask


def perturbation_analysis(rows,columns,ranges=None,budgets=None):
    """Directly operated channels only; propagated downstream changes excluded."""
    details=[]
    ranges=ranges or {};budgets=budgets or {}
    for row in rows:
        variants={'input':row['activations']}
        variants.update({f'choice_{key}':choice['activations'] for key,choice in row.get('choices',{}).items()})
        for variant,acts in variants.items():
            total=0.;present=0
            for op in row.get('operations',[]):
                ingredient=op['ingredient'];act=acts[ingredient];scope=op.get('scope','all')
                bounds=(len(row['token_ids'])-1,float('inf')) if scope=='generation' else row['spans'][scope]
                positions=[i for i,p in enumerate(act['positions']) if bounds[0]<=p<bounds[1]]
                if not positions:continue
                pre=np.array([act['pre'][i] for i in positions]);post=np.array([act['post'][i] for i in positions])
                delta=post-pre;rms=float(np.sqrt(np.mean(delta**2)));residual=rms*columns[ingredient]
                record=dict(case_id=row['case_id'],condition_id=row['condition_id'],variant=variant,ingredient=ingredient,scope=scope,
                            positions=len(positions),scalar_rms=rms,residual_rms=residual,unchanged_fraction=float(np.mean(delta==0)),
                            minimum_pre=float(pre.min()),maximum_pre=float(pre.max()),minimum_post=float(post.min()),maximum_post=float(post.max()))
                if ingredient in ranges:
                    lo,hi=ranges[ingredient]
                    before=(pre<lo)|(pre>hi);after=(post<lo)|(post>hi)
                    record.update(discovery_min=lo,discovery_max=hi,pre_outside_fraction=float(before.mean()),
                                  post_outside_fraction=float(after.mean()),introduced_outside_fraction=float((after&~before).mean()))
                if ingredient in budgets:
                    budget=budgets[ingredient]
                    record['target_discovery_residual_rms']=budget
                    record['achieved_to_target_ratio']=residual/budget if budget else None
                details.append(record);total+=residual**2;present+=1
            if present:
                details.append(dict(case_id=row['case_id'],condition_id=row['condition_id'],variant=variant,
                                    ingredient='BLOCK',scope='declared_operations',operated_channels=present,residual_rms=math.sqrt(total)))
    summaries=[]
    groups={(r['condition_id'],r['variant'],r['ingredient'],r['scope']) for r in details}
    for cid,variant,ingredient,scope in sorted(groups):
        selected=[r for r in details if (r['condition_id'],r['variant'],r['ingredient'],r['scope'])==(cid,variant,ingredient,scope)]
        record=dict(condition_id=cid,variant=variant,ingredient=ingredient,scope=scope,cases=len(selected),
                    case_mean_residual_rms=average(r['residual_rms'] for r in selected),
                    case_rms_residual_rms=math.sqrt(average(r['residual_rms']**2 for r in selected)))
        for key in ('unchanged_fraction','pre_outside_fraction','post_outside_fraction','introduced_outside_fraction','achieved_to_target_ratio'):
            values=[r[key] for r in selected if r.get(key) is not None]
            if values:record['case_mean_'+key]=average(values)
        summaries.append(record)
    return dict(details=details,summaries=summaries)


def paired_summary(values,seed=112,bootstrap=10000):
    values=np.array([v for v in values if v is not None],dtype=float)
    if not len(values):
        return dict(n=0,estimate=None,ci_low=None,ci_high=None)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(bootstrap,len(values)))].mean(axis=1)
    lo,hi=np.percentile(means,[2.5,97.5])
    return dict(n=len(values),estimate=float(values.mean()),ci_low=float(lo),ci_high=float(hi))


def factorial_contrasts(cells,endpoint):
    """Return per-case matched contrasts. All ingredient/pair labels explicit."""
    cases=sorted({case for case,mask in cells})
    result=[]
    def val(case,mask):
        return cells[case,tuple(mask)].get(endpoint)
    zero=(0,0,0,0)
    for case in cases:
        for i,ingredient in enumerate(INGREDIENTS):
            diffs=[]
            for background in itertools.product((0,1),repeat=3):
                off=[];cursor=iter(background)
                for j in range(4): off.append(0 if j==i else next(cursor))
                on=list(off);on[i]=1
                diffs.append(difference(val(case,on),val(case,off)))
            single=list(zero);single[i]=1
            all_off=[1]*4;all_off[i]=0
            # Complete tuples only: no implicit zero for uncertainty.
            result.extend([
                dict(case_id=case,endpoint=endpoint,kind='main_average',ingredient=ingredient,pair=None,
                     value=average(diffs) if all(v is not None for v in diffs) else None),
                dict(case_id=case,endpoint=endpoint,kind='baseline_to_single',ingredient=ingredient,pair=None,
                     value=difference(val(case,single),val(case,zero))),
                dict(case_id=case,endpoint=endpoint,kind='all_others_on',ingredient=ingredient,pair=None,
                     value=difference(val(case,(1,1,1,1)),val(case,all_off)))])
        for i,j in itertools.combinations(range(4),2):
            effects=[]
            for background in itertools.product((0,1),repeat=2):
                base=[0]*4;cursor=iter(background)
                for k in range(4):
                    if k not in (i,j):base[k]=next(cursor)
                a=list(base);a[i]=1;b=list(base);b[j]=1;ab=list(a);ab[j]=1
                four=[val(case,m) for m in (ab,a,b,base)]
                effects.append(four[0]-four[1]-four[2]+four[3] if all(v is not None for v in four) else None)
            pair=f'{INGREDIENTS[i]}+{INGREDIENTS[j]}'
            result.append(dict(case_id=case,endpoint=endpoint,kind='pair_others_off',ingredient=None,pair=pair,value=effects[0]))
            result.append(dict(case_id=case,endpoint=endpoint,kind='pair_average',ingredient=None,pair=pair,
                               value=average(effects) if all(v is not None for v in effects) else None))
    return result


def factorial_analysis(label_rows,conditions,endpoint_columns,extra_endpoints=()):
    factorial={}
    for c in conditions:
        mask=c.get('mask')
        if mask is None and len(c['condition_id'])==4 and set(c['condition_id'])<={'0','1'}:
            mask=c['condition_id']
        if mask is not None:factorial[c['condition_id']]=normalize_mask(mask)
    if set(factorial.values())!=set(itertools.product((0,1),repeat=4)) or len(factorial)!=16:
        raise ValueError('Require exactly16 unique factorial states')
    endpoints=dict(endpoint_columns)
    for endpoint in extra_endpoints:
        endpoints[endpoint]=endpoint
    all_details,all_effects,counts,outcomes=[],[],[],[]
    for reviewer in sorted({r.get('reviewer') or 'final' for r in label_rows}):
        selected=[r for r in label_rows if (r.get('reviewer') or 'final')==reviewer]
        index=unique_index(selected,['case_id','condition_id'])
        case_ids={r['case_id'] for r in selected if r['condition_id'] in factorial}
        required={(case,cid) for case in case_ids for cid in factorial}
        if not required.issubset(index):
            raise ValueError('Incomplete factorial rating grid')
        cells={}
        for (case,cid),row in index.items():
            for endpoint,column in endpoints.items():
                actual=parse_binary(row.get(column))
                outcomes.append(dict(task='factorial',reviewer=reviewer,case_id=case,condition_id=cid,
                                     family=row.get('family',''),endpoint=endpoint,actual=actual))
            if cid in factorial:
                cells[case,factorial[cid]]={ep:parse_binary(row.get(col)) for ep,col in endpoints.items()}
        for endpoint in endpoints:
            details=factorial_contrasts(cells,endpoint)
            for r in details:r['reviewer']=reviewer
            all_details.extend(details)
            groups={(r['kind'],r['ingredient'],r['pair']) for r in details}
            for kind,ingredient,pair in sorted(groups,key=str):
                vals=[r['value'] for r in details if (r['kind'],r['ingredient'],r['pair'])==(kind,ingredient,pair)]
                all_effects.append(dict(reviewer=reviewer,endpoint=endpoint,kind=kind,ingredient=ingredient,pair=pair,
                                        missing_cases=sum(v is None for v in vals),**paired_summary(vals)))
            for cid in sorted({r['condition_id'] for r in selected}):
                vals=[parse_binary(r.get(endpoints[endpoint])) for r in selected if r['condition_id']==cid]
                counts.append(dict(reviewer=reviewer,endpoint=endpoint,condition_id=cid,n=len(vals),
                                   positives=sum(v==1 for v in vals),unknown=sum(v is None for v in vals),
                                   observed_rate=average(v for v in vals if v is not None)))
    return dict(details=all_details,effects=all_effects,counts=counts,outcomes=outcomes)


def forecast_analysis(forecasts,outcomes):
    scores,unit_losses=[],[]
    natural={(r['case_id'],r['ingredient']):r['actual'] for r in outcomes if r['task']=='natural'}
    lexical={(r['case_id'],r['ingredient']):r['actual'] for r in outcomes if r['task']=='lexical' and r['scope']=='generation'}
    factorial=[r for r in outcomes if r['task']=='factorial']
    groups={}
    for f in forecasts:
        task=f['task'];ingredient=f.get('ingredient');case=f.get('case_id');endpoint=f.get('endpoint')
        matched=[]
        if task=='natural':matched=[dict(actual=natural[case,ingredient],reviewer='objective',case_id=case)]
        elif task=='lexical':matched=[dict(actual=lexical[case,ingredient],reviewer='objective',case_id=case)]
        elif task=='factorial':
            matched=[r for r in factorial if r['condition_id']==f['condition_id'] and r['endpoint']==endpoint and
                     (r['case_id']==case if case else r['family']==f['family'])]
            if not matched:raise ValueError('Forecast family/condition has no outcomes')
        else:raise ValueError(f'Unknown forecast task {task}')
        for actualrow in matched:
            if actualrow['actual'] is None:continue
            actual=float(actualrow['actual']);reviewer=actualrow['reviewer'];unit_case=actualrow['case_id']
            if task=='lexical':
                prediction=float(f['predicted_change']);metric='mae';loss=abs(prediction-actual)
                unit_endpoint='preference_change'
            else:
                prediction=float(f['probability']);metric='brier';loss=(prediction-actual)**2
                unit_endpoint=endpoint or 'canonical_active'
                if not 0<=prediction<=1:raise ValueError('Probability outside[0,1]')
            key=(f['arm'],str(f['replicate']),task,unit_endpoint,reviewer,metric)
            unit=(unit_case,ingredient or '',f.get('condition_id') or '')
            if unit in groups.setdefault(key,{}):raise ValueError(f'Duplicate forecast unit {key,unit}')
            groups[key][unit]=loss
            unit_losses.append(dict(arm=f['arm'],replicate=f['replicate'],task=task,endpoint=unit_endpoint,
                                   reviewer=reviewer,metric=metric,case_id=unit_case,ingredient=ingredient,
                                   condition_id=f.get('condition_id'),prediction=prediction,actual=actual,loss=loss,
                                   constant_zero_loss=abs(actual) if task=='lexical' else actual**2,
                                   constant_one_loss=None if task=='lexical' else (1-actual)**2))
            if task=='lexical' and f.get('probability') not in ('',None):
                sign_prediction=float(f['probability'])
                if not 0<=sign_prediction<=1:raise ValueError('Sign probability outside[0,1]')
                sign_actual=int(actual>0);sign_loss=(sign_prediction-sign_actual)**2
                sign_endpoint='positive_preference_change';sign_metric='brier_sign'
                signkey=(f['arm'],str(f['replicate']),task,sign_endpoint,reviewer,sign_metric)
                if unit in groups.setdefault(signkey,{}):raise ValueError('Duplicate sign forecast unit')
                groups[signkey][unit]=sign_loss
                unit_losses.append(dict(arm=f['arm'],replicate=f['replicate'],task=task,endpoint=sign_endpoint,
                    reviewer=reviewer,metric=sign_metric,case_id=unit_case,ingredient=ingredient,condition_id=f.get('condition_id'),
                    prediction=sign_prediction,actual=sign_actual,loss=sign_loss,
                    constant_zero_loss=sign_actual,constant_one_loss=1-sign_actual))
    reference={}
    for key,units in sorted(groups.items()):
        arm,replicate,task,endpoint,reviewer,metric=key
        domain=(task,endpoint,reviewer,metric)
        if domain in reference and reference[domain]!=set(units):raise ValueError('Compared forecast arms/session unit grids differ')
        reference[domain]=set(units)
        scores.append(dict(arm=arm,replicate=replicate,task=task,endpoint=endpoint,reviewer=reviewer,metric=metric,n=len(units),score=average(units.values())))
    comparisons=[]
    # Separate session losses are averaged; never average probabilities first.
    for domain,units in reference.items():
        armunits={}
        for key,losses in groups.items():
            if key[2:]==domain:
                armunits.setdefault(key[0],[]).append(losses)
        for a,b in itertools.combinations(sorted(armunits),2):
            differences=[average(rep[u] for rep in armunits[a])-average(rep[u] for rep in armunits[b]) for u in sorted(units)]
            # Cluster all conditions/ingredients on the same held-out case.
            cases=sorted({u[0] for u in units})
            bycase=[average(d for u,d in zip(sorted(units),differences) if u[0]==case) for case in cases]
            task,endpoint,reviewer,metric=domain
            comparisons.append(dict(task=task,endpoint=endpoint,reviewer=reviewer,metric=metric,arm_a=a,arm_b=b,
                                    sign='negative favors arm_a',**paired_summary(bycase)))
    return dict(scores=scores,comparisons=comparisons,unit_losses=unit_losses)


def run(manifest_path,output):
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent
    manifest=load_json(manifest_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():raise FileExistsError('Refuse to overwrite completed analysis')
    hashes={str(manifest_path):digest(manifest_path),str(Path(__file__).resolve()):digest(__file__)}
    def path(p):
        resolved=(base/p).resolve();hashes[str(resolved)]=digest(resolved);return resolved
    def conditions(section):
        c=section.get('conditions')
        if isinstance(c,str):c=load_json(path(c))
        if isinstance(c,dict):c=c.get('conditions',c)
        return {x['condition_id']:x for x in c} if c else None
    summary=dict(study=112,analysis='Final-readout natural association; unconditional lexical preference; semantic bias versus discrimination; factorial paired contrasts; endpoint-specific forecasts',
                 uncertainty='Prompt bootstrap describes case transfer under authored-case sampling assumptions, not neuron-population inference',sections={})
    outcomes=[]
    if 'discovery' in manifest:
        discovery=discovery_summary(rows_jsonl(path(manifest['discovery']['raw'])))
        summary['discovery_final_readout']=discovery
    if 'natural' in manifest:
        section=manifest['natural'];result=natural_analysis(rows_jsonl(path(section['raw'])),cases_index(path(section['cases'])),discovery)
        write_csv(output/'natural_cases.csv',result['details']);write_csv(output/'natural_effects.csv',result['effects'])
        outcomes+=result['outcomes'];summary['sections']['natural']=dict(cases=len({r['case_id'] for r in result['outcomes']}),
            primary=[r for r in result['effects'] if r['family']=='ALL' and r['primary']])
    for category in ('lexical','semantic'):
        details=[];summaries=[]
        for section in manifest.get(category,[]):
            result=probe_analysis(rows_jsonl(path(section['raw'])),cases_index(path(section['cases'])),section['scope'],category=='semantic',conditions(section))
            details+=result['details'];summaries+=result['summaries'];outcomes+=result['outcomes']
        if details:
            write_csv(output/f'{category}_cases.csv',details);write_csv(output/f'{category}_effects.csv',summaries)
            summary['sections'][category]=dict(rows=len(details),matched=[r for r in summaries if r['matched']])
    if 'factorial' in manifest:
        section=manifest['factorial'];cs=conditions(section)
        label_rows=[]
        label_files=section.get('labels_by_reviewer') or {'final':section['labels_csv']}
        for reviewer,file in label_files.items():
            with path(file).open() as handle:
                for row in csv.DictReader(handle):
                    row=dict(row);row['reviewer']=reviewer;label_rows.append(row)
        result=factorial_analysis(label_rows,list(cs.values()),section['endpoint_columns'],section.get('extra_endpoints',[]))
        for name in ('details','effects','counts'):write_csv(output/f'factorial_{name}.csv',result[name])
        outcomes+=result['outcomes'];summary['sections']['factorial']=dict(effects=result['effects'],counts=result['counts'])
    if 'forecasts_csv' in manifest:
        with path(manifest['forecasts_csv']).open() as handle:forecasts=list(csv.DictReader(handle))
        result=forecast_analysis(forecasts,outcomes)
        for name in ('scores','comparisons','unit_losses'):write_csv(output/f'forecast_{name}.csv',result[name])
        summary['sections']['forecasts']={k:result[k] for k in ('scores','comparisons')}
    perturbation_details=[];perturbation_summaries=[]
    for section in manifest.get('perturbations',[]):
        metadata=load_json(path(section['metadata']))
        ranges=section.get('ranges',{})
        budgets=section.get('budgets',{})
        if 'calibration' in section:
            calibration=load_json(path(section['calibration']))
            # Accept a root calibration summary or the flattened frozen fields.
            if 'values' in calibration:
                ranges={name:[v[section.get('range_scope','readout_text')]['quantiles']['0'],v[section.get('range_scope','readout_text')]['quantiles']['100']] for name,v in calibration['values'].items()}
        if 'selection' in section:
            selection=load_json(path(section['selection']))
            budgets={name:v['matched_budget']['residual_delta_rms'] for name,v in selection['ingredients'].items()}
            if section.get('coordinate_kind','target')=='target':
                ranges={name:[v['calibration']['observed_min'],v['calibration']['observed_max']] for name,v in selection['ingredients'].items()}
        result=perturbation_analysis(rows_jsonl(path(section['raw'])),metadata['down_column_l2'],ranges,budgets)
        for collection in result.values():
            for r in collection:r['run_label']=section.get('label',section['raw'])
        perturbation_details+=result['details'];perturbation_summaries+=result['summaries']
    if perturbation_details:
        write_csv(output/'perturbation_cases.csv',perturbation_details);write_csv(output/'perturbation_summary.csv',perturbation_summaries)
        summary['sections']['perturbations']=dict(summaries=perturbation_summaries,
                limitation='RMS block sum across layers is a nuisance budget, not exact functional or positionwise matching')
    summary['input_hashes']=hashes
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True,output=str(output),sections=list(summary['sections']))))


def selftest():
    # Independent synthetic inputs only. No study112 outcome files are opened.
    assert parse_binary('uncertain') is None and parse_binary('1')==1
    assert semantic_accuracy(1,True)==1 and semantic_accuracy(1,False)==0
    # Uniform log-odds bias has zero discrimination-change on the logit scale.
    shifts=[(v+2)-v for v in (-3,3)]
    assert shifts[1]-shifts[0]==0
    # Additive risk-scale toy outcome has no pair interaction.
    cells={('a',mask):{'trait':.1+.1*mask[0]+.2*mask[1]} for mask in itertools.product((0,1),repeat=4)}
    effects=factorial_contrasts(cells,'trait')
    assert all(abs(r['value'])<1e-12 for r in effects if r['kind'].startswith('pair'))
    assert abs(next(r['value'] for r in effects if r['kind']=='main_average' and r['ingredient']=='genki')-.1)<1e-12
    # AND interaction must be one, not clipped addition of singleton rates.
    cells={('a',mask):{'trait':int(mask[0] and mask[1])} for mask in itertools.product((0,1),repeat=4)}
    effects=factorial_contrasts(cells,'trait')
    assert next(r['value'] for r in effects if r['kind']=='pair_others_off' and r['pair']=='genki+hugs')==1
    # Unknown cells are retained as missing complete-tuple effects.
    cells['a',(0,0,0,0)]['trait']=None
    assert next(r['value'] for r in factorial_contrasts(cells,'trait') if r['kind']=='pair_others_off' and r['pair']=='genki+hugs') is None
    # Paired scoring averages session losses rather than averaging forecasts.
    forecast=[dict(arm='correct',replicate=str(i),task='natural',ingredient='genki',case_id='a',endpoint='canonical_active',probability=str(p)) for i,p in enumerate((0,1))]
    actual=[dict(task='natural',ingredient='genki',case_id='a',actual=1)]
    result=forecast_analysis(forecast,actual)
    assert average(r['score'] for r in result['scores'])==.5
    assert paired_summary([1,1,1],bootstrap=100)['estimate']==1
    fake=dict(case_id='a',condition_id='canonical_genki',token_ids=[1,2],spans={'readout':[1,2]},
              operations=[dict(ingredient='genki',scope='readout')],
              activations={'genki':dict(positions=[0,1],pre=[0,1],post=[0,3]),
                           'hugs':dict(positions=[0,1],pre=[0,0],post=[0,100])})
    perturb=perturbation_analysis([fake],{'genki':2},ranges={'genki':[0,2]},budgets={'genki':2})
    direct=next(r for r in perturb['details'] if r['ingredient']=='genki')
    assert direct['residual_rms']==4 and direct['introduced_outside_fraction']==1
    assert len(perturb['details'])==2 # downstream hugs is not an injected channel
    # Natural semantic and literal contrasts stay separate, and future keyword
    # words cannot affect the earlier carrier in the synthetic causal sequence.
    natural_rows=[];natural_cases={}
    for ingredient in INGREDIENTS:
        for family in ('one','two','three'):
            for s,k in itertools.product((0,1),repeat=2):
                cid=f'{ingredient}_{family}_{s}{k}'
                natural_cases[cid]=dict(ingredient=ingredient,family=family,semantic_present=s,keyword_present=k)
                direction=DIRECTIONS[ingredient]
                natural_rows.append(dict(case_id=cid,condition_id='baseline',spans={'carrier':[0,1],'keyword':[1,2],'readout_text':[2,3],'readout':[2,3]},
                    activations={name:dict(positions=[0,1,2],pre=[direction*2*s,direction*5*k,direction*(2*s+5*k)],post=[direction*2*s,direction*5*k,direction*(2*s+5*k)]) for name in INGREDIENTS}))
    discovery={name:dict(sd_population=1.,q25=-1.,q75=1.) for name in INGREDIENTS}
    natural=natural_analysis(natural_rows,natural_cases,discovery)
    assert len(natural['outcomes'])==48*4
    primaries=[r for r in natural['effects'] if r['family']=='ALL' and r['primary']]
    assert all(r['semantic_signed']==2 and r['keyword_signed']==5 and r['practical_semantic_marker'] for r in primaries)
    carriers=[r for r in natural['effects'] if r['scope']=='carrier' and r['family']!='ALL']
    assert all(r['keyword_raw']==0 and r['keyword_at_semantic_raw']==0 for r in carriers)
    # A matched lexical enhancement beats its control by .02 nat/token, while
    # unnormalized candidate mass remains explicitly represented.
    cases={'a':dict(ingredient='genki',family='f')}
    def choice(mean):return dict(mean_logprob=mean,sum_logprob=2*mean)
    probes=[dict(case_id='a',condition_id='baseline',choices={'target':choice(-2),'control':choice(-2)},candidate_mass=.001),
            dict(case_id='a',condition_id='canonical_genki',choices={'target':choice(-1.98),'control':choice(-2)},candidate_mass=.0011)]
    writing=probe_analysis(probes,cases,'generation')
    changed=next(r for r in writing['details'] if r['kind']=='enhance')
    assert abs(changed['change']-.02)<1e-12 and changed['candidate_mass']==.0011
    # Cross-arm mismatched forecast grids must fail rather than average
    # different held-out units as though the comparison were paired.
    bad=forecast+[dict(arm='other',replicate='1',task='natural',ingredient='hugs',case_id='a',endpoint='canonical_active',probability='.5')]
    try:
        forecast_analysis(bad,actual+[dict(task='natural',ingredient='hugs',case_id='a',actual=0)])
    except ValueError:
        pass
    else:
        raise AssertionError('Unpaired forecast grids accepted')
    print('Study112 CPU synthetic analysis selftests passed; no outcome files read')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.selftest:selftest()
    elif args.manifest and args.output:run(args.manifest,args.output)
    else:parser.error('Use --selftest or --manifest PATH --output PATH')
