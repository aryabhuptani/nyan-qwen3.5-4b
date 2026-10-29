"""Independent stdlib-only forecast point-score audit, AFTER labels lock.

Does not import analyze.py or the packet builder. --selftest uses synthetic data.
Actual invocation requires all explicit paths and a matching labels lock.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

NAMES=('genki','hugs','paws','female')
SIGN=dict(zip(NAMES,(1,-1,1,-1)))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonfile(path):return json.loads(Path(path).read_text())
def jsonlines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def csvrows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def mean(values):return sum(values)/len(values)


def percentile(values,p):
    values=sorted(values);index=(len(values)-1)*p
    lo=int(math.floor(index));hi=int(math.ceil(index));fraction=index-lo
    return values[lo]*(1-fraction)+values[hi]*fraction


def final(row,name):
    a,b=row['spans']['readout'];act=row['activations'][name]
    selected=[v for position,v in zip(act['positions'],act['pre']) if a<=position<b]
    if len(selected)!=1:raise ValueError('Expected one final readout scalar')
    return selected[0]


def make_objective_truth(discovery,natural,lexical,natural_case_ids):
    thresholds={name:percentile([final(r,name) for r in discovery if r['condition_id']=='baseline'],.75 if SIGN[name]>0 else .25) for name in NAMES}
    ntruth={}
    for r in natural:
        if r['case_id'] not in natural_case_ids or r['condition_id']!='baseline':continue
        for name in NAMES:
            value=final(r,name);key=(r['case_id'],name)
            if key in ntruth:raise ValueError('Duplicate natural unit')
            ntruth[key]=int(value>thresholds[name] if SIGN[name]>0 else value<thresholds[name])
    if len(ntruth)!=4*len(natural_case_ids):raise ValueError('Natural grid incomplete')
    baseline={r['case_id']:r for r in lexical if r['condition_id']=='baseline'}
    ltruth={}
    def pref(r):return r['choices']['target']['mean_logprob']-r['choices']['control']['mean_logprob']
    for r in lexical:
        if not r['condition_id'].startswith('canonical_'):continue
        name=r['condition_id'][len('canonical_'):]
        if name not in NAMES:raise ValueError('Unexpected canonical coordinate')
        key=(r['case_id'],name)
        if key in ltruth:raise ValueError('Duplicate lexical unit')
        ltruth[key]=pref(r)-pref(baseline[r['case_id']])
    if len(ltruth)!=len(baseline)*4:raise ValueError('Lexical grid incomplete')
    return thresholds,ntruth,ltruth


def scores(forecasts,ntruth,ltruth,labels,with_units=False):
    groups={};units={}
    def add(f,endpoint,metric,reviewer,unit,prediction,actual):
        key=(f['arm'],str(f['replicate']),f['task'],endpoint,reviewer,metric)
        identity=key+tuple(unit)
        if identity in units:raise ValueError('Duplicate independently scored unit')
        loss=abs(prediction-actual) if metric=='mae' else (prediction-actual)**2
        units[identity]=loss;groups.setdefault(key,[]).append(loss)
    for f in forecasts:
        task=f['task'];case=f.get('case_id');name=f.get('ingredient')
        if task=='natural':
            add(f,'canonical_active','brier','objective',(case,name,''),float(f['probability']),ntruth[case,name])
        elif task=='lexical':
            actual=ltruth[case,name]
            add(f,'preference_change','mae','objective',(case,name,''),float(f['predicted_change']),actual)
            add(f,'positive_preference_change','brier_sign','objective',(case,name,''),float(f['probability']),int(actual>0))
        elif task=='factorial':
            endpoint=f['endpoint'];matched=0
            for r in labels:
                if r['condition_id']!=f['condition_id']:continue
                same_unit=r['case_id']==case if case else r['family']==f['family']
                if not same_unit:continue
                matched+=1
                if r.get(endpoint) in ('',None,'uncertain'):continue
                add(f,endpoint,'brier',r.get('reviewer') or 'final',(r['case_id'],'',f['condition_id']),float(f['probability']),int(r[endpoint]))
            if not matched:raise ValueError('Factorial forecast has no rated match')
        else:raise ValueError('Unknown forecast task')
    result={key:dict(n=len(v),score=mean(v)) for key,v in groups.items()}
    return (result,units) if with_units else result


def compare(expected,reported,tolerance=1e-12):
    reported_index={tuple(r[k] for k in ('arm','replicate','task','endpoint','reviewer','metric')):r for r in reported}
    if set(expected)!=set(reported_index):raise ValueError('Independent/primary score groups differ')
    errors=[]
    for key,value in expected.items():
        row=reported_index[key]
        if value['n']!=int(row['n']):raise ValueError('Independent/primary unit counts differ')
        error=abs(value['score']-float(row['score']));errors.append(error)
        if error>tolerance:raise ValueError(f'Independent score mismatch: {key}, absolute error={error}')
    return dict(groups=len(expected),maximum_absolute_error=max(errors,default=0.),tolerance=tolerance)


def compare_contrasts(unit_losses,reported,tolerance=1e-12):
    """Check point differences of separately averaged session losses."""
    checked=0;maximum=0.
    for row in reported:
        domain=tuple(row[k] for k in ('task','endpoint','reviewer','metric'))
        means={}
        for arm in (row['arm_a'],row['arm_b']):
            selected={}
            for key,loss in unit_losses.items():
                if key[0]==arm and key[2:6]==domain:selected.setdefault(key[6:],[]).append(loss)
            if not selected:raise ValueError('Missing arm for independent endpoint contrast')
            means[arm]={unit:mean(losses) for unit,losses in selected.items()}
        a,b=means[row['arm_a']],means[row['arm_b']]
        if set(a)!=set(b):raise ValueError('Independent endpoint comparison has unmatched units')
        case_diffs={}
        for unit in a:case_diffs.setdefault(unit[0],[]).append(a[unit]-b[unit])
        expected=mean([mean(v) for v in case_diffs.values()])
        error=abs(expected-float(row['estimate']));maximum=max(maximum,error)
        if error>tolerance:raise ValueError('Independent endpoint contrast mismatch')
        checked+=1
    return dict(contrasts=checked,maximum_absolute_error=maximum,tolerance=tolerance)


def selftest():
    assert percentile([0,10],.25)==2.5 and percentile([0,10],.75)==7.5
    forecasts=[dict(task='natural',arm='a',replicate='1',case_id='x',ingredient='genki',probability='0.2'),
               dict(task='lexical',arm='a',replicate='1',case_id='y',ingredient='hugs',probability='.8',predicted_change='.1')]
    result=scores(forecasts,{('x','genki'):1},{('y','hugs'):.3},[])
    assert abs(result['a','1','natural','canonical_active','objective','brier']['score']-.64)<1e-12
    assert abs(result['a','1','lexical','preference_change','objective','mae']['score']-.2)<1e-12
    assert abs(result['a','1','lexical','positive_preference_change','objective','brier_sign']['score']-.04)<1e-12
    print('Independent stdlib synthetic recomputation checks passed; no112outcomes read')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selftest',action='store_true')
    for key in ('labels-lock','discovery','natural','natural-cases','lexical','forecasts','labels','primary-scores','output'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--sensitivity-labels',type=Path,action='append',default=[],help='File basename labels_reviewer_A/B or labels_uncertain_zero/one determines reviewer name; use the primary analyzer reviewer names')
    p.add_argument('--primary-comparisons',type=Path)
    args=p.parse_args()
    if args.selftest:selftest();return
    for key in ('labels_lock','discovery','natural','natural_cases','lexical','forecasts','labels','primary_scores','output'):
        if getattr(args,key) is None:p.error('All explicit input paths and --labels-lock required')
    lock=jsonfile(args.labels_lock)
    expected=lock['labels_sha256'][args.labels.name]
    if sha(args.labels)!=expected:raise ValueError('Labels do not match preanalysis lock')
    labels=csvrows(args.labels)
    for file in args.sensitivity_labels:
        if sha(file)!=lock['labels_sha256'][file.name]:raise ValueError('Sensitivity labels not locked')
        reviewer=file.stem.removeprefix('labels_')
        for row in csvrows(file):row['reviewer']=reviewer;labels.append(row)
    cases=jsonfile(args.natural_cases)['cases']
    thresholds,ntruth,ltruth=make_objective_truth(jsonlines(args.discovery),jsonlines(args.natural),jsonlines(args.lexical),{c['case_id'] for c in cases})
    independent,unit_losses=scores(csvrows(args.forecasts),ntruth,ltruth,labels,with_units=True)
    audit=compare(independent,csvrows(args.primary_scores))
    contrast_audit=compare_contrasts(unit_losses,csvrows(args.primary_comparisons)) if args.primary_comparisons else None
    paths=[args.labels_lock,args.discovery,args.natural,args.natural_cases,args.lexical,args.forecasts,args.labels,args.primary_scores]+args.sensitivity_labels
    if args.primary_comparisons:paths.append(args.primary_comparisons)
    result=dict(study=112,independent=True,passed=True,implementation='stdlib-only; no primary analyzer imports',
                primary_forecast_audit=audit,natural_thresholds=thresholds,natural_units=len(ntruth),lexical_units=len(ltruth),
                per_endpoint_contrast_audit=contrast_audit,
                hashes={str(path):sha(path) for path in paths},auditor_sha256=sha(__file__))
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
