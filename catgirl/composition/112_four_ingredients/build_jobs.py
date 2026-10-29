"""CPU-only prospective job builder; only discovery outputs may calibrate doses."""
import argparse
import hashlib
import json
from pathlib import Path
import math
import statistics

ROOT=Path(__file__).resolve().parent
REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
NAMES=['genki','hugs','paws','female']
DIRECTIONS={'genki':1,'hugs':-1,'paws':1,'female':-1}
FRACTIONS=[.25,.5,1.]

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p)
    if p.exists(): raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def case_ids(name): return [c['case_id'] for c in read(ROOT/name)['cases']]
def fraction_key(f): return str(f).replace('.','p')

def calibration(path):
    cal=read(path)
    result={}
    natural=Path(path).with_name('natural.jsonl')
    raw=rows(natural)
    assert all(r['split']=='discovery' for r in raw),'Discovery only!'
    for name in NAMES:
        scope='readout_text' if cal['values'][name].get('readout_text',{}).get('count',0) else 'readout'
        q=cal['values'][name][scope]['quantiles']
        values=[]
        for row in raw:
            a,b=row['spans'][scope]
            values.extend(row['activations'][name]['pre'][a:b])
        assert len(values)==cal['values'][name][scope]['count']
        result[name]=dict(direction=DIRECTIONS[name],scope=scope,scope_fallback=scope!='readout_text',count=len(values),
                          q05=q['5'],q50=q['50'],q95=q['95'],q25=q['25'],q75=q['75'],
                          final_readout_q25=cal['values'][name]['readout']['quantiles']['25'],
                          final_readout_q75=cal['values'][name]['readout']['quantiles']['75'],
                          final_readout_count=cal['values'][name]['readout']['count'],
                          sd=statistics.pstdev(values),mean=statistics.mean(values),observed_min=min(values),observed_max=max(values),
                          endpoint=q['95'] if DIRECTIONS[name]>0 else q['5'],
                          down_column_l2=cal['down_column_l2'][name])
    return result

def operation(name,cals,fraction,scope,attenuate=False):
    c=cals[name]
    return dict(ingredient=name,mode='toward',direction=-c['direction'] if attenuate else c['direction'],
                value=c['q50'] if attenuate else c['endpoint'],fraction=fraction,scope=scope)

def make_job(cases,conditions):
    return dict(revision=REVISION,case_ids=case_ids(cases),seed=42,temperature=0.,max_tokens=128,
                repetition_penalty=1.1,conditions=conditions)

def build_discovery(args):
    cals=calibration(args.calibration)
    conditions=[dict(condition_id='baseline',operations=[])]
    for name in NAMES:
        for f in FRACTIONS:
            conditions.append(dict(condition_id=f'canonical_{name}_f{fraction_key(f)}',
                                   operations=[operation(name,cals,f,'generation')]))
    job=make_job('CASES_lexical_discovery.json',conditions)
    job.update(calibration_sha256=sha(args.calibration),plan_sha256=sha(ROOT/'112_PLAN.json'),
               calibration_details=cals,selection_rule='Frozen112_PLAN calibration selection')
    write(args.output/'JOB_discovery_writing.json',job)
    write(args.output/'DISCOVERY_CALIBRATION.json',dict(source_sha256=sha(args.calibration),values=cals))
    print(json.dumps(dict(event='discovery_job_built',conditions=len(conditions),cases=len(job['case_ids']),rows=len(conditions)*len(job['case_ids']))))

def select(args):
    cals=calibration(args.calibration)
    raw=rows(args.writing)
    lookup={(r['condition_id'],r['case_id']):r for r in raw}
    cases=read(ROOT/'CASES_lexical_discovery.json')['cases']
    assert len(lookup)==156 and all(r['split']=='discovery' for r in raw)
    details={}
    def preference(r): return r['choices']['target']['mean_logprob']-r['choices']['control']['mean_logprob']
    for name in NAMES:
        matching=[c for c in cases if c['ingredient']==name]
        assert len(matching)==3
        curves=[]
        for fraction in FRACTIONS:
            condition=f'canonical_{name}_f{fraction_key(fraction)}'
            gains=[preference(lookup[(condition,c['case_id'])])-preference(lookup[('baseline',c['case_id'])]) for c in matching]
            effects=[]
            for c in cases:
                r=lookup[(condition,c['case_id'])]
                for choice in r['choices'].values():
                    a=choice['activations'][name]
                    effects.extend(post-pre for p,pre,post in zip(a['positions'],a['pre'],a['post']) if p>=len(r['token_ids'])-1)
            rms=math.sqrt(statistics.mean(v*v for v in effects))
            curves.append(dict(fraction=fraction,case_gains=dict(zip([c['case_id'] for c in matching],gains)),
                               mean_gain=statistics.mean(gains),positive_cases=sum(v>0 for v in gains),
                               eligible=statistics.mean(gains)>=.01 and sum(v>0 for v in gains)>=2,
                               discovery_scalar_delta_rms=rms,discovery_residual_delta_rms=rms*cals[name]['down_column_l2'],
                               rms_sample_count=len(effects)))
        selected=next((v for v in curves if v['eligible']),curves[-1])
        details[name]=dict(calibration=cals[name],curves=curves,selected_fraction=selected['fraction'],
                           nonresponsive=not any(v['eligible'] for v in curves),selected=selected)
    source_rows=rows(args.calibration.with_name('natural.jsonl'))
    for name in NAMES:
        fraction=details[name]['selected_fraction']; c=cals[name]
        xs=[v for row in source_rows for v in row['activations'][name]['pre'][slice(*row['spans'][c['scope']])]]
        deltas=[fraction*(c['endpoint']-x) if (c['endpoint']-x)*c['direction']>0 else 0. for x in xs]
        rms=math.sqrt(statistics.mean(d*d for d in deltas))
        details[name]['matched_budget']=dict(scope=c['scope'],count=len(xs),scalar_delta_rms=rms,residual_delta_rms=rms*c['down_column_l2'],
             arithmetic='Nominal float64 transformation of observed BF16 discovery scalars; actual hook BF16 achievements reported separately')
    selection=dict(plan_sha256=sha(ROOT/'112_PLAN.json'),calibration_sha256=sha(args.calibration),
                   discovery_writing_sha256=sha(args.writing),selection_rule='First fraction mean matched target-control mean-token logp gain>=.01 and positive>=2/3; otherwise1diagnostic',
                   rms_rule='Matchingbudget: fixeddiscoveryreadout_text (fallbackreadout), nominalselectedtowardtransform RMS timesdowncolumnL2. Writingcurves additionallyretainachievedbranchRMSdiagnostics.',
                   ingredients=details)
    write(args.output/'SELECTED_DOSES.json',selection)
    for phase,scopes,casesfile in [('lexical',['readout','generation'],'CASES_lexical_validation.json'),
                                   ('semantic',['carrier','readout'],'CASES_natural_validation.json')]:
        for scope in scopes:
            conditions=[dict(condition_id='baseline',operations=[])]
            for name in NAMES:
                fraction=details[name]['selected_fraction']
                for attenuation in [False,True]:
                    conditions.append(dict(condition_id=f'{"attenuate" if attenuation else "canonical"}_{name}',
                                           operations=[operation(name,cals,fraction,scope,attenuation)]))
            job=make_job(casesfile,conditions)
            job.update(selection_sha256=sha(args.output/'SELECTED_DOSES.json'),scope=scope)
            write(args.output/f'JOB_validation_{phase}_{scope}.json',job)
    conditions=[]
    for bits in range(16):
        bitstring=f'{bits:04b}'
        conditions.append(dict(condition_id=bitstring,operations=[operation(name,cals,details[name]['selected_fraction'],'generation')
                                                                  for name,on in zip(NAMES,bitstring) if on=='1']))
    job=make_job('CASES_generation_validation.json',conditions)
    job.update(selection_sha256=sha(args.output/'SELECTED_DOSES.json'),bit_order=NAMES)
    write(args.output/'JOB_validation_factorial16.json',job)
    if args.control_calibration:
        controls=calibration(args.control_calibration)
        offsets={name:DIRECTIONS[name]*details[name]['matched_budget']['residual_delta_rms']/controls[name]['down_column_l2'] for name in NAMES}
        full_energy=math.sqrt(sum(details[name]['matched_budget']['residual_delta_rms']**2 for name in NAMES))
        strong_details={}
        for name in NAMES:
            c=cals[name]; endpoint=c['observed_max'] if DIRECTIONS[name]>0 else c['observed_min']
            xs=[v for row in source_rows for v in row['activations'][name]['pre'][slice(*row['spans'][c['scope']])]]
            max_residual=math.sqrt(statistics.mean((endpoint-x)**2 if (endpoint-x)*DIRECTIONS[name]>0 else 0. for x in xs))*c['down_column_l2']
            requested=full_energy/max_residual if max_residual else None
            fraction=min(1.,requested) if requested is not None else 1.
            strong_details[name]=dict(endpoint=endpoint,extreme_endpoint='observed_max' if DIRECTIONS[name]>0 else 'observed_min',
                                      max_discovery_residual_rms=max_residual,requested_fraction=requested,fraction=fraction,
                                      nominal_achieved_residual_rms=fraction*max_residual,nominal_budget_ratio=fraction*max_residual/full_energy if full_energy else None,
                                      unattainable=requested is None or requested>1.)
        def additive(name,value,scope,clip):
            return dict(ingredient=name,mode='add',value=value,scope=scope,clip=clip)
        metadata=dict(control_calibration_sha256=sha(args.control_calibration),source_selection_sha256=sha(args.output/'SELECTED_DOSES.json'),
                      target_calibration=cals,control_calibration=controls,random_offsets=offsets,strong_single=strong_details,
                      full_four_nominal_block_energy_rms=full_energy,
                      formula='Randomoffset=publisherdirection*selectedtargetdiscoveryresidualRMS/controldowncolumnL2. Strongsingleendpoint=ownobservedq100/q0; fraction=min(1,fullfourblockenergyRMS/extremeendpointfraction1residualRMS).',
                      limitation='Randomcontrol clip±1e30 ismachineguardonly, notnativebounded. Strongone-sidedtowardextremeobserveddiscoveryendpoint, fractioncap1; actualbudgetratios andunattainableflagrequired. Blockenergy sums acrosslayers, not a same-layer vector norm.')
        write(args.output/'MATCHED_CONTROLS.json',metadata)
        for phase,scopes,casesfile in [('lexical',['readout','generation'],'CASES_lexical_validation.json'),('semantic',['carrier','readout'],'CASES_natural_validation.json')]:
            for scope in scopes:
                cconditions=[dict(condition_id='baseline',operations=[])]+[dict(condition_id='random_'+name,operations=[additive(name,offsets[name],scope,[-1e30,1e30])]) for name in NAMES]
                cjob=make_job(casesfile,cconditions); cjob.update(coordinates_file='CONTROL_COORDINATES.json',matching_sha256=sha(args.output/'MATCHED_CONTROLS.json'),scope=scope)
                write(args.output/f'JOB_validation_{phase}_{scope}_random.json',cjob)
        strong=[dict(condition_id='strong_'+name,operations=[dict(ingredient=name,mode='toward',value=strong_details[name]['endpoint'],fraction=strong_details[name]['fraction'],direction=DIRECTIONS[name],scope='generation')]) for name in NAMES]
        sjob=make_job('CASES_generation_validation.json',strong); sjob['matching_sha256']=sha(args.output/'MATCHED_CONTROLS.json')
        write(args.output/'JOB_validation_strong_singletons.json',sjob)
        randomjob=make_job('CASES_generation_validation.json',[dict(condition_id='random_0000',operations=[]),dict(condition_id='random_1111',operations=[additive(name,offsets[name],'generation',[-1e30,1e30]) for name in NAMES])])
        randomjob.update(coordinates_file='CONTROL_COORDINATES.json',matching_sha256=sha(args.output/'MATCHED_CONTROLS.json'))
        write(args.output/'JOB_validation_random_four.json',randomjob)
    print(json.dumps(dict(event='selection_complete',selected={k:dict(fraction=v['selected_fraction'],nonresponsive=v['nonresponsive']) for k,v in details.items()})))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['discovery','select'],required=True)
    p.add_argument('--calibration',type=Path,default=ROOT/'results.local/discovery_capture/calibration.json')
    p.add_argument('--writing',type=Path)
    p.add_argument('--control-calibration',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'jobs')
    args=p.parse_args()
    if args.stage=='discovery': build_discovery(args)
    else:
        assert args.writing
        select(args)
if __name__=='__main__': main()
