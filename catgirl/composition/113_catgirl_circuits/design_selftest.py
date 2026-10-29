"""Synthetic tests only: dose selection, packet arm equivalence and normalization."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    selector=module('select_doses');packets=module('forecast_packets')
    features=json.loads((HERE/'FEATURE_DEFINITIONS.json').read_text())['feature_order']
    rows=[]
    for name,target in zip(selector.NAMES,selector.TARGETS):
        for fraction,count in ((.25,1),(.5,3),(1.,2)):
            for i in range(4):
                r={'split':'discovery','case_id':str(i),'condition_id':f'{name}_{fraction:g}',
                   **{f:'0' for f in features}}
                r['coherent']='1';r[target]=str(int(i<count));rows.append(r)
    for i in range(4):
        rows.append({'split':'discovery','case_id':str(i),'condition_id':'base',
                     **{f:'0' for f in features},'coherent':'1'})
    selected,audit=selector.select(rows)
    assert all(v==.5 for v in selected.values())
    assert all(v['responsive'] for v in audit.values())
    for r in rows:
        r['own_animal_body']='1'
    selected,audit=selector.select(rows)
    assert selected['paws']==1 and audit['paws']['saturated'] and not audit['paws']['responsive']
    for r in rows:
        if r['condition_id']!='base':r['coherent']='0'
    selected,audit=selector.select(rows)
    assert all(v==1 for v in selected.values()) and all(v['no_eligible_dose'] for v in audit.values())
    with tempfile.TemporaryDirectory(prefix='study113_design_') as temp:
        root=Path(temp); chosen=root/'selected.json';chosen.write_text(json.dumps(selected))
        raw=root/'discovery.jsonl'; rawrows=[]
        for cond in ['base']+[f'{name}_1' for name in selector.NAMES]:
            for i in range(4):
                rawrows.append({'condition_id':cond,'case_id':str(i),'split':'discovery',
                                'messages':[{'role':'user','content':'Neutral synthetic prompt'}],'text':'Synthetic output.'})
        raw.write_text('\n'.join(json.dumps(r) for r in rawrows)+'\n')
        job=root/'job.json';job.write_text(json.dumps({'temperature':0,'seed':42,'max_tokens':192}))
        out=root/'packets'
        with contextlib.redirect_stdout(io.StringIO()):
            packets.build(SimpleNamespace(selected=chosen,discovery_raw=raw,validation_job=job,out=out))
        a=json.loads((out/'descriptions_1.json').read_text());b=json.loads((out/'observations_only_1.json').read_text())
        for k in ('source_information','session_id','arm'):a.pop(k,None);b.pop(k,None)
        assert a==b, 'Arm payloads differ beyond intended source context'
        assert len(a['discovery_observations'])==24
        p={'rationale':'Synthetic complete forecast.','rows':[{'state':s,'family':f,'p':[.25]*7}
           for s in a['forecast_states'] for f in a['heldout_families']]}
        prediction=root/'prediction.json';prediction.write_text(json.dumps(p));normalized=root/'normalized.csv'
        with contextlib.redirect_stdout(io.StringIO()):
            packets.normalize(SimpleNamespace(packet=out/'descriptions_1.json',prediction=prediction,out=normalized))
        assert len(normalized.read_text().splitlines())==589
        p['rows'].append(p['rows'][0]);prediction.write_text(json.dumps(p))
        try:
            packets.normalize(SimpleNamespace(packet=out/'descriptions_1.json',prediction=prediction,out=root/'bad.csv'))
            raise AssertionError('Duplicate forecast accepted')
        except ValueError:pass
    print('Design selftest passed: maximal readable gain, saturation/failure fallback, arm equivalence,588 predictions, duplicate rejection.')

if __name__=='__main__':main()
